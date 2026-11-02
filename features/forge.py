"""导出：把 safetensors 权重转成 GGUF 二进制，按档量化，或原样另存 safetensors、ONNX、MLX。

调用示例：
  from features.forge import Choice, Export
  job = Export(Choice(source=Path("ckpt"), out=Path("m.gguf"), quant="q8_0"))
  report = job.run()          # {"tensors": 12, "bytes": 8123, "dropped": ["tokenizer"]}
  read(Path("m.gguf"))        # 自带的极简 reader，回读断言和自检都用它

  checked = Export(Choice(     # 量化前后跑同一批内置套件
      source=Path("ckpt"), out=Path("m.gguf"), quant="q8_0",
      self_check=True, suites=["zh-lite", "format-lite"],
  )).run()
  checked["self_check"]["degradation"]  # 正数表示量化后掉分

命令行那一层是这样调的：
  ori forge sft.warp.yaml --out out/gguf --format gguf --quant q8_0

GGUF v3 的字节布局，严格按顺序写，一处都不能调：
  magic "GGUF" → 版本 3 → tensor 数（u64）→ kv 数（u64）
  → kv 段（键 + 类型 + 值）→ tensor 信息段（名字 + 维数 + 各维 + 类型 + 偏移）
  → 补 0 到 32 字节对齐 → 数据段
偏移一律相对数据段开头，读出的一方加上数据段起点才是真位置。

为什么自己写而不调 llama.cpp：基础安装零编译，pip 装完就能出文件。
量化块照 ggml 的块定义写，一个字节都不改：
  q8_0   一块 32 个值 = 1 个 fp16 比例 + 32 个 int8，合计 34 字节
  q5_k   一块 256 个值 = 2 个 fp16 比例 + 12 字节 6bit 比例 + 32 字节第 5 位 + 128 字节低 4 位，合计 176 字节
  q6_k   一块 256 个值 = 128 字节低 4 位 + 64 字节高 2 位 + 16 个 int8 子块比例 + 1 个 fp16 超块比例，合计 210 字节
q5_k 里 5 位拆成低 4 位和一张第 5 位的位图，是 ggml 的 Q5_K 超块结构；
q6_k 的 6 位同样拆成低 4 位与高 2 位，是 ggml 的 Q6_K 超块结构。

档位怎么分派（llama.cpp 的 _M 就是靠这一手把输出层抬高）：
  output.weight、token_embd.weight 这两层喂给每个 token 的 logits 和梯度，误差最显眼，
  q5_k_m 把这两层升到 q6_k，其余仍走 q5_k；q5_k_s 是同一档位但不抬升，全走 q5_k。
分派只看张量名，写死在 OUTPUT_LAYER 里，别的名字不受影响。

行宽不是块宽（q8_0 是 32，k-quant 是 256）时补零到块宽再量化，不整体退 fp32：
补出来的零只占字节，读的一方按 dims 取前面的真值，尾巴丢掉。
三维张量按最外层摊平成行，走同一条行量化路径，dims 原样写回。

q6_k 的逐子块比例不是拍出来的数，是像上游 make_qx_quants 那样迭代搜出来的：
初值仍是 -max/32，之后每轮在基准值附近摆一圈候选，量一次算一次 rmse，只留最好的，3 轮收住。
标定见 tests/test_forge.py 里那份对照用例：同一份张量，迭代版的重构误差要比 -max/32 低一截。

tokenizer 从基座目录的 tokenizer.json（或 tokenizer.model）导出词表与 merges，
kv 缺哪一条、读不出来都记进 dropped：产物离开仓库能不能独立跑，看这一栏。
onnx 走 torch.onnx.export 导整张因果图（词表 → N 层 → 归一 → 输出层），权重落旁边同名 safetensors，
这张图能增量解码：每层都有 past_k.N / past_v.N，回出 present_k.N / present_v.N，一段喂完接着下一段；
导完过 onnx.checker，再拿 onnxruntime 跟 torch 对拍 logits；mlx 走 mlx_lm.convert 子进程，
Linux 上跑不了 mlx，就把命令、参数、临时目录布局打出来（--verbose），--mlx-print-only 只打印不执行。
"""

import json
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from utils.fail import fail

GGUF_MAGIC = b"GGUF"  # 文件头四字节，固定这串
GGUF_VERSION = 3  # 写死 v3，v2 少了 kv 里的对齐字段
ALIGN = 32  # 数据段对齐到 32 字节，ggml 读的时候按这个数跳
GGML_F32, GGML_F16, GGML_Q8_0, GGML_Q5_K, GGML_Q6_K = 0, 1, 8, 13, 14  # ggml 的类型号
QUANTS = {"f32": GGML_F32, "f16": GGML_F16, "q8_0": GGML_Q8_0, "q5_k_s": GGML_Q5_K, "q5_k_m": GGML_Q5_K, "q6_k": GGML_Q6_K}  # 认的档
FILE_TYPE = {"f32": 0, "f16": 1, "q8_0": 7, "q5_k_s": 16, "q5_k_m": 17, "q6_k": 18}  # general.file_type 的取值，照 ggml 的枚举
BLOCK = {GGML_Q8_0: 32, GGML_Q5_K: 256, GGML_Q6_K: 256}  # 一块装多少个值
BLOCK_BYTES = {GGML_Q8_0: 34, GGML_Q5_K: 176, GGML_Q6_K: 210}  # 一块占多少字节，回读断言用它
OUTPUT_LAYER = ("output.weight", "token_embd.weight")  # _M 档要抬到 q6_k 的输出层，按 ggml 的名字写

K_U32, K_I32, K_F32, K_BOOL, K_STRING, K_ARRAY = 4, 5, 6, 7, 8, 9  # kv 值的类型号，字符串数组走 K_ARRAY
KV_FMT = {K_U32: ("<I", 4), K_I32: ("<i", 4), K_F32: ("<f", 4), K_BOOL: ("<?", 1)}  # 标量值的打包格式与步长
TOKEN_TYPE = {"control": 3, "user": 4, "normal": 1}  # ggml 的 token 类型号，3 控制符 4 用户自定义 1 普通词
TOKEN_NAME = {"bos_token": "<s>", "eos_token": "</s>", "pad_token": "<pad>"}  # 配置没指时按惯例的名字找
NO_ID = 0xFFFFFFFF  # 没有这个特殊 token 时写的 u32，-1 的补码写法

ARCHS = {"llama", "qwen2", "qwen3", "mistral", "gemma", "gemma2", "phi3", "deepseek2"}  # 认得的模型结构

# config.json 字段 → GGUF 的 kv 名。键是 llama.cpp 认的那套，前缀在写的时候拼上结构名。
CONFIG_KEYS = {
    "num_hidden_layers": ("block_count", K_U32),  # 层数
    "hidden_size": ("embedding_length", K_U32),  # 隐藏维度
    "intermediate_size": ("feed_forward_length", K_U32),  # 前馈宽度
    "num_attention_heads": ("attention.head_count", K_U32),  # 注意力头数
    "num_key_value_heads": ("attention.head_count_kv", K_U32),  # kv 头数，MQA/GQA 靠它
    "max_position_embeddings": ("context_length", K_U32),  # 上下文长度
    "vocab_size": ("vocab_size", K_U32),  # 词表大小
    "rope_theta": ("rope.freq_base", K_F32),  # rope 底数
    "rms_norm_eps": ("attention.layer_norm_rms_epsilon", K_F32),  # RMSNorm 的 eps
}

# 张量名照 HF 的叫法换成 ggml 的叫法，逐个写死。改名的映射只在这里维护。
NAMES = {
    "model.embed_tokens.weight": "token_embd.weight",
    "model.norm.weight": "output_norm.weight",
    "lm_head.weight": "output.weight",
}
# 层内后缀的映射，前面拼 blk.<层号>.
LAYER_NAMES = {
    "self_attn.q_proj.weight": "attn_q.weight",
    "self_attn.k_proj.weight": "attn_k.weight",
    "self_attn.v_proj.weight": "attn_v.weight",
    "self_attn.o_proj.weight": "attn_output.weight",
    "mlp.gate_proj.weight": "ffn_gate.weight",
    "mlp.up_proj.weight": "ffn_up.weight",
    "mlp.down_proj.weight": "ffn_down.weight",
    "input_layernorm.weight": "attn_norm.weight",
    "post_attention_layernorm.weight": "ffn_norm.weight",
}
LAYER = re.compile(r"model\.layers\.(\d+)\.(.+)")  # 认层号和后缀
BLK = re.compile(r"blk\.(\d+)\.(.+)")  # onnx 那边反着认：ggml 名字拆回（层号, 后缀）

# onnx 图里的参数名，由 NAMES / LAYER_NAMES 反推着写。point 是给 torch 模块挂参数用的名字。
ONNX_NAMES = {
    "token_embd.weight": "embed.weight",
    "output_norm.weight": "norm.weight",
    "output.weight": "head.weight",
}
ONNX_LAYER_NAMES = {
    "attn_q.weight": "attn.q.weight",
    "attn_k.weight": "attn.k.weight",
    "attn_v.weight": "attn.v.weight",
    "attn_output.weight": "attn.o.weight",
    "ffn_gate.weight": "feed.gate.weight",
    "ffn_up.weight": "feed.up.weight",
    "ffn_down.weight": "feed.down.weight",
    "attn_norm.weight": "attention_norm.weight",
    "ffn_norm.weight": "feed_norm.weight",
}
ONNX_SEQ = 8  # 导出用的序列长度：图按这个长度固定下来，跑的时候也要喂同样长
ONNX_RELATIVE_MAX = 1e-4  # 自检的 logits 相对误差上限，差过线就是图导坏了
SELF_CHECK_TOKENS = 64  # 每道内置题最多生成 64 token，短答案够用，也避免坏模型一直吐
SELF_CHECK_TIMEOUT = 300  # 一道题最多跑五分钟，和 judge 的答案命令上限一致

Q6_UNITS = (1.0, 0.125, 0.03125)  # 迭代的轮数与每轮候选的步长单位，一条一轮，先粗后细，照 make_qx_quants 的 3 轮
Q6_STEPS = (-3, -2, -1, 1, 2, 3)  # 每轮的候选偏移，正负都有，避开 0（比例不动就白量一次）


@dataclass
class Choice:
    """这一次导什么、导到哪、按哪档量化。由命令行填。"""

    source: Path  # 权重文件或目录
    out: Path  # 产物路径
    format: str = "gguf"  # safetensors、gguf、onnx、mlx
    quant: str = "f16"  # 量化档，f16 就是不量化
    config: Path = None  # config.json，没有就不写结构参数
    tokenizer: Path = None  # 词表所在目录，缺了就按 source 旁边找
    with_tokenizer: bool = True  # 要不要导词表，--no-tokenizer 时关掉
    name: str = "hoshiori-export"  # 写进 general.name
    dropped: list = field(default_factory=list)  # 映射不上、只能丢的项，跑完给人看
    tensors: dict = None  # 外面已经读过（比如合并完）就直接给数组，不再读盘
    verbose: bool = False  # 把要执行的命令、参数、目录布局都打出来
    mlx_print_only: bool = False  # mlx 只打印不执行，方便在 macOS 上照抄
    self_check: bool = False  # 量化前后跑同一批套件并写退化报告
    suites: list = field(default_factory=lambda: ["zh-lite", "format-lite"])  # 自评要跑的内置套件或文件


class Export:
    """一次导出。一次 job 用一个实例。"""

    def __init__(self, choice: Choice) -> None:
        self.choice = choice

    def run(self) -> dict:
        """主流程：读权重 → 改名 → 可选基线 → 按格式写盘 → 可选量化评测 → 出账。"""
        tensors = self.choice.tensors or load(self.choice.source)  # 读进来就是 np 数组，坏文件当场报错
        named = rename(tensors, self.choice.dropped)  # 换成 ggml 的名字，没映射的照原名留着
        if self.choice.format == "safetensors":  # 安全格式就是原样另存
            return self._save(named)
        if self.choice.format == "onnx":  # 图另存，权重跟着落旁边
            return self._onnx(named)
        if self.choice.format == "mlx":  # 交给 mlx_lm 自己转
            return self._mlx(named)
        named = gguf_rope(named, self.choice.config)  # HF Llama 的半向量旋转需转成 GGML 的相邻对
        kv = metadata(self.choice.config, self.choice.name, self.choice.quant, self.choice.dropped)  # 元数据转 kv
        if self.choice.with_tokenizer:  # 不导词表就不塞这些 kv
            kv.update(tokenizer(self._token_dir(), self.choice.dropped))  # 词表照着基座目录读，缺了记进 dropped
        baseline = self._baseline(named, kv) if self.choice.self_check else None  # 先跑 f16，量化产物还没写
        blob = gguf_bytes(named, kv, self.choice.quant, self.choice.dropped)  # 拼整个文件，一回写出去
        self.choice.out.parent.mkdir(parents=True, exist_ok=True)
        self.choice.out.write_bytes(blob)  # 数据段一次性落盘
        summary = {"tensors": len(named), "bytes": len(blob), "dropped": self.choice.dropped}  # 本次导出的账
        if baseline is not None:  # 基线跑过才评量化产物
            summary["self_check"] = self._finish_check(baseline)  # 量化分与基线分落成旁边两份报告
        return summary

    def _baseline(self, tensors: dict, kv: dict) -> dict:
        """临时写一份 f16 GGUF 并先跑基准；临时模型在评完后立刻清掉。"""
        base_kv = dict(kv)  # 结构、词表与目标产物完全相同
        base_kv["general.file_type"] = (K_U32, FILE_TYPE["f16"])  # 只有权重档位换回 f16
        with tempfile.TemporaryDirectory(prefix="hoshiori-forge-base-") as scratch:  # 基线不留成第三份产物
            model = Path(scratch) / "baseline.f16.gguf"  # llama-cli 要吃一个真实文件
            model.write_bytes(gguf_bytes(tensors, base_kv, "f16", []))  # 当前权重原精度写成可运行的 GGUF
            return benchmark(model, self.choice.suites, "f16")  # 必须在临时目录收掉前跑完

    def _finish_check(self, baseline: dict) -> dict:
        """跑量化产物，把两份原始分数交给 self_check，并只留下最终比较报告。"""
        quantized = benchmark(self.choice.out, self.choice.suites, self.choice.quant)  # 同一套题、同一推理参数
        folder = self.choice.out.parent  # 最终 report.md / report.json 就放在产物旁边
        base_file = tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=folder, prefix=".forge-base-", suffix=".json", delete=False)  # 不覆盖已有隐藏文件
        quant_file = tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=folder, prefix=".forge-quant-", suffix=".json", delete=False)  # 同目录让最终报告落在产物旁
        base_path, quant_path = Path(base_file.name), Path(quant_file.name)  # self_check 的两份输入
        try:
            with base_file, quant_file:  # 中间报告写完先关，Windows 上也能马上重读
                json.dump(baseline, base_file, ensure_ascii=False, indent=2)  # 基线原始分
                json.dump(quantized, quant_file, ensure_ascii=False, indent=2)  # 量化原始分
            return self_check(base_path, quant_path)  # 汇总总分差与逐项差
        finally:
            base_path.unlink(missing_ok=True)  # 只保留用户要的两份比较报告
            quant_path.unlink(missing_ok=True)  # 原始答案可能很大，不在产物旁重复堆

    def _token_dir(self) -> Path:
        """词表在哪个目录。显式给了就用它，否则按权重路径推。"""
        if self.choice.tokenizer:  # 命令行从基座目录传进来的
            return Path(self.choice.tokenizer)
        source = Path(self.choice.source)  # 单个权重文件就取它的旁边
        return source if source.is_dir() else source.parent

    def _save(self, tensors: dict) -> dict:
        """另存成 safetensors，原文件的元数据跟着带过去。"""
        from safetensors.numpy import save_file  # 只有这条路要库，进来再导

        single = Path(self.choice.source).is_file()  # 目录进来的没有元数据
        meta = read_metadata(self.choice.source) if single else {}  # 单个文件的元数据原样带
        self.choice.out.parent.mkdir(parents=True, exist_ok=True)
        save_file(tensors, str(self.choice.out), metadata=meta)  # 整份写出去
        return {"tensors": len(tensors), "bytes": self.choice.out.stat().st_size, "dropped": self.choice.dropped}

    def _onnx(self, tensors: dict) -> dict:
        """导 ONNX：整张因果图，权重落旁边，导完过 checker，再跟 torch 对拍 logits。"""
        import torch  # 只有这条路要 torch
        from safetensors.numpy import save_file  # 权重落旁边，onnx 文件里只存图

        config = json.loads(Path(self.choice.config).read_text(encoding="utf-8")) if self.choice.config else {}  # 结构配置，没有就全靠权重形状推
        module, feeds = onnx_graph(tensors, config, torch)  # 同构模块加一批假 token 和空的 past
        names, out_names = onnx_names(len(feeds[1]))  # 输入输出名按层展开
        self.choice.out.parent.mkdir(parents=True, exist_ok=True)
        side = self.choice.out.with_suffix(".safetensors")  # 权重文件，和 onnx 同名不同后缀
        save_file({name: np.ascontiguousarray(one.astype(np.float32)) for name, one in tensors.items()}, str(side))  # 权重整份落盘
        if self.choice.verbose:  # 把这张图的输入输出摆出来
            for name, one in zip(names, flat(feeds)):  # 一个输入一行，形状照实打
                print(f"onnx 输入 {name} {one.dtype} {tuple(one.shape)}")
            print(f"onnx 输出 {', '.join(out_names)}，opset 14")
        # dynamo=False 走 TorchScript 那条老路：只要 torch 和 onnx，不额外要 onnxscript。
        torch.onnx.export(
            module,  # 拼好的模块
            feeds,  # 一批一个，长度 ONNX_SEQ 的假 token，past 长度 0
            str(self.choice.out),  # 图写到哪
            input_names=names,  # 输入名
            output_names=out_names,  # 输出名：每个位置的词表打分，加各层回出去的 present
            opset_version=14,  # 14 是各家运行时都吃得下的版本
            dynamo=False,  # 走老 exporter，见上面那行
            dynamic_axes=past_axes(names, out_names),  # 序列维和 past_len 维全放开，长度不再钉死
        )
        import onnx  # 只有这条路要 onnx

        onnx.checker.check_model(onnx.load(str(self.choice.out)))  # 图本身要合法，不合规当场报
        error, note = verify_graph(self.choice.out, module, feeds, names)  # 有 onnxruntime 就真跑一遍对数
        if error is not None and error > ONNX_RELATIVE_MAX:  # 差过线就是图导坏了
            fail("error", f"onnx 图和 torch 对不上，相对误差 {error:.2e}", hint="换 opset 或看这个算子有没有实现")
        if self.choice.verbose:  # 自检结论也打出来
            print(f"  {note}")
        kept = self.choice.out.stat().st_size + side.stat().st_size  # 图加权重一起算产出
        return {"tensors": len(tensors), "bytes": kept, "dropped": self.choice.dropped + [f"权重另存 {side.name}，onnx 只存图", note]}

    def _mlx(self, tensors: dict) -> dict:
        """导 MLX：把 mlx_lm.convert 的命令、参数、临时目录布局打出来；--mlx-print-only 只打印不跑。"""
        from safetensors.numpy import save_file  # 合并过的权重得先落盘才交给子进程

        folder = self.choice.out  # mlx 的产物是目录，不是单文件
        folder.parent.mkdir(parents=True, exist_ok=True)
        source = Path(self.choice.source)  # 一般直接指基座目录
        merged = bool(self.choice.tensors)  # 合并过的权重基座目录上没有，只能摊到临时目录
        scratch = Path(tempfile.mkdtemp(prefix="hoshiori-mlx-")) if merged else None  # 临时目录，打印时也得有个路径
        command = [  # 要执行的命令，转换器只认整目录
            sys.executable,
            "-m",
            "mlx_lm.convert",
            "--input",
            str(scratch if merged else source),
            "--mlx-path",
            str(folder),
        ]
        if self.choice.verbose or self.choice.mlx_print_only:  # 要在 macOS 上照抄的就是这几行
            print(f"mlx 转换（产物目录 {folder}）：")
            if merged:  # 合并过的权重要先摊到临时目录
                print(f"  临时目录 {scratch}：写 model.safetensors、config.json、tokenizer*")
            else:  # 直接读基座目录，不摊
                print(f"  直接读基座目录 {source}，不摊临时目录")
            print("  命令：" + " ".join(f'"{one}"' if " " in one else one for one in command))  # 路径带空格就加引号，照抄不会断
        if self.choice.mlx_print_only:  # 只打印，不执行
            if scratch:  # 打印用的临时目录顺手收掉
                shutil.rmtree(scratch, ignore_errors=True)
            return {"tensors": len(tensors), "bytes": 0, "dropped": self.choice.dropped + ["--mlx-print-only，只打印没执行"]}
        try:
            if scratch:  # 合并后的权重、结构配置、词表都摊到临时目录
                save_file(tensors, str(scratch / "model.safetensors"))  # 合并后的权重
                shutil.copy2(self.choice.config, scratch / "config.json")  # 结构配置照抄
                words = Path(self.choice.tokenizer or self.choice.source)  # 词表跟着基座，没有就用权重目录
                for one in ("tokenizer.json", "tokenizer.model", "tokenizer_config.json"):  # 词表也带上
                    if (words / one).exists():
                        shutil.copy2(words / one, scratch / one)
            run = subprocess.run(command, capture_output=True, text=True)  # 输出自己收
        finally:
            if scratch:  # 成不成都把临时目录扫干净
                shutil.rmtree(scratch, ignore_errors=True)
        if run.returncode:  # 转失败了把最后一段 stderr 摆出来
            fail("error", f"mlx 转换失败: {run.stderr.strip()[-400:]}", hint="单独跑一次 mlx_lm.convert 看完整输出")
        written = sum(one.stat().st_size for one in folder.rglob("*") if one.is_file())  # 目录里写出来的全部字节
        return {"tensors": len(tensors), "bytes": written, "dropped": self.choice.dropped}


# --- 量化自评 ---

def benchmark(model: Path, suites: list, label: str) -> dict:
    """用 llama.cpp 批处理入口跑套件并按 scorer 的规则出一份 judge 同形报告。

    model 是 f16 基线或刚写出的量化 GGUF；suites 可给内置名，也可给套件文件路径。
    固定 seed、temperature 与生成上限，避免采样抖动被误认成量化退化。
    新版优先 llama-completion；旧版沿用 llama-cli。stderr 保留诊断，stdout 只判生成正文。
    """
    from features import report, scorer, suite  # 自评才需要套件与报告，普通导出不多载模块

    program = shutil.which("llama-completion") or shutil.which("llama-cli")  # 新版 CLI 是交互 UI，批处理用 completion
    if program is None:  # 没运行时就不能声称跑过模型
        fail("missing", "量化自评要 llama.cpp", hint="将 llama-completion（新版）或 llama-cli（旧版）放进 PATH")

    found = []  # 已装载的套件，命令层验过后这里直接读
    for given in suites:  # 内置名和显式文件都收
        path = Path(given)  # 先按文件看
        path = path if path.is_file() else suite.locate(suite.folder(), str(given))  # 否则回内置目录按名找
        found.append(suite.read(path))  # 坏套件在第一次答题前停

    rows = []  # judge 报告同形的逐题结果
    for one in found:  # 一套一套跑，输出顺序可复现
        for task in one.tasks:  # 每题单起一次，失败能准确点到题号
            command = [
                program,  # llama.cpp 的非交互命令
                "-m", str(model),  # 当前要评的 GGUF
                "-p", task.prompt,  # 原样题面
                "-n", str(SELF_CHECK_TOKENS),  # 短答案的生成上限
                "--seed", "1",  # 两边固定同一随机种子
                "--temp", "0",  # 贪心解码，去掉采样噪声
                "--no-display-prompt",  # stdout 只留模型答案
                "--simple-io",  # 子进程管道下不用终端控制符
                "--single-turn",  # 一题答完就退，不等下一轮输入
            ]
            try:
                done = subprocess.run(command, capture_output=True, text=True, timeout=SELF_CHECK_TIMEOUT)  # 真跑模型
            except subprocess.TimeoutExpired:  # 卡死不能让整次导出永远挂着
                fail("input", f"自评题 {one.name}/{task.id} 超过 {SELF_CHECK_TIMEOUT} 秒", where=str(model))
            if done.returncode:  # 运行时拒绝模型或执行失败都不能当零分吞掉
                detail = done.stderr.strip()[-400:] or done.stdout.strip()[-400:]  # 最后一段通常就是根因
                fail("input", f"自评题 {one.name}/{task.id} 跑失败（退出码 {done.returncode}）", where=str(model), hint=detail)
            answer = done.stdout.strip()  # 装载与计时在 stderr，不关闭会连生成文本一起关闭的日志器
            answer = re.sub(r"\s*\[end of text\]\s*$", "", answer).strip()  # 只剥运行时终止标记，不改模型正文
            if "available commands:" in answer and "Loading model" in answer:
                fail("missing", "llama-cli 输出了交互界面，不能作为评测答案", hint="请安装同版本的 llama-completion")
            verdict = scorer.grade(task, answer)  # 与 ori judge 共用同一套判分
            rows.append({
                "suite": one.name,  # 哪套题
                "id": task.id,  # 哪一题
                "score": verdict.score,  # 0~1 分
                "reason": verdict.reason,  # 为什么得这个分
                "answer": answer,  # 原始答案留在中间报告供比较
            })

    return report.make(  # 字段与 judge/report.json 一致，self_check 只认这一种形状
        model=label,  # f16 或具体量化档
        backend=Path(program).name,  # 报告记录实际批处理程序
        suite=", ".join(one.name for one in found),  # 实际跑过的套件
        recipe="forge:self-check",  # 这不是另一次训练配方
        rows=rows,  # 全部逐题结果
    )


def self_check(baseline_report, quant_report) -> dict:
    """读两份 judge 形报告，计算量化减基线的总分差与逐项差，落 report.md / report.json。

    两份原始报告应放在量化产物所在目录；最终报告写到 quant_report 的同一目录。
    delta 为「量化 - 基线」，负数是掉分；degradation 为相反数，正数是退化幅度。
    """
    paths = [Path(baseline_report), Path(quant_report)]  # 统一成路径，字符串调用也收
    reports = []  # 基线、量化各一份
    for path in paths:  # 外部报告只在这个边界读和验一次
        try:
            data = json.loads(path.read_text(encoding="utf-8"))  # 报告是 UTF-8 JSON
        except (OSError, json.JSONDecodeError) as bad:  # 文件缺失或 JSON 写坏都说清位置
            fail("input", f"自评报告读不出来: {bad}", where=str(path))
        if "total" not in data or not isinstance(data.get("tasks"), list):  # 比较至少要总分与逐题分
            fail("input", "不像 judge 的 report.json", where=str(path), hint="报告要有 total 和 tasks")
        reports.append(data)  # 过边界后内部直接信任字段
    baseline, quantized = reports  # 名字固定，后面不再靠下标猜

    before = {(row.get("suite") or "", row["id"]): row for row in baseline["tasks"]}  # 基线逐项索引
    after = {(row.get("suite") or "", row["id"]): row for row in quantized["tasks"]}  # 量化逐项索引
    keys = list(before) + [key for key in after if key not in before]  # 基线顺序为主，新项目接在末尾
    items = []  # 每项的两边分数、差值与退化
    for key in keys:  # 同一套里的同一题才互相比
        base_score = float(before[key]["score"]) if key in before else 0.0  # 量化多出的题以 0 为基线
        quant_score = float(after[key]["score"]) if key in after else 0.0  # 量化漏题按 0 分，不能悄悄消失
        delta = round(quant_score - base_score, 4)  # 量化减基线，负数是掉分
        degradation = round(base_score - quant_score, 4)  # 直接相减，持平时写 0.0 而不是 -0.0
        items.append({
            "suite": key[0],  # 套件名参与唯一定位
            "id": key[1],  # 题号
            "baseline": base_score,  # 原权重得分
            "quantized": quant_score,  # 量化权重得分
            "delta": delta,  # 量化减基线
            "degradation": degradation,  # 正数更适合拿来跟阈值比较
        })
    items.sort(key=lambda one: (-one["degradation"], one["suite"], one["id"]))  # 掉最多的排最前，命令直接取前三

    baseline_total, quantized_total = float(baseline["total"]), float(quantized["total"])  # 两边总分只取一次
    total_delta = round(quantized_total - baseline_total, 4)  # 总分同样是量化减基线
    degradation = round(baseline_total - quantized_total, 4)  # 持平明确写 0.0
    data = {
        "schema": "hoshiori/forge-self-check/1",  # 机器读时先认版本
        "baseline": {"model": baseline.get("model", "f16"), "total": baseline_total},  # 基线摘要
        "quantized": {"model": quantized.get("model", "quantized"), "total": quantized_total},  # 量化摘要
        "total_delta": total_delta,  # 负数表示总分退化
        "degradation": degradation,  # 正数表示退化幅度
        "count": len(items),  # 比了多少项
        "items": items,  # 逐项差，按退化从大到小
        "raw_reports": {"baseline": baseline, "quantized": quantized},  # 留住实际答案/运行时/硬件，便于审查零分与解析污染
    }
    folder = paths[1].parent  # 量化报告在哪，最终比较报告就落在哪
    markdown_path, json_path = folder / "report.md", folder / "report.json"  # 用户约定的固定文件名
    data["files"] = {"markdown": str(markdown_path), "json": str(json_path)}  # 命令层打印时不用再猜路径
    markdown_path.write_text(self_check_markdown(data), encoding="utf-8")  # 给人看的退化表
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")  # 给机器与 CI 的同源数据
    return data  # Export 与命令层继续用同一份数据做门控


def self_check_markdown(data: dict) -> str:
    """把自评差值排成一张 Markdown 表；项目已经按退化从大到小排好。"""
    lines = [
        "# 量化自评报告",  # 报告标题
        "",  # 标题与摘要分开
        f"- 基线: {data['baseline']['model']}，总分 {data['baseline']['total']}",  # 原权重摘要
        f"- 量化: {data['quantized']['model']}，总分 {data['quantized']['total']}",  # 量化权重摘要
        f"- 总分差（量化 - 基线）: {data['total_delta']:+.4f}",  # 符号方向写进标题，免得猜
        f"- 退化幅度: {data['degradation']:+.4f}",  # 命令门控使用的数
        "",  # 摘要与表格分开
        "| 套件 | 项目 | 基线 | 量化 | 差值（量化 - 基线） | 退化 |",  # 逐项表头
        "|---|---|---:|---:|---:|---:|",  # 数字列右对齐
    ]
    for one in data["items"]:  # 掉分最多的先列，打开报告先看到风险
        suite_name = str(one["suite"]).replace("|", "\\|").replace("\n", " ")  # 不让名字撑破 Markdown 表
        item_name = str(one["id"]).replace("|", "\\|").replace("\n", " ")  # 题号同样收成单元格
        lines.append(
            f"| {suite_name} | {item_name} | {one['baseline']:.4f} | {one['quantized']:.4f} | "
            f"{one['delta']:+.4f} | {one['degradation']:+.4f} |"
        )
    return "\n".join(lines) + "\n"  # 文本文件用换行收尾


# --- onnx 整图 ---

def onnx_local_name(name: str):
    """ggml 名字换成 onnx 模块里的参数名。不进图的返回 None。"""
    if name in ONNX_NAMES:  # 词表、输出层这种整名
        return ONNX_NAMES[name]
    found = BLK.fullmatch(name)  # 层内的按后缀
    if found and found.group(2) in ONNX_LAYER_NAMES:
        return f"layers.{found.group(1)}.{ONNX_LAYER_NAMES[found.group(2)]}"
    return None


def onnx_names(layers: int) -> tuple:
    """图的输入输出名，按层展开。返回（输入名, 输出名）。

    每层一对 past 进、一对 present 出；past_len 为 0、present 原样给回时，行为和整段前向一样。
    """
    past = [f"past_{s}.{one}" for s in ("k", "v") for one in range(layers)]  # past_k.0 / past_v.0 …
    present = [f"present_{s}.{one}" for s in ("k", "v") for one in range(layers)]  # present_k.0 / present_v.0 …
    return ["input_ids", *past], ["logits", *present]


def past_axes(names: list, out_names: list) -> dict:
    """批维、序列维、past_len 维全放开，图才能吃任意批、任意长度、任意已有长度。"""
    axes = {"input_ids": {0: "batch", 1: "seq"}, "logits": {0: "batch", 1: "seq"}}  # 批与本步长度两头都动态
    for name in names[1:]:  # past：批、头、past_len、头宽
        axes[name] = {0: "batch", 2: "past_len"}
    for name in out_names[1:]:  # present：长度是 past_len + seq，另用一个维名
        axes[name] = {0: "batch", 2: "total_len"}
    return axes


def flat(feeds: tuple) -> list:
    """把（假 token, past_k 列表, past_v 列表）摊成一列，顺序和输入名一致。"""
    ids, past_k, past_v = feeds
    return [ids, *past_k, *past_v]


def onnx_graph(tensors: dict, config: dict, torch):
    """按权重里的名字拼一个和 features.model 同构的最小模块，权重真灌进去，返回（模块, 一批输入）。

    输入是（假 token, 各层空 past_k, 各层空 past_v），past 形状 (批, 头, past_len=0, 头宽)，
    前向回（logits, 各层 present_k …, 各层 present_v …）；把 present 当下一段的 past 喂回去就是增量解码。

    名字对不上 features.model 的张量（没映射过的）不进图，缺的张量保持随机初值（种子写死）。
    这样只有几个张量的小权重也能导出一张完整图，全张量的权重导出来就是原样的那一摞。
    几何从权重形状推，推不出来才看 config：形状是硬事实，config 只是补充。
    """
    from torch import nn  # 只有这条路要 nn

    torch.manual_seed(0)  # 缺的张量按这个种子起，两次跑出来的图一样
    found = {name: torch.from_numpy(np.ascontiguousarray(one.astype(np.float32))) for name, one in tensors.items()}  # 全换成 torch 张量

    def pick(suffix: str):  # 第一个带这个层内后缀的张量，小权重常常只有第 0 层
        return next((one for name, one in found.items() if BLK.fullmatch(name) and name.endswith(suffix)), None)

    embed = found.get("token_embd.weight")  # 词表权重决定词表大小和隐藏维
    hidden = int(embed.shape[1]) if embed is not None else int(config.get("hidden_size") or 128)  # 隐藏维
    vocab = int(embed.shape[0]) if embed is not None else int(config.get("vocab_size") or 128)  # 词表大小
    layers = 1 + max((int(one.split(".")[1]) for one in found if BLK.fullmatch(one)), default=-1)  # 层数按最大的层号推
    layers = layers or int(config.get("num_hidden_layers") or 1)  # 一个层权重都没有才照配置，至少一层
    query = pick("attn_q.weight")  # 注意力的输出宽度从 q_proj 的形状来
    d_attn = int(query.shape[0]) if query is not None else hidden  # 没有 q_proj 就按隐藏维
    gate = pick("ffn_gate.weight")  # 前馈的中间宽度从门控那一路来
    wide = int(gate.shape[0]) if gate is not None else hidden * 4  # 没有就按惯例 4 倍
    wanted = int(config.get("head_dim") or 0)  # 配置给的头宽优先，但得能整除且是偶数（rope 要成对转）
    count = int(config.get("num_attention_heads") or 0)  # 配置给的头数
    if not wanted and count and d_attn % count == 0:
        wanted = d_attn // count  # 只给头数就自己除出来
    head_dim = wanted if wanted and wanted % 2 == 0 and d_attn % wanted == 0 else (2 if d_attn % 2 == 0 else 1)  # 用不了就退到最保守的偶数
    heads = d_attn // head_dim  # 头数由头宽定，保证 heads × head_dim 正好铺满
    ctx = int(config.get("max_position_embeddings") or ONNX_SEQ)  # 上下文长度
    seq = max(1, min(ONNX_SEQ, ctx))  # 输入长度不超出上下文
    theta = float(config.get("rope_theta") or 10000.0)  # rope 底数
    tie = "output.weight" not in found  # 没有独立输出层就拿词表权重转置当输出层

    pair = torch.arange(0, head_dim, 2, dtype=torch.float32)  # 每两维一组
    freq = (theta ** (-pair / head_dim))[None, :]  # 每组的角速度，序列长度无关，可以预算

    def angles(steps, past):  # 角度按实际长度现算，导出的图才能吃任意长度
        spot = torch.arange(steps, dtype=torch.float32) + past  # 位置接着 past_len 往后数，不是从 0 起
        turn = spot[:, None] * freq  # 位置乘角速度
        return turn.cos()[None, None], turn.sin()[None, None]  # 补批和头两维，好跟张量对齐

    def rotate(x, steps, past):  # 相邻两维当一对转，和 features/model.py 的 rotate 一个算法
        if head_dim % 2:  # 头宽是奇数就配不成对，原样返回
            return x
        cos, sin = angles(steps, past)  # 按这一批的真实长度和起点取角度
        pairs = x.view(*x.shape[:-1], -1, 2)  # 头宽拆成 组 × 组内两值
        left, right = pairs[..., 0], pairs[..., 1]  # 一组的两个分量
        turned = torch.stack((-right, left), dim=-1)  # 旋转 90 度
        return (pairs * cos.unsqueeze(-1) + turned * sin.unsqueeze(-1)).flatten(-2)  # 按角度混完拼回头宽

    class RMS(nn.Module):
        """按最后一维的均方根归一。eps 和 features/model.py 一样是 1e-6。"""

        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(hidden))  # 学出来的缩放，初始 1

        def forward(self, x):
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.weight  # 归一后乘回缩放

    class Attention(nn.Module):
        """多头因果注意力，参数名缩写成 q / k / v / o。past 拼在本步前面一起算，回出拼好的 present。"""

        def __init__(self) -> None:
            super().__init__()
            self.q = nn.Linear(hidden, d_attn, bias=False)  # 查询
            self.k = nn.Linear(hidden, d_attn, bias=False)  # 键
            self.v = nn.Linear(hidden, d_attn, bias=False)  # 值
            self.o = nn.Linear(d_attn, hidden, bias=False)  # 拼回隐藏维

        def forward(self, x, past_k, past_v):
            batch, steps, _ = x.shape  # 批、本步长度、隐藏维
            past = past_k.shape[2]  # 前面已经有了多少位，query 的起点和遮罩都靠它
            newest = past + steps  # 拼完之后的键长
            split = lambda part: part.view(batch, steps, heads, head_dim).transpose(1, 2)  # 拆成（批, 头, 序列, 头宽）
            query = rotate(split(self.q(x)), steps, past)  # 查询从这个起点开始转位置
            key = torch.cat([past_k, rotate(split(self.k(x)), steps, past)], dim=2)  # 旧的键在前，本步的键接在后
            value = torch.cat([past_v, split(self.v(x))], dim=2)  # 值同样接在后面，值不转位置
            scores = (query @ key.transpose(-2, -1)) * (head_dim ** -0.5)  # 缩放点积，键长是 past + 本步
            reach = torch.arange(steps) + past  # 每个 query 能看到的最后一个键位置（含自己）
            ahead = torch.arange(newest)[None, :] > reach[:, None]  # 比它更靠右的键看不见
            scores = scores.masked_fill(ahead, float("-inf"))  # 左对齐的因果遮罩
            out = torch.softmax(scores, dim=-1) @ value  # 加权求和
            return self.o(out.transpose(1, 2).reshape(batch, steps, -1)), key, value  # 投影回去，键值当 present 交出去

    class Feed(nn.Module):
        """前馈：门控乘值，再压回隐藏维。"""

        def __init__(self) -> None:
            super().__init__()
            self.gate = nn.Linear(hidden, wide, bias=False)  # 门控那一路
            self.up = nn.Linear(hidden, wide, bias=False)  # 值那一路
            self.down = nn.Linear(wide, hidden, bias=False)  # 压回隐藏维

        def forward(self, x):
            return self.down(torch.nn.functional.silu(self.gate(x)) * self.up(x))  # 两路相乘就是门控

    class Block(nn.Module):
        """一层：两道 pre-norm 裹着注意力和前馈，各自带残差。"""

        def __init__(self) -> None:
            super().__init__()
            self.attention_norm = RMS()  # 进注意力前先归一
            self.attn = Attention()  # 注意力
            self.feed_norm = RMS()  # 进前馈前再归一
            self.feed = Feed()  # 前馈

        def forward(self, x, past_k, past_v):
            out, key, value = self.attn(self.attention_norm(x), past_k, past_v)  # 注意力这一跳，顺手带出 present
            x = x + out  # 残差
            return x + self.feed(self.feed_norm(x)), key, value  # 前馈这一跳，键值原样往上交

    class Graph(nn.Module):
        """整摞：查表 → 逐层 → 收尾归一 → 输出层。每层一份 past 进，一份 present 出。"""

        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Embedding(vocab, hidden)  # 词表
            self.layers = nn.ModuleList([Block() for _ in range(layers)])  # 一摞层
            self.norm = RMS()  # 收尾归一
            self.head = None if tie else nn.Linear(hidden, vocab, bias=False)  # 独立输出层，共用时就是 None

        def forward(self, ids, past_k, past_v):
            x = self.embed(ids)  # 查表
            keys, values = [], []  # 各层回出来的 present
            for block, one_k, one_v in zip(self.layers, past_k, past_v):  # 一层一层往下走，past 按层对号
                x, key, value = block(x, one_k, one_v)
                keys.append(key)  # 这一层拼好 past 的键
                values.append(value)  # 这一层的值
            x = self.norm(x)  # 收尾归一
            logits = x @ self.embed.weight.T if self.head is None else self.head(x)  # 共用词表就直接拿转置投影
            return (logits, *keys, *values)  # 每个位置的打分，加各层的 present

    module = Graph().eval()  # 先搭好架子
    state = {}  # 要灌进去的权重
    for name, one in found.items():
        local = onnx_local_name(name)  # 换算成模块里的参数名
        if local is not None:  # 认不出的不进图
            state[local] = one
    module.load_state_dict(state, strict=False)  # 缺的保持随机初值
    ids = (torch.arange(seq, dtype=torch.long) % vocab)[None]  # 一批一个，长度 seq 的假 token
    empty = lambda: torch.zeros(1, heads, 0, head_dim)  # past_len 为 0 的空 past：导出时就走整段前向那条路
    return module, (ids, [empty() for _ in range(layers)], [empty() for _ in range(layers)])  # 每层一份，键和值各一批


def verify_graph(path: Path, module, feeds: tuple, names: list) -> tuple:
    """有 onnxruntime 就真跑一遍对拍，返回（logits 相对误差, 说明）；没装返回（None, 说明）。

    past 用空的那一份（past_len 为 0），对拍的还是整段前向这一条路：图松了能拦住。
    """
    import torch

    try:
        import onnxruntime as runtime  # 只有这条路要它
    except ImportError:  # 缺包只跳过数值对拍，checker 已经过过
        return None, "没装 onnxruntime，数值没对拍（只过了 checker）"
    session = runtime.InferenceSession(str(path), providers=["CPUExecutionProvider"])  # 只在本机 CPU 上跑
    with torch.no_grad():  # 对拍不需要梯度
        want = module(*feeds)[0].numpy()  # torch 这一侧的结果，第 0 个回的是 logits
    feed = {name: one.numpy() for name, one in zip(names, flat(feeds))}  # 输入按名字喂
    got = session.run(None, feed)[0]  # onnx 这一侧的结果
    error = float(np.max(np.abs(got - want)) / max(float(np.max(np.abs(want))), 1e-12))  # 除以最大值算相对误差
    return error, f"onnx 对拍 logits 相对误差 {error:.2e}（上限 {ONNX_RELATIVE_MAX:g}）"


# --- 读权重 ---

def load(source: Path) -> dict:
    """读一个 safetensors 文件，或一个目录里的全部 safetensors。返回 名字 → 数组。"""
    path = Path(source)
    files = sorted(path.glob("*.safetensors")) if path.is_dir() else [path]  # 目录就全收，文件就一个
    if not files:  # 目录里一个权重都没有
        fail("input", f"这里没有 safetensors: {source}", hint="指到 .safetensors 文件或含权重的目录")
    tensors = {}
    for one in files:
        tensors.update(read_tensors(one))  # 一个个读进来
    return tensors


def read_tensors(path: Path) -> dict:
    """读一个 safetensors 文件。bf16 的位宽 numpy 不认，按偏移自己解。"""
    from safetensors import safe_open  # 库只管切张量和读元数据，够用

    raw = Path(path).read_bytes()  # 先整份读进来，bf16 要用原始字节
    size = struct.unpack("<Q", raw[:8])[0]  # 头 8 字节是头长度
    head = json.loads(raw[8 : 8 + size])  # 头是一段 JSON
    base = 8 + size  # 数据从头部之后开始
    out = {}
    with safe_open(str(path), framework="np") as handle:  # 库自己按名字切
        for name in head:
            if name == "__metadata__":  # 元数据不是张量
                continue
            start, stop = head[name]["data_offsets"]  # 该张量在数据段的起止
            try:
                out[name] = handle.get_tensor(name)  # 常规精度直接拿
            except TypeError:  # bf16：numpy 不认识这个类型
                bits = np.frombuffer(raw[base + start : base + stop], dtype=np.uint16)  # 取原始 16 位
                out[name] = (bits.astype(np.uint32) << 16).view(np.float32)  # 左移 16 位当 fp32 的高半
                out[name] = out[name].reshape(head[name]["shape"])  # 按头里的形状摆回去
    return out


def read_metadata(path: Path) -> dict:
    """读 safetensors 的自带元数据，没有就是空字典。"""
    from safetensors import safe_open

    with safe_open(str(path), framework="np") as handle:
        return dict(handle.metadata() or {})


# --- 改名 ---

def gguf_rope(tensors: dict, config: Path) -> dict:
    """将 HF Llama/Mistral Q/K 的半向量布局转成 GGML 相邻旋转对；星织权重已是相邻对。

    与 llama.cpp b12040 conversion/llama.py 的 permute 保持一致；只用于 GGUF，
    safetensors 原始权重和 ONNX 不经过这一步。每个 KV 头单独置换，支持 GQA。
    """
    if config is None:                                       # 没有结构不能推断头数，沿用原始布局
        return tensors
    data = json.loads(Path(config).read_text(encoding="utf-8"))  # 配置为外部模型的结构事实
    if data.get("model_type") not in {"llama", "mistral"} or data.get("rope_interleaved"):
        return tensors                                      # 星织、Qwen 等结构不套用 Llama 的转换
    out = dict(tensors)                                     # 不改调用方拿来做其他格式导出的权重
    for name, weight in tensors.items():
        if not name.endswith(("attn_q.weight", "attn_q.bias", "attn_k.weight", "attn_k.bias")):
            continue                                        # V、O、MLP 不参与旋转位置编码
        heads = data["num_attention_heads"]                 # 查询头的分组数
        if ".attn_k." in name:
            heads = data.get("num_key_value_heads", heads)  # GQA 键头可能比查询头少
        shape = (heads, 2, weight.shape[0] // heads // 2, *weight.shape[1:])
        out[name] = weight.reshape(shape).swapaxes(1, 2).reshape(weight.shape)  # 半向量变成交错配对
    return out


def ggml_name(name: str):
    """HF 张量名换成 ggml 的名字。不认识返回 None。"""
    if name in NAMES:  # 词表、输出层这种整名映射
        return NAMES[name]
    found = LAYER.fullmatch(name)  # 层内的按后缀映射
    if found and found.group(2) in LAYER_NAMES:
        return f"blk.{found.group(1)}.{LAYER_NAMES[found.group(2)]}"
    return None


def rename(tensors: dict, dropped: list) -> dict:
    """整份改名。换不了的照原名留着，并记进 dropped。"""
    out = {}
    for name, one in tensors.items():
        mapped = ggml_name(name)
        if mapped is None:  # 认不出的名字照样写，读的一方自己判断
            dropped.append(f"张量名未映射: {name}")
            mapped = name
        out[mapped] = one
    return out


# --- 元数据 ---

def metadata(config: Path, name: str, quant: str, dropped: list) -> dict:
    """config.json 尽力映射成 kv。映射不上的字段进 dropped。"""
    kv = {
        "general.architecture": (K_STRING, "llama"),  # 结构名，认不出就用 llama
        "general.name": (K_STRING, name),  # 这次导出的名字
        "general.quantization_version": (K_U32, 2),  # ggml 的量化版本，2 是当前值
        "general.file_type": (K_U32, FILE_TYPE[quant]),  # 整份文件的档
        "general.alignment": (K_U32, ALIGN),  # 数据段对齐，读的一方按它跳
    }
    if config is None:  # 没给 config 就只有上面这几条
        dropped.append("config.json 没给，结构参数一个没写")
        return kv
    data = json.loads(Path(config).read_text(encoding="utf-8"))  # 读结构配置
    arch = str(data.get("model_type") or "").lower()  # 结构名取 model_type
    if arch in ARCHS:  # 认得就换上
        kv["general.architecture"] = (K_STRING, arch)
    else:  # 认不得就留 llama，并记一笔
        dropped.append(f"model_type 不认识: {arch or '<空>'}")
    arch = kv["general.architecture"][1]  # 后面拼前缀用换过的那个
    for key, value in data.items():
        if key == "model_type":  # 已经换成 general.architecture 了，不算丢
            continue
        if key not in CONFIG_KEYS:  # 不在映射表里的字段只能丢
            dropped.append(f"config 字段未映射: {key}")
            continue
        suffix, kind = CONFIG_KEYS[key]
        kv[f"{arch}.{suffix}"] = (kind, value)  # 前缀加结构名，llama.cpp 只认这套
    return kv


# --- 词表 ---

def tokenizer(folder: Path, dropped: list) -> dict:
    """从基座目录拼出 tokenizer 的全部 kv。缺哪条、读不出来都记进 dropped。"""
    json_path = Path(folder) / "tokenizer.json"  # HF 那份，词表和 merges 都在里面
    model_path = Path(folder) / "tokenizer.model"  # SentencePiece 那份
    if json_path.exists():
        return _tokenizer_json(json_path, Path(folder), dropped)
    if model_path.exists():
        return _tokenizer_model(model_path, dropped)
    dropped.append("基座目录没有 tokenizer.json / tokenizer.model，产物不能独立跑")  # 缺词表就是不能独立跑
    return {}


def _tokenizer_json(path: Path, folder: Path, dropped: list) -> dict:
    """读 HF 的 tokenizer.json：词表、merges、特殊 token 都从这一份里抠。"""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))  # 整份读进来
    except (OSError, json.JSONDecodeError) as bad:  # 坏了也要说清是坏在哪一步
        dropped.append(f"tokenizer.json 读不出来（{bad}），产物不能独立跑")
        return {}
    model = data.get("model") or {}  # 词表本体
    vocab = model.get("vocab") or {}  # 词 → id
    if not vocab:  # 没有词表等于白导
        dropped.append("tokenizer.json 里没有 model.vocab，产物不能独立跑")
        return {}
    added = {int(one["id"]): one for one in data.get("added_tokens") or []}  # 特殊 token 自带 id
    size = max([*vocab.values(), *added] or [0]) + 1  # 词表按最大 id 撑开，id 就是下标
    tokens = [""] * size  # 先摆空位，缺 id 由空串占着
    for text, index in vocab.items():
        tokens[index] = text  # 词按 id 就位
    types = [TOKEN_TYPE["normal"]] * size  # 默认全是普通词
    for index, one in added.items():  # 特殊 token 覆盖类型
        tokens[index] = one.get("content", tokens[index])  # 文字用它的内容
        types[index] = TOKEN_TYPE["control"] if one.get("special") else TOKEN_TYPE["user"]  # 控制符 / 用户词
    merges = [" ".join(one) if isinstance(one, list) else one for one in model.get("merges") or []]  # BPE 的合并表
    if not merges and model.get("type") == "BPE":  # 有 BPE 却一条 merges 都没有，产物跑起来会掉词
        dropped.append("tokenizer.json 是 BPE 但没有 merges，产物不能独立跑")
    kv = {
        "tokenizer.ggml.model": (K_STRING, "gpt2" if model.get("type") == "BPE" else "llama"),  # 词表算法名
        "tokenizer.ggml.tokens": (K_ARRAY, (K_STRING, tokens)),  # 词表本体
        "tokenizer.ggml.merges": (K_ARRAY, (K_STRING, merges)),  # 合并表
        "tokenizer.ggml.token_type": (K_ARRAY, (K_I32, types)),  # 每个词的类型
    }
    # SmolLM 在 ByteLevel 前先拆单个数字；默认 GPT-2 会把多位数字合并，实际推理会变。
    # 布局对应 llama.cpp b12040 conversion/base.py 的 smollm，按完整预分词结构识别。
    smollm = {"type": "Sequence", "pretokenizers": [
        {"type": "Digits", "individual_digits": True},
        {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": True},
    ]}
    if data.get("pre_tokenizer") == smollm:
        kv["tokenizer.ggml.pre"] = (K_STRING, "smollm")        # 数字和空白必须与源 tokenizer 保持一致
    settings_path = folder / "tokenizer_config.json"          # 对话模板和特殊词表开关跟模型一起导出
    if settings_path.is_file():
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
        if isinstance(settings.get("chat_template"), str):  # 多模板映射需用户选择，不能随意猜一种
            kv["tokenizer.chat_template"] = (K_STRING, settings["chat_template"])
        for flag in ("add_bos_token", "add_eos_token"):
            if isinstance(settings.get(flag), bool):         # 仅导出明确给出的开关
                kv[f"tokenizer.ggml.{flag}"] = (K_BOOL, settings[flag])
    for key, label, default in (  # 三个特殊 id，配置里指到哪个词就用哪个 id
        ("bos_token", "bos_token_id", 1),
        ("eos_token", "eos_token_id", 2),
        ("pad_token", "padding_token_id", NO_ID),
    ):
        found = _token_id(folder, key, added)  # 配置里的名字换成 id
        if found is None:  # 配置没指就按惯例的名字在词表里找
            found = next((index for index, one in added.items() if one.get("content") == TOKEN_NAME[key]), None)
        if found is None:  # 还是没有就写默认，并记一笔
            dropped.append(f"tokenizer 缺 {label}，产物不能独立跑")
        kv[f"tokenizer.ggml.{label}"] = (K_U32, default if found is None else found)
    return kv


def _token_id(folder: Path, key: str, added: dict):
    """按 tokenizer_config.json 里的名字在 added_tokens 里换 id，没有返回 None。"""
    config_path = folder / "tokenizer_config.json"  # 配置和词表放同一处
    if not config_path.exists():  # 没有配置就没有名字
        return None
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))  # 配置是 JSON
    except (OSError, json.JSONDecodeError) as bad:
        return None  # 读不出来当没有，缺的那条由上层记
    raw = config.get(key)  # bos_token / eos_token / pad_token
    text = raw.get("content") if isinstance(raw, dict) else raw  # 兼容 {"content": ...} 写法
    for index, one in added.items():
        if one.get("content") == text:  # 名字对上就是它
            return index
    return None


def _tokenizer_model(path: Path, dropped: list) -> dict:
    """读 SentencePiece 的 tokenizer.model。只要 pieces 那一段，明文 protobuf 自己走。"""
    raw = path.read_bytes()
    tokens, types = [], []
    pos = 0
    while pos < len(raw):  # 一个字段一个字段往前推
        key, pos = _varint(raw, pos)  # 字段号和类型
        field, wire = key >> 3, key & 7
        if field == 1 and wire == 2:  # 一条 piece，模型文件的主体
            size, pos = _varint(raw, pos)  # 这条多长
            text, one_type, pos = _piece(raw, pos, pos + size)  # 词和类型
            tokens.append(text)
            types.append(one_type)
            continue
        if wire == 2:  # 别的长度段整个跳过
            size, pos = _varint(raw, pos)
            pos += size
        elif wire == 0:  # 变长整数
            _, pos = _varint(raw, pos)
        elif wire == 5:  # 定长 4 字节
            pos += 4
        else:  # 定长 8 字节
            pos += 8
    if not tokens:  # 一条都没解出来
        dropped.append(f"{path.name} 里没解出词表，产物不能独立跑")
        return {}
    ids = {text: index for index, text in enumerate(tokens)}  # 名字 → id，找特殊 token 用
    kv = {
        "tokenizer.ggml.model": (K_STRING, "llama"),  # SentencePiece 就是 llama 这套
        "tokenizer.ggml.tokens": (K_ARRAY, (K_STRING, tokens)),  # 词表本体
        "tokenizer.ggml.merges": (K_ARRAY, (K_STRING, [])),  # SentencePiece 没有 merges
        "tokenizer.ggml.token_type": (K_ARRAY, (K_I32, types)),  # 每个词的类型，和 ggml 的枚举一致
    }
    for key, label, default in (  # 概型里这几个是固定写法，缺了写默认
        ("<s>", "bos_token_id", 1),
        ("</s>", "eos_token_id", 2),
        ("<pad>", "padding_token_id", NO_ID),
    ):
        found = ids.get(key)
        if found is None:  # 缺了就记一笔
            dropped.append(f"tokenizer 缺 {label}，产物不能独立跑")
        kv[f"tokenizer.ggml.{label}"] = (K_U32, default if found is None else found)
    return kv


def _piece(raw: bytes, start: int, stop: int):
    """解一条 SentencePiece piece，返回（词, 类型, 新位置）。类型缺省是普通词。"""
    text, one_type = "", TOKEN_TYPE["normal"]
    pos = start
    while pos < stop:
        key, pos = _varint(raw, pos)
        field, wire = key >> 3, key & 7
        if field == 1 and wire == 2:  # piece 文本
            size, pos = _varint(raw, pos)
            text = raw[pos : pos + size].decode("utf-8", "replace")
            pos += size
            continue
        if field == 3 and wire == 0:  # 类型枚举，编号和 ggml 的一致
            one_type, pos = _varint(raw, pos)
            continue
        if wire == 2:  # 别的长度段，整个跳过（score 是定长 4 字节，走下面）
            size, pos = _varint(raw, pos)
            pos += size
        elif wire == 0:
            _, pos = _varint(raw, pos)
        elif wire == 5:
            pos += 4
        else:
            pos += 8
    return text, one_type, stop


def _varint(raw: bytes, pos: int):
    """读一个 protobuf 变长整数，返回（值, 新位置）。"""
    value = shift = 0
    while True:
        byte = raw[pos]
        pos += 1
        value |= (byte & 0x7F) << shift  # 低 7 位是数据，高位是续读标记
        if byte < 0x80:
            return value, pos
        shift += 7


# --- 量化 ---

def tensor_bytes(kind: int, shape) -> int:
    """按形状算一份数据多少字节。行宽补齐过，按补齐后的宽度算。"""
    if kind in BLOCK:  # 块类型按行算：行数 × 每行块数 × 块字节
        width = int(shape[-1])
        rows = int(np.prod(shape[:-1])) if len(shape) > 1 else 1
        return rows * (-(-width // BLOCK[kind])) * BLOCK_BYTES[kind]
    return int(np.prod(shape)) * (2 if kind == GGML_F16 else 4)  # 不量化的按元素宽度


def q8_0(rows: np.ndarray) -> bytes:
    """每32值写 fp16 比例与32个 int8；分批向量化，避免大模型逐块进入Python。"""
    chunks = rows.astype(np.float32, copy=False).reshape(-1, 32)  # prepare 已补齐每行块宽
    layout = np.dtype([("scale", "<f2"), ("values", "i1", (32,))])  # GGML 紧凑34字节，不插结构体padding
    out = bytearray()                                         # 每批写完就释放中间浮点矩阵
    for start in range(0, len(chunks), 8192):                  # 每批约1MiB像素，峰值不随模型块数增长
        batch = chunks[start:start + 8192]                   # 保持行优先，和旧标量实现同一字节序
        scales = (np.maximum(np.abs(batch).max(axis=1), 1e-8) / 127).astype(np.float16)
        values = np.zeros_like(batch)                        # 全零及低于fp16范围的块写成零，避免0/0
        np.divide(batch, scales[:, None].astype(np.float32), out=values, where=scales[:, None] != 0)
        packed = np.empty(len(batch), dtype=layout)            # 每一项恰是一块GGML数据
        packed["scale"] = scales                              # 使用真实存盘比例计算，回读才对得上
        packed["values"] = np.clip(np.rint(values), -127, 127).astype(np.int8)
        out.extend(packed.tobytes())                          # 比例在前、量化数在后，一批顺序追加
    return bytes(out)


def _pack_scales(ls: np.ndarray, lm: np.ndarray) -> bytes:
    """把 8 个 6bit 比例和 8 个 6bit 下限压进 12 字节，照 ggml 的排法。"""
    out = bytearray(12)  # 全部先置 0，靠下面按位拼
    for j in range(8):  # 8 个子块逐个来
        if j < 4:  # 前 4 块的低 6 位各占一个字节
            out[j] = int(ls[j])
            out[j + 4] = int(lm[j])
            continue
        out[j + 4] = (int(ls[j]) & 0xF) | ((int(lm[j]) & 0xF) << 4)  # 后 4 块的低 4 位合成一个字节
        out[j - 4] |= (int(ls[j]) >> 4) << 6  # 高 2 位别回去，挤在前 4 字节的高位
        out[j] |= (int(lm[j]) >> 4) << 6  # 下限同理
    return bytes(out)


def q5_k(rows: np.ndarray) -> bytes:
    """q5_k（q5_k_s 与 q5_k_m 的块本体）：每 256 个值一个超块，5 位量化。"""
    out = bytearray()
    rows = rows.astype(np.float32)
    for row in rows:  # 一行行来
        for start in range(0, row.size, 256):  # 256 个值一个超块
            block = row[start : start + 256].reshape(8, 32)  # 8 个子块，每块 32 个值
            lo, hi = block.min(1), block.max(1)  # 每个子块的范围
            lo = np.minimum(lo, 0)  # 下限夹到 0 以内（ggml 的规矩），全正数的块就从 0 起算
            span = hi - lo  # 跨度，除 31 就是每一步
            scale = np.where(span > 0, span / 31, 1 / 31)  # 全平的子块给个最小步长，免得除零
            mins = -lo  # 下限取负，公式是 值 = 比例*q - 下限，所以它非负
            top_scale, top_min = float(scale.max()), float(max(mins.max(), 1e-8))  # 超块里最大的两个当基准
            d, dmin = np.float16(top_scale / 63), np.float16(top_min / 63)  # 基准再除 63 落进 6bit
            ls = np.clip(np.rint(scale / top_scale * 63), 0, 63).astype(np.uint8)  # 各子块比例对基准的比值
            lm = np.clip(np.rint(mins / top_min * 63), 0, 63).astype(np.uint8)  # 各子块下限对基准的比值
            low, high = float(d), float(dmin)  # 用存下去的比例回算 q，读回来才一致
            q = np.zeros(256, dtype=np.uint8)
            for j in range(8):  # 8 个子块逐个量化
                step = low * int(ls[j])  # 这个子块实际的步长
                base = high * int(lm[j])  # 这个子块实际的下限
                if step:  # 步长为 0 就整块留 0
                    q[j * 32 : (j + 1) * 32] = np.clip(np.rint((block[j] + base) / step), 0, 31)  # 5 位范围 0~31
            qh = bytearray(32)  # 第 5 位的位图
            qs = bytearray(128)  # 低 4 位，两个值挤一个字节
            for group in range(4):  # 4 组，每组 64 个值
                m1, m2 = 1 << (2 * group), 2 << (2 * group)  # 高低两个值各占位图里的一位
                for j in range(32):  # 每组 32 对
                    left, right = int(q[group * 64 + j]), int(q[group * 64 + 32 + j])
                    if left > 15:  # 超了 4 位就把第 5 位写进位图
                        qh[j] |= m1
                        left -= 16
                    if right > 15:
                        qh[j] |= m2
                        right -= 16
                    qs[group * 32 + j] = left | (right << 4)  # 低 4 位两个挤一起
            out += d.tobytes() + dmin.tobytes() + _pack_scales(ls, lm) + bytes(qh) + bytes(qs)  # 照结构拼块
    return bytes(out)


def q6_error(block: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """按一组比例量化再还原，返回每个子块的均方误差。"""
    with np.errstate(divide="ignore", invalid="ignore"):  # 候选比例可能正好是 0（第一轮 offset -1），除法会喊一声
        q = np.clip(np.rint(block / scale[:, None]), -32, 31)  # 6 位有符号就是 -32~31，inf 被 clip 成 31
    return np.mean((q * scale[:, None] - block) ** 2, axis=1)  # 均方误差，比 rmse 少开一次方


def q6_scales(block: np.ndarray) -> np.ndarray:
    """给 16 个子块各挑一个比例：初值 -max/32，再迭代几轮候选，只留误差最低的。

    初值和符号照 make_qx_quants 的约定：取子块绝对值最大的那个值取负再除 32。
    取负是因为 q6_k 存的是有符号 6 位、范围 -32~31，正极值要落在负端那 32 格上。
    上游在初值之后接着走几轮最小二乘；这里是同一个意思的直接搜法：每轮在基准值附近摆 ±1 到 ±3 个单位，
    量一次算一次 rmse，只留最好的，3 轮收住（Q6_UNITS 一条一轮）。
    单位要从 1 收到 1/8 再到 1/32，是因为子块之间的量级可能差几倍（只给细的走不出去），
    最优也常常只离初值几个百分点（只给粗的收不准）。
    实测：把极值正好压在 -32 上的 -max/32 对 rmse 偏大；极值不占主导的子块（极值被远值顶走、
    或者斜坡这种整段都要精度）能降两成以上，普通正态张量也有几个百分点。对照用例见 tests/test_forge.py。
    """
    base = -block[np.arange(block.shape[0]), np.abs(block).argmax(1)] / 32  # 初值，子块极值取负除 32
    best = base.copy()
    best_err = q6_error(block, best)  # 初值的误差，后面只跟它比
    for unit in Q6_UNITS:  # 一轮比一轮细
        for offset in Q6_STEPS:  # 六个候选逐个量
            candidate = best * (1 + offset * unit)  # 只缩放比例，不动值
            error = q6_error(block, candidate)  # 这个候选的误差
            better = error < best_err  # 逐子块比，好的就留下
            best = np.where(better, candidate, best)
            best_err = np.where(better, error, best_err)
    return best


def q6_k(rows: np.ndarray, refine: bool = True) -> bytes:
    """q6_k：每 256 个值一块，6 位有符号，低 4 位与高 2 位分开存。

    refine=False 退回旧的一拍脑袋版本（比例就是 -max/32，不迭代），对照用例拿它当基准。
    """
    out = bytearray()
    for row in rows.astype(np.float32):  # 一行行来
        for start in range(0, row.size, 256):  # 256 个值一个超块
            block = row[start : start + 256].reshape(16, 16)  # 16 个子块，每块 16 个值
            scale = q6_scales(block) if refine else -block[np.arange(16), np.abs(block).argmax(1)] / 32  # 子块比例
            peak = scale[int(np.abs(scale).argmax())]  # 超块里绝对值最大的那个比例，带符号
            if not peak:  # 整块全零，比例和量化值全留 0
                out += bytes(208) + np.float16(0).tobytes()  # 128+64+16 个零加一个 0 比例
                continue
            iscale = -128 / peak  # 比例之间换算用的倒数，照 ggml 的写法
            ls = np.clip(np.rint(iscale * scale), -128, 127).astype(np.int8)  # 子块比例相对超块取整，-128 也留着
            flat = ls == 0  # 取整成 0 的子块
            ls[flat & (scale != 0)] = np.where(scale[flat & (scale != 0)] > 0, 1, -1)  # 有值却落到 0 的给一格，不然整块被抹平
            d = np.float16(-peak / 128)  # 超块比例存 fp16，符号带着极值的方向
            q = np.zeros(256, dtype=np.int8)
            for j in range(16):  # 16 个子块逐个量化
                if not ls[j]:  # 比例 0 的子块整块留 0
                    continue
                step = float(d) * int(ls[j])  # 这个子块实际的步长，两个符号相乘落地
                q[j * 16 : (j + 1) * 16] = np.clip(np.rint(block[j] / step), -32, 31)  # 6 位有符号范围
            u = (q.astype(np.int16) + 32).astype(np.uint8)  # 转成 6 位无符号存，读的一方减 32
            ql, qh = bytearray(128), bytearray(64)  # 低 4 位 128 字节，高 2 位 64 字节
            for half in range(2):  # 两个 128 值的大半块
                base = half * 128  # 这一大半块的起点
                for l in range(32):  # 每轮摆 4 个值
                    ql[half * 64 + l] = int(u[base + l]) & 0xF | (int(u[base + 64 + l]) & 0xF) << 4  # 第 1、3 个值挤一个字节
                    ql[half * 64 + 32 + l] = int(u[base + 32 + l]) & 0xF | (int(u[base + 96 + l]) & 0xF) << 4  # 第 2、4 个值挤一个字节
                    qh[half * 32 + l] = (  # 四个值的高 2 位各占一个字节的两位
                        (int(u[base + l]) >> 4)
                        | (int(u[base + 32 + l]) >> 4) << 2
                        | (int(u[base + 64 + l]) >> 4) << 4
                        | (int(u[base + 96 + l]) >> 4) << 6
                    )
            out += bytes(ql) + bytes(qh) + ls.tobytes() + d.tobytes()  # 照结构拼块：低 4 位、高 2 位、比例、超块比例
    return bytes(out)


QUANTIZE = {GGML_Q8_0: q8_0, GGML_Q5_K: q5_k, GGML_Q6_K: q6_k}  # 类型号 → 量化函数


def kind_for(quant: str, name: str) -> int:
    """这个张量用哪种块。_M 档把输出层抬到 q6_k，其余按档位本身。"""
    if quant == "q5_k_m" and name in OUTPUT_LAYER:  # 输出层误差最显眼，_M 就是靠抬它换质量
        return GGML_Q6_K
    return QUANTS[quant]


def prepare(tensor: np.ndarray, quant: str, name: str, dropped: list):
    """决定一个张量怎么存，返回（字节, 类型号）。一维的留 fp32，矩阵行宽不足就补零。"""
    if tensor.ndim < 2:  # 范数、偏置这类一维量，fp32 是 ggml 的常规做法
        if QUANTS[quant] in BLOCK:  # 量化档下才值得一提
            dropped.append(f"{name} 是一维，留 fp32")
        return tensor.astype(np.float32).tobytes(), GGML_F32
    kind = kind_for(quant, name)  # 档位加名字定块型
    if kind not in BLOCK:  # f16 / f32 不量化
        return tensor.astype(np.float16 if kind == GGML_F16 else np.float32).tobytes(), kind
    width = int(tensor.shape[-1])
    pad = -width % BLOCK[kind]  # 补到块宽的整数倍，正好齐就是 0
    rows = tensor.reshape(-1, width).astype(np.float32)  # 三维的按最外层摊平成行，走同一条路
    if pad:  # 尾巴补零只占字节，dims 照原样写
        dropped.append(f"{name} 行宽 {width} 补 {pad} 个零到 {width + pad} 再量化")
        padded = np.zeros((rows.shape[0], width + pad), dtype=np.float32)
        padded[:, :width] = rows  # 真值放前面，零在尾巴
        rows = padded
    return QUANTIZE[kind](rows), kind  # 走这条块型的量化函数


# --- 写文件 ---

def put_text(buf: bytearray, text: str) -> None:
    """写一个 GGUF 字符串：u64 长度 + 字节。"""
    raw = text.encode("utf-8")
    buf += struct.pack("<Q", len(raw)) + raw


def _put_value(buf: bytearray, kind: int, value) -> None:
    """写一个标量或数组元素。字符串有自己的长度前缀。"""
    if kind == K_STRING:
        put_text(buf, str(value))
        return
    buf += struct.pack(KV_FMT[kind][0], value)


def put_kv(buf: bytearray, key: str, kind: int, value) -> None:
    """写一条 kv：键、类型、值。数组先写元素类型和个数。"""
    put_text(buf, key)  # 键
    buf += struct.pack("<I", kind)  # 类型
    if kind != K_ARRAY:  # 标量直接写
        _put_value(buf, kind, value)
        return
    item_kind, items = value  # 数组是（元素类型, 列表）
    buf += struct.pack("<I", item_kind)  # 元素类型
    buf += struct.pack("<Q", len(items))  # 元素个数
    for one in items:  # 逐个元素
        _put_value(buf, item_kind, one)


def gguf_bytes(tensors: dict, kv: dict, quant: str, dropped: list = None) -> bytes:
    """把整份文件拼出来：头、kv、张量信息、对齐填充、数据段。"""
    if quant not in QUANTS:  # 档位要在表里
        fail("input", f"不认识的量化档 {quant}", hint="可用 " + " / ".join(sorted(QUANTS)))
    dropped = dropped if dropped is not None else []  # 没给就现建一个，退回的记不记都行
    prepared = []  # (名字, 维度, 类型, 偏移, 字节)
    offset = 0
    for name, tensor in tensors.items():  # 每个张量先算字节和偏移
        payload, kind = prepare(tensor, quant, name, dropped)
        offset = (offset + ALIGN - 1) // ALIGN * ALIGN  # 每份数据前面对齐
        prepared.append((name, tensor.shape, kind, offset, payload))
        offset += len(payload)

    buf = bytearray()
    buf += GGUF_MAGIC  # magic
    buf += struct.pack("<I", GGUF_VERSION)  # 版本
    buf += struct.pack("<Q", len(prepared))  # tensor 数
    buf += struct.pack("<Q", len(kv))  # kv 数
    for key in sorted(kv):  # 键排好序写，读的一方好查
        kind, value = kv[key]
        put_kv(buf, key, kind, value)
    for name, shape, kind, at, _ in prepared:  # 张量信息段
        put_text(buf, name)  # 名字
        buf += struct.pack("<I", len(shape))  # 维数
        for size in reversed(shape):  # ggml 的维度是反着写的，最内层在前
            buf += struct.pack("<Q", size)
        buf += struct.pack("<I", kind)  # 类型号
        buf += struct.pack("<Q", at)  # 数据段内偏移
    pad = (ALIGN - len(buf) % ALIGN) % ALIGN  # 补齐到对齐边界
    buf += b"\0" * pad
    base = len(buf)  # 数据段的起点，偏移都相对它
    for name, shape, kind, at, payload in prepared:  # 数据段按偏移就位
        buf += b"\0" * (at - (len(buf) - base))  # 补到这一份的偏移
        buf += payload
    return bytes(buf)


def read(path: Path) -> dict:
    """极简 reader：回读 magic、版本、kv、张量字节。测试和自检拿它断言。"""
    raw = Path(path).read_bytes()  # 整份读进来
    if raw[:4] != GGUF_MAGIC:  # 头四字节不对就不是 GGUF
        fail("input", f"不是 GGUF 文件: {path}", hint="头四字节要 GGUF")
    pos = 4
    version = struct.unpack_from("<I", raw, pos)[0]  # 版本
    pos += 4
    count = struct.unpack_from("<Q", raw, pos)[0]  # tensor 数
    pos += 8
    kv_count = struct.unpack_from("<Q", raw, pos)[0]  # kv 数
    pos += 8
    kv = {}
    for _ in range(kv_count):  # 逐条读 kv
        key, pos = _get_text(raw, pos)
        kind = struct.unpack_from("<I", raw, pos)[0]
        pos += 4
        if kind == K_ARRAY:  # 数组：元素类型、个数、逐个值
            item_kind = struct.unpack_from("<I", raw, pos)[0]
            pos += 4
            size = struct.unpack_from("<Q", raw, pos)[0]
            pos += 8
            items = []
            for _ in range(size):
                if item_kind == K_STRING:
                    one, pos = _get_text(raw, pos)
                else:
                    one = struct.unpack_from(KV_FMT[item_kind][0], raw, pos)[0]
                    pos += KV_FMT[item_kind][1]
                items.append(one)
            kv[key] = items
            continue
        if kind == K_STRING:  # 字符串值
            kv[key], pos = _get_text(raw, pos)
            continue
        kv[key] = struct.unpack_from(KV_FMT[kind][0], raw, pos)[0]
        pos += KV_FMT[kind][1]
    infos = []
    for _ in range(count):  # 逐个读张量信息
        name, pos = _get_text(raw, pos)
        dims = struct.unpack_from("<I", raw, pos)[0]
        pos += 4
        shape = []
        for _ in range(dims):  # 维度反着写，读回来再倒回去
            shape.append(struct.unpack_from("<Q", raw, pos)[0])
            pos += 8
        kind = struct.unpack_from("<I", raw, pos)[0]
        pos += 4
        at = struct.unpack_from("<Q", raw, pos)[0]
        pos += 8
        infos.append((name, tuple(reversed(shape)), kind, at))
    start = (pos + ALIGN - 1) // ALIGN * ALIGN  # 数据段从对齐边界开始
    data_len = len(raw) - start  # 数据段总长，断言用
    tensors = {}
    for name, shape, kind, at in infos:  # 按偏移切出每份数据，宽度按补齐后算
        tensors[name] = raw[start + at : start + at + tensor_bytes(kind, shape)]
    out = {"magic": raw[:4], "version": version, "kv": kv, "data_len": data_len}  # 头、版本、kv、数据段长度
    out["infos"], out["tensors"] = infos, tensors  # 张量信息与按偏移切出来的字节
    return out


def _get_text(raw: bytes, pos: int):
    """读一个 GGUF 字符串，返回（文本, 新位置）。"""
    size = struct.unpack_from("<Q", raw, pos)[0]
    pos += 8
    return raw[pos : pos + size].decode("utf-8"), pos + size
