"""导出：把 safetensors 权重转成 GGUF 二进制，按档量化，或原样另存 safetensors、ONNX、MLX。

调用示例：
  from features.forge import Choice, Export
  job = Export(Choice(source=Path("ckpt"), out=Path("m.gguf"), quant="q8_0"))
  report = job.run()          # {"tensors": 12, "bytes": 8123, "dropped": ["tokenizer"]}
  read(Path("m.gguf"))        # 自带的极简 reader，回读断言和自检都用它

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

tokenizer 从基座目录的 tokenizer.json（或 tokenizer.model）导出词表与 merges，
kv 缺哪一条、读不出来都记进 dropped：产物离开仓库能不能独立跑，看这一栏。
onnx 走 torch.onnx.export（图 + 旁边的 safetensors 权重），mlx 走 mlx_lm.convert 子进程。
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


class Export:
    """一次导出。一次 job 用一个实例。"""

    def __init__(self, choice: Choice) -> None:
        self.choice = choice

    def run(self) -> dict:
        """主流程：读权重 → 改名 → 收元数据 → 按格式写盘 → 出账。"""
        tensors = self.choice.tensors or load(self.choice.source)  # 读进来就是 np 数组，坏文件当场报错
        named = rename(tensors, self.choice.dropped)  # 换成 ggml 的名字，没映射的照原名留着
        if self.choice.format == "safetensors":  # 安全格式就是原样另存
            return self._save(named)
        if self.choice.format == "onnx":  # 图另存，权重跟着落旁边
            return self._onnx(named)
        if self.choice.format == "mlx":  # 交给 mlx_lm 自己转
            return self._mlx(named)
        kv = metadata(self.choice.config, self.choice.name, self.choice.quant, self.choice.dropped)  # 元数据转 kv
        if self.choice.with_tokenizer:  # 不导词表就不塞这些 kv
            kv.update(tokenizer(self._token_dir(), self.choice.dropped))  # 词表照着基座目录读，缺了记进 dropped
        blob = gguf_bytes(named, kv, self.choice.quant, self.choice.dropped)  # 拼整个文件，一回写出去
        self.choice.out.parent.mkdir(parents=True, exist_ok=True)
        self.choice.out.write_bytes(blob)  # 数据段一次性落盘
        return {"tensors": len(named), "bytes": len(blob), "dropped": self.choice.dropped}

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
        """导 ONNX：图只做词表投影，权重另外落一份 safetensors，dims 全在权重里。"""
        import torch  # 只有这条路要 torch
        from safetensors.numpy import save_file  # 权重落旁边，onnx 文件里只存图

        head = "token_embd.weight" if "token_embd.weight" in tensors else next(  # 前向用词表，没有就用第一个二维权
            name for name, one in tensors.items() if one.ndim == 2
        )
        weights = {name: torch.from_numpy(np.ascontiguousarray(one.astype(np.float32))) for name, one in tensors.items()}  # torch 张量

        class Project(torch.nn.Module):
            """把词表挂成参数的空图：前向就是一次投影，够导出、够对拍。"""

            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(weights[head])  # 词表当参数挂住

            def forward(self, ids):
                return torch.nn.functional.embedding(ids, self.weight)  # 查表，batch × 1 个 id → 隐藏向量

        self.choice.out.parent.mkdir(parents=True, exist_ok=True)
        side = self.choice.out.with_suffix(".safetensors")  # 权重文件，和 onnx 同名不同后缀
        save_file({name: one.numpy() for name, one in weights.items()}, str(side))  # 权重整份落盘
        # dynamo=False 走 TorchScript 那条老路：只要 torch 和 onnx，不额外要 onnxscript。
        torch.onnx.export(
            Project().eval(),  # 模块本身
            torch.zeros(1, 1, dtype=torch.long),  # 一批一个假 token id
            str(self.choice.out),  # 图写到哪
            input_names=["ids"],  # 输入名
            output_names=["hidden"],  # 输出名
            opset_version=14,  # 14 是各家运行时都吃得下的版本
            dynamo=False,  # 走老 exporter，见上面那行
        )
        kept = self.choice.out.stat().st_size + side.stat().st_size  # 图加权重一起算产出
        return {"tensors": len(tensors), "bytes": kept, "dropped": self.choice.dropped + [f"权重另存 {side.name}，onnx 只存图"]}

    def _mlx(self, tensors: dict) -> dict:
        """导 MLX：交给 mlx_lm.convert 子进程，产物是一个目录。"""
        from safetensors.numpy import save_file  # 合并过的权重得先落盘才交给子进程

        folder = self.choice.out  # mlx 的产物是目录，不是单文件
        folder.parent.mkdir(parents=True, exist_ok=True)
        source = Path(self.choice.source)  # 一般直接指基座目录
        with tempfile.TemporaryDirectory() as scratch:  # 合并结果基座目录上没有，摊到临时目录去
            if self.choice.tensors:  # 不摊的话转出来的是没合并的基座
                source = Path(scratch)
                save_file(tensors, str(source / "model.safetensors"))  # 合并后的权重
                shutil.copy2(self.choice.config, source / "config.json")  # 结构配置照抄
                words = Path(self.choice.tokenizer or self.choice.source)  # 词表跟着基座，没有就用权重目录
                for one in ("tokenizer.json", "tokenizer.model", "tokenizer_config.json"):  # 词表也带上
                    if (words / one).exists():
                        shutil.copy2(words / one, source / one)
            run = subprocess.run(  # 转换器要整目录，指给它要转的那份
                [sys.executable, "-m", "mlx_lm.convert", "--input", str(source), "--mlx-path", str(folder)],
                capture_output=True,  # 输出自己收
                text=True,
            )
        if run.returncode:  # 转失败了把最后一段 stderr 摆出来
            fail("error", f"mlx 转换失败: {run.stderr.strip()[-400:]}", hint="单独跑一次 mlx_lm.convert 看完整输出")
        written = sum(one.stat().st_size for one in folder.rglob("*") if one.is_file())  # 目录里写出来的全部字节
        return {"tensors": len(tensors), "bytes": written, "dropped": self.choice.dropped}


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
    """q8_0：每 32 个值一块，一块一个 fp16 比例加 32 个 int8。"""
    out = bytearray()
    for row in rows.astype(np.float32):  # 一行行来
        for start in range(0, row.size, 32):  # 32 个一段
            chunk = row[start : start + 32]
            d = np.float16(max(abs(chunk).max(), 1e-8) / 127)  # 比例存 fp16，除 127 落在 int8 上半区
            step = float(d)  # 量化用存下去的那个值再算，读回来才对得上
            q = np.clip(np.rint(chunk / step), -127, 127).astype(np.int8)  # 归一到 int8
            out += d.tobytes() + q.tobytes()  # 比例在前，32 个 int8 在后
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


def q6_k(rows: np.ndarray) -> bytes:
    """q6_k：每 256 个值一块，6 位有符号，低 4 位与高 2 位分开存。"""
    out = bytearray()
    for row in rows.astype(np.float32):  # 一行行来
        for start in range(0, row.size, 256):  # 256 个值一个超块
            block = row[start : start + 256].reshape(16, 16)  # 16 个子块，每块 16 个值
            top = block[np.arange(16), np.abs(block).argmax(1)]  # 各子块绝对值最大的那个值，符号跟着它
            scale = -top / 32  # 子块的最优比例，符号与极值相反，是 make_qx_quants 的约定
            peak = scale[int(np.abs(scale).argmax())]  # 超块里绝对值最大的那个比例，带符号
            if not peak:  # 整块全零，比例和量化值全留 0
                out += bytes(208) + np.float16(0).tobytes()  # 128+64+16 个零加一个 0 比例
                continue
            iscale = -128 / peak  # 比例之间换算用的倒数，照 ggml 的写法
            ls = np.clip(np.rint(iscale * scale), -128, 127).astype(np.int8)  # 子块比例相对超块取整，-128 也留着
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
