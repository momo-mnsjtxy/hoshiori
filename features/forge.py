"""导出：把 safetensors 权重转成 GGUF 二进制，按档量化，或原样另存 safetensors。

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
  q5_k_m 一块 256 个值 = 2 个 fp16 比例 + 12 字节 6bit 比例 + 32 字节第 5 位 + 128 字节低 4 位，合计 176 字节
q5_k_m 里 5 位拆成低 4 位和一张第 5 位的位图，是 ggml 的 Q5_K 超块结构；
行宽不够 256 的矩阵退回 q8_0，一维的范数、偏置一律 fp32，跟 llama.cpp 的做法一致。
"""

import json
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from utils.fail import fail

GGUF_MAGIC = b"GGUF"  # 文件头四字节，固定这串
GGUF_VERSION = 3  # 写死 v3，v2 少了 kv 里的对齐字段
ALIGN = 32  # 数据段对齐到 32 字节，ggml 读的时候按这个数跳
GGML_F32, GGML_F16, GGML_Q8_0, GGML_Q5_K = 0, 1, 8, 13  # ggml 的类型号
QUANTS = {"f32": GGML_F32, "f16": GGML_F16, "q8_0": GGML_Q8_0, "q5_k_m": GGML_Q5_K}  # 认的档
FILE_TYPE = {"f32": 0, "f16": 1, "q8_0": 7, "q5_k_m": 17}  # general.file_type 的取值，11 是 q5_k_m 那一档
BLOCK = {GGML_Q8_0: 32, GGML_Q5_K: 256}  # 一块装多少个值
BLOCK_BYTES = {GGML_Q8_0: 34, GGML_Q5_K: 176}  # 一块占多少字节，回读断言用它

K_U32, K_F32, K_BOOL, K_STRING = 4, 6, 7, 8  # kv 值的类型号，只用到这四种

ARCHS = {"llama", "qwen2", "qwen3", "mistral", "gemma", "gemma2", "phi3", "deepseek2"}  # 认得模型结构

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
    format: str = "gguf"  # safetensors 或 gguf
    quant: str = "f16"  # 量化档，f16 就是不量化
    config: Path = None  # config.json，没有就不写结构参数
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
        kv = metadata(self.choice.config, self.choice.name, self.choice.quant, self.choice.dropped)  # 元数据转 kv
        blob = gguf_bytes(named, kv, self.choice.quant, self.choice.dropped)  # 拼整个文件，一回写出去
        self.choice.out.parent.mkdir(parents=True, exist_ok=True)
        self.choice.out.write_bytes(blob)  # 数据段一次性落盘
        return {"tensors": len(named), "bytes": len(blob), "dropped": self.choice.dropped}

    def _save(self, tensors: dict) -> dict:
        """另存成 safetensors，原文件的元数据跟着带过去。"""
        from safetensors.numpy import save_file  # 只有这条路要库，进来再导

        single = Path(self.choice.source).is_file()  # 目录进来的没有元数据
        meta = read_metadata(self.choice.source) if single else {}  # 单个文件的元数据原样带
        self.choice.out.parent.mkdir(parents=True, exist_ok=True)
        save_file(tensors, str(self.choice.out), metadata=meta)  # 整份写出去
        return {"tensors": len(tensors), "bytes": self.choice.out.stat().st_size, "dropped": self.choice.dropped}


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


# --- 量化 ---

def block_bytes(kind: int, count: int) -> int:
    """按类型算 count 个值要占多少字节。回读断言和自检都用它。"""
    if kind in BLOCK:  # 有块的按块算
        return (count // BLOCK[kind]) * BLOCK_BYTES[kind]
    return count * (2 if kind == GGML_F16 else 4)  # fp16 两字节，fp32 四字节


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
    """q5_k（q5_k_m 的块本体）：每 256 个值一个超块，5 位量化。"""
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


def prepare(tensor: np.ndarray, quant: str, name: str, dropped: list):
    """决定一个张量怎么存，返回（字节, 类型号）。一维的、行宽不够的都退回 fp32。"""
    if tensor.ndim != 2:  # 范数、偏置这类一维量，还有极少数三维的，都留 fp32
        if quant in ("q8_0", "q5_k_m"):
            dropped.append(f"{name} 不是二维，留 fp32")
        return tensor.astype(np.float32).tobytes(), GGML_F32
    if quant in ("q8_0", "q5_k_m") and tensor.shape[1] % 32:  # 行宽连 32 都对不齐，量化不了
        dropped.append(f"{name} 行宽 {tensor.shape[1]} 不到 32 的倍数，留 fp32")
        return tensor.astype(np.float32).tobytes(), GGML_F32
    if quant == "q5_k_m" and tensor.shape[1] % 256 == 0:  # 行宽整好是 256 的倍数才能上超块
        return q5_k(tensor), GGML_Q5_K
    if quant in ("q8_0", "q5_k_m"):  # q8_0 只要求 32 对齐，行宽不够 256 的 q5_k 也落这里
        return q8_0(tensor), GGML_Q8_0
    kind = QUANTS[quant]
    return tensor.astype(np.float16 if kind == GGML_F16 else np.float32).tobytes(), kind  # f16 / f32


# --- 写文件 ---

def put_text(buf: bytearray, text: str) -> None:
    """写一个 GGUF 字符串：u64 长度 + 字节。"""
    raw = text.encode("utf-8")
    buf += struct.pack("<Q", len(raw)) + raw


def put_kv(buf: bytearray, key: str, kind: int, value) -> None:
    """写一条 kv：键、类型、值。"""
    put_text(buf, key)  # 键
    buf += struct.pack("<I", kind)  # 类型
    if kind == K_STRING:  # 字符串值有自己的长度前缀
        put_text(buf, str(value))
        return
    buf += struct.pack({K_U32: "<I", K_F32: "<f", K_BOOL: "<?", K_STRING: "<Q"}[kind], value)  # 标量值


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
        if kind == K_STRING:  # 字符串值
            kv[key], pos = _get_text(raw, pos)
            continue
        fmt, step = {K_U32: ("<I", 4), K_F32: ("<f", 4), K_BOOL: ("<?", 1)}[kind]
        kv[key] = struct.unpack_from(fmt, raw, pos)[0]
        pos += step
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
    for name, shape, kind, at in infos:  # 按偏移切出每份数据
        size = block_bytes(kind, int(np.prod(shape)) if shape else 1)
        tensors[name] = raw[start + at : start + at + size]
    out = {"magic": raw[:4], "version": version, "kv": kv, "data_len": data_len}  # 头、版本、kv、数据段长度
    out["infos"], out["tensors"] = infos, tensors  # 张量信息与按偏移切出来的字节
    return out


def _get_text(raw: bytes, pos: int):
    """读一个 GGUF 字符串，返回（文本, 新位置）。"""
    size = struct.unpack_from("<Q", raw, pos)[0]
    pos += 8
    return raw[pos : pos + size].decode("utf-8"), pos + size
