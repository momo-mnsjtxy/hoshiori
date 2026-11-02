"""LoRA 合并：把 adapter 的低秩增量按 alpha/r 放大，加回基座权重。

调用示例：
  from features.merge import Adapter, merge
  base = {"model.layers.0.self_attn.q_proj.weight": W}      # 基座，名字 → 数组
  out = merge(base, [Adapter(Path("out/adapter"))])         # 返回合并后的新字典
  out["model.layers.0.self_attn.q_proj.weight"]             # 已是 W + B@A*(alpha/r)

adapter 是一个目录或一份 safetensors，张量名成对出现：
  <层名>.lora_A.weight   形状 (r, in)
  <层名>.lora_B.weight   形状 (out, r)
旁边的 adapter_config.json 里有 r 和 lora_alpha，没给就用参数里的值。

合并在 fp32 里算，算完保持 fp32：fp16 里直接加会把小增量抹平。
多个 adapter 按列表顺序依次叠上去，先叠的和后叠的等价于分别加。
基座里找不到对应的名字就当场报错，写清是哪个 adapter 的哪一对，别静默少加。
"""

from pathlib import Path

import numpy as np

from utils.fail import fail

SUFFIX_A, SUFFIX_B = ".lora_A.weight", ".lora_B.weight"  # 一对低秩矩阵的名字尾巴


class Adapter:
    """一个 adapter：从哪读、放大多少。r 和 alpha 决定放大倍数 alpha/r。"""

    def __init__(self, path: Path, alpha: float = None, r: int = None) -> None:
        self.path = Path(path)  # adapter 目录或文件
        config = (self.path if self.path.is_dir() else self.path.parent) / "adapter_config.json"
        data = {}  # 配置里读出来的东西
        if config.exists():  # 有配置就先读，参数里给了的后面覆盖
            import json

            data = json.loads(config.read_text(encoding="utf-8"))
        self.variant = data.get("variant") or ("loha" if data.get("peft_type") == "LOHA" else "dora" if data.get("use_dora") else "lora")  # 配置决定实际权重公式
        self.r = int(r or data.get("r") or 16)  # 秩数，缺了就按 16
        self.alpha = float(alpha or data.get("lora_alpha") or self.r)  # 缩放分子，缺了就按 r，倍数等于 1

    @property
    def scale(self) -> float:
        """增量要乘的倍数。"""
        return self.alpha / self.r

    def tensors(self) -> dict:
        """读出挂件张量；三种变体都复用同一文件读取路径。"""
        from features.forge import read_tensors                                   # 共用 BF16 位解码，不让 numpy 假装支持 bfloat16
        files = sorted(self.path.glob("*.safetensors")) if self.path.is_dir() else [self.path]
        if not files:
            fail("input", f"adapter 里没有 safetensors: {self.path}")
        state = {}                                                               # 张量保留磁盘上的完整名称
        for path in files:
            state.update({name: value.astype(np.float32) for name, value in read_tensors(path).items()})
        return state

    def pairs(self) -> dict:
        """兼容标准 LoRA 的旧接口，返回基座权重名到 A/B 的映射。"""
        state = self.tensors()                                                   # 读取一次全部张量
        found = {}                                                               # 每层都必须有完整的一对
        for name in state:
            if name.endswith(SUFFIX_A):
                base = _base(name, SUFFIX_A)
                if base + SUFFIX_B not in state:
                    fail("input", f"adapter 里 {base} 只配了一半")
                found[base] = (state[name], state[base + SUFFIX_B])
            elif name.endswith(SUFFIX_B) and _base(name, SUFFIX_B) + SUFFIX_A not in state:
                fail("input", f"adapter 里 {_base(name, SUFFIX_B)} 只配了一半")
        return found

    def apply(self, base: dict) -> None:
        """按存档变体更新一份 fp32 基座；缺张量在读取边界明确报错。"""
        state = self.tensors()                                                   # 不把 LoHa 当 LoRA 配对而静默跳过
        first = ".hada_w1_a" if self.variant == "loha" else SUFFIX_A
        names = [name[:-len(first)] for name in state if name.endswith(first)]
        if not names:
            fail("input", f"adapter 没有 {self.variant} 权重: {self.path}")
        suffixes = [".hada_w1_a", ".hada_w1_b", ".hada_w2_a", ".hada_w2_b"] if self.variant == "loha" else [SUFFIX_A, SUFFIX_B]
        if self.variant == "dora":
            suffixes.append(".lora_magnitude_vector")                            # DoRA 缺幅值就不能重建有效权重
        for name in names:
            if name not in base:
                fail("input", f"基座里没有 {name}", hint=f"核对 {self.path} 挂载的基座")
            missing = [suffix for suffix in suffixes if name + suffix not in state]
            if missing:
                fail("input", f"adapter 里 {name} 缺少 {missing[0]}")
            if self.variant == "loha":
                delta = (state[name + suffixes[0]] @ state[name + suffixes[1]]) * (state[name + suffixes[2]] @ state[name + suffixes[3]])
            else:
                delta = state[name + SUFFIX_B] @ state[name + SUFFIX_A]             # 常规低秩方向更新
            weight = base[name] + delta * self.scale                              # 先合方向，再处理 DoRA 幅值
            if self.variant == "dora":
                norm = np.maximum(np.linalg.norm(weight, axis=1), 1e-12)
                weight *= (state[name + ".lora_magnitude_vector"] / norm)[:, None]
            base[name] = weight                                                   # 后一个挂件以此结果为起点


def _base(name: str, suffix: str) -> str:
    """去掉 A/B 尾巴，得到基座里的权重名。"""
    return name.replace(suffix, "")


def merge(base: dict, adapters: list) -> dict:
    """把每个 adapter 的增量加进去，返回新字典，不动传进来的基座。"""
    out = {name: one.astype(np.float32) for name, one in base.items()}  # 拷成 fp32，下面直接改
    for adapter in adapters:                                                     # 顺序叠加，DoRA 的归一使次序有意义
        adapter.apply(out)                                                       # 每个挂件知道自己的权重公式
    return out
