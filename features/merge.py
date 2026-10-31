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
        self.r = int(r or data.get("r") or 16)  # 秩数，缺了就按 16
        self.alpha = float(alpha or data.get("lora_alpha") or self.r)  # 缩放分子，缺了就按 r，倍数等于 1

    @property
    def scale(self) -> float:
        """增量要乘的倍数。"""
        return self.alpha / self.r

    def pairs(self) -> dict:
        """读出全部 A/B 对，返回 基座名 → (A, B)。"""
        from safetensors.numpy import load_file  # 只有这里要库

        files = sorted(self.path.glob("*.safetensors")) if self.path.is_dir() else [self.path]  # 目录全收
        if not files:  # 一个权重都没有
            fail("input", f"adapter 里没有 safetensors: {self.path}", hint="指到 adapter 目录或 .safetensors 文件")
        found = {}
        for one in files:  # 逐个文件往里并
            for name, tensor in load_file(str(one)).items():
                if SUFFIX_A in name:  # A 矩阵先记下，等它的 B
                    found.setdefault(_base(name, SUFFIX_A), [None, None])[0] = tensor.astype(np.float32)
                elif SUFFIX_B in name:  # B 矩阵补上另一半
                    found.setdefault(_base(name, SUFFIX_B), [None, None])[1] = tensor.astype(np.float32)
        for name, pair in found.items():  # 缺一半的对就是坏 adapter
            if pair[0] is None or pair[1] is None:
                fail("input", f"adapter 里 {name} 只配了一半", hint="A 和 B 要成对出现")
        return found


def _base(name: str, suffix: str) -> str:
    """去掉 A/B 尾巴，得到基座里的权重名。"""
    return name.replace(suffix, "")


def merge(base: dict, adapters: list) -> dict:
    """把每个 adapter 的增量加进去，返回新字典，不动传进来的基座。"""
    out = {name: one.astype(np.float32) for name, one in base.items()}  # 拷成 fp32，下面直接改
    for adapter in adapters:  # 一个 adapter 一趟
        for name, (left, right) in adapter.pairs().items():  # 一对一对来
            if name not in out:  # 基座里没有这个权重
                fail("input", f"基座里没有 {name}", hint=f"核对 {adapter.path} 是不是挂在别的模型上")
            out[name] += right @ left * adapter.scale  # B@A 是低秩增量，乘 alpha/r 再加
    return out
