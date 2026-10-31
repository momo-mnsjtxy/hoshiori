"""低秩挂件：给指定的线性层接一条 A/B 旁路，只训这条旁路，基座一个数都不动。

调用示例：
  from features import lora
  lora.attach(model, ["w_q", "w_v"], r=8, alpha=16, dropout=0.05)  # 挂上，返回挂了几层
  lora.trainable(model)          # 只留 A/B 可训，其余参数全部冻死，返回 A/B 列表
  lora.save(model, Path("out/1"))  # 只存 A/B，名字用 merge.py 认的 .lora_A / .lora_B
  merged = lora.merge(model)     # 把 B@A×alpha/r 加回基座，返回 名字 → 张量

前向是 base(x) + B@A@x×alpha/r。A 用 kaiming 起，B 全零，所以刚挂上时输出和基座一模一样。
targets 认简写：w_q / w_k / w_v / w_o / w_gate / w_up / w_down；也认完整后缀 q_proj、mlp.down_proj 这些。
认不出任何一层当场报错，免得配方拼错了还静默跑完。
"""

from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn

from features.model import whole
from utils.fail import fail

SUFFIX_A, SUFFIX_B = ".lora_A.weight", ".lora_B.weight"  # 一对低秩矩阵的名字尾巴，merge.py 按它剥基座名
TARGETS = {
    "w_q": ".q_proj",  # 查询投影
    "w_k": ".k_proj",  # 键投影
    "w_v": ".v_proj",  # 值投影
    "w_o": ".o_proj",  # 输出投影
    "w_gate": ".gate_proj",  # 前馈门控
    "w_up": ".up_proj",  # 前馈值那一路
    "w_down": ".down_proj",  # 前馈压回
}


class LowRank(nn.Module):
    """一层低秩旁路：基座照跑，旁边加一条 B@A@x 的增量。"""

    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float) -> None:
        super().__init__()
        self.base = base  # 原来的那层，权重冻着不动
        self.r = r  # 秩数
        self.scale = alpha / r  # 增量放大倍数
        self.A = nn.Parameter(torch.empty(r, base.in_features))  # 降维那一半，形状 (r, 进)
        self.B = nn.Parameter(torch.zeros(base.out_features, r))  # 升维那一半，先全零，挂上时输出等于基座
        nn.init.kaiming_uniform_(self.A, a=5**0.5)  # 跟 Linear 一样的起法
        self.dropout = nn.Dropout(dropout)  # 增量上的丢弃

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """基座输出 + 低秩增量，形状和基座一致。"""
        delta = self.dropout(x) @ self.A.T @ self.B.T  # 先进 A 再出 B，就是低秩那一截
        return self.base(x) + delta * self.scale  # 乘倍率加回基座


def attach(model: nn.Module, targets: list, r: int, alpha: float, dropout: float) -> int:
    """把 targets 指定的线性层逐个包成 LowRank，返回挂了几层。"""
    wanted = [TARGETS.get(one, one if one.startswith(".") else "." + one) for one in targets]  # 简写翻成后缀
    for one in model.parameters():  # 先全冻上，只放开后面挂的 A/B
        one.requires_grad_(False)
    count = 0  # 挂了几层
    for name, module in list(model.named_modules()):  # 名字要连着父模块一起看
        if not isinstance(module, nn.Linear):  # 只包线性层
            continue
        if not any(name.endswith(tail) for tail in wanted):  # 不在名单里就跳过
            continue
        parent = model.get_submodule(name.rsplit(".", 1)[0])  # 父模块
        setattr(parent, name.rsplit(".", 1)[1], LowRank(module, r, alpha, dropout))  # 就地换掉那一层
        count += 1  # 记一层
    if not count:  # 一层都没挂上说明名单拼错了
        fail("input", f"这些目标一层都没挂上: {targets}", where="train.lora.target", hint="可用: " + " / ".join(TARGETS))
    return count


def layers(model: nn.Module) -> dict:
    """找出身上全部的 LowRank 层，返回 层名 → 层。"""
    return {name: one for name, one in model.named_modules() if isinstance(one, LowRank)}


def trainable(model: nn.Module) -> list:
    """只留 A/B 可训，其余全部冻死，返回可训参数列表。"""
    for name, one in model.named_parameters():  # 按名字判断
        one.requires_grad_(name.endswith(".A") or name.endswith(".B"))  # 只有 A 和 B 放开
    return [one for one in model.parameters() if one.requires_grad]


def save(model: nn.Module, folder: Path, base: Path = None, write: bool = True) -> dict:
    """只把 A/B 存成 folder/adapter.safetensors，旁边写一份 adapter_config.json。

    base 是挂载的基座目录，写进配置给 forge 合并用；从随机权重起的就是空串。
    write 为假时只拼张量不落盘，理由同 Model.save：fsdp2 下拼分片每个进程都得走一遍。
    """
    import json

    found = layers(model)  # 全部挂件
    state = {}  # 要落盘的张量
    for name, one in found.items():  # 一层的 A、B 配成一对
        state[name + ".weight" + SUFFIX_A] = whole(one.A)  # 基座名 + lora_A
        state[name + ".weight" + SUFFIX_B] = whole(one.B)  # 基座名 + lora_B
    if write:  # 只有 0 号真写文件
        folder.mkdir(parents=True, exist_ok=True)  # 目录先建
        save_file(state, str(folder / "adapter.safetensors"))  # 一对对写出去
        first = next(iter(found.values()))  # 取一层问形状，配置里要写清维度
        config = {"r": first.r, "lora_alpha": round(first.scale * first.r, 6), "target_modules": list(found), "peft_type": "LORA", "base_model_name_or_path": str(base or "")}  # merge.py 读 r 和 lora_alpha，forge 读基座路径
        (folder / "adapter_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")  # 配置跟权重放一起
    return state


def load(model: nn.Module, source: Path) -> None:
    """从文件或目录把 A/B 灌回已经挂好的那些层。"""
    files = sorted(source.glob("*.safetensors")) if source.is_dir() else [source]  # 目录就全收
    state = {}  # 攒起来的张量
    for one in files:  # 逐个文件读
        state.update(load_file(str(one), device="cpu"))  # 并进同一份
    with torch.no_grad():  # 参数是叶子又要梯度，原地灌得关掉记录
        for name, one in layers(model).items():  # 只灌自己认识的层
            for tensor, tail in ((one.A, SUFFIX_A), (one.B, SUFFIX_B)):  # A、B 各管各的
                key = name + ".weight" + tail  # 对应的键
                if key not in state:  # 缺一半就是坏档
                    fail("input", f"adapter 里没有 {key}", where=str(source), hint="挂载目标和存盘那份要一致")
                tensor.copy_(state[key].to(tensor.dtype))  # 灌回参数


def merge(model: nn.Module) -> dict:
    """把每层增量按 alpha/r 加回基座，返回 权重名 → 张量，不带任何 lora 键。"""
    out = {}  # 合并后的整份权重
    for name, tensor in model.state_dict().items():  # 先拷一遍基座，跳过挂件的 A/B
        if name.endswith(".A") or name.endswith(".B"):  # 挂件本身不进产物
            continue
        out[_plain(name)] = tensor.detach().clone().float()  # 统一在 fp32 里算，小数增量才不被抹平
    for name, one in layers(model).items():  # 一层一层加增量
        key = name + ".weight"  # 基座里对应的权重名
        out[key] += one.B.detach().float() @ one.A.detach().float() * one.scale  # B@A 就是低秩增量
    return out


def _plain(name: str) -> str:
    """把包在挂件里的权重名还原成基座的名字。"""
    return name.replace(".base.weight", ".weight")  # LowRank 把原层藏成 base 子模块，产物里不该带这一层
