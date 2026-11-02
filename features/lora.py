"""低秩挂件：给指定的线性层接一条 A/B 旁路，只训这条旁路，基座一个数都不动。

调用示例：
  from features import lora
  lora.attach(model, ["w_q", "w_v"], r=8, alpha=16, dropout=0.05)  # 挂上，返回挂了几层
  lora.trainable(model)          # 只留 A/B 可训，其余参数全部冻死，返回 A/B 列表
  lora.save(model, Path("out/1"))  # 只存 A/B，名字用 merge.py 认的 .lora_A / .lora_B
  merged = lora.merge(model)     # 把 B@A×alpha/r 加回基座，返回 名字 → 张量

variant="dora" 使用可训幅值与低秩方向，variant="loha" 使用两组低秩的 Hadamard 积。
保存、载入和 merge 使用同一变体定义；不把浮点变体伪装成可支持量化权重。

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

    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float, variant: str = "lora") -> None:
        super().__init__()
        self.variant = variant  # lora 标准旁路、dora 幅值/方向分离、loha 两组低秩的逐元素积
        self.base = base  # 原来的那层，权重冻着不动
        self.r = r  # 秩数
        self.scale = alpha / r  # 增量放大倍数
        dtype = base.weight.dtype if base.weight.is_floating_point() else torch.float32  # 量化基座整数存储不能成为挂件 dtype
        self.A = nn.Parameter(torch.empty(r, base.in_features, device=base.weight.device, dtype=dtype))  # 降维那一半，形状 (r, 进)
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device, dtype=dtype))  # 升维那一半，先全零，挂上时输出等于基座
        nn.init.kaiming_uniform_(self.A, a=5**0.5)  # 跟 Linear 一样的起法
        if variant == "loha":                                                    # 第二组低秩不能也全零，否则 B 的梯度为零
            self.C = nn.Parameter(torch.empty_like(self.A))                       # 第二组输入方向
            self.D = nn.Parameter(torch.empty_like(self.B))                       # 第二组输出方向
            nn.init.kaiming_uniform_(self.C, a=5**0.5)                             # 非零起步保证第一组能学习
            nn.init.kaiming_uniform_(self.D, a=5**0.5)                             # 初始总增量仍因 B=0 而为零
        if variant == "dora":                                                    # 每个输出通道独立学习幅值
            self.magnitude = nn.Parameter(base.weight.detach().float().norm(dim=1))
        self.dropout = nn.Dropout(dropout)  # 增量上的丢弃

    def delta(self, dtype: torch.dtype = None) -> torch.Tensor:
        """返回增量矩阵；LoHa 使用两组低秩矩阵逐元素相乘。"""
        dtype = dtype or self.A.dtype  # 导出可显式要求整个乘法在 fp32 中进行
        change = self.B.to(dtype) @ self.A.to(dtype)                                                  # 第一组低秩恢复成权重形状
        if self.variant == "loha":
            change = change * (self.D.to(dtype) @ self.C.to(dtype))                                   # Hadamard 积提高可表达的有效秩
        return change * self.scale                                               # 三种变体共用 alpha/r

    def merged(self) -> torch.Tensor:
        """返回可导出的有效权重；DoRA 方向归一后乘学习到的幅值。"""
        weight = self.base.weight.float() + self.delta(torch.float32)                  # 在 fp32 合并，避免小增量被舍掉
        if self.variant == "dora":
            weight = weight * (self.magnitude.float() / weight.norm(dim=1).clamp_min(1e-12)).unsqueeze(1)
        return weight                                                            # 不修改冻结基座

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """返回基座与可训增量；DoRA 的归一范数停止求导，幅值仍可训练。"""
        base = self.base(x)                                                       # 量化基座保留自己的计算路径
        if self.variant == "lora":
            return base + (self.dropout(x) @ self.A.T @ self.B.T) * self.scale    # 常规 LoRA 避免生成完整增量矩阵
        delta = self.delta()                                                      # DoRA/LoHa 需要显式方向矩阵
        output = base + self.dropout(x) @ delta.T                                 # dropout 只作用于可训更新
        if self.variant == "dora":
            norm = (self.base.weight.float() + delta.float()).norm(dim=1).detach().clamp_min(1e-12)
            ratio = (self.magnitude.float() / norm).to(output.dtype)               # 每个输出通道有自己的幅值比例
            bias = self.base.bias if self.base.bias is not None else 0.0          # 幅值只缩放权重贡献，不缩放偏置
            output = (output - bias) * ratio + bias
        return output                                                            # 形状与原线性层相同


def attach(model: nn.Module, targets: list, r: int, alpha: float, dropout: float, variant: str = "lora") -> int:
    """把 targets 指定的线性层逐个包成 LowRank，返回挂了几层。"""
    if variant not in {"lora", "dora", "loha"} or r < 1 or not 0 <= dropout < 1:
        fail("input", "挂件变体、秩或 dropout 不合法", where="train.lora")
    if variant != "lora" and getattr(model, "quantized", False):
        fail("input", "DoRA/LoHa 当前需要浮点基座", where="train.lora.variant", hint="使用 fp32/bf16/fp16，或量化基座配标准 LoRA")
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
        setattr(parent, name.rsplit(".", 1)[1], LowRank(module, r, alpha, dropout, variant))  # 就地换掉那一层
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
        one.requires_grad_(name.rsplit(".", 1)[-1] in {"A", "B", "C", "D", "magnitude"})  # 只有 A 和 B 放开
    return [one for one in model.parameters() if one.requires_grad]


def save(model: nn.Module, folder: Path, base: Path = None, write: bool = True) -> dict:
    """只把 A/B 存成 folder/adapter.safetensors，旁边写一份 adapter_config.json。

    base 是挂载的基座目录，写进配置给 forge 合并用；从随机权重起的就是空串。
    write 为假时只拼张量不落盘，理由同 Model.save：fsdp2 下拼分片每个进程都得走一遍。
    """
    import json

    found = layers(model)  # 全部挂件
    state = {}  # 要落盘的张量
    for name, one in found.items():                                               # 每个变体按自己的实际可训张量存档
        for parameter, suffix in _names(one.variant).items():
            state[name + ".weight" + suffix] = whole(getattr(one, parameter))     # 只存挂件，不复制冻结基座
    if write:  # 只有 0 号真写文件
        folder.mkdir(parents=True, exist_ok=True)  # 目录先建
        save_file(state, str(folder / "adapter.safetensors"))  # 一对对写出去
        first = next(iter(found.values()))  # 取一层问形状，配置里要写清维度
        config = {"r": first.r, "lora_alpha": round(first.scale * first.r, 6), "target_modules": list(found), "peft_type": "LOHA" if first.variant == "loha" else "LORA", "variant": first.variant, "use_dora": first.variant == "dora", "base_model_name_or_path": str(base or "")}  # merge.py 读 r 和 lora_alpha，forge 读基座路径
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
            for parameter, tail in _names(one.variant).items():  # A、B 各管各的
                key = name + ".weight" + tail  # 对应的键
                if key not in state:  # 缺一半就是坏档
                    fail("input", f"adapter 里没有 {key}", where=str(source), hint="挂载目标和存盘那份要一致")
                tensor = getattr(one, parameter)  # 按变体找到这份可训参数
                tensor.copy_(state[key].to(tensor.dtype))  # 灌回参数


def merge(model: nn.Module) -> dict:
    """把每层增量按 alpha/r 加回基座，返回 权重名 → 张量，不带任何 lora 键。"""
    out = {}  # 合并后的整份权重
    for name, tensor in model.state_dict().items():  # 先拷一遍基座，跳过挂件的 A/B
        if name.rsplit(".", 1)[-1] in {"A", "B", "C", "D", "magnitude"}:  # 挂件本身不进产物
            continue
        out[_plain(name)] = tensor.detach().clone().float()  # 统一在 fp32 里算，小数增量才不被抹平
    for name, one in layers(model).items():  # 一层一层加增量
        key = name + ".weight"  # 基座里对应的权重名
        out[key] = one.merged().detach().clone()  # B@A 就是低秩增量
    return out


def _plain(name: str) -> str:
    """把包在挂件里的权重名还原成基座的名字。"""
    return name.replace(".base.weight", ".weight").replace(".base.bias", ".bias")  # LowRank 把原层藏成 base 子模块，产物里不该带这一层


def _names(variant: str) -> dict:
    """挂件参数名与磁盘后缀的唯一映射；保存与载入共用。"""
    if variant == "loha":
        return {"B": ".hada_w1_a", "A": ".hada_w1_b", "D": ".hada_w2_a", "C": ".hada_w2_b"}
    names = {"A": SUFFIX_A, "B": SUFFIX_B}                                        # 标准 LoRA 磁盘格式保持不变
    if variant == "dora":
        names["magnitude"] = ".lora_magnitude_vector"                            # 幅值不能在合并时从基座重新猜
    return names
