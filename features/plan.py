"""训练计划：开跑前把参数摊成一张账，先看账再跑。

调用示例：
  from features import plan
  table = plan.build(recipe)
  print(table.show())          # 人看的账，带显存、成本和复用配方
  table.needs_gpu             # 这套精度要不要 GPU

数字是量级判断，不是精确值：乘完权重、梯度、优化器态、激活留量再按卡数切。
用来拦一眼就能看出要爆显存的配方，别拿它当容量保证。
"""

from dataclasses import dataclass, field

from features import cost

BYTES = {"bf16": 2, "fp16": 2, "fp32": 4, "fp8": 1, "nf4": 0.5, "int8": 1}  # 每参数占几个字节
ACTIVATION = 6.0  # 激活留量相对权重的倍数，经验值
FRAGMENT = 1.15  # 显存碎片余量
LORA_SHARE = 0.05  # 挂 LoRA 时优化器态只按 5% 参数算


@dataclass
class Plan:
    """一张训练账。"""

    task: str
    how: str
    devices: list
    precision: str
    batch: int
    grad_accum: int
    epochs: float
    lora: dict
    vram_gb: float
    notes: list = field(default_factory=list)
    steps: int = 0  # 配方明确给出的总更新步数；0 表示要由数据量和 epochs 决定
    tokens: int = 128  # 每条样本的最长文本 token 数，来自模型几何
    params_b: float = 0.5  # 模型十亿参数量，复用卡要拿它和显存一起分档

    @property
    def needs_gpu(self) -> bool:
        """这套精度要不要 GPU。"""
        return self.precision != "fp32"

    def show(self) -> str:
        """把账排成人读的几行。"""
        lines = [
            "训练计划",
            f"  任务      {self.task}",
            f"  方式      {self.how} · 设备 {self.devices or [0]}",
            f"  精度      {self.precision}",
            f"  批        单卡 {self.batch} × 累积 {self.grad_accum} × 卡 {max(1, len(self.devices))}",
            f"  轮        {self.epochs}",
        ]
        if self.lora:  # 挂 LoRA 才提，不挂不提
            lines.append(f"  LoRA      r={self.lora.get('r')} alpha={self.lora.get('alpha')}")
        lines.append(f"  显存      ≈{self.vram_gb:.1f} GiB / 卡")
        for note in self.notes:  # 提醒跟在账后面
            lines.append(f"  ! {note}")
        if self.steps:  # 只有配方明确给了总步数，才有足够信息报成本
            tokens_per_step = self.tokens * self.batch * self.grad_accum  # 单卡一个更新步的 token 上限
            quote = cost.estimate(self, tokens_per_step=tokens_per_step)  # 默认单价只作开跑前占位
            lines.extend(["", *cost.render(quote).splitlines()])  # 成本表与训练账隔一行
        lines.append("")  # 推荐另起一段，没步数时它就是唯一新增段
        lines.append("复用卡推荐")  # 三条都能直接抄回 recipe 的 train 节
        for card in cost.recommend(self.vram_gb, self.params_b):  # 按当前显存量级和模型大小分档
            lines.append(
                f"  {card['strategy']} · {card['precision']} · batch {card['batch_size']} — {card['why']}"
            )
        return "\n".join(lines)


def estimate(params_b: float, precision: str, how: str, cards: int) -> float:
    """估一张卡的峰值显存（GiB）。"""
    per = BYTES.get(precision, 2)  # 每个参数占几个字节
    weights = params_b * per  # 权重本体
    grads = 0.0 if precision == "nf4" else params_b * 2  # 量化时梯度在权重外，不按字节算
    optimizer = params_b * 8 * (LORA_SHARE if precision in {"nf4", "int8"} else 1.0)  # AdamW 的一阶二阶
    activation = params_b * per * ACTIVATION  # 前向激活，跟着精度走
    shard = cards if how == "fsdp2" else 1  # 只有 fsdp2 会把参数摊到各卡，别的都是一卡一份
    return (weights + grads + optimizer + activation) * FRAGMENT / max(1, shard)  # 加碎片余量再按卡切


def table_overflow(params_b: float, precision: str, how: str, cards: int, limit: float = 24.0) -> bool:
    """粗判会不会明显放不下：估算超过常见单卡容量就提示。"""
    return estimate(params_b, precision, how, cards) > limit


def build(recipe: dict) -> Plan:
    """从配方算出一张账。"""
    train = recipe.get("train") or {}  # 没写 train 节就当空
    geometry = (recipe.get("model") or {}).get("geometry") or {}  # 序列上限在模型几何里
    how = train.get("strategy", "single")  # 没写就单卡
    devices = [int(one) for one in train.get("devices") or []]  # 卡号转整数，写照应列表
    precision = train.get("precision", "bf16")  # 默认 bf16
    params = float((recipe.get("model") or {}).get("params_b") or 0.5)  # 没写就按 0.5B 估
    notes = []  # 收集要给用户看的提醒
    if how == "ddp":  # ddp 每张卡都存全量，省时间不省显存
        notes.append("ddp 每卡存全量权重，省的是时间不是显存")
    if how == "single" and len(devices) > 1:  # 给了卡却不用
        notes.append("给了多张卡却在 single 上跑，第二张往后都闲着")
    if precision == "fp16":  # fp16 要防梯度下溢
        notes.append("fp16 要先开 loss scale，否则梯度会掉成 0")
    if table_overflow(params, precision, how, max(1, len(devices))):  # 明显放不下就先说
        notes.append("按这个配置很可能爆显存，先把 batch_size 或 seq 长度压下来")
    return Plan(
        task=recipe.get("task", "sft"),
        how=how,
        devices=devices,
        precision=precision,
        batch=int(train.get("batch_size", 1)),
        grad_accum=int(train.get("grad_accum", 1)),
        epochs=float(train.get("epochs", 1)),
        lora=train.get("lora"),
        vram_gb=estimate(params, precision, how, max(1, len(devices))),
        notes=notes,
        steps=int(train.get("max_steps", 0)),
        tokens=int(geometry.get("max_seq_len", 128)),
        params_b=params,
    )
