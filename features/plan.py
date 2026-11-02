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
import json                            # 校准和价钱快照从用户指定的 JSON 文件加载
from pathlib import Path              # 相对路径跟配方目录走，不跟启动目录漂移

from features import cost

BYTES = {"bf16": 2, "fp16": 2, "fp32": 4, "fp8": 2, "nf4": 0.5, "int8": 1}  # FP8 前向仍保留 BF16 主权重
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
    profile: dict = None               # 真机短跑证据；没有时不生成看似已验证的配方
    price: dict = None                 # 带来源与时间的云价快照
    configuration: dict = None         # 与 profile 逐项匹配的实际训练配置

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
        lines.append(f"  显存估算  ≈{self.vram_gb:.1f} GiB / 卡")  # 理论估算与实测分开，不当硬件容量
        measured = cost.matching_run(self.profile, self.configuration) if self.profile else None
        if measured:
            lines.append(f"  实测峰值  {measured['peak_reserved_gib']:.3f} GiB · {self.profile['hardware']['name']}")
        for note in self.notes:  # 提醒跟在账后面
            lines.append(f"  ! {note}")
        if self.steps:  # 只有配方明确给了总步数，才有足够信息报成本
            tokens_per_step = self.tokens * self.batch * self.grad_accum  # 单卡一个更新步的 token 上限
            quote = cost.estimate(self, tokens_per_step=tokens_per_step, profile=self.profile,
                                  configuration=self.configuration, price=self.price)  # 缺少实测或单价则明确未知
            lines.extend(["", *cost.render(quote).splitlines()])  # 成本表与训练账隔一行
        lines.append("")  # 推荐另起一段，没步数时它就是唯一新增段
        lines.append("复用卡推荐")  # 只列同一硬件、同几何下实际量过的批量
        cards = cost.calibrated_recommend(self.profile, self.configuration) if self.profile else []
        if not cards:
            lines.append("  未知：缺少同模型、精度、训练配置的 GPU 校准；运行 ori cost calibrate")
        for card in cards:
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


def build(recipe: dict, base_dir=None) -> Plan:
    """从配方算出一张账。"""
    train = recipe.get("train") or {}  # 没写 train 节就当空
    geometry = (recipe.get("model") or {}).get("geometry") or {}  # 序列上限在模型几何里
    how = train.get("strategy", "single")  # 没写就单卡
    devices = [int(one) for one in train.get("devices") or []]  # 卡号转整数，写照应列表
    precision = train.get("precision", "fp32")  # 与 spin 真正开跑时的默认精度一致
    evidence = recipe.get("cost") or {}  # 只有显式提供的证据才参与费用和复用推荐
    base = Path(base_dir) if base_dir is not None else Path.cwd()
    profile = json.loads((base / evidence["profile"]).read_text(encoding="utf-8")) if evidence.get("profile") else None
    price = json.loads((base / evidence["price"]).read_text(encoding="utf-8")) if evidence.get("price") else None
    config = cost.configuration(recipe) if profile else None
    measured = cost.matching_run(profile, config) if profile else None
    actual_params = measured.get("parameters") if measured else None  # 真正匹配时优先用构建出来的参数数目
    params = float((recipe.get("model") or {}).get("params_b") or (actual_params / 1e9 if actual_params else 0.5))
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
        profile=profile,
        price=price,
        configuration=config,
    )
