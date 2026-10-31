"""迁移：把 lmlab 的 preset yaml 转成星织的配方，顺带记下哪些字段没处放。

调用示例：
  from features import migration
  job = migration.convert(preset, where="configs/sft.yaml")   # 转一份
  job.recipe      # 转好的配方，可以直接写文件
  job.report()    # 逐条差异清单，人对着改
  job.blocked     # 缺基座路径就是 True，转完也跑不起来

做法是照表直落，不猜语义。表里有的照搬；表里没有的原样记进"待确认"，
既不静默丢掉，也不自作主张塞到别处。丢掉的那几类要写清为什么丢。
"""

import json
from dataclasses import dataclass, field

from utils import yaml_io
from utils.fail import fail

# lmlab 的字段 → 星织的字段。值是 None 表示只记一笔、不搬。
MAP = {
    "train.batch_size": "train.batch_size",
    "train.learning_rate": "train.lr",
    "train.num_epochs": "train.epochs",
    "train.seed": "train.seed",
    "train.max_grad_norm": "train.max_grad_norm",
    "train.warmup_steps": "train.warmup_steps",
    "train.gradient_accumulation_steps": "train.grad_accum",
    "train.gradient_checkpointing": "train.grad_checkpoint",
    "peft.rank": "train.lora.r",
    "peft.alpha": "train.lora.alpha",
    "peft.dropout": "train.lora.dropout",
    "peft.target_modules": "train.lora.target",
    "finetune.base_checkpoint_path": "model.path",
    "data.train_path": "data.files",
    "peft.quant_type": "train.precision",  # nf4 直接对上我们的精度档
}

# 值需要顺手改一下类型的：lmlab 的 alpha 写成 16.0，星织收整数
CAST = {"peft.alpha": lambda one: round(float(one))}

# 这几类有意不搬，写清为什么，免得接手者以为漏了
SKIPPED = {
    "finetune.freeze_encoder": "还没做逐层冻结，整层一起训",
    "finetune.freeze_embeddings": "还没做逐层冻结，整层一起训",
    "finetune.freeze_output_projection": "还没做逐层冻结，整层一起训",
}

MODE_TASK = {"pretrain": "pretrain", "finetune": "sft", "sft": "sft", "distill": "distill"}  # 训练模式换名

# 模型几何、可视化这些节由基座 config 决定，整节记成丢弃不进配方
GEOMETRY = ("model.", "objective.", "inference.", "visualization.", "dashboard.")

FINE_TUNE = {"sft", "distill", "dpo", "grpo", "ppo", "rloo"}


@dataclass
class Job:
    """一次迁移。三份清单随转随记。"""

    recipe: dict = field(default_factory=dict)
    mapped: list = field(default_factory=list)
    unknown: list = field(default_factory=list)  # 没处放，要人看
    dropped: list = field(default_factory=list)  # 有意不搬，写清原因

    @property
    def blocked(self) -> bool:
        """缺基座路径就没法跑。"""
        return any(item["path"] == "model.path" for item in self.unknown)

    def report(self) -> str:
        """差异清单，三段分开列。"""
        lines = ["## 已搬运"]
        for item in self.mapped:
            lines.append(f"- `{item['from']}` = {json.dumps(item['value'], ensure_ascii=False)} → {item['to']}")
        lines.append(f"\n## 待人工确认（{len(self.unknown)}）")
        for item in self.unknown:
            lines.append(f"- `{item['path']}` = {json.dumps(item['value'], ensure_ascii=False)} ← {item['why']}")
        lines.append(f"\n## 丢弃（{len(self.dropped)}）")
        for item in self.dropped:
            lines.append(f"- `{item['path']}` = {json.dumps(item['value'], ensure_ascii=False)} ← {item['why']}")
        return "\n".join(lines)


def convert(preset: dict, where: str = "") -> Job:
    """转一份 preset。"""
    job = Job()
    mode = (preset.get("train") or {}).get("mode")  # 他们的训练模式
    task = MODE_TASK.get(str(mode))  # 换成我们的任务名
    if task:  # 认得就直接定下任务
        job.recipe["task"] = task
    else:  # 认不得的记下来，别猜
        job.unknown.append({"path": "train.mode", "value": mode, "why": "认不出的训练模式"})

    for path, value in _flatten(preset):  # 摊平后逐个字段过；顺序是特判 → 搬 → 丢 → 待确认
        if path == "train.mode":  # 模式换名后由上面那段定 task，这里只做记录
            job.mapped.append({"from": path, "to": "task", "value": task})
            continue
        if path == "finetune.method":  # 全量还是 LoRA，决定带不带 lora 节，值本身不搬
            job.mapped.append({"from": path, "to": "train.lora" if value == "lora" else "（全量）", "value": value})
            continue
        if path in SKIPPED:  # 有意不搬的那几项，原因写在 SKIPPED 里
            job.dropped.append({"path": path, "value": value, "why": SKIPPED[path]})
            continue
        if path in MAP:  # 表里有的照搬
            if path in CAST:  # 类型对不上先归整
                value = CAST[path](value)
            _put(job.recipe, MAP[path], value if path != "data.train_path" else [value])  # 单文件包成列表
            job.mapped.append({"from": path, "to": MAP[path], "value": value})  # 搬过的逐条记账
            continue
        if path in {"project_name", "output_root", "checkpoint_path"}:  # 由命令行接管
            job.dropped.append({"path": path, "value": value, "why": "输出目录改由命令行给"})
            continue
        if path.startswith(GEOMETRY):  # 几何跟着基座走
            job.dropped.append({"path": path, "value": value, "why": "模型几何由基座 config 决定"})
            continue
        job.unknown.append({"path": path, "value": value, "why": "星织里没有对应字段"})

    if job.recipe.get("task") in FINE_TUNE and "model" not in job.recipe:  # 缺基座就标阻塞
        job.unknown.append({"path": "model.path", "value": None, "why": "preset 没写基座，转完要手填"})
    if (preset.get("finetune") or {}).get("method") == "full":  # 全量微调不带 lora 节
        (job.recipe.get("train") or {}).pop("lora", None)
    job.recipe.setdefault("version", 1)  # 配方格式版本，固定 1
    job.recipe.setdefault("name", where.rsplit("/", 1)[-1].rsplit(".", 1)[0] or "imported")  # 用文件名当名字
    return job


def _flatten(node, prefix: str = ""):
    """把嵌套 preset 摊成 点分路径 → 值。"""
    if isinstance(node, dict):  # 还是映射就继续往里拆
        for key, value in node.items():
            yield from _flatten(value, f"{prefix}.{key}" if prefix else key)  # 前缀一层层接上去
        return
    yield prefix, node  # 到了叶子，交出路径和值


def _put(recipe: dict, path: str, value) -> None:
    """按点分路径往结果里塞值，中间层不存在就建。"""
    parts = path.split(".")
    node = recipe
    for part in parts[:-1]:  # 中间层没有就建一个空映射
        node = node.setdefault(part, {})
    node[parts[-1]] = value  # 末段落值


def read_preset(path) -> dict:
    """读一份 preset 文件。"""
    return yaml_io.read(path)  # 读法跟配方一样，直接用同一套


def save(job: Job, target) -> None:
    """把转换结果写成配方文件，头上带一句提醒。"""
    if job.blocked:
        head = "# 由 ori import lmlab 生成。缺 model.path，补上再跑\n"
    else:
        head = "# 由 ori import lmlab 生成，对照 .diff.md 复核后跑\n"
    yaml_io.write(target, job.recipe, head)
