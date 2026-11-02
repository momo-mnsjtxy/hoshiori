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
from pathlib import Path  # 外部配置的相对路径按源文件定位
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
        return any(item["path"] == "model.path" or item.get("blocking") for item in self.unknown)

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


def convert(preset: dict, where: str = "", framework: str = "lmlab") -> Job:
    """转一份 preset。"""
    if framework != "lmlab":  # 旧入口继续默认 lmlab，新来源显式选择
        return convert_framework(preset, where, framework)
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
        head = "# 由 ori import 生成。仍有阻塞项，请看 .diff.md 补完再跑\n"
    else:
        head = "# 由 ori import 生成，对照 .diff.md 复核后跑\n"
    yaml_io.write(target, job.recipe, head)


# --- Axolotl / LLaMA-Factory 配置 ---

COMMON = {  # 两家都直接使用的训练字段，不把相同映射写两遍
    "learning_rate": "train.lr", "num_train_epochs": "train.epochs",
    "gradient_accumulation_steps": "train.grad_accum", "seed": "train.seed",
    "max_grad_norm": "train.max_grad_norm", "warmup_steps": "train.warmup_steps",
    "weight_decay": "train.weight_decay", "max_steps": "train.max_steps",
    "save_steps": "train.save_every", "logging_steps": "train.log_every",
    "lora_alpha": "train.lora.alpha", "lora_dropout": "train.lora.dropout",
}
FRAMEWORKS = {  # 官方字段含义按来源分开，便于对照迁移报告
    "axolotl": {"base_model": "model.path", "micro_batch_size": "train.batch_size",
                "num_epochs": "train.epochs", "sequence_len": "model.geometry.max_seq_len",
                "gradient_checkpointing": "train.grad_checkpoint", "lora_r": "train.lora.r",
                "lora_target_modules": "train.lora.target"},
    "llama-factory": {"model_name_or_path": "model.path", "per_device_train_batch_size": "train.batch_size",
                      "cutoff_len": "model.geometry.max_seq_len", "lora_rank": "train.lora.r",
                      "lora_target": "train.lora.target"},
}


def convert_framework(preset: dict, where: str, framework: str) -> Job:
    """迁移两家常用 SFT/预训练配置；未支持的语义逐条保留，绝不假称等价。"""
    if framework not in FRAMEWORKS:  # framework 是公开入口，不接受拼错的来源
        fail("input", f"不认识迁移来源 {framework}")
    source = Path(where or ".").resolve()  # 数据路径按原配置所在位置确定
    job = Job(recipe={"version": 1, "name": source.stem or "imported"})
    task = preset.get("stage", "sft") if framework == "llama-factory" else preset.get("rl", "sft")
    tasks = {"pt": "pretrain", "sft": "sft", "dpo": "dpo", "ppo": "ppo", "grpo": "grpo"}
    if task not in tasks:  # 没有对应任务时阻止产物被当成可直接运行
        job.unknown.append({"path": "task", "value": task, "why": "这个训练阶段尚不支持", "blocking": True})
    else:
        job.recipe["task"] = tasks[task]  # 同名任务或明确的 pt → pretrain 转换
    mapping = {**COMMON, **FRAMEWORKS[framework]}  # 来源专用字段覆盖共同字段
    handled = {"datasets", "dataset", "dataset_dir", "stage", "rl", "adapter", "finetuning_type",
               "bf16", "fp16", "load_in_4bit", "load_in_8bit", "quantization_bit", "lora_target_linear"}
    for key, value in preset.items():  # 外部顶层字段只在这里解析一次
        if key in mapping:
            if mapping[key] == "model.path" and (source.parent / str(value)).is_dir():
                value = str((source.parent / str(value)).resolve())  # 本地基座迁出后仍指向原目录
            if mapping[key] == "train.lora.target":  # 两家都允许用逗号分隔目标名
                value = [part.strip() for part in value.split(",")] if isinstance(value, str) else value
                if value == ["all"]:  # all-linear 显式展开为本模型真实支持的七类投影
                    value = ["w_q", "w_k", "w_v", "w_o", "w_gate", "w_up", "w_down"]
            if mapping[key] in {"train.lr", "train.weight_decay", "train.max_grad_norm", "train.lora.dropout"}:
                value = float(value)  # yaml 科学计数法有时被读成字符串
            _put(job.recipe, mapping[key], value)  # 映射结果仍走统一的嵌套写入
            job.mapped.append({"from": key, "to": mapping[key], "value": value})
        elif key not in handled:
            job.unknown.append({"path": key, "value": value, "why": "尚无等价字段，原值保留供复核"})
    precision = "bf16" if preset.get("bf16") is True else "fp16" if preset.get("fp16") is True else "fp32"
    bits = preset.get("quantization_bit", 4 if preset.get("load_in_4bit") else 8 if preset.get("load_in_8bit") else 0)
    if bits not in (0, 4, 8):  # 未支持量化不能悄悄落为全精度
        job.unknown.append({"path": "quantization_bit", "value": bits, "why": "只支持 4/8 位量化", "blocking": True})
    precision = {4: "nf4", 8: "int8"}.get(bits, precision)  # QLoRA 与 int8 映射明确
    _put(job.recipe, "train.precision", precision)
    job.mapped.append({"from": "precision/quantization", "to": "train.precision", "value": precision})
    method = preset.get("adapter", "") if framework == "axolotl" else preset.get("finetuning_type", "full")
    if method not in ("", "full", "lora", "qlora"):  # freeze 等方式不能等价冒充 full
        job.unknown.append({"path": "finetuning_type", "value": method, "why": "该微调方式没有等价实现", "blocking": True})
    if method in ("", "full"):  # 全量训练不能带着残留的 LoRA 字段
        job.recipe["train"].pop("lora", None)
    else:
        _put(job.recipe, "train.lora.r", job.recipe["train"].get("lora", {}).get("r", 8))
        if preset.get("lora_target_linear") or not job.recipe["train"]["lora"].get("target"):
            _put(job.recipe, "train.lora.target", ["w_q", "w_k", "w_v", "w_o", "w_gate", "w_up", "w_down"])
    files = dataset_files(preset, source, framework, job)  # 注册表解析与路径定位集中在一个动作
    if files:
        _put(job.recipe, "data.files", files)
        job.mapped.append({"from": "datasets" if framework == "axolotl" else "dataset", "to": "data.files", "value": files})
    if not (job.recipe.get("model") or {}).get("path"):
        job.unknown.append({"path": "model.path", "value": None, "why": "缺基座路径"})
    return job


def dataset_files(preset: dict, source: Path, framework: str, job: Job) -> list:
    """把本地训练文件定位为绝对路径；Hub 数据和特殊字段格式保留为待处理项。"""
    records = preset.get("datasets", []) if framework == "axolotl" else []
    directory = source.parent  # Axolotl 相对文件跟随配置文件
    if framework == "llama-factory":
        directory = source.parent / preset.get("dataset_dir", "data")  # 官方默认目录名
        registry_path = directory / "dataset_info.json"  # 名称到文件的注册表
        registry = json.loads(registry_path.read_text()) if registry_path.is_file() else {}
        names = preset.get("dataset", [])  # CLI 常用逗号分隔数据集名
        names = [name.strip() for name in names.split(",")] if isinstance(names, str) else names
        records = [dict(registry.get(name, {}), name=name) for name in names]
    files = []  # 只把明确存在的本地文件交给训练器
    for item in records:
        record = {"path": item} if isinstance(item, str) else item
        given = record.get("path") or record.get("file_name")
        path = directory / given if given else None  # 绝对给定路径会覆盖 directory
        if path is None or not path.is_file():
            job.unknown.append({"path": "data.files", "value": record, "why": "数据不是现有本地文件；先下载或补 dataset_info.json", "blocking": True})
            continue
        files.append(str(path.resolve()))  # 迁出到另一个目录后仍然找得到原语料
        extras = {key: value for key, value in record.items() if key not in {"path", "file_name", "name"}}
        if extras:  # type/template/columns 不具有通用等价转换，明确要求核对
            job.unknown.append({"path": "dataset.format", "value": extras, "why": "请确认 JSONL 已是星织 text/messages 格式；未自动转换模板"})
    if not records:  # 缺训练数据同样不应该声称可直接运行
        job.unknown.append({"path": "data.files", "value": None, "why": "没有训练数据", "blocking": True})
    return files
