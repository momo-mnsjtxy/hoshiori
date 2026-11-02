"""lm-eval 适配：把任务定义导成自己的套件，也能让答题后端把导入套件真跑一遍。

调用示例：
  from pathlib import Path
  from features import lm_eval, suite
  one = lm_eval.import_file(Path("tasks/demo.yaml"), suite.folder())
  one.tasks      # 导了几题
  one.dropped    # 认不出的指标名，进报告，不静默丢
  one.report     # 报告文件在哪
  scored = lm_eval.run_backend("printf 12", suite.read(one.path), limit=1)
  scored["total"]  # 后端在第一题上的平均分

认两种题库来源：
  1. yaml 里的 dataset_path（或 dataset_kwargs.data_files）指到本地 jsonl
  2. 纯 yaml 里 samples / fewshot / fewshot_samples 内联的示例
  3. HF Hub dataset_path + dataset_name + test_split（可选依赖 datasets）
num_fewshot 控制真实示范前缀，fewshot_split/training_split 或 fewshot_config.samples 提供示范池。

doc_to_text / doc_to_target 是 jinja 片段，这里只认 {{字段}} 和 {{字段|过滤器}}（过滤器可串，
只有 lower / upper / strip），不引 jinja 依赖。别的写法不猜，原样留着并写进报告。

metric_list 只认三种：exact_match → exact、contains → contains、regex → regex。
认不出的指标名记进报告，题目照导；一条都认不出时按 exact 兜底，也在报告里写明。

为什么兜底而不丢题：丢题是静默的损失，兜底至少题面还在，人看报告就能自己改规则。
"""

import json
import re
import random  # few-shot 抽样独立种子，导入结果可重现
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from features import scorer, suite
from utils import command, yaml_io
from utils.fail import fail

METRICS = {                                                                   # lm-eval 指标 → 我们的规则
    "exact_match": "exact",                                                   # 全等
    "contains": "contains",                                                   # 关键词
    "regex": "regex",                                                         # 正则
}
FILTERS = {"lower": str.lower, "upper": str.upper, "strip": str.strip}        # 认的过滤器只有这三个
PLACE = re.compile(r"\{\{\s*([^{}]*?)\s*\}\}")                                   # 整个 {{...}} 段
SAMPLE_KEYS = ("samples", "fewshot", "fewshot_samples")                       # 内联示例可能用的键名
SPLIT = ";"                                                                   # contains 的多个答案按分号分
TIMEOUT = 300                                                                 # 单题最多等五分钟，坏后端不能把整批题拖死


@dataclass
class Import:
    """一份 yaml 导完的结果，用来回话和写报告。"""

    source: Path    # 从哪个文件来的
    name: str       # 任务名，已洗成能进文件名的样子
    path: Path      # 写出的套件文件
    report: Path    # 报告文件
    tasks: int      # 导了几题
    dropped: list = field(default_factory=list)  # 认不出的指标名
    notes: list = field(default_factory=list)    # 其它说明：兜底、跳过的记录、认不出的写法

    def say(self) -> str:
        """一行回话：导了哪份、多少题、丢弃多少规则、报告在哪。"""
        line = f"{self.source.name} → {self.path.name}：{self.tasks} 题，丢弃规则 {len(self.dropped)}"
        if self.dropped:                                                      # 丢弃的规则名摆在明处
            line += "（" + " / ".join(self.dropped) + "）"
        return line + f"；报告 {self.report.name}"


# --- 跑分 ---

def run_backend(backend_cmd: str, suite, limit: int) -> dict:
    """拿一个答题命令跑一整套题，返回总分和逐题判定。

    backend_cmd 里的 ``{prompt}`` 会换成题面；空命令按回显题面处理。
    limit 为 0 时跑完整套件，正整数只跑开头几题。
    """
    if limit < 0:                                                            # 这是公开边界，负数没有可解释的跑法
        fail("input", "limit 不能是负数", hint="给 0 或正整数，0 表示不限")
    tasks = suite.tasks[:limit] if limit else suite.tasks                    # 先定下本次实际跑的题
    rows = []                                                                # 一题一行，交给 judge 拼完整报告
    for task in tasks:                                                       # 同一后端依次答完选中的题
        text = answer(backend_cmd, task.prompt)                              # 把题面交给外部答题命令
        verdict = scorer.grade(task, text)                                  # 用套件自带规则判这份答案
        rows.append({                                                        # 留下逐题分数与可追溯答案
            "suite": suite.name,                                           # 题来自哪一套
            "id": task.id,                                                 # 题号供基线逐题对齐
            "score": verdict.score,                                        # 这题的 0~1 分
            "reason": verdict.reason,                                      # 为什么得这分
            "answer": text.strip(),                                        # 后端实际答了什么
        })
    total = round(sum(row["score"] for row in rows) / len(rows), 4) if rows else 0.0  # 平均分就是套件分
    return {                                                                 # 这是一份可直接检查的分数报告
        "backend": backend_cmd or "echo",                                 # 空命令的真实行为是回显
        "suite": suite.name,                                               # 跑的是哪一套
        "count": len(rows),                                                # 实际答了几题
        "total": total,                                                    # 这一套的平均分
        "tasks": rows,                                                     # 每题的答案与判定
    }


def answer(backend_cmd: str, prompt: str) -> str:
    """先解析可信命令模板，再把题面填入独立 argv；不执行 shell 管道或重定向。

    占位符作为参数使用，例如 python backend.py '{prompt}'；固定 python -c 程序也可读取
    sys.argv[1]。不要把题面嵌入程序源码，后端脚本也不应再次 eval 输入。
    命令失败、找不到或超时沿用 input 错误出口。
    """
    if not backend_cmd:                                                      # 没接模型时保留原来的回显占位行为
        return prompt                                                        # 题面本身就是占位答案
    try:
        arguments = command.argv(backend_cmd, prompt=prompt)                     # 题面的空格、换行与引号始终属于同一参数
        done = subprocess.run(arguments, shell=False, capture_output=True, text=True, timeout=TIMEOUT)
    except (ValueError, OSError) as error:                                    # 模板或可执行文件错误属于可修正的后端配置
        fail("input", f"答案命令不能启动: {error}", where=backend_cmd[:100])
    except subprocess.TimeoutExpired:                                       # 卡死的后端不能拖住整批评测
        fail("input", f"答案命令 {TIMEOUT} 秒没跑完", where=backend_cmd[:100])
    if done.returncode:                                                      # 程序自己的失败不伪装成空答案
        fail("input", f"答案命令退出码 {done.returncode}", where=backend_cmd[:100], hint=done.stderr.strip()[:300])
    return done.stdout                                                       # 原样交给判分器，清理规则由题型决定


def import_all(paths: list, out_dir: Path) -> list:
    """一份一份导，全导完再把结果交回去。缺输出目录就先建好。"""
    out_dir.mkdir(parents=True, exist_ok=True)                                # 目录先备好
    return [import_file(Path(one), out_dir) for one in paths]                 # 逐个文件导


def import_file(source: Path, out_dir: Path) -> Import:
    """把一份 lm-eval 任务定义导成我们的套件文件，返回导了什么。"""
    data = yaml_io.read(source)                                               # 坏 yaml 在这里停
    name = clean_name(str(data.get("task") or data.get("task_name") or source.stem))
    notes = []                                                                # 给人看的说明，最后进报告
    rule, dropped = pick_rule(data, notes)                                    # 判分规则只取一条
    records = load_records(data, source)                                      # 内联示例或本地 jsonl
    examples = fewshot_records(data, source, records)                         # 示范题与评分题分别装载
    tasks = build(data, records, rule, notes, examples)                                 # 一条一条渲染成题目
    if not tasks:                                                             # 一条都没成
        fail("input", f"{source.name} 没导出一题", hint="核对题库字段和 doc_to_text 里写的字段名")

    where = out_dir / f"lm-eval-{name}.yaml"                                  # 落到套件目录，名字带前缀
    if where.exists():                                                        # 同名文件会被盖掉
        note(notes, f"{where.name} 已经在了，已覆盖")
    desc = desc_of(data, len(tasks))                                          # 套件说明
    one = suite.Suite(name=f"lm-eval-{name}", desc=desc, tasks=tasks, path=where)
    suite.write(where, one, head=f"# 由 lm-eval 任务 {source.name} 导入，共 {len(tasks)} 题。\n")
    report = out_dir / f"lm-eval-{name}.report.md"                            # 报告放在套件旁边
    report.write_text(render_report(source, where, len(tasks), dropped, notes), encoding="utf-8")
    return Import(source=source, name=name, path=where, report=report, tasks=len(tasks),
                  dropped=dropped, notes=notes)


# --- 名字与说明 ---

def clean_name(text: str) -> str:
    """任务名洗成能进文件名的样子：留字母数字下划线点和横线。"""
    return re.sub(r"[^\w.-]+", "-", text).strip("-.") or "task"               # 洗空了就退回 task


def desc_of(data: dict, count: int) -> str:
    """套件说明：有描述就用描述，接一句来源和题数。"""
    said = str(data.get("description") or data.get("task") or "").strip()     # 原定义里的描述
    return f"{said}（lm-eval 导入，{count} 题）" if said else f"lm-eval 导入，{count} 题"


# --- 判分规则 ---

def pick_rule(data: dict, notes: list) -> tuple:
    """挑一条判分规则。认不出的指标名原样记下来，一条都不认就按 exact 兜底。"""
    listed = data.get("metric_list") or []                                    # 指标清单
    if not isinstance(listed, list):                                          # 写成映射的那种
        note(notes, "metric_list 不是清单，按 exact 兜底")
        return "exact", []
    names = [metric_name(one) for one in listed]                              # 指标名，空的也留着
    known = [METRICS[one] for one in names if one in METRICS]                 # 认得的那几条
    dropped = [one or "(没写 metric)" for one in names if one not in METRICS]  # 认不出的进报告
    if not names:                                                             # 压根没写 metric_list
        note(notes, "没给 metric_list，按 exact 兜底")
        return "exact", dropped
    if not known:                                                             # 写了但一条都不认
        note(notes, "指标一条都认不出，按 exact 兜底")
        return "exact", dropped
    if len(known) > 1:                                                        # 一套题只跟一条规则
        note(notes, f"认得 {len(known)} 条规则，只取第一条 {known[0]}")
    return known[0], dropped


def metric_name(one) -> str:
    """取一条指标的指标名。正常写成 {metric: xx}，直接写名字的也认。"""
    if isinstance(one, dict):                                                 # 正常形状
        return str(one.get("metric") or "")
    return str(one or "")                                                     # 直接写名字那种


# --- 题库 ---

def load_records(data: dict, source: Path, split: str = "") -> list:
    """装载评分或示范 split：先本地内联/JSONL，再使用 datasets 访问 Hugging Face Hub。"""
    selected = split or data.get("test_split") or data.get("validation_split") or data.get("training_split") or "test"
    if not split and isinstance(data.get("samples"), list) and data["samples"]:
        return [dict(one) for one in data["samples"]]  # 显式评分样本的优先级保持兼容
    where = local_file(data, source, selected)  # 只读所选 split，不误把训练集当测试集
    if where:
        return read_jsonl(where)
    dataset = data.get("dataset_path")  # 有 dataset_path 时 few-shot 示例不能覆盖它
    if dataset:
        if str(dataset).startswith((".", "/", "~")) or str(dataset).endswith(".jsonl"):
            fail("input", f"本地题库不存在: {dataset}", where=str(source))
        try:
            from datasets import load_dataset  # 本地任务不要求安装网络数据依赖
        except ImportError:
            fail("missing", "HF Hub 题库需要 datasets", hint="pip install datasets")
        options = dict(data.get("dataset_kwargs") or {})  # 原字典不改，示范 split 还会再用
        if options.pop("trust_remote_code", False):  # 任务导入只取数据，不执行仓库 Python
            fail("input", "不支持执行 dataset 的远端代码，请先导出 JSONL")
        options.pop("split", None)  # split 只由任务配置中的用途决定
        if options.get("data_files"):
            options["data_files"] = data_paths(options["data_files"], source.parent)  # datasets 也从任务目录解析文件
        try:
            loaded = load_dataset(str(dataset), name=data.get("dataset_name"), split=selected, **options)
            records = [dict(one) for one in loaded]  # 将 Hub 记录转成模板可消费的映射
        except Exception as wrong:  # 远端数据边界保留网络、鉴权、split 的真实失败原因
            fail("input", f"HF 数据集 {dataset} / {selected} 读取失败: {wrong}", where=str(source))
        if not records:
            fail("input", f"HF 数据集 {dataset} / {selected} 为空")
        return records
    few = data.get("fewshot_config") or {}  # 兼容旧的仅内联示例任务
    for holder in (data, few):
        for key in SAMPLE_KEYS:
            if isinstance(holder.get(key), list) and holder[key]:
                return [dict(one) for one in holder[key]]
    fail("input", f"{source.name} 没有题库", hint="设置 samples、本地 JSONL 或 HF dataset_path")


def local_file(data: dict, source: Path, split: str = "test") -> Path:
    """选定 split 的本地 JSONL；多 split 时不悄悄取第一个文件。"""
    given = data.get("dataset_path")  # 直接给 JSONL 不需要 datasets 包
    if given:
        path = Path(str(given)).expanduser()
        path = path if path.is_absolute() else source.parent / path
        if path.is_file():
            return path
    files = (data.get("dataset_kwargs") or {}).get("data_files")
    if isinstance(files, dict):
        split_name = split.split("[", 1)[0]  # train[:3] 仍属于 train，但切片交给 datasets
        if split_name not in files:
            fail("input", f"data_files 没有 split {split_name}，可用: {', '.join(files)}", where=str(source))
        files = files[split_name] if split == split_name else None
    files = files if isinstance(files, list) else [files] if files else []
    if len(files) > 1:  # 多文件交给 datasets 合并，不能静默少读
        return None
    if files:
        path = Path(str(files[0])).expanduser()
        path = path if path.is_absolute() else source.parent / path
        return path if path.is_file() else None
    return None


def data_paths(files, directory: Path):
    """把 datasets 多文件/多 split 配置中的本地文件定位到任务目录，URL 保持原样。"""
    if isinstance(files, dict):  # 每个 split 独立保留，不能合并为评分题库
        return {name: data_paths(value, directory) for name, value in files.items()}
    if isinstance(files, list):  # 一个 split 可以来自多个文件
        return [data_paths(value, directory) for value in files]
    if "://" in str(files):  # HTTP、HF、S3 等地址交给 datasets
        return files
    return str((directory / str(files)).resolve())  # 相对路径只在边界归一化一次


def fewshot_records(data: dict, source: Path, records: list) -> list:
    """按 num_fewshot 装载示范池；优先显式示范，再用 training/fewshot split。"""
    count = data.get("num_fewshot", 0)  # 没要求就保留原来的 zero-shot
    if type(count) is not int or count < 0:
        fail("input", "num_fewshot 必须是非负整数", where=str(source))
    if type(data.get("fewshot_seed", 0)) is not int:
        fail("input", "fewshot_seed 必须是整数", where=str(source))
    if not count:
        return []
    for holder, keys in ((data.get("fewshot_config") or {}, SAMPLE_KEYS), (data, ("fewshot_samples", "fewshot"))):
        for key in keys:
            if isinstance(holder.get(key), list) and holder[key]:
                return [dict(one) for one in holder[key]]
    split = data.get("fewshot_split") or data.get("training_split")
    if split:
        return load_records(data, source, split)  # 别拿 test split 自动冒充训练示范
    return records  # 同池抽样在 build 中逐题排除自身与重复题面


def read_jsonl(path: Path) -> list:
    """一行一条 JSON 的题库。哪一行坏了当场指出行号。"""
    records = []                                                              # 攒下来的记录
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():                                                  # 空行不算记录
            continue
        try:
            one = json.loads(line)                                            # 一行一条
        except json.JSONDecodeError as wrong:                                 # 坏行要指到行号
            fail("input", f"第 {number} 行不是 JSON: {wrong.msg}", where=str(path))
        if not isinstance(one, dict):                                         # 每条要是 键: 值
            fail("input", f"第 {number} 行不是 键: 值 那种", where=str(path))
        records.append(one)                                                   # 收下这条
    if not records:                                                           # 空文件不成题库
        fail("input", f"{path.name} 一行记录都没有", where=str(path))
    return records


# --- 模板渲染 ---

def render(template: str, record: dict, notes: list) -> str:
    """只替换 {{字段}} 和 {{字段|过滤器}}（过滤器可串），别的写法原样留着。"""
    def swap(match) -> str:
        parts = [one.strip() for one in match.group(1).split("|")]            # 字段和过滤器链
        value = field_of(record, parts[0], notes)                             # 先取字段，缺了记一笔
        for name in parts[1:]:                                                # 过滤器从左往右套
            if name not in FILTERS:                                           # 过滤器不认
                note(notes, f"过滤器 {name} 认不出，原样留着")
                continue
            value = FILTERS[name](value)                                      # 认得就套上去
        return value
    text = PLACE.sub(swap, template)                                          # 逐个占位换掉
    if "{%" in template:                                                      # 语句块没法跑
        note(notes, "模板里有 {% ... %} 语句块，认不出，原样留着")
    return text


def field_of(record: dict, path: str, notes: list) -> str:
    """按点号取字段，取不到当空串并记一笔。列表值拼成一行。"""
    here = record                                                             # 从整条记录开始往里走
    for part in path.split("."):                                              # 一层一层取
        here = here.get(part) if isinstance(here, dict) else None             # 走岔了就当没有
    if here is None:                                                          # 模板写了数据里没有的字段
        note(notes, f"字段 {path} 数据里没有，渲染成空")
        return ""
    if isinstance(here, list):                                                # 列表拼成一行
        return ", ".join(str(one) for one in here)
    return str(here)


# --- 题目 ---

def build(data: dict, records: list, rule: str, notes: list, examples: list = None) -> list:
    """把每条记录渲染成一道题。渲染空了就跳过，跳几条记进报告。"""
    text = str(data.get("doc_to_text") or "{{input}}")                        # 题面模板，缺了退回 input 字段
    target = str(data.get("doc_to_target") or "{{target}}")                   # 答案模板，缺了退回 target 字段
    if not data.get("doc_to_text") or not data.get("doc_to_target"):          # 缺模板要说明
        note(notes, "缺 doc_to_text 或 doc_to_target，按 input / target 字段拼")
    choices = str(data.get("doc_to_choice") or "")                            # 选择题的选项模板
    if choices:                                                               # 选项拼进题面，不丢
        note(notes, "doc_to_choice 认不出，把选项按 A/B/C 接到题面后面")

    count = data.get("num_fewshot", 0)                                         # 是否在题目前拼示范
    chance = random.Random(data.get("fewshot_seed", 0))                        # 不碰全局随机状态
    separator = str(data.get("fewshot_delimiter", "\n\n"))                 # lm-eval 默认示范间空一行
    target_separator = str(data.get("target_delimiter", " "))                # 题面与答案之间的分隔符
    examples = examples or []                                                  # zero-shot 的示范池为空
    tasks = []                                                                # 攒下来的题目
    seen = set()                                                              # 已经用过的题号
    empty = 0                                                                 # 渲染空了的记录数
    for index, record in enumerate(records, 1):                               # 一条一条来
        prompt = render(text, record, notes).strip()                          # 题面
        answer = render(target, record, notes).strip()                        # 答案
        if not prompt or not answer:                                          # 空了就没法出题
            empty += 1
            continue
        if choices:                                                           # 选项接在题面后面
            prompt = add_choices(prompt, render(choices, record, notes))
        if count:                                                             # 示范答案只进前缀，不改评分答案
            candidates = [one for one in examples if render(text, one, notes).strip() != render(text, record, notes).strip()]
            if len(candidates) < count:                                        # 不足时明确失败，避免实际 few-shot 数悄悄变化
                fail("input", f"题目 {index} 只有 {len(candidates)} 条无泄漏示范，少于 num_fewshot={count}")
            chosen = chance.sample(candidates, count)                         # 同一 seed、同一输入得到相同前缀
            prefix = []                                                       # 每条示范先题面、后正确答案
            for example in chosen:
                question = render(text, example, notes).strip()
                if choices:
                    question = add_choices(question, render(choices, example, notes))
                prefix.append(question + target_separator + render(target, example, notes).strip())
            prompt = separator.join([*prefix, prompt])                        # 真正发送给后端的提示词包含示范
        tasks.append(one_task(ident_of(record, index, seen), prompt, answer, rule))
    if count:
        note(notes, f"已拼接 {count}-shot 示范，seed={data.get('fewshot_seed', 0)}，排除与当前题面相同的记录")
    if empty:                                                                 # 跳过的条数要说
        note(notes, f"{empty} 条记录渲染后缺题面或答案，跳过")
    return tasks


def one_task(ident: str, prompt: str, answer: str, rule: str) -> suite.Task:
    """按规则把答案放进它该在的字段。"""
    if rule == "contains":                                                    # 多个答案按分号分
        words = [one.strip() for one in answer.split(SPLIT) if one.strip()]
        return suite.Task(id=ident, prompt=prompt, rule="contains", keywords=words)
    if rule == "regex":                                                       # 答案就是正则原文
        return suite.Task(id=ident, prompt=prompt, rule="regex", pattern=answer)
    return suite.Task(id=ident, prompt=prompt, rule="exact", answer=answer)


def add_choices(prompt: str, rendered: str) -> str:
    """把渲染出来的选项接成 A. / B. / C. 几行。"""
    raw = [one.strip(" '\"") for one in rendered.strip("[]").split(",")]       # 拆开，去掉 jinja 带的引号
    options = [one for one in raw if one]                                     # 空的不要
    lines = [prompt, "选项："]                                                # 题面后面起一行
    lines += [f"{letter}. {option}" for letter, option in zip("ABCDEFGHIJ", options)]
    return "\n".join(lines)


def ident_of(record: dict, index: int, seen: set) -> str:
    """题号：用记录自带的 id / idx，重了往后编，保证套件内唯一。"""
    base = clean_name(str(record.get("id") or record.get("idx") or f"q{index}"))
    ident = base                                                              # 先试原样的
    count = 2                                                                 # 撞了就加序号
    while ident in seen:
        ident = f"{base}-{count}"
        count += 1
    seen.add(ident)                                                           # 记下用过
    return ident


# --- 报告 ---

def note(notes: list, text: str) -> None:
    """记一笔说明，同一句只留一次。"""
    if text not in notes:                                                     # 重复的别再记
        notes.append(text)


def render_report(source: Path, where: Path, count: int, dropped: list, notes: list) -> str:
    """拼一份导入报告：导了哪份、多少题、哪些规则没认出来。"""
    lines = [f"# lm-eval 导入报告：{source.name}", "",
             f"- 套件: {where.name}", f"- 题数: {count}", f"- 丢弃规则: {len(dropped)}"]
    if dropped:                                                               # 认不出的规则单独一块
        lines += ["", "## 认不出的指标（没静默丢，都在这，自己改成 exact / contains / regex）"]
        lines += [f"- {one}" for one in dropped]
    if notes:                                                                 # 其它说明单独一块
        lines += ["", "## 说明"]
        lines += [f"- {one}" for one in notes]
    return "\n".join(lines) + "\n"
