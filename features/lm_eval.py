"""lm-eval 导入器：把 lm-evaluation-harness 的任务定义读成我们自己的套件文件。

调用示例：
  from pathlib import Path
  from features import lm_eval, suite
  one = lm_eval.import_file(Path("tasks/demo.yaml"), suite.folder())
  one.tasks      # 导了几题
  one.dropped    # 认不出的指标名，进报告，不静默丢
  one.report     # 报告文件在哪

认两种题库来源：
  1. yaml 里的 dataset_path（或 dataset_kwargs.data_files）指到本地 jsonl
  2. 纯 yaml 里 samples / fewshot / fewshot_samples 内联的示例

doc_to_text / doc_to_target 是 jinja 片段，这里只认 {{字段}} 和 {{字段|过滤器}}（过滤器可串，
只有 lower / upper / strip），不引 jinja 依赖。别的写法不猜，原样留着并写进报告。

metric_list 只认三种：exact_match → exact、contains → contains、regex → regex。
认不出的指标名记进报告，题目照导；一条都认不出时按 exact 兜底，也在报告里写明。

为什么兜底而不丢题：丢题是静默的损失，兜底至少题面还在，人看报告就能自己改规则。
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from features import suite
from utils import yaml_io
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
    tasks = build(data, records, rule, notes)                                 # 一条一条渲染成题目
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

def load_records(data: dict, source: Path) -> list:
    """拿题库：yaml 里内联的示例优先，否则读本地 jsonl。"""
    few = data.get("fewshot_config") or {}                                    # 有的任务把示例塞在这一层
    others = [few] if isinstance(few, dict) else []                           # 能看的另一层
    for holder in [data, *others]:                                            # 两层都看
        for key in SAMPLE_KEYS:                                               # 内联示例
            if isinstance(holder.get(key), list) and holder[key]:
                return [dict(one) for one in holder[key] if isinstance(one, dict)]  # 只要 键: 值 那种
    where = local_file(data, source)                                          # yaml 里指的本地题库
    if not where:                                                             # 两头都没有
        fail("input", f"{source.name} 既没有内联示例，也没有本地 jsonl",
             hint="用 dataset_path 指到 .jsonl，或写 samples 内联")
    return read_jsonl(where)


def local_file(data: dict, source: Path) -> Path:
    """从 dataset_path / dataset_kwargs.data_files 里找一个真的存在的文件。"""
    given = [data.get("dataset_path")]                                        # 直接指文件那种
    files = (data.get("dataset_kwargs") or {}).get("data_files")              # 按 split 分组的写法
    if isinstance(files, dict):                                               # 分组里的每个都试
        given += list(files.values())
    elif files:                                                               # 一个路径或一列路径
        given += files if isinstance(files, list) else [files]
    for one in given:                                                         # 逐个按 yaml 所在目录找
        if not one:                                                           # 空值跳过
            continue
        path = Path(str(one))                                                 # 相对路径按 yaml 所在目录算
        if not path.is_absolute():
            path = source.parent / path
        if path.is_file():                                                    # 找到就停
            return path
    return None


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

def build(data: dict, records: list, rule: str, notes: list) -> list:
    """把每条记录渲染成一道题。渲染空了就跳过，跳几条记进报告。"""
    text = str(data.get("doc_to_text") or "{{input}}")                        # 题面模板，缺了退回 input 字段
    target = str(data.get("doc_to_target") or "{{target}}")                   # 答案模板，缺了退回 target 字段
    if not data.get("doc_to_text") or not data.get("doc_to_target"):          # 缺模板要说明
        note(notes, "缺 doc_to_text 或 doc_to_target，按 input / target 字段拼")
    choices = str(data.get("doc_to_choice") or "")                            # 选择题的选项模板
    if choices:                                                               # 选项拼进题面，不丢
        note(notes, "doc_to_choice 认不出，把选项按 A/B/C 接到题面后面")

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
        tasks.append(one_task(ident_of(record, index, seen), prompt, answer, rule))
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
