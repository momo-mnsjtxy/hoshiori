"""评测套件：一个 yaml/json 文件就是一整套题，往目录里放一个就多一个套件。

调用示例：
  from features import suite
  where = suite.folder()                    # 内置套件目录
  names = suite.scan(where)                 # 目录里现在有哪些套件文件
  one = suite.read(where / "zh-lite.yaml")  # 读一整套，坏文件当场报错
  one.tasks[0].prompt                       # 第一题的题面

套件长这样（tasks 里每项一题，rule 决定这题怎么判）：
  name: zh-lite
  desc: 中文常识与格式跟随
  tasks:
    - {id: units, prompt: 一打鸡蛋是几个？, rule: contains, keywords: ["12"], why: 常识}

rule 只有三种，各自要的字段不同：
  exact    answer    答案去掉空白后要一模一样
  contains keywords  答案里要出现全部关键词
  regex    pattern   答案要匹配这条正则

为什么按目录扫描：加一套题只要往目录里放一个文件，删掉就少一套，不用改任何代码。
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

from utils import yaml_io
from utils.fail import fail

SUFFIXES = {".yaml", ".yml", ".json"}                                    # 认作套件的扩展名；json 也是合法 yaml，读法同一套
RULES = {"exact": "answer", "contains": "keywords", "regex": "pattern"}  # 规则 → 它必须配的字段


@dataclass
class Task:
    """一道题。"""

    id: str                                       # 题号，基线按它对比，套件内唯一
    prompt: str                                   # 题面，原样交给答题后端
    rule: str                                     # 判分规则，只有 exact / contains / regex
    answer: str = ""                              # exact 用的标准答案
    keywords: list = field(default_factory=list)  # contains 用的关键词
    pattern: str = ""                             # regex 用的正则
    why: str = ""                                 # 这题考什么，给人看，不参与判分


@dataclass
class Suite:
    """一整套题，来自一个文件。"""

    name: str    # 套件名，报告和基线都用它
    desc: str    # 一句话说明这套题考什么
    tasks: list  # 全部题目
    path: Path   # 来自哪个文件，报错和追溯用


def folder() -> Path:
    """内置套件目录。按本文件的位置找，换一个目录启动也找得到。"""
    return Path(__file__).parent / "suites"  # 内置套件就在这个包旁边


def scan(where: Path) -> list:
    """列出目录里的套件文件，按文件名排序。目录不在就报。"""
    if not where.is_dir():                                                     # 目录没了就无从扫
        fail("missing", f"套件目录不在: {where}", hint="建好这个目录，或换一个目录")
    return sorted(one for one in where.iterdir() if one.suffix in SUFFIXES)    # 只要认得作套件的那几类


def locate(where: Path, name: str) -> Path:
    """按名字找套件文件。名字可带扩展名，也可以直接给一个存在的路径。"""
    given = Path(name)                                 # 命令行写的东西
    if given.suffix in SUFFIXES and given.exists():    # 直接给了文件路径
        return given                                   # 直接用这个文件
    for one in scan(where):                            # 否则按文件名逐个对，带不带扩展名都行
        if one.stem == given.stem:                     # 名字对上，扩展名写没写都算
            return one                                 # 就是它
    fail("input", f"找不到套件 {name}", where=str(where), hint="--suite 用文件名去掉扩展名，例: --suite zh-lite")


def read(path: Path) -> Suite:
    """读一整套题并当场验它。坏文件在这里停，后面的评测不用再防。"""
    data = yaml_io.read(path)                                              # 坏 yaml、顶层不是映射都在这里停
    check(data, path)                                                      # 字段、必填、题号唯一先过一遍
    tasks = [                                                              # 逐题搬成题目对象
        Task(
            id=str(one["id"]).strip(),                                     # 题号
            prompt=one["prompt"].strip(),                                  # 题面
            rule=one["rule"],                                              # 判分规则
            answer=(one.get("answer") or "").strip(),                      # exact 的标准答案
            keywords=[str(word) for word in one.get("keywords") or []],    # contains 的关键词
            pattern=one.get("pattern") or "",                              # regex 的正则原文
            why=(one.get("why") or "").strip(),                            # 出题意图，给人看
        )
        for one in data["tasks"]                                           # 一题一题搬
    ]
    return Suite(                                                          # 装成一套交出去
        name=(data.get("name") or path.stem).strip(),                      # 没写名字就用文件名
        desc=(data.get("desc") or "").strip(),                             # 说明可以留空
        tasks=tasks,                                                       # 全部题目
        path=path,                                                         # 来自哪个文件
    )


def check(data: dict, path: Path) -> None:
    """验一整套题。缺东西、题号重了、正则写坏了都在这里报。"""
    where = str(path)                                                   # 每条报错都带上文件位置
    tasks = data.get("tasks")                                           # 题目清单
    if not isinstance(tasks, list) or not tasks:                        # 没题就不成套装
        fail("input", "套件要有 tasks，而且至少一题", where=where, hint="照着 features/suites/zh-lite.yaml 写")
    seen = set()                                                        # 已经用过的题号
    for index, one in enumerate(tasks, 1):                              # 一题一题看
        if not isinstance(one, dict):                                   # 每题要写成 键: 值
            fail("input", f"第 {index} 题不是 键: 值 那种", where=where)
        ident = str(one.get("id") or "").strip()                        # 题号
        if not ident:                                                   # 没题号，基线就对不上
            fail("input", f"第 {index} 题缺 id", where=where, hint="例: id: units")
        if ident in seen:                                               # 题号重了，两份结果会互相盖
            fail("input", f"题号重了: {ident}", where=where)
        seen.add(ident)                                                 # 记下用过，防重号
        if not str(one.get("prompt") or "").strip():                    # 没题面就问不出答案
            fail("input", f"{ident} 缺 prompt", where=where)
        rule = one.get("rule")                                          # 这题怎么判
        if rule not in RULES:                                           # 只认那三种
            fail("input", f"{ident} 的 rule 不认识: {rule}", where=where, hint="可用: " + " / ".join(RULES))
        need = RULES[rule]                                              # 这条规则必须配的字段
        if rule == "contains":                                          # 关键词要一份非空清单
            if not isinstance(one.get(need), list) or not one[need]:    # 清单不能空
                fail("input", f"{ident} 用 contains 就要给 keywords 清单", where=where)
            continue
        if not str(one.get(need) or "").strip():                        # exact 要答案、regex 要正则，都不能空
            fail("input", f"{ident} 用 {rule} 就要给 {need}", where=where)
        if rule == "regex":                                             # 正则先编译一次，坏正则出现在装载时而不是判分时
            try:
                re.compile(one[need])                                   # 先试着编译
            except re.error as wrong:                                   # 写错了当场指出
                fail("input", f"{ident} 的 pattern 编译不过: {wrong}", where=where)
