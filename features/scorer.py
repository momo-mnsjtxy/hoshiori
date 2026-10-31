"""判分：拿答案跟题目要求比，给 0~1 分，并说清为什么得这个分。

调用示例：
  from features import scorer
  verdict = scorer.grade(task, "12 个")
  verdict.score   # 1.0
  verdict.reason  # "答案里有全部关键词"
  verdict = scorer.grade(task, "十来个")
  verdict.score  # 0.0

三条规则，跟套件里的 rule 一一对应：
  exact     去掉全部空白后全等才给 1 分
  contains  命中的关键词占多少就给多少分，全中才 1 分
  regex     正则能搜到才给 1 分

为什么 contains 给的是比例：只命中一半的答案明显比一个都没中好，报告里要看得出来。
"""

import re
from dataclasses import dataclass

BLANK = re.compile(r"\s+")  # 全部空白，exact 拿来两边一起清掉


@dataclass
class Verdict:
    """一道题的判定。"""

    score: float  # 0~1，1 是满分
    reason: str   # 给人看的判定理由


def grade(task, answer: str) -> Verdict:
    """按题目自带的 rule 判一份答案。题目已经装载验过，这里直接用。"""
    return HANDLERS[task.rule](task, answer)  # 三条规则各一个处理函数


def exact(task, answer: str) -> Verdict:
    """去掉空白后要一模一样。空格、换行、缩进都不算差别。"""
    if BLANK.sub("", answer) == BLANK.sub("", task.answer):    # 两边都清掉空白再比
        return Verdict(1.0, "去掉空白后跟标准答案一致")                    # 全等给满分
    return Verdict(0.0, f"跟标准答案不一致，要的是「{task.answer}」")        # 差一个字也是 0


def contains(task, answer: str) -> Verdict:
    """答案里出现多少个关键词就给多少分。"""
    hit = [one for one in task.keywords if one in answer]            # 命中的关键词
    missing = [one for one in task.keywords if one not in answer]    # 没命中的
    score = round(len(hit) / len(task.keywords), 4)                  # 命中占比就是分
    if not missing:                                                  # 全中
        return Verdict(score, "答案里有全部关键词")                           # 全中就是 1 分
    return Verdict(score, "缺关键词: " + " / ".join(missing))            # 缺几个扣几个


def regex(task, answer: str) -> Verdict:
    """答案里能搜到题目给的正则就给满分。多行靠 MULTILINE 管。"""
    if re.search(task.pattern, answer, re.MULTILINE | re.DOTALL):    # 搜到就算过
        return Verdict(1.0, f"匹配上了正则 {task.pattern}")                # 搜到就给满分
    return Verdict(0.0, f"没匹配上正则 {task.pattern}")                    # 格式不对


HANDLERS = {"exact": exact, "contains": contains, "regex": regex}    # rule → 处理函数，加规则只加这一行
