"""计算答案与参考答案的字符重合率，重复字符按出现次数计算。

调用方式：run(answer="模型答案", reference="参考答案")。
调用示例：run(answer="星织", reference="星绒") -> 0.5。
"""

from collections import Counter                 # 统计每个字符出现了几次

NAME = "char_hit"                              # 命令行使用的插件名
SUMMARY = "答案与参考答案的字符重合率"           # 插件列表展示的用途


def run(answer: str, reference: str) -> float:       # 指标插件统一公开 run
    """返回 0～1 的字符重合率，空答案与空参考记 0。"""
    hit = sum((Counter(answer) & Counter(reference)).values())  # 两边都有的字符数
    total = max(len(answer), len(reference), 1)                 # 较长一边作分母，空对空记 0
    return round(hit / total, 4)                               # 四位小数够报告展示
