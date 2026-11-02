"""答案超过免费长度后，按每个字符的费率给负扣分。

调用方式：run(answer="模型答案", limit=100, rate=0.01)。
调用示例：run(answer="abcdef", limit=3, rate=0.5) -> -1.5。
"""

NAME = "length_penalty"                         # 命令行使用的插件名
SUMMARY = "答案超过长度上限后按字符扣分"          # 插件列表展示的用途


def run(answer: str, limit: int = 100, rate: float = 0.01) -> float:  # 奖励插件统一公开 run
    """返回超过 limit 部分的负扣分。"""
    extra = max(0, len(answer) - limit)           # 只计算超出免费长度的字符
    return -round(extra * rate, 4)                # 负数表示从总奖励里扣掉
