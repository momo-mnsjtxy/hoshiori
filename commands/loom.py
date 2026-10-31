"""命令 loom：打开面板看跑次、指标和样本。

调用示例：
  ori loom

面板读的是产物目录里的 loom.jsonl，所以先跑过 weave 才有东西看。
"""


def command(args) -> int:
    """报出面板会读哪里。"""
    print("面板读产物目录里的 loom.jsonl，先跑一趟 weave 再来看")
    print("这一步还没接面板服务")
    return 0
