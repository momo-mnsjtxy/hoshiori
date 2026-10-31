"""做一个带出口类型的错误，全项目都用它报错。

调用示例：
  from utils.fail import fail
  fail("input", "找不到 data.jsonl", hint="核对路径")
  # 抛出一个 Failure，main.py 按 kind 决定退出码与说法

kind 决定退出码，也决定这批话是给谁看的：
  input   2  用户填的东西有问题，改一下就能过
  schema  3  recipe 内容不合规范，指到具体字段
  missing 4  机器上缺东西，要装
  error   1  程序自己的故障，需要看堆栈

为什么不用继承分几个异常类：接手者要来回跳文件才拼得出有哪些错误类型。
一个 Failure 加 kind 字段，读一遍这个文件就全知道了。
"""


class Failure(Exception):
    """一次失败。kind 说类型，where 指位置，hint 说下一步怎么做。"""

    def __init__(self, kind: str, message: str, hint: str = "", where: str = "") -> None:
        super().__init__(message)
        self.kind = kind
        self.message = message
        self.hint = hint
        self.where = where

    @property
    def exit_code(self) -> int:
        return {"input": 2, "schema": 3, "missing": 4}.get(self.kind, 1)

    def say(self) -> str:
        """拼成给人看的三行：错在哪、什么位置、怎么办。"""
        lines = [f"{self.kind}: {self.message}"]
        if self.where:
            lines.append(f"  位置: {self.where}")
        if self.hint:
            lines.append(f"  提示: {self.hint}")
        return "\n".join(lines)


def fail(kind: str, message: str, hint: str = "", where: str = "") -> None:
    """抛一个 Failure。调用行就是它的位置来源，不用自己填。"""
    raise Failure(kind, message, hint, where)
