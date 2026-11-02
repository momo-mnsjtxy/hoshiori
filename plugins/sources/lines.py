"""逐行读取 UTF-8 文本，保留空行并返回字符串列表。

调用方式：run(path="data/raw.txt")。
调用示例：run(path="notes.txt") -> ["第一行", "第二行"]。
"""

from pathlib import Path                    # 按用户给的路径读取文本

NAME = "lines"                             # 命令行使用的插件名
SUMMARY = "按行读取一个 UTF-8 文本文件"      # 插件列表展示的用途


def run(path: str) -> list:                      # 来源插件统一公开 run
    """读取一个文本文件，返回其中每一行。"""
    text = Path(path).read_text(encoding="utf-8")  # 一次读完整份文本
    return text.splitlines()                        # 按换行拆开，空行也保留
