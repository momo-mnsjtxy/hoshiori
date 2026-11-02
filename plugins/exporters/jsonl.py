"""把一份报告写成单行 UTF-8 JSONL 文件。

调用方式：run(report={"total": 1.0}, path="out/report.jsonl")。
调用示例：run(report={"count": 2}, path="report.jsonl") -> "report.jsonl"。
"""

import json                                  # 报告转成标准 JSON
from pathlib import Path                     # 按用户给的路径写文件

NAME = "jsonl"                              # 命令行使用的插件名
SUMMARY = "把一份报告写成 JSONL"             # 插件列表展示的用途


def run(report: dict, path: str) -> str:             # 导出插件统一公开 run
    """写一行报告 JSON，返回写成的路径。"""
    target = Path(path)                       # 输出位置只认这一处
    text = json.dumps(report, ensure_ascii=False) + "\n"  # JSONL 一条占一行
    target.write_text(text, encoding="utf-8")              # 用 UTF-8 写完整份文件
    return str(target)                                      # 命令行把路径打印出来
