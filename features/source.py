"""来源：按一个地址吐出原始记录，供 weave 一条条消费。

调用示例：
  from features import source
  for record, where in source.read("data/a.jsonl"):   # where 是行号，写进 rejects 方便回查
      ...
  source.read("-")        # 从标准输入读
  source.read("https://…")  # 从链接读
  source.read("data/")      # 是一个目录就递归读光里面的文件

认的行格式：一行一个 JSON 对象；认不出 JSON 的整行当正文。整份是 JSON 数组的文件也认。
大小写、编码、坏字节都在这里统一兜住，下游拿到就是干净的对象。
"""

import io
import json
import sys
from pathlib import Path
from urllib.request import urlopen

from utils.fail import fail

PACKED = {".jsonl", ".json", ".ndjson"}  # 一行一条的扩展名


def read(spec: str):
    """按地址产出 (记录, 位置)。所有入口都在这里分开，调用方只有一条路。"""
    if spec == "-":  # 标准输入，接管道用
        yield from _lines(sys.stdin.buffer, "<stdin>")
        return
    if spec.startswith(("http://", "https://")):  # 远端文件，整份拉下来再按行走
        with urlopen(spec, timeout=60) as answer:
            yield from _lines(io.BytesIO(answer.read()), spec)
        return
    path = Path(spec)  # 剩下的都是本地路径
    if not path.exists():
        fail("input", f"找不到 {path}", hint="核对路径，或用 - 走标准输入")
    if path.is_dir():  # 目录：递归读光，前台不传一堆路径
        for child in sorted(path.rglob("*")):
            if child.is_file():
                yield from file(child)
        return
    yield from file(path)


def file(path: Path):
    """读一个文件。压缩过的先解压。"""
    handle = _open(path)  # 拿到字节流，编码在下一层统一处理
    with handle:
        yield from _lines(handle, str(path))


def _open(path: Path):
    """打开文件；.zst 先包一层解压。"""
    if path.suffix == ".zst":
        try:
            import zstandard
        except ImportError:
            fail("missing", "zst 压缩样本要 zstandard", hint="pip install zstandard")
        return zstandard.ZstdDecompressor().stream_reader(path.open("rb"))
    return path.open("rb")


def _lines(stream, where: str):
    """把字节流按行读成记录。解不出的行整行当正文，不丢。"""
    text = io.TextIOWrapper(stream, encoding="utf-8", errors="replace")  # 坏字节换成占位符，别中断整趟
    for number, line in enumerate(text, 1):
        line = line.strip()
        if not line:  # 空行跳过
            continue
        if line.startswith("{"):  # 一行一个对象
            try:
                yield json.loads(line), f"{where}:{number}"
            except json.JSONDecodeError:
                yield {"text": line}, f"{where}:{number}"  # 不是 JSON 就当纯文本行
            continue
        if line.startswith("[") and where.endswith((".json", ".ndjson")):  # 整份是数组的 JSON
            yield from _array(json.loads(line), where, number)
            continue
        yield {"text": line}, f"{where}:{number}"  # 纯文本行


def _array(items, where: str, number: int):
    """展开一个 JSON 数组里的对象。"""
    for index, item in enumerate(items):
        if isinstance(item, dict):  # 只要对象，别的类型无从判断字段
            yield item, f"{where}:{number}#{index}"
