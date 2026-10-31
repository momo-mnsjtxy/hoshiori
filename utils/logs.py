"""进度与事件落盘。终端一行，同时追加进 loom.jsonl 给面板读。

调用示例：
  from utils.logs import begin, step, warn, done
  begin("out/bolt")               # 之后的事件同时写 out/bolt/loom.jsonl
  step("weave", "读到 1000 条", read=1000)
  warn("weave", "跳过一条超长样本")

为什么不是事件总线：总线要订阅方启动时登记，接手者看不出谁会响应。
这里就是几个直接调用，写终端和写文件都在这一个文件里，找人只找一个地方。
"""

import json
import sys
import time

_sink = None  # 打开的文件句柄；begin() 前是 None，此时只写终端


def begin(folder) -> None:
    """把后续事件追加写进 folder/loom.jsonl。"""
    global _sink
    folder.mkdir(parents=True, exist_ok=True)
    _sink = (folder / "loom.jsonl").open("a", encoding="utf-8")


def end() -> None:
    """收尾关文件，可以重复调。"""
    global _sink
    if _sink is not None:
        _sink.close()
        _sink = None


def _write(stage: str, text: str, level: str, extra: dict) -> None:
    """写出处。终端正常内容进 stdout，警告和错误进 stderr。"""
    line = f"[{stage}] {text}" + "".join(f" {k}={v}" for k, v in extra.items())
    print(("! " if level == "warn" else "") + line, file=sys.stderr if level != "info" else sys.stdout, flush=True)
    if _sink is not None:  # 面板侧要能按时间排，带上时间戳
        record = {"t": round(time.time(), 3), "stage": stage, "text": text, "level": level, **extra}
        _sink.write(json.dumps(record, ensure_ascii=False) + "\n")
        _sink.flush()  # 每行都刷，进程被 kill 也留得下最后一步


def step(stage: str, text: str, **extra) -> None:
    """记一步正常进度。"""
    _write(stage, text, "info", extra)


def warn(stage: str, text: str, **extra) -> None:
    """记一条跳过或异常但能继续的情况。"""
    _write(stage, text, "warn", extra)


def done(stage: str, text: str, **extra) -> None:
    """记一步收尾。"""
    _write(stage, text, "info", extra)
