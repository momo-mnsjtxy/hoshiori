"""公开榜提交：从评测报告抽出可公开字段 POST；暂时发不出就留下待补文件。

调用示例：
  from pathlib import Path
  from features import board
  result = board.submit(Path("out/judge/report.json"), "https://hoshiori.ai/board")
  result["submitted"]  # True 表示已发出；False 时看 result["pending"]

POST 负载只含 model、score、recipe、hardware、time。没有 URL 或网络失败时，
同一份负载会写到 report.json 旁边的 board-pending.json，并把目标 URL 一并留下。
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

from utils.fail import fail

PENDING = "board-pending.json"                                               # 离线待补文件的固定名字
TIMEOUT = 20                                                                 # 榜单接口不该拖住本地评测太久


def submit(report, url: str) -> dict:
    """提交一份 report.json；无 URL 或网络失败时改写 ``board-pending.json``。

    report 可以是报告路径，也可以是已经读出的报告映射。返回值里的 payload 是实际
    尝试发送的五个公开字段，pending 是待补文件路径或 None。
    """
    data, pending = read_report(report)                                      # 外部报告先在边界读好、验好
    missing = [name for name in ("model", "total", "recipe", "hardware") if name not in data]
    if missing:                                                              # 缺字段就不是 judge 能提交的报告
        fail("input", "公开榜报告缺字段: " + " / ".join(missing),
             hint="要提交 ori judge 生成的 report.json")
    payload = {                                                              # 榜单只收到复现与排序真正需要的内容
        "model": data["model"],                                             # 被评模型名
        "score": data["total"],                                            # judge 的总分在榜上叫 score
        "recipe": data["recipe"],                                          # 配方 SHA-256 哈希
        "hardware": data["hardware"],                                      # CPU、内存、GPU 与系统
        "time": data.get("time") or datetime.now(timezone.utc).isoformat(), # 报告没时间时记本次提交的 UTC 时间
    }
    if not url or not url.strip():                                           # 没目标时先落盘，等用户补 URL
        write_pending(pending, "", payload)                                 # 五个字段原样保住
        return {"submitted": False, "payload": payload, "pending": pending}

    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")           # HTTP 请求体统一用 UTF-8 JSON
    request = Request(url.strip(), data=body,                                # 有 data 就是 POST，再显式写清方法
                      headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(request, timeout=TIMEOUT) as response:                  # 真发出去并读完响应，连接才能及时归还
            response.read()                                                  # 榜单响应正文不参与本地评测结论
    except (OSError, ValueError):                                            # 断网、超时、坏 URL 都先排队，不丢报告
        write_pending(pending, url.strip(), payload)                         # URL 也留下，之后能原样补发
        return {"submitted": False, "payload": payload, "pending": pending}
    return {"submitted": True, "payload": payload, "pending": None}      # 服务已接收，没有待补文件


def read_report(report) -> tuple:
    """读报告并决定待补文件位置；映射输入落在当前目录。"""
    if isinstance(report, dict):                                             # 命令层若已读过可直接交映射
        return report, Path(PENDING)                                         # 没原文件就落在当前工作目录
    path = Path(report)                                                      # 正常用法是 report.json 路径
    if not path.is_file():                                                   # 路径不在时直接指出，不伪装成离线
        fail("input", f"评测报告不在: {path}", hint="先跑 ori judge 生成 report.json")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))                 # 报告是 UTF-8 JSON
    except json.JSONDecodeError as wrong:                                    # 坏 JSON 属于用户可修的输入
        fail("input", f"评测报告不是合法 json: {wrong}", where=str(path))
    if not isinstance(data, dict):                                           # 榜单字段必须从映射里按名提取
        fail("input", f"{path} 不像评测报告", hint="要提交 ori judge 生成的 report.json")
    return data, path.parent / PENDING                                       # 待补文件跟原报告放在一起


def write_pending(path: Path, url: str, payload: dict) -> None:
    """把目标 URL 和原提交字段写成一份可人工补发的 JSON。"""
    pending = {"url": url, **payload}                                      # 去掉 url 后就是原 POST 负载
    path.parent.mkdir(parents=True, exist_ok=True)                           # 报告目录若刚被搬走也能补建
    path.write_text(json.dumps(pending, ensure_ascii=False, indent=2), encoding="utf-8")
