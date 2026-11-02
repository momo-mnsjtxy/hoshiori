"""公开榜：提交评测结果，也把收到的提交排成可直接托管的静态榜。

调用示例：
  from pathlib import Path
  from features import board

  rows = board.collect(Path("board-inbox"))
  board.render(rows)                                  # 一整张静态 HTML
  board.publish(Path("board-inbox"), Path("site"))  # site/index.html + board.json

  result = board.submit(Path("out/judge/report.json"), "https://hoshiori.ai/board")
  result["submitted"]  # True 表示已发出；False 时看 result["pending"]

collect 只让 model、score、recipe、hardware、time 五个公开字段进榜，分数高的在前；
同分时先提交的在前。缺字段或读不成 JSON 的文件不进 board.json，原因列在 HTML 底部。
POST 负载只含 model、score、recipe、hardware、time。没有 URL 或网络失败时，
同一份负载会写到 report.json 旁边的 board-pending.json，并把目标 URL 一并留下。
"""

import html
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

from utils.fail import fail

PENDING = "board-pending.json"                                               # 离线待补文件的固定名字
TIMEOUT = 20                                                                 # 榜单接口不该拖住本地评测太久
FIELDS = ("model", "score", "recipe", "hardware", "time")                 # 榜上允许公开的五项
STYLE = """
:root { color-scheme: dark; --bg:#14161a; --panel:#1c1f26; --line:#2c313b;
        --text:#e6e8ec; --dim:#8d95a3; --accent:#8fb6e0; --warn:#d9847f; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--text); font:14px/1.6 ui-sans-serif,system-ui,sans-serif; }
main { max-width:1100px; margin:0 auto; padding:36px 18px 64px; }
h1 { margin:0; font-size:24px; } h2 { margin:30px 0 8px; font-size:16px; }
.note { color:var(--dim); margin:4px 0 22px; }
table { width:100%; border-collapse:collapse; background:var(--panel); border:1px solid var(--line); }
th, td { padding:10px 12px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top; }
th { color:var(--dim); font-weight:normal; white-space:nowrap; }
td:first-child { color:var(--dim); width:44px; } .score { color:var(--accent); font-variant-numeric:tabular-nums; }
code { color:var(--accent); } .empty { color:var(--dim); text-align:center; }
.skipped { color:var(--warn); } .skipped ul { margin:6px 0 0; padding-left:20px; }
@media (max-width:760px) { table { display:block; overflow-x:auto; } th, td { white-space:nowrap; } }
"""


class BoardRows(list):
    """有效榜单行；skipped 留着未进榜的文件和原因，供页面底部说明。"""

    def __init__(self, rows=(), skipped=()):
        super().__init__(rows)                                                # 对调用方仍是一份可遍历、可 JSON 化的列表
        self.skipped = list(skipped)                                          # 未进榜记录不混进有效行


# --- 收榜与发布 ---

def collect(folder) -> BoardRows:
    """读取 ``folder/board-*.json``，返回按分数和提交时间排好的有效行。"""
    root = Path(folder)                                                       # 收件目录由命令层先检查
    rows = []                                                                # 只有五项齐全、分数可排序的行进这里
    skipped = []                                                             # 坏文件留原因，页面底部会列出来
    for path in sorted(root.glob("board-*.json")):                            # 文件名排序让完全同分同刻也稳定
        try:
            data = json.loads(path.read_text(encoding="utf-8"))              # 一份文件就是一次提交
        except (OSError, UnicodeError, json.JSONDecodeError) as wrong:
            skipped.append({"file": path.name, "reason": f"读不成 JSON：{wrong}"})
            continue                                                         # 坏文件不挡住其余提交出榜
        if not isinstance(data, dict):                                        # 五个公开字段必须按名字取
            skipped.append({"file": path.name, "reason": "内容不是 JSON 对象"})
            continue
        missing = [name for name in FIELDS if name not in data]               # 边界上一次列全缺项
        if missing:
            skipped.append({
                "file": path.name,
                "model": data.get("model", ""),
                "reason": "缺字段：" + " / ".join(missing),
            })                                                               # 文件名外也留模型名，页面上更容易找回原提交
            continue
        try:
            score = float(data["score"])                                    # 排名需要能比较的有限数字
        except (TypeError, ValueError):
            skipped.append({"file": path.name, "reason": "score 不是数字"})
            continue
        if not math.isfinite(score):                                          # NaN 会让排序结果失去意义
            skipped.append({"file": path.name, "reason": "score 不是有限数字"})
            continue
        rows.append({name: data[name] for name in FIELDS})                    # 额外字段不公开，也不写进机器榜
    rows.sort(key=_order)                                                     # 分高在前；同分时提交早的在前
    return BoardRows(rows, skipped)


def render(rows) -> str:
    """把榜单行排成一张自带样式的 HTML；所有外来内容都先做文本转义。"""
    ready, skipped = _ready(rows)                                             # 直接传普通列表时也会挡掉缺字段行
    lines = []                                                               # 榜单表格，一份提交一行
    for rank, row in enumerate(ready, 1):
        lines.append(
            "<tr>"
            f"<td>{rank}</td>"
            f"<td>{_esc(row['model'])}</td>"
            f"<td class='score'>{_esc(row['score'])}</td>"
            f"<td><code>{_esc(_recipe_head(row['recipe']))}</code></td>"
            f"<td>{_esc(_hardware(row['hardware']))}</td>"
            f"<td>{_esc(row['time'])}</td>"
            "</tr>"
        )
    body = "".join(lines) or "<tr><td class='empty' colspan='6'>还没有有效提交</td></tr>"
    omitted = ""
    if skipped:                                                              # 坏行放在榜后，不冒充一个排名
        items = "".join(_skipped_line(one) for one in skipped)              # 每条外来文字仍各自转义
        omitted = f"<section class='skipped'><h2>跳过的提交</h2><ul>{items}</ul></section>"
    return (
        "<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>星织公开榜</title><style>{STYLE}</style></head><body><main>"
        f"<h1>星织公开榜</h1><p class='note'>共 {len(ready)} 份有效提交 · 分数降序，同分先到先排</p>"
        "<table><thead><tr><th>#</th><th>模型</th><th>分数</th><th>配方</th>"
        f"<th>硬件</th><th>提交时间</th></tr></thead><tbody>{body}</tbody></table>"
        f"{omitted}</main></body></html>"
    )


def publish(folder, out) -> dict:
    """收齐提交，在 ``out`` 写 ``index.html`` 与只含有效行的 ``board.json``。"""
    rows = collect(folder)                                                    # HTML 与 JSON 同源，顺序不会各算一遍
    target = Path(out)                                                       # 静态站点目录
    target.mkdir(parents=True, exist_ok=True)                                # 第一次发布时建好目录
    paths = {"html": target / "index.html", "json": target / "board.json"} # 两份固定产物
    paths["html"].write_text(render(rows), encoding="utf-8")                # 给人看的完整静态页
    paths["json"].write_text(
        json.dumps(list(rows), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )                                                                         # 给机器读的有效榜单
    return paths                                                             # 命令层用路径回话


def _ready(rows) -> tuple:
    """整理 render 收到的行；collect 的跳过记录也一并带过去。"""
    ready = []                                                               # 能进表格的行
    skipped = list(getattr(rows, "skipped", []))                             # collect 已经记下的坏文件
    for number, row in enumerate(rows, 1):                                   # 普通列表也可能直接交给 render
        if not isinstance(row, dict):
            skipped.append({"file": f"第 {number} 行", "reason": "内容不是 JSON 对象"})
            continue
        missing = [name for name in FIELDS if name not in row]               # HTML 不渲染残缺的一行
        if missing:
            skipped.append({
                "file": f"第 {number} 行",
                "model": row.get("model", ""),
                "reason": "缺字段：" + " / ".join(missing),
            })                                                               # 普通列表没有文件名，就用行号和模型名辨认
            continue
        try:
            score = float(row["score"])                                     # render 自己也保证排序键可用
        except (TypeError, ValueError):
            skipped.append({"file": f"第 {number} 行", "reason": "score 不是数字"})
            continue
        if not math.isfinite(score):
            skipped.append({"file": f"第 {number} 行", "reason": "score 不是有限数字"})
            continue
        ready.append({name: row[name] for name in FIELDS})                   # 只把公开字段交给模板
    ready.sort(key=_order)                                                    # 单独调用 render 也遵守同一套榜序
    return ready, skipped


def _order(row: dict) -> tuple:
    """榜序：分数倒排，提交时间正排。完全相同由稳定排序保住原顺序。"""
    return -float(row["score"]), _time_order(row["time"])


def _time_order(value) -> tuple:
    """把 ISO 时间换成可比较时刻；旧数据若不是 ISO，仍按原文字稳定排列。"""
    if isinstance(value, (int, float)) and not isinstance(value, bool):       # 也接受已有的 Unix 时间戳
        return 0, float(value)
    text = str(value)                                                        # JSON 时间通常是 ISO 字符串
    iso = text[:-1] + "+00:00" if text.endswith("Z") else text             # Python 3.10 不认结尾 Z
    try:
        moment = datetime.fromisoformat(iso)                                 # 带偏移时按真实时刻比较
    except ValueError:
        return 1, text                                                       # 旧的非 ISO 时间放在正常时间之后
    if moment.tzinfo is None:                                                # 没写时区的旧提交按 UTC 理解
        moment = moment.replace(tzinfo=timezone.utc)
    return 0, moment.timestamp()


def _recipe_head(value) -> str:
    """只展示配方摘要的前八位；``sha256:`` 算算法名，不算摘要。"""
    text = str(value)                                                        # 页面显示统一走文字
    digest = text.split(":", 1)[1] if ":" in text else text                # sha256: 后面才是哈希本体
    return digest[:8]                                                        # 八位够人眼辨认，完整值仍在 board.json


def _hardware(value) -> str:
    """硬件映射压成一格紧凑 JSON；字符串型旧数据原样显示。"""
    if isinstance(value, str):                                               # 旧提交可能已经自己排好硬件说明
        return value
    return json.dumps(value, ensure_ascii=False, separators=(", ", ": "))   # 嵌套 GPU 清单也能完整留在一行


def _skipped_line(one: dict) -> str:
    """把一条未进榜记录排成页尾列表项。"""
    model = f"（{_esc(one['model'])}）" if one.get("model") else ""           # 有模型名就帮人辨认是哪次提交
    return f"<li><code>{_esc(one['file'])}</code>{model} · {_esc(one['reason'])}</li>"


def _esc(value) -> str:
    """把外来值转成 HTML 文本，尖括号、引号和 & 都不能变成标记。"""
    return html.escape("" if value is None else str(value), quote=True)       # 与 loom 一样，None 显示为空白


# --- 提交 ---


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
