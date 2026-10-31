"""面板：把产物目录摊成五个只读页面，起在本机给人看。

调用示例：
  from pathlib import Path
  from features import loom
  loom.serve(Path("bolt"), "127.0.0.1", 8765)       # 前台起，Ctrl+C 停
  server = loom.make(Path("bolt"), "127.0.0.1", 0)  # 测试用：起在空口上，自己管收摊
  server.server_address[1]                          # 系统挑到的真实端口

五个页面：
  /           首页      数据集卡摘要，收下 / 重复 / 拒收 的比例条
  /samples    样本浏览  bolt.jsonl 分页，一页 20 条，能搜关键词
  /rejects    拒收档    rejects.jsonl 按原因码分组，点开看样本
  /progress   进度      loom.jsonl 时间线
  /compare    对比      两份 card.json 并排

安全面就三条，都只在这个文件里：
  1. 只读：一个 GET 路由，不写盘，不收表单。
  2. 路径钉死在产物根目录内：请求里的相对路径展开后不在根下面，回 404。
  3. 正文和出处一律 HTML 转义：样本里有尖括号也不会把版式咬穿。
默认只监听 127.0.0.1。

为什么用标准库 http.server 服务端拼页面，不上前端框架：面板要能断网打开，也不想给接手者
多一套 node 构建。五个页面都短，拼字符串看得完；分页和搜索都在服务端做，浏览器只发一个 GET。
"""

import functools
import html
import http.server
import json
import time
import urllib.parse
from pathlib import Path

PAGE = 20  # 一页几条样本。20 条差不多一屏，再多就得滚很久

STYLE = """
:root { color-scheme: dark; --bg:#14161a; --panel:#1c1f26; --line:#2c313b; --text:#e6e8ec;
        --dim:#8d95a3; --keep:#4f9d69; --dup:#c9a227; --gone:#b5504b; }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--text); font:14px/1.6 ui-sans-serif, system-ui, sans-serif; }
nav { display:flex; gap:4px; padding:10px 16px; border-bottom:1px solid var(--line); background:var(--panel); }
nav a { color:var(--dim); text-decoration:none; padding:4px 10px; border-radius:6px; }
nav a.on, nav a:hover { color:var(--text); background:#2a2f38; }
main { max-width:960px; margin:0 auto; padding:20px 16px 60px; }
h1 { font-size:18px; margin:0 0 14px; }
table { width:100%; border-collapse:collapse; }
th, td { text-align:left; padding:6px 8px; border-bottom:1px solid var(--line); vertical-align:top; }
th { color:var(--dim); font-weight:normal; width:32%; }
a { color:#8fb6e0; }
.bars { display:flex; height:14px; border-radius:7px; overflow:hidden; background:#22262e; margin:10px 0 6px; }
.bars span { display:block; }
.kept { background:var(--keep); } .dup { background:var(--dup); } .gone { background:var(--gone); }
.legend { display:flex; gap:16px; color:var(--dim); font-size:12px; }
.sample { border:1px solid var(--line); border-radius:8px; margin:0 0 12px; background:var(--panel); }
.sample header { padding:6px 10px; color:var(--dim); border-bottom:1px solid var(--line); font-size:12px; }
.sample pre { margin:0; padding:10px; white-space:pre-wrap; word-break:break-word; font:13px/1.6 ui-monospace, monospace; }
form { display:flex; gap:8px; margin:0 0 16px; }
input { flex:1; padding:7px 10px; border-radius:6px; border:1px solid var(--line); background:#0f1115; color:var(--text); }
button, .page a { padding:7px 12px; border-radius:6px; border:1px solid var(--line); background:var(--panel); color:var(--text); cursor:pointer; text-decoration:none; }
.page { display:flex; align-items:center; gap:12px; margin:16px 0; color:var(--dim); }
.grid { display:grid; grid-template-columns:1fr 1fr; gap:14px; }
.card { border:1px solid var(--line); border-radius:8px; padding:10px 12px; background:var(--panel); }
.tl { border-left:2px solid var(--line); margin-left:6px; padding-left:12px; }
.tl div { padding:4px 0; color:var(--dim); }
.tl time { color:var(--text); margin-right:8px; }
.note { color:var(--dim); }
.warn { color:var(--gone); }
"""

LINKS = (("/", "首页"), ("/samples", "样本"), ("/rejects", "拒收档"), ("/progress", "进度"), ("/compare", "对比"))


# --- 拼页面的小件 ---

def _esc(value) -> str:
    """转义成能塞进 HTML 的文本。样本里的尖括号不转义就会破版。"""
    return html.escape("" if value is None else str(value), quote=True)


def _nav(on: str) -> str:
    """顶部五个页面的入口，当前页点亮。"""
    parts = []
    for path, name in LINKS:
        mark = " class='on'" if path == on else ""        # 当前页给个高亮
        parts.append(f"<a href='{path}'{mark}>{name}</a>")
    return "".join(parts)


def _page(title: str, body: str, on: str = "") -> str:
    """拼一整页。样式内联成一份，断网也开得出来。"""
    return (
        "<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>"
        f"<title>{_esc(title)} · 星织面板</title><style>{STYLE}</style></head><body>"
        f"<nav>{_nav(on)}</nav><main>{body}</main></body></html>"
    )


def _pager(here: str, page: int, total: int, extra: str = "") -> str:
    """上一页 / 下一页。extra 带上搜索词或原因码，翻页不丢条件。"""
    if total <= PAGE:                                     # 只有一页就不用翻
        return ""
    before = f"<a href='{here}?page={page - 1}{extra}'>上一页</a>" if page > 1 else ""     # 第一页没有上一条
    after = f"<a href='{here}?page={page + 1}{extra}'>下一页</a>" if page * PAGE < total else ""  # 尾页没有下一条
    sheets = max(1, -(-total // PAGE))                    # 向上取整，算出总页数
    return f"<div class='page'>{before}<span>第 {page} 页 / 共 {sheets} 页</span>{after}</div>"


def _cards(rows: list, page: int) -> str:
    """把一页样本拼成卡片。正文和细节都转义，pre 保住换行。"""
    parts = []
    for index, row in enumerate(rows, (page - 1) * PAGE + 1):     # 行号接着整份往下数
        origin = row.get("origin", "")
        detail = row.get("detail", "")
        head = f"#{index} · {_esc(origin)}" + (f" · 细节 {_esc(detail)}" if detail else "")  # 拒收档带细节
        text = row.get("text", "")
        shown = f"<pre>{_esc(text)}</pre>" if text else ""   # 拒收档没有正文就不占一行
        parts.append(f"<article class='sample'><header>{head}</header>{shown}</article>")
    return "".join(parts)


def _card_html(card: dict, title: str) -> str:
    """一份卡片的字段。首页和对比页都用它，字段顺序才不会各写一套。"""
    counts = card.get("counts", {})
    gone = sum(counts.get("refused", {}).values())        # 拒收原因码加起来
    rows = [
        ("名称", card.get("name", "")),
        ("许可", card.get("license", "")),
        ("条数 读/收/重复/拒收", f"{counts.get('read', 0)} / {counts.get('kept', 0)} / {counts.get('duplicated', 0)} / {gone}"),
        ("指纹", card.get("fingerprint", "")),
        ("去重", card.get("dedup", "")),
        ("试跑", "是" if card.get("dry_run") else "否"),
        ("来源", "; ".join(str(one.get("uri")) for one in card.get("sources", []))),
    ]
    body = "".join(f"<tr><th>{name}</th><td>{_esc(value)}</td></tr>" for name, value in rows)
    return f"<div class='card'><h1>{_esc(title)}</h1><table>{body}</table></div>"


# --- 找产物 ---

def _at(root: Path, rel: str):
    """把请求里的相对路径钉在根目录内，越界返回 None。"""
    target = (root / rel).resolve()                       # resolve 把 .. 和软链都展开
    return target if target.is_relative_to(root) else None  # 展开后不在根下面就是逃出去了


def _load(root: Path, params: dict, default: str):
    """读这一页要的 jsonl 行。默认名字能被 ?p= 换掉，换出根目录返回 None。"""
    rel = (params.get("p") or [default])[0]               # 没给 p 就读默认那份产物
    path = _at(root, rel)                                 # 只在根目录里找
    if path is None or not path.is_file():                # 越界，或者文件不在
        return None
    lines = path.read_text(encoding="utf-8").splitlines()  # 一行一条 JSON
    return [json.loads(line) for line in lines if line]


def _read_json(path: Path) -> dict:
    """读一份卡片。"""
    return json.loads(path.read_text(encoding="utf-8"))


def _num(params: dict, key: str) -> int:
    """取一个正整数的页码，没给或乱填都按第 1 页。"""
    raw = (params.get(key) or ["1"])[0]
    return int(raw) if raw.isdigit() and int(raw) > 0 else 1  # 非数字退回第 1 页


# --- 服务本体 ---

class Panel(http.server.BaseHTTPRequestHandler):
    """一次请求。产物根目录在 make() 里塞进来，这个类只读它。"""

    def __init__(self, request, address, server, root):
        self.root = Path(root).resolve()                  # 入口就定死绝对路径，越界判断靠它比
        super().__init__(request, address, server)

    def do_GET(self):
        """一个 GET 进来，按路径找页面；没登记的路径就是 404。"""
        url = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(url.query)
        route = ROUTES.get(url.path)                      # 路径 → 页面函数
        if route is None:                                 # 没有这一页
            self.reply(404, _page("没有这一页", f"<h1>没有这一页</h1><p class='note'>{_esc(url.path)}</p>"))
            return
        code, body = route(self, params)                  # 页面自己说清用哪个状态码
        self.reply(code, body)

    def reply(self, code: int, body: str) -> None:
        """发一整页。"""
        raw = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))  # 说清长度，浏览器才知道发完了
        self.end_headers()
        self.wfile.write(raw)

    def gone(self, note: str):
        """一页读不到数据的提示，统一回 404。"""
        return 404, _page("读不到", f"<h1>读不到</h1><p class='warn'>{_esc(note)}</p>")

    # --- 五个页面 ---

    def index(self, params):
        """首页：数据集卡摘要，加一条收下 / 重复 / 拒收的比例条。"""
        card = _at(self.root, "card.json")
        if card is None or not card.is_file():            # 没有卡就别装了，先跑 weave
            return self.gone("这个目录里没有 card.json，先跑一趟 ori weave")
        data = _read_json(card)
        counts = data.get("counts", {})
        total = counts.get("read", 0) or 1                # 读 0 条时按 1 算，好除比例
        kept, dup = counts.get("kept", 0), counts.get("duplicated", 0)  # 收下的和重复的
        gone = sum(counts.get("refused", {}).values())    # 拒收原因码加起来
        body = (
            f"<div class='bars'><span class='kept' style='width:{kept / total * 100:.1f}%'></span>"
            f"<span class='dup' style='width:{dup / total * 100:.1f}%'></span>"
            f"<span class='gone' style='width:{gone / total * 100:.1f}%'></span></div>"
            f"<div class='legend'><span>收下 {kept}</span><span>重复 {dup}</span><span>拒收 {gone}</span></div>"
            "<br>"
            + _card_html(data, f"{data.get('name', '无名数据集')} · 数据集卡")
        )
        return 200, _page("首页", body, "/")

    def samples(self, params):
        """样本浏览器：bolt.jsonl 分页，一页 20 条，可按关键词过滤。"""
        rows = _load(self.root, params, "bolt.jsonl")
        if rows is None:                                  # 越界，或者没有正文档
            return self.gone("bolt.jsonl 读不到，路径可能跑出了产物目录")
        q = (params.get("q") or [""])[0].strip()          # 搜索词，先去掉首尾空格
        if q:                                             # 有关键词就按正文和出处过滤
            rows = [row for row in rows if q.lower() in (row.get("text", "") + row.get("origin", "")).lower()]
        page = _num(params, "page")
        shown = rows[(page - 1) * PAGE : page * PAGE]     # 只切出这一页
        extra = f"&q={urllib.parse.quote(q)}" if q else ""  # 翻页时把搜索词带上
        body = (
            f"<h1>样本浏览器</h1><form><input name='q' value='{_esc(q)}' placeholder='搜正文或出处'>"
            "<button>搜</button></form>"
            f"<p class='note'>命中 {len(rows)} 条，一页 {PAGE} 条</p>"
            + (_cards(shown, page) or "<p class='note'>这一页没有样本</p>")
            + _pager("/samples", page, len(rows), extra)
        )
        return 200, _page("样本浏览器", body, "/samples")

    def rejects(self, params):
        """拒收档：按原因码分组统计，点了某个原因码才展开它的样本。"""
        rows = _load(self.root, params, "rejects.jsonl")
        if rows is None:                                  # 越界，或者没有拒收档
            return self.gone("rejects.jsonl 读不到，路径可能跑出了产物目录")
        code = (params.get("code") or [""])[0].strip()    # 点开的原因码
        tally = {}                                        # 原因码 → 条数
        for row in rows:
            reason = row.get("reason", "未标")
            tally[reason] = tally.get(reason, 0) + 1
        lines = "".join(
            f"<tr><td><a href='/rejects?code={urllib.parse.quote(name)}'>{_esc(name)}</a></td><td>{count}</td></tr>"
            for name, count in sorted(tally.items(), key=lambda one: -one[1])  # 条数多的排前面
        )
        body = f"<h1>拒收档</h1><p class='note'>共 {len(rows)} 条，按原因码分组</p><table>{lines}</table>"
        if code:                                          # 点了原因码就展开这一类
            picked = [row for row in rows if row.get("reason") == code]
            page = _num(params, "page")
            extra = f"&code={urllib.parse.quote(code)}"   # 翻页时把原因码带上
            body += f"<h2>原因码 {_esc(code)}</h2>"
            body += _cards(picked[(page - 1) * PAGE : page * PAGE], page) or "<p class='note'>没有样本</p>"
            body += _pager("/rejects", page, len(picked), extra)
        return 200, _page("拒收档", body, "/rejects")

    def progress(self, params):
        """进度：loom.jsonl 时间线，一行一条事件。"""
        rows = _load(self.root, params, "loom.jsonl")
        if rows is None:                                  # 越界，或者还没有事件
            return self.gone("loom.jsonl 读不到，可能还没跑过 weave")
        parts = []
        for row in rows:
            clock = time.strftime("%H:%M:%S", time.localtime(row.get("t", 0)))  # 时间戳换成钟点
            extra = " ".join(f"{k}={v}" for k, v in row.items() if k not in ("t", "stage", "text", "level"))  # 附带字段
            level = "warn" if row.get("level") != "info" else ""  # 警告和报错标红
            parts.append(
                f"<div class='{level}'><time>{clock}</time>[{_esc(row.get('stage'))}] "
                f"{_esc(row.get('text'))} {_esc(extra)}</div>"
            )
        body = f"<h1>进度</h1><p class='note'>共 {len(rows)} 条事件</p><div class='tl'>{''.join(parts)}</div>"
        return 200, _page("进度", body, "/progress")

    def compare(self, params):
        """对比：这一边和另一个目录的卡片并排。"""
        other = (params.get("other") or [""])[0].strip()  # 另一个产物目录
        card = _at(self.root, "card.json")
        if card is None or not card.is_file():            # 这一边都没有卡，比不了
            return self.gone("这一边没有 card.json，先跑一趟 ori weave")
        left = _read_json(card)
        right = None
        if other:                                         # other 是另一份产物，本来就在本根之外，所以故意不过 _at
            other_card = Path(other).expanduser().resolve() / "card.json"
            right = _read_json(other_card) if other_card.is_file() else None  # 那边没卡就留空
        rhs = _card_html(right, "那边") if right else "<div class='card'><h1>那边</h1><p class='note'>没给目录，或者那边没有 card.json</p></div>"
        body = (
            f"<h1>对比</h1><form><input name='other' value='{_esc(other)}' placeholder='另一个产物目录，例如 out/bolt'>"
            "<button>比</button></form>"
            "<p class='note'>对比的目录不受本面板根目录限制（比的就是别处的产物），只读它的 card.json。</p>"
            f"<div class='grid'>{_card_html(left, '这一边')}{rhs}</div>"
        )
        return 200, _page("对比", body, "/compare")


ROUTES = {                                # 路径 → 页面函数，加一页就在这里加一行
    "/": Panel.index,
    "/samples": Panel.samples,
    "/rejects": Panel.rejects,
    "/progress": Panel.progress,
    "/compare": Panel.compare,
}


def make(root, host: str = "127.0.0.1", port: int = 0) -> http.server.ThreadingHTTPServer:
    """建一个服务但先不转，测试拿它自己起线程。端口给 0 让系统挑空口。"""
    return http.server.ThreadingHTTPServer((host, port), functools.partial(Panel, root=root))


def serve(root, host: str = "127.0.0.1", port: int = 0) -> None:
    """前台起服务，Ctrl+C 停。默认只监听本机。"""
    server = make(root, host, port)
    print(f"面板起在 http://{host}:{server.server_address[1]}/  根目录 {Path(root).resolve()}")
    print("只读服务，Ctrl+C 停")
    try:
        server.serve_forever()
    except KeyboardInterrupt:                             # 手动停不算故障
        print("\n收工")
    finally:
        server.server_close()
