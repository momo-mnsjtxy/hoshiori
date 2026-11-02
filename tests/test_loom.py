"""面板的测试：五个页面与指标流能不能开，越界、转义、缺卡三件挡不挡得住。

约定：面板起在 0 端口由系统挑空口，每个用例一份临时产物目录，用完收摊。
样本里固定放一段 <script>，用来验转义；根目录外固定放一份 jsonl，用来验 ../ 逃逸。
"""

import json
import threading
import urllib.error
import urllib.parse
import urllib.request

import pytest

import main as cli
from features import loom

SECRET = "<script>alert('x')</script>"      # 尖括号样本，转义不住就会破版


def _get(url: str):
    """打一个页面，拿 (状态码, 正文)。404 不算异常，照样读回来。"""
    try:
        with urllib.request.urlopen(url, timeout=5) as reply:
            return reply.status, reply.read().decode("utf-8")
    except urllib.error.HTTPError as bad:                                # 4xx 也把正文交出来
        return bad.code, bad.read().decode("utf-8")


def _fill(root, outside) -> None:
    """造一份假产物：卡片、正文、拒收、事件各一份，根外再放一份。"""
    card = {
        "schema": "itoguchi/1",
        "name": "测试集",
        "license": "cc0",
        "sources": [{"uri": "a.jsonl", "license": "cc0"}],
        "counts": {"read": 4, "kept": 2, "duplicated": 1, "refused": {"too_short": 1}},
        "fingerprint": "blake3:abc",
        "dedup": "阈值 0.8 / 窗口 5 / 指纹 128",
        "dry_run": False,
    }
    (root / "card.json").write_text(json.dumps(card, ensure_ascii=False), encoding="utf-8")
    kept = [{"text": "第一段正文", "origin": "a.jsonl"}, {"text": SECRET, "origin": "a.jsonl"}]   # 两条收下的
    (root / "bolt.jsonl").write_text(
        "\n".join(json.dumps(one, ensure_ascii=False) for one in kept) + "\n", encoding="utf-8"
    )
    reject = {"reason": "too_short", "detail": "1", "origin": "a.jsonl"}  # 一条拒收的
    (root / "rejects.jsonl").write_text(json.dumps(reject, ensure_ascii=False) + "\n", encoding="utf-8")
    events = [
        {"t": 1700000000.0, "stage": "spin", "text": "训练", "level": "info", "loss": 1.25, "step": 1},
        {"t": 1700000001.0, "stage": "spin", "text": "训练", "level": "info", "loss": 0.75, "step": 2},
        {"t": 1700000002.0, "stage": "weave", "text": "读完", "level": "info", "kept": 2},
    ]
    (root / "loom.jsonl").write_text(
        "\n".join(json.dumps(one, ensure_ascii=False) for one in events) + "\n", encoding="utf-8"
    )
    (outside / "逃出.jsonl").write_text(json.dumps({"text": "根外的正文"}, ensure_ascii=False) + "\n", encoding="utf-8")


@pytest.fixture
def panel(tmp_path):
    """起面板，交回地址、根目录和另一份产物目录；用例完了关服务。"""
    root = tmp_path / "bolt"
    root.mkdir()
    other = tmp_path / "别的"
    other.mkdir()
    _fill(root, tmp_path)
    other_card = {"name": "对照组", "counts": {"read": 1, "kept": 1}}
    (other / "card.json").write_text(json.dumps(other_card, ensure_ascii=False), encoding="utf-8")

    server = loom.make(root, "127.0.0.1", 0)                             # 0 端口，系统挑空口
    thread = threading.Thread(target=server.serve_forever, daemon=True)  # 后台转，别卡住用例
    thread.start()
    yield {"base": f"http://127.0.0.1:{server.server_address[1]}", "root": root, "other": str(other)}
    server.shutdown()
    server.server_close()


def test_five_pages_render(panel):
    """五个页面都 200，且各自的关键字段在。"""
    code, home = _get(panel["base"] + "/")
    assert code == 200 and "测试集" in home and "class='bars'" in home          # 首页有卡和比例条

    code, samples = _get(panel["base"] + "/samples")
    assert code == 200 and "第一段正文" in samples and "#1" in samples          # 样本页有正文和行号

    code, rejects = _get(panel["base"] + "/rejects")
    assert code == 200 and "too_short" in rejects                              # 拒收档按原因码分组

    code, progress = _get(panel["base"] + "/progress")
    assert code == 200 and "读完" in progress and "weave" in progress          # 进度有时间线

    code, compare = _get(panel["base"] + "/compare?other=" + urllib.parse.quote(panel["other"]))
    assert code == 200 and "测试集" in compare and "对照组" in compare         # 对比两张卡并排


def test_metrics_json_returns_only_time_and_loss(panel):
    """指标地址 200，并把普通事件字段剥掉，只留画图所需两项。"""
    code, body = _get(panel["base"] + "/metrics.json")
    points = json.loads(body)
    assert code == 200
    assert points == [
        {"t": 1700000000.0, "loss": 1.25},
        {"t": 1700000001.0, "loss": 0.75},
    ]
    assert all(set(point) == {"t", "loss"} for point in points)               # stage、step 等字段不外送


def test_metrics_since_returns_only_newer_points(panel):
    """since 是时间游标，只回严格晚于它的 loss 点。"""
    code, body = _get(panel["base"] + "/metrics.json?since=1700000000.0")
    assert code == 200
    assert json.loads(body) == [{"t": 1700000001.0, "loss": 0.75}]


def test_metrics_over_limit_merge_adjacent_points(panel):
    """超过 2000 点后相邻 loss 取均值，奇数末点原样保留。"""
    events = [{"t": float(index), "loss": float(index), "step": index} for index in range(2001)]
    (panel["root"] / "loom.jsonl").write_text(
        "\n".join(json.dumps(one) for one in events) + "\n", encoding="utf-8"
    )
    code, body = _get(panel["base"] + "/metrics.json")
    points = json.loads(body)
    assert code == 200 and len(points) == 1001
    assert points[0] == {"t": 1.0, "loss": 0.5}                              # 前两个点合成一个
    assert points[-1] == {"t": 2000.0, "loss": 2000.0}                      # 奇数尾点不丢


def test_home_has_live_loss_svg_and_fetch_script(panel):
    """首页内联 SVG，并按两秒、前台可见、增量三项约定自己刷新。"""
    code, home = _get(panel["base"] + "/")
    assert code == 200 and "<svg id='loss-chart'" in home and "id='loss-line'" in home
    assert 'fetch("/metrics.json" + query' in home and '"?since="' in home     # 拉同源增量地址
    assert "setTimeout(updateLoss, 2000)" in home                              # 每两秒排下一轮
    assert '"visibilitychange"' in home and "document.hidden" in home          # 后台标签停拉
    assert "while (points.length > 2000)" in home                              # 多轮增量累计也会抽稀


def test_search_filters_samples(panel):
    """搜关键词只留命中的那条。"""
    code, body = _get(panel["base"] + "/samples?q=" + urllib.parse.quote("第一段"))
    assert code == 200 and "第一段正文" in body and "alert" not in body


def test_outside_path_blocked(panel):
    """../ 跑出产物目录被挡；同一条路由给根内的相对路径能通。"""
    code, body = _get(panel["base"] + "/samples?p=" + urllib.parse.quote("../逃出.jsonl"))
    assert code == 404 and "跑出" in body                                       # 越界回 404

    code, body = _get(panel["base"] + "/samples?p=bolt.jsonl")
    assert code == 200 and "第一段正文" in body                                 # 根内的相对路径能读


def test_script_in_sample_escaped(panel):
    """样本里的 <script> 转义后才发出去，不破版。"""
    code, body = _get(panel["base"] + "/samples")
    assert code == 200
    assert "&lt;script&gt;" in body                                            # 尖括号被转义
    assert "<script>alert" not in body                                         # 原样的一处都不留


def test_missing_card_exits_2(tmp_path, capsys):
    """目录里没有 card.json 就退回 2，并让人先跑 weave。"""
    empty = tmp_path / "空目录"
    empty.mkdir()
    assert cli.run(["loom", str(empty)]) == 2
    assert "weave" in capsys.readouterr().err


def test_bad_port_exits_2(tmp_path, capsys):
    """端口越界也在命令层挡下，不去 bind。"""
    root = tmp_path / "bolt"
    root.mkdir()
    (root / "card.json").write_text("{}", encoding="utf-8")
    assert cli.run(["loom", str(root), "--port", "70000"]) == 2
    assert "端口" in capsys.readouterr().err
