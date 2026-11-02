"""公开榜：提交端负载，以及收榜、静态页面、发布命令的完整边界。"""

import json

import main as cli
from features import board


def _report(tmp_path):
    path = tmp_path / "report.json"
    path.write_text(json.dumps({
        "model": "starfur-1b",
        "total": 0.875,
        "recipe": "sha256:abc",
        "hardware": {"cpu": "demo", "gpu": ["card"]},
    }), encoding="utf-8")
    return path


def _submission(folder, name, model, score, time, **extra):
    """写一份收榜用提交；extra 可以覆盖字段，也可以夹一项不该公开的内容。"""
    data = {
        "model": model,
        "score": score,
        "recipe": "sha256:1234567890abcdef",
        "hardware": {"cpu": "demo", "gpu": ["card"]},
        "time": time,
        **extra,
    }
    path = folder / f"board-{name}.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


class Reply:
    """给 urlopen 用的最小响应：能进 with，也能读完。"""

    def __enter__(self):
        return self

    def __exit__(self, kind, value, trace):
        return False

    def read(self):
        return b'{"ok": true}'


def test_submit_posts_public_fields(monkeypatch, tmp_path):
    seen = {}

    def open_url(request, timeout):
        seen["request"] = request
        seen["timeout"] = timeout
        return Reply()

    monkeypatch.setattr(board, "urlopen", open_url)
    result = board.submit(_report(tmp_path), "https://board.example/submit")
    payload = json.loads(seen["request"].data.decode("utf-8"))
    assert seen["request"].get_method() == "POST"
    assert set(payload) == {"model", "score", "recipe", "hardware", "time"}
    assert payload["model"] == "starfur-1b"
    assert payload["score"] == 0.875
    assert payload["recipe"] == "sha256:abc"
    assert payload["hardware"] == {"cpu": "demo", "gpu": ["card"]}
    assert result["submitted"] and result["pending"] is None


def test_submit_without_url_writes_pending(tmp_path):
    result = board.submit(_report(tmp_path), "")
    pending = tmp_path / "board-pending.json"
    saved = json.loads(pending.read_text(encoding="utf-8"))
    assert not result["submitted"] and result["pending"] == pending
    assert saved["url"] == ""
    assert saved["model"] == "starfur-1b"
    assert saved["score"] == 0.875
    assert saved["recipe"] == "sha256:abc"
    assert saved["hardware"] == {"cpu": "demo", "gpu": ["card"]}
    assert saved["time"]


def test_collect_sorts_and_skips_missing_fields(tmp_path):
    """分高在前，同分先到在前；缺字段文件不混进有效行。"""
    _submission(tmp_path, "late", "同分后到", 0.8, "2026-11-02T10:00:00+00:00")
    _submission(tmp_path, "low", "低分", 0.4, "2026-11-02T08:00:00+00:00")
    _submission(tmp_path, "early", "同分先到", 0.8, "2026-11-02T09:00:00+00:00")
    bad = _submission(tmp_path, "bad", "残缺", 1.0, "2026-11-02T07:00:00+00:00")
    broken = json.loads(bad.read_text(encoding="utf-8"))
    del broken["hardware"]
    bad.write_text(json.dumps(broken, ensure_ascii=False), encoding="utf-8")

    rows = board.collect(tmp_path)

    assert [row["model"] for row in rows] == ["同分先到", "同分后到", "低分"]
    assert len(rows.skipped) == 1
    assert rows.skipped[0]["file"] == "board-bad.json"
    assert "hardware" in rows.skipped[0]["reason"]


def test_render_escapes_fields_and_orders_ties():
    """直接渲染普通列表也重排同分行，并转义每一项外来文字。"""
    rows = [
        {
            "model": "后到<script>alert(1)</script>",
            "score": 0.8,
            "recipe": "sha256:1234567890abcdef",
            "hardware": {"cpu": "<b>late</b>"},
            "time": "2026-11-02T10:00:00+00:00",
        },
        {
            "model": "先到 & 安全",
            "score": 0.8,
            "recipe": "sha256:abcdef0123456789",
            "hardware": {"cpu": "early"},
            "time": "2026-11-02T09:00:00+00:00",
        },
        {"model": "残缺", "score": 1.0, "recipe": "sha256:bad", "time": "2026-11-02T08:00:00Z"},
    ]

    page = board.render(rows)

    assert page.index("先到 &amp; 安全") < page.index("后到&lt;script&gt;")
    assert "<script>alert(1)</script>" not in page
    assert "&lt;b&gt;late&lt;/b&gt;" in page
    assert "12345678" in page and "1234567890abcdef" not in page
    assert "abcdef01" in page
    assert page.index("跳过的提交") > page.index("后到&lt;script&gt;")
    assert "第 3 行" in page and "hardware" in page


def test_publish_writes_html_and_json(tmp_path):
    """一次发布落两份文件，机器榜沿用页面的排序且不泄露额外字段。"""
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    _submission(
        inbox,
        "one",
        "starfur-1b",
        0.875,
        "2026-11-02T09:00:00+00:00",
        private="不该公开",
    )
    out = tmp_path / "site"

    paths = board.publish(inbox, out)
    saved = json.loads((out / "board.json").read_text(encoding="utf-8"))

    assert paths == {"html": out / "index.html", "json": out / "board.json"}
    assert (out / "index.html").is_file() and (out / "board.json").is_file()
    assert "starfur-1b" in (out / "index.html").read_text(encoding="utf-8")
    assert saved[0]["model"] == "starfur-1b"
    assert set(saved[0]) == {"model", "score", "recipe", "hardware", "time"}


def test_collect_command_missing_folder_exits_2(tmp_path, capsys):
    """收件目录不存在、或目录里没有提交，都属于用户可改的输入错误。"""
    missing = tmp_path / "没有这个目录"

    assert cli.run(["board", "collect", str(missing), "--out", str(tmp_path / "site")]) == 2
    assert "公开榜目录不在" in capsys.readouterr().err

    empty = tmp_path / "空目录"
    empty.mkdir()
    assert cli.run(["board", "collect", str(empty), "--out", str(tmp_path / "site")]) == 2
    assert "没有 board-*.json" in capsys.readouterr().err
