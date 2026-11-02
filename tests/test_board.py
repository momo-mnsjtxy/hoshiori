"""公开榜：在线时核对 POST 负载，没 URL 时核对待补文件。"""

import json

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
