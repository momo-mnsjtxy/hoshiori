"""网关的测试：真起一个服务，用 urllib 打接口，再验限流、鉴权和 usage 的算法来源。

约定：端口一律给 0，由系统挑一个空的；起了就一定要收掉，别把端口漏在机器上。
"""

import json
import sys
import urllib.error
import urllib.request

import pytest

from features.gateway import Gateway


@pytest.fixture()
def alive():
    """起一个回声网关，用完收掉。"""
    gate = Gateway("echo-model", backend="echo", limit=60, log=None)  # 不写日志，免得建文件
    server = gate.start(0, in_thread=True)                            # 0 端口，后台线程跑
    yield gate, f"http://127.0.0.1:{server.server_address[1]}"        # 把真端口给用例
    server.shutdown()                                                 # 停循环
    server.server_close()                                             # 放端口
    gate.close()                                                      # 关手柄


@pytest.fixture()
def guarded():
    """起一个要 token 的回声网关，用完收掉。"""
    gate = Gateway("echo-model", backend="echo", token="s3cret", limit=60, log=None)  # 限流放宽，只测鉴权
    server = gate.start(0, in_thread=True)
    yield gate, f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()
    gate.close()


def get(url, path, token=None):
    """打一条 GET，返回状态码、解析好的 JSON、响应头。"""
    ask = urllib.request.Request(url + path, headers=bearer(token))
    try:
        with urllib.request.urlopen(ask, timeout=5) as back:
            return back.status, json.loads(back.read()), back.headers
    except urllib.error.HTTPError as bad:  # 4xx 也会抛，状态码要拿回来
        return bad.code, json.loads(bad.read()), bad.headers


def post(url, path, body, token=None):
    """打一条 POST，返回状态码、原文、响应头。"""
    raw = json.dumps(body).encode("utf-8")
    ask = urllib.request.Request(url + path, data=raw, headers={"Content-Type": "application/json", **bearer(token)})
    try:
        with urllib.request.urlopen(ask, timeout=5) as back:
            return back.status, back.read().decode("utf-8"), back.headers
    except urllib.error.HTTPError as bad:  # 401、429 走这里
        return bad.code, bad.read().decode("utf-8"), bad.headers


def bearer(token):
    """给了 token 才拼 Authorization 头，没给就是空字典。"""
    return {"Authorization": f"Bearer {token}"} if token is not None else {}


def ask(url, token=None):
    """问一句最普通的聊天，返回状态码、解析好的 JSON、响应头。"""
    status, raw, headers = post(url, "/v1/chat/completions", {"model": "echo-model", "messages": [{"role": "user", "content": "你好"}]}, token)
    try:
        return status, json.loads(raw), headers
    except ValueError:  # 流式和纯文本不是 JSON
        return status, raw, headers


def test_models_lists_one(alive):
    _, url = alive
    status, body, _ = get(url, "/v1/models")
    assert status == 200
    assert isinstance(body["data"], list)
    assert body["data"][0]["id"] == "echo-model"


def test_chat_answers_string(alive):
    _, url = alive
    status, body, _ = ask(url)
    assert status == 200
    assert body["choices"][0]["message"]["content"] == "你好"


def test_chat_streams_sse(alive):
    _, url = alive
    status, body, _ = post(url, "/v1/chat/completions", {"model": "echo-model", "messages": [{"role": "user", "content": "数到十"}], "stream": True})
    assert status == 200
    lines = [line for line in body.splitlines() if line.strip()]       # 空行是块分隔，不看
    assert lines and all(line.startswith("data: ") for line in lines)  # 每一行都得是事件行
    assert lines[-1] == "data: [DONE]"                                 # 末尾必须是结束标记
    said = "".join(json.loads(line[6:])["choices"][0]["delta"].get("content", "") for line in lines[:-1])
    assert said == "数到十"                                               # 拼回来就是原文


def test_limit_returns_429():
    gate = Gateway("echo-model", backend="echo", limit=1, log=None)     # 一分钟只准一条
    server = gate.start(0, in_thread=True)
    url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        first, _, _ = ask(url)
        second, body, _ = ask(url)
        assert first == 200
        assert second == 429
        assert body["error"]["type"] == "rate_limit_error"  # OpenAI 的错误形状
    finally:
        server.shutdown()
        server.server_close()
        gate.close()


# --- 鉴权 ---

def test_no_token_on_both_routes_401(guarded):
    """设了 token，两条路由不带 Authorization 都要 401，且错误体是 OpenAI 形状。"""
    _, url = guarded
    for status, body, _ in (get(url, "/v1/models"), ask(url)):
        assert status == 401
        assert body["error"]["type"] == "invalid_request_error"  # 照 OpenAI 的 error 对象
        assert body["error"]["code"] == 401
        assert body["error"]["message"]                            # 要说清怎么带


def test_wrong_token_401(guarded):
    _, url = guarded
    status, body, _ = ask(url, "nope")
    assert status == 401
    assert body["error"]["type"] == "invalid_request_error"
    status, body, _ = get(url, "/v1/models", "nope")
    assert status == 401


def test_right_token_200(guarded):
    _, url = guarded
    status, body, _ = ask(url, "s3cret")
    assert status == 200
    assert body["choices"][0]["message"]["content"] == "你好"
    status, body, _ = get(url, "/v1/models", "s3cret")
    assert status == 200


def test_bearer_scheme_ignores_case(guarded):
    """Bearer 这个方案名大小写不敏感，令牌本身必须一字不差。"""
    _, url = guarded
    for scheme in ("bearer", "BEARER", "BeArEr"):
        ask_ = urllib.request.Request(url + "/v1/models", headers={"Authorization": f"{scheme} s3cret"})
        with urllib.request.urlopen(ask_, timeout=5) as back:
            assert back.status == 200
    ask_ = urllib.request.Request(url + "/v1/models", headers={"Authorization": "bearer S3CRET"})  # 令牌大小写要一样
    with pytest.raises(urllib.error.HTTPError) as bad:
        urllib.request.urlopen(ask_, timeout=5)
    assert bad.value.code == 401


# --- 用量 ---

def test_usage_source_is_estimate_without_tokenizer(alive):
    """模型目录里没分词器就走字数估，响应头要标 estimate。"""
    _, url = alive
    status, body, headers = ask(url)
    assert status == 200
    assert headers["x-usage-source"] == "estimate"
    assert body["usage"]["prompt_tokens"] == 2  # "你好" 两个字


def test_usage_counts_with_tokenizer(tmp_path):
    """模型目录里有分词器就真数 token，响应头标 tokenizer，和字数不一样。"""
    pytest.importorskip("tokenizers")  # 没装这个库就跳过，估算那条已经单独覆盖
    from tokenizers import Tokenizer, models, pre_tokenizers

    made = Tokenizer(models.WordLevel(vocab={"[UNK]": 0, "你好": 1, "世界": 2}, unk_token="[UNK]"))
    made.pre_tokenizer = pre_tokenizers.Whitespace()  # 按空格切词，一次 encode 就是真 token 数
    where = tmp_path / "m"
    where.mkdir()
    made.save(str(where / "tokenizer.json"))

    gate = Gateway(str(where), backend="echo", limit=60, log=None)
    server = gate.start(0, in_thread=True)
    try:
        sentence = "你好 世界"  # 2 个 token，但 5 个字，两者能区分开
        status, body, headers = post(
            f"http://127.0.0.1:{server.server_address[1]}",
            "/v1/chat/completions",
            {"messages": [{"role": "user", "content": sentence}]},
        )
        assert gate.usage_source == "tokenizer"  # 加载一次就定了，不随请求变
        assert status == 200
        assert headers["x-usage-source"] == "tokenizer"
        assert json.loads(body)["usage"] == {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4}
    finally:
        server.shutdown()
        server.server_close()
        gate.close()


def test_missing_tokenizers_falls_back_to_estimate(tmp_path, monkeypatch):
    """目录里有 tokenizer.json 但没装 tokenizers 库，退回估算并在头上标出来。"""
    where = tmp_path / "m"
    where.mkdir()
    (where / "tokenizer.json").write_text("{}", encoding="utf-8")   # 内容无所谓，库都没装
    monkeypatch.setitem(sys.modules, "tokenizers", None)            # 让 from tokenizers import 直接失败

    gate = Gateway(str(where), backend="echo", limit=60, log=None)
    server = gate.start(0, in_thread=True)
    try:
        assert gate.usage_source == "estimate"                      # 缺库就当没有分词器
        status, body, headers = ask(f"http://127.0.0.1:{server.server_address[1]}")
        assert status == 200
        assert headers["x-usage-source"] == "estimate"
        assert body["usage"]["prompt_tokens"] == 2                  # 退回按字数估
    finally:
        server.shutdown()
        server.server_close()
        gate.close()
