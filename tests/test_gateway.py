"""网关的测试：真起一个服务，用 urllib 打接口，再验限流。

约定：端口一律给 0，由系统挑一个空的；起了就一定要收掉，别把端口漏在机器上。
"""

import json
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


def get(url, path):
    """打一条 GET，返回状态码和解析好的 JSON。"""
    try:
        with urllib.request.urlopen(url + path, timeout=5) as back:
            return back.status, json.loads(back.read())
    except urllib.error.HTTPError as bad:  # 4xx 也会抛，状态码要拿回来
        return bad.code, json.loads(bad.read())


def post(url, path, body):
    """打一条 POST，返回状态码和原文。"""
    raw = json.dumps(body).encode("utf-8")
    ask = urllib.request.Request(url + path, data=raw, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(ask, timeout=5) as back:
            return back.status, back.read().decode("utf-8")
    except urllib.error.HTTPError as bad:  # 429 走这里
        return bad.code, bad.read().decode("utf-8")


def test_models_lists_one(alive):
    gate, url = alive
    status, body = get(url, "/v1/models")
    assert status == 200
    assert isinstance(body["data"], list)
    assert body["data"][0]["id"] == gate.name


def test_chat_answers_string(alive):
    _, url = alive
    status, body = post(url, "/v1/chat/completions", {"model": "echo-model", "messages": [{"role": "user", "content": "你好"}]})
    assert status == 200
    assert json.loads(body)["choices"][0]["message"]["content"] == "你好"


def test_chat_streams_sse(alive):
    _, url = alive
    status, body = post(url, "/v1/chat/completions", {"model": "echo-model", "messages": [{"role": "user", "content": "数到十"}], "stream": True})
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
        first, _ = post(url, "/v1/chat/completions", {"messages": [{"role": "user", "content": "一"}]})
        second, body = post(url, "/v1/chat/completions", {"messages": [{"role": "user", "content": "二"}]})
        assert first == 200
        assert second == 429
        assert json.loads(body)["error"]["type"] == "rate_limit_error"  # OpenAI 的错误形状
    finally:
        server.shutdown()
        server.server_close()
        gate.close()
