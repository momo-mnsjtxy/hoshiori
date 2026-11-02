"""MCP stdio 客户端测试：真实起假服务器，覆盖帧顺序、调用和失败出口。

调用方式：在项目根目录运行 ``python -m pytest tests/test_mcp.py -q``。
调用示例：``python -m pytest tests/test_mcp.py::test_handshake_lists_tools -q``。

假服务器脚本由每个测试写进临时目录。它逐行读 JSON，只有先收到 initialize 和
initialized 才回答 tools/list，因此测试同时约束了一行一帧和生命周期顺序。
"""

import json                                               # 配置、事件日志和命令结果都用 JSON
import sys                                                # 假服务器用当前 Python，机器无需额外命令
import textwrap                                           # 测试内嵌脚本保持可读缩进

import pytest                                             # 断言 Failure 与 argparse 退出

import main                                               # 从真实入口核对命令退出码
from features.mcp import TIMEOUT, Servers                 # 直接核对连接生命周期和默认时限
from utils.fail import Failure                            # 区分 input/schema/missing/error


FAKE_SERVER = r"""
import json
import sys
import time

mode, events_path = sys.argv[1:]
initialized = False


def record(message):
    with open(events_path, "a", encoding="utf-8") as stream:
        stream.write(json.dumps(message, ensure_ascii=False, separators=(",", ":")) + "\n")


def send(message):
    sys.stdout.write(json.dumps(message, ensure_ascii=False, separators=(",", ":")) + "\n")
    sys.stdout.flush()


for line in sys.stdin:
    message = json.loads(line)
    record(message)
    method = message.get("method")

    if method == "initialize":
        if mode == "timeout":
            time.sleep(60)
        if mode == "crash":
            print("假服务器故意崩溃", file=sys.stderr, flush=True)
            raise SystemExit(7)
        send({
            "jsonrpc": "2.0",
            "id": message["id"],
            "result": {
                "protocolVersion": message["params"]["protocolVersion"],
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "fake", "version": "1"}
            }
        })
        continue

    if method == "notifications/initialized":
        initialized = True
        continue

    if method == "tools/list":
        if not initialized:
            print("没有 initialized 通知", file=sys.stderr, flush=True)
            raise SystemExit(8)
        send({
            "jsonrpc": "2.0",
            "id": message["id"],
            "result": {"tools": [{
                "name": "echo",
                "description": "回显文字",
                "inputSchema": {"type": "object"}
            }]}
        })
        continue

    if method == "tools/call":
        if mode == "rpc-error":
            send({
                "jsonrpc": "2.0",
                "id": message["id"],
                "error": {"code": -32601, "message": "没有这个工具"}
            })
            continue
        arguments = message["params"]["arguments"]
        send({
            "jsonrpc": "2.0",
            "id": message["id"],
            "result": {
                "content": [{"type": "text", "text": arguments.get("text", "固定回复")}],
                "structuredContent": {"tool": message["params"]["name"], "arguments": arguments}
            }
        })
"""


def fake(tmp_path, mode="ok"):
    """写一台假服务器和一份指向它的 mcp.yaml，返回配置与事件日志。"""
    script = tmp_path / "fake_mcp.py"                     # 子进程真正执行的脚本
    events = tmp_path / "events.jsonl"                    # 每条收到的帧单独记一行
    config = tmp_path / "mcp.yaml"                        # CLI 也认这个固定文件名
    script.write_text(textwrap.dedent(FAKE_SERVER), encoding="utf-8")  # 测试自带服务器
    config.write_text(                                     # JSON 本身也是合法 yaml
        json.dumps(
            {"fake": {"command": sys.executable, "args": [str(script), mode, str(events)]}},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return config, events                                  # 用例可直接读回真实帧


def frames(path):
    """读取假服务器收到的一行一个 JSON 帧。"""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]  # 顺序不能变


def test_handshake_lists_tools(tmp_path):
    """start 按 initialize、initialized、tools/list 的顺序握手。"""
    config, events = fake(tmp_path)                        # 一台正常回答的服务器
    servers = Servers(config, timeout=0.5)                 # 测试不需要等满生产十秒
    try:
        tools = servers.start("fake")                     # 真起子进程并完成握手
        assert [tool["name"] for tool in tools] == ["echo"]  # 工具清单原样返回
    finally:
        servers.stop("fake")                              # 用例自己也验证 stop 可收尾

    sent = frames(events)                                  # 每行都能独立解析为 JSON
    assert [one["method"] for one in sent] == [            # 生命周期顺序固定
        "initialize",
        "notifications/initialized",
        "tools/list",
    ]
    assert sent[0]["id"] == 1 and sent[2]["id"] == 2     # 请求编号递增
    assert "id" not in sent[1]                            # initialized 是通知，不等响应


def test_call_returns_tool_result(tmp_path):
    """call 发送标准 tools/call 参数并拿回完整结果。"""
    config, events = fake(tmp_path)                        # 假工具会回显 arguments.text
    servers = Servers(config, timeout=0.5)                 # 每次请求都独立计时
    try:
        servers.start("fake")                             # call 前先完成工具清单握手
        result = servers.call("fake", "echo", {"text": "星织"})  # 只调用一次
    finally:
        servers.stop("fake")                              # 成功路径也不留子进程

    assert result["content"] == [{"type": "text", "text": "星织"}]  # CallToolResult 原样交回
    called = frames(events)[-1]                            # 最后一帧就是 tools/call
    assert called["id"] == 3                              # 两条握手请求之后的编号
    assert called["method"] == "tools/call"              # MCP 标准方法名
    assert called["params"] == {"name": "echo", "arguments": {"text": "星织"}}  # 参数不改写


def test_timeout_is_clear_error_and_stops_server(tmp_path):
    """服务器活着但不回答时按截止时间失败，并主动收进程。"""
    config, _ = fake(tmp_path, "timeout")                 # initialize 后睡六十秒
    assert TIMEOUT == 10.0                                 # 生产路径严格使用十秒上限
    servers = Servers(config, timeout=0.1)                 # 用短时限验证同一条路径
    with pytest.raises(Failure) as caught:
        servers.start("fake")                             # 不应真的等六十秒
    assert caught.value.kind == "error"                   # 运行时故障退出码为 1
    assert "0.1 秒内没有回应" in caught.value.message     # 错误串说明是超时，不说成崩溃
    assert not servers.running                             # 超时连接已从运行表摘掉


def test_crash_reports_exit_code_and_stderr(tmp_path):
    """服务器握手时崩溃会带回名字、退出码和 stderr。"""
    config, _ = fake(tmp_path, "crash")                   # initialize 后以 7 退出
    servers = Servers(config, timeout=0.5)                 # 崩溃会立即唤醒，不等超时
    with pytest.raises(Failure) as caught:
        servers.start("fake")                             # stdout EOF 触发崩溃分支
    assert caught.value.kind == "error"                   # main 会返回 1
    assert "fake" in caught.value.message                 # 指明是哪台服务器
    assert "退出码 7" in caught.value.message             # 保留系统退出状态
    assert "假服务器故意崩溃" in caught.value.message     # stderr 给出真正原因


def test_json_rpc_error_is_not_called_a_crash(tmp_path):
    """服务器正常返回 error 对象时保留 code 与 message。"""
    config, _ = fake(tmp_path, "rpc-error")               # 只拒绝 tools/call
    servers = Servers(config, timeout=0.5)                 # 握手本身仍成功
    try:
        servers.start("fake")                             # 先进入可调用状态
        with pytest.raises(Failure) as caught:
            servers.call("fake", "missing", {})          # 远端返回 -32601
        assert "JSON-RPC 错误 -32601" in caught.value.message  # 与进程崩溃分开说
        assert "没有这个工具" in caught.value.message     # 对端原因不丢
    finally:
        servers.stop("fake")                              # error 响应后连接仍可正常停止


def test_cli_list_tools_call_and_bad_args_exit_codes(tmp_path, monkeypatch, capsys):
    """三条有效命令返回 0，命令开头发现坏 JSON 返回 2。"""
    fake(tmp_path)                                         # 当前目录准备固定名 mcp.yaml
    monkeypatch.chdir(tmp_path)                            # CLI 不需要额外配置路径参数

    assert main.run(["mcp", "list"]) == 0               # 第一类：只列配置
    assert capsys.readouterr().out.strip() == "fake"       # 不会启动服务器

    assert main.run(["mcp", "tools", "fake"]) == 0     # 第二类：握手并列工具
    assert "echo  回显文字" in capsys.readouterr().out     # 名字可复制给 call

    assert main.run(["mcp", "call", "fake", "echo", "--args", '{"text":"CLI"}']) == 0  # 第三类
    output = json.loads(capsys.readouterr().out)            # call 输出保持机器可读
    assert output["content"][0]["text"] == "CLI"        # JSON 参数已传到假服务器

    code = main.run(["mcp", "call", "fake", "echo", "--args", "[]"])  # 第四类：坏输入
    assert code == 2                                       # JSON 数组不是 arguments 对象
    assert "--args 必须是 JSON 对象" in capsys.readouterr().err  # 原因可直接修改


def test_cli_error_schema_and_missing_exit_codes(tmp_path, monkeypatch, capsys):
    """崩溃返回 1，坏配置返回 3，缺可执行文件返回 4。"""
    config, _ = fake(tmp_path, "crash")                   # 先核对真实崩溃出口
    monkeypatch.chdir(tmp_path)                            # CLI 从这里找 mcp.yaml
    assert main.run(["mcp", "tools", "fake"]) == 1     # 运行时错误统一为 1
    assert "退出码 7" in capsys.readouterr().err           # main 输出清楚原因

    config.write_text(                                     # 再把命令换成机器上一定没有的路径
        json.dumps({"fake": {"command": str(tmp_path / "不存在"), "args": []}}, ensure_ascii=False),
        encoding="utf-8",
    )
    assert main.run(["mcp", "tools", "fake"]) == 4     # missing 类型对应退出码 4
    assert "命令不存在" in capsys.readouterr().err         # 与服务器自己崩溃分开

    config.write_text(                                     # 最后造一份字段类型不合规的配置
        json.dumps({"fake": {"command": sys.executable, "args": "不是列表"}}, ensure_ascii=False),
        encoding="utf-8",
    )
    assert main.run(["mcp", "list"]) == 3               # schema 类型对应退出码 3
    assert "args 要是字符串列表" in capsys.readouterr().err  # 指到填错的字段


def test_cli_requires_an_action():
    """mcp 后没给 list/tools/call 时由 argparse 返回用法错误 2。"""
    with pytest.raises(SystemExit) as stopped:             # argparse 在命令模块导入前拦住
        main.run(["mcp"])                                 # 少一层动作
    assert stopped.value.code == 2                         # 标准命令行用法错误码
