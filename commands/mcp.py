"""命令 mcp：列服务器、看工具，或用 JSON 参数调用一次工具。

调用方式：
  ori mcp list
  ori mcp tools <服务器名>
  ori mcp call <服务器名> <工具名> --args '<JSON 对象>'

调用示例：
  ori mcp call local echo --args '{"text":"星织"}'

配置固定读当前目录的 mcp.yaml。检查集中在命令开头：服务器名和工具名先去空白，
--args 必须是 JSON 对象；通过后才启动外部进程。每条命令结束都会停止服务器。
"""

import json                                               # 命令行参数和 MCP 结果都用 JSON
from pathlib import Path                                  # 默认配置明确落在当前工作目录

from features.mcp import Servers                          # 连接、握手、调用都归 MCP 主体
from utils.fail import fail                               # 用户能改的参数按退出码 2 返回

CONFIG = Path("mcp.yaml")                                # CLI 没有隐藏搜索路径，只认这一份


def command(args) -> int:
    """执行 mcp 的 list、tools 或 call 动作。"""
    servers = Servers(CONFIG)                             # 先读配置，名字清单只有一个来源
    if args.action == "list":                            # list 没有别的用户输入要检查
        return show_servers(servers)                      # 不启动任何服务器就能列出配置

    name = args.name.strip()                              # 纯空白名字也算没填
    if not name:                                          # argparse 只检查参数存在，不看内容
        fail("input", "MCP 服务器名字不能为空")         # 用户换一个有内容的名字即可
    if name not in servers.names():                       # 启动进程前先核对配置里的名字
        available = ", ".join(servers.names()) or "（没有已配置服务器）"  # 把可选项带回
        fail("input", f"找不到 MCP 服务器 {name}", hint=f"可用: {available}")

    if args.action == "tools":                           # tools 没有 JSON 参数要解析
        try:                                              # 成功失败都要收掉 stdio 子进程
            tools = servers.start(name)                   # initialize 后拿 tools/list
            return show_tools(tools)                      # 展示名字和说明
        finally:
            servers.stop(name)                            # CLI 一次调用不保留后台连接

    tool = args.tool.strip()                              # 工具名来自命令行，同样先整理
    raw = args.args.strip()                               # JSON 外侧空白不影响含义
    if not tool:                                          # 空工具名无法组成 tools/call
        fail("input", "MCP 工具名字不能为空")           # 在启动服务器前退回
    try:                                                  # JSON 是系统边界，只在这里解析一次
        arguments = json.loads(raw)                       # 后面直接交给 features.mcp
    except json.JSONDecodeError as bad:                   # 语法位置对用户修输入有用
        fail("input", "--args 不是合法 JSON", where=str(bad))
    if not isinstance(arguments, dict):                   # MCP tools/call.arguments 必须是对象
        fail("input", "--args 必须是 JSON 对象", hint='例: {"text":"星织"}')

    try:                                                  # call 会在需要时自己 start
        result = servers.call(name, tool, arguments)      # 只调一次，返回完整 CallToolResult
        print(json.dumps(result, ensure_ascii=False, indent=2))  # 保留 content、isError 等字段
        return 0                                          # JSON-RPC 成功收到结果
    finally:
        servers.stop(name)                                # 超时或远端错误也不留孤儿进程


def show_servers(servers: Servers) -> int:
    """把配置中的服务器名字逐行列出。"""
    names = servers.names()                               # 顺序照用户的 yaml
    if not names:                                         # 空配置也明确说明读成功了
        print("（无）")                                  # 与插件清单的空态说法一致
    for name in names:                                    # 名字可直接复制给 tools 或 call
        print(name)                                       # 一台服务器占一行，方便管道处理
    return 0                                              # 读配置成功就是成功


def show_tools(tools: list) -> int:
    """把 tools/list 的名字和说明逐行列出。"""
    if not tools:                                         # 服务器可以合法地没有工具
        print("（无）")                                  # 用户知道握手确实完成了
    for tool in tools:                                    # start 已核对每项都有 name
        description = tool.get("description", "").strip()  # 说明是可选字段
        suffix = f"  {description}" if description else ""  # 没说明就只打印名字
        print(f"{tool['name']}{suffix}")                 # 名字可直接复制给 call
    return 0                                              # 工具清单完整展示
