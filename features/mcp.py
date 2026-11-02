"""MCP 服务器：从 yaml 找到命令，用逐行 JSON-RPC 连接它的 stdio。

调用方式：
  servers = Servers(Path("mcp.yaml"))
  tools = servers.start("local")
  result = servers.call("local", "echo", {"text": "星织"})
  servers.stop("local")

配置示例：
  local:
    command: python
    args: [server.py]

一条 stdout 行就是一个 JSON-RPC 帧。start 依次发 initialize、initialized 通知和
tools/list；call 发一次 tools/call。服务器的 stderr 不混进协议流，只在崩溃时带回。
"""

import json                                                # JSON-RPC 帧只用标准库编解码
import queue                                               # 阻塞读 stdout 放到队列，主线程才能计时
import subprocess                                          # MCP 服务器就是一个 stdio 子进程
import tempfile                                            # stderr 单独落临时文件，写满也不会卡服务器
import threading                                           # 每个运行中的服务器有一个逐行读取线程
import time                                                # 每次请求共用一个截止时间
from dataclasses import dataclass, field                   # 把一条连接的状态放在一起
from pathlib import Path                                   # 配置文件由调用方明确给出
from typing import BinaryIO                                # 标明子进程和临时文件都是二进制流

from utils import yaml_io                                  # 沿用全项目一致的 yaml 错误说法
from utils.fail import fail                                # 失败类型交给 main.py 翻译成退出码

PROTOCOL_VERSION = "2025-11-25"                           # initialize 生命周期最后一版协议
TIMEOUT = 10.0                                             # 每个 JSON-RPC 请求最多等十秒
from main import VERSION as CLIENT_VERSION                # 源码与 wheel 都复用入口的版本来源


@dataclass
class _Connection:
    """一个已启动服务器的进程、收件队列和这次会话的编号。"""

    process: subprocess.Popen                              # 正在说 MCP 的子进程
    errors: BinaryIO                                       # stderr 独立保存，不污染 stdout 帧
    messages: queue.Queue = field(default_factory=queue.Queue)  # stdout 每行按到达顺序排队
    reader: threading.Thread = None                        # 只负责 readline 的后台线程
    tools: list = field(default_factory=list)              # start 时 tools/list 的结果
    next_id: int = 1                                       # 请求 id 在本连接内递增


class Servers:
    """一份 MCP 配置和其中当前运行的服务器。"""

    def __init__(self, path: Path, timeout: float = TIMEOUT) -> None:
        """读取并核对一份 ``name: {command, args}`` 配置。"""
        self.path = Path(path).expanduser()                 # 配置位置只在入口定一次
        self.timeout = timeout                              # 测试可缩短，正常调用固定十秒
        self.settings = yaml_io.read(self.path)             # 文件缺失与坏 yaml 沿用统一错误
        self.running = {}                                   # 名字 → 当前连接，不藏全局进程

        for name, choice in self.settings.items():          # yaml 是外部输入，在读入边界核对一次
            if not isinstance(name, str) or not name.strip():  # 名字会出现在命令行里
                fail("schema", "MCP 服务器名字必须是非空字符串", where=str(self.path))
            where = f"{self.path}:{name}"                 # 后面的错误都能指回具体服务器
            if not isinstance(choice, dict):                # 每台服务器要有 command 与 args
                fail("schema", "MCP 服务器配置要写成映射", where=where)
            unknown = [key for key in choice if key not in {"command", "args"}]  # 不默默吞错字段
            if unknown:                                     # 拼错配置应在启动进程前发现
                fail("schema", f"MCP 配置不认识字段 {unknown[0]}", where=where)
            command = choice.get("command")                # 可执行文件由这一项给出
            arguments = choice.get("args", [])             # 不需要参数时可以省略 args
            if not isinstance(command, str) or not command.strip():  # 空命令无法启动
                fail("schema", "MCP 配置要有非空 command", where=f"{where}.command")
            if not isinstance(arguments, list) or any(not isinstance(one, str) for one in arguments):
                fail("schema", "MCP 的 args 要是字符串列表", where=f"{where}.args")
            choice["command"] = command.strip()            # 首尾空白不是命令的一部分
            choice["args"] = arguments                     # 省略时补成统一的空列表

    # --- 查看配置 ---

    def names(self) -> tuple:
        """按 yaml 中的顺序返回服务器名字。"""
        return tuple(self.settings)                         # 字典顺序就是用户写下的顺序

    # --- 启动并握手 ---

    def start(self, name: str) -> list:
        """启动一台服务器，完成 initialize 与 tools/list，返回工具列表。"""
        if name not in self.settings:                       # 直接调用特性层也要给可改的名字错误
            available = ", ".join(self.names()) or "（没有已配置服务器）"  # 把能选的名字带回
            fail("input", f"找不到 MCP 服务器 {name}", hint=f"可用: {available}")
        old = self.running.get(name)                        # 重复 start 不另起一个相同进程
        if old and old.process.poll() is None:              # 现有连接仍活着
            return old.tools                                # 返回第一次握手拿到的工具
        if old:                                             # 旧进程已经退出，先收干净再重连
            self.stop(name)                                 # 临时文件和管道也一起关掉

        choice = self.settings[name]                        # 配置在 __init__ 已核对完整
        errors = tempfile.TemporaryFile()                   # stderr 可持续写，不占管道容量
        try:                                                # 启动错误要区分没装和程序故障
            process = subprocess.Popen(                     # 不走 shell，配置不会被当成命令串执行
                [choice["command"], *choice["args"]],     # command 与 args 原样组成 argv
                stdin=subprocess.PIPE,                      # JSON-RPC 请求写这里
                stdout=subprocess.PIPE,                     # JSON-RPC 响应只能走这里
                stderr=errors,                              # 日志另存，不能破坏一行一个 JSON
                bufsize=0,                                  # 每帧写完立即 flush，不攒在客户端
            )
        except FileNotFoundError:                           # 可执行文件不在机器上
            errors.close()                                  # 进程没起来也要收临时文件
            fail("missing", f"MCP 服务器 {name} 的命令不存在: {choice['command']}")
        except OSError as bad:                              # 权限、格式等其余启动失败
            errors.close()                                  # 不把系统文件描述符留到进程结束
            fail("error", f"MCP 服务器 {name} 启动失败: {bad}")

        connection = _Connection(process=process, errors=errors)  # 从此处开始统一由 stop 收尾
        connection.reader = threading.Thread(              # readline 可以阻塞，放后台等完整帧
            target=_read_lines,                             # 一行一行送回当前连接的队列
            args=(process.stdout, connection.messages),    # 只共享这两个简单对象
            name=f"mcp-{name}-stdout",                    # 排查挂线程时看得出是哪台服务器
            daemon=True,                                    # 主程序异常退出时不被读线程拖住
        )
        connection.reader.start()                          # 先开始收，再发第一条 initialize
        self.running[name] = connection                     # 后续请求都按名字找到同一会话

        try:                                                # 握手任一步失败都不能留下子进程
            initialized = self._request(                    # 第一条请求声明客户端能力
                name,
                "initialize",
                {
                    "protocolVersion": PROTOCOL_VERSION,  # 走用户要求的 initialize 生命周期
                    "capabilities": {},                    # 这个小客户端不接服务器反向能力
                    "clientInfo": {"name": "hoshiori", "version": CLIENT_VERSION},  # 对端日志可辨认来源
                },
            )
            if not isinstance(initialized, dict) or not isinstance(initialized.get("protocolVersion"), str):
                self._protocol_error(name, "initialize 结果缺少 protocolVersion")  # 没完成协商不能继续
            self._send(name, {"jsonrpc": "2.0", "method": "notifications/initialized"})  # 确认协商完成
            listed = self._request(name, "tools/list", {})  # 握手后立即拿一份可调用工具清单
            if not isinstance(listed, dict) or not isinstance(listed.get("tools"), list):
                self._protocol_error(name, "tools/list 结果缺少 tools 列表")  # 外部响应只在边界验一次
            for tool in listed["tools"]:                   # 每个工具至少要有可调用的名字
                if not isinstance(tool, dict) or not isinstance(tool.get("name"), str):
                    self._protocol_error(name, "tools/list 里有工具缺少 name")
            connection.tools = listed["tools"]             # call 与命令层共用这一份清单
            return connection.tools                         # 调用方可直接展示
        except BaseException:                               # 资源清理也覆盖 Ctrl-C，原异常照常抛出
            self.stop(name)                                 # 不让握手失败的服务器留在后台
            raise                                           # Failure 或中断仍交给项目入口

    # --- 调一次工具 ---

    def call(self, name: str, tool: str, arguments: dict):
        """在已启动的服务器上调用一次工具，返回 MCP 的 CallToolResult。"""
        if name not in self.running:                        # 直接调用时也能一条语句完成
            self.start(name)                                # start 同时完成工具清单握手
        return self._request(                               # isError 属于工具结果，不改写成客户端异常
            name,
            "tools/call",
            {"name": tool, "arguments": arguments},      # MCP 标准字段原样送给服务器
        )

    # --- 停止服务器 ---

    def stop(self, name: str) -> None:
        """停止一台服务器；没有运行时也可安全调用。"""
        connection = self.running.pop(name, None)           # 先摘掉，收尾中不会被再次使用
        if connection is None:                              # finally 重复 stop 不算失败
            return                                          # 没有资源需要处理
        _finish(connection)                                 # 关 stdin，必要时 terminate 再 kill

    # --- JSON-RPC 帧 ---

    def _request(self, name: str, method: str, params: dict):
        """发一条有 id 的请求，在同一截止时间内等到对应响应。"""
        connection = self.running[name]                     # start 已建立这条会话
        request_id = connection.next_id                     # 响应必须原样带回这个 id
        connection.next_id += 1                             # 下一条请求使用新编号
        self._send(                                         # 每个字段都按 JSON-RPC 2.0 写出
            name,
            {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params},
        )
        deadline = time.monotonic() + self.timeout          # 通知不能反复刷新十秒上限

        while True:                                         # 对端可能先发日志通知，再发响应
            message = self._receive(name, deadline)         # 取一整行并解析成对象
            if "id" not in message and isinstance(message.get("method"), str):
                continue                                    # 单向通知不需要客户端回复
            if message.get("id") != request_id:            # 串行客户端不该收到别的响应
                self._protocol_error(name, f"响应 id 应为 {request_id}，实际是 {message.get('id')}")
            if "error" in message:                          # JSON-RPC 层明确拒绝了请求
                error = message["error"]                   # code/message/data 一起保留给用户
                if isinstance(error, dict):                 # 标准错误能给出更紧凑的说明
                    code = error.get("code", "?")         # 非标准缺 code 也不遮住 message
                    words = error.get("message", json.dumps(error, ensure_ascii=False))
                    fail("error", f"MCP 服务器 {name} 对 {method} 返回 JSON-RPC 错误 {code}: {words}")
                fail("error", f"MCP 服务器 {name} 对 {method} 返回 JSON-RPC 错误: {error}")
            if "result" not in message:                     # 成功响应必须有 result，哪怕是 null
                self._protocol_error(name, f"{method} 响应缺少 result")
            return message["result"]                       # 结果结构由 start 或调用方按边界处理

    def _send(self, name: str, message: dict) -> None:
        """把一个对象写成单行 UTF-8 JSON-RPC 帧并立即送出。"""
        connection = self.running[name]                     # 每台服务器各有自己的 stdin
        frame = json.dumps(message, ensure_ascii=False, separators=(",", ":")) + "\n"  # 换行就是帧边界
        try:                                                # 对端突然退出时写端会直接断
            connection.process.stdin.write(frame.encode("utf-8"))  # stdio 明确使用 UTF-8 JSON
            connection.process.stdin.flush()               # 下一次 readline 现在就能收到
        except (BrokenPipeError, OSError):                  # 统一补上退出码和 stderr
            self._crashed(name)                             # 不留下模糊的 BrokenPipeError

    def _receive(self, name: str, deadline: float) -> dict:
        """在截止时间前取下一帧，并核对 JSON-RPC 外壳。"""
        connection = self.running[name]                     # 读线程只给这条连接投递
        left = deadline - time.monotonic()                  # 整个请求共用剩余时间
        if left <= 0:                                       # 收到再多通知也不能无限等
            self._timed_out(name)                           # 报十秒超时并收掉进程
        try:                                                # Queue.get 提供跨平台的超时等待
            line = connection.messages.get(timeout=left)    # None 表示 stdout 已到 EOF
        except queue.Empty:                                 # 到点时再区分崩溃与仍在运行
            if connection.process.poll() is not None:       # 进程刚好在等待期间退出
                self._crashed(name)                         # 给退出码和 stderr
            self._timed_out(name)                           # 活着但没回就是协议超时
        if line is None:                                    # 读线程看到了 EOF
            self._crashed(name)                             # 明确说明服务器退出或关了 stdout
        try:                                                # stdout 是外部系统边界，只在这里验
            message = json.loads(line.decode("utf-8"))     # 每行只能装一个 JSON 对象
        except (UnicodeDecodeError, json.JSONDecodeError) as bad:
            self._protocol_error(name, f"不是合法 UTF-8 JSON: {bad}")  # stdout 日志也会在这里指出
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
            self._protocol_error(name, "帧不是 JSON-RPC 2.0 对象")  # 数组和缺版本都不是响应
        return message                                      # 请求层继续核对 id 与 result/error

    # --- 失败时收尾 ---

    def _timed_out(self, name: str) -> None:
        """停止无响应的服务器，抛出带等待时长的错误。"""
        connection = self.running.pop(name)                 # 超时后的会话已不能继续复用
        _, errors = _finish(connection)                     # 停进程后 stderr 才不会继续变化
        detail = _stderr_detail(errors)                     # 对端自己的提示也保留下来
        fail("error", f"MCP 服务器 {name} 在 {self.timeout:g} 秒内没有回应{detail}")

    def _crashed(self, name: str) -> None:
        """收掉提前结束的连接，抛出退出码和 stderr。"""
        connection = self.running.pop(name)                 # 崩掉的连接不再留在运行表里
        try:                                                # EOF 到进程状态更新之间可能差一瞬
            code = connection.process.wait(timeout=0.05)    # 尽量拿到真实退出码
        except subprocess.TimeoutExpired:                   # 只关 stdout、进程仍活着
            code = None                                     # _finish 会负责终止它
        final_code, errors = _finish(connection)            # 所有管道与线程一起收尾
        detail = _stderr_detail(errors)                     # stderr 是最直接的崩溃原因
        if code is None:                                    # 没退出却关 stdout 也是协议断线
            fail("error", f"MCP 服务器 {name} 提前关闭了 stdout（终止码 {final_code}）{detail}")
        fail("error", f"MCP 服务器 {name} 崩了（退出码 {code}）{detail}")

    def _protocol_error(self, name: str, reason: str) -> None:
        """停止已失去同步的服务器，并说明是哪一种坏帧。"""
        connection = self.running.pop(name)                 # 坏帧后无法可靠找到下一条边界
        _, errors = _finish(connection)                     # 协议错误同样不能遗留进程
        detail = _stderr_detail(errors)                     # 对端可能把解释写在 stderr
        fail("error", f"MCP 服务器 {name} 的协议响应无效: {reason}{detail}")


def _read_lines(stdout: BinaryIO, messages: queue.Queue) -> None:
    """从一台服务器持续读取完整行；EOF 用 None 通知等待方。"""
    while True:                                             # 一条连接只需要这一个顺序读取者
        line = stdout.readline()                            # 半帧留在这里，不阻塞有超时的主线程
        if not line:                                        # 子进程退出或主动关闭 stdout
            messages.put(None)                              # 唤醒正在等响应的请求
            return                                          # 这条 stdout 不会再有消息
        messages.put(line)                                  # 保留原始字节，边界处统一解码


def _finish(connection: _Connection) -> tuple:
    """关闭一条连接，返回最终退出码和 stderr 文本。"""
    process = connection.process                            # 后面几步都处理同一个子进程
    if process.poll() is None:                              # 活着时先用 EOF 请它正常收尾
        try:
            process.stdin.close()                           # stdio 服务器通常在输入 EOF 后退出
        except (BrokenPipeError, OSError):                  # 已断开的 stdin 不影响后续回收
            pass                                            # 退出状态会在 wait 里拿到
        try:
            process.wait(timeout=0.2)                       # 给正常清理留一个很短的窗口
        except subprocess.TimeoutExpired:                   # 不响应 EOF 就明确终止
            process.terminate()                             # 先给可捕获的终止信号
            try:
                process.wait(timeout=0.5)                   # 让服务器清自己的临时资源
            except subprocess.TimeoutExpired:               # 连终止信号也不理
                process.kill()                              # 最后才强制收掉，不能留孤儿进程
                process.wait()                              # 确认系统已经回收
    if process.stdin and not process.stdin.closed:          # 已退出但 stdin 还在本进程手里
        process.stdin.close()                               # 释放写端文件描述符
    if process.stdout and not process.stdout.closed:        # 读线程已从 EOF 退出或即将退出
        process.stdout.close()                              # 释放读端文件描述符
    if connection.reader:                                  # 启动完成后一定有线程
        connection.reader.join(timeout=0.2)                 # 不把已结束线程留在调试列表里
    connection.errors.seek(0)                              # 子进程结束后 stderr 已固定
    errors = connection.errors.read().decode("utf-8", errors="replace")  # 非 UTF-8 日志也能展示
    connection.errors.close()                              # 临时文件至此不再需要
    return process.returncode, errors                       # 错误路径用，普通 stop 会忽略


def _stderr_detail(text: str) -> str:
    """把 stderr 压成适合接在错误串后的短说明。"""
    words = " ".join(text.strip().splitlines())            # 多行日志不冲散 Failure 的三行格式
    if not words:                                           # 没写 stderr 就不加空标签
        return ""                                          # 主错误串已经足够明确
    return f"；stderr: {words[-1000:]}"                    # 只留末尾，通常是最近的根因


# --- 把星织作为 stdio MCP 服务器 ---

class Server:
    """在明确根目录内提供卡片、配方校验和套件清单，不启动任意 shell。

    调用示例：Server(Path("out")).serve()；CLI 为 ``ori mcp serve --root out``。
    协议参考：https://modelcontextprotocol.io/specification/2025-11-25/basic/lifecycle
    stdout 只发送 JSON-RPC；现有 Servers 客户端可以直接连接这个服务端。
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root).expanduser().resolve()       # 一次固定可访问的工作区
        if not self.root.is_dir():                          # root 是命令行外部输入
            fail("input", f"MCP 根目录不存在: {self.root}")
        self.initialized = False                            # 通知到达前不开放业务方法
        self.negotiated = False                             # initialize 成功和 initialized 分开记录

    def tools(self) -> list:
        """返回实际可执行的三种工具；输入参数只允许声明的字段。"""
        path_schema = {"type": "object", "properties": {"path": {"type": "string"}},
                       "required": ["path"], "additionalProperties": False}
        return [                                             # 少量固定工具直接列出，避免额外注册层
            {"name": "read_card", "description": "读取根目录内的数据集 card.json", "inputSchema": path_schema},
            {"name": "validate_recipe", "description": "校验根目录内的完整星织配方（不展开 extends）", "inputSchema": path_schema},
            {"name": "list_suites", "description": "列出安装内置的评测套件", "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False}},
        ]

    def call(self, name: str, arguments: dict) -> dict:
        """执行工具并返回 MCP content 与 structuredContent，业务错误由协议边界翻译。"""
        from features import recipe, suite                 # 业务能力按需导入，MCP 本身仍是轻依赖
        if name == "list_suites":
            if arguments:                                  # 不接受拼错的无效参数
                fail("input", "list_suites 不接受参数")
            result = {"suites": [path.stem for path in sorted(suite.folder().glob("*.yaml"))]}
        else:
            if name not in {"read_card", "validate_recipe"}:
                fail("input", f"没有工具 {name}")
            if set(arguments) != {"path"} or not isinstance(arguments["path"], str) or not arguments["path"].strip():
                fail("input", "工具需要非空 path，且不接受额外参数")
            path = (self.root / arguments["path"].strip()).resolve()  # resolve 同时消除 .. 和符号链接
            if not path.is_relative_to(self.root):          # MCP 客户端不能读取根目录外的文件
                fail("input", "path 跑出了 MCP 根目录")
            if name == "read_card":
                path = path / "card.json" if path.is_dir() else path
                path = path.resolve()                      # 目录里的 card.json 本身也可能是越界链接
                if not path.is_relative_to(self.root):
                    fail("input", "card.json 跑出了 MCP 根目录")
                if path.name != "card.json":              # 这个工具只提供卡片，不能当任意文件读取器
                    fail("input", "read_card 只读取 card.json")
                result = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(result, dict):            # 卡片是外部文件，必须是 JSON 对象
                    fail("input", "card.json 必须是对象")
            else:
                data = yaml_io.read(path)                   # 不展开 extends，避免间接读到工作区外
                if data.get("extends"):
                    fail("input", "先将 extends 展开为完整配方，再用 MCP 校验")
                recipe.check(data)                         # 使用 CLI 相同的配方规则
                result = {"valid": True, "path": str(path.relative_to(self.root)), "stamp": recipe.stamp(data)}
        return {"content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}], "structuredContent": result}

    def handle(self, message) -> dict | None:
        """处理一帧 JSON-RPC；通知不返回响应，协议错误与工具执行错误分开。"""
        from utils.fail import Failure                     # 只在统一入口翻译项目错误
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0" or not isinstance(message.get("method"), str):
            return self.error(None, -32600, "无效 JSON-RPC 请求")
        ident = message.get("id")                          # 通知没有 id
        notification = "id" not in message
        if not notification and (isinstance(ident, bool) or not isinstance(ident, (int, str))):
            return self.error(None, -32600, "请求 id 必须是字符串或整数")
        method = message["method"]
        params = message.get("params", {})                 # 无参数方法统一用空对象
        if not isinstance(params, dict):
            return None if notification else self.error(ident, -32602, "params 必须是对象")
        if notification:
            if method == "notifications/initialized" and self.negotiated:
                self.initialized = True                    # 只有握手完成后才接收就绪通知
            return None                                    # 其他通知同样不得回复
        if method == "initialize":
            if self.negotiated:
                return self.error(ident, -32600, "连接已经 initialize")
            if not isinstance(params.get("protocolVersion"), str) or not isinstance(params.get("capabilities"), dict) or not isinstance(params.get("clientInfo"), dict):
                return self.error(ident, -32602, "initialize 缺 protocolVersion/capabilities/clientInfo")
            self.negotiated = True                         # 客户端不支持返回版本时应自行断开
            result = {"protocolVersion": PROTOCOL_VERSION, "capabilities": {"tools": {}},
                      "serverInfo": {"name": "hoshiori", "version": CLIENT_VERSION}}
        elif method == "ping":
            result = {}                                    # ping 在生命周期各阶段都可用
        elif not self.initialized:
            return self.error(ident, -32002, "先 initialize 并发送 notifications/initialized")
        elif method == "tools/list":
            result = {"tools": self.tools()}               # 当前工具数固定，无需分页游标
        elif method == "tools/call":
            if not isinstance(params.get("name"), str) or not isinstance(params.get("arguments", {}), dict):
                return self.error(ident, -32602, "tools/call 需要 name 和对象 arguments")
            try:
                result = self.call(params["name"], params.get("arguments", {}))
            except (Failure, OSError, ValueError) as wrong:  # 外部文件与业务参数错误按 MCP 工具失败返回
                result = {"isError": True, "content": [{"type": "text", "text": str(wrong)}]}
        else:
            return self.error(ident, -32601, f"没有方法 {method}")
        return {"jsonrpc": "2.0", "id": ident, "result": result}

    @staticmethod
    def error(ident, code: int, message: str) -> dict:
        """构造一条标准 JSON-RPC 错误，保留对应请求编号。"""
        return {"jsonrpc": "2.0", "id": ident, "error": {"code": code, "message": message}}

    def serve(self, incoming=None, outgoing=None) -> None:
        """逐行处理 stdio，收到 EOF 自然结束；测试可传 StringIO 观察完整协议。"""
        import sys                                         # 默认协议流是当前进程的标准输入输出
        incoming = incoming or sys.stdin
        outgoing = outgoing or sys.stdout
        for line in incoming:                              # 一行一帧；不会把日志打印到协议 stdout
            try:
                response = self.handle(json.loads(line))   # JSON 解析只在协议边界做一次
            except json.JSONDecodeError:
                response = self.error(None, -32700, "帧不是合法 JSON")
            if response is not None:                       # 通知没有响应帧
                outgoing.write(json.dumps(response, ensure_ascii=False, separators=(",", ":")) + "\n")
                outgoing.flush()                           # 对端立刻能收到，不等待缓冲区装满
