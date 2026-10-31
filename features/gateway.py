"""网关：把 answer() 挂到 OpenAI 兼容的接口上，管起服务、限流、记日志。

调用示例：
  from features.gateway import Gateway
  Gateway("out/merged", answer_cmd="ollama run {model} {prompt}", limit=30).run(8080)
  Gateway("qwen2.5", backend="echo").run(8081)  # 不回后端，原样回声

接口只有两条，形状照 OpenAI，前端换一家模型不用改代码：
  POST /v1/chat/completions  收 model、messages、temperature、max_tokens、stream，回 choices
  GET  /v1/models            报出身上挂的这一个模型

后端抽成一个纯函数 answer(model, messages, options)：默认拿外部命令，把模板里的 {prompt}
换成最后一条消息。换 vllm / llama.cpp 只换这一个函数，接口和字段都不动。

网关自己真干的事只有四件：设了 token 就验 Authorization；按 IP+自然分钟限流，超了回 429；
每个请求一行 JSON 落日志；一个请求一个线程。计数和日志两个线程都会改，所以都要过锁。

Gateway 的参数：
  model       模型目录或模型名，填进模板的 {model}
  backend     echo 原样回声；cmd 走外部命令
  answer_cmd  命令模板，{prompt} 换成最后一条消息，{model} 换成模型
  token       要求的令牌，None 就不鉴权；给了就必须带 Authorization: Bearer <token>
              令牌自己可以带空格，方案名后面整段都算令牌，首尾空白不算
  limit       每个 IP 每分钟几条，0 不限
  log         日志路径，None 不写

鉴权在 do_GET 和 do_POST 一开头各拦一次，都走同一个 check_token()，路由分支里不再重复。
usage 有分词器就真数，没有就按字数估，响应头 x-usage-source 标出用的是哪种。

流式默认不给 usage，最后一块是空 delta 加 finish_reason。
请求体带 stream_options: {"include_usage": true} 时，在 [DONE] 前多补一块 choices 为空的块，
带上完整 usage：不少客户端靠这一块记用量，缺了它上游就算不出账。

run(port) 一直占着直到 Ctrl-C；start(port, in_thread=True) 只把服务起好、把 server 交回来，测试用。
"""

import hmac
import json
import subprocess
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from utils.fail import fail

PIECE = 8  # 流式一块吐几个字。太小 HTTP 头开销大，太大不像逐字


# --- 后端 ---

def answer(model: str, messages: list, options: dict) -> str:
    """默认后端：把最后一条消息填进外部命令，跑它，它的输出就是回复。"""
    prompt = messages[-1]["content"]                                            # 只把最后一条当问题，历史上下文由模板自己接
    command = options["answer_cmd"]                                             # 命令模板由命令层给，这里不兜底
    command = command.replace("{prompt}", prompt)                               # 填问题占位
    command = command.replace("{model}", model)                                 # 填模型占位
    done = subprocess.run(command, shell=True, capture_output=True, text=True)  # 起外部命令，收输出
    if done.returncode:                                                         # 外部命令自己的故障要连它的 stderr 一起报
        raise RuntimeError(f"后端命令失败({done.returncode}): {done.stderr.strip() or '没给输出'}")
    return done.stdout.strip()                                                  # 命令的标准输出就是回复正文


def echo(model: str, messages: list, options: dict) -> str:
    """回声后端：把最后一条消息原样回过去，不碰外部命令。签名跟 answer 一样。"""
    return messages[-1]["content"]  # 收到什么回什么，用来验接口通不通


def chunks(text: str):
    """把正文按 PIECE 个字切开，逐块吐。"""
    for start in range(0, len(text), PIECE):  # 从 0 开始，每 PIECE 个字一段
        yield text[start : start + PIECE]     # 抛出这一小段


def new_id() -> str:
    """回执 id，照 OpenAI 的 chatcmpl- 开头。"""
    return "chatcmpl-" + uuid.uuid4().hex[:24]  # 取 24 位十六进制，够唯一


def piece(base: dict, delta: dict, finish) -> bytes:
    """拼一块 SSE。base 是每块共用的 id、模型、时间。"""
    choice = {"index": 0, "delta": delta, "finish_reason": finish}               # 这一块的增量
    chunk = {**base, "choices": [choice]}                                        # 拼成完整的一块
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode("utf-8")  # 一行 data: 加空行收尾


def usage_piece(base: dict, usage: dict) -> bytes:
    """拼带用量的那一块。choices 是空数组，用量单独占一块，照 OpenAI 的 include_usage。"""
    chunk = {**base, "choices": [], "usage": usage}                               # 空 choices 加用量
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode("utf-8")    # 一行 data: 加空行收尾


def estimate(messages: list, text: str) -> dict:
    """按字数粗估用量。真分词走不通时用它，够回执形状用。"""
    asked = sum(len(str(m.get("content", ""))) for m in messages)  # 所有消息加起来几个字
    said = len(text)                                               # 回复几个字
    return {"prompt_tokens": asked, "completion_tokens": said, "total_tokens": asked + said}


def load_tokenizer(model: str):
    """模型目录里有 tokenizer.json 就加载它，没有、或者缺 tokenizers 库就回 None。

    None 表示退回字数估算，不是故障：tokenizers 是可选依赖，装上才用得上真分词。
    只在这里加载一次，之后整个服务生命周期共用，别每个请求重建（重建一次几百毫秒）。
    """
    path = Path(model) / "tokenizer.json"  # 分词器跟着模型目录走，只认这一处
    if not path.is_file():                 # 只给了模型名的，目录里没有分词器
        return None
    try:                                   # 缺库就回 None，让上层走估算，这里不抛
        from tokenizers import Tokenizer
    except ImportError:
        return None
    return Tokenizer.from_file(str(path))  # 读一次，交给网关拿着


# --- 收请求 ---

class Handler(BaseHTTPRequestHandler):
    """一条请求一个线程。只认两条路由，别的回 404。"""

    def do_GET(self) -> None:
        if not self.check_token():                                      # 先验身份，401 不占限流配额
            return
        gate = self.server.gateway                                      # 这个服务带的网关
        if not gate.allow(self.client_address[0]):                      # 过了身份再看限流
            return self.reject(429, "rate_limit_error", "这个 IP 这一分钟用完了")
        if self.route() != "/v1/models":                                # 只有模型列表这一条
            return self.reject(404, "invalid_request_error", "没有这个路径")
        self.send_json(200, {"object": "list", "data": [gate.card()]})  # 身上只挂一个模型，照列表形状回

    def do_POST(self) -> None:
        if not self.check_token():                                                   # 先验身份，401 不占限流配额
            return
        gate = self.server.gateway                                                   # 这个服务带的网关
        if not gate.allow(self.client_address[0]):                                   # 过了身份再看限流
            return self.reject(429, "rate_limit_error", "这个 IP 这一分钟用完了")
        if self.route() != "/v1/chat/completions":                                   # 只有聊天这一条
            return self.reject(404, "invalid_request_error", "没有这个路径")
        try:                                                                         # 请求体是外面的东西，解析放在这道边界上
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))  # 按长度读完再解
        except (ValueError, KeyError):                                               # 不是合法 JSON，或者压根没给长度
            return self.reject(400, "invalid_request_error", "请求体不是合法 JSON")
        messages = body.get("messages") or []                                        # OpenAI 的字段，原样收
        if not messages:                                                             # 一条消息都没有就无从回答
            return self.reject(400, "invalid_request_error", "messages 是空的")
        if "content" not in messages[-1]:                                            # 后端只读最后一条，它必须带正文
            return self.reject(400, "invalid_request_error", "最后一条消息没有 content")
        options = {                                                                  # 后端要的选项
            "temperature": body.get("temperature"),                                  # 温度原样往下传
            "max_tokens": body.get("max_tokens"),                                    # 上限原样往下传
            "answer_cmd": gate.answer_cmd,                                           # 命令模板由网关带着
        }
        try:                                                                         # 后端是外部程序，可能失败
            text = gate.reply(body.get("model") or gate.name, messages, options)     # 后端给的整段回复
        except RuntimeError as bad:                                                  # 后端自己报的错，原样告诉调用方
            return self.reject(500, "server_error", str(bad))
        if body.get("stream"):                                                       # 要流就一块块吐
            want_usage = bool((body.get("stream_options") or {}).get("include_usage"))  # 要不要那块用量
            return self.send_stream(gate, messages, text, want_usage)
        self.send_json(200, gate.completion(messages, text))                         # 不要流就整段回

    def check_token(self) -> bool:
        """比对 Authorization 头。网关没设 token 就一律放行；不对就回 401 并返回 False。"""
        token = self.server.gateway.token              # 网关要求的令牌，None 表示不鉴权
        if token is None:                              # 没开鉴权，默认行为不变
            return True
        given = self.headers.get("Authorization", "")  # 形如 "Bearer abc"
        scheme, _, value = given.partition(" ")        # 只按第一个空格把方案名切下来
        value = value.strip()                          # 剩下的整段都是令牌：令牌自己带空格也不会被切断，首尾空白不算
        if scheme.lower() != "bearer":                 # 方案大小写不敏感：Bearer、bearer 都认
            self.reject(401, "invalid_request_error", "要给 Authorization: Bearer <token>")
            return False
        if not hmac.compare_digest(value.encode(), token.encode()):  # 定长比，不按位提前返回
            self.reject(401, "invalid_request_error", "token 不对")
            return False
        return True

    def route(self) -> str:
        """去掉查询串的路径。"""
        return self.path.split("?")[0]  # 只比对路径本身

    def send_json(self, status: int, body: dict) -> None:
        """回一段 JSON。"""
        raw = json.dumps(body, ensure_ascii=False).encode("utf-8")  # 中文原样，不转义成 \u
        self.send_response(status)                                  # 状态行
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))           # 长度先说完
        self.send_header("x-usage-source", self.server.gateway.usage_source)  # 用量是真数还是估的
        self.end_headers()
        self.wfile.write(raw)                                       # 正文
        self.record(status)                                         # 记一行日志

    def reject(self, status: int, kind: str, message: str) -> None:
        """按 OpenAI 的错误形状回一个失败。"""
        self.send_json(status, {"error": {"message": message, "type": kind, "code": status}})

    def send_stream(self, gate, messages: list, text: str, with_usage: bool = False) -> None:
        """把回复拆成 SSE 块逐条吐，末尾给 [DONE]。要用量就在 [DONE] 前补一块空的 choices 带 usage。"""
        base = gate.chunk_base()                                              # 每块共用的 id、模型、时间
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")  # 逐块吐字用事件流
        self.send_header("Cache-Control", "no-cache")                         # 中间别缓存
        self.send_header("x-usage-source", gate.usage_source)                 # 用量是真数还是估的
        self.end_headers()
        self.wfile.write(piece(base, {"role": "assistant"}, None))            # 头一块只说角色，照 OpenAI 的规矩
        for part in chunks(text):                                             # 正文一块块来
            self.wfile.write(piece(base, {"content": part}, None))
        self.wfile.write(piece(base, {}, "stop"))                             # 倒数第二块带 stop
        if with_usage:                                                        # 要了用量才补这一块，不要就保持原样
            self.wfile.write(usage_piece(base, gate.usage(messages, text)))    # 和整段回执同一套算法，不另算一份
        self.wfile.write(b"data: [DONE]\n\n")                                 # 最后一块是结束标记
        self.wfile.flush()
        self.record(200)                                                      # 记一行日志

    def record(self, status: int) -> None:
        """写一行请求日志。"""
        line = {"t": round(time.time(), 3), "ip": self.client_address[0]}  # 时间与人
        line["method"] = self.command                                      # 方法
        line["path"] = self.path                                           # 路径
        line["status"] = status                                            # 结果码
        self.server.gateway.write_log(line)                                # 交给网关落盘

    def log_message(self, form: str, *parts) -> None:
        """关掉 http.server 自己往 stderr 打的访问日志，日志已经进文件了。"""
        pass  # 不做事，只为覆盖父类


# --- 网关 ---

class Gateway:
    """一个网关。模型、后端、限流计数、日志都挂在这个实例上。"""

    def __init__(self, model: str, backend: str = "cmd", answer_cmd: str = "", token: str = None, limit: int = 60, log=None, host: str = "127.0.0.1") -> None:
        self.model = model                                  # 模型目录或名字，填进 {model}
        self.name = Path(model).name or model               # 回执里的模型 id，取末段
        self.backend = backend                              # echo 或 cmd
        self.answer_cmd = answer_cmd                        # cmd 后端的命令模板
        self.reply = echo if backend == "echo" else answer  # 后端定一次，别每个请求再判断
        self.token = token or None                          # 要求的令牌，None 就不鉴权
        self.tokenizer = load_tokenizer(model)              # 分词器只在构造里加载一次，没有就 None
        self.usage_source = "tokenizer" if self.tokenizer else "estimate"  # 用量从哪来，进响应头
        self.limit = limit                                  # 每 IP 每分钟几条，0 不限
        self.host = host                                    # 绑哪个地址，默认只本机
        self.hits = {}                                      # IP → (哪一分钟, 几次)
        self.lock = threading.Lock()                        # 计数和日志写都要过这把锁
        self.log_path = Path(log) if log else None          # 日志路径，None 就不写
        self.sink = None                                    # 日志手柄，开不开看下面
        if log:                                             # 要写才开，手柄一直开着免得每行开一次
            self.sink = self.log_path.open("a", encoding="utf-8")

    def allow(self, ip: str) -> bool:
        """这个 IP 这一分钟还准不准再要一次。"""
        if not self.limit:                                                          # 0 表示不限，直接准
            return True
        minute = int(time.time() // 60)                                             # 按自然分钟切桶
        with self.lock:                                                             # 计数会被好几个请求线程一起改
            last, count = self.hits.get(ip, (minute, 0))                            # 这个 IP 上次记在哪一分钟、几次
            if last != minute:                                                      # 换分钟了就从 0 重数
                count = 0
            self.hits[ip] = (minute, count + 1)                                     # 先记上这一次
            gone = [key for key, value in self.hits.items() if value[0] != minute]  # 上一分钟留下的 IP
            for key in gone:                                                        # 顺手清掉，不然字典只长不缩
                del self.hits[key]
        return count < self.limit                                                   # 这次是第 count+1 次，没到线才准

    def write_log(self, record: dict) -> None:
        """追加一行 JSON 日志。没给路径就不写。"""
        if self.sink is None:                                               # 不写日志的模式
            return
        with self.lock:                                                     # 多线程同时写会串行
            self.sink.write(json.dumps(record, ensure_ascii=False) + "\n")  # 一行一条
            self.sink.flush()                                               # 每行都刷，进程被 kill 也留得下

    def close(self) -> None:
        """关日志手柄。可以重复调。"""
        if self.sink is not None:  # 没开过就不用关
            self.sink.close()
            self.sink = None

    def card(self) -> dict:
        """模型列表里的一项。"""
        return {"id": self.name, "object": "model", "created": int(time.time()), "owned_by": "hoshiori"}

    def usage(self, messages: list, text: str) -> dict:
        """算用量。接了分词器就真数，没有就退回字数估算。"""
        if self.tokenizer is None:                                                    # 没分词器，粗估
            return estimate(messages, text)
        asked = sum(len(self.tokenizer.encode(str(m.get("content", ""))).ids) for m in messages)  # 每条消息各数一遍
        said = len(self.tokenizer.encode(text).ids)                                   # 回复也真数一遍
        return {"prompt_tokens": asked, "completion_tokens": said, "total_tokens": asked + said}

    def completion(self, messages: list, text: str) -> dict:
        """非流式的整段回执，字段照 OpenAI。"""
        answer = {"role": "assistant", "content": text}  # 助手这一条
        return {
            "id": new_id(),                              # 回执 id
            "object": "chat.completion",                 # 类型
            "created": int(time.time()),                 # 生成时间
            "model": self.name,                          # 落到哪个模型
            "choices": [{"index": 0, "message": answer, "finish_reason": "stop"}],
            "usage": self.usage(messages, text),         # 用量，真数或估的，看响应头
        }

    def chunk_base(self) -> dict:
        """流式每块共用的那部分。"""
        base = {"id": new_id(), "object": "chat.completion.chunk"}  # 回执 id 与类型
        base["created"] = int(time.time())                          # 生成时间
        base["model"] = self.name                                   # 哪个模型回的
        return base

    def start(self, port: int = 0, in_thread: bool = False):
        """建服务，返回 server。port 给 0 由系统挑一个空端口。"""
        server = ThreadingHTTPServer((self.host, port), Handler)  # 每个请求起一个线程
        server.gateway = self                                     # 处理器从 server 上取网关
        if in_thread:                                             # 测试要拿回控制权
            threading.Thread(target=server.serve_forever, daemon=True).start()
        return server

    def run(self, port: int) -> None:
        """起服务并一直跑。端口被占说清原因并给一个能用的下一个端口，Ctrl-C 收摊。"""
        try:                       # 占没占端口在这一步才知道
            server = self.start(port)
        except OSError as busy:    # 端口被别人占了
            next_port = port + 1 if port < 65535 else 1024                  # 顺手给一个还合法的下一个口
            fail("missing", f"端口 {port} 起不来: {busy.strerror}", hint=f"换一个端口，比如 --port {next_port}")
        print(f"网关已起: http://127.0.0.1:{port}/v1  (POST /v1/chat/completions · GET /v1/models)")
        print(f"模型 {self.name} · 后端 {self.backend} · 每分钟 {self.limit or '不限'} 条")
        print(f"日志 {self.log_path or '不写'}")
        try:                       # try 是为了 Ctrl-C 也能收干净
            server.serve_forever()
        except KeyboardInterrupt:  # 手动中断
            print("\n收到 Ctrl-C，收摊")
        finally:
            server.server_close()  # 放开端口
            self.close()           # 关日志
