"""命令 serve：把一个模型起成 OpenAI 兼容的网关。

调用示例：
  ori serve out/merged --port 8080
  ori serve qwen2.5 --backend echo --port 8080
  ori serve out/merged --answer-cmd "ollama run {model} {prompt}" --limit 30 --log run.jsonl
  ori serve out/merged --token abc123

给了 --token 就要求每条请求带 Authorization: Bearer <token>，不对或缺都回 401；不给就谁都能打。

检查只在这里做一次：模型在不在、端口范围、后端取值、cmd 后端有没有命令、日志目录在不在、token 空不空。
起服务、鉴权、限流、落日志都在 features.gateway，这个文件只把参数搬过去。
"""

from pathlib import Path

from features.gateway import Gateway
from utils.fail import fail


def command(args) -> int:
    """核对参数后起网关，一直跑到 Ctrl-C。"""
    token = (args.token or "").strip()                 # 首尾空格看不见，先去掉再比
    if args.token and not token:                       # 给了 --token 却只有空格
        fail("input", "--token 是空的", hint="换一个没有空格的令牌，或者干脆不给")
    given = Path(args.model).expanduser()              # 参数可能是目录，也可能只是个模型名
    if "/" in args.model and not given.is_dir():       # 带斜杠的当路径，目录必须在
        fail("input", f"模型目录不在: {args.model}", hint="先训练或合并出一个目录，或者只给模型名")
    if not 0 < args.port < 65536:                      # 端口范围
        fail("input", f"端口越界: {args.port}", hint="给 1024 到 65535 之间的数")
    if args.backend not in ("echo", "cmd"):            # 后端只有这两种
        fail("input", f"不认识的 backend: {args.backend}", hint="echo 原样回声，cmd 走外部命令")
    if args.backend == "cmd" and not args.answer_cmd:  # cmd 后端没命令就没人回答
        fail("input", "cmd 后端要给 --answer-cmd", hint="例: --answer-cmd 'ollama run {model} {prompt}'")
    logs_dir = Path(args.log).expanduser().parent      # 日志要写进哪个目录
    if args.log and not logs_dir.is_dir():             # 目录不在就先说一声，别等第一行才炸
        fail("input", f"日志目录不在: {logs_dir}", hint="先建好目录，或者换个路径")

    gate = Gateway(                                    # 参数原样搬给网关
        str(given) if given.is_dir() else args.model,  # 目录在就用绝对路径，否则当模型名
        backend=args.backend,                          # 哪个后端
        answer_cmd=args.answer_cmd,                    # 命令模板
        token=token,                                   # 要求的令牌，空就不鉴权
        limit=args.limit,                              # 每分钟每 IP 的条数
        log=args.log,                                  # 日志路径
        host=args.host,                                # 绑哪个地址
    )
    gate.run(args.port)                                # 一直跑到中断
    return 0
