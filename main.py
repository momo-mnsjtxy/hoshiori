"""ori 的入口：认命令、把参数交给对应的命令文件、统一接住并翻译错误。

调用示例：
  ori weave --in data/a.jsonl --out bolt
  ori validate sft.warp.yaml
  python main.py --help          # 没装包时直接跑

入口只干三件事：拆参数、找命令、接错误。业务判断都在 commands/ 里。
整个项目只有这里接错误，接住后按 Failure.kind 决定退出码：
填错 2、配方不合规 3、缺东西 4、程序故障 1。用户能自己改的那些会带着原因返回。
"""

import argparse
import sys
from pathlib import Path

from utils.fail import Failure, fail

VERSION = (Path(__file__).parent / "VERSION").read_text().strip()  # 版本只有 VERSION 文件一处

# 命令名 → 命令文件。加一个命令就在这里加一行，文件自己写在 commands/ 里。
COMMANDS = {
    "weave": "commands.weave",
    "import": "commands.import_lmlab",
    "validate": "commands.validate",
    "vocab": "commands.vocab",
    "spin": "commands.spin",
    "judge": "commands.judge",
    "forge": "commands.forge",
    "serve": "commands.serve",
    "loom": "commands.loom",
    "doctor": "commands.doctor",
    "plugin": "commands.plugin",  # 单文件插件的清单与运行入口
    "mcp": "commands.mcp",        # stdio MCP 服务器的工具清单与单次调用
}


def parser() -> argparse.ArgumentParser:
    """拼出全部参数。每个命令自己的参数在它自己的段落里。"""
    top = argparse.ArgumentParser(prog="ori", description="星织 Hoshiori — 线在你手里。")
    top.add_argument("--version", action="version", version=f"ori {VERSION}")
    below = top.add_subparsers(dest="cmd", metavar="<命令>")

    weave = below.add_parser("weave", help="数据：收、洗、去重、打指纹、出卡")
    weave.add_argument("--in", dest="inputs", action="append", required=True, help="输入，可给多次；- 走标准输入")
    weave.add_argument("--out", default="bolt", help="产物目录，默认 bolt")
    weave.add_argument("--store", default=None, help="去重索引落盘到 <目录>/index，百万级语料用；默认只在内存")
    weave.add_argument("--name", default="weave", help="数据集名")
    weave.add_argument("--license", default="unknown", help="来源许可")
    weave.add_argument("--fields", default="text,content,raw,body", help="正文字段优先序，逗号分")
    weave.add_argument("--min-chars", type=int, default=1, help="短于这个长度算空样本")
    weave.add_argument("--max-chars", type=int, default=0, help="长于这个长度拒收，0 不限")
    weave.add_argument("--min-score", type=float, default=0.0, help="质量分低于此线拒收，默认 0 不筛")
    privacy = weave.add_mutually_exclusive_group()
    privacy.add_argument("--scrub", dest="scrub", action="store_true", default=True, help="替换敏感信息（默认）")
    privacy.add_argument("--no-scrub", dest="scrub", action="store_false", help="保留正文里的敏感信息")
    weave.add_argument("--fake", dest="fake_files", action="append", default=[], help="合成样本来源，可给多次")
    weave.add_argument("--fake-ratio", type=float, default=0.1, help="混合结果里的合成样本占比，默认 0.1")
    weave.add_argument("--size", type=int, default=5, help="比对窗口几个字")
    weave.add_argument("--threshold", type=float, default=0.8, help="重合度到这条线算重复")
    weave.add_argument("--count", type=int, default=128, help="指纹长度")
    weave.add_argument("--no-exact", action="store_true", help="不回算真实重合度，快但会误杀")
    weave.add_argument("--keep-duplicates", action="store_true", help="判掉的也留进正文档")
    weave.add_argument("--algo", default="sha256", choices=["sha256", "blake3"], help="指纹算法")
    weave.add_argument("--dry-run", action="store_true", help="照常跑，卡片里标一笔")

    into = below.add_parser("import", help="把别家的配置迁过来")
    kinds = into.add_subparsers(dest="kind", required=True)
    old = kinds.add_parser("lmlab", help="lmlab preset yaml → 配方 + 差异清单")
    old.add_argument("files", nargs="+", help="一个或多个 preset")
    old.add_argument("--out", default="imported", help="输出目录")
    old.add_argument("--force", action="store_true", help="覆盖已有文件")

    vocab = below.add_parser("vocab", help="从语料学一个词表")
    vocab.add_argument("--in", dest="inputs", action="append", required=True, help="语料，可给多次")
    vocab.add_argument("--out", required=True, help="词表写到哪")
    vocab.add_argument("--size", type=int, default=8000, help="词表大小，至少 259")
    vocab.add_argument("--limit", type=int, default=2000, help="每种来源最多读几条")

    check = below.add_parser("validate", help="验一份配方")
    check.add_argument("recipe")

    spin = below.add_parser("spin", help="训练：出账后真跑")
    spin.add_argument("recipe")
    spin.add_argument("--dry-run", action="store_true", help="出完账就停")
    spin.add_argument("--steps", type=int, default=0, help="跑多少步，给了就覆盖 epochs 换算；带 --resume 时是从断点再跑这么多步")
    spin.add_argument("--save-every", type=int, help="每多少步存一档，0 只存收尾")
    spin.add_argument("--resume", help="从哪一档接着跑；给目录，auto 取产物目录里最新")
    spin.add_argument("--out", help="产物目录，默认 out/run-<时间戳>")
    spin.add_argument("--launch", default="plain", choices=["plain", "torchrun"], help="怎么起进程：plain 本进程跑；torchrun 按配方的 devices 起多卡")

    judge = below.add_parser("judge", help="评测")
    judge.add_argument("recipe", nargs="?")
    judge.add_argument("--suite", action="append", help="评测套件，可给多次")
    judge.add_argument("--answer-cmd", help="拿答案的外部命令，{prompt} 换成题面；不给就回显")
    judge.add_argument("--arena", help="另一个答题命令；与 --answer-cmd 乱序对打并算 Elo / BT")
    judge.add_argument("--baseline", help="基线 report.json，给了就走回归门")
    judge.add_argument("--min-delta", type=float, help="允许掉多少总分，缺省读配方 eval.gate")
    judge.add_argument("--limit", type=int, default=0, help="每套最多跑几题，0 不限")
    judge.add_argument("--out", default="out/judge", help="报告目录，默认 out/judge")
    judge.add_argument("--submit", help="跑完把公开字段 POST 到这个榜单 URL；断网就留待补文件")
    judge.add_argument("--import-lm-eval", action="append", help="导入 lm-eval 任务成套件，给了就只导入")

    forge = below.add_parser("forge", help="导出")
    forge.add_argument("recipe")
    forge.add_argument("--out", required=True, help="产物文件或目录，必填")
    forge.add_argument("--format", action="append", dest="formats", help="导出格式，可给多次")
    forge.add_argument("--quant", action="append", help="量化档，可给多次：f16 / q8_0 / q6_k / q5_k_s / q5_k_m")
    forge.add_argument("--self-check", action="store_true", help="用 llama-cli 跑量化前后基准，写退化报告")
    forge.add_argument("--suites", nargs="+", help="自评套件，空格或逗号分；默认 zh-lite,format-lite")
    forge.add_argument("--self-check-threshold", type=float, help="允许的总分退化；默认读 eval.gate.min_delta，再默认 0.05")
    forge.add_argument("--no-tokenizer", action="store_true", help="不导词表，产物就不能独立跑")
    forge.add_argument("--verbose", action="store_true", help="把要执行的命令、参数、目录布局都打出来")
    forge.add_argument("--mlx-print-only", action="store_true", help="mlx 只打印命令和目录布局，不执行")

    serve = below.add_parser("serve", help="起一个 OpenAI 兼容的服务")
    serve.add_argument("model")
    serve.add_argument("--port", type=int, default=8080)
    serve.add_argument("--backend", default="cmd", help="回声 echo / 外部命令 cmd")
    serve.add_argument("--answer-cmd", default="", help="后端命令，{prompt} 换成最后一条消息")
    serve.add_argument("--limit", type=int, default=60, help="每 IP 每分钟几条，0 不限")
    serve.add_argument("--log", default="gateway.jsonl", help="请求日志路径，一行一条 JSON")
    serve.add_argument("--host", default="127.0.0.1", help="监听地址，默认只本机")
    serve.add_argument("--token", default="", help="要求 Authorization: Bearer <token>，不给就不鉴权")

    loom = below.add_parser("loom", help="打开面板")
    loom.add_argument("out", nargs="?", default="bolt", help="产物目录，默认 bolt")
    loom.add_argument("--port", type=int, default=8765, help="监听端口，0 表示随便挑一个空口")
    loom.add_argument("--host", default="127.0.0.1", help="监听地址，默认只本机")
    plugins = below.add_parser("plugin", help="扫描并运行单文件插件")                                  # 插件命令入口
    plugin_actions = plugins.add_subparsers(dest="action", required=True)                             # list 与 run 两个动作
    plugin_actions.add_parser("list", help="列出四类插件")                                             # 只展示目录现状
    plugin_run = plugin_actions.add_parser("run", help="运行一个插件")                                 # 点名运行一份插件
    plugin_run.add_argument("kind", metavar="类", help="sources / rewards / metrics / exporters")    # 四类目录中的一类
    plugin_run.add_argument("name", metavar="名", help="插件的 NAME")                                 # 文件声明的插件名
    plugin_run.add_argument("--args", default="{}", help="传给 run 的 JSON 对象，默认 {}")             # 关键字参数对象
    mcp = below.add_parser("mcp", help="连接 stdio MCP 服务器")                                         # MCP 命令入口
    mcp_actions = mcp.add_subparsers(dest="action", required=True)                                      # list、tools、call 三个动作
    mcp_actions.add_parser("list", help="列出 mcp.yaml 里的服务器")                                    # 只读配置，不起进程
    mcp_tools = mcp_actions.add_parser("tools", help="列出一台服务器的工具")                            # 完成握手后展示 tools/list
    mcp_tools.add_argument("name", metavar="名", help="mcp.yaml 里的服务器名")                         # 点名一台服务器
    mcp_call = mcp_actions.add_parser("call", help="调用一台服务器的一个工具")                          # 一次 tools/call 后退出
    mcp_call.add_argument("name", metavar="名", help="mcp.yaml 里的服务器名")                          # 点名一台服务器
    mcp_call.add_argument("tool", metavar="工具", help="tools 列出的工具名")                           # 点名一个工具
    mcp_call.add_argument("--args", default="{}", help="工具参数 JSON 对象，默认 {}")                  # MCP arguments 字段
    below.add_parser("doctor", help="看看这台机器缺什么")
    return top


def run(argv: list = None) -> int:
    """跑一条命令，返回退出码。所有 Failure 在这里收口。"""
    args = parser().parse_args(argv)
    if not args.cmd:  # 什么也没给就列帮助
        parser().print_help()
        return 0
    if args.cmd == "import" and not args.kind:  # import 还要再点一层来源
        fail("input", "import 后面要说迁哪家的", hint="ori import lmlab <文件>")
    module = __import__(COMMANDS[args.cmd], fromlist=["command"])  # 用到才导，启动不被依赖拖慢
    try:
        return module.command(args)
    except Failure as bad:  # 已知失败，按 kind 给退出码
        print(bad.say(), file=sys.stderr)
        return bad.exit_code
    except KeyboardInterrupt:  # 手动中断不算故障
        print("已中断", file=sys.stderr)
        return 130


def main() -> int:
    """给 pyproject 的脚本入口和 python -m 用。"""
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
