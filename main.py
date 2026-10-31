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
    "spin": "commands.spin",
    "judge": "commands.judge",
    "forge": "commands.forge",
    "serve": "commands.serve",
    "loom": "commands.loom",
    "doctor": "commands.doctor",
}


def parser() -> argparse.ArgumentParser:
    """拼出全部参数。每个命令自己的参数在它自己的段落里。"""
    top = argparse.ArgumentParser(prog="ori", description="星织 Hoshiori — 线在你手里。")
    top.add_argument("--version", action="version", version=f"ori {VERSION}")
    below = top.add_subparsers(dest="cmd", metavar="<命令>")

    weave = below.add_parser("weave", help="数据：收、洗、去重、打指纹、出卡")
    weave.add_argument("--in", dest="inputs", action="append", required=True, help="输入，可给多次；- 走标准输入")
    weave.add_argument("--out", default="bolt", help="产物目录，默认 bolt")
    weave.add_argument("--name", default="weave", help="数据集名")
    weave.add_argument("--license", default="unknown", help="来源许可")
    weave.add_argument("--fields", default="text,content,raw,body", help="正文字段优先序，逗号分")
    weave.add_argument("--min-chars", type=int, default=1, help="短于这个长度算空样本")
    weave.add_argument("--max-chars", type=int, default=0, help="长于这个长度拒收，0 不限")
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

    check = below.add_parser("validate", help="验一份配方")
    check.add_argument("recipe")

    spin = below.add_parser("spin", help="训练：开跑前先出账")
    spin.add_argument("recipe")
    spin.add_argument("--dry-run", action="store_true", help="出完账就停")

    judge = below.add_parser("judge", help="评测")
    judge.add_argument("recipe", nargs="?")
    judge.add_argument("--suite", action="append", help="评测套件，可给多次")
    judge.add_argument("--answer-cmd", help="拿答案的外部命令，{prompt} 换成题面；不给就回显")
    judge.add_argument("--baseline", help="基线 report.json，给了就走回归门")
    judge.add_argument("--min-delta", type=float, help="允许掉多少总分，缺省读配方 eval.gate")
    judge.add_argument("--limit", type=int, default=0, help="每套最多跑几题，0 不限")
    judge.add_argument("--out", default="out/judge", help="报告目录，默认 out/judge")

    forge = below.add_parser("forge", help="导出")
    forge.add_argument("recipe")
    forge.add_argument("--out", required=True, help="产物文件或目录，必填")
    forge.add_argument("--format", action="append", dest="formats", help="导出格式，可给多次")
    forge.add_argument("--quant", action="append", help="量化档，可给多次：f16 / q8_0 / q5_k_m")

    serve = below.add_parser("serve", help="起一个 OpenAI 兼容的服务")
    serve.add_argument("model")
    serve.add_argument("--port", type=int, default=8080)
    serve.add_argument("--backend", default="cmd", help="回声 echo / 外部命令 cmd")
    serve.add_argument("--answer-cmd", default="", help="后端命令，{prompt} 换成最后一条消息")
    serve.add_argument("--limit", type=int, default=60, help="每 IP 每分钟几条，0 不限")
    serve.add_argument("--log", default="gateway.jsonl", help="请求日志路径，一行一条 JSON")
    serve.add_argument("--host", default="127.0.0.1", help="监听地址，默认只本机")

    loom = below.add_parser("loom", help="打开面板")
    loom.add_argument("out", nargs="?", default="bolt", help="产物目录，默认 bolt")
    loom.add_argument("--port", type=int, default=8765, help="监听端口，0 表示随便挑一个空口")
    loom.add_argument("--host", default="127.0.0.1", help="监听地址，默认只本机")
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
