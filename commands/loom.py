"""命令 loom：起一个只读面板，看一个产物目录。

调用示例：
  ori loom
  ori loom bolt --port 8765
  ori loom out/bolt --host 0.0.0.0        # 想让别人看才放开

面板只读，路径钉在你给的那个产物目录里。检查只在这里做一次：目录有没有卡、端口在不在范围
里。页面本身全在 features/loom.py。这个命令前台运行，Ctrl+C 停。
"""

import os  # 令牌只从指定环境变量读取，不出现在进程参数里
from pathlib import Path

from features import loom
from utils.fail import fail


def command(args) -> int:
    """核对产物目录和端口，然后把面板交出去。"""
    out = Path(args.out)                            # 产物目录
    if not (out / "card.json").is_file():           # 没有卡说明这目录还没跑过 weave
        fail("input", f"{out} 里没有 card.json", hint=f"先跑 ori weave --out {out} --in <语料>")
    if not 0 <= args.port < 65536:                  # 端口越界，bind 会报得很难懂
        fail("input", f"端口越界: {args.port}", hint="常用 8000 到 9000，0 表示随便挑一个")
    token = os.environ.get(getattr(args, "token_env", "ORI_LOOM_TOKEN"), "")  # 只读用户明确指定的令牌变量
    loom.serve(out, args.host, args.port, token=token)
    return 0
