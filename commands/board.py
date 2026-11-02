"""命令 board：收齐一个目录里的公开榜提交，发布成静态 HTML 与 JSON。

调用示例：
  ori board collect board-inbox --out site/board

collect 只读 ``board-*.json``。目录不存在或一份提交都没有时在开头按输入错误停下；
字段检查、排序、转义和落盘都交给 features/board.py，命令层只编排这一条流程。
"""

from pathlib import Path

from features import board
from utils.fail import fail


def command(args) -> int:
    """检查收件目录，再发布 ``index.html`` 与 ``board.json``。"""
    folder = Path(args.folder)                                               # 收到提交的目录
    if not folder.is_dir():                                                  # 路径写错时不要悄悄发布空榜
        fail("input", f"公开榜目录不在: {folder}", hint="给一个装着 board-*.json 的目录")
    if not any(path.is_file() for path in folder.glob("board-*.json")):       # 同名目录不能冒充一份提交文件
        fail("input", f"{folder} 里没有 board-*.json", hint="先放入至少一份公开榜提交")

    paths = board.publish(folder, Path(args.out))                            # 收、排、渲染、落盘一次做完
    print(f"公开榜 {paths['html']} · {paths['json']}")                       # 把两份产物位置交回用户
    return 0                                                                # 两份都写成才算成功
