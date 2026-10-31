"""命令 spin：训练前把账摊开。数据在不在、显存放不放得下，先说清楚。

调用示例：
  ori spin sft.warp.yaml --dry-run    # 只出账，不跑
  ori spin sft.warp.yaml              # 出账，再查这台机器能不能跑

实跑要 torch 和显卡驱动，缺了就在这一步说，别等跑到一半。
"""

import shutil
from pathlib import Path

from features import plan, recipe
from utils.fail import fail


def command(args) -> int:
    """验配方、查数据、出账。"""
    data = recipe.load(Path(args.recipe))
    recipe.check(data)

    for one in (data.get("data") or {}).get("files") or []:  # 数据不在就别开跑
        if not Path(one).exists():
            fail("input", f"训练数据不在: {one}", hint="先用 ori weave 出 bolt/，或改 data.files")

    table = plan.build(data)
    print(table.show())
    print(f"\n配方哈希 {recipe.stamp(data)}")
    if args.dry_run:  # 只出账就停
        return 0

    import importlib.util

    if importlib.util.find_spec("torch") is None:  # 真跑要 torch
        fail("missing", "实际开跑要 torch", hint="pip install torch")
    if table.needs_gpu and shutil.which("nvidia-smi") is None:  # 这套精度要卡
        fail("missing", "这套精度要 NVIDIA 卡", hint="换 precision: fp32，或装驱动")
    print("\n账没问题，可以接训练后端")
    return 0
