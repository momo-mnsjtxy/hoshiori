"""命令 validate：验一份配方，过了就把哈希打出来。

调用示例：
  ori validate sft.warp.yaml
  ori validate sft.warp.yaml && ori spin sft.warp.yaml

配方其实是能跑起来的，这一步也能过；哈希是给配方留的锁，贴进 issue 或提交记录里。
"""

from pathlib import Path

from features import recipe


def command(args) -> int:
    """读、验、报哈希。"""
    path = Path(args.recipe)
    data = recipe.load(path)  # 展开 extends，循环会在这里报
    recipe.check(data)  # 字段名、类型、必备项
    print(f"✓ {path} 过了（哈希 {recipe.stamp(data)}）")
    return 0
