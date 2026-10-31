"""命令 import lmlab：把他们的 preset 转成配方，同时留一份差异清单。

调用示例：
  ori import lmlab ~/LMLab/configs/sft_lora.yaml --out imported
  ori import lmlab configs/*.yaml --out imported --force

每份 preset 出两个文件：转好的 .warp.yaml 和逐条差异 .diff.md。
缺基座路径的那份返回 1，提醒补完再跑。
"""

from pathlib import Path

from features import migration
from utils.fail import fail


def command(args) -> int:
    """逐个文件转。"""
    if not args.files:  # 少了文件
        fail("input", "要给至少一个 preset 文件", hint="例: ori import lmlab configs/sft_lora.yaml")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)  # 输出目录先备好

    blocked = []
    total = 0
    for spec in args.files:
        source = Path(spec)
        preset = migration.read_preset(source)  # 读，坏 yaml 在这里就停
        job = migration.convert(preset, where=str(source))  # 转，unknown 一项都不丢
        name = source.stem
        recipe_path = out / f"{name}.warp.yaml"
        if recipe_path.exists() and not args.force:  # 不覆盖默认
            fail("input", f"{recipe_path} 已经在了", hint="加 --force 覆盖")
        migration.save(job, recipe_path)
        (out / f"{name}.diff.md").write_text(job.report(), encoding="utf-8")
        total += len(job.mapped)
        if job.blocked:  # 缺基座，记下来统一报
            blocked.append(name)
        print(f"{name}: 搬了 {len(job.mapped)} 项，待确认 {len(job.unknown)} 项，丢弃 {len(job.dropped)} 项")

    print(f"\n共搬 {total} 项。先看 {out}/*.diff.md 再 ori validate。")
    if blocked:
        print(f"这几份缺 model.path，补上才能跑: {', '.join(blocked)}")
        return 1
    return 0
