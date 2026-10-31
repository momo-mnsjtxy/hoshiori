"""命令 judge：拿模型跑评测套件，出分数和报告。

调用示例：
  ori judge sft.warp.yaml --suite zh-lite

眼下先核对套件名并把要跑的事报一遍，评分接上后在这里出报告和回归门。
"""

from pathlib import Path

from features import recipe


def command(args) -> int:
    """核对配方与套件，报出计划。"""
    if not args.recipe:  # 没给配方
        print("要给一份配方，例: ori judge sft.warp.yaml --suite zh-lite")
        return 2
    data = recipe.load(Path(args.recipe))  # 读配方
    recipe.check(data)  # 先过校验
    suites = args.suite or (data.get("eval") or {}).get("suites") or []  # 命令行优先
    print(f"要评 {args.recipe}，套件 {', '.join(suites) if suites else '（没给）'}")
    print("这一步还没接评分后端")
    return 0
