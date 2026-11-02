"""命令 cost：抓取公开价钱、在当前 GPU 校准，或使用两份证据生成训练账。

调用示例：
  ori cost prices --gpu 'NVIDIA GeForce RTX 5090' --cloud secure --out price.json
  ori cost calibrate demo.warp.yaml --batches 1,2,4 --steps 10 --out profile.json
  ori cost estimate demo.warp.yaml --profile profile.json --price price.json
"""

import json                              # 证据 JSON 与命令输出使用同一结构
from pathlib import Path                 # 用户指定的配方和产物位置

from features import cost, plan, recipe   # 指令只编排价格、实测与训练账
from utils.fail import fail              # 命令入口统一呈现用户可修正的错误


def command(args) -> int:
    """按 action 执行；校准必须明确给出产物路径，不能跑完丢掉证据。"""
    if args.action == "prices":
        if not args.gpu:
            fail("input", "prices 需要 --gpu 的精确 GPU id")
        result = cost.prices(args.gpu, args.cloud)  # 公共查询，不创建实例或产生云端费用
    else:
        if not args.recipe:
            fail("input", f"{args.action} 需要配方路径")
        path = Path(args.recipe).resolve()
        data = recipe.load(path)         # 继承展开仍使用项目原有配方逻辑
        recipe.check(data)
        if args.action == "calibrate":
            if not args.out:
                fail("input", "calibrate 需要 --out 保存完整真机证据")
            try:
                batches = [int(value.strip()) for value in args.batches.split(",")]
            except ValueError:
                fail("input", "--batches 要用逗号分隔正整数，例如 1,2,4")
            result = cost.calibrate(data, batches=batches, steps=args.steps, warmup=args.warmup)
        else:
            evidence = data.setdefault("cost", {})  # CLI 显式路径覆盖配方中对应字段
            if args.profile:
                evidence["profile"] = str(Path(args.profile).resolve())
            if args.price:
                evidence["price"] = str(Path(args.price).resolve())
            table = plan.build(data, base_dir=path.parent)
            print(table.show())         # 费用和推荐在同一张训练账，未知值也明确显示
            result = cost.estimate(table, tokens_per_step=table.tokens * table.batch * table.grad_accum,
                                   profile=table.profile, configuration=table.configuration, price=table.price)
    if args.out:
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(str(target))              # 给调用者一个能直接打开的证据位置
    elif args.action != "estimate":
        print(json.dumps(result, ensure_ascii=False, indent=2))  # 价格也可直接交给下游管道
    return 0
