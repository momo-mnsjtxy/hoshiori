"""命令 weave：收一批语料，洗、去重、打指纹、出卡。

调用示例：
  ori weave --in data/raw.jsonl --out bolt --name zh-mix --license cc0
  ori weave --in a.jsonl --in b.jsonl --threshold 0.9
  cat a.jsonl | ori weave --in - --out bolt

检查只在这里做一次：输入有没有给、长度范围合不合逻辑。剩下的规矩原样交给 features。
"""

import json
from pathlib import Path

from features import duplicate
from features.dataset import Choice, Dataset
from features.sample import Policy
from utils import logs
from utils.fail import fail


def command(args) -> int:
    """跑一趟 weave。"""
    if not args.inputs:  # 没有输入就无从收
        fail("input", "至少要给一个 --in", hint="例: --in data/a.jsonl")
    if args.min_chars < 0 or args.max_chars < 0:
        fail("input", "长度范围不能是负数", hint="min-chars 和 max-chars 都给正数")
    if args.max_chars and args.max_chars < args.min_chars:  # 上下限写反了
        fail("input", "max-chars 比 min-chars 还小，一条都过不了", hint="把两个数对调")
    if not 0 < args.threshold <= 1:  # 阈值超出 0~1
        fail("input", "threshold 落在 0 到 1 之间", hint="常规用 0.8，严一点 0.9")

    out = Path(args.out)  # 产物目录
    job = Choice(  # 把命令行参数原样搬成一份选择，规矩都在这里定死
        out=out,
        name=args.name,
        license=args.license,
        policy=Policy(fields=tuple(args.fields.split(",")), min_chars=args.min_chars, max_chars=args.max_chars),
        rule=duplicate.Rule(threshold=args.threshold, size=args.size, count=args.count, exact=not args.no_exact),
        algo=args.algo,
        keep_duplicates=args.keep_duplicates,
        dry_run=args.dry_run,
    )

    logs.begin(out)  # 面板侧的记录进 out/loom.jsonl
    try:  # try 是为了中途出错也把文件关掉
        counts = Dataset(job).make(args.inputs)  # 整趟交给数据集，命令层不碰产物
    finally:
        logs.end()

    summary = counts.summary()  # 这一趟的账
    print(json.dumps(summary, ensure_ascii=False, indent=2))  # 机器能直接读
    print(f"\n产物 {out}/bolt.jsonl · {out}/card.json · {out}/fingerprint.json")
    print(f"收下 {summary['kept']} 条（{summary['keep_rate']:.1%}），重复 {counts.duplicated} 条，拒收 {sum(counts.refused.values())} 条")
    return 0
