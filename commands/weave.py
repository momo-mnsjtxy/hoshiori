"""命令 weave：收一批语料，洗、去重、打指纹、出卡。

调用示例：
  ori weave --in data/raw.jsonl --out bolt --name zh-mix --license cc0
  ori weave --in a.jsonl --in b.jsonl --threshold 0.9
  ori weave --in data/raw.jsonl --out bolt --store bolt     # 索引落到 bolt/index，百万级语料用
  cat a.jsonl | ori weave --in - --out bolt

检查只在这里做一次：输入有没有给、长度范围合不合逻辑。剩下的规矩原样交给 features。
"""

import json
import os
from pathlib import Path

from features import duplicate
from features.dataset import Choice, Dataset
from features.sample import Policy
from utils import logs
from utils.fail import fail


def command(args) -> int:
    """跑一趟 weave。"""
    store = Path(args.store) if args.store else None  # 索引落盘目录，不给就全程在内存
    if store is not None:  # 落盘先试目录，建不出来就别等跑到一半才炸
        if store.exists() and not store.is_dir():  # 指到一个文件上，mkdir 只会给出奇怪的报错
            fail("input", f"--store 指到了一个文件: {store}", hint="换一个目录")
        _check_store(store / "index")
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
        store_dir=store,
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


def _check_store(store: Path) -> None:
    """落盘目录要能建、能写。"""
    if store.exists() and not store.is_dir():  # 同名文件占了位置
        fail("input", f"落盘位置不是目录: {store}", hint="--store 指一个目录")
    try:
        store.mkdir(parents=True, exist_ok=True)  # 目录先建出来
    except OSError as bad:  # 没权限、路径上挡着别的
        fail("input", f"落盘目录建不出来: {store}", hint=str(bad))
    if not os.access(store, os.W_OK):  # 建得出来但写不进
        fail("input", f"落盘目录写不了: {store}", hint="换一个有写权限的目录")
