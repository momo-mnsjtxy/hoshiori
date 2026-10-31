"""命令 vocab：从语料学一个字节级 BPE 词表，训练和导出共用这一份。

调用示例：
  ori vocab --in bolt/bolt.jsonl --out vocab.json --size 32000
  ori vocab --in a.jsonl --in b.jsonl --out vocab.json --size 8000 --limit 5000

学出来的 vocab.json 放进配方的 model.tokenizer，训练按它编码，forge 又把同一份词表写进 GGUF，
两边一份词表，导出的模型才认得训练时编出来的号。
检查只在这里做一遍：输入在不在、词表大小够不够、输出目录��不能写。
"""

from pathlib import Path

from features import tokenizer
from utils.fail import fail


def command(args) -> int:
    """学词表并落盘。"""
    if not args.inputs:  # 没有语料就无从学
        fail("input", "至少要给一个 --in", hint="例: --in bolt/bolt.jsonl")
    if args.size < 259:  # 三个特符加 256 个字节是底线
        fail("input", "词表至少 259（3 个特符 + 256 个字节）", hint="常规正文用 8000 到 32000")
    out = Path(args.out)
    if not out.parent.is_dir():  # 输出目录得先在
        fail("input", f"输出目录不在: {out.parent}", hint="先建好目录")

    sources = [Path(one) for one in args.inputs]  # 语料路径
    vocab = tokenizer.Vocab.train(sources, args.size, limit=args.limit)  # 学词表
    vocab.save(out)  # 落盘
    print(f"词表 {vocab.size} 个词 → {out}")
    print(f"把 model.tokenizer 指向它，训练会按这份词表编码；导出时同一份词表会写进产物")
    return 0
