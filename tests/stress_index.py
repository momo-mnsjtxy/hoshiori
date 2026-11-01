#!/usr/bin/env python
"""落盘索引的压测：同一批语料，纯内存跑一遍、落盘跑一遍，各自在独立子进程里量峰值内存。

调用示例：
  python tests/stress_index.py             # 默认 20 万条
  python tests/stress_index.py 1000000     # 100 万条，几分钟，进度每 10 万条一行
  python tests/stress_index.py 200000      # 二十万，一两分钟

为什么先落语料文件：100 万条要是每次现摇随机词，光造数就能吃掉大半时间，
两种模式还各摇一遍。先写一次文件，两个子进程都读它，才量的是索引本身的代价。

为什么开子进程：VmHWM 是进程级的只升不降高水位。两种模式挤在一个进程里，
后跑的那种会把前一种的峰值一起算进来，量不出差，所以一种一个子进程。

输出：每 10 万条一行进度；结束时两行峰值与耗时，末行是内存比（纯内存 ÷ 落盘，越大越省）
和落盘模式的热层峰值条数。热层峰值不该随语料条数涨，涨了就是热层没管住。
"""

import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # 仓库根，子进程要把它塞进 sys.path
WORDS = ["断线", "张力", "校准", "布面", "打纬", "综框", "真布", "毛边", "纱支", "捻度"]
STEP = 100_000  # 每多少条报一次进度

# 子进程里干的事：逐行读语料文件、一路 check 下去，报出进度、热层条数、峰值内存、耗时。
# store 传 None 就是纯内存；传目录就落盘，跑完目录整棵删掉。
CHILD = """
import sys, time
sys.path.insert(0, {root!r})
from features import duplicate

def peak():  # 这个进程自己的峰值常驻内存，kB
    return int([line.split()[1] for line in open("/proc/self/status") if line.startswith("VmHWM")][0])

dup = duplicate.Duplicate(duplicate.Rule(), store={store!r}, hot={hot})  # 落盘热层就是容量
started = time.perf_counter()
seen = 0
with open({corpus!r}, encoding="utf-8") as lines:
    for text in lines:
        dup.check(text.rstrip("\\n"))
        seen += 1
        if seen % {step} == 0:  # 打点走 stderr，跟最后的结果行分开
            print(f"  {{seen}} 条 · {{time.perf_counter() - started:.0f}} s", file=sys.stderr, flush=True)
spent = time.perf_counter() - started
assert dup.result.kept == {count}, dup.result.kept  # 语料互不重复，一条都不该判掉
hot_rows = len(dup._index._order) if dup._index.store else -1  # 纯内存没有热层淘汰
print(hot_rows, peak(), round(spent, 2), flush=True)
"""


def make_corpus(path: Path, count: int) -> None:
    """摇一份语料写进文件，同一个种子每次一样。"""
    rng = random.Random(1)  # 固定种子，跑几次都是同一批
    with path.open("w", encoding="utf-8") as out:
        for _ in range(count):
            out.write("".join(rng.choice(WORDS) for _ in range(20)) + "。\n")  # 每条 20 个词，长度稳定


def run(store, hot: int, count: int, corpus: Path) -> tuple:
    """在一个子进程里跑一遍，返回（热层条数，峰值 kB，耗时秒）。"""
    snippet = CHILD.format(
        root=str(ROOT), store=str(store) if store else None, hot=hot, count=count, corpus=str(corpus), step=STEP
    )
    began = subprocess.Popen([sys.executable, "-c", snippet], stdout=subprocess.PIPE, stderr=None, text=True)
    result = began.communicate()[0]  # 进度直接透到终端，这里只等最后的数字
    if began.returncode != 0:  # 子进程挂了，主进程也挂
        sys.exit(f"子进程退出码 {began.returncode}")
    hot_rows, peak, spent = result.split()
    return int(hot_rows), int(peak), float(spent)


def main() -> None:
    """跑两种模式并打印对比。参数给了就是语料条数。"""
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000
    hot = 50_000  # 和默认热层一样，这个数就是落盘模式的内存地板
    print(f"语料 {count} 条 · 纯内存与落盘各跑一个子进程，量 VmHWM")

    held = tempfile.mkdtemp(prefix="stress-index-")  # 落盘模式的目录，跑完删
    corpus = Path(held) / "corpus.txt"
    try:
        print("摇语料…", flush=True)
        make_corpus(corpus, count)
        print(f"语料 {corpus.stat().st_size / 1e6:.0f} MB，纯内存跑起", flush=True)
        _, memory_peak, memory_spent = run(None, hot, count, corpus)  # 纯内存：全留在内存里
        print("落盘跑起", flush=True)
        hot_rows, disk_peak, disk_spent = run(held, hot, count, corpus)  # 落盘：热层封顶，其余落盘
    finally:
        import shutil

        shutil.rmtree(held, ignore_errors=True)  # 100 万条能摊出好几个 G，别留着

    print(f"纯内存  峰值 {memory_peak / 1024:8.1f} MB  耗时 {memory_spent:6.2f} s")
    print(f"落盘    峰值 {disk_peak / 1024:8.1f} MB  耗时 {disk_spent:6.2f} s")
    print(f"内存比 {memory_peak / disk_peak:.1f}  热层峰值 {hot_rows} 条")


if __name__ == "__main__":
    main()
