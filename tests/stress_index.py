#!/usr/bin/env python
"""落盘索引的压测：同一批语料，纯内存跑一遍、落盘跑一遍，各自在独立子进程里量峰值内存。

调用示例：
  python tests/stress_index.py             # 默认 20 万条
  python tests/stress_index.py 1000000     # 100 万条，几分钟，进度每 10 万条一行
  python tests/stress_index.py 200000      # 二十万，一两分钟
  python tests/stress_index.py 200000 190000  # 二十万条，温层上限压到十九万：末尾转冷一片

为什么先落语料文件：100 万条要是每次现摇随机词，光造数就能吃掉大半时间，
两种模式还各摇一遍。先写一次文件，两个子进程都读它，才量的是索引本身的代价。

为什么开子进程：VmHWM 是进程级的只升不降高水位。两种模式挤在一个进程里，
后跑的那种会把前一种的峰值一起算进来，量不出差，所以一种一个子进程。

输出：每 10 万条一行进度；结束时把落盘模式的峰值分成两段报：热层跑完（首次转冷那一刻）、
转冷之后（跑完那一刻），再报内存比（纯内存 ÷ 落盘，越大越省）和落盘模式的热层峰值条数。
分成两段是为了看清内存到底是涨在哪一层：热层段只跟热层容量有关，段数一多就该是平的；
转冷段涨，就是转冷时攒在内存里的键缓冲没管住。两句都该是平的，才说明曲线走平。
默认温层上限 100 万条，二十万条以内不会转冷，转冷段就是空的；想看转冷那一段要传第二个参数。
温层上限压到比一片稍小，就在末尾转一片冷出来；压得太低（比如两万）会摊出十几个冷片，
每查一个键都要把这十几个冷片各翻一个桶，整体慢好几倍。
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

# 子进程里干的事：逐行读语料文件、一路 check 下去，报出进度、热层条数、两段峰值、耗时。
# store 传 None 就是纯内存；传目录就落盘，跑完目录整棵删掉。
# 两段峰值靠 watch 钩子抓：转冷每走一步都会调一次，第一条就是"热层刚跑完、还没开始转冷"那一刻。
# 没有转冷就直接报 0，主进程按"这次没转冷"打印。
CHILD = """
import sys, time
sys.path.insert(0, {root!r})
from features import duplicate

def peak():  # 这个进程自己的峰值常驻内存，kB
    return int([line.split()[1] for line in open("/proc/self/status") if line.startswith("VmHWM")][0])

dup = duplicate.Duplicate(duplicate.Rule(), store={store!r}, hot={hot})  # 落盘热层就是容量
if {warm} > 0:  # 温层上限压小一点，二十万条也能转冷
    dup._index.warm = {warm}
before_cold = []  # 首次转冷那一刻的峰值；整轮不转冷就一直是空的

def probe(_buffered):  # 转冷每走一步调一次，只记第一条
    if not before_cold:
        before_cold.append(peak())

dup._index.watch = probe
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
print(hot_rows, before_cold[0] if before_cold else 0, peak(), round(spent, 2), flush=True)
"""


def make_corpus(path: Path, count: int) -> None:
    """摇一份语料写进文件，同一个种子每次一样。"""
    rng = random.Random(1)  # 固定种子，跑几次都是同一批
    with path.open("w", encoding="utf-8") as out:
        for _ in range(count):
            out.write("".join(rng.choice(WORDS) for _ in range(20)) + "。\n")  # 每条 20 个词，长度稳定


def run(store, hot: int, count: int, corpus: Path, warm: int = 0) -> tuple:
    """在一个子进程里跑一遍，返回（热层条数，热层段峰值 kB，跑完峰值 kB，耗时秒）。"""
    snippet = CHILD.format(
        root=str(ROOT),
        store=str(store) if store else None,
        hot=hot,
        count=count,
        corpus=str(corpus),
        step=STEP,
        warm=warm,
    )
    began = subprocess.Popen([sys.executable, "-c", snippet], stdout=subprocess.PIPE, stderr=None, text=True)
    result = began.communicate()[0]  # 进度直接透到终端，这里只等最后的数字
    if began.returncode != 0:  # 子进程挂了，主进程也挂
        sys.exit(f"子进程退出码 {began.returncode}")
    hot_rows, before_cold, peak, spent = result.split()
    return int(hot_rows), int(before_cold), int(peak), float(spent)


def main() -> None:
    """跑两种模式并打印对比。第一个参数是语料条数，第二个是温层上限（不给就用默认）。"""
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000
    warm = int(sys.argv[2]) if len(sys.argv) > 2 else 0  # 0 就是默认温层上限
    hot = 50_000  # 和默认热层一样，这个数就是落盘模式的内存地板
    print(f"语料 {count} 条 · 纯内存与落盘各跑一个子进程，量 VmHWM")

    held = tempfile.mkdtemp(prefix="stress-index-")  # 落盘模式的目录，跑完删
    corpus = Path(held) / "corpus.txt"
    try:
        print("摇语料…", flush=True)
        make_corpus(corpus, count)
        print(f"语料 {corpus.stat().st_size / 1e6:.0f} MB，纯内存跑起", flush=True)
        _, _, memory_peak, memory_spent = run(None, hot, count, corpus)  # 纯内存：全留在内存里
        print("落盘跑起", flush=True)
        hot_rows, before_cold, disk_peak, disk_spent = run(held, hot, count, corpus, warm)  # 落盘：热层封顶
    finally:
        import shutil

        shutil.rmtree(held, ignore_errors=True)  # 100 万条能摊出好几个 G，别留着

    print(f"纯内存  峰值 {memory_peak / 1024:8.1f} MB  耗时 {memory_spent:6.2f} s")
    print(f"落盘    热层段峰值 {before_cold / 1024:8.1f} MB  转冷段峰值 {disk_peak / 1024:8.1f} MB  耗时 {disk_spent:6.2f} s")
    if before_cold:  # 这一轮转过冷，两段能分开看
        print(f"        转冷这一段涨了 {(disk_peak - before_cold) / 1024:.1f} MB")
    else:
        print("        这一轮没转冷（语料没到温层上限），转冷段就是跑完那一刻的数")
    print(f"内存比 {memory_peak / disk_peak:.1f}  热层峰值 {hot_rows} 条")


if __name__ == "__main__":
    main()
