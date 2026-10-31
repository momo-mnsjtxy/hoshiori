#!/usr/bin/env python
"""落盘索引的压测：同一批语料，纯内存跑一遍、落盘跑一遍，各自在独立子进程里量峰值内存。

调用示例：
  python tests/stress_index.py            # 默认 20 万条
  python tests/stress_index.py 1000000    # 100 万条
  python tests/stress_index.py 200000     # 跑完大概一两分钟，看最后两行就够

为什么开子进程：VmHWM 是进程级的只升不降高水位。两种模式挤在一个进程里，
后跑的那种会把前一种的峰值一起算进来，量不出差，所以一种一个子进程。

输出：两行各自的峰值与耗时，末行是内存比（纯内存峰值 ÷ 落盘峰值，越大越省）
和落盘模式的热层峰值条数。热层峰值不该随语料条数涨，涨了就是热层没管住。
"""

import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # 仓库根，子进程要把它塞进 sys.path
WORDS = ["断线", "张力", "校准", "布面", "打纬", "综框", "真布", "毛边", "纱支", "捻度"]

# 子进程里干的事：造 count 条语料一路 check 下去，报出热层条数、峰值内存、耗时。
# store 传 None 就是纯内存；传目录就落盘，跑完目录整棵删掉。
CHILD = """
import random, sys, time
sys.path.insert(0, {root!r})
from features import duplicate

WORDS = {words!r}

def corpus(count):  # 同一个种子，两种模式吃的是同一批语料
    rng = random.Random(1)
    for at in range(count):
        yield "".join(rng.choice(WORDS) for _ in range(20)) + "。"

def peak():  # 这个进程自己的峰值常驻内存，kB
    return int([line.split()[1] for line in open("/proc/self/status") if line.startswith("VmHWM")][0])

dup = duplicate.Duplicate(duplicate.Rule(), store={store!r}, hot={hot})  # 落盘热层就是容量
started = time.perf_counter()
for text in corpus({count}):
    dup.check(text)
spent = time.perf_counter() - started
assert dup.result.kept == {count}, dup.result.kept  # 语料互不重复，一条都不该判掉
hot_rows = len(dup._index._order) if dup._index.store else -1  # 纯内存没有热层淘汰
print(hot_rows, peak(), round(spent, 2))
"""


def run(store, hot: int, count: int) -> tuple:
    """在一个子进程里跑一遍，返回（热层条数，峰值 kB，耗时秒）。"""
    snippet = CHILD.format(root=str(ROOT), words=WORDS, store=str(store) if store else None, hot=hot, count=count)
    done = subprocess.run([sys.executable, "-c", snippet], capture_output=True, text=True)
    if done.returncode != 0:  # 子进程挂了，原样把它的输出丢出来
        sys.exit(done.stderr)
    hot_rows, peak, spent = done.stdout.split()
    return int(hot_rows), int(peak), float(spent)


def main() -> None:
    """跑两种模式并打印对比。参数给了就是语料条数。"""
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000
    hot = 50_000  # 和默认热层一样，这个数就是落盘模式的内存地板
    print(f"语料 {count} 条 · 纯内存与落盘各跑一个子进程，量 VmHWM")

    held = tempfile.mkdtemp(prefix="stress-index-")  # 落盘模式的目录，跑完删
    try:
        _, memory_peak, memory_spent = run(None, hot, count)  # 纯内存：全留在内存里
        hot_rows, disk_peak, disk_spent = run(held, hot, count)  # 落盘：热层封顶，其余落盘
    finally:
        import shutil

        shutil.rmtree(held, ignore_errors=True)  # 100 万条能摊出好几个 G，别留着

    print(f"纯内存  峰值 {memory_peak / 1024:8.1f} MB  耗时 {memory_spent:6.2f} s")
    print(f"落盘    峰值 {disk_peak / 1024:8.1f} MB  耗时 {disk_spent:6.2f} s")
    print(f"内存比 {memory_peak / disk_peak:.1f}  热层峰值 {hot_rows} 条")


if __name__ == "__main__":
    main()
