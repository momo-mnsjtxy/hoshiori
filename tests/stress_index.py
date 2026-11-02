#!/usr/bin/env python
"""同一语料分别跑纯内存与默认 v5 前缀索引，子进程测耗时和 VmHWM。

调用：python tests/stress_index.py 200000；百万级可用 --disk-only 避免纯内存基线耗尽机器。
--json /tmp/index.json 保存实测数字。--directory 指定有足够空间的临时工作盘。
旧位置参数 warm 与 --no-pack 仍接受，但 v5 不做转冷，因此不改变新布局行为。
耗时包含 save 与 close；完成后重开抽查首条候选，避免只测未提交的写入。
"""

import argparse  # 压测参数与输出位置由命令行明确指定
import json      # 保存机器可读的实测记录
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

if {no_pack}:  # 冷片不压 zlib，省掉整片的压缩
    from features import index as index_mod
    index_mod.COLD_PACK = False
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
dup._index.settle()  # 旧布局等后台，新布局没有待搬任务
if dup._index.store:
    dup.save()  # 耗时包含持久化，不能只量尚未提交的写入
spent = time.perf_counter() - started
assert dup.result.kept == {count}, dup.result.kept  # 语料互不重复，一条都不该判掉
hot_rows = len(dup._index._order) if dup._index.store else -1  # 纯内存没有热层淘汰
dup.close()  # 关闭也算在本轮耗时里
spent = time.perf_counter() - started
if {store!r}:
    back = duplicate.Duplicate(duplicate.Rule(), store={store!r})
    with open({corpus!r}, encoding="utf-8") as source:
        first = next(source).rstrip("\\n")
    assert back.check(first) is not None  # 重开确实命中已保存的首条语料
    back.close()
print(hot_rows, before_cold[0] if before_cold else 0, peak(), round(spent, 2), flush=True)
"""


def make_corpus(path: Path, count: int) -> None:
    """摇一份语料写进文件，同一个种子每次一样。"""
    rng = random.Random(1)  # 固定种子，跑几次都是同一批
    with path.open("w", encoding="utf-8") as out:
        for _ in range(count):
            out.write("".join(rng.choice(WORDS) for _ in range(20)) + "。\n")  # 每条 20 个词，长度稳定


def run(store, hot: int, count: int, corpus: Path, warm: int = 0, no_pack: bool = False) -> tuple:
    """在一个子进程里跑一遍，返回（热层条数，热层段峰值 kB，跑完峰值 kB，耗时秒）。"""
    snippet = CHILD.format(
        root=str(ROOT),
        store=str(store) if store else None,
        hot=hot,
        count=count,
        corpus=str(corpus),
        step=STEP,
        warm=warm,
        no_pack=no_pack,
    )
    began = subprocess.Popen([sys.executable, "-c", snippet], stdout=subprocess.PIPE, stderr=None, text=True)
    result = began.communicate()[0]  # 进度直接透到终端，这里只等最后的数字
    if began.returncode != 0:  # 子进程挂了，主进程也挂
        sys.exit(f"子进程退出码 {began.returncode}")
    hot_rows, before_cold, peak, spent = result.split()
    return int(hot_rows), int(before_cold), int(peak), float(spent)


def main() -> None:
    """跑两种模式并打印对比。第一个参数是语料条数，第二个是温层上限（不给就用默认），--no-pack 不压冷片。"""
    parser = argparse.ArgumentParser(description=__doc__)  # 参数错误在启动时直接报清
    parser.add_argument("count", type=int, nargs="?", default=200_000)
    parser.add_argument("warm", type=int, nargs="?", default=0)
    parser.add_argument("--no-pack", action="store_true")
    parser.add_argument("--disk-only", action="store_true")
    parser.add_argument("--json", type=Path)
    parser.add_argument("--directory", type=Path)
    args = parser.parse_args()
    count, warm, no_pack = args.count, args.warm, args.no_pack
    if count < 1:
        parser.error("count 必须大于零")
    hot = 50_000  # 和默认热层一样，这个数就是落盘模式的内存地板
    print(f"语料 {count} 条 · v5 哈希前缀布局 · 每种模式独立子进程量 VmHWM")

    held = tempfile.mkdtemp(prefix="stress-index-", dir=args.directory)  # 落盘模式的目录，跑完删
    corpus = Path(held) / "corpus.txt"
    try:
        print("摇语料…", flush=True)
        make_corpus(corpus, count)
        print(f"语料 {corpus.stat().st_size / 1e6:.0f} MB", flush=True)
        memory_peak = memory_spent = None  # disk-only 不捏造内存基线
        if not args.disk_only:
            print("纯内存跑起", flush=True)
            _, _, memory_peak, memory_spent = run(None, hot, count, corpus)
        print("落盘跑起", flush=True)
        hot_rows, before_cold, disk_peak, disk_spent = run(Path(held) / "index", hot, count, corpus, warm, no_pack)  # 落盘：热层封顶
    finally:
        import shutil

        shutil.rmtree(held, ignore_errors=True)  # 100 万条能摊出好几个 G，别留着

    if memory_peak is not None:
        print(f"纯内存  峰值 {memory_peak / 1024:.1f} MiB  耗时 {memory_spent:.2f} s")
        print(f"内存比 {memory_peak / disk_peak:.1f}")
    print(f"v5 前缀  峰值 {disk_peak / 1024:.1f} MiB  耗时 {disk_spent:.2f} s  {count / disk_spent:.0f} 条/秒")
    result = {"count": count, "layout": "hash-prefix-v5", "disk_peak_kib": disk_peak,
              "disk_seconds": disk_spent, "memory_peak_kib": memory_peak, "memory_seconds": memory_spent,
              "memory_ratio": memory_peak / disk_peak if memory_peak else None, "includes_save_close": True}
    if args.json:
        args.json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")  # 实测不混进历史报告


if __name__ == "__main__":
    main()
