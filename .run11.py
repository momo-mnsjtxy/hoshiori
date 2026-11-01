"""1.1M 条落盘跑一遍（不压冷片），只跑落盘这一侧，语料复用文件。"""
import os, random, sys, tempfile, time
sys.path.insert(0, ".")
from pathlib import Path
from features import index as index_mod
from features import duplicate

WARM = int(sys.argv[1]) if len(sys.argv) > 1 else 0
COUNT = int(sys.argv[2]) if len(sys.argv) > 2 else 1_100_000
corpus = Path(".bigcorpus")
WORDS = ["断线", "张力", "校准", "布面", "打纬", "综框", "真布", "毛边", "纱支", "捻度"]
if not corpus.exists():
    rng = random.Random(1)
    with corpus.open("w", encoding="utf-8") as out:
        for _ in range(COUNT):
            out.write("".join(rng.choice(WORDS) for _ in range(20)) + "。\n")
    print("语料摇好了", flush=True)
index_mod.COLD_PACK = False
store = Path(tempfile.mkdtemp(prefix="big11-"))
def peak():
    return int([l.split()[1] for l in open("/proc/self/status") if l.startswith("VmHWM")][0])
dup = duplicate.Duplicate(duplicate.Rule(), store=store, hot=50_000)
if WARM:
    dup._index.warm = WARM
before = []
dup._index.watch = lambda n: before.append(peak()) if not before else None
started = time.perf_counter()
seen = 0
with corpus.open(encoding="utf-8") as lines:
    for text in lines:
        dup.check(text.rstrip("\n"))
        seen += 1
        if seen % 100_000 == 0:
            print(f"  {seen} 条 · {time.perf_counter() - started:.0f} s · {seen / (time.perf_counter() - started):.0f} 条/秒", flush=True)
        if seen >= COUNT:
            break
dup._index.settle()
spent = time.perf_counter() - started
print(f"跑完 {seen} 条 · {spent:.0f} s · {seen / spent:.0f} 条/秒 · 峰值 {peak() / 1024:.1f} MB · 热层段 { (before[0] if before else 0) / 1024:.1f} MB · 冷片 {len(dup._index._cold)}")
dup.close()
