"""只量索引本身：20 万条 × 32 个 16 字节键，先全灌再全查。"""
import random, tempfile, time
from pathlib import Path
from features.index import Index

rng = random.Random(1)
N = 200_000
K = 32
d = Path(tempfile.mkdtemp())
keys = [[rng.randbytes(16) for _ in range(K)] for _ in range(N)]
idx = Index(d, hot=50_000, key_size=16)
t = time.perf_counter()
for sub, row in enumerate(keys):
    for k in row:
        idx.put(k, sub, [sub] * 4, {sub})
put = time.perf_counter() - t
t = time.perf_counter()
hits = 0
for row in keys:
    for k in row:
        hits += len(idx.get(k))
get = time.perf_counter() - t
print(f"put {put:.1f}s  get {get:.1f}s  命中 {hits}")
idx.close()
