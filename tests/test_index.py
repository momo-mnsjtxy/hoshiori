"""落盘索引的测试：三层各自能查、热层满了会挤、片会滚、转冷是流式的、落盘能读回来、两种模式判定一样。

约定：用 8 字节的假键和 3 个假窗口，不跑真实的指纹，省时间；
分带键的真实长度由 features/duplicate.py 按段内行数给。
"""

import random
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

from features import duplicate
from features.index import Index

ROOT = Path(__file__).resolve().parent.parent  # 仓库根，压测的子进程要在这儿跑


def keyed(sample: int, band: int) -> bytes:
    """一条样本第 band 个键：样本号和段号各占 4 字节。"""
    return sample.to_bytes(4, "big") + band.to_bytes(4, "big")


def windows_for(sample: int) -> set:
    """这条样本的窗口集合，带点样本号，好认。"""
    return {sample, sample + 1, sample + 2}


def fill(index: Index, count: int, bands: int = 2) -> None:
    """塞 count 条样本，每条 bands 个键。"""
    for sample in range(count):
        for band in range(bands):
            index.put(keyed(sample, band), sample, [sample] * 4, windows_for(sample))


# ---------- 纯内存 ----------

def test_memory_put_get_record():
    index = Index(key_size=8)
    index.put(keyed(1, 0), 1, [7] * 4, {1, 2})
    assert index.get(keyed(1, 0)) == [1]
    assert index.get(keyed(9, 0)) == []
    assert index.record(1) == ([7, 7, 7, 7], {1, 2})
    assert index.size() == 1


def test_memory_does_not_evict():
    index = Index(hot=2, key_size=8)
    fill(index, 20)
    assert len(index._order) == 40  # 纯内存热层没有上限
    assert index.get(keyed(0, 0)) == [0]  # 头一条也还在
    assert index.size() == 20


def test_memory_save_needs_dir():
    with pytest.raises(ValueError):
        Index(key_size=8).save()


# ---------- 落盘三层 ----------

def test_hot_overflow_evicts_but_still_found(tmp_path):
    index = Index(tmp_path, hot=4, per_shard=100, warm=1000, key_size=8)
    fill(index, 30)
    assert len(index._order) <= 4  # 热层被压回容量以内
    assert index._warm  # 挤出去的东西落在温层
    for sample in range(30):  # 挤出去的照样查得到，窗口也读得回来
        assert index.get(keyed(sample, 0)) == [sample]
        assert index.record(sample)[1] == windows_for(sample)


def test_warm_rolls_new_shard(tmp_path):
    index = Index(tmp_path, hot=1, per_shard=3, warm=100, key_size=8)
    fill(index, 12)
    assert len(index._warm) > 1  # 一片满了换新片
    assert sum(shard.count for shard in index._warm) + len(index._live) == 12  # 记录不丢


def test_cold_keeps_every_key(tmp_path):
    index = Index(tmp_path, hot=1, per_shard=2, warm=2, key_size=8)
    fill(index, 40, bands=3)  # 四十条三条键：桶只有 256 个，必然有记录几条键落进同一个桶
    assert index._cold  # 温层攒够上限后最老的片转冷
    for sample in range(40):  # 每一条的每一条键都要撞得上，不能只撞第一条
        for band in range(3):
            assert index.get(keyed(sample, band)) == [sample]
        assert index.record(sample)[1] == windows_for(sample)  # 窗口压过也读得回来
    assert list(tmp_path.glob("cold-*.k*"))  # 全部键按桶压在盘上，不是只留第一条


def test_cold_conversion_is_streaming(tmp_path):
    index = Index(tmp_path, hot=1, per_shard=2, warm=2, key_size=8)
    seen = []  # 转冷每走一步，钩子报一次内存里还压着几条记录
    index.watch = seen.append
    fill(index, 12)
    assert index._cold  # 确实转过冷
    assert seen and max(seen) == 1  # 峰值就是一条，不攒整片
    assert set(seen) <= {0, 1}  # 每一步都只可能是 0 条或 1 条


def test_cold_removes_original_shard(tmp_path):
    index = Index(tmp_path, hot=1, per_shard=2, warm=2, key_size=8)
    fill(index, 12)
    assert index._cold
    alive = {shard.serial for shard in index._warm}  # 还在温层的片号
    for pattern in ("warm-*.rec", "warm-*.off", "warm-*.key", "warm-*.bkt"):
        for one in tmp_path.glob(pattern):  # 转过冷的片文件要当场删干净
            assert int(one.stem.split("-")[1]) in alive, one.name
    assert len(list(tmp_path.glob("warm-*.rec"))) == len(alive)  # 只剩没转冷的那几片


def test_warm_files_carry_key_table(tmp_path):
    index = Index(tmp_path, hot=1, per_shard=2, warm=100, key_size=8)  # 温层管够，不转冷
    fill(index, 6)
    assert list(tmp_path.glob("warm-*.rec"))
    assert list(tmp_path.glob("warm-*.bkt"))  # 温层带键表桶
    assert not list(tmp_path.glob("cold-*.rec"))


# 子进程里一路塞到转冷若干次，量这个进程的峰值内存。单片峰值和连转峰值必须分开进程，
# 同一进程里第二次的峰值会把前一次的算进去。
COLD_PEAK = """
import sys
sys.path.insert(0, {root!r})
from features.index import Index

def keyed(sample, band):
    return sample.to_bytes(4, "big") + band.to_bytes(4, "big")

index = Index({store!r}, hot=2000, per_shard=2000, warm=2000, key_size=8)  # 片缩到 2000 条，跑得快
for sample in range({count}):  # 每条两个键，跟别的用例一个形状
    for band in range(2):
        index.put(keyed(sample, band), sample, [sample] * 4, {{sample}})
assert len(index._cold) >= {cold}, len(index._cold)  # 确实连转了这么多次
print(int([line.split()[1] for line in open("/proc/self/status") if line.startswith("VmHWM")][0]))
"""


def _cold_peak(store: Path, count: int, cold: int) -> int:
    """在子进程里塞到转冷 cold 片，返回峰值内存 kB。"""
    snippet = COLD_PEAK.format(root=str(ROOT), store=str(store), count=count, cold=cold)
    done = subprocess.run([sys.executable, "-c", snippet], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    return int(done.stdout.strip())


def test_multi_cold_peak_bounded(tmp_path):
    single = _cold_peak(tmp_path / "one", 4_500, 1)  # 只转一片
    multi = _cold_peak(tmp_path / "many", 12_000, 5)  # 连转五片
    assert multi <= single * 1.2  # 每片搬完就放开，峰值不跟着片数涨


# ---------- 落盘与读回 ----------

def test_save_reload_keeps_hits(tmp_path):
    store = tmp_path / "a"
    index = Index(store, hot=2, per_shard=100, warm=1000, key_size=8)
    fill(index, 10)
    index.save()
    index.close()

    back = Index(store, key_size=8)
    assert back.size() == 10
    for sample in range(10):  # 命中不丢，窗口也还在
        assert back.get(keyed(sample, 0)) == [sample]
        assert back.record(sample)[1] == windows_for(sample)
    back.close()


def test_save_elsewhere_leaves_original(tmp_path):
    index = Index(tmp_path / "a", hot=2, per_shard=100, key_size=8)
    fill(index, 4)
    other = index.save(tmp_path / "b")
    assert (other / "index.json").exists()
    assert (tmp_path / "a" / "index.json").exists()  # 原目录照样完整
    index.close()


def test_reload_after_cold(tmp_path):
    store = tmp_path / "a"
    index = Index(store, hot=1, per_shard=2, warm=2, key_size=8)
    fill(index, 8)
    index.save()
    index.close()

    assert list(store.glob("cold-*.k*"))  # 冷层键桶跟着片落了盘
    assert not list(store.glob("cold-*.key"))  # 键表随片一起不要了
    back = Index(store, key_size=8)
    assert back.get(keyed(0, 0)) == [0]
    assert back.get(keyed(0, 1)) == [0]  # 第二条键读回来也撞得上
    assert back.record(0)[1] == windows_for(0)  # 压缩的记录还能拆开
    back.close()


# ---------- 两种模式判定一样 ----------

BASE = "星织把散落的星点织成整匹布，训练管线就是织机，梭子交到手上，经线与纬线各自就位。"
WORDS = ["断线", "张力", "校准", "布面", "打纬", "综框", "真布", "毛边", "纱支", "捻度"]


def corpus(count: int = 240) -> list:
    """一批文本：前三成是原文和只动空白的改写，其余是同样词表的随机拼接。"""
    rng = random.Random(11)
    texts = []
    for at in range(count):
        pick = at % 10
        if pick < 2:
            texts.append(BASE * 2)
        elif pick < 3:
            texts.append(BASE.replace("，", " ,  ") * 2)
        else:
            texts.append("".join(rng.choice(WORDS) for _ in range(20)))
    return texts


def test_same_verdict_memory_and_disk(tmp_path):
    texts = corpus()
    memory = duplicate.Duplicate(duplicate.Rule(threshold=0.8))
    disk = duplicate.Duplicate(duplicate.Rule(threshold=0.8), store=tmp_path, hot=3)
    assert [memory.check(text) for text in texts] == [disk.check(text) for text in texts]
    assert memory.result.dropped == disk.result.dropped
    disk.close()


def test_disk_reload_verdict_unchanged(tmp_path):
    texts = corpus()
    memory = duplicate.Duplicate(duplicate.Rule(threshold=0.8))
    disk = duplicate.Duplicate(duplicate.Rule(threshold=0.8), store=tmp_path, hot=3)
    for text in texts:  # 先跑一遍把索引喂满
        memory.check(text)
        disk.check(text)
    disk.save()
    disk.close()

    again = duplicate.Duplicate(duplicate.Rule(threshold=0.8), store=tmp_path, hot=3)
    assert [again.check(text) for text in texts] == [memory.check(text) for text in texts]  # 读回来的索引判定不变
    again.close()


# ---------- 压测 ----------

# 子进程里跑同一批语料，各自报自己进程的峰值内存（VmHWM）。同一进程连着跑两种，
# 后一种的峰值会把前一种的算进去，量不出差，所以必须分开进程。
BENCH = """
import random, sys
sys.path.insert(0, {root!r})
from features import duplicate

WORDS = {words!r}

def corpus(count):
    rng = random.Random(1)
    for at in range(count):
        yield "".join(rng.choice(WORDS) for _ in range(20)) + "。"

dup = duplicate.Duplicate(duplicate.Rule(), store={store!r}, hot={hot})
for text in corpus({count}):
    dup.check(text)
assert dup.result.kept == {count}, dup.result.kept
print(len(dup._index._order) if dup._index.store else -1)  # 热层还剩几条键
print(int([line.split()[1] for line in open("/proc/self/status") if line.startswith("VmHWM")][0]))
"""


def _bench(root: Path, store, hot: int, count: int) -> tuple:
    """在子进程里跑一批语料，返回（热层条数，峰值内存 kB）。"""
    snippet = BENCH.format(root=str(root), words=WORDS, store=str(store) if store else None, hot=hot, count=count)
    done = subprocess.run([sys.executable, "-c", snippet], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    hot_rows, peak = done.stdout.split()
    return int(hot_rows), int(peak)


@pytest.mark.slow  # 十万条，默认跑会拖慢一分钟，pytest -m slow 单独跑
def test_stress_100k_hot_bounded(tmp_path):
    count = 100_000
    hot_rows, disk_peak = _bench(ROOT, tmp_path / "index", 50_000, count)  # 落盘：热层照容量封顶
    assert hot_rows <= 50_000  # 热层没超量，内存就不会被语料条数拖着涨
    _, memory_peak = _bench(ROOT, None, 50_000, count)  # 纯内存做基线
    warnings.warn(  # -q 会把 print 吃掉，用警告让这条实测数出现在总结里
        f"10 万条压测：纯内存峰值 {memory_peak} kB · 落盘峰值 {disk_peak} kB · 热层 {hot_rows} 条",
        stacklevel=1,
    )
    assert disk_peak * 5 <= memory_peak  # 峰值不到纯内存版的五分之一
