"""落盘索引的测试：三层各自能查、热层满了会挤、片会滚、桶表按片落在盘上、转冷是流式且按批刷、
转冷在后台不挡 put、关闭会排队排空、压与不压判定一样、落盘能读回来、两种模式判定一样，
外加键表按条分片、冷片单桶文件两桶不串、桶链按序走、查询只读目标桶。

约定：用 8 字节的假键和 3 个假窗口，不跑真实的指纹，省时间；
分带键的真实长度由 features/duplicate.py 按段内行数给。
转冷改成后台以后，断言「已经转过冷」之前要 settle()：不刹车时后台可能还排着队。
"""

import array
import os
import random
import subprocess
import sys
import threading
import time
import warnings
import zlib
from pathlib import Path

import pytest

from features import duplicate, index as index_mod
from features.index import COLD_BATCH, COLD_HEAD, KEY_SLICE, LegacyIndex as Index, Shard

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
    index.settle()  # 等后台把那几片搬完
    assert index._cold  # 温层攒够上限后最老的片转冷
    for sample in range(40):  # 每一条的每一条键都要撞得上，不能只撞第一条
        for band in range(3):
            assert index.get(keyed(sample, band)) == [sample]
        assert index.record(sample)[1] == windows_for(sample)  # 窗口压过也读得回来
    assert list(tmp_path.glob("cold-*.chk"))  # 全部键按桶串在盘上，不是只留第一条


def test_cold_conversion_is_streaming(tmp_path):
    index = Index(tmp_path, hot=1, per_shard=2, warm=2, key_size=8)
    seen = []  # 转冷每走一步，钩子报一次缓冲里还压着几条记录
    index.watch = seen.append
    fill(index, 12)
    index.settle()
    assert index._cold  # 确实转过冷
    assert seen  # 钩子确实报过
    assert max(seen) < COLD_BATCH  # 这一片连一批都没攒满，更不会攒整片
    assert min(seen) == 1  # 一条一条攒，不一次压一批进去


def test_cold_flushes_in_batches(tmp_path):
    """片长过一批：满一批就刷出去、缓冲清空，峰值还是一批的量。"""
    index = Index(tmp_path, hot=1, per_shard=COLD_BATCH + 100, warm=COLD_BATCH + 100, key_size=8)
    seen = []  # 每搬一条报一次缓冲里压着几条
    index.watch = seen.append
    fill(index, COLD_BATCH + 200)
    index.settle()
    assert index._cold  # 攒满的片转过冷了
    assert max(seen) <= COLD_BATCH  # 从没超过一批
    assert 0 in seen  # 满一批就刷，刷完缓冲是空的
    for sample in (0, COLD_BATCH, COLD_BATCH + 100):  # 转过冷的键照样查得到
        assert index.get(keyed(sample, 0)) == [sample]


def test_cold_removes_original_shard(tmp_path):
    index = Index(tmp_path, hot=1, per_shard=2, warm=2, key_size=8)
    fill(index, 12)
    index.settle()
    assert index._cold
    alive = {shard.serial for shard in index._warm}  # 还在温层的片号
    for pattern in ("warm-*.rec", "warm-*.off", "warm-*.ket[0-9]*", "warm-*.bkt"):
        for one in tmp_path.glob(pattern):  # 转过冷的片文件要当场删干净
            assert int(one.stem.split("-")[1]) in alive, one.name
    assert len(list(tmp_path.glob("warm-*.rec"))) == len(alive)  # 只剩没转冷的那几片


def test_cold_deletes_bucket_table(tmp_path):
    """转过冷的片，桶表跟着片号一起删掉，不留死文件。"""
    index = Index(tmp_path, hot=1, per_shard=2, warm=2, key_size=8)
    fill(index, 12)
    index.settle()
    assert index._cold
    alive = {shard.serial for shard in index._warm}
    left = [one for one in tmp_path.glob("warm-*.bkt") if int(one.stem.split("-")[1]) not in alive]
    assert not left  # 转冷的片一片桶片都不剩
    assert list(tmp_path.glob("cold-*.chk"))  # 键没丢，换到冷层的桶链文件里了


def test_warm_files_carry_key_table(tmp_path):
    index = Index(tmp_path, hot=1, per_shard=2, warm=100, key_size=8)  # 温层管够，不转冷
    fill(index, 6)
    index.save()  # 桶表跟着落盘
    assert list(tmp_path.glob("warm-*.rec"))
    assert list(tmp_path.glob("warm-*.bkt"))  # 温层带键表桶，桶表落盘（整块一个文件）
    assert not list(tmp_path.glob("cold-*.rec"))


# ---------- 转冷在后台 ----------

def slow_conversion(index: Index, spend: float, seen: list = None) -> None:
    """把转冷换成一个慢的：睡 spend 秒再真搬。同步实现下这 sleep 就是主流程白等的时间。"""
    real = index._to_cold

    def slow(shard, cold):
        if seen is not None:  # 谁被搬了，记下来
            seen.append(shard.serial)
        time.sleep(spend)
        real(shard, cold)

    index._to_cold = slow


def test_put_does_not_wait_for_cold(tmp_path):
    """转冷挪到后台以后，灌语料的耗时不再跟着转冷次数长。

    每片转冷故意拖到 0.3 秒：同步实现下这 0.3 秒是灌到一半时白等的，
    两千条会白等五六秒；后台跑的话灌完只用几十毫秒，片还在队里排着。
    """
    index = Index(tmp_path, hot=200, per_shard=200, warm=400, key_size=8)
    slow_conversion(index, 0.3)
    began = time.perf_counter()
    fill(index, 2000)
    spent = time.perf_counter() - began
    queued = index._left  # 灌完这会儿后台还剩几片没搬
    index.settle()
    assert queued >= 1  # 片还排着队，主流程没等它
    assert spent < 1.0  # 灌完只用了几十毫秒，不是一路白等到转完
    assert len(index._cold) == queued  # 排着的都真搬成了冷片
    assert not index._jobs  # 队列排空


def test_close_drains_queue(tmp_path):
    """close 先把排着队的片搬完再关：关完冷层片数就是触发过转冷的片数。"""
    index = Index(tmp_path, hot=5, per_shard=20, warm=40, key_size=8)
    moved = []  # 后台真正搬过的片号
    slow_conversion(index, 0.01, moved)
    fill(index, 400)
    index.close()
    assert len(index._cold) == len(moved)  # 排着队的全搬完了，一片不剩
    assert not index._jobs  # 队列空了
    assert len(list(tmp_path.glob("cold-*.rec"))) == len(moved)  # 真落了盘，不是只在内存记账


def test_worker_error_reaches_close(tmp_path):
    """后台转冷抛异常要能从 close 抛出来，不能静默。"""
    index = Index(tmp_path, hot=5, per_shard=20, warm=40, key_size=8)

    def boom(shard, cold):
        raise RuntimeError("转冷挂了")

    index._to_cold = boom
    fill(index, 100)
    with pytest.raises(RuntimeError):
        index.close()


def test_read_while_background_converts(tmp_path):
    """一边灌一边读：后台正搬着片，主线程按老下标回读也不能炸。

    真机上就是这么用的：后台搬片的同一秒，主流程还在按候选下标回读签名。
    搬完的片文件是删掉的，句柄留着才读得到；这一步没接好会报 mmap 长度大于文件。
    """
    index = Index(tmp_path, hot=100, per_shard=1000, warm=2000, key_size=8)  # 千条一片、两千温层
    stop = threading.Event()
    hurt = []  # 读挂了就把原因记在这儿，只记第一条

    def reader():
        while not stop.is_set():
            for sub in range(index.size()):
                try:
                    index.record(sub)  # 按老下标回读
                    index.get(keyed(sub, 0))  # 顺带查一下键
                except Exception as boom:
                    hurt.append(repr(boom))
                    return

    other = threading.Thread(target=reader)
    other.start()
    fill(index, 4000)  # 四片都会转冷；片放大到千条，一次转冷里主线程要回读很多次
    stop.set()
    other.join()
    index.settle()
    assert not hurt
    for sample in range(4000):  # 搬完照样查得到
        assert index.get(keyed(sample, 1)) == [sample]


# ---------- 桶表落盘 ----------

def test_bucket_table_lands_on_disk_whole(tmp_path):
    """桶表整块落在盘上，读回来照样按桶查得到。"""
    store = tmp_path / "s"
    index = Index(store, hot=1, per_shard=200_000, warm=100_000, key_size=8)
    fill(index, 400)  # 八百条键
    index.save()
    index.close()

    ones = list(store.glob("warm-*.bkt"))
    assert len(ones) == 1  # 一片一个文件，不按桶摊
    assert ones[0].stat().st_size > 0  # 盘上有内容，不是空壳
    back = Index(store, key_size=8)  # 桶头整块读回来
    for sample in (0, 137, 399):
        assert back.get(keyed(sample, 0)) == [sample]
    back.close()


def test_bucket_table_size_follows_the_shard(tmp_path):
    """桶表就一格定长数组：条数只跟片大小有关，不跟着灌进去多少条涨。"""
    index = Index(tmp_path, hot=1, per_shard=200_000, warm=100_000, key_size=8)
    fill(index, 400)
    for shard in index._warm:
        assert len(shard.table.heads) == shard.buckets  # 一个桶一格，整片就这么多
        assert index.get(keyed(0, 0)) == [0]  # 照样查得到
    index.close()


# ---------- 键表按条分片 ----------

def test_key_entry_slice_reads_that_entry(tmp_path):
    """按条号读回来的就是那一条：片号、片内位置全按条号算，不整份映射。"""
    shard = Shard(tmp_path, 1, "warm", 8, 1024)
    for i in range(5):
        shard.bump_key(keyed(i, 3), i)  # 键表末尾接五条
    for i in range(5):  # 还在缓冲里，直接从内存切
        entry = shard._key_at(i)  # 读第 i 条
        assert entry[:8] == keyed(i, 3)  # 键就是它自己那条，没串位
        assert int.from_bytes(entry[8:16], "big") == i  # 下��也是
    shard.flush()  # 落地，同一条从盘上 pread 回来也要一样
    for i in range(5):
        assert shard._key_at(i)[:8] == keyed(i, 3)
    shard.close()


def test_key_entry_slices_cross_boundary(tmp_path):
    """跨片边界：一片满了就换下一片，两边的条目各自都读得回来。"""
    shard = Shard(tmp_path, 1, "warm", 8, 1024)
    for i in range(KEY_SLICE + 2):
        shard.bump_key(keyed(i, 0), i)
    shard.flush()  # 缓冲写出去才量得到文件长短
    files = sorted(tmp_path.glob("warm-*.ket*"))
    assert len(files) == 2  # 满了换片，不挤在一个文件里
    assert files[0].stat().st_size == KEY_SLICE * shard.entry  # 上一片正好满
    assert files[1].stat().st_size == 2 * shard.entry  # 多出来的两条落在下一片
    assert shard._key_at(KEY_SLICE - 1)[:8] == keyed(KEY_SLICE - 1, 0)  # 片尾那条还在上一片
    assert shard._key_at(KEY_SLICE)[:8] == keyed(KEY_SLICE, 0)  # 跨片的第一条在新片
    shard.close()


def test_key_chain_crosses_slice(tmp_path):
    """同一个键的链跨过片边界，find 顺着链照样把两边的下标都取回来。"""
    shard = Shard(tmp_path, 1, "warm", 8, 8)  # 桶只有八个，同一个键一直落同一个桶、链一路往下长
    for i in range(KEY_SLICE + 1):
        shard.bump_key(keyed(0, 0), i)
    assert sorted(shard.find(keyed(0, 0))) == list(range(KEY_SLICE + 1))
    shard.close()


# ---------- 冷片单文件桶链 ----------

def key_in(bucket: int, buckets: int) -> bytes:
    """找一个正好落在 bucket 号桶上的键。"""
    for n in range(100_000):
        key = n.to_bytes(8, "big")
        if zlib.crc32(key) % buckets == bucket:
            return key
    raise AssertionError(f"找不到落在第 {bucket} 桶上的键")


def test_cold_two_buckets_stay_apart(tmp_path):
    """一个冷片一个桶文件，两个桶各走各的链，谁也不串到谁身上。"""
    shard = Shard(tmp_path, 1, "cold", 8, 4)
    shard.start_heads()  # 桶头先挂内存，跟转冷时一样
    one = key_in(1, 4)
    other = key_in(2, 4)
    shard.bump(one, 11)
    shard.bump(other, 22)
    shard.flush()
    assert shard.candidates(one) == [11]  # 自己的桶里只有自己那条
    assert shard.candidates(other) == [22]
    assert shard._heads[1] and shard._heads[2]  # 两个桶各有一个链头
    assert shard._heads[1] != shard._heads[2]  # 链头不在同一处
    assert len(list(tmp_path.glob("cold-*.chk"))) == 1  # 全片就一个桶文件，不是按桶摊一堆
    shard.close()


def test_cold_chain_walks_newest_first(tmp_path):
    """桶链按写进去的先后倒着走，最后收在 0。"""
    shard = Shard(tmp_path, 1, "cold", 8, 2)
    shard.start_heads()
    key = key_in(0, 2)
    for sub in (7, 8, 9):
        shard.bump(key, sub)
    shard.flush()
    assert shard.candidates(key) == [9, 8, 7]  # 新写的在链头，老的往后
    at = shard._heads[zlib.crc32(key) % 2]
    walked = []
    while at:  # 自己顺着链走一遍，确认收在 0
        head = os.pread(shard._chk, COLD_HEAD, at)
        walked.append(int.from_bytes(head[4:8], "big"))
        at = int.from_bytes(head[:4], "big")
    assert walked == [9, 8, 7]
    assert at == 0  # 链到头
    shard.close()


def test_query_reads_one_bucket_only(tmp_path, monkeypatch):
    """查一次只读目标那一条链：几百条条目铺在文件里，读到的只有一条头加一块键。"""
    buckets = 256
    shard = Shard(tmp_path, 1, "cold", 8, buckets)
    shard.start_heads()
    for n in range(buckets):  # 每个桶都塞一条，别的桶全都有东西
        shard.bump(key_in(n, buckets), n)
    shard.flush()
    shard.drop_heads()  # 搬完把内存那份桶头放掉，查询只能读盘上的
    read = []  # 桶链文件上每次读了多少字节

    def spy(fd, size, offset):
        if fd == shard._chk:  # 只数桶链文件上的读
            read.append(size)
        return real(fd, size, offset)

    real = os.pread
    monkeypatch.setattr(index_mod.os, "pread", spy)
    assert shard.candidates(key_in(99, buckets)) == [99]
    assert sum(read) == 4 + COLD_HEAD + shard.key_size  # 一个桶头、一条条目头、一块键，就这么多
    assert sum(read) < shard._chk_bytes // 32  # 远小于整片的字节数（一个桶平均也有一条以上的量）
    shard.close()


def test_key_table_and_cold_survive_reload(tmp_path):
    """键表分片和冷层桶链都要过 save/reload：读回来照样查得到，条目一条不错位。"""
    store = tmp_path / "s"
    index = Index(store, hot=2, per_shard=40, warm=80, key_size=8)
    fill(index, 200)  # 前面几片会转冷，末尾还留一片温层
    index.save()
    index.close()

    back = Index(store, key_size=8)
    assert back._cold  # 冷片读回来了
    for sample in range(200):  # 冷层、温层两边都撞得上
        assert back.get(keyed(sample, 1)) == [sample]
    warm = max(back._warm, key=lambda one: one._key_entries)  # 还有键表的那一片（末尾那片可能是空的）
    assert warm._key_entries  # 没转冷那片的键表读回来还在
    for at in range(warm._key_entries):  # 逐条读，位置不错位
        entry = warm._key_at(at)
        assert int.from_bytes(entry[8:16], "big") in warm.find(entry[:8])  # 顺着链走得到它自己的下标
    back.close()

# 子进程里一路塞到转冷若干次，量这个进程的峰值内存。单片峰值和连转峰值必须分开进程，
# 同一进程里第二次的峰值会把前一次的算进去。
COLD_PEAK = """
import sys
sys.path.insert(0, {root!r})
from features.index import LegacyIndex as Index

def keyed(sample, band):
    return sample.to_bytes(4, "big") + band.to_bytes(4, "big")

index = Index({store!r}, hot=2000, per_shard=2000, warm=2000, key_size=8)  # 片缩到 2000 条，跑得快
for sample in range({count}):  # 每条两个键，跟别的用例一个形状
    for band in range(2):
        index.put(keyed(sample, band), sample, [sample] * 4, {{sample}})
index.settle()  # 后台搬完，量过峰值
while len(index._cold) < {cold}:  # 再一片一片推，推够这么多次连转
    index._park_oldest()  # 直接排队最老的那片
    index.settle()  # 等它搬完再推下一片，峰值才量得出是连着的
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
    multi = _cold_peak(tmp_path / "many", 12_000, 4)  # 连转四片
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

    assert list(store.glob("cold-*.chk"))  # 冷层键链跟着片落了盘
    assert not list(store.glob("cold-*.ket*"))  # 冷层的键不再单留一份键表，全在桶链里
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


def test_same_verdict_through_cold(tmp_path, monkeypatch):
    """一路转冷也不改判定：同一批语料，内存版和落盘版逐条一模一样。"""
    monkeypatch.setattr(index_mod, "Index", Index)  # 此用例专测旧 v4 转冷；v5 见 test_prefix_index.py
    texts = corpus()
    memory = duplicate.Duplicate(duplicate.Rule(threshold=0.8))
    disk = duplicate.Duplicate(duplicate.Rule(threshold=0.8), store=tmp_path, hot=3)
    disk._index.per_shard = 4  # 片缩到四条、温层只留八条，几百条语料会一路转冷
    disk._index.warm = 8
    assert [memory.check(text) for text in texts] == [disk.check(text) for text in texts]
    disk._index.settle()
    assert disk._index._cold  # 确实走到了冷层那条路
    assert memory.result.dropped == disk.result.dropped
    disk.close()


def test_pack_switch_keeps_verdict(tmp_path, monkeypatch):
    """压与不压，同一批语料的判定逐条一样；不压确实更占盘。"""
    monkeypatch.setattr(index_mod, "Index", Index)  # 保留旧压缩格式的回归覆盖
    texts = corpus()
    memory = duplicate.Duplicate(duplicate.Rule(threshold=0.8))
    want = [memory.check(text) for text in texts]
    got = []
    for name, pack in (("pack", True), ("raw", False)):  # 两种模式各跑一遍，都跟内存版比
        monkeypatch.setattr(index_mod, "COLD_PACK", pack)
        disk = duplicate.Duplicate(duplicate.Rule(threshold=0.8), store=tmp_path / name, hot=3)
        disk._index.per_shard = 4  # 片缩到四条、温层只留八条，几百条语料一路转冷
        disk._index.warm = 8
        got.append([disk.check(text) for text in texts])
        disk.close()
    assert got[0] == want
    assert got[1] == want
    packed = sum(one.stat().st_size for one in (tmp_path / "pack").glob("cold-*.rec"))
    raw = sum(one.stat().st_size for one in (tmp_path / "raw").glob("cold-*.rec"))
    assert raw > packed  # 不压确实原样存，占盘更多


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
