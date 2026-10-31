"""落盘索引：把「分带键 → 下标」这层映射从进程内存挪到磁盘上，百万级语料也不会把人撑爆。

调用示例：
  from pathlib import Path
  from features.index import Index

  idx = Index()                                  # 纯内存，跟没有这个类时一样
  idx.put(b"某一分带键", 7, sig, windows)          # 7 号样本的这个键进索引
  idx.get(b"某一分带键")                          # [7]，可能重复，调用方用集合收
  idx.record(7)                                  # (签名, 窗口集合)，回算真实重合度用

  idx = Index(Path("bolt/index"))                # 落盘：同样四件事，热层满了自己往盘上挤
  idx.put(b"某一分带键", 8)                       # 不给签名和窗口就只做键→下标
  idx.save()                                     # 落盘；换一个进程用同一条路径构造就自动读回来
  idx.out()                                      # 把热层挤干净，挤出去的东西照样查得到

接口：put 登记、get 取候选下标、out/save 落盘与读回、size 报已收条数；
另有一个 record 按下标读回签名与窗口，回算真实重合度要用，落盘的记录靠它按需取。

三层，越往下越省内存、越慢：
  热层  内存里的 LRU，默认 5 万条键，查得最快；装不下就从头成批挤出去。
  温层  每片 20 万条的顺序文件，记录靠偏移表按需映射读，写只往片尾追加；键走盘上哈希桶。
  冷层  温层攒够一个上限，最老的片转成压缩片：签名和窗口压进文件，内存只留第一条键。

为什么分层：占内存的不是语料条数，是热层条数。一百万条语料的热层照样只有 5 万。
为什么冷层只留第一条键：一条样本 32 个分带键，全留在内存就白分了；留最前面那条，
撞上就算候选，漏掉的那些温层还兜着，冷层只是最后的粗筛。
为什么用内存映射：温层是按偏移跳着读的，mmap 切片比拼一次 seek 再 read 少一次拷贝。
为什么键表是哈希桶不是排序段：排序段查一次要把所有段各二分一遍，实测 6 万次查要 12 秒；
哈希桶查一次只翻一两条链，同样的次数一秒出头。
为什么位置表压成一个数：一条样本一行位置，两个 8 字节数组是 16MB/百万条，
压成「片号 << 21 | 片内位置」一个数只要 8MB，片内位置 2^21 远大于一片的 20 万条。
"""

import json
import mmap
import shutil
import zlib
from array import array
from collections import OrderedDict
from pathlib import Path

HOT = 50_000  # 热层默认能装多少条键。这个数就是内存峰值，别随手调大
SHARD = 200_000  # 温层一片多少条记录。片小一点，转冷时一次处理的量就小
WARM = 1_000_000  # 温层最多留多少条记录，超了最老的片转冷
POS_BITS = 21  # 位置表里，片内位置占多少位
POS_MASK = (1 << POS_BITS) - 1  # 从位置表里抠回片内位置的掩码
NOWHERE = (1 << 64) - 1  # 位置表里的空值：这条还在内存里，盘上没有位置
MANIFEST = "index.json"  # 落盘的清单文件名，加载时按它认目录
EVICT = 20  # 热层满了每次挤掉容量的几分之一；一条一条挤太零碎
BUCKET = 8  # 桶数取一片能装多少条键的几倍。桶多撞车少，桶本身也要占文件


class Live:
    """还在内存里的一条记录。热层里它还剩几条键，一条都不剩了才准离开内存。"""

    __slots__ = ("sig", "windows", "first", "keys")

    def __init__(self, sig, windows, first) -> None:
        self.sig = sig or []  # 指纹；只做键→下标时可以没有
        self.windows = windows or set()  # 窗口集合，回算真实重合度用
        self.first = first  # 第一条键，转冷后靠它粗筛
        self.keys = 0  # 热层里还剩几条键指着它


class Shard:
    """一片盘上的文件。

    温层：`warm-NNNN.rec` 顺序记记录（每条带长度头），`warm-NNNN.bkt` 是哈希桶，
    `warm-NNNN.key` 是键条目（键 + 下标 + 同桶的下一条）。
    冷层：`cold-NNNN.rec` 同样的记录格式，但每条单独 zlib 压缩；键表不要了，
    只在内存里留第一条键 → 下标，落到 `cold-NNNN.first`。
    """

    def __init__(self, store: Path, serial: int, layer: str, key_size: int, buckets: int = 0) -> None:
        self.serial = serial  # 全库唯一的片号，位置表里存的就是它
        self.layer = layer  # "warm" 还是 "cold"
        self.key_size = key_size  # 键固定多长，键条目按这个宽度切
        self.buckets = buckets  # 桶数；冷层为 0
        self.entry = key_size + 16  # 键条目的宽度：键 + 下标 + 同桶下一条
        self.rec = store / f"{layer}-{serial:04d}.rec"
        self.key = store / f"warm-{serial:04d}.key" if layer == "warm" else None
        self.bkt = store / f"warm-{serial:04d}.bkt" if layer == "warm" else None
        self.first = store / f"cold-{serial:04d}.first"  # 冷层粗筛表（加载时才读）
        self.offsets = array("Q")  # 第 i 条在 rec 里的字节偏移
        self.cold_keys = {}  # 冷层粗筛：第一条键 → 下标们
        self.count = 0  # 这片收了多少条
        self._out = self.rec.open("a+b")  # 记录只往片尾追加；a+ 才能给 mmap 当底座
        self._keys_out = self.key.open("a+b") if self.key else None
        self._rec_bytes = 0  # 记录文件写了多少字节，不每次都去问系统
        self._key_bytes = 0  # 键条目文件写了多少字节
        self._rec_map = None  # 记录文件的映射，读到才铺
        self._key_map = None  # 键条目文件的映射，读到才铺
        self._rec_span = 0  # 记录映射铺到多大
        self._key_span = 0  # 键条目映射铺到多大
        self._buckets = None  # 桶表的读写映射，开片时按桶数一次铺好
        if self.buckets:
            self._bkt = self.bkt.open("a+b")
            self._bkt.truncate(self.buckets * 8)  # 桶表定长，开头就占住
            self._buckets = mmap.mmap(self._bkt.fileno(), self.buckets * 8)

    def append(self, body: bytes) -> int:
        """追加一条记录，返回片内位置。冷层的 body 进文件前先压缩。"""
        blob = zlib.compress(body) if self.layer == "cold" else body  # 冷层压一把
        at = self._rec_bytes  # 这条落在片尾
        self._out.write(len(blob).to_bytes(4, "big") + blob)  # 长度打头，读的时候才知道读多长
        self._rec_bytes = at + 4 + len(blob)
        self.offsets.append(at)  # 偏移记下来，供按下标回读
        self.count += 1
        return self.count - 1

    def read(self, pos: int) -> bytes:
        """按片内位置读回一条记录，冷层顺带解压。"""
        handle = self._mapped_records()  # 映射盖到片尾
        at = self.offsets[pos]  # 这条的起点
        size = int.from_bytes(handle[at : at + 4], "big")  # 长度打头，先读它
        blob = handle[at + 4 : at + 4 + size]  # 再按长度切片
        return zlib.decompress(blob) if self.layer == "cold" else blob

    def bump(self, key: bytes, sub: int) -> None:
        """把「键 → 下标」挂进桶表。同一个桶里的条目串成一条链。"""
        if not self.buckets:  # 冷层没有键表
            return
        where = zlib.crc32(key) % self.buckets  # crc32 把整个键混一遍；只取前 8 字节会撞在一起
        at = self._key_bytes  # 条目落在条目文件尾部
        head = self._bucket(where)  # 桶里原来那串的头
        self._keys_out.write(key + sub.to_bytes(8, "big") + head.to_bytes(8, "big"))  # 新条目接在最前
        self._key_bytes = at + self.entry
        self._buckets[where * 8 : where * 8 + 8] = (at // self.entry + 1).to_bytes(8, "big")  # 桶改指新条目

    def find(self, key: bytes) -> list:
        """找键，返回落在它上面的下标。顺着桶里的链一条条比。"""
        if not self.buckets:
            return []
        handle = self._mapped_keys()  # 条目文件的映射
        if handle is None:  # 键表还是空的
            return []
        found = []
        at = self._bucket(zlib.crc32(key) % self.buckets)  # 这个桶里那串的头
        while at:  # 0 表示链到头
            base = (at - 1) * self.entry  # 条目号转字节位置
            if handle[base : base + self.key_size] == key:  # 键一样才算命中
                found.append(int.from_bytes(handle[base + self.key_size : base + self.key_size + 8], "big"))
            at = int.from_bytes(handle[base + self.key_size + 8 : base + self.entry], "big")  # 顺着链往下
        return found

    def _bucket(self, where: int) -> int:
        """桶里那串的头（条目号 + 1）。0 就是空桶。"""
        at = where * 8
        return int.from_bytes(self._buckets[at : at + 8], "big")

    def _mapped_records(self):
        """记录文件的只读映射，盖不住片尾就重铺。"""
        if self._rec_map is None or self._rec_bytes > self._rec_span:
            self._out.flush()  # 新写的内容先落盘，映射才看得到
            return self._remap("rec")
        return self._rec_map

    def _mapped_keys(self):
        """键条目文件的只读映射，盖不住表尾就重铺。空文件返回 None。"""
        if self._key_bytes == 0:  # 一条键都还没写
            return None
        if self._key_map is None or self._key_bytes > self._key_span:
            self._keys_out.flush()
            return self._remap("key")
        return self._key_map

    def _remap(self, which: str) -> mmap.mmap:
        """按文件当前长度铺一份映射。长过文件的映射有些系统不收，只能铺到头。"""
        if which == "rec":
            if self._rec_map is not None:
                self._rec_map.close()
            self._rec_span = self._rec_bytes
            self._rec_map = mmap.mmap(self._out.fileno(), self._rec_span, access=mmap.ACCESS_READ)
            return self._rec_map
        if self._key_map is not None:
            self._key_map.close()
        self._key_span = self._key_bytes
        self._key_map = mmap.mmap(self._keys_out.fileno(), self._key_span, access=mmap.ACCESS_READ)
        return self._key_map

    def flush(self) -> None:
        """把缓冲着的都写出去。复制给别人之前必须调。"""
        self._out.flush()
        if self._keys_out is not None:
            self._keys_out.flush()
        if self._buckets is not None:
            self._buckets.flush()

    def close(self) -> None:
        """关映射、关文件。"""
        for one in (self._rec_map, self._key_map, self._buckets):
            if one is not None:
                one.close()
        self._rec_map = self._key_map = self._buckets = None
        self._out.close()
        if self._keys_out is not None:
            self._keys_out.close()
        if self.buckets:
            self._bkt.close()


class Index:
    """分带键 → 下标的映射。一次 weave 用一个实例。

    store 给 None 就是纯内存：热层不淘汰，记录全留在内存里，行为和没有这个类时一样。
    """

    def __init__(self, store=None, hot: int = HOT, per_shard: int = SHARD, warm: int = WARM, key_size: int = 32) -> None:
        self.store = Path(store) if store else None  # None 就是纯内存
        self.hot = hot  # 热层最多几条键
        self.per_shard = per_shard  # 温层一片几条记录
        self.warm = warm  # 温层总共最多几条记录
        self.key_size = key_size  # 键固定多长，分带键是 4 字节 × 段内行数
        self.count = 0  # 已经登记过多少条样本

        self._live = {}  # 下标 → Live，还在内存里的记录
        self._keys = {}  # 键 → 下标们，热层
        self._order = OrderedDict()  # 键的插入顺序，从头就是最旧的那条，淘汰从头取
        self._where = array("Q")  # 下标 → 盘上位置（片号 << 21 | 片内位置），纯内存时不动
        self._shards = {}  # 片号 → Shard
        self._warm = []  # 温层片，按新旧排，末尾最新
        self._cold = []  # 冷层片
        self._warm_rows = 0  # 温层现在有多少条记录
        self._serial = 0  # 片号发号器

        if self.store is not None:  # 落盘模式：目录先建好，有旧清单就读回来
            self.store.mkdir(parents=True, exist_ok=True)
            if (self.store / MANIFEST).exists():
                self._load()

    # --- 登记 ---

    def put(self, key: bytes, sub: int, sig: list = None, windows: set = None) -> None:
        """登记「这个键上有这个下标」。同一条样本的各个键各调一次。

        sig 和 windows 只在第一次见到 sub 时用上，之后忽略：它们就是被索引的那条记录本身。
        """
        if len(key) != self.key_size:
            raise ValueError(f"键要正好 {self.key_size} 字节，拿到 {len(key)} 字节")
        self._touch(key, sub, sig, windows)
        if self.store is not None and len(self._order) > self.hot:  # 热层满了，成批挤最旧的出去
            self.out(max(1, self.hot // EVICT))

    def _touch(self, key: bytes, sub: int, sig, windows) -> None:
        """把键和下标挂上，第一次见到的下标顺手建记录。"""
        record = self._live.get(sub)
        if record is None:  # 新样本，先建记录
            record = self._live[sub] = Live(sig, windows, key)
            self.count += 1
            if self.store is not None:  # 位置表按下标铺开，中间的洞先填空值
                while len(self._where) <= sub:
                    self._where.append(NOWHERE)
        record.keys += 1  # 这条样本多了一条键指着它
        if key in self._order:  # 老键，接在后面
            self._keys[key].append(sub)
        else:  # 新键，记进插入顺序
            self._keys[key] = [sub]
            self._order[key] = None

    # --- 查候选 ---

    def get(self, key: bytes) -> list:
        """拿这个键上的候选下标。同一个下标可能出现多次，调用方用集合收。"""
        found = list(self._keys.get(key, ()))  # 热层
        for shard in reversed(self._warm):  # 温层各片，新的先找
            found += shard.find(key)
        for shard in self._cold:  # 冷层只剩第一条键，撞上就算粗筛候选
            found += shard.cold_keys.get(key, ())
        return found

    def record(self, sub: int) -> tuple:
        """按下标读回（签名，窗口集合）。还在内存就直接给，落了盘就跳着读回来。"""
        record = self._live.get(sub)
        if record is not None:  # 还在内存
            return record.sig, record.windows
        packed = self._where[sub]  # 盘上位置
        body = self._shards[packed >> POS_BITS].read(packed & POS_MASK)  # 片号 + 片内位置，读回来
        return _unpack(body, self.key_size)[2:]  # 取签名和窗口，前两样是下标和第一条键

    def size(self) -> int:
        """已经收了多少条样本。"""
        return self.count

    # --- 淘汰与落盘 ---

    def out(self, count: int = None) -> None:
        """把热层最旧的 count 条键挤到温层，不给 count 就挤干净。

        挤出去的键照样查得到：条目进键表，记录跟着样本走，最后一条键走了记录才落盘。
        """
        if self.store is None:  # 纯内存没有可挤的地方
            return
        count = len(self._order) if count is None else count
        while count > 0 and self._order:
            key, _ = self._order.popitem(last=False)  # 最旧的那条，C 实现，不绕迭代器
            subs = self._keys.pop(key)
            shard = self._shard()  # 挤到当前这片
            for sub in subs:  # 键落到盘上，热层不再留
                shard.bump(key, sub)
            for sub in subs:  # 每条挂着它的样本少一条键
                record = self._live[sub]
                record.keys -= 1
                if record.keys == 0:  # 一条键都不剩了，记录才准离开内存
                    self._park(sub, record)
                    del self._live[sub]
            count -= 1

    def save(self, directory=None) -> Path:
        """落盘。directory 给了就写到那儿，不给就写自己的存储目录。

        落的是整个索引：记录、键表、冷层粗筛表全在。换一个进程构造同一个目录就全部回来。
        """
        if self.store is None:
            raise ValueError("纯内存索引落不了盘，构造时给 store 目录")
        target = Path(directory) if directory is not None else self.store
        self.out()  # 热层挤干净，内存里的东西全写到盘上
        self.store.mkdir(parents=True, exist_ok=True)
        if target != self.store:  # 另存一份，目标目录先建出来
            target.mkdir(parents=True, exist_ok=True)
        shards = []
        for shard in list(self._shards.values()):  # 清单里只记位置和形状，记录本身在文件里
            shard.flush()  # 先把缓冲写出去，否则复制到的是半截文件
            if shard.layer == "cold":  # 冷层粗筛表跟片一起落
                self._put_first(shard)
            if target != self.store:  # 换目录就复制一份，原目录保持不动
                for one in (shard.rec, shard.key, shard.bkt, shard.first):
                    if one is not None and one.exists():
                        shutil.copyfile(one, target / one.name)  # 分块复制，文件再大也不整块读进内存
            shards.append({"serial": shard.serial, "layer": shard.layer, "count": shard.count, "buckets": shard.buckets})
        manifest = {
            "version": 1,  # 清单格式版本
            "key_size": self.key_size,
            "hot": self.hot,
            "per_shard": self.per_shard,
            "warm": self.warm,
            "count": self.count,
            "shards": shards,
        }
        text = json.dumps(manifest, indent=2)
        (target / MANIFEST).write_text(text, encoding="utf-8")  # 清单写目标目录
        if target != self.store:  # 自己那份也写，两个目录都读得回来
            (self.store / MANIFEST).write_text(text, encoding="utf-8")
        return target

    def close(self) -> None:
        """关掉全部文件。关之前不落盘的话，热层里剩的东西就没了。"""
        for shard in self._shards.values():
            shard.close()

    # --- 挤出去以后怎么落 ---

    def _park(self, sub: int, record: Live) -> None:
        """把一条记录写到盘上，位置表记下它在哪。"""
        shard = self._shard()
        pos = shard.append(_pack(sub, record.first, record.sig, record.windows))
        self._where[sub] = (shard.serial << POS_BITS) | pos
        self._warm_rows += 1
        if shard.count >= self.per_shard:  # 这片满了，换新片
            self._open_warm()
        if self._warm_rows > self.warm:  # 温层攒太多了，最老的片转冷
            self._to_cold(self._warm[0])

    def _shard(self) -> Shard:
        """当前该写的温层片，末尾就是；一片都没有就开一片。"""
        if not self._warm:
            self._open_warm()
        return self._warm[-1]

    def _open_warm(self) -> Shard:
        """开一片新的温层。"""
        self._serial += 1
        shard = Shard(self.store, self._serial, "warm", self.key_size, max(1024, self.per_shard * BUCKET))
        self._shards[shard.serial] = shard
        self._warm.append(shard)
        return shard

    def _to_cold(self, shard: Shard) -> None:
        """把一片温层转成冷层：记录压起来，键表只要第一条键。"""
        self._warm.remove(shard)
        self._warm_rows -= shard.count
        self._serial += 1
        cold = Shard(self.store, self._serial, "cold", self.key_size)
        self._shards[cold.serial] = cold
        self._cold.append(cold)
        for pos in range(shard.count):  # 逐条搬过去，位置表跟着改
            body = shard.read(pos)
            sub, first = _unpack(body, self.key_size)[:2]
            at = cold.append(body)
            cold.cold_keys.setdefault(first, []).append(sub)  # 粗筛只留第一条键
            self._where[sub] = (cold.serial << POS_BITS) | at
        shard.close()
        for one in (shard.rec, shard.key, shard.bkt):  # 键表随片一起不要了
            if one is not None:
                one.unlink()
        del self._shards[shard.serial]

    def _put_first(self, shard: Shard) -> None:
        """冷层粗筛表落盘：第一条键 + 下标。"""
        with shard.first.open("wb") as out:
            for key, subs in shard.cold_keys.items():
                for sub in subs:
                    out.write(key + sub.to_bytes(8, "big"))

    # --- 读回来 ---

    def _load(self) -> None:
        """照清单把索引读回来。记录文件扫一遍重建位置表和下标。

        热层容量、片大小这些跟着清单走：接着上一次的设置继续跑，不会突然换一套。
        """
        manifest = json.loads((self.store / MANIFEST).read_text(encoding="utf-8"))
        if manifest["version"] != 1:  # 格式对不上就别硬读
            raise ValueError(f"索引清单版本 {manifest['version']} 不认识")
        self.key_size = manifest["key_size"]
        self.hot = manifest["hot"]
        self.per_shard = manifest["per_shard"]
        self.warm = manifest["warm"]
        self.count = manifest["count"]
        for one in manifest["shards"]:  # 按清单把片开回来
            shard = Shard(self.store, one["serial"], one["layer"], self.key_size, one["buckets"])
            shard._key_bytes = shard.key.stat().st_size if shard.key else 0
            self._serial = max(self._serial, one["serial"])
            self._shards[shard.serial] = shard
            if one["layer"] == "warm":
                self._warm.append(shard)
                self._warm_rows += one["count"]
            else:
                self._cold.append(shard)
                self._read_first(shard)
            self._scan(shard)  # 扫一遍记录，位置表和下标对齐
            shard.count = one["count"]

    def _scan(self, shard: Shard) -> None:
        """扫一片的记录：先把偏移全记下，再逐条解出下标填回内存。"""
        at = 0
        with shard.rec.open("rb") as handle:  # 第一趟只看长度头，不解压，省内存
            while True:
                size = int.from_bytes(handle.read(4), "big")
                if size == 0:  # 长度头都没读到，到底了
                    break
                shard.offsets.append(at)  # 偏移先记下
                at += 4 + size
                handle.seek(at)
        shard._rec_bytes = at  # 文件就这么长，之后追加从这儿接着算
        for pos in range(len(shard.offsets)):  # 第二趟才解记录，只要前 8 字节的下标
            sub = int.from_bytes(shard.read(pos)[:8], "big")
            while len(self._where) <= sub:  # 位置表按下标铺开
                self._where.append(NOWHERE)
            self._where[sub] = (shard.serial << POS_BITS) | pos

    def _read_first(self, shard: Shard) -> None:
        """把冷层粗筛表读回内存。"""
        raw = shard.first.read_bytes()
        step = self.key_size + 8
        for at in range(0, len(raw), step):
            key = raw[at : at + self.key_size]
            shard.cold_keys.setdefault(key, []).append(int.from_bytes(raw[at + self.key_size : at + step], "big"))


def _pack(sub: int, first: bytes, sig: list, windows) -> bytes:
    """一条记录压成字节：下标、第一条键、签名、窗口集合。"""
    return sub.to_bytes(8, "big") + first + _words(sig) + _words(windows)


def _words(values) -> bytes:
    """一串 32 位整数按大端拼成字节，前头带上个数。逐个 to_bytes 太慢，走 array 一把过。"""
    block = array("I", values)
    block.byteswap()  # array 按本机字节序，换成大端存
    return len(values).to_bytes(4, "big") + block.tobytes()


def _read_words(body: bytes, at: int) -> tuple:
    """从 at 读回一串整数和下一个位置。"""
    count = int.from_bytes(body[at : at + 4], "big")
    block = array("I")
    block.frombytes(body[at + 4 : at + 4 + 4 * count])
    block.byteswap()
    return block, at + 4 + 4 * count


def _unpack(body: bytes, key_size: int) -> tuple:
    """字节拆回（下标，第一条键，签名，窗口集合）。"""
    sub = int.from_bytes(body[:8], "big")
    first = body[8 : 8 + key_size]
    sig, at = _read_words(body, 8 + key_size)
    windows, at = _read_words(body, at)
    return sub, first, list(sig), set(windows)

