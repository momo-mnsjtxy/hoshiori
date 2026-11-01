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
  idx.settle()                                   # 等后台把排着队的片都转冷（save 和 close 自己会等）
  idx.watch = lambda n: ...                      # 转冷途中每走一步报一下内存里还压着几条记录

接口：put 登记、get 取候选下标、out/save 落盘与读回、size 报已收条数；
另有一个 record 按下标读回签名与窗口，回算真实重合度要用，落盘的记录靠它按需取。

三层，越往下越省内存、越慢：
  热层  内存里的 LRU，默认 5 万条键，查得最快；装不下就从头成批挤出去。
  温层  每片 50 万条的顺序文件，记录靠偏移表按需映射读，写只往片尾追加；
        键走盘上哈希桶，桶表本身也按片落在盘上，内存里只留几百个最近碰过的桶；
        键条目本身按条切成 4096 条一片的 `warm-NNNN.ketSSSS`，按条号 pread，不整份映射。
  冷层  温层攒够一个上限，最老的片转成压缩片：整片逐条流式搬，一条读完压完写完就放开，
        原片文件当场删掉（不留空壳占 inode）；每条的全部键按桶串进 `cold-NNNN.chk`
        （桶头在文件开头定长区，条目顺序追加在尾部），查的时候只顺着一个桶的链走。

键表为什么按条分片：键条目是 4096 条一片（KEY_SLICE），一条 32 字节，一片才 128KB。
一片五十万条温层的键表是 16MB，原来整份 mmap 铺着，映射过的页一直算在常驻内存里。
切成小文件以后按条号算片号、按条号取余算片内偏移，pread 读哪条是哪儿条，整片不用铺。
冷片为什么单文件顺序追加：原来一条记录要把键按桶 pread/pwrite 散到几千个桶文件上，
一批两千条就是几千次开文件加几千次 seek；改成一个冷片一个 `.chk`，条目只管往尾部 append，
桶头在开头那块定长区里、全片只有几 KB，攒一批整块写回一次。
查询为什么只读一条链：桶头直接读内存里那张表，顺着链 pread 它自己的条目，别的桶一个字节不碰。

转冷在后台线程里做：`_park` 只把「这片该转冷了」塞进队列就返回，主流程接着登记；
真压 zlib、删原片都由后台一片一片来，主流程不陪着等盘。close/save 会先把队列排空。

为什么分层：占内存的不是语料条数，是热层条数。一百万条语料的热层照样只有 5 万。
为什么冷层留全部键：原来只留第一条键，被转冷的老片第二条以后的键全撞不上，阈值一调低召回就掉；
全键按桶 zlib 压着，一段键表落地就从内存丢掉，盘上贵一点，内存不涨。
为什么冷层桶要分得密：链是一条条 pread 出来的，链长就是每条键的读次数。桶按每四条键一个分，
一条链只有几条，查一次只有几回 pread；桶稀了链就几千条，一条记录几十个键能查上好几秒。
为什么冷层要键集求交：桶按 crc32 分，桶里还落着别的记录的键，撞上同一个桶不等于撞上键；
条目的键集和记录里的键集是同一份，求交对上了才交出下标，误报交不到下游。
为什么偏移表放盘上：一片 50 万条 × 8 字节就是 4MB，冷层留几百片光偏移就白分了内存；
按条追加进 `.off`，读时 mmap 切一刀只碰要读的那 8 字节。
为什么用内存映射：温层是按偏移跳着读的，mmap 切片比拼一次 seek 再 read 少一次拷贝。
为什么桶表也要落盘：一片 50 万条的桶表是 32MB，温层留几片就是几十上百 MB。原来整份铺成
映射，映射过的页会一直算在常驻内存里，百万条实测把这个进程顶到 2GB（内存比从 10.1 掉到 7.9）。
改成按桶切片落盘、内存只留最近碰过的几百个桶，这部分从几十 MB 压到几百个整数。
为什么桶片里用 pread/pwrite 而不是映射：桶是 8 字节的链头，读一次写一次就完，铺映射会把页
重新顶回常驻内存；桶片数按片算下来几十个，一个温片（1M 上限两片）一百多个文件，一直在
1024 的 fd 上限里头。
为什么键表是哈希桶不是排序段：排序段查一次要把所有段各二分一遍，实测 6 万次查要 12 秒；
哈希桶查一次只翻一两条链，同样的次数一秒出头。
为什么键表条目不压缩：一条记录一个桶一般只落一条键，条目块十六七字节，zlib 压完反而更长；
块攒到 COLD_KEY_PACK 字节以上才压，压出来确实短了才用，长的不留压过的。
为什么冷片桶头留在内存：一片最多 MAX_BUCKET 个桶，四字节一个也就 8KB，比每条键一次 pwrite
便宜得多；攒一批整块写回开头那块定长区，读的那边连桶头都不用读盘。
为什么位置表压成一个数：一条样本一行位置，两个 8 字节数组是 16MB/百万条，
压成「片号 << 21 | 片内位置」一个数只要 8MB，片内位置 2^21 远大于一片的 50 万条。
"""

import json
import mmap
import os
import shutil
import threading
import zlib
from array import array
from collections import OrderedDict, deque
from pathlib import Path

HOT = 50_000  # 热层默认能装多少条键。这个数就是内存峰值，别随手调大
SHARD = 500_000  # 温层一片多少条记录。为什么不比这更小：每片转一次冷就要整片压一遍再删，
                 # 二十万条时一百万条语料要转五次，转冷次数直接决定 IO 总量；片放大到五十万，
                 # 转冷次数少一多半，而单片的内存（键表在盘上、只留几千条一批的键缓冲）没变。
                 # 也不比这更大：单片搬完要几十秒，片再大后台就跟不上主流程，队列会越积越长。
WARM = 1_000_000  # 温层最多留多少条记录，超了最老的片转冷
POS_BITS = 21  # 位置表里，片内位置占多少位
POS_MASK = (1 << POS_BITS) - 1  # 从位置表里抠回片内位置的掩码
NOWHERE = (1 << 64) - 1  # 位置表里的空值：这条还在内存里，盘上没有位置
MANIFEST = "index.json"  # 落盘的清单文件名，加载时按它认目录
EVICT = 20  # 热层满了每次挤掉容量的几分之一；一条一条挤太零碎
BUCKET = 16  # 温层桶数取一片能装多少条键的几倍。桶多撞车少，桶本身也要占文件：
             # 桶是查一次键就要走一遍的链头，链长一半、查就快一半；一片五十万条是 800 万桶
             # × 8 字节 = 64MB，比 8 的 32MB 贵一倍，换来的是别把时间花在链上
MIN_BUCKET = 1024  # 冷层一片最少几个桶。桶太少，每条链都要走很长
MAX_BUCKET = 4_000_000  # 冷层一片最多几个桶。再多，桶头区自己就比键还占地方
PER_BUCKET = 4  # 一个桶平均装几条键，按片里的键数反推桶数。链长就是这个数：
                # 查一次冷层只要按桶号 pread 一个桶头再走几条，链长十几条时一条记录三十多个键
                # 光查就能吃掉几十毫秒
BUCKET_TABLE = 32_000_000  # 一片的桶表多大就多重：五十万条 × 8（BUCKET）= 400 万个桶 × 8 字节
                           # = 32MB。这是温层唯一一块按片大小定的内存，片一搬完就放掉
KET_KEEP = 256  # 全库共用一个键片句柄池。共用是为了跟片数脱钩：片再多也只开这么多
                 # 只当上限用（fd 上限五十多万，几片全开也就一万多）

KEY_SLICE = 4096  # 温层键条目一片多少条。一条 16~48 字节，一片一两百 KB：
                  # 小到 pread 一条不浪费，又不会像整份映射那样把页顶进常驻内存。
                  # 是 2 的幂：条号算片号、算片内位置走移位，比 divmod 快一倍
KEY_SHIFT = 12  # KEY_SLICE 是 2 的几次方，条号右移它就是片号
KEY_MASK = KEY_SLICE - 1  # 条号按位与它就是片内第几条
COLD_KEY_PACK = 64  # 冷层一个桶的键块攒到多少字节才值得压。一条记录一个桶一般只落一条键，
                    # 十六七字节压完反而更长，块大了才压（还没压短就仍原样存）
COLD_HEAD = 15  # 冷层一条条目的头：下一条 4 + 下标 4 + 键数 2 + 压没压 1 + 块长 4。
               # 下一条和下标都用 4 字节：一片冷片的链文件也就几百 MB，装得下 4G 偏移
COLD_BATCH = 2_000  # 转冷时每搬多少条就把攒着的桶头写回一次。批的大小见 _to_cold 里的说明
COLD_PACK = True  # 冷片记录压不压缩。为假就原样存：省掉整片的 zlib，占盘翻几倍；
                  # 测试和压测脚本可以换成假跑一遍（读回时按清单记的那个值来）
COLD_WAIT = 120.0  # settle 最多等几秒。一片五十万搬完要几十秒，再留一倍的余量


class Live:
    """还在内存里的一条记录。热层里它还剩几条键，一条都不剩了才准离开内存。"""

    __slots__ = ("sig", "windows", "keys", "left")

    def __init__(self, sig, windows, key) -> None:
        self.sig = sig or []  # 指纹；只做键→下标时可以没有
        self.windows = windows or set()  # 窗口集合，回算真实重合度用
        self.keys = bytearray(key)  # 这条的全部键，按登记顺序定长拼着，转冷后靠它留全键
        self.left = 0  # 热层里还剩几条键指着它，归零才准落盘


class BucketTable:
    """温层的哈希桶表：桶号 → 链头，整片放在内存一个数组里，只在落盘和读回的时候碰文件。

    为什么放内存：查一次键要一个桶头，桶头一次 pread 就是每条键一次系统调用；一片五十万条
    是四百万个桶，一个桶头八字节也就是三十二兆，换掉每条键两次系统调用划得来。
    为什么不用映射：映射过的页算进常驻内存，跟没释放一样；这里就是堆上的一格定长数组，
    大小只跟片有关、不随语料涨。
    盘上还是 `warm-NNNN.bkt` 一个文件，只整块读、整块写，建片、落盘、读回来的时候才动它。
    """

    __slots__ = ("path", "heads")

    def __init__(self, store: Path, serial: int, buckets: int) -> None:
        self.path = store / f"warm-{serial:04d}.bkt"  # 桶头落盘就这一个文件
        self.heads = array("Q", bytes(8 * buckets))  # 桶号 → 链头（条目号 + 1），全零就是空桶

    def get(self, where: int) -> int:
        """桶里那串的头（条目号 + 1）。0 就是空桶；桶表已经放掉就当空桶。"""
        return 0 if self.heads is None else self.heads[where]

    def set(self, where: int, head: int) -> None:
        """改桶的头。"""
        self.heads[where] = head

    def files(self) -> list:
        """桶表落过盘就返回那个文件。落盘复制和转冷后删除都按它走。"""
        return [self.path] if self.path.exists() else []

    def load(self) -> None:
        """读回来的片：桶头整块从盘上抄回内存。"""
        block = array("Q")
        block.frombytes(self.path.read_bytes())
        block.byteswap()  # 盘上是大端
        self.heads = block

    def flush(self) -> None:
        """把整片桶头写出去。转过冷、数组已经放掉的片不用再写。"""
        if self.heads is None:  # 这片搬完了，桶表在盘上已经没人看
            return
        block = array("Q", self.heads)
        block.byteswap()  # 盘上用大端，换台机器也一样
        self.path.write_bytes(block.tobytes())

    def drop(self) -> None:
        """转冷搬完把数组放掉：这片没有活人再按桶查了。"""
        self.heads = None

    def close(self) -> None:
        """关片时调一下。数组没有句柄要关，留着这个口子给关片的地方按顺序走。"""


_KET_CACHE = OrderedDict()  # 路径 → 打开着的键片文件
_KET_LOCK = threading.Lock()  # 护着上面那个池子


class Shard:
    """一片盘上的文件。

    温层：`warm-NNNN.rec` 顺序记记录（每条带长度头），`warm-NNNN.off` 是按条追加的偏移表，
    `warm-NNNN.bktSSSS` 是分片的哈希桶，`warm-NNNN.ketSSSS` 是键条目
    （键 + 下标 + 同桶的下一条），一片 KEY_SLICE 条，按条号 pread。
    冷层：`cold-NNNN.rec` 同样的记录格式，但每条单独 zlib 压缩（COLD_PACK 为假时就原样存）；偏移表照旧；
    `cold-NNNN.chk` 一个文件装全部桶：开头 buckets 个四字节桶头，条目顺序追加在尾部，
    一条条目是一个桶里一群键（下一条 4 + 下标 4 + 键数 2 + 压没压 1 + 块长 4 + 键块）。
    """

    def __init__(self, store: Path, serial: int, layer: str, key_size: int, buckets: int = 0) -> None:
        self.serial = serial  # 全库唯一的片号，位置表里存的就是它
        self.layer = layer  # "warm" 还是 "cold"
        self.key_size = key_size  # 键固定多长，键条目按这个宽度切
        self.buckets = buckets  # 桶数；温层是哈希表宽度，冷层是 chk 里几个桶头
        self.entry = key_size + 16  # 温层键条目的宽度：键 + 下标 + 同桶下一条
        self.rec = store / f"{layer}-{serial:04d}.rec"
        self.off = store / f"{layer}-{serial:04d}.off"  # 按条追加的偏移表
        self.key = store / f"warm-{serial:04d}.ket" if layer == "warm" else None  # 键条目片名的前缀
        self.chk = store / f"cold-{serial:04d}.chk" if layer == "cold" else None  # 冷层桶链的单文件
        self.table = BucketTable(store, serial, buckets) if buckets and layer == "warm" else None
        self.count = 0  # 这片收了多少条
        self._out = self.rec.open("a+b")  # 记录只往片尾追加；a+ 才能给 mmap 当底座
        self._off_out = self.off.open("a+b")  # 偏移表同样只往后追加
        self._rec_bytes = 0  # 记录文件写了多少字节，不每次都去问系统
        self._off_bytes = 0  # 偏移表写了多少字节
        self._rec_map = None  # 记录文件的映射，读到才铺
        self._off_map = None  # 偏移表的映射，读到才铺
        self._rec_span = 0  # 记录映射铺到多大
        self._off_span = 0  # 偏移表映射铺到多大
        self._key_entries = 0  # 键表收了多少条条目，片号就是它除以 KEY_SLICE
        self._ket_out = None  # 正在写的那片键条目文件，只在片尾那条上
        self._ket_seg = (bytearray(), 0, 0)  # 当前片还没落盘的条目：(缓冲, 第一条的条号, 后一条的条号)。
                                            # 换片、落盘时整个换一份新的，读的那边捏着旧的那份照样是完整的一段

        self._heads = None  # 冷层正在搬时的桶头暂存：每个桶链头在文件里的偏移，搬完就丢
        self._chk = None  # 冷层桶链文件的句柄
        self._chk_bytes = 0  # 桶链文件写到哪了，条目就接在这儿
        self._lock = threading.Lock()  # 只护着映射与句柄的铺和撤；后台转冷会撤它们，主流程还在读
        if layer == "cold":  # 冷片一建出来就把桶头区按定长占好
            self._chk = os.open(self.chk, os.O_RDWR | os.O_CREAT)  # 不开 append：Linux 上 append
                                                                     # 的 pwrite 会写到文件尾，桶头就写不回开头
            size = os.fstat(self._chk).st_size  # 读回来的片已经有内容，别再按桶头区截短它
            os.ftruncate(self._chk, max(size, buckets * 4))  # 开头那块桶头区占够，条目接在它后面
            self._chk_bytes = max(size, buckets * 4)

    def append(self, body: bytes) -> int:
        """追加一条记录，返回片内位置。冷层的 body 进文件前先压缩（COLD_PACK 为假就原样存）。

        写和读抢同一把锁：缓冲里的字节还没落地，读的那边就会按刚加的字节数去铺映射，
        文件却还短着，mmap 直接报长度大于文件。
        """
        with self._lock:
            blob = zlib.compress(body) if self.layer == "cold" and COLD_PACK else body  # 冷层压一把
            at = self._rec_bytes  # 这条落在片尾
            self._out.write(len(blob).to_bytes(4, "big") + blob)  # 长度打头，读的时候才知道读多长
            self._off_out.write(at.to_bytes(8, "big"))  # 偏移按条追加，不留在内存里
            self._rec_bytes = at + 4 + len(blob)
            self._off_bytes += 8  # 偏移表跟着长一条
        self.count += 1
        return self.count - 1

    def read(self, pos: int) -> bytes:
        """按片内位置读回一条记录，冷层顺带解压。"""
        with self._lock:  # 后台可能正在撤这几份映射
            handle = self._mapped_records()  # 映射盖到片尾
            at = int.from_bytes(self._mapped_offsets()[pos * 8 : pos * 8 + 8], "big")  # 偏移表里取这条的起点
            size = int.from_bytes(handle[at : at + 4], "big")  # 长度打头，先读它
            blob = handle[at + 4 : at + 4 + size]  # 再按长度切片
        return zlib.decompress(blob) if self.layer == "cold" and COLD_PACK else blob

    def bump(self, keys: bytes, sub: int) -> None:
        """冷层把一条记录按桶摊开写：每个落到的桶把这条的键接到它的链最前。

        条目里只放这条记录落在这个桶的键，块长跟着是真实的键数——一条记录的键散在几个桶里，
        按总键数写块长，读的时候会读串。
        """
        if not self.buckets or self.layer != "cold":  # 温层的键走下面的 bump_key
            return
        spread = {}
        for at in range(0, len(keys), self.key_size):  # 键逐个摊到它自己的桶
            key = keys[at : at + self.key_size]
            spread.setdefault(zlib.crc32(key) % self.buckets, bytearray()).extend(key)  # 同桶的键攒一块
        for where, block in spread.items():  # 摊完了才写，块里的键数才是真数
            self._push(where, sub, block)

    def bump_key(self, key: bytes, sub: int) -> None:
        """温层把「键 → 下标」挂进桶表。同一个桶里的条目串成一条链。

        和 find 抢同一把锁：条目是写到文件尾部的，读的那边按条目数算位置会撞上写了一半的那条。
        """
        if not self.buckets:  # 冷层没有这种桶表
            return
        where = zlib.crc32(key) % self.buckets  # crc32 把整个键混一遍；只取前 8 字节会撞在一起
        with self._lock:
            head = self._bucket(where)  # 桶里原来那串的头
            at = self._key_push(key + sub.to_bytes(8, "big") + head.to_bytes(8, "big"))  # 新条目接在最前
            self.table.set(where, at // self.entry + 1)  # 桶改指新条目

    def find(self, key: bytes) -> list:
        """找键，返回落在它上面的下标。顺着桶里的链一条条比，只读链上那几条。"""
        if not self.buckets:
            return []
        found = []
        size = self.entry  # 一条条目多宽
        with self._lock:  # 后台转冷会关掉键表的句柄，主流程还在顺着链找
            at = self._bucket(zlib.crc32(key) % self.buckets)  # 这个桶里那串的头
            buf, first, end = self._ket_seg  # 这一段取在循环外，一次到齐，不怕别人换段
            while at:  # 0 表示链到头
                n = at - 1  # 条目号
                margin = n - first  # 离缓冲区第一条差几条
                if 0 <= margin < end - first:  # 还在缓冲区里，直接从内存切
                    base = margin * size
                    row = buf[base : base + size]
                else:  # 落过盘的那片，按片号和片内位置 pread 它自己这一条
                    row = os.pread(self._ket_use(n >> KEY_SHIFT), size, (n & KEY_MASK) * size)
                if row[: self.key_size] == key:  # 键一样才算命中
                    found.append(int.from_bytes(row[self.key_size : self.key_size + 8], "big"))
                at = int.from_bytes(row[self.key_size + 8 :], "big")  # 顺着链往下
        return found

    def candidates(self, key: bytes) -> list:
        """冷层找键：先取这个桶的头，只走这一条链，命中了才交出下标。

        桶头正在搬的时候在内存数组里（搬完才写盘、再把数组放掉），搬完以后按桶号 pread：
        搬的时候读内存，读到的就是当下最新的链头，不比盘上那批慢一拍。
        """
        found = []
        where = zlib.crc32(key) % self.buckets  # 和写的时候同一套散列，算得回来
        at = self._heads[where] if self._heads is not None else self._head_at(where)
        while at:  # 0 表示链到头
            head = os.pread(self._chk, COLD_HEAD, at)  # 只读这一条的头
            if len(head) < COLD_HEAD:  # 这条正写到一半，当它还没写完
                break
            size = int.from_bytes(head[11:COLD_HEAD], "big")  # 块多长
            block = os.pread(self._chk, size, at + COLD_HEAD)  # 再按块长读它自己的键
            if len(block) < size:  # 同上，半截的跳过
                break
            if head[10]:  # 压过的先解开
                block = zlib.decompress(block)
            if key in _split_keys(block, self.key_size):  # 键集求交：真在这条的键集里
                found.append(int.from_bytes(head[4:8], "big"))
            at = int.from_bytes(head[:4], "big")  # 顺着链往下
        return found

    def files(self) -> list:
        """这片在盘上占的文件。落盘复制和转冷后删除都按它走。"""
        ones = [self.rec, self.off]
        if self.layer == "warm":
            ones += sorted(self.rec.parent.glob(f"warm-{self.serial:04d}.ket*"))  # 键条目用几片建几片
            ones += self.table.files()  # 桶片
        else:
            ones.append(self.chk)  # 键全串在这一个文件里
        return [one for one in ones if one is not None and one.exists()]

    def _push(self, where: int, sub: int, block) -> None:
        """把一条条目接到桶链最前：条目顺序追加到文件尾，桶头改成它的偏移。

        桶头先只在内存那张表里改，攒一批由 flush 整块写回开头；每条键因此只要一次 write，
        不用再为桶头单独 pwrite 一次。
        块小了压出来反而更长（一条记录一个桶一般只落一条键），COLD_KEY_PACK 字节以上才试，
        压短了才用压过的。
        """
        flag = 0  # 这条存的是原样
        if COLD_PACK and len(block) >= COLD_KEY_PACK:  # 块够大才值得压
            short = zlib.compress(block, 1)  # 一级就够：键块没什么结构，压太狠白烧 CPU
            if len(short) < len(block):  # 真压短了才用
                block, flag = short, 1
        at = self._chk_bytes  # 这条落在文件尾
        head = self._heads[where].to_bytes(4, "big") + sub.to_bytes(4, "big")  # 头里存的下一条是桶原来的链头
        head += (len(block) // self.key_size).to_bytes(2, "big") + bytes([flag]) + len(block).to_bytes(4, "big")
        os.pwrite(self._chk, head + block, at)  # 一次写完，头带块，接在文件尾
        self._chk_bytes = at + COLD_HEAD + len(block)  # 下一条接在它后面
        self._heads[where] = at  # 桶头改成这条

    def _key_push(self, entry: bytes) -> int:
        """把一条键条目接到键表末尾，返回它的字节位置。

        条目按条号算片，片满了就换下一片，换片时把上一片一次写出去。
        为什么不在每条后面写一次：一片 4096 条正好一个块，攒着写是每片一次 pwrite，
        每条一次就是 4096 次系统调用；这批条目里的大头（刚写的那片）还能直接从内存读回来。
        """
        if self._key_entries % KEY_SLICE == 0:  # 上一片满了（或者还一片都没有），换下一片
            self._key_flush()  # 上一片缓冲里的先落地
            self._ket_seg = (bytearray(), self._key_entries, self._key_entries)
        buf, first, end = self._ket_seg  # 这一段就是当前片的尾巴
        at = self._key_entries * self.entry  # 条目号转字节位置，片内的位置也这么算
        buf += entry  # 先进缓冲，不留一次一次的系统调用
        self._ket_seg = (buf, first, end + 1)  # 段往后长一条
        self._key_entries += 1
        return at

    def _key_at(self, entry: int) -> bytes:
        """按条号读回一条键条目：还在缓冲里就直接切，落过盘的那片 pread 它自己那一块。

        片号按条号整除，片内位置按条号取余。
        """
        buf, first, end = self._ket_seg  # 现在这一段的缓冲和两头
        margin = entry - first  # 离缓冲区第一条差几条
        if 0 <= margin < end - first:  # 这条还在缓冲区里
            at = margin * self.entry
            return bytes(buf[at : at + self.entry])
        return self._ket_read(entry >> KEY_SHIFT, self.entry, (entry & KEY_MASK) * self.entry)

    def _key_flush(self) -> None:
        """把缓冲里的条目一次写回当前这一片。换片、落盘、关文件之前都要调。"""
        buf, first, end = self._ket_seg
        if not buf:  # 缓冲是空的
            return
        inside = (first & KEY_MASK) * self.entry  # 缓冲从片内第几条开始，就是片内偏移
        self._ket_write(first >> KEY_SHIFT, bytes(buf), inside)  # 整块写回片内原位，句柄在一步里取完
        self._ket_seg = (bytearray(), self._key_entries, self._key_entries)  # 换一份空段，捏着旧段的读那边照旧

    def _ket_use(self, at: int):
        """第 at 片键条目文件的句柄，走全库共用的池子：开多少只跟 KET_KEEP 有关，跟片数无关。"""
        path = f"{self.key}{at:04d}"  # 片文件名，池子按完整路径认
        with _KET_LOCK:  # 池子是全局的，几片同时读也要岔开
            handle = _KET_CACHE.get(path)  # 池里有就直接用
            if handle is not None:
                _KET_CACHE.move_to_end(path)  # 用过挪到末尾，别被当最旧的挤掉
                return handle
            handle = os.open(path, os.O_RDWR | os.O_CREAT)  # 没有就开一个
            _KET_CACHE[path] = handle
            if len(_KET_CACHE) > KET_KEEP:  # 到上限了，关掉最久没用的那个
                _, dead = _KET_CACHE.popitem(last=False)
                os.close(dead)
            return handle

    def _ket_read(self, at: int, size: int, offset: int) -> bytes:
        """从第 at 片读一段。句柄的取用和读在同一步里做完，不会被中途挤掉。"""
        return os.pread(self._ket_use(at), size, offset)

    def _ket_write(self, at: int, data: bytes, offset: int) -> None:
        """往第 at 片写一段。同样一口气做完，避免句柄在两步之间被挤掉。"""
        os.pwrite(self._ket_use(at), data, offset)

    def start_heads(self) -> None:
        """开搬之前把桶头表挂进内存当写入的暂存，一整片就这几 MB。"""
        self._heads = array("I", bytes(self.buckets * 4))  # 整片桶头，全零

    def drop_heads(self) -> None:
        """搬完把桶头表丢掉：盘上那份是全的，查询按桶号 pread 就行。"""
        self._heads = None

    def _head_at(self, where: int) -> int:
        """按桶号从盘上读桶头：就四字节，别的桶一个字节不碰。"""
        return int.from_bytes(os.pread(self._chk, 4, where * 4), "big")

    def _bucket(self, where: int) -> int:
        """桶里那串的头（条目号 + 1）。0 就是空桶。"""
        return self.table.get(where)

    def _mapped_records(self):
        """记录文件的只读映射，盖不住片尾就重铺。"""
        if self._rec_map is None or self._rec_bytes > self._rec_span:
            self._out.flush()  # 新写的内容先落盘，映射才看得到
            return self._remap("rec")
        return self._rec_map

    def _mapped_offsets(self):
        """偏移表的只读映射。表是按条追加的，读到哪铺到哪。"""
        if self._off_bytes == 0:  # 一条都还没写
            return b""
        if self._off_map is None or self._off_bytes > self._off_span:
            self._off_out.flush()
            return self._remap("off")
        return self._off_map

    def _remap(self, which: str) -> mmap.mmap:
        """按文件当前长度铺一份映射。长过文件的映射有些系统不收，只能铺到头。"""
        if which == "rec":
            if self._rec_map is not None:
                self._rec_map.close()
            self._rec_span = self._rec_bytes
            self._rec_map = mmap.mmap(self._out.fileno(), self._rec_span, access=mmap.ACCESS_READ)
            return self._rec_map
        if self._off_map is not None:
            self._off_map.close()
        self._off_span = self._off_bytes
        self._off_map = mmap.mmap(self._off_out.fileno(), self._off_span, access=mmap.ACCESS_READ)
        return self._off_map

    def stream_from_file(self):
        """转冷前收拾好自己：缓冲写出去，映射和键表句柄撤掉，桶表跟着关。

        映射的页是常驻的，整片顺序搬的时候用不着它；撤掉再顺手顺序读，内存里只压着一条。
        键条目按条读，搬的时候要顺着每条键翻桶，句柄留着就够，不留映射。
        桶表留着：转冷要按它把键摊回冷片，数组本身就是要用的。
        主流程这时候还可能按下标读这片的老记录，所以撤映射要跟 read 抢同一把锁。
        """
        self.flush()
        with self._lock:
            if self._rec_map is not None:
                self._rec_map.close()
                self._rec_map = None
            self._off_map_close()
            with _KET_LOCK:  # 池子是全局的，只摘自己这几片的
                mine = [one for one in _KET_CACHE if one.startswith(str(self.key))]
                for one in mine:
                    os.close(_KET_CACHE.pop(one))
            self._ket_out = None
    def _off_map_close(self) -> None:
        """撤掉偏移表映射。"""
        if self._off_map is not None:
            self._off_map.close()
            self._off_map = None

    def flush(self) -> None:
        """把缓冲着的都写出去；冷层的桶头整块写回开头那块定长区。复制给别人之前必须调。"""
        self._out.flush()
        self._off_out.flush()
        self._key_flush()  # 键条目缓冲也落地
        if self._heads is not None:  # 冷层正在搬：桶头在内存里攒着，这里整块写回去
            block = array("I", self._heads)
            block.byteswap()  # 盘上用大端
            os.pwrite(self._chk, block.tobytes(), 0)

    def close(self) -> None:
        """把缓冲写出去，再关映射、关句柄、关文件。不落盘就关的话，热层里剩的东西才真没了。"""
        self.flush()  # 盘上已经有的先补齐，冷层的桶头落完地再关
        if self.table is not None:  # 温层的桶表整块写出去，关掉以后再打开也查得到
            self.table.flush()
        for one in (self._rec_map, self._off_map):
            if one is not None:
                one.close()
        self._rec_map = self._off_map = None
        self._ket_out = None
        self._out.close()
        self._off_out.close()
        if self._chk is not None:
            os.close(self._chk)
            self._chk = None

    def remove_files(self) -> None:
        """把这片在盘上的文件删掉。转冷搬完原片就没用了，空壳也占 inode。"""
        for one in self.files():
            one.unlink()


class Index:
    """分带键 → 下标的映射。一次 weave 用一个实例。

    store 给 None 就是纯内存：热层不淘汰，记录全留在内存里，行为和没有这个类时一样。
    watch 给一个可调用对象，转冷时会把「缓冲里还压着几条记录」报给它用，刷出去就报 0。
    转冷在后台线程里一片一片做，put 只排队不等；排着的片由 settle/save/close 负责等干净。
    """

    def __init__(self, store=None, hot: int = HOT, per_shard: int = SHARD, warm: int = WARM, key_size: int = 32) -> None:
        self.store = Path(store) if store else None  # None 就是纯内存
        self.hot = hot  # 热层最多几条键
        self.per_shard = per_shard  # 温层一片几条记录
        self.warm = warm  # 温层总共最多几条记录
        self.key_size = key_size  # 键固定多长，分带键是 4 字节 × 段内行数
        self.count = 0  # 已经登记过多少条样本
        self.watch = None  # 转冷时的计数钩子，平时是 None

        self._live = {}  # 下标 → Live，还在内存里的记录
        self._keys = {}  # 键 → 下标们，热层
        self._order = OrderedDict()  # 键的插入顺序，从头就是最旧的那条，淘汰从头取
        self._where = array("Q")  # 下标 → 盘上位置（片号 << 21 | 片内位置），纯内存时不动
        self._shards = {}  # 片号 → Shard；转冷搬完的温片也留着，飞行中的 record 还要读它
        self._warm = []  # 温层片，按新旧排，末尾最新
        self._moving = set()  # 正在转冷的片：已经不在 _warm 里，但盘上的键还没搬完，查询还要算上
        self._cold = []  # 冷层片
        self._warm_rows = 0  # 温层现在有多少条记录
        self._serial = 0  # 片号发号器

        self._lock = threading.Lock()  # 只护着 _keys/_order/_live/_warm 这些顺序结构；转冷本身在锁外做
        self._jobs = deque()  # 等着转冷的片，先进先出；顺序定死才不会乱着搬
        self._queued = set()  # 已经排进队里的片号，同一片不排两次
        self._left = 0  # 排着队加正在搬的一共几片，等排空的人看这个
        self._done = threading.Condition()  # 片数变了就叫醒 settle
        self._worker = None  # 后台转冷线程；有活才起，队列空了就自己收
        self._error = None  # 转冷抛出来的异常，settle 交给 close/save 往外抛

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
        with self._lock:  # 顺序结构会被后台线程动，动的时候不能插进去
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
        elif self.store is not None:  # 老样本再添一条键，键也要跟着记录进盘
            record.keys += key
        record.left += 1  # 这条样本多了一条键指着它
        if key in self._order:  # 老键，接在后面
            self._keys[key].append(sub)
        else:  # 新键，记进插入顺序
            self._keys[key] = [sub]
            self._order[key] = None

    # --- 查候选 ---

    def get(self, key: bytes) -> list:
        """拿这个键上的候选下标。同一个下标可能出现多次，调用方用集合收。"""
        with self._lock:  # 快照一份片序，查盘的时候不占着锁
            found = list(self._keys.get(key, ()))  # 热层
            warm = list(self._warm) + list(self._moving)  # 温层各片，连正在搬的也算上
            cold = list(self._cold)  # 冷层各片
        for shard in reversed(warm):  # 新的先找
            found += shard.find(key)
        for shard in cold:  # 冷层全键按桶压着，桶里求交命中才交出下标
            found += shard.candidates(key)
        return found

    def record(self, sub: int) -> tuple:
        """按下标读回（签名，窗口集合）。还在内存就直接给，落了盘就跳着读回来。"""
        record = self._live.get(sub)
        if record is not None:  # 还在内存
            return record.sig, record.windows
        packed = self._where[sub]  # 盘上位置
        body = self._shards[packed >> POS_BITS].read(packed & POS_MASK)  # 片号 + 片内位置，读回来
        return _unpack(body, self.key_size)[2:]  # 取签名和窗口，前两样是下标和全部键

    def size(self) -> int:
        """已经收了多少条样本。"""
        return self.count

    # --- 淘汰与落盘 ---

    def out(self, count: int = None) -> None:
        """把热层最旧的 count 条键挤到温层，不给 count 就挤干净。

        挤出去的键照样查得到：条目进键表，记录跟着样本走，最后一条键走了记录才落盘。
        全程占着锁：挤出来的记录往哪片写要和后台「摘片」同一个先后，否则记录会追加到
        一片已经搬完的片上去，它的键就再也摊不进冷层。
        """
        if self.store is None:  # 纯内存没有可挤的地方
            return
        count = len(self._order) if count is None else count
        with self._lock:
            while count > 0 and self._order:
                key, _ = self._order.popitem(last=False)  # 最旧的那条，C 实现，不绕迭代器
                subs = self._keys.pop(key)
                shard = self._shard()  # 挤到当前这片
                for sub in subs:  # 键落到盘上，热层不再留
                    shard.bump_key(key, sub)
                for sub in subs:  # 每条挂着它的样本少一条键
                    record = self._live[sub]
                    record.left -= 1
                    if record.left == 0:  # 一条键都不剩了，记录才准离开内存
                        self._park(sub, record)
                        del self._live[sub]
                count -= 1

    def save(self, directory=None) -> Path:
        """落盘。directory 给了就写到那儿，不给就写自己的存储目录。

        落的是整个索引：记录、偏移表、键表、冷层键桶全在。换一个进程构造同一个目录就全部回来。
        """
        if self.store is None:
            raise ValueError("纯内存索引落不了盘，构造时给 store 目录")
        target = Path(directory) if directory is not None else self.store
        self.out()  # 热层挤干净，内存里的东西全写到盘上
        boom = self.settle()  # 排着的片搬完再写清单：否则清单里会有删掉文件的片、复制到半截
        if boom is not None:
            raise boom
        self.store.mkdir(parents=True, exist_ok=True)
        if target != self.store:  # 另存一份，目标目录先建出来
            target.mkdir(parents=True, exist_ok=True)
        shards = []
        for shard in list(self._shards.values()):  # 清单里只记位置和形状，记录本身在文件里
            if shard.table is not None:  # 温层的桶表先落盘
                shard.table.flush()
            ones = shard.files()  # 这片在盘上占的文件；搬完就删掉文件的温片这里是空的
            if not ones:  # 记录和键都在冷片里了，不再记它
                continue
            shard.flush()  # 先把缓冲写出去，否则复制到的是半截文件
            if target != self.store:  # 换目录就复制一份，原目录保持不动
                for one in ones:
                    shutil.copyfile(one, target / one.name)  # 分块复制，文件再大也不整块读进内存
            shards.append({"serial": shard.serial, "layer": shard.layer, "count": shard.count, "buckets": shard.buckets})
        manifest = {
            "version": 4,  # 清单格式版本。3 起温层桶表是分片文件；4 起键条目按条分片、冷层键桶是单文件
            "pack": COLD_PACK,  # 冷片压没压，读回来时按它解
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

    def settle(self, wait: float = COLD_WAIT) -> BaseException:
        """等后台把排着队的片都转完。转冷抛过异常就把那个异常交回去，没有就是 None。

        wait 秒还没排完就打印还剩几片：一片五十万条搬完要几十秒，卡在这个数外面说明后台跟不上。
        """
        if self._worker is None:  # 从来没起过后台线程
            return self._error
        with self._done:
            if not self._done.wait_for(lambda: self._left == 0, wait):  # 等到搬空为止
                print(f"转冷还没排完，还剩 {self._left} 片")
        return self._error

    def close(self) -> None:
        """关掉全部文件。关之前不落盘的话，热层里剩的东西就没了。

        排队等着的片先搬完再关；转冷抛过异常，文件关完再把异常抛出来，不静默吞掉。
        """
        boom = self.settle()
        for shard in self._shards.values():
            shard.close()
        if boom is not None:
            raise boom

    # --- 挤出去以后怎么落 ---

    def _park(self, sub: int, record: Live) -> None:
        """把一条记录写到盘上，位置表记下它在哪；片满了就交给后台去转冷。

        调用的时候就已经占着 _lock 了（只有 out 调它）：追加和「后台摘走这片」不能岔开。
        """
        shard = self._shard()
        pos = shard.append(_pack(sub, bytes(record.keys), self.key_size, record.sig, record.windows))
        self._where[sub] = (shard.serial << POS_BITS) | pos
        self._warm_rows += 1
        if shard.count >= self.per_shard:  # 这片满了，换新片
            self._open_warm()
        if self._warm_rows > self.warm:  # 温层攒太多了，最老的片扔给后台去转冷
            self._park_oldest()

    def _park_oldest(self) -> None:
        """把最老的片排进转冷队列。已经在队里就不重复排，转不完下一片也不排。"""
        shard = self._warm[0]
        if shard.serial in self._queued:  # 排过还没轮到
            return
        self._queued.add(shard.serial)
        self._jobs.append(shard)
        with self._done:  # 让 settle 知道多了一片要等
            self._left += 1
        if self._worker is None:  # 没有后台线程就起一个
            self._worker = threading.Thread(target=self._work, name="cold", daemon=True)
            self._worker.start()

    def _work(self) -> None:
        """后台线程：一片一片把排着队的温片搬成冷片。转冷本身在锁外做，主流程照常登记。"""
        while True:
            with self._lock:  # 摘片、发冷片号都要和主流程岔开
                if not self._jobs:  # 队列空了就收工，下次排片再起
                    self._worker = None
                    return
                shard = self._jobs.popleft()
                self._queued.discard(shard.serial)
                self._warm.remove(shard)
                self._moving.add(shard)  # 搬完之前它还查得到，不能有中间空窗
                self._warm_rows -= shard.count
                self._serial += 1
                cold = Shard(self.store, self._serial, "cold", self.key_size, self._cold_buckets(shard))
                cold.start_heads()  # 桶头先挂内存当暂存，搬完再整块写回开头那块定长区
                self._shards[cold.serial] = cold
                self._cold.append(cold)  # 摘温片和挂冷片在同一把锁里，中间不留一段两头都查不到
            try:
                self._to_cold(shard, cold)
            except Exception as boom:  # 转冷挂了就记下来，由 close/save 抛给调用方
                self._error = boom
                with self._lock:
                    self._moving.discard(shard)
                with self._done:  # 别再让 settle 空等到超时
                    self._left = 0
                    self._done.notify_all()
                with self._lock:
                    self._worker = None
                return
            with self._lock:  # 盘上的键全搬完了，它才从查询里退出去
                self._moving.discard(shard)
            with self._done:  # 搬完一片，片数减一
                self._left -= 1
                if self._left == 0:
                    self._done.notify_all()

    def _shard(self) -> Shard:
        """当前该写的温层片，末尾就是；一片都没有就开一片。"""
        if not self._warm:
            self._open_warm()
        return self._warm[-1]

    def _open_warm(self) -> Shard:
        """开一片新的温层。桶表按这片能装多少条键放宽度，桶多撞车少。"""
        self._serial += 1
        shard = Shard(self.store, self._serial, "warm", self.key_size, max(1024, self.per_shard * BUCKET))
        self._shards[shard.serial] = shard
        self._warm.append(shard)
        return shard

    def _to_cold(self, shard: Shard, cold: Shard) -> None:
        """把一片温层转成冷层：逐条流式搬，搬完把原片文件、桶片一起删掉。

        每条只走「读一条 → 压一条 → 写一条」，写完当场放开，内存里不攒整片；
        键按桶串进冷片的 chk 文件，桶头在内存里攒着，攒够 COLD_BATCH 条整块写回一次。
        为什么是 2000：一条记录一个桶只落一条键，一批两千条就是两千次八字节位移的桶头回写，
        摊到两千条也就是每片的桶头区写两遍（8KB）；批再大，桶头在内存里攒着不落地，
        中途崩了就整片对不上；批再小则是白多写几遍桶头。
        搬完不关源片、也不从 _shards 摘掉：还在飞行中的 record 按下标读它，句柄留着就读得到；
        文件已经删了，多占的是每片三个 fd，关整个索引时一起收。片是温层，记录原样没压。
        冷片由调用方在锁里建好传进来，这边只管搬。
        """
        shard.stream_from_file()  # 缓冲落盘、映射与桶表关掉，接着顺序读
        buffered = 0  # 这一批已经攒了几条键还没刷出去
        try:
            with shard.rec.open("rb") as source:
                for _ in range(shard.count):  # 逐条搬，一条一条来
                    size = int.from_bytes(source.read(4), "big")  # 长度头
                    body = source.read(size)  # 只解这一条
                    sub = int.from_bytes(body[:8], "big")  # 这条的下标
                    keys = _keys_of(body, self.key_size)  # 这条的全部键
                    at = cold.append(body)  # 压一条、写一条
                    cold.bump(keys, sub)  # 键按桶攒进缓冲
                    self._where[sub] = (cold.serial << POS_BITS) | at  # 位置表改指冷片
                    del body  # 立刻放开，常驻不攒
                    buffered += 1
                    if buffered >= COLD_BATCH:  # 一批攒够了，桶头整块写回一次
                        cold.flush()
                        buffered = 0
                    if self.watch is not None:  # 钩子：这时候缓冲里还压着几条
                        self.watch(buffered)
        finally:
            cold.flush()  # 最后不满一批的桶头也写回去
            cold.drop_heads()  # 盘上那份齐了，内存里那份丢掉
            shard.table.drop()  # 源片的桶表也没人再查，三十二兆放掉
            shard.remove_files()  # 原片连键条目、桶片一起删，不留空壳

    def _cold_buckets(self, shard: Shard) -> int:
        """冷片该开几个键桶：按这片要搬多少条键来定。桶大了每条链都走很长。"""
        keys = shard._key_entries  # 这片有多少条键条目
        return max(MIN_BUCKET, min(MAX_BUCKET, keys // PER_BUCKET))

    # --- 读回来 ---

    def _load(self) -> None:
        """照清单把索引读回来。记录文件扫一遍重建位置表和下标。

        热层容量、片大小这些跟着清单走：接着上一次的设置继续跑，不会突然换一套。
        冷片压没压也照清单来（清单是压着写的，读的时候却按别的模式解，等于读一堆乱码），
        所以这里改的是模块里的 COLD_PACK，同进程里再写新片也跟着这个值。
        """
        global COLD_PACK
        manifest = json.loads((self.store / MANIFEST).read_text(encoding="utf-8"))
        if manifest["version"] != 4:  # 格式对不上就别硬读（3 的键条目是整份一个文件、冷层按桶摊文件）
            raise ValueError(f"索引清单版本 {manifest['version']} 不认识")
        COLD_PACK = manifest.get("pack", True)  # 冷片压没压，照写的时候那个值来；老清单一律是压着的
        self.key_size = manifest["key_size"]
        self.hot = manifest["hot"]
        self.per_shard = manifest["per_shard"]
        self.warm = manifest["warm"]
        self.count = manifest["count"]
        for one in manifest["shards"]:  # 按清单把片开回来
            shard = Shard(self.store, one["serial"], one["layer"], self.key_size, one["buckets"])
            if one["layer"] == "warm":  # 键表按片数回来，接着往后写
                left = sorted(self.store.glob(f"warm-{one['serial']:04d}.ket*"))  # 键条目片，按片号就是条号顺序
                shard._key_entries = sum(one.stat().st_size for one in left) // shard.entry
                if shard._key_entries:  # 有一片写到半截，就从那一片接着写
                    shard._ket_out = shard._ket_use(shard._key_entries >> KEY_SHIFT)
                    shard._ket_seg = (bytearray(), shard._key_entries, shard._key_entries)
                shard.table.load()  # 桶头整块读回来

            self._serial = max(self._serial, one["serial"])
            self._shards[shard.serial] = shard
            if one["layer"] == "warm":
                self._warm.append(shard)
                self._warm_rows += one["count"]
            else:
                self._cold.append(shard)
            self._scan(shard)  # 扫一遍记录，偏移表按条重建，下标填回位置表
            shard.count = one["count"]

    def _scan(self, shard: Shard) -> None:
        """扫一遍记录：偏移按条写进偏移表，下标同时填回位置表。只解当前这一条。"""
        at = 0
        pos = 0
        with shard.rec.open("rb") as handle, shard.off.open("wb") as off:
            while True:
                size = int.from_bytes(handle.read(4), "big")
                if size == 0:  # 长度头都没读到，到底了
                    break
                off.write(at.to_bytes(8, "big"))  # 这条的起点按条追加
                body = handle.read(size)  # 只解这一条
                if shard.layer == "cold" and COLD_PACK:  # 压过的冷片才要解
                    body = zlib.decompress(body)
                sub = int.from_bytes(body[:8], "big")
                while len(self._where) <= sub:  # 位置表按下标铺开
                    self._where.append(NOWHERE)
                self._where[sub] = (shard.serial << POS_BITS) | pos
                at += 4 + size
                pos += 1
        shard._rec_bytes = at  # 文件就这么长，之后追加从这儿接着算
        shard._off_bytes = pos * 8


def _pack(sub: int, keys: bytes, key_size: int, sig: list, windows) -> bytes:
    """一条记录压成字节：下标、全部键、签名、窗口集合。"""
    return sub.to_bytes(8, "big") + (len(keys) // key_size).to_bytes(2, "big") + keys + _words(sig) + _words(windows)


def _keys_of(body: bytes, key_size: int) -> bytes:
    """从一条记录里抠出它全部的键（定宽拼着的字节; 前头是条数）。"""
    count = int.from_bytes(body[8:10], "big")
    return body[10 : 10 + count * key_size]


def _split_keys(keys: bytes, key_size: int) -> set:
    """一串定宽键拆成一个集合，求交用。"""
    return {keys[at : at + key_size] for at in range(0, len(keys), key_size)}


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
    """字节拆回（下标，全部键，签名，窗口集合）。"""
    sub = int.from_bytes(body[:8], "big")
    keys = _keys_of(body, key_size)
    sig, at = _read_words(body, 10 + len(keys))
    windows, at = _read_words(body, at)
    return sub, keys, list(sig), set(windows)
