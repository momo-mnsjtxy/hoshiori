"""重复判定：把一条正文压成指纹，跟已收的比，够像就算重复。

调用示例：
  from features.duplicate import Duplicate, Rule
  dup = Duplicate(Rule(threshold=0.8))
  dup.check("一段话")        # 第一次 None
  dup.check("同一段话，改了空格")  # 返回 0，表示跟第 0 条重复
  dup.result.dropped         # 一共判掉几条

三段配合，缺一段就废：
  1. 视图 view()：先剔掉空白和标点，只留字。改空格、换行、标点不影响判定。
  2. 签名 sign()：从视图取 5 字窗口，哈希后按 128 个排列各取最小值，压成定长指纹。
  3. 分带 bands：把指纹切成 8 段比，任意一段全等就是候选；候选再回算真实重合度过线才算重复。

为什么最后要回算：分带只看"有没有一段全等"，低相似度也会撞车，不回算会误杀。
分带这张表交给 features.index：默认还在内存，给了 store 目录，新建时用哈希前缀分片落盘，
百万级语料不再顶内存。窗口集合也不留内存，回算时按下标从索引里读回来。
"""

import hashlib
import re
from array import array
from dataclasses import dataclass, field

from features import index

KEEP = re.compile(r"[^\W_]+", re.UNICODE)  # 只留字母、数字、汉字；下划线和标点都不要
PRIME = 2147483647  # 2^31-1。哈希乘数压在这个宽度内，乘积超不过 2^63，numpy 不会静默溢出


def view(text: str) -> str:
    """判定用的视图：全小写、只留字。纯符号的正文留原样，免得变成空串。"""
    kept = "".join(KEEP.findall(text)).lower()
    return kept or text


def shingles(text: str, size: int = 5) -> list:
    """视图的 5 字窗口哈希，去重后保序返回。短文本自动缩小窗口，别只剩一个整串。"""
    body = view(text)
    size = min(size, max(2, len(body) // 3)) if len(body) > 3 else 1  # 太短就退到 1，短句也能比
    pieces = [body] if len(body) <= size else [body[i : i + size] for i in range(len(body) - size + 1)]
    seen, out = set(), []
    for piece in pieces:
        digest = hashlib.blake2b(piece.encode("utf-8"), digest_size=4).digest()  # 4 字节够用，省内存
        value = int.from_bytes(digest, "big")
        if value in seen:  # 同一窗口重复出现只算一次
            continue
        seen.add(value)
        out.append(value)
    return out


def pairs(count: int, seed: int):
    """按种子生成 count 组 (a, b) 排列参数，a 取奇数且小于 PRIME。"""
    for i in range(count):
        raw = hashlib.blake2b(f"{seed}:{i}".encode(), digest_size=16).digest()
        a = int.from_bytes(raw[:8], "big") % (PRIME - 2) + 1
        yield (a | 1), int.from_bytes(raw[8:], "big") % PRIME


def sign_plain(values: list, seed: int, count: int) -> list:
    """纯标准库的签名：逐个排列求最小值。慢，但没有 numpy 也跑得动。"""
    if not values:
        return [0] * count
    sig = []
    for a, b in pairs(count, seed):
        sig.append(min((a * v + b) % PRIME for v in values))  # 每个排列取全场最小，就是该排列的指纹
    return sig


def make_sign(seed: int, count: int):
    """挑一个签名实现。机器上有 numpy 就用矩阵算，没有退回纯 Python。"""
    try:
        import numpy
    except ImportError:
        return lambda values: sign_plain(values, seed, count)

    rows = numpy.array([a for a, _ in pairs(count, seed)], dtype=numpy.uint64)  # 排列的第一半
    adds = numpy.array([b for _, b in pairs(count, seed)], dtype=numpy.uint64)  # 排列的第二半

    def sign_fast(values: list) -> list:
        if not values:
            return [0] * count
        column = numpy.array(values, dtype=numpy.uint64).reshape(-1, 1)  # 竖着排，好跟每个排列相乘
        return [int(v) for v in ((rows * column + adds) % PRIME).min(axis=0)]  # 一次算出全部排列的最小值

    return sign_fast


def overlap(left: set, right: set) -> float:
    """两堆窗口的真实重合度，交并比。"""
    if not left and not right:
        return 1.0
    common = len(left & right)
    return 0.0 if not common else common / len(left | right)


@dataclass
class Rule:
    """判定规矩。rows 留 0 表示按阈值自动推。"""

    threshold: float = 0.8  # 重合度到这条线就算重复
    size: int = 5  # 窗口字数
    count: int = 128  # 指纹长度
    rows: int = 0  # 一段几行；越小越容易撞上，阈值低时必须小
    seed: int = 4242
    exact: bool = True  # 是否回算真实重合度（关掉会误杀低相似度文本）


def rows_for(threshold: float, count: int) -> int:
    """按阈值反推段内行数：从大往小试，第一个仍在阈值处大概率撞上的就是它。

    阈值 0.8 得 4 行 32 段，0.95 得 16 行 8 段。阈值越低越要少行，
    否则真重复也撞不上同一段，全被漏掉。
    """
    for rows in range(max(1, count // 2), 0, -1):
        if count % rows:  # 指纹必须能被段数整除
            continue
        bands = count // rows
        if 1 - (1 - threshold**rows) ** bands >= 0.95:  # 撞车概率够高
            return rows
    return 1


@dataclass
class Result:
    """统计：留了几条、判掉几条、判掉的对谁。"""

    kept: int = 0
    dropped: int = 0
    pairs: list = field(default_factory=list)


class Duplicate:
    """重复判定的索引。一次 weave 用一个实例。

    store 给 None 就纯内存；给了目录就把分带索引交给 features.index 落盘，
    窗口集合跟着记录走，回算时按下标读回来，不留内存。hot 仅控制旧 v4 热层；
    新 v5 使用固定 SQLite 页缓存。已有 v4 目录不会静默迁移，详见 docs/INDEX.md。
    """

    def __init__(self, rule: Rule, store=None, hot: int = index.HOT) -> None:
        self.rule = rule
        rule.rows = rule.rows or rows_for(rule.threshold, rule.count)  # 没给就按阈值推算
        if rule.count % rule.rows:
            raise ValueError("指纹长度要被段内行数整除")
        self.result = Result()
        self._sign = make_sign(rule.seed, rule.count)  # 提前选好实现，别在循环里反复判断
        self._index = index.Index(store, hot=hot, key_size=rule.rows * 4)  # 段内的值拼成定长键，键 → 下标
        self._taken = 0  # 已收样本数，同时也是下一条样本的下标

    def check(self, text: str) -> int:
        """返回命中的下标；没命中就入索引并返回 None。"""
        values = shingles(text, self.rule.size)
        windows = set(values)  # 回算要的是集合
        sig = self._sign(values)  # 本条指纹

        keys = list(self._bands(sig))  # 本条的分带键，查和登记都用它，只算一次
        hits = set()  # 候选下标
        for key in keys:
            hits.update(self._index.get(key))  # 这一段全等的历史下标

        best, hit = 0.0, None
        for index_at in sorted(hits):  # 候选不多，逐个回算真实重合度
            other_sig, other_windows = self._index.record(index_at)  # 对方的签名和窗口，盘上的读回来
            score = overlap(windows, other_windows) if self.rule.exact else self._estimate(sig, other_sig)
            if score >= self.rule.threshold and score > best:  # 过线且比之前的更像
                best, hit = score, index_at

        mine = self._taken
        self._taken += 1
        if hit is not None:  # 判掉，只记不重复登记
            self.result.dropped += 1
            self.result.pairs.append((mine, hit, round(best, 4)))
            return hit
        self.result.kept += 1
        for key in keys:  # 不重复才进索引
            self._index.put(key, mine, sig, windows)  # 记录随第一条键一起进索引
        return None

    def save(self) -> None:
        """落盘。纯内存时什么也不做。"""
        if self._index.store is not None:
            self._index.save()

    def close(self) -> None:
        """关掉索引占的文件。纯内存时什么也不做。"""
        if self._index.store is not None:
            self._index.close()

    def _bands(self, sig: list):
        """指纹切成段，每段的值按大端拼成一个定长键。逐个 to_bytes 太慢，走 array 一把过。"""
        for band in range(self.rule.count // self.rule.rows):
            start = band * self.rule.rows
            block = array("I", sig[start : start + self.rule.rows])  # 指纹值都小于 2^31，4 字节够
            block.byteswap()  # 统一成大端，换台机器也一样
            yield block.tobytes()

    def _estimate(self, left: list, right: list) -> float:
        """不回算时用指纹估算：两个指纹一样的位置占多少。"""
        return sum(1 for a, b in zip(left, right) if a == b) / len(left)
