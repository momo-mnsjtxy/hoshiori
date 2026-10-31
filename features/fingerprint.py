"""指纹：把整份数据的正文压成一串摘要，改一个字都会变。

调用示例：
  from features import fingerprint
  stamp = fingerprint.Stamp()
  stamp.feed("一行正文")        # 逐条喂
  result = stamp.seal()        # {"content": "sha256:…", "count": 12, "chunks": [...]}
  fingerprint.compare(a, b)    # 两份指纹的差异清单，空列表表示一致

指纹进数据集卡，也进配方的哈希锁。换批数据没换配方，靠它就能查出来。
每 500 条封一段，段摘要单独存，这样改一行只会动到所在那一段，差异定位得到具体位置。
"""

import hashlib

CHUNK = 500  # 每多少条封一段
SEP = b"\x1e"  # 记录分隔符，防止两条内容首尾拼起来正好等于第三条


class Stamp:
    """边喂边算的指纹器。"""

    def __init__(self, algo: str = "sha256") -> None:
        self.algo = algo
        self._total = _digest(algo)  # 整份的总摘要
        self._chunk = _digest(algo)  # 当前段的摘要
        self._chunks = []
        self._count = 0

    def feed(self, text: str) -> None:
        """喂一条正文。"""
        payload = text.encode("utf-8") + SEP  # 定死分隔符，拼接歧义没了
        self._total.update(payload)  # 总摘要跟着走
        self._chunk.update(hashlib.sha256(payload).digest())  # 段摘要只吃定长摘要，段长不影响它
        self._count += 1
        if self._count % CHUNK == 0:  # 到段长就封一段
            self._seal()

    def _seal(self) -> None:
        """封当前段，换一个新的段摘要器。"""
        self._chunks.append(f"{self.algo}:{self._chunk.hexdigest()}")
        self._chunk = _digest(self.algo)

    def seal(self) -> dict:
        """收尾，交出整份指纹。"""
        if self._count % CHUNK:  # 尾部不满一段也封上
            self._seal()
        return {"algo": self.algo, "content": f"{self.algo}:{self._total.hexdigest()}", "count": self._count, "chunks": self._chunks}


def _digest(algo: str):
    """按算法名取一个哈希器。blake3 快，但要多装一个包。"""
    if algo == "blake3":
        import blake3

        return blake3.blake3()
    return hashlib.sha256()


def compare(left: dict, right: dict) -> list:
    """两份指纹的差异，逐项列成人能读的话。"""
    diffs = []
    if left["content"] != right["content"]:  # 总摘要不一样，内容肯定变了
        diffs.append(f"内容 {left['content']} ≠ {right['content']}")
    if left["count"] != right["count"]:  # 条数不一样，删了或多了
        diffs.append(f"条数 {left['count']} ≠ {right['count']}")
    if left["chunks"] != right["chunks"]:  # 段摘要不同，改动落在哪几段
        diffs.append("分段不一致")
    return diffs
