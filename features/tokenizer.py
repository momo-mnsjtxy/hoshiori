"""分词器：从 jsonl 语料学一份字节级 BPE 词表，用词表编解码，并导出成 GGUF 的 tokenizer kv。

调用示例：
  from features.tokenizer import Vocab
  vocab = Vocab.train([Path("bolt/bolt.jsonl")], 4096, limit=5000)  # 从语料学词表
  vocab.save(Path("out/tokenizer/vocab.json"))                      # 落盘，配方指它
  vocab = Vocab.load(Path("out/tokenizer/vocab.json"))              # 下次原样读回来
  ids = vocab.encode("星织把散落的星点织成整匹布")                  # 编成号
  vocab.decode(ids)                                                 # 还原成字
  ids = vocab.ids_of(text, max_len)                                 # 训练端取一条，收尾补位都做好

怎么接：
  trainer 这样接：Vocab.load(配方 model.tokenizer)，ids_of 每批取，模型 embed 词表按 vocab.size 定。
  forge 这样接：直接 kv() 写进 GGUF。

学会之后词表就是死的，encode 只做查表和按 rank 合并，不再改词表。
编码是字节级的：语料先变成 UTF-8 字节，再按学到的 rank 从相邻对往下并，任何字都不会掉。
词表长这样：id 0/1/2 固定是 <s>、</s>、<pad>，3 到 258 是 256 个单字节，259 往后才是学到的合并词。
token 文本用 GPT-2 那套字节映射（0x00 这类控制字节配成 U+0100 起的字符），
这样每个 token 都能原样写进 JSON 和 GGUF 字符串，decode 再按反向表还原成字节。
"""

import json
import re
from pathlib import Path

from features.forge import K_ARRAY, K_I32, K_STRING, K_U32, TOKEN_TYPE
from utils.fail import fail

SPECIALS = ("<s>", "</s>", "<pad>")                      # 三个控制符，占 id 0/1/2
SPECIAL_ID = {"<s>": 0, "</s>": 1, "<pad>": 2}          # 名字到 id，构造时就定死
SPLIT = re.compile("(" + "|".join(re.escape(one) for one in SPECIALS) + ")")  # 带括号切开，控制符自己留在结果里


def _byte_text() -> list:
    """造 256 个字节到字符的映射表，可逆，照 GPT-2 的排法。

    可打印的那批字节直接用自己，控制字节（换行、空格以外那些）配到 U+0100 往后，
    免得 token 里夹控制符，写 JSON 和 GGUF 时被转义得看不清。
    """
    plain = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    rest = [one for one in range(256) if one not in plain]  # 剩下的控制字节
    chars = [chr(one) for one in plain] + [chr(256 + at) for at in range(len(rest))]  # 可打印的原样，其余往后排
    out = [""] * 256  # 按字节值摆回去
    for at, byte in enumerate(plain + rest):  # 两张表一一对上
        out[byte] = chars[at]
    return out


BYTE_TEXT = _byte_text()  # 字节值 → token 文本


class Vocab:
    """一份学好的词表。构造只接 merges，id 空间由构造过程摆好。"""

    def __init__(self, merges: list) -> None:
        self.merges = [(left, right) for left, right in merges]  # 学到的合并表，顺序就是 rank
        self.tokens = list(SPECIALS)  # 词表本体，id 就是下标，控制符先占前排
        self.raw = {at: one.encode("utf-8") for at, one in enumerate(SPECIALS)}  # 控制符的字节，decode 用
        index = {one: at for at, one in enumerate(SPECIALS)}  # 文本 → id，合并时要查
        self.byte_id = {}  # 字符 → id，编码时按字节查
        for byte in range(256):  # 256 个单字节紧跟在控制符后面
            self.byte_id[BYTE_TEXT[byte]] = len(self.tokens)  # 记下这个字节的号
            index[BYTE_TEXT[byte]] = len(self.tokens)  # 同上，合并时要查
            self.raw[len(self.tokens)] = bytes([byte])  # 一个字节还原成一个字节
            self.tokens.append(BYTE_TEXT[byte])  # 摆进词表
        self.ranks = {}  # 相邻对 → 先后，越小越先并
        self.new = {}  # 相邻对 → 并出来的新号
        for at, (left, right) in enumerate(self.merges):  # 合并词接着往后排
            pair = (index[left], index[right])  # 两条 token 的号
            self.ranks[pair] = at  # 学到的先后
            self.new[pair] = len(self.tokens)  # 新号的落点
            self.raw[len(self.tokens)] = self.raw[pair[0]] + self.raw[pair[1]]  # 新号的字节是两条接起来
            self.tokens.append(left + right)  # 新词文本是两条接起来
            index[left + right] = len(self.tokens) - 1  # 后面还要按文本查它

    @property
    def size(self) -> int:
        """词表大小，模型 embed 和输出层按它定。"""
        return len(self.tokens)

    @property
    def specials(self) -> dict:
        """三个控制符和它们的 id。bos、eos、pad 就是这三个。"""
        return dict(SPECIAL_ID)

    # --- 学词表 ---

    @classmethod
    def train(cls, paths: list, vocab_size: int, *, limit: int = 2000) -> "Vocab":
        """从 jsonl 语料学词表。paths 是文件列表，每个文件一行一条、正文在 text 字段。

        limit：最多读多少条，默认 2000；语料大时先掐量，学得快也够用。
        vocab_size：学完的词表大小，含 3 个控制符和 256 个单字节，至少 259。
        频次一样时按字节序取小的那对，同样语料学出来的词表一字不差。
        """
        if vocab_size < len(SPECIALS) + 256:  # 装不下控制符加全部字节就没法编
            fail("input", f"vocab_size 至少 {len(SPECIALS) + 256}，给了 {vocab_size}", where="model.tokenizer.vocab_size")
        texts = _read(paths, limit)  # 语料一次读进来
        if not texts:  # 一条都没有就学不成
            fail("input", "语料里没有能用的样本", where=", ".join(str(one) for one in paths), hint="先 ori weave 出一份 bolt/")
        seqs = [list(one.encode("utf-8")) for one in texts]  # 每条先摊成字节
        text_of = list(BYTE_TEXT)  # 号的文本，新号往后接
        merges = []  # 学到的合并表
        count, where = _count_pairs(seqs)  # 相邻对各出现几次、分别在哪几条里
        while len(SPECIALS) + 256 + len(merges) < vocab_size:  # 学到目标大小为止
            if not count:  # 全语料都并成一条了，学不动
                break
            left, right = max(count, key=lambda one: (count[one], -one[0], -one[1]))  # 最多的一对，平局取小号
            new = len(text_of)  # 新号
            text_of.append(text_of[left] + text_of[right])  # 新词的文本
            merges.append((text_of[left], text_of[right]))  # 记进合并表，按学的顺序
            for at in where.pop((left, right)):  # 只重扫带这一对的那几条，不整份重来
                seq = seqs[at]
                for pos in range(len(seq) - 1):  # 先把这条的旧账撤掉
                    pair = (seq[pos], seq[pos + 1])
                    count[pair] -= 1
                    if not count[pair]:
                        del count[pair]
                        where.pop(pair, None)
                seq = _join(seq, left, right, new)  # 真的并掉
                seqs[at] = seq
                for pos in range(len(seq) - 1):  # 再把这条的新账记上
                    pair = (seq[pos], seq[pos + 1])
                    count[pair] = count.get(pair, 0) + 1
                    where.setdefault(pair, set()).add(at)
        return cls(merges)

    # --- 存读 ---

    def save(self, path: Path) -> None:
        """写成 vocab.json，只有 merges 和特殊符两张表，词表本体由它们推出来。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)  # 目录没建就先建
        data = {"model": "gpt2", "special_tokens": list(SPECIALS), "merges": [list(one) for one in self.merges]}
        path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")  # 整份落盘

    @classmethod
    def load(cls, path: Path) -> "Vocab":
        """读 vocab.json。格式不对当场说清是哪一条。"""
        path = Path(path)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))  # 整份读进来
        except json.JSONDecodeError as bad:  # 文件坏了要说清坏在哪
            fail("input", f"{path} 不是合法 json（{bad}）", hint="重新 save 一份")
        if data.get("special_tokens") != list(SPECIALS):  # 控制符对不上，id 就全错位
            fail("input", f"{path} 的特殊符不是 {list(SPECIALS)}", hint="用这份分词器自己 save 出来的 vocab.json")
        if "merges" not in data:  # 没有合并表等于空词表
            fail("input", f"{path} 里没有 merges", hint="重新 train 一份")
        return cls([tuple(one) for one in data["merges"]])

    # --- 编解码 ---

    def encode(self, text: str) -> list:
        """把一段字编成号。控制符单独切出来按 id 走，其余按字节 BPE。"""
        ids = []  # 编出来的号
        for piece in SPLIT.split(text):  # 按控制符切成几段
            if not piece:  # 切开后的空段
                continue
            if piece in SPECIAL_ID:  # 就是控制符本身
                ids.append(SPECIAL_ID[piece])  # 用它的固定 id
                continue
            ids.extend(self._merge([self.byte_id[BYTE_TEXT[byte]] for byte in piece.encode("utf-8")]))  # 其余先摊成字节再并
        return ids

    def _merge(self, ids: list) -> list:
        """按 rank 从小到大把相邻对并掉，直到没有能并的。"""
        while True:
            best = None  # 这一步要先并哪一对
            at = -1  # 并在哪一位
            for pos in range(len(ids) - 1):  # 扫一遍相邻对
                rank = self.ranks.get((ids[pos], ids[pos + 1]))  # 这一对学过没、多早
                if rank is not None and (best is None or rank < best):  # 更早并的先并
                    best, at = rank, pos
            if best is None:  # 没有能并的了
                return ids
            ids[at : at + 2] = [self.new[(ids[at], ids[at + 1])]]  # 两条合成一条

    def decode(self, ids: list) -> str:
        """把号还原成字。半个字也照出，坏字节换成替换符。"""
        raw = bytearray()  # 先拼字节
        for one in ids:  # 一个号一条
            raw += self.raw[one]  # 取它的字节接上
        return raw.decode("utf-8", errors="replace")  # 再整段当 UTF-8 解

    def ids_of(self, text: str, max_len: int) -> list:
        """训练端取一条：编码、</s> 收尾、超长截断到 max_len-1、不足拿 <pad> 补齐。"""
        ids = self.encode(text)[: max_len - 1] + [SPECIAL_ID["</s>"]]  # 留一位给收尾符
        return ids + [SPECIAL_ID["<pad>"]] * (max_len - len(ids))  # 补到正好 max_len

    # --- 导出 ---

    def kv(self) -> dict:
        """七个 tokenizer.ggml.* 键，形状和 forge 的 tokenizer() 一样，直接 update 进 kv。"""
        kinds = [TOKEN_TYPE["control"]] * len(SPECIALS) + [TOKEN_TYPE["normal"]] * (self.size - len(SPECIALS))  # 控制符标 3，其余标 1
        return {
            "tokenizer.ggml.model": (K_STRING, "gpt2"),  # 字节级 BPE 就是 gpt2 这套
            "tokenizer.ggml.tokens": (K_ARRAY, (K_STRING, list(self.tokens))),  # 词表本体
            "tokenizer.ggml.merges": (K_ARRAY, (K_STRING, [" ".join(one) for one in self.merges])),  # 合并表，ggml 要空格分隔
            "tokenizer.ggml.token_type": (K_ARRAY, (K_I32, kinds)),  # 每个词的类型
            "tokenizer.ggml.bos_token_id": (K_U32, SPECIAL_ID["<s>"]),  # 开头符
            "tokenizer.ggml.eos_token_id": (K_U32, SPECIAL_ID["</s>"]),  # 收尾符
            "tokenizer.ggml.padding_token_id": (K_U32, SPECIAL_ID["<pad>"]),  # 补位符
        }


def _count_pairs(seqs: list) -> tuple:
    """数一遍全语料的相邻对，同时记下每一对出现在哪几条里。"""
    count = {}  # 相邻对出现几次
    where = {}  # 相邻对 → 出现的序列号
    for at, seq in enumerate(seqs):  # 一条一条数
        for pos in range(len(seq) - 1):  # 每两个相邻的号
            pair = (seq[pos], seq[pos + 1])  # 这一对
            count[pair] = count.get(pair, 0) + 1  # 记一次
            where.setdefault(pair, set()).add(at)  # 记下它在哪条里
    return count, where


def _join(seq: list, left: int, right: int, new: int) -> list:
    """把 seq 里每一处 left right 挨着的地方换成一个 new。"""
    out = []  # 换出来的新序列
    at = 0  # 走到哪
    while at < len(seq):  # 一位一位过
        if at < len(seq) - 1 and seq[at] == left and seq[at + 1] == right:  # 这里正好是要并的一对
            out.append(new)  # 换成新号
            at += 2  # 两条算一条
            continue
        out.append(seq[at])  # 其余原样搬
        at += 1
    return out


def _read(paths: list, limit: int) -> list:
    """把语料读成正文列表，最多 limit 条，空行跳过。"""
    out = []  # 攒起来的正文
    for one in paths:  # 一个文件一趟
        for line in Path(one).read_text(encoding="utf-8").splitlines():  # 一行一条
            if len(out) >= limit:  # 够了就不读了
                return out
            if not line.strip():  # 空行
                continue
            text = json.loads(line).get("text") or ""  # 正文在 text 字段
            if text:  # 有正文才收
                out.append(text)
    return out
