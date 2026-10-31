"""分词器的测试：能学、能存读、编解码往返、超长截断补齐、控制符、kv 七键与 forge 对得上、空语料报错。

约定：
  语料是几十条重复句，学到 300 个词就够验合并的那条路；词表小，几秒跑完。
  kv 的用例真走一遍 forge 的 gguf_bytes 和 read，不是只看键名。
"""

import json

import numpy as np
import pytest

from features import forge, tokenizer
from features.tokenizer import Vocab
from utils.fail import Failure


@pytest.fixture
def corpus(tmp_path):
    """一段有重复的语料：重复是 BPE 能学出合并的前提。"""
    texts = [f"星织把散落的星点织成整匹布，第{i}条。" for i in range(40)]  # 40 条，句式高度重复
    path = tmp_path / "bolt.jsonl"  # 跟 weave 的产物同名
    path.write_text("\n".join(json.dumps({"text": one}, ensure_ascii=False) for one in texts), encoding="utf-8")
    return [path]


@pytest.fixture
def vocab(corpus):
    """学到 300 个词的一份词表。"""
    return Vocab.train(corpus, 300, limit=40)


# ---------- 学词表 ----------

def test_train_reaches_requested_size(vocab):
    assert vocab.size == 300  # 控制符 3 + 字节 256 + 学到的 41
    assert len(vocab.merges) == 300 - 259  # 合并表刚好补满差额


def test_train_is_stable(corpus):
    one = Vocab.train(corpus, 280, limit=40)
    other = Vocab.train(corpus, 280, limit=40)
    assert one.merges == other.merges  # 同语料同参数，学出来一字不差


def test_limit_caps_records(corpus):
    one = Vocab.train(corpus, 259, limit=1)  # 只要一条，没有可学的合并
    assert one.merges == []


def test_empty_corpus_fails(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("\n\n", encoding="utf-8")
    with pytest.raises(Failure) as err:
        Vocab.train([path], 300)
    assert err.value.kind == "input"  # 空语料是用户该看一眼的那种错


def test_vocab_size_too_small_fails(corpus):
    with pytest.raises(Failure) as err:
        Vocab.train(corpus, 200)
    assert "至少 259" in err.value.message


# ---------- 编解码 ----------

def test_roundtrip(vocab):
    for text in ("星织", "hello world", "第 7 条。", "a\tb\nc", "😀 emoji"):
        assert vocab.decode(vocab.encode(text)) == text  # 任何字都要原样回来


def test_roundtrip_uses_merges(vocab):
    ids = vocab.encode("星织" * 5)
    assert len(ids) < len("星织".encode("utf-8")) * 5  # 学过合并之后比纯字节短


def test_specials_have_fixed_ids(vocab):
    assert vocab.specials == {"<s>": 0, "</s>": 1, "<pad>": 2}
    assert vocab.encode("<s>x</s><pad>")[0] == 0  # 控制符按自己的 id 走
    assert vocab.decode([0, 1, 2]) == "<s></s><pad>"  # 也能还原


def test_ids_of_truncates_and_pads(vocab):
    long_ids = vocab.ids_of("星织" * 400, 16)
    assert len(long_ids) == 16  # 超长截到定长
    assert long_ids[-1] == vocab.specials["</s>"]  # 最后一位是收尾符
    short_ids = vocab.ids_of("星", 16)
    assert short_ids[-1] == vocab.specials["<pad>"]  # 不足的补位
    assert short_ids[1] == vocab.specials["</s>"]  # 收尾符紧跟在正文后


# ---------- 存读 ----------

def test_save_and_load_roundtrip(vocab, tmp_path):
    path = tmp_path / "vocab.json"
    vocab.save(path)
    back = Vocab.load(path)
    assert back.tokens == vocab.tokens  # 词表一条不差
    assert back.encode("星织") == vocab.encode("星织")  # 编出来的号也一样


def test_load_rejects_wrong_specials(vocab, tmp_path):
    path = tmp_path / "vocab.json"
    path.write_text(json.dumps({"special_tokens": ["<s>"], "merges": []}), encoding="utf-8")
    with pytest.raises(Failure):
        Vocab.load(path)


def test_load_rejects_broken_json(tmp_path):
    path = tmp_path / "vocab.json"
    path.write_text("{不好", encoding="utf-8")
    with pytest.raises(Failure) as err:
        Vocab.load(path)
    assert err.value.kind == "input"


# ---------- 导出 ----------

def test_kv_has_seven_keys_with_ggml_shape(vocab):
    kv = vocab.kv()
    assert set(kv) == {  # 七个键，一个不多一个不少
        "tokenizer.ggml.model",
        "tokenizer.ggml.tokens",
        "tokenizer.ggml.merges",
        "tokenizer.ggml.token_type",
        "tokenizer.ggml.bos_token_id",
        "tokenizer.ggml.eos_token_id",
        "tokenizer.ggml.padding_token_id",
    }
    assert kv["tokenizer.ggml.model"] == (forge.K_STRING, "gpt2")  # 标记按 gpt2 走
    kind, (item_kind, tokens) = kv["tokenizer.ggml.tokens"]  # 词表是字符串数组
    assert (kind, item_kind, tokens) == (forge.K_ARRAY, forge.K_STRING, vocab.tokens)
    kind, (item_kind, merges) = kv["tokenizer.ggml.merges"]  # merges 同样是字符串数组
    assert (kind, item_kind, len(merges)) == (forge.K_ARRAY, forge.K_STRING, len(vocab.merges))
    assert all(one.count(" ") == 1 for one in merges)  # ggml 要空格分隔的两条
    kind, (item_kind, types) = kv["tokenizer.ggml.token_type"]  # 类型是 i32 数组
    assert (kind, item_kind) == (forge.K_ARRAY, forge.K_I32)
    assert types[:3] == [forge.TOKEN_TYPE["control"]] * 3  # 控制符标 3
    assert set(types[3:]) == {forge.TOKEN_TYPE["normal"]}  # 其余全是普通词
    assert kv["tokenizer.ggml.bos_token_id"] == (forge.K_U32, 0)  # 三个特殊 id 指到控制符
    assert kv["tokenizer.ggml.eos_token_id"] == (forge.K_U32, 1)
    assert kv["tokenizer.ggml.padding_token_id"] == (forge.K_U32, 2)


def test_kv_goes_through_forge_gguf(vocab, tmp_path):
    kv = vocab.kv()  # 我们的键直接交给 forge
    tensors = {"token_embd.weight": np.zeros((4, 4), dtype=np.float32)}  # 一份占位的权重
    path = tmp_path / "t.gguf"
    path.write_bytes(forge.gguf_bytes(tensors, kv, "f16"))
    got = forge.read(path)["kv"]  # 用 forge 自己的 reader 回读
    assert got["tokenizer.ggml.model"] == "gpt2"  # 算法名一字不差
    assert got["tokenizer.ggml.tokens"] == vocab.tokens  # 词表整份回得来
    assert got["tokenizer.ggml.merges"] == [" ".join(one) for one in vocab.merges]  # merges 也回得来
    assert got["tokenizer.ggml.token_type"] == [3, 3, 3] + [1] * (vocab.size - 3)  # 类型没串位
    assert got["tokenizer.ggml.bos_token_id"] == 0  # 三个 id 也在
    assert got["tokenizer.ggml.eos_token_id"] == 1
    assert got["tokenizer.ggml.padding_token_id"] == 2


def test_byte_text_is_reversible():
    assert len(tokenizer.BYTE_TEXT) == 256  # 256 个字节各有字符
    assert len(set(tokenizer.BYTE_TEXT)) == 256  # 互不重复
    assert all(len(one) == 1 for one in tokenizer.BYTE_TEXT)  # 每条都是一个字符


def test_vocab_command(tmp_path):
    """命令层：学出来的词表能落盘，参数检查在开头。"""
    import main

    src = tmp_path / "a.jsonl"
    src.write_text('\n'.join('{"text": "织布第%d条"}' % i for i in range(20)), encoding="utf-8")
    out = tmp_path / "v.json"
    assert main.run(["vocab", "--in", str(src), "--out", str(out), "--size", "300", "--limit", "20"]) == 0
    assert out.is_file()  # 真落盘了
    assert main.run(["vocab", "--in", str(src), "--out", str(out), "--size", "100"]) == 2  # 词表太小
