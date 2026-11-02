"""样本、判重、指纹、来源、数据集、混配的测试。

约定：短文本用重复句拼长，免得 5 字窗口在短句上全崩；改写只动空白或固定间隔换字。
"""

import json

import pytest

from features import dataset, duplicate, fingerprint, source, woof
from features.clean import blend, score, scrub
from features.sample import Policy, Refused, clean, read, text_of
from main import parser as command_parser


# ---------- 样本 ----------

def test_clean_rejects_blank_and_long():
    with pytest.raises(Refused) as blank:
        clean("  ", Policy())
    assert blank.value.code == "too_short"
    with pytest.raises(Refused) as long:
        clean("x" * 50, Policy(max_chars=10))
    assert long.value.code == "too_long"


def test_clean_strips_ctrl_keeps_width():
    assert clean("Ａ\u0007ｂ", Policy()) == "Ａｂ"  # NFC 不改字宽，只剔控制符


def test_clean_collapses_ws():
    assert clean("a  \t b\n\n\n\nc", Policy()) == "a b\n\nc"


def test_text_of_prefers_field_then_messages():
    assert text_of({"text": "正文", "messages": [{"role": "user", "content": "忽略"}]}, Policy()) == "正文"
    joined = text_of({"messages": [{"role": "user", "content": "问"}, {"role": "assistant", "content": "答"}]}, Policy())
    assert "<|user|>" in joined


def test_unknown_role_refused():
    with pytest.raises(Refused) as bad:
        text_of({"messages": [{"role": "tool", "content": "x"}]}, Policy())
    assert bad.value.code == "unknown_role"


def test_no_field_refused():
    with pytest.raises(Refused) as bad:
        text_of({"别的": "值"}, Policy())
    assert bad.value.code == "no_text_field"


def test_read_keeps_messages_and_meta():
    out = read({"text": "正文", "messages": [{"role": "user", "content": "问"}], "meta": {"k": 1}}, Policy())
    assert out["text"] == "正文"
    assert out["messages"]
    assert out["meta"]["k"] == 1


@pytest.mark.parametrize(
    ("raw", "safe"),
    [
        ("联系 fox+star@example.com", "联系 «邮箱»"),
        ("电话 13812345678", "电话 «手机号»"),
        ("证件 11010519491231002X", "证件 «身份证号»"),
        ("令牌 sk-abcdefghijklmnopqrstuvwxyz0123456789", "令牌 «密钥»"),
        ("节点 192.168.1.42", "节点 «IP»"),
    ],
)
def test_scrub_replaces_each_sensitive_kind(raw, safe):
    assert scrub(raw) == (safe, 1)


def test_read_obeys_scrub_policy():
    raw = "联系 fox@example.com"
    record = {"text": raw, "messages": [{"role": "user", "content": raw}]}
    safe = read(record, Policy())
    kept = read(record, Policy(scrub=False))
    assert safe["text"] == "联系 «邮箱»"
    assert safe["messages"][0]["content"] == "联系 «邮箱»"
    assert kept["text"] == raw
    assert kept["messages"][0]["content"] == raw


def test_score_stays_in_bounds_and_rewards_usable_text():
    good = (
        "星织沿着清楚的来源标记整理语料，每一条记录都能回到原文件。"
        "清洗阶段保留正文含义，去除无效空白，再检查机械重复。"
        "完整的句子以标点收尾，方便模型学到稳定的语言边界。"
    )
    repeated = "重复重复重复重复" * 100
    assert score("") == 0.0
    assert 0.0 <= score("短") <= 1.0
    assert 0.0 <= score("很长。" * 10_000) <= 1.0
    assert score(good) > score(repeated)


def test_min_score_refuses_low_quality_text():
    with pytest.raises(Refused) as bad:
        read({"text": "重复重复重复重复" * 30}, Policy(min_score=0.9))
    assert bad.value.code == "low_score"


def test_read_keeps_images_and_refuses_bad_paths(tmp_path):                          # 单路径、路径数组和两种拒收一次覆盖
    first = tmp_path / "first.png"                                                  # 样本层只确认路径存在
    second = tmp_path / "second.png"                                                # 图片内容留给训练准备入口读取
    first.write_bytes(b"first")                                                     # 造两个真实存在的文件
    second.write_bytes(b"second")                                                   # 路径顺序应原样保留
    one = read({"text": "正文", "images": str(first)}, Policy())                   # 字符串路径也统一成数组
    many = read({"text": "正文", "images": [str(first), str(second)]}, Policy())   # 数组路径逐个保留
    assert one["images"] == [str(first)]                                            # 单图下游也得到同一种结构
    assert many["images"] == [str(first), str(second)]                              # 多图顺序没有变化
    with pytest.raises(Refused) as missing:                                          # 不存在的图片在样本入口拒收
        read({"text": "正文", "images": [str(tmp_path / "missing.png")]}, Policy())  # 不留到训练时才报错
    assert missing.value.code == "no_image"                                        # 缺图有独立原因码
    with pytest.raises(Refused) as crowded:                                          # 默认一条最多四张图
        read({"text": "正文", "images": [str(first)] * 5}, Policy())               # 路径都存在，只考数量上限
    assert crowded.value.code == "too_many_images"                                 # 超限也能在拒收档单独统计


# ---------- 判重 ----------

BASE = "星织把散落的星点织成整匹布，训练管线就是织机，梭子交到手上，经线与纬线各自就位。" * 3


def _light(text):
    """只动空白和标点，视图不变。"""
    return text.replace("，", " ,  ").replace("。", "\n。")


def _heavy(text):
    """每 7 字换一个字，只有放宽阈值才该判重。"""
    chars = list(text)
    for i in range(0, len(chars), 7):
        chars[i] = "龍"
    return "".join(chars)


def test_view_drops_space_and_punct():
    assert duplicate.view("a， b。 c") == "abc"


def test_light_mutation_caught():
    dup = duplicate.Duplicate(duplicate.Rule(threshold=0.9))
    assert dup.check(BASE) is None
    assert dup.check(_light(BASE)) == 0
    assert dup.result.dropped == 1


def test_light_mutation_caught_with_store(tmp_path):
    """同一段话只动空白，落盘索引也判得出来；热层故意压到 2 条键，逼它往盘上挤。"""
    dup = duplicate.Duplicate(duplicate.Rule(threshold=0.9), store=tmp_path, hot=2)
    assert dup.check(BASE) is None
    assert dup.check(_light(BASE)) == 0
    assert dup.result.dropped == 1
    dup.close()


def test_threshold_layers():
    tight = duplicate.Duplicate(duplicate.Rule(threshold=0.95))
    tight.check(BASE)
    assert tight.check(_heavy(BASE)) is None

    loose = duplicate.Duplicate(duplicate.Rule(threshold=0.15))
    loose.check(BASE)
    assert loose.check(_heavy(BASE)) == 0


def test_different_texts_both_kept():
    dup = duplicate.Duplicate(duplicate.Rule())
    dup.check(BASE)
    dup.check("另一段完全无关的话，讲织机保养、梭子替换、断线接头与张力校准。" * 2)
    assert dup.result.kept == 2


def test_rows_follow_threshold():
    assert duplicate.rows_for(0.15, 128) < duplicate.rows_for(0.95, 128)


def test_plain_engine_repeatable():
    values = duplicate.shingles(BASE)
    assert duplicate.sign_plain(values, 4242, 8) == duplicate.sign_plain(values, 4242, 8)


def test_short_text_keeps_shingles():
    assert duplicate.shingles("短文本")


def test_bad_divisor():
    with pytest.raises(ValueError):
        duplicate.Duplicate(duplicate.Rule(count=100, rows=7))


# ---------- 指纹 ----------

def test_fingerprint_stable_and_sensitive():
    left, right = fingerprint.Stamp(), fingerprint.Stamp()
    for line in ("甲", "乙"):
        left.feed(line)
        right.feed(line)
    assert left.seal()["content"] == right.seal()["content"]

    other = fingerprint.Stamp()
    for line in ("甲", "丙"):
        other.feed(line)
    assert other.seal()["content"] != left.seal()["content"]


def test_fingerprint_compare():
    left, right = fingerprint.Stamp(), fingerprint.Stamp()
    left.feed("甲")
    right.feed("乙")
    assert fingerprint.compare(left.seal(), right.seal())


# ---------- 来源 ----------

def test_source_reads_jsonl_and_text(tmp_path):
    path = tmp_path / "a.jsonl"
    path.write_text('{"text": "一行"}\n裸文本\n', encoding="utf-8")
    rows = list(source.file(path))
    assert rows[0][0]["text"] == "一行"
    assert rows[1][0]["text"] == "裸文本"


def test_source_reads_array(tmp_path):
    path = tmp_path / "a.json"
    path.write_text(json.dumps([{"text": "甲"}, {"text": "乙"}]), encoding="utf-8")
    assert len(list(source.file(path))) == 2


def test_source_missing(tmp_path):
    with pytest.raises(Exception):
        list(source.read(str(tmp_path / "无.jsonl")))


# ---------- 数据集 ----------

def test_dataset_writes_products(tmp_path):
    src = tmp_path / "in.jsonl"
    lines = [{"text": BASE}, {"text": _light(BASE)}, {"text": "另一段完全不同的话，讲断线接头。"}]
    src.write_text("\n".join(json.dumps(one, ensure_ascii=False) for one in lines), encoding="utf-8")
    out = tmp_path / "bolt"
    job = dataset.Choice(out=out, name="t", license="cc0", rule=duplicate.Rule(threshold=0.9))
    counts = dataset.Dataset(job).make([str(src)])

    assert counts.read == 3
    assert counts.kept == 2
    assert counts.duplicated == 1
    assert (out / "card.json").exists()
    assert (out / "fingerprint.json").exists()
    card = json.loads((out / "card.json").read_text(encoding="utf-8"))
    assert card["counts"]["kept"] == 2
    assert card["sources"][0]["uri"] == str(src)


def test_dataset_keeps_duplicates_when_asked(tmp_path):
    src = tmp_path / "in.jsonl"
    body = json.dumps({"text": BASE}, ensure_ascii=False) + "\n" + json.dumps({"text": _light(BASE)}, ensure_ascii=False) + "\n"
    src.write_text(body, encoding="utf-8")
    out = tmp_path / "o"
    job = dataset.Choice(out=out, rule=duplicate.Rule(threshold=0.9), keep_duplicates=True)
    dataset.Dataset(job).make([str(src)])
    assert len((out / "bolt.jsonl").read_text(encoding="utf-8").splitlines()) == 2


def test_dataset_records_refuse_reason(tmp_path):
    src = tmp_path / "in.jsonl"
    src.write_text('{"text": "  "}\n', encoding="utf-8")
    out = tmp_path / "o"
    counts = dataset.Dataset(dataset.Choice(out=out)).make([str(src)])
    first = json.loads((out / "rejects.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert first["reason"] == "too_short"
    assert counts.refused == {"too_short": 1}


def test_dataset_blends_fake_and_counts_kept_fake(tmp_path):
    real = tmp_path / "real.jsonl"
    fake = tmp_path / "fake.jsonl"
    real_rows = ["春风越过青山。", "潮水拍打礁石。", "夜航灯照海面。", "松鼠藏起榛果。"]
    fake_rows = ["纸鸢掠过城墙。", "细雨落入池塘。", "晨钟唤醒村庄。", "列车驶出隧道。"]
    real.write_text("\n".join(real_rows), encoding="utf-8")
    fake.write_text("\n".join(fake_rows), encoding="utf-8")
    out = tmp_path / "mixed"
    choice = dataset.Choice(out=out, fake_files=[str(fake)], fake_ratio=0.25)
    counts = dataset.Dataset(choice).make([str(real)])

    assert counts.read == 4
    assert counts.kept == 4
    assert counts.kept_fake == 1
    assert counts.summary()["kept_fake"] == 1


def test_weave_cli_collects_scrub_score_and_fake_choices():
    args = command_parser().parse_args(
        [
            "weave",
            "--in",
            "real.jsonl",
            "--no-scrub",
            "--min-score",
            "0.6",
            "--fake",
            "first.jsonl",
            "--fake",
            "second.jsonl",
            "--fake-ratio",
            "0.25",
        ]
    )
    assert args.scrub is False
    assert args.min_score == 0.6
    assert args.fake_files == ["first.jsonl", "second.jsonl"]
    assert args.fake_ratio == 0.25


# ---------- 混配 ----------

def test_blend_follows_fake_ratio():
    real = [f"real-{number}" for number in range(8)]
    fake = [f"fake-{number}" for number in range(8)]
    mixed = blend(real, fake, 0.25)
    assert len(mixed) == 8
    assert sum(row.startswith("fake-") for row in mixed) == 2
    assert sum(row.startswith("real-") for row in mixed) == 6


def test_blend_keeps_the_nonempty_side():
    real = ["real-a", "real-b"]
    fake = ["fake-a", "fake-b"]
    assert blend(real, [], 0.5) == real
    assert blend([], fake, 0.5) == fake
    assert blend([], [], 0.5) == []


def test_woof_split_follows_weight():
    plan = woof.Woof(sources=[woof.Source("a", 1.0), woof.Source("b", 3.0)], target=100).split()
    assert plan["b"] > plan["a"] > 0
    assert sum(plan.values()) <= 100


def test_woof_take_hard():
    assert woof.Woof(sources=[woof.Source("a", 1.0, take=10)]).split()["a"] == 10


def test_woof_stamp_changes():
    one = woof.Woof(sources=[woof.Source("a", 1.0)], target=10)
    two = woof.Woof(sources=[woof.Source("a", 2.0)], target=10)
    assert one.stamp() != two.stamp()


def test_woof_needs_budget():
    with pytest.raises(Exception):
        woof.Woof(sources=[woof.Source("a", 1.0)]).split()
