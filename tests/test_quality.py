"""脱敏与人工校准的行为验证：保留语义标签、独立留出、weave 真正使用档案。

调用：python -m pytest tests/test_quality.py
"""
import json
import pytest
import main
from features import clean, sample
from utils.fail import Failure


def test_sensitive_fields_and_card_validation():
    text, hits = clean.scrub("姓名：张三，地址：北京市海淀区测试路 12 号；银行卡：4111 1111 1111 1111")
    assert text == "姓名：«姓名»，地址：«地址»；银行卡：«银行卡»"
    assert hits == 3
    assert clean.scrub("编号 4111 1111 1111 1112")[0] == "编号 4111 1111 1111 1112"
    assert clean.scrub("自然语言中的张三未给字段标签")[0] == "自然语言中的张三未给字段标签"


def test_calibration_uses_labels_and_reports_heldout():
    # 人工接口的测试夹具；程序生成的分数只用来检查算法，不作为人工校准证据。
    rows = [{"text": str(i) + "测试文本" * (i + 1), "score": 0.0} for i in range(20)]
    fitted = clean.fit(rows, "synthetic-test-fixture")
    assert fitted["weights"]["punctuation"] == 1
    assert fitted["heldout_count"] == 4 and fitted["train_count"] == 16
    assert fitted["heldout_mse"] == 0 < fitted["default_heldout_mse"]
    assert fitted == clean.fit(rows * 2, "synthetic-test-fixture")  # 重复标注不泄漏进留出集
    with pytest.raises(Failure):
        clean.fit(rows[:2], "test")
    with pytest.raises(Failure):
        clean.fit([{**rows[0], "score": float("nan")}], "test")


def test_profile_controls_weave_and_preserves_difficulty(tmp_path):
    labels = tmp_path / "labels.jsonl"
    labels.write_text("\n".join(json.dumps({"text": str(i) + "测试" * (i+1), "score": 0}) for i in range(12)))
    profile = tmp_path / "quality.json"
    assert main.run(["quality", "fit", str(labels), "--annotator", "test-fixture", "--out", str(profile)]) == 0
    data = tmp_path / "input.jsonl"
    data.write_text(json.dumps({"text": "没有标点", "difficulty": .8}) + "\n" + json.dumps({"text": "保留句号。", "difficulty": .2}))
    out = tmp_path / "out"
    assert main.run(["weave", "--in", str(data), "--out", str(out), "--quality-profile", str(profile), "--min-score", "0.5"]) == 0
    kept = [json.loads(line) for line in (out / "bolt.jsonl").read_text().splitlines()]
    assert len(kept) == 1 and kept[0]["difficulty"] == .2
    with pytest.raises(sample.Refused):
        sample.read({"text": "文本", "difficulty": float("inf")}, sample.Policy())


def test_bad_quality_profile_is_an_input_error(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"weights": {"length": -1, "repeat": 1, "punctuation": 1}}))
    with pytest.raises(Failure):
        clean.profile(path)
