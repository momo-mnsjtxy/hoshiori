"""回归门：掉分与不掉分两种情况的返回码、差异清单，以及命令里的接线。"""

import json

import pytest

import main
from features import gate, report
from utils.fail import Failure


def _doc(*scores):
    """造一份够用的报告数据。不落盘，直接给 check。"""
    rows = [{"suite": "s", "id": f"t{i}", "score": score, "reason": "", "answer": ""} for i, score in enumerate(scores)]
    return {"total": round(sum(scores) / len(scores), 4), "tasks": rows}


def _baseline(tmp_path, payload):
    path = tmp_path / "base.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


# ---------- 比较 ----------

def test_same_total_passes():
    out = gate.check(_doc(1.0, 0.0), _doc(1.0, 0.0), 0.0)
    assert out.ok and out.exit_code == 0 and out.delta == 0.0


def test_small_drop_within_allowance_passes():
    out = gate.check(_doc(1.0, 0.5), _doc(1.0, 1.0), 0.3)  # 掉 0.25
    assert out.ok and out.exit_code == 0


def test_big_drop_fails_and_lists_tasks():
    out = gate.check(_doc(1.0, 0.0), _doc(1.0, 1.0), 0.1)  # 掉 0.5
    assert not out.ok and out.exit_code == 1
    assert out.dropped == ["t1"]


def test_dropped_list_is_sorted_and_ignores_rises():
    out = gate.check(_doc(0.0, 0.0, 1.0), _doc(1.0, 1.0, 0.0), 0.0)
    assert out.dropped == ["t0", "t1"]


def test_zero_allowance_fails_on_any_drop():
    assert gate.check(_doc(0.9), _doc(1.0), 0.0).exit_code == 1


def test_rise_passes_with_negative_delta():
    out = gate.check(_doc(1.0), _doc(0.5), 0.0)
    assert out.ok and out.delta == 0.5 and out.dropped == []


def test_say_names_dropped_tasks():
    said = gate.check(_doc(0.0), _doc(1.0), 0.0).say()
    assert "不过" in said and "t0" in said


def test_new_task_is_not_counted_as_drop():
    current = {"total": 0.75, "tasks": [{"id": "t0", "score": 1.0}, {"id": "new", "score": 0.5}]}
    baseline = {"total": 1.0, "tasks": [{"id": "t0", "score": 1.0}]}
    out = gate.check(current, baseline, 0.3)
    assert out.dropped == [] and out.ok


# ---------- 读基线 ----------

def test_load_missing_file(tmp_path):
    with pytest.raises(Failure) as bad:
        gate.load(tmp_path / "nope.json")
    assert bad.value.kind == "input"


def test_load_broken_json(tmp_path):
    path = tmp_path / "base.json"
    path.write_text("{不好", encoding="utf-8")
    with pytest.raises(Failure):
        gate.load(path)


def test_load_non_report(tmp_path):
    path = tmp_path / "base.json"
    path.write_text('{"a": 1}', encoding="utf-8")
    with pytest.raises(Failure):
        gate.load(path)


def test_load_roundtrip_with_report(tmp_path):
    payload = _doc(1.0)
    payload.update({"schema": "hoshiori/judge/1", "model": "m", "backend": "echo", "suite": "s", "recipe": "sha256:x", "count": 1, "hardware": report.hardware()})
    paths = report.save(payload, tmp_path / "r")
    assert gate.load(paths["json"])["total"] == 1.0


# ---------- 命令里的接线 ----------

def _recipe(tmp_path):
    path = tmp_path / "r.yaml"
    path.write_text("version: 1\ntask: sft\nmodel: {path: base}\nname: demo\n", encoding="utf-8")
    return path


def test_judge_with_baseline_holds_the_line(tmp_path):
    base = _baseline(tmp_path, _doc(0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    out = tmp_path / "r"
    code = main.run(["judge", str(_recipe(tmp_path)), "--suite", "format-lite", "--answer-cmd", "echo 3.14", "--baseline", str(base), "--out", str(out)])
    assert code == 0  # 全答对，比基线高


def test_judge_with_baseline_fails_on_drop(tmp_path):
    base = _baseline(tmp_path, _doc(1.0, 1.0, 1.0, 1.0, 1.0, 1.0))
    out = tmp_path / "r"
    code = main.run(["judge", str(_recipe(tmp_path)), "--suite", "format-lite", "--answer-cmd", "echo 不知道", "--baseline", str(base), "--out", str(out)])
    assert code == 1
    assert (out / "report.json").exists()  # 拦住也照样出报告


def test_judge_missing_baseline_fails(tmp_path):
    out = tmp_path / "r"
    assert main.run(["judge", str(_recipe(tmp_path)), "--suite", "format-lite", "--baseline", str(tmp_path / "no.json"), "--out", str(out)]) == 2
