"""套件装载、坏文件报错、三种判分规则的边界，以及 judge 命令的落地。"""

import json

import pytest

import main
from features import scorer, suite
from utils.fail import Failure


# ---------- 内置套件 ----------

def test_builtin_suites_scanned():
    found = suite.scan(suite.folder())
    assert [one.name for one in found] == ["format-lite.yaml", "zh-lite.yaml"]


def test_builtin_suites_have_expected_size():
    counts = {one.name: len(suite.read(one).tasks) for one in suite.scan(suite.folder())}
    assert counts == {"format-lite.yaml": 6, "zh-lite.yaml": 10}


def test_builtin_tasks_carry_their_rule_field():
    for path in suite.scan(suite.folder()):
        for task in suite.read(path).tasks:
            assert task.prompt
            assert task.rule in suite.RULES
            need = suite.RULES[task.rule]
            assert getattr(task, need)


def test_builtin_suites_grade_correct_answers():
    right = {
        "units": "12 个",
        "nasal": "后鼻音",
        "fruits": "苹果、香蕉、橘子",
        "json-name": '{"name": "星织"}',
        "five-chars": "今天吃饭了",
        "markdown-list": "- 红色\n- 蓝色",
        "kilometer": "1000",
        "upper": "HOSHIORI",
        "kv-line": "lang: zh",
        "sort-numbers": "1, 2, 3",
        "json-array": "[1, 2, 3]",
        "json-bool": "false",
        "json-ok": '{"ok": true}',
        "gram": "3500",
        "pipe-path": "a|b|c",
        "pi-two": "3.14",
    }
    for path in suite.scan(suite.folder()):
        for task in suite.read(path).tasks:
            assert scorer.grade(task, right[task.id]).score == 1.0


def test_locate_by_name_with_and_without_suffix():
    where = suite.folder()
    assert suite.locate(where, "zh-lite").name == "zh-lite.yaml"
    assert suite.locate(where, "zh-lite.yaml").name == "zh-lite.yaml"


def test_locate_unknown_name_points_at_dir():
    with pytest.raises(Failure) as bad:
        suite.locate(suite.folder(), "nope")
    assert bad.value.kind == "input"
    assert "zh-lite" in bad.value.hint


def test_scan_missing_dir(tmp_path):
    with pytest.raises(Failure) as bad:
        suite.scan(tmp_path / "nope")
    assert bad.value.kind == "missing"


# ---------- 坏文件 ----------

def _write(tmp_path, text, name="bad.yaml"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_missing_tasks(tmp_path):
    with pytest.raises(Failure) as bad:
        suite.read(_write(tmp_path, "name: x\n"))
    assert "tasks" in bad.value.message


def test_empty_tasks(tmp_path):
    with pytest.raises(Failure):
        suite.read(_write(tmp_path, "name: x\ntasks: []\n"))


def test_task_not_a_mapping(tmp_path):
    with pytest.raises(Failure) as bad:
        suite.read(_write(tmp_path, "tasks:\n  - 一句话\n"))
    assert "第 1 题" in bad.value.message


def test_missing_id(tmp_path):
    with pytest.raises(Failure) as bad:
        suite.read(_write(tmp_path, "tasks:\n  - {prompt: 问, rule: exact, answer: 答}\n"))
    assert "id" in bad.value.message


def test_duplicate_id(tmp_path):
    body = "tasks:\n  - {id: a, prompt: 问, rule: exact, answer: 答}\n  - {id: a, prompt: 问, rule: exact, answer: 答}\n"
    with pytest.raises(Failure) as bad:
        suite.read(_write(tmp_path, body))
    assert "重了" in bad.value.message


def test_missing_prompt(tmp_path):
    with pytest.raises(Failure):
        suite.read(_write(tmp_path, "tasks:\n  - {id: a, rule: exact, answer: 答}\n"))


def test_unknown_rule(tmp_path):
    with pytest.raises(Failure) as bad:
        suite.read(_write(tmp_path, "tasks:\n  - {id: a, prompt: 问, rule: fuzzy, answer: 答}\n"))
    assert "exact" in bad.value.hint


def test_contains_needs_keywords(tmp_path):
    with pytest.raises(Failure):
        suite.read(_write(tmp_path, "tasks:\n  - {id: a, prompt: 问, rule: contains}\n"))


def test_exact_needs_answer(tmp_path):
    with pytest.raises(Failure) as bad:
        suite.read(_write(tmp_path, "tasks:\n  - {id: a, prompt: 问, rule: exact}\n"))
    assert "answer" in bad.value.message


def test_broken_pattern_caught_at_load(tmp_path):
    with pytest.raises(Failure) as bad:
        suite.read(_write(tmp_path, "tasks:\n  - {id: a, prompt: 问, rule: regex, pattern: '([a'}\n"))
    assert "pattern" in bad.value.message


def test_name_defaults_to_file_stem(tmp_path):
    one = suite.read(_write(tmp_path, "tasks:\n  - {id: a, prompt: 问, rule: exact, answer: 答}\n", name="mini.yaml"))
    assert one.name == "mini"


def test_json_suite_loads(tmp_path):
    body = {"tasks": [{"id": "a", "prompt": "问", "rule": "exact", "answer": "答"}]}
    path = _write(tmp_path, json.dumps(body), name="mini.json")
    assert suite.read(path).tasks[0].answer == "答"


# ---------- 判分边界 ----------

def _task(**over):
    base = {"id": "t", "prompt": "问", "rule": "exact", "answer": "答"}
    base.update(over)
    return suite.Task(**base)


def test_exact_ignores_all_whitespace():
    task = _task(answer="1, 2, 3")
    assert scorer.grade(task, "1,2,3").score == 1.0
    assert scorer.grade(task, " 1 ,\n2,\t3 ").score == 1.0


def test_exact_rejects_near_miss():
    assert scorer.grade(_task(answer="12"), "12 个").score == 0.0
    assert scorer.grade(_task(answer="abc"), "ABC").score == 0.0


def test_contains_full_half_none():
    task = _task(rule="contains", keywords=["红", "蓝"])
    assert scorer.grade(task, "红色和蓝色").score == 1.0
    assert scorer.grade(task, "只有红色").score == 0.5
    assert scorer.grade(task, "都没有").score == 0.0


def test_contains_reason_names_missing():
    task = _task(rule="contains", keywords=["红", "蓝"])
    assert "蓝" in scorer.grade(task, "红色").reason


def test_regex_matches_anywhere_in_answer():
    task = _task(rule="regex", pattern=r"\d{3}")
    assert scorer.grade(task, "答案是 123 米").score == 1.0


def test_regex_multiline_and_dotall():
    task = _task(rule="regex", pattern=r"(?m)^-\s+\S+\n-\s+\S+")
    assert scorer.grade(task, "- 红\n- 蓝").score == 1.0
    span = _task(rule="regex", pattern=r"a.b")
    assert scorer.grade(span, "a\nb").score == 1.0


def test_regex_no_match():
    assert scorer.grade(_task(rule="regex", pattern=r"^\d+$"), "十二").score == 0.0


def test_scores_stay_in_range():
    for answer in ["", "答", "不知道" * 10]:
        for task in [_task(), _task(rule="contains", keywords=["答"]), _task(rule="regex", pattern="答")]:
            assert 0.0 <= scorer.grade(task, answer).score <= 1.0


# ---------- 命令接得上 ----------

def _recipe(tmp_path):
    path = tmp_path / "r.yaml"
    path.write_text("version: 1\ntask: sft\nmodel: {path: base}\nname: demo\n", encoding="utf-8")
    return path


def test_judge_writes_three_reports(tmp_path):
    out = tmp_path / "r"
    code = main.run(["judge", str(_recipe(tmp_path)), "--suite", "format-lite", "--answer-cmd", "echo 3.14", "--out", str(out)])
    assert code == 0
    for name in ["report.md", "report.json", "report.csv"]:
        assert (out / name).exists()
    data = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert data["backend"] == "echo 3.14"
    assert data["suite"] == "format-lite"
    assert data["count"] == 6
    assert data["hardware"]["cpu"]


def test_judge_echo_backend_marked(tmp_path):
    out = tmp_path / "r"
    assert main.run(["judge", str(_recipe(tmp_path)), "--suite", "format-lite", "--limit", "1", "--out", str(out)]) == 0
    data = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert data["backend"] == "echo"
    assert data["count"] == 1


def test_judge_limit_zero_means_all(tmp_path):
    out = tmp_path / "r"
    main.run(["judge", str(_recipe(tmp_path)), "--suite", "zh-lite", "--limit", "0", "--out", str(out)])
    assert json.loads((out / "report.json").read_text(encoding="utf-8"))["count"] == 10
