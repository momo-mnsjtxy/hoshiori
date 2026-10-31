"""lm-eval 导入器：内联示例、外部 jsonl、占位替换、过滤器、认不出的规则、坏文件、导出形状。

最要紧的一条是 test_export_reads_back_by_suite：导出来的文件必须能被 suite.read 装回去，
形状对不上就在这里停，而不是等到 ori judge 真跑的时候才发现。
"""

import pytest

import main
from features import lm_eval, suite
from utils.fail import Failure


def _write(tmp_path, text, name="task.yaml"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


INLINE = """
task: demo
doc_to_text: "问题：{{question}}"
doc_to_target: "{{answer}}"
metric_list:
  - metric: exact_match
samples:
  - {question: 一打鸡蛋是几个, answer: "12"}
  - {question: 太阳从哪边升起, answer: 东}
"""


# ---------- 两种题库来源 ----------

def test_inline_samples_exported(tmp_path):
    one = lm_eval.import_file(_write(tmp_path, INLINE), tmp_path)
    assert one.tasks == 2
    assert one.dropped == []
    assert one.path.name == "lm-eval-demo.yaml"


def test_external_jsonl_loaded(tmp_path):
    (tmp_path / "bank.jsonl").write_text('{"q": "1+1", "a": "2"}\n', encoding="utf-8")
    body = """
task: plus
dataset_path: ./bank.jsonl
doc_to_text: "算：{{q}}"
doc_to_target: "{{a}}"
metric_list: [{metric: exact_match}]
"""
    one = lm_eval.import_file(_write(tmp_path, body), tmp_path)
    assert suite.read(one.path).tasks[0].prompt == "算：1+1"


def test_dataset_kwargs_data_files(tmp_path):
    (tmp_path / "bank.jsonl").write_text('{"q": "2+2", "a": "4"}\n', encoding="utf-8")
    body = """
task: plus
dataset_path: json
dataset_kwargs:
  data_files: {test: ./bank.jsonl}
doc_to_text: "{{q}}"
doc_to_target: "{{a}}"
metric_list: [{metric: exact_match}]
"""
    assert lm_eval.import_file(_write(tmp_path, body), tmp_path).tasks == 1


def test_fewshot_config_samples_and_metric_as_string(tmp_path):
    body = """
task: t
doc_to_text: "{{q}}"
doc_to_target: "{{a}}"
metric_list: [exact_match]
fewshot_config:
  samples: [{q: 问, a: 答}]
"""
    task = suite.read(lm_eval.import_file(_write(tmp_path, body), tmp_path).path).tasks[0]
    assert task.prompt == "问"
    assert task.rule == "exact"


def test_no_question_bank(tmp_path):
    body = ("task: x\ndoc_to_text: '{{q}}'\ndoc_to_target: '{{a}}'\n"
            "metric_list: [{metric: exact_match}]\n")
    with pytest.raises(Failure) as bad:
        lm_eval.import_file(_write(tmp_path, body), tmp_path)
    assert bad.value.kind == "input"


# ---------- 占位与过滤器 ----------

def test_field_placeholders_replaced(tmp_path):
    body = """
task: t
doc_to_text: "问：{{question}}"
doc_to_target: "{{answer}}"
metric_list: [{metric: exact_match}]
samples: [{question: 天是什么色, answer: 蓝}]
"""
    task = suite.read(lm_eval.import_file(_write(tmp_path, body), tmp_path).path).tasks[0]
    assert task.prompt == "问：天是什么色"
    assert task.answer == "蓝"


def test_filters_lower_upper_strip(tmp_path):
    body = """
task: t
doc_to_text: "{{name|upper}}/{{name|lower}}/{{spaced|strip}}"
doc_to_target: "{{name|upper}}"
metric_list: [{metric: exact_match}]
samples: [{name: Ab, spaced: "  留白  "}]
"""
    task = suite.read(lm_eval.import_file(_write(tmp_path, body), tmp_path).path).tasks[0]
    assert task.prompt == "AB/ab/留白"
    assert task.answer == "AB"


def test_filter_chain_in_order(tmp_path):
    body = """
task: t
doc_to_text: "{{name|strip|upper}}"
doc_to_target: "{{name}}"
metric_list: [{metric: exact_match}]
samples: [{name: " ab "}]
"""
    task = suite.read(lm_eval.import_file(_write(tmp_path, body), tmp_path).path).tasks[0]
    assert task.prompt == "AB"


def test_unknown_filter_kept_and_noted(tmp_path):
    body = """
task: t
doc_to_text: "{{name|title}}"
doc_to_target: "{{name}}"
metric_list: [{metric: exact_match}]
samples: [{name: ab}]
"""
    one = lm_eval.import_file(_write(tmp_path, body), tmp_path)
    assert suite.read(one.path).tasks[0].prompt == "ab"
    assert any("title" in line for line in one.notes)


def test_missing_field_renders_empty_and_noted(tmp_path):
    body = """
task: t
doc_to_text: "{{nope}}尾"
doc_to_target: "{{answer}}"
metric_list: [{metric: exact_match}]
samples: [{answer: 答}]
"""
    one = lm_eval.import_file(_write(tmp_path, body), tmp_path)
    assert suite.read(one.path).tasks[0].prompt == "尾"
    assert any("nope" in line for line in one.notes)


def test_doc_to_choice_appended(tmp_path):
    body = """
task: t
doc_to_text: "选一个：{{q}}"
doc_to_target: "{{a}}"
doc_to_choice: "{{opts}}"
metric_list: [{metric: exact_match}]
samples: [{q: 颜色, a: 红, opts: ["红", "蓝"]}]
"""
    one = lm_eval.import_file(_write(tmp_path, body), tmp_path)
    prompt = suite.read(one.path).tasks[0].prompt
    assert "A. 红" in prompt and "B. 蓝" in prompt
    assert any("doc_to_choice" in line for line in one.notes)


# ---------- 规则映射与丢弃 ----------

def test_contains_splits_target(tmp_path):
    body = """
task: t
doc_to_text: "{{q}}"
doc_to_target: "{{a}}"
metric_list: [{metric: contains}]
samples: [{q: 水果, a: "红;蓝"}]
"""
    task = suite.read(lm_eval.import_file(_write(tmp_path, body), tmp_path).path).tasks[0]
    assert task.rule == "contains"
    assert task.keywords == ["红", "蓝"]


def test_regex_maps_to_pattern(tmp_path):
    body = r"""
task: t
doc_to_text: "{{q}}"
doc_to_target: "{{a}}"
metric_list: [{metric: regex}]
samples: [{q: 数, a: '^\d+$'}]
"""
    task = suite.read(lm_eval.import_file(_write(tmp_path, body), tmp_path).path).tasks[0]
    assert task.rule == "regex"
    assert task.pattern == "^\\d+$"


def test_unknown_metric_goes_to_report(tmp_path):
    body = """
task: t
doc_to_text: "{{q}}"
doc_to_target: "{{a}}"
metric_list: [{metric: bleu}, {metric: rouge}]
samples: [{q: 问, a: 答}]
"""
    one = lm_eval.import_file(_write(tmp_path, body), tmp_path)
    assert one.dropped == ["bleu", "rouge"]
    assert one.tasks == 1
    text = one.report.read_text(encoding="utf-8")
    assert "bleu" in text and "rouge" in text
    assert suite.read(one.path).tasks[0].rule == "exact"


def test_known_metric_wins_over_unknown(tmp_path):
    body = """
task: t
doc_to_text: "{{q}}"
doc_to_target: "{{a}}"
metric_list: [{metric: bleu}, {metric: regex}]
samples: [{q: 问, a: "答"}]
"""
    one = lm_eval.import_file(_write(tmp_path, body), tmp_path)
    assert one.dropped == ["bleu"]
    assert suite.read(one.path).tasks[0].rule == "regex"


# ---------- 坏文件 ----------

def test_bad_yaml_reported(tmp_path):
    with pytest.raises(Failure) as bad:
        lm_eval.import_file(_write(tmp_path, "task: [x\n"), tmp_path)
    assert bad.value.kind == "input"


def test_bad_jsonl_reported_with_line(tmp_path):
    (tmp_path / "bank.jsonl").write_text('{"q": 1}\n{oops\n', encoding="utf-8")
    body = "task: t\ndataset_path: ./bank.jsonl\ndoc_to_text: '{{q}}'\ndoc_to_target: '{{a}}'\n"
    with pytest.raises(Failure) as bad:
        lm_eval.import_file(_write(tmp_path, body), tmp_path)
    assert "第 2 行" in bad.value.message


def test_empty_question_bank_reported(tmp_path):
    (tmp_path / "bank.jsonl").write_text("\n\n", encoding="utf-8")
    body = "task: t\ndataset_path: ./bank.jsonl\ndoc_to_text: '{{q}}'\ndoc_to_target: '{{a}}'\n"
    with pytest.raises(Failure):
        lm_eval.import_file(_write(tmp_path, body), tmp_path)


# ---------- 导出形状 ----------

def test_export_reads_back_by_suite(tmp_path):
    one = lm_eval.import_file(_write(tmp_path, INLINE), tmp_path)
    back = suite.read(one.path)
    assert back.name == "lm-eval-demo"
    assert [task.rule for task in back.tasks] == ["exact", "exact"]
    assert back.tasks[0].prompt == "问题：一打鸡蛋是几个"
    assert back.tasks[0].answer == "12"


def test_duplicate_record_ids_not_collide(tmp_path):
    body = """
task: t
doc_to_text: "{{q}}"
doc_to_target: "{{a}}"
metric_list: [{metric: exact_match}]
samples: [{id: same, q: 一, a: 1}, {id: same, q: 二, a: 2}]
"""
    back = suite.read(lm_eval.import_file(_write(tmp_path, body), tmp_path).path)
    ids = [task.id for task in back.tasks]
    assert len(set(ids)) == 2


def test_report_written_next_to_suite(tmp_path):
    one = lm_eval.import_file(_write(tmp_path, INLINE), tmp_path)
    assert one.report.exists()
    assert one.report.name == "lm-eval-demo.report.md"


# ---------- 命令接得上 ----------

def test_judge_import_flag_writes_builtin_folder(tmp_path):
    source = _write(tmp_path, INLINE, name="cli.yaml")
    made = [suite.folder() / "lm-eval-demo.yaml", suite.folder() / "lm-eval-demo.report.md"]
    try:
        assert main.run(["judge", "--import-lm-eval", str(source)]) == 0
        assert made[0].exists() and made[1].exists()
        assert len(suite.read(made[0]).tasks) == 2
    finally:
        for path in made:  # 内置目录留着干净，test_suite 的清单才对得上
            path.unlink(missing_ok=True)
