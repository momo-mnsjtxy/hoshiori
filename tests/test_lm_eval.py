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


# ---------- 导入后真跑 ----------

def test_run_backend_scores_imported_suite(tmp_path):
    imported = lm_eval.import_file(_write(tmp_path, INLINE), tmp_path)
    result = lm_eval.run_backend("printf 12", suite=suite.read(imported.path), limit=1)
    assert result["count"] == 1
    assert result["total"] == 1.0
    assert result["tasks"][0]["answer"] == "12"


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


def test_fewshot_prefix_is_sent_without_evaluation_answer_leak(tmp_path):
    """评分题答案留在判分字段，示范问题和答案进入真正发送的提示词。"""
    body = """
task: shots
num_fewshot: 1
target_delimiter: ' => '
fewshot_delimiter: '\n\n'
doc_to_text: '{{q}}'
doc_to_target: '{{a}}'
samples: [{q: test-question, a: secret-answer}]
fewshot_config:
  samples: [{q: demo-question, a: demo-answer}]
"""
    one = lm_eval.import_file(_write(tmp_path, body), tmp_path)
    task = suite.read(one.path).tasks[0]
    assert "demo-question => demo-answer" in task.prompt
    assert task.prompt.endswith("test-question") and "secret-answer" not in task.prompt
    assert task.answer == "secret-answer"
    assert any("1-shot" in line for line in one.notes)


def test_hf_dataset_and_training_split_are_separate(tmp_path, monkeypatch):
    """Datasets 边界传入仓库、配置、revision 与正确 split，测试集不会被示范覆盖。"""
    import sys
    from types import SimpleNamespace
    calls = []

    def load_dataset(path, name, split, **options):
        calls.append((path, name, split, options))
        return [{"q": "held-out", "a": "held-out-label"}] if split == "validation" else [{"q": "demo", "a": "demo-label"}]

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=load_dataset))
    body = """
task: hub
dataset_path: organization/data
dataset_name: subset
validation_split: validation
training_split: train
dataset_kwargs: {revision: pinned-commit}
num_fewshot: 1
doc_to_text: '{{q}}'
doc_to_target: '{{a}}'
"""
    imported = lm_eval.import_file(_write(tmp_path, body), tmp_path)
    task = suite.read(imported.path).tasks[0]
    assert calls == [("organization/data", "subset", "validation", {"revision": "pinned-commit"}),
                     ("organization/data", "subset", "train", {"revision": "pinned-commit"})]
    assert "demo-label" in task.prompt and "held-out-label" not in task.prompt
    assert task.answer == "held-out-label"


def test_fewshot_never_reuses_current_question(tmp_path):
    """只有自己可供示范时明确报错，不能把正确答案偷偷拼到问题前面。"""
    body = "task: leak\nnum_fewshot: 1\nsamples: [{input: q, target: a}]\n"
    with pytest.raises(Failure, match="无泄漏示范"):
        lm_eval.import_file(_write(tmp_path, body), tmp_path)


def test_local_split_selection_prefers_test(tmp_path):
    """data_files 字典里 train 排在前面也不会被误当测试集。"""
    (tmp_path / "train.jsonl").write_text('{"input":"train","target":"train-label"}\n')
    (tmp_path / "test.jsonl").write_text('{"input":"test","target":"test-label"}\n')
    body = "task: split\ndataset_path: json\ndataset_kwargs:\n  data_files: {train: train.jsonl, test: test.jsonl}\n"
    task = suite.read(lm_eval.import_file(_write(tmp_path, body), tmp_path).path).tasks[0]
    assert task.prompt == "test" and task.answer == "test-label"


def test_missing_test_split_never_falls_back_to_only_training_file(tmp_path):
    """显式 test 不存在时必须失败，不能因只有一份 train 就自动拿来评分。"""
    (tmp_path / "train.jsonl").write_text('{"input":"seen","target":"seen-label"}\n')
    body = "task: wrong-split\ndataset_path: json\ntest_split: test\ndataset_kwargs:\n  data_files: {train: train.jsonl}\n"
    with pytest.raises(Failure, match="没有 split test"):
        lm_eval.import_file(_write(tmp_path, body), tmp_path)


# ---------- 外部题面只能作为参数 ----------

def test_backend_receives_shell_characters_without_executing(tmp_path):
    import shlex  # 这里只引用可信 Python 路径和固定程序
    import sys  # 让测试与当前解释器使用同一版本

    marker = tmp_path / "command-executed"
    prompt = f'引号 " 与单引号 \'\n第二行 $(touch {marker}) `touch {marker}` ; touch {marker}; # {{model}}'
    program = "import sys; sys.stdout.write(sys.argv[1])"  # 程序自身固定，外部文本只在 argv[1]
    template = f"{shlex.quote(sys.executable)} -c {shlex.quote(program)} {{prompt}}"
    assert lm_eval.answer(template, prompt) == prompt  # 换行、引号与占位字面全部原样到达后端
    assert not marker.exists()  # shell 命令替换和分号都没有执行
    assert lm_eval.answer("echo {prompt}", "ordinary words").strip() == "ordinary words"  # 原有回声命令照常可用


@pytest.mark.parametrize("template", [
    "python -c 'print({prompt})'",
    "python -c'print({prompt})'",
    'sh -c "{prompt}"',
    'bash -lc "{prompt}"',
    'bash --command="{prompt}"',
])
def test_backend_rejects_direct_code_interpolation(template):
    with pytest.raises(Failure, match="不要把占位符"):
        lm_eval.answer(template, "untrusted text")  # 不让题面落入已知直接代码参数


def test_backend_reports_invalid_template_and_missing_program():
    with pytest.raises(Failure, match="不能启动"):
        lm_eval.answer("echo 'unterminated", "text")  # 模板引号错误不能在主入口变成未知崩溃
    with pytest.raises(Failure, match="不能启动"):
        lm_eval.answer("/does-not-exist/ori-backend {prompt}", "text")  # shell=False 的 OSError 仍走原来的 input 出口
