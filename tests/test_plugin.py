"""插件扩展点测试：覆盖目录扫描、坏文件隔离、四类样例与命令行出口。

调用方式：在项目根目录运行 python -m pytest tests/test_plugin.py -q。
调用示例：python -m pytest tests/test_plugin.py::test_examples_are_scanned -q。
"""

import json                                                # 造 CLI 参数并读取 JSONL

import pytest                                              # 断言警告和异常

import main                                                # 从真实入口检查退出码
from features import plugin                               # 直接检查扫描与加载
from utils.fail import Failure                            # 未知名字应走 input 错误


def test_examples_are_scanned():
    """四类目录都能自动发现各自的示例。"""
    found = {kind: {item.NAME for item in plugin.scan(kind)} for kind in plugin.kinds()}  # 每类名字集合
    assert "lines" in found["sources"]                    # 文本来源示例
    assert "length_penalty" in found["rewards"]          # 长度奖励示例
    assert "char_hit" in found["metrics"]                # 字符指标示例
    assert "jsonl" in found["exporters"]                 # 报告导出示例


def test_bad_file_only_warns_and_keeps_others(tmp_path, monkeypatch):
    """一个文件导入失败时，旁边的好插件仍会出现。"""
    folder = tmp_path / "sources"                         # 只造本用例需要的一类
    folder.mkdir()                                         # 临时目录与真实布局相同
    (folder / "good.py").write_text(                      # 一份契约完整的插件
        'NAME = "good"\nSUMMARY = "能用"\ndef run():\n    return 1\n', encoding="utf-8"
    )
    (folder / "bad.py").write_text("raise RuntimeError('坏掉')\n", encoding="utf-8")  # 导入时直接失败
    monkeypatch.setattr(plugin, "ROOT", tmp_path)          # 扫描临时目录，不碰内置插件
    with pytest.warns(UserWarning, match="bad.py"):        # 坏文件只给一条警告
        found = plugin.scan("sources")                     # 扫描仍会走到 good.py
    assert [item.NAME for item in found] == ["good"]       # 好插件照常留下


def test_load_unknown_name_is_input_failure():
    """未知名字给用户可改的 input 错误。"""
    with pytest.raises(Failure) as bad:                     # load 自己说明可选名字
        plugin.load("metrics", "missing")                  # 这一名字不在目录里
    assert bad.value.kind == "input"                       # main 会把它翻成退出码 2
    assert "char_hit" in bad.value.hint                    # 提示里列出现有插件


def test_source_example_runs(tmp_path):
    """来源插件按行返回文本。"""
    path = tmp_path / "raw.txt"                            # 一份两行文本
    path.write_text("甲\n乙\n", encoding="utf-8")         # 最后一行也带换行
    assert plugin.load("sources", "lines").run(path=str(path)) == ["甲", "乙"]  # 行顺序不变


def test_metric_example_runs():
    """指标插件返回 0～1 的字符重合率。"""
    score = plugin.load("metrics", "char_hit").run(answer="星织", reference="星绒")  # 命中一半
    assert score == 0.5                                    # 两个字符里重合一个


def test_reward_example_runs():
    """奖励插件只扣超出长度的部分。"""
    reward = plugin.load("rewards", "length_penalty").run(answer="abcdef", limit=3, rate=0.5)  # 超三字
    assert reward == -1.5                                  # 三个字符各扣 0.5


def test_exporter_example_runs(tmp_path):
    """导出插件写成一行 JSONL。"""
    path = tmp_path / "report.jsonl"                       # 输出落在临时目录
    result = plugin.load("exporters", "jsonl").run(report={"total": 1.0}, path=str(path))  # 写一份报告
    assert result == str(path)                              # 返回实际输出路径
    assert json.loads(path.read_text(encoding="utf-8")) == {"total": 1.0}  # 一行能读回原报告


def test_cli_list_exit_code(capsys):
    """plugin list 成功并列出四类示例。"""
    assert main.run(["plugin", "list"]) == 0              # 正常列清单返回 0
    output = capsys.readouterr().out                        # 收下命令行输出
    for text in ["sources:", "lines", "rewards:", "length_penalty", "metrics:", "char_hit", "exporters:", "jsonl"]:
        assert text in output                               # 类别与名字都看得到


def test_cli_runs_four_examples(tmp_path, capsys):
    """四类示例都能从真实入口运行一次并返回 0。"""
    source = tmp_path / "raw.txt"                          # 来源插件的输入
    source.write_text("甲\n乙\n", encoding="utf-8")       # 两行足够验证
    target = tmp_path / "report.jsonl"                     # 导出插件的落点
    calls = [                                               # 四类各一条真实命令
        ["sources", "lines", {"path": str(source)}],
        ["metrics", "char_hit", {"answer": "星织", "reference": "星绒"}],
        ["rewards", "length_penalty", {"answer": "abcdef", "limit": 3, "rate": 0.5}],
        ["exporters", "jsonl", {"report": {"total": 1.0}, "path": str(target)}],
    ]
    for kind, name, values in calls:                        # 逐条从 main.run 进入
        code = main.run(["plugin", "run", kind, name, "--args", json.dumps(values, ensure_ascii=False)])
        assert code == 0                                    # 每类正常运行都返回 0
    assert target.exists()                                  # 导出插件确实写了文件
    assert capsys.readouterr().err == ""                    # 正常运行没有错误输出


def test_cli_bad_json_exits_two(capsys):
    """--args 解析失败走 input 的退出码 2。"""
    code = main.run(["plugin", "run", "metrics", "char_hit", "--args", "{bad"])  # 故意少引号和右括号
    assert code == 2                                        # 用户输入错误的统一退出码
    assert "--args 不是合法 JSON" in capsys.readouterr().err  # 原因直接可改
