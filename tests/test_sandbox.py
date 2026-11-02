"""奖励沙箱的测试：正常分数与五类失败都留在子进程，训练器能用它覆盖旧奖励跑一步。

调用示例：python -m pytest tests/test_sandbox.py -q
"""

import json
import math

import main
from features import model, sandbox, trainer


def _code(tmp_path, body: str):
    """写一份最小奖励文件，供真实子进程载入。"""
    path = tmp_path / "reward.py"                                                # 每个测试都有自己的临时目录
    path.write_text(body, encoding="utf-8")                                      # 内容就是外部用户代码
    return path                                                                   # 交给沙箱的公开入口


def test_returns_number(tmp_path):
    path = _code(tmp_path, "def reward(payload):\n    return payload['value'] * 2\n")  # 读取 JSON payload 算分
    score, error = sandbox.run(path, {"value": 1.25})                            # 真起隔离子进程
    assert score == 2.5 and error is None                                         # 成功只返回有限浮点数


def test_timeout_is_error(tmp_path):
    path = _code(tmp_path, "def reward(payload):\n    while True:\n        pass\n")    # 持续占 CPU，等父进程到点杀
    score, error = sandbox.run(path, {}, timeout=0.2)                              # 墙钟上限故意压短
    assert score is None and "超时" in error                                      # 主进程没有被忙循环拖住


def test_exception_is_error(tmp_path):
    path = _code(tmp_path, "def reward(payload):\n    raise ValueError('坏奖励')\n")  # 外部异常只在子进程出现
    score, error = sandbox.run(path, {})                                          # 调用方仍拿普通返回值
    assert score is None and "执行异常" in error and "ValueError: 坏奖励" in error  # 类型和原因都留得住


def test_non_number_is_error(tmp_path):
    path = _code(tmp_path, "def reward(payload):\n    return {'score': 1}\n")       # 映射不能当训练分数
    score, error = sandbox.run(path, {})                                          # 类型检查发生在隔离进程
    assert score is None and "不是数字" in error                                  # 错误类别明确


def test_non_finite_is_error(tmp_path):
    path = _code(tmp_path, "def reward(payload):\n    return float('nan')\n")       # NaN 会污染整条损失
    score, error = sandbox.run(path, {})                                          # 有数字类型仍要查有限性
    assert score is None and "不是有限数" in error                                # 与非数字分开报告


def test_memory_limit_is_error(tmp_path):
    path = _code(tmp_path, "def reward(payload):\n    data = bytearray(256 * 1024 * 1024)\n    return len(data)\n")  # 明显超过测试上限
    score, error = sandbox.run(path, {}, memory_mb=64)                            # Python 本身仍装得进 64 MB
    assert score is None and "内存超限" in error                                  # MemoryError 或资源信号归为同一类


def test_spin_checks_reward_file(tmp_path, capsys):
    path = tmp_path / "group.warp.yaml"                                          # 相对奖励路径要按这份配方的位置查
    path.write_text(                                                             # 在建模型前就应发现文件缺失
        "version: 1\ntask: rloo\nmodel:\n  path: base\ntrain:\n  reward: missing.py\n",
        encoding="utf-8",
    )
    assert main.run(["spin", str(path), "--dry-run"]) == 2                     # 文件问题属于用户输入错误
    assert "奖励文件不在" in capsys.readouterr().err                              # 原因直接指向 reward 文件


def test_trainer_uses_sandbox_reward_for_one_step(tmp_path):
    path = _code(                                                                 # 当前候选末字决定外部奖励
        tmp_path,
        "def reward(payload):\n    return 2.0 if payload['response'].endswith('甲') else -1.0\n",
    )
    data = tmp_path / "groups.jsonl"                                             # 原 rewards 故意与外部结果不同
    row = {"prompt": "织：", "responses": ["候选甲", "候选乙"], "rewards": [99, 99]}
    data.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")  # 一行就是一组
    geo = model.Geometry(d_model=16, layers=1, heads=1, d_head=16, vocab=64, max_seq_len=16)  # CPU 秒过的小模型
    choice = trainer.Choice(                                                      # RLOO 不需要额外参考模型
        out=tmp_path / "out",
        files=[data],
        task="rloo",
        group=2,
        steps=1,
        save_every=0,
        reward=path,
    )
    job = trainer.Trainer(model.Model(geo), geo, choice)                          # 读数据时就在沙箱里重算两条奖励
    assert job.samples[0][1] == [2.0, -1.0]                                      # 数据里的 99 已被完整覆盖
    assert job.run() == 1 and math.isfinite(job.history[-1])                     # 新奖励真能参与一步反向
