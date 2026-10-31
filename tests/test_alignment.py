"""对齐损失的测试：五种损失各自一个数值用例，外加偏好对的读法与缺字段。

约定：
  数组都手写几行，断言全是能手算出来的数：DPO 平局是 ln2、GRPO 优势组内均值为 0、
  裁剪在比率出区间时生效、蒸馏温度趋 1 时贴近 KL、留一基线不含自己。
  torch 只挑一条验张量能过、能反传，其余走 numpy，缺 torch 也不影响这批用例。
"""

import json
import math

import numpy as np
import pytest

from features import alignment
from utils.fail import Failure


# ---------- 偏好数据 ----------

def test_pairs_maps_each_row_to_two_samples(tmp_path):
    path = tmp_path / "pref.jsonl"
    row = {"prompt": "问：", "chosen": "好答案", "rejected": "差答案"}
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    got = alignment.pairs([path])
    assert got == [({"text": "问：好答案"}, {"text": "问：差答案"})]  # 上下文接在两边前面


def test_pairs_accepts_dict_list():
    got = alignment.pairs([{"chosen": "a", "rejected": "b"}])
    assert got == [({"text": "a"}, {"text": "b"})]  # 已经解析好的也接


def test_pairs_missing_field_fails():
    with pytest.raises(Failure) as err:
        alignment.pairs([{"chosen": "a"}])
    assert err.value.kind == "input" and "rejected" in err.value.message


# ---------- DPO ----------

def test_dpo_tie_is_ln2():
    logp = np.array([-1.0, -2.0])
    out = alignment.dpo(logp, logp, logp, logp, beta=0.1)  # 两边完全一样
    assert float(out["loss"]) == pytest.approx(math.log(2))  # 平局就是 ln2
    assert float(out["accuracy"]) == pytest.approx(0.5)  # 平局算半条


def test_dpo_prefers_chosen():
    chosen = np.array([-0.5])
    rejected = np.array([-1.5])
    ref = np.array([-1.0])
    out = alignment.dpo(chosen, rejected, ref, ref, beta=0.5)  # chosen 比 rejected 高 1
    assert float(out["loss"]) < math.log(2)  # 拉到比平局好
    assert float(out["accuracy"]) == 1.0  # 每条都赢


def test_dpo_beta_steepens():
    chosen, rejected, ref = np.array([0.0]), np.array([-1.0]), np.array([0.0])
    flat = float(alignment.dpo(chosen, rejected, ref, ref, beta=0.1)["loss"])
    steep = float(alignment.dpo(chosen, rejected, ref, ref, beta=1.0)["loss"])
    assert steep < flat  # beta 越大，同样的差距罚得越轻


# ---------- GRPO ----------

def test_grpo_advantages_mean_zero_per_group():
    logps = np.zeros((2, 4))
    rewards = np.array([[1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 1.0, 1.0]])
    out = alignment.grpo(logps, rewards, ref_logps=logps, clip=0.2, kl_coef=0.0)
    assert np.allclose(out["advantages"].mean(axis=-1), 0.0)  # 每组归一后均值为 0
    assert float(out["loss"]) == pytest.approx(0.0)  # 比率为 1、KL 不罚，损失就是 0
    assert np.allclose(out["kl"], 0.0)  # 参考模型一样，KL 是 0


def test_grpo_kl_penalty_is_positive():
    logps = np.zeros((1, 2))
    rewards = np.array([[0.0, 1.0]])
    ref = np.full((1, 2), -1.0)
    out = alignment.grpo(logps, rewards, ref_logps=ref, kl_coef=0.1)
    assert float(out["kl"]) > 0  # 参考模型更低，KL 是正的
    assert float(out["loss"]) > 0.1 * float(out["kl"]) * 0.9  # 罚项进了损失


# ---------- RLOO ----------

def test_rloo_baseline_excludes_self():
    logps = np.zeros((1, 3))
    rewards = np.array([[1.0, 2.0, 3.0]])
    out = alignment.rloo(logps, rewards)
    assert np.allclose(out["baseline"], [[2.5, 2.0, 1.5]])  # 基线是另外两条的平均
    assert np.allclose(out["advantages"], [[-1.5, 0.0, 1.5]])  # 优势对称
    assert np.allclose(out["advantages"].mean(axis=-1), 0.0)  # 组内也归零


# ---------- PPO ----------

def test_ppo_clips_out_of_range_ratio():
    logps = np.array([math.log(3.0)])  # 比率 3
    old = np.zeros(1)
    advantage = np.array([1.0])
    out = alignment.ppo(logps, old, advantage, clip=0.2)
    assert float(out["ratio"][0]) == pytest.approx(3.0)  # 比率照实算
    assert bool(out["clipped"][0])  # 3 超过了 1.2
    assert float(out["loss"]) == pytest.approx(-1.2)  # 损失按边界 1.2 算，不按 3


def test_ppo_keeps_in_range_ratio():
    logps = np.array([math.log(1.1)])  # 比率 1.1，在区间内
    out = alignment.ppo(logps, np.zeros(1), np.array([1.0]), clip=0.2)
    assert not bool(out["clipped"][0])  # 没被裁
    assert float(out["loss"]) == pytest.approx(-1.1)  # 原样出力


# ---------- 蒸馏 ----------

def test_distill_matches_kl_at_temperature_one():
    teacher = np.array([[2.0, 1.0, 0.0]])
    student = np.array([[0.5, 1.5, 0.0]])
    p = np.exp(teacher) / np.exp(teacher).sum(axis=-1, keepdims=True)  # 教师分布
    q = np.exp(student) / np.exp(student).sum(axis=-1, keepdims=True)  # 学生分布
    want = float((p * np.log(p / q)).sum())  # 手算的 KL(教师‖学生)
    got = float(alignment.distill(student, teacher, temperature=1.0))
    assert got == pytest.approx(want, rel=1e-9)  # 温度 1 不加缩放


def test_distill_temperature_flattens():
    teacher = np.array([[4.0, 0.0]])
    student = np.array([[0.0, 4.0]])
    cold = float(alignment.distill(student, teacher, temperature=0.5))
    hot = float(alignment.distill(student, teacher, temperature=4.0))
    assert cold / 0.5**2 > hot / 4.0**2  # 去掉 T² 之后，温度高了分布变平、KL 变小


def test_distill_scales_by_temperature_squared():
    teacher = np.array([[3.0, 1.0]])
    student = np.array([[1.0, 2.0]])
    t = 2.0
    p = np.exp(teacher / t) / np.exp(teacher / t).sum(axis=-1, keepdims=True)
    q = np.exp(student / t) / np.exp(student / t).sum(axis=-1, keepdims=True)
    want = t**2 * float((p * np.log(p / q)).sum())  # 缩放后的 KL 再乘 T²
    assert float(alignment.distill(student, teacher, temperature=t)) == pytest.approx(want, rel=1e-9)


# ---------- torch 一路 ----------

def test_torch_tensors_work_and_backprop():
    torch = pytest.importorskip("torch")
    chosen = torch.tensor([-0.5], requires_grad=True)
    rejected = torch.tensor([-1.5])
    ref = torch.tensor([-1.0])
    out = alignment.dpo(chosen, rejected, ref, ref, beta=0.5)
    assert out["loss"].detach().item() > 0  # 算得出标量
    out["loss"].backward()
    assert chosen.grad is not None and chosen.grad.item() < 0  # 抬高 chosen 能把损失压下去
