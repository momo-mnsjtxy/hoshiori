"""对齐：偏好对数据与五种对齐损失。纯张量函数，不碰 trainer，手里也没有模型。

调用示例：
  from pathlib import Path
  from features import alignment
  rows = alignment.pairs([Path("pref.jsonl")])                                  # 一条偏好对拆成两个样本
  out = alignment.dpo(logp_c, logp_r, ref_c, ref_r, beta=0.1)                   # {"loss", "accuracy", "margin"}
  out = alignment.grpo(logps, rewards, clip=0.2, kl_coef=0.01, ref_logps=refs)  # {"loss", "advantages", "kl"}
  out = alignment.rloo(logps, rewards)                                          # {"loss", "advantages", "baseline"}
  out = alignment.ppo(logps, old_logps, advantages, clip=0.2)                   # {"loss", "ratio", "clipped"}
  loss = alignment.distill(student_logits, teacher_logits, temperature=2.0)     # 标量

trainer 按 recipe.task 挑函数，每个任务要的额外数据：
  dpo      走 dpo()，数据要有 chosen / rejected 两列，另拿冻结基座算 ref_* 两列
  grpo     走 grpo()，数据要有 rewards，另拿冻结基座算 ref_logps
  rloo     走 rloo()，数据要有 rewards，同组的几条要一起喂
  ppo      走 ppo()，数据要有 rewards，另拿上一版模型算 old_logps
  distill  走 distill()，数据要有 teacher_logits（教师模型的前向）
  sft / pretrain 不走这个文件，直接交叉熵。

形状：logp_* 每条样本一个数；logps / rewards 走 (组, 组内条数)，组内至少两条；
distill 的 logits 是 (…, 类数)，最后一维就是词表。
输入给 numpy 数组就按 numpy 算，给 torch 张量就走 torch（带梯度），两边都不用到对方。
KL 用 k3 估计 exp(ref-logp)-(ref-logp)-1，它恒非负，不用再估 log 的期望。
"""

import json
from pathlib import Path

from utils.fail import fail

EPS = 1e-6  # 组内标准差的底，全组奖励一样时也不至于除零


# --- 偏好数据 ---

def pairs(records) -> list:
    """读偏好对，每条映射成 (chosen 样本, rejected 样本)，各是一个 {"text": 正文}。

    records：jsonl 文件路径的列表，也可以直接给已经解析好的 dict 列表。
    每条要有 chosen 和 rejected 两个字符串；写了 prompt 的话会接在两边前面。
    返回列表，长度等于记录条数。缺字段当场说清是第几条缺哪个。
    """
    rows = []  # 摊平的记录
    for one in records:  # 路径和 dict 混着给也行
        rows.extend(_lines(one) if isinstance(one, (str, Path)) else [one])
    out = []  # 拆出来的样本对
    for at, row in enumerate(rows):  # 一条一条来
        for key in ("chosen", "rejected"):  # 两个字段都得有
            if not row.get(key):  # 缺了或者空串都算没有
                fail("input", f"第 {at + 1} 条缺 {key}", hint="偏好对要有 chosen 和 rejected 两个字段")
        head = row.get("prompt") or ""  # 有上下文就接在前面
        out.append(({"text": head + row["chosen"]}, {"text": head + row["rejected"]}))  # 一条变两个样本
    return out


def _lines(path: Path) -> list:
    """一个 jsonl 文件读成 dict 列表，空行跳过。"""
    return [json.loads(one) for one in Path(path).read_text(encoding="utf-8").splitlines() if one.strip()]


# --- 后端 ---

def _lib(*values):
    """看入参是 torch 张量还是 numpy，返回对应的库；给了张量才 import torch。"""
    if any(type(one).__module__.split(".")[0] == "torch" for one in values):  # 有一个是张量就走 torch
        import torch  # 只有这条路要 torch
        return torch
    import numpy as np  # 其余一律 numpy
    return np


def _as(lib, value):
    """转成这个后端的浮点数组，整数列表也接得住。"""
    if lib.__name__ == "torch":  # torch 走 as_tensor，和入参共用 dtype 规则
        return lib.as_tensor(value, dtype=lib.float32)
    return lib.asarray(value, dtype=lib.float64)  # numpy 默认双精度，数值用例好对


def _mean(lib, value, keep: bool = False):
    """按最后一维求均值。"""
    if lib.__name__ == "torch":
        return lib.mean(value, dim=-1, keepdim=keep)
    return lib.mean(value, axis=-1, keepdims=keep)


def _std(lib, value, keep: bool = False):
    """按最后一维求标准差，总体口径（除以 n），和 numpy 默认一致。"""
    if lib.__name__ == "torch":
        return lib.std(value, dim=-1, keepdim=keep, correction=0)
    return lib.std(value, axis=-1, keepdims=keep)


def _sum(lib, value):
    """按最后一维求和。"""
    if lib.__name__ == "torch":
        return lib.sum(value, dim=-1)
    return lib.sum(value, axis=-1)


def _minimum(lib, one, other):
    """逐位取小。"""
    return lib.minimum(one, other)


def _detach(lib, value):
    """梯度在这截断的地方用。"""
    return value.detach() if lib.__name__ == "torch" else value


def _sigmoid(lib, value):
    """logistic 函数。"""
    if lib.__name__ == "torch":
        return lib.sigmoid(value)
    return 1 / (1 + lib.exp(-value))


def _log_softmax(lib, value):
    """按最后一维取 log softmax，减最大值防溢出。"""
    if lib.__name__ == "torch":
        return lib.log_softmax(value, dim=-1)
    top = value - value.max(axis=-1, keepdims=True)  # 先挪到 0 以下
    return top - lib.log(lib.exp(top).sum(axis=-1, keepdims=True))  # 再减配分函数的对数


# --- 损失 ---

def dpo(logp_chosen, logp_rejected, logp_ref_chosen, logp_ref_rejected, beta: float = 0.1) -> dict:
    """DPO：让 chosen 的似然相对 rejected 拉开，同时别离参考模型太远。

    logp_chosen / logp_rejected：本次模型对两条样本的 log 概率，每条一个数。
    logp_ref_chosen / logp_ref_rejected：冻结参考模型对同两条的 log 概率，形状一样。
    beta：0.01 到 0.5。越大越贴参考模型、越难拉开；0.1 是常用的起步值。
    返回 {"loss", "accuracy", "margin"}：margin 是每条的优势差，两模型完全一样时是 0，损失正好 ln2。
    """
    lib = _lib(logp_chosen, logp_rejected, logp_ref_chosen, logp_ref_rejected)  # 按入参挑后端
    chosen, rejected, ref_chosen, ref_rejected = (_as(lib, one) for one in (logp_chosen, logp_rejected, logp_ref_chosen, logp_ref_rejected))
    margin = (chosen - rejected) - (ref_chosen - ref_rejected)  # 本次拉开的幅度减参考模型拉开的幅度
    loss = -_mean(lib, lib.log(_sigmoid(lib, beta * margin)))  # margin 是 0 时 sigmoid 是 0.5，损失 ln2
    accuracy = _mean(lib, (margin > 0) + 0.5 * (margin == 0))  # 平局按半条算
    return {"loss": loss, "accuracy": accuracy, "margin": margin}


def grpo(logps, rewards, *, clip: float = 0.2, kl_coef: float = 0.0, ref_logps) -> dict:
    """GRPO：同一组内把奖励归一成优势，比裁剪区间再用 KL 罚拉住参考模型。

    logps：本次模型的每条 log 概率，形状 (组, 组内条数)，组内至少两条。
    rewards：同形状的奖励，绝对值不重要，只有组内相对高低有用。
    clip：0.1 到 0.3，比率超出 1±clip 就按边界算，0.2 是常用值。
    kl_coef：0 到 0.1，KL 罚的权重，0 就是纯组内优化。
    ref_logps：同形状，冻结参考模型给的 log 概率。
    返回 {"loss", "advantages", "kl"}：优势按组归一，每组均值正好是 0，kl 是整批平均的那个数。
    """
    lib = _lib(logps, rewards, ref_logps)  # 按入参挑后端
    new, reward, ref = (_as(lib, one) for one in (logps, rewards, ref_logps))
    advantages = (reward - _mean(lib, reward, keep=True)) / (_std(lib, reward, keep=True) + EPS)  # 组内减均值除标准差
    ratio = lib.exp(new - _detach(lib, new))  # 只走一步，旧概率就是这条自己，比率此时是 1
    clipped = lib.clip(ratio, 1 - clip, 1 + clip)  # 卡在区间里
    surrogate = _mean(lib, _minimum(lib, ratio * advantages, clipped * advantages))  # 每组保守的那个乘优势
    gap = ref - new  # 参考模型比本次多出来的对数比
    kl = lib.mean(lib.exp(gap) - gap - 1)  # k3 估计，恒非负，整批取一个数
    return {"loss": -lib.mean(surrogate) + kl_coef * kl, "advantages": advantages, "kl": kl}


def rloo(logps, rewards) -> dict:
    """RLOO：拿组内去掉自己的平均奖励当基线，优势是奖励减基线。

    logps / rewards：形状 (组, 组内条数)，组内至少两条，只有两条时基线就是另一条。
    返回 {"loss", "advantages", "baseline"}：每条的基线里都不含自己。
    """
    lib = _lib(logps, rewards)  # 按入参挑后端
    new, reward = _as(lib, logps), _as(lib, rewards)
    total = _mean(lib, reward, keep=True) * reward.shape[-1]  # 组内总和
    baseline = (total - reward) / max(reward.shape[-1] - 1, 1)  # 除掉自己再平均
    advantages = reward - baseline  # 高于同伴是正的
    return {"loss": -_mean(lib, new * advantages), "advantages": advantages, "baseline": baseline}


def ppo(logps, old_logps, advantages, clip: float = 0.2) -> dict:
    """PPO：裁掉离旧策略太远的更新，只让区间内的比率出力。

    logps / old_logps：新、旧两版模型对同一批样本的 log 概率，形状一样。
    advantages：每条的优势，可以正可以负，形状一样。
    clip：0.1 到 0.3，比率卡在 1±clip。
    返回 {"loss", "ratio", "clipped"}：ratio 是本次与旧版的概率比，clipped 标出超出区间的条目。
    """
    lib = _lib(logps, old_logps, advantages)  # 按入参挑后端
    new, old, advantage = (_as(lib, one) for one in (logps, old_logps, advantages))
    ratio = lib.exp(new - old)  # 新旧概率比
    clipped = (ratio < 1 - clip) | (ratio > 1 + clip)  # 谁被卡了
    objective = _minimum(lib, ratio * advantage, lib.clip(ratio, 1 - clip, 1 + clip) * advantage)  # 保守的上界
    return {"loss": -lib.mean(objective), "ratio": ratio, "clipped": clipped}


def distill(student_logits, teacher_logits, temperature: float = 2.0):
    """蒸馏：教师分布到学生分布的 KL，按温度缩放再乘 T²。

    student_logits：学生每个位置的打分，形状 (…, 类数)。
    teacher_logits：教师同位置的打分，形状一样。
    temperature：0.5 到 10，越大分布越平、暗知识越显；趋 1 时就是原始 KL。
    返回标量损失。乘 T² 是因为缩放后的梯度会缩到 1/T²。
    """
    lib = _lib(student_logits, teacher_logits)  # 按入参挑后端
    student, teacher = _as(lib, student_logits) / temperature, _as(lib, teacher_logits) / temperature
    log_teacher, log_student = _log_softmax(lib, teacher), _log_softmax(lib, student)  # 两边都走 log softmax
    kl = _sum(lib, lib.exp(log_teacher) * (log_teacher - log_student))  # 每个位置一条 KL
    return temperature**2 * _mean(lib, kl)
