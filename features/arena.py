"""成对擂台：两个答题后端乱序回答同一批题，再按 Elo 与 Bradley-Terry 排名。

调用示例：
  from features import arena, suite
  one = suite.read(suite.folder() / "zh-lite.yaml")
  result = arena.run("model-a {prompt}", "model-b {prompt}", [one], limit=10)
  result["ranking"][0]  # Elo 排在前面的后端，同时带 BT 强度与胜平负

每题是一场对局：两份答案各自按题目规则得 0~1 分，分高者胜，相同就是平局。
后端调用顺序逐题打乱，避免总让某一方先答；结果始终换回 A 方视角，所以排名不受展示顺序影响。
Elo 用 K=32、初始 1500；Bradley-Terry 返回两方归一化后、和为 1 的强度。
"""

import math
import random

from features import lm_eval, scorer
from utils.fail import fail

ELO_K = 32.0                                                                 # 需求指定的单场更新幅度
ELO_INITIAL = 1500.0                                                         # 两个新后端从同一起点开打
BT_EPSILON = 1e-9                                                            # 全胜时把无限强弱比收成有限数，排序不变


def run(first_backend: str, second_backend: str, suites: list, limit: int = 0,
        seed: int = 0) -> dict:
    """让两个后端对打若干套题，返回逐题对局和排好序的强弱表。

    limit 按套件生效，0 表示每套全跑；seed 只控制谁先答，不改变题目顺序。
    """
    if not second_backend.strip():                                           # 第二个后端是擂台必填输入
        fail("input", "擂台要给另一个答题命令", hint="例: --arena 'model-b {prompt}'")
    if limit < 0:                                                            # 负数没有可解释的取题方式
        fail("input", "limit 不能是负数", hint="给 0 或正整数，0 表示不限")

    backends = (first_backend, second_backend)                               # A/B 身份全程固定，调用先后另算
    chance = random.Random(seed)                                             # 固定种子让同一批评测可复现
    matches = []                                                             # 一道题留下一场完整对局
    for one in suites:                                                       # 套件按用户给的顺序跑
        tasks = one.tasks[:limit] if limit else one.tasks                    # 每套最多取 limit 题
        for task in tasks:                                                   # 两个后端都答这一道题
            order = [0, 1]                                                   # 0 是 A，1 是 B
            chance.shuffle(order)                                            # 每题重新决定谁先答
            answers = ["", ""]                                             # 固定槽位保证结果仍按 A/B 对齐
            verdicts = [None, None]                                          # 两方各自的题目判定
            for side in order:                                               # 按本题抽到的先后调用后端
                answers[side] = lm_eval.answer(backends[side], task.prompt)   # 同一题面原样交给这一方
                verdicts[side] = scorer.grade(task, answers[side])           # 各自按同一规则判分
            first_score = verdicts[0].score                                  # A 方这题得分
            second_score = verdicts[1].score                                 # B 方这题得分
            outcome = 0.5 if first_score == second_score else float(first_score > second_score)  # 平局折半分
            matches.append({                                                 # 一场对局足够复算两种排名
                "suite": one.name,                                          # 来自哪套题
                "id": task.id,                                              # 哪一道题
                "order": ["A" if side == 0 else "B" for side in order],  # 实际谁先答
                "result": outcome,                                          # A 方视角：胜 1、平 0.5、负 0
                "players": [                                                # 两方原始答案和判定都留下
                    player("A", backends[0], answers[0], verdicts[0]),       # A 方
                    player("B", backends[1], answers[1], verdicts[1]),       # B 方
                ],
            })
    outcomes = [match["result"] for match in matches]                       # 两种排名共用同一份赛果
    return {                                                                 # 命令层可直接塞进 JSON 报告
        "elo": {"k": ELO_K, "initial": ELO_INITIAL},                     # 算法参数写明，报告可复算
        "matches": matches,                                                 # 每题的对局明细
        "ranking": rank(backends, outcomes),                                # 按 Elo 从强到弱排好
    }


def player(side: str, backend: str, answer: str, verdict) -> dict:
    """把一方在一道题上的答案与判定收成可写进 JSON 的记录。"""
    return {                                                                 # 字段与普通 judge 逐题报告保持相近
        "side": side,                                                       # A 或 B，命令相同时也分得清
        "backend": backend or "echo",                                     # 空命令就是回显后端
        "score": verdict.score,                                             # 这方在本题的分数
        "reason": verdict.reason,                                           # 判分理由
        "answer": answer.strip(),                                           # 实际答案
    }


def elo(outcomes: list) -> tuple:
    """按 A 方视角的赛果顺序更新两方 Elo，返回 ``(A, B)``。"""
    first = ELO_INITIAL                                                      # A 方从 1500 开始
    second = ELO_INITIAL                                                     # B 方从 1500 开始
    for outcome in outcomes:                                                 # 每题更新一次，平局 outcome=0.5
        expected = 1.0 / (1.0 + math.pow(10.0, (second - first) / 400.0))    # Elo 对 A 胜率的估计
        change = ELO_K * (outcome - expected)                                # 实际结果减预期，再乘 K
        first += change                                                      # A 加多少
        second -= change                                                     # B 就减多少，总分守恒
    return round(first, 4), round(second, 4)                                 # 报告保留足够复核的精度


def bradley_terry(outcomes: list) -> tuple:
    """拟合两方 Bradley-Terry 强度，返回和为 1 的 ``(A, B)``。"""
    if not outcomes:                                                         # 没比赛就谈不上拟合强弱
        fail("input", "擂台没有可排名的对局")
    first_wins = sum(outcomes)                                               # 胜记 1、平记 0.5，正是分数胜场
    probability = first_wins / len(outcomes)                                 # 两人 BT 的极大似然胜率有闭式解
    probability = min(1.0 - BT_EPSILON, max(BT_EPSILON, probability))       # 全胜/全负是无穷比，收成有限值
    return round(probability, 9), round(1.0 - probability, 9)                # 归一化强度也就是两人对局胜率


def rank(backends: tuple, outcomes: list) -> list:
    """把两方赛果算成 Elo、BT 与胜平负，按 Elo 再按 BT 从强到弱返回。"""
    if not outcomes:                                                         # 排名公开接口在边界直接说明空输入
        fail("input", "擂台没有可排名的对局")
    elo_scores = elo(outcomes)                                               # 顺序模型：越近的赛果影响仍由当前分差决定
    bt_scores = bradley_terry(outcomes)                                      # 汇总模型：只看分数胜场占比
    wins = (sum(one > 0.5 for one in outcomes), sum(one < 0.5 for one in outcomes))  # 两方胜场
    ties = sum(one == 0.5 for one in outcomes)                               # 平局两边相同
    losses = (wins[1], wins[0])                                              # A 的负场就是 B 的胜场，反之亦然
    rows = [                                                                 # 先按固定 A/B 身份造两行
        {
            "side": "A" if side == 0 else "B",                           # 身份不随名次变化
            "backend": backends[side] or "echo",                          # 后端命令就是参赛者名
            "elo": elo_scores[side],                                        # Elo 分
            "bt": bt_scores[side],                                          # BT 归一化强度
            "wins": wins[side],                                             # 胜场
            "ties": ties,                                                   # 平场
            "losses": losses[side],                                         # 负场
        }
        for side in (0, 1)                                                   # 两方各一行
    ]
    rows.sort(key=lambda row: (row["elo"], row["bt"]), reverse=True)       # 先看 Elo，同分再看 BT
    for position, row in enumerate(rows, 1):                                 # 排完再写名次，避免两份顺序来源
        row["rank"] = position                                              # 1 就是擂主
    return rows                                                              # 已从强到弱排好
