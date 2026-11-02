"""成本与复用卡测试：估价保持线性，显存分档会给出不同且可读的配方。

调用示例：python -m pytest tests/test_cost.py -q
"""

from dataclasses import replace

import pytest

from features import cost, plan


def _plan(steps: int = 100):
    """造一张带明确步数和序列上限的训练账。"""
    return plan.build(                                                           # 走真实 build，顺带核对新字段来自配方
        {
            "task": "sft",                                                     # 普通监督微调足够覆盖成本入口
            "model": {"params_b": 7, "geometry": {"max_seq_len": 512}},       # 七十亿参数、每条最多 512 token
            "train": {"max_steps": steps, "batch_size": 2, "grad_accum": 4},  # 一个更新步处理八条序列
        }
    )


def test_price_and_hours_scale_linearly():
    table = _plan(steps=100)                                                     # 固定一张单卡训练账
    base = cost.estimate(table, price_per_hour=2.5, tokens_per_step=2_000, tokens_per_second=1_000)        # 基准单价和工作量
    longer = cost.estimate(replace(table, steps=200), price_per_hour=2.5, tokens_per_step=2_000, tokens_per_second=1_000)  # 步数翻倍
    dearer = cost.estimate(table, price_per_hour=5.0, tokens_per_step=2_000, tokens_per_second=1_000)       # 单价翻倍
    assert longer["hours"] == pytest.approx(base["hours"] * 2)                   # 总时长随总步数线性增长
    assert longer["total_price"] == pytest.approx(base["total_price"] * 2)       # 同一单价下总价也跟着翻倍
    assert dearer["hours"] == pytest.approx(base["hours"])                       # 换单价不应改变训练时间
    assert dearer["total_price"] == pytest.approx(base["total_price"] * 2)       # 价格只按小时单价线性变化


def test_recommendation_changes_with_vram_tier():
    small = cost.recommend(vram_gb=8, params_b=7)                                # 连七十亿 NF4 都吃紧的小卡
    medium = cost.recommend(vram_gb=48, params_b=7)                              # 能放量化、放不下全量状态
    large = cost.recommend(vram_gb=256, params_b=7)                              # 足够覆盖 BF16 全量训练量级
    assert len(small) == len(medium) == len(large) == 3                          # 每档都固定给三条可复用配方
    assert small != medium != large                                              # 显存档变化必须真正改变建议
    assert {one["precision"] for one in large} == {"bf16"}                       # 充裕档无需靠量化换空间


def test_small_vram_starts_conservatively():
    first = cost.recommend(vram_gb=8, params_b=7)[0]                             # 第一条是默认优先项
    assert first["strategy"] == "single"                                        # 先避开分布式额外复杂度
    assert first["precision"] == "nf4"                                          # 小显存优先四位权重
    assert first["batch_size"] == 1                                              # 微批从最保守的 1 起步
    assert first["why"]                                                         # 建议必须把原因带在旁边


def test_show_contains_cost_and_recommendation_sections():
    table = _plan(steps=120)                                                     # 有总步数时两段都应出现
    text = table.show()                                                          # 走用户实际看到的整张训练账
    assert table.steps == 120 and table.tokens == 512                            # 新字段确实由 recipe 填入
    assert "成本估算" in text and "复用卡推荐" in text                            # 成本与三条建议接在原账末尾
    assert "120 步" in text and "未知" in text                                  # 估算工作量和推荐字段都能读到


def test_show_without_steps_only_adds_recommendations():
    text = _plan(steps=0).show()                                                 # 没给 max_steps 就无法可靠估时
    assert "成本估算" not in text                                                # 不拿 epochs 硬猜总步数
    assert "复用卡推荐" in text                                                   # 无成本时仍给可复用配方
