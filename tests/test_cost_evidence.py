"""真实证据驱动的成本与推荐回归：禁止缺省占位、配置外推和跨卡冒用。

运行：python -m pytest tests/test_cost.py tests/test_cost_evidence.py
fixture 只验证证据匹配规则；真机实测另保存在 docs/evidence，不能用这些 fixture 证明性能。
"""

import copy                               # 独立改一个配置维度，验证不能冒用旧证据
import io                                 # 公共价格响应的内存文件
import json                               # 测试真实 JSON 读取路径

import pytest                             # 错误与浮点近似断言

from features import cost, plan           # 产品实际使用的估价与训练计划入口


def recipe():
    """一份小文本配方，参数量不伪装成 7B。"""
    return {"task": "pretrain", "model": {"geometry": {"layers": 2, "d_model": 64, "heads": 2, "d_head": 32}},
            "train": {"precision": "bf16", "batch_size": 2, "grad_accum": 1, "max_steps": 100}}


def profile(data):
    """构造匹配逻辑用的显式测试记录，不冒充实测产物。"""
    config = cost.configuration(data)
    return {"version": 1, "kind": "cuda-training-calibration",
            "hardware": {"name": "fixture GPU", "total_gib": 24.0, "free_gib_before": 20.0},
            "runs": [{"status": "ok", "configuration": config, "seconds_per_step": 0.5,
                      "tokens_per_second": 512.0, "peak_reserved_gib": 4.0}]}


def test_missing_price_or_throughput_stays_unknown():
    table = plan.build(recipe())
    unknown = cost.estimate(table)
    assert unknown["hours"] is None and unknown["total_price"] is None
    assert cost.estimate(table, price_per_hour=3)["hours"] is None
    assert cost.estimate(table, tokens_per_second=100)["total_price"] is None
    assert "未知" in cost.render(unknown)


@pytest.mark.parametrize("section,key,value", [
    ("geometry", "layers", 3), ("geometry", "max_seq_len", 256), ("geometry", "d_model", 128),
    ("train", "precision", "nf4"), ("train", "grad_accum", 2), ("train", "batch_size", 4),
])
def test_mismatch_does_not_extrapolate_cost(section, key, value):
    data = recipe()
    evidence = profile(data)
    changed = copy.deepcopy(data)
    target = changed["model"]["geometry"] if section == "geometry" else changed["train"]
    target[key] = value
    if key == "d_model":
        target["heads"] = 4                    # 保持合法几何，只改变模型大小
    quote = cost.estimate(plan.build(changed), price_per_hour=1, profile=evidence, configuration=cost.configuration(changed))
    assert quote["hours"] is None and quote["total_price"] is None


def test_matched_configuration_and_usd_price_produce_measured_cost():
    data = recipe()
    evidence = profile(data)
    price = {"gpu": "fixture GPU", "currency": "USD", "unit": "GPU-hour", "price_per_hour": 3}
    quote = cost.estimate(plan.build(data), profile=evidence, configuration=cost.configuration(data), price=price)
    assert quote["hours"] == pytest.approx(100 * 0.5 / 3600)
    assert quote["total_price"] == pytest.approx(100 * 0.5 / 3600 * 3)
    assert "USD" in cost.render(quote)
    price["gpu"] = "different GPU"
    mismatch = cost.estimate(plan.build(data), profile=evidence, configuration=cost.configuration(data), price=price)
    assert mismatch["hours"] is not None and mismatch["total_price"] is None
    assert not mismatch["hardware_match"]


def test_recommend_only_measured_batches_with_capacity_margin():
    data = recipe()
    evidence = profile(data)
    candidate = copy.deepcopy(evidence["runs"][0])
    candidate["configuration"]["batch_size"] = 4
    candidate["peak_reserved_gib"] = 19           # 乘15%余量后超过实测空闲容量
    evidence["runs"].append(candidate)
    cards = cost.calibrated_recommend(evidence, cost.configuration(data))
    assert [row["batch_size"] for row in cards] == [2]
    assert cost.calibrated_recommend(evidence, cost.configuration(data), gpu_name="other") == []
    larger = copy.deepcopy(data)
    larger["model"]["geometry"]["layers"] = 12
    assert cost.calibrated_recommend(evidence, cost.configuration(larger)) == []


def test_plan_resolves_evidence_relative_to_recipe_directory(tmp_path):
    data = recipe()
    (tmp_path / "profile.json").write_text(json.dumps(profile(data)))
    data["cost"] = {"profile": "profile.json"}
    table = plan.build(data, base_dir=tmp_path)
    assert table.profile["hardware"]["name"] == "fixture GPU"
    assert "实测" in table.show()
    assert "未知" in table.show()                # 没有价钱仍然不能报总费用


def test_public_price_preserves_source_currency_unit_and_time(monkeypatch):
    payload = {"data": {"gpuTypes": [{"id": "fixture GPU", "securePrice": 0.73}]}}
    requests = []
    def open_response(request, timeout):
        requests.append(request)
        return io.BytesIO(json.dumps(payload).encode())
    monkeypatch.setattr(cost.urllib.request, "urlopen", open_response)
    result = cost.prices("fixture GPU")
    assert result["price_per_hour"] == 0.73
    assert result["currency"] == "USD" and result["unit"] == "GPU-hour"
    assert result["source"] == requests[0].full_url == cost.PRICE_URL
    assert result["fetched_at"]
    payload["data"]["gpuTypes"][0]["securePrice"] = 0
    with pytest.raises(ValueError, match="没有有效"):
        cost.prices("fixture GPU")


def test_cpu_cannot_claim_gpu_calibration(monkeypatch):
    import torch                              # 没有卡时必须明确拒绝校准
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="真实 CUDA"):
        cost.calibrate(recipe())
