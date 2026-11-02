"""训练深度验收：异构教师、健康验证、难度课程、低秩变体和跨批图文检索。

运行：OMP_NUM_THREADS=1 python -m pytest tests/test_training_depth.py -q
测试使用真实 CPU 前向/反向与 safetensors 存档，不把注入分数当作检索实测。
"""

import json  # 构建训练集并核对实际落盘状态
from dataclasses import replace  # 只修改要验证的模型几何

import numpy as np  # 对照导出合并结果
import pytest  # 独立临时目录和参数化用例
import torch  # 实际梯度更新
from torch.nn import functional as F  # 合并前后线性计算对照

from features import lora, merge, model, recipe, trainer  # 使用产品实际入口
from utils.fail import Failure  # 错误路径也要有明确原因


def geometry(**over):
    """一套 CPU 可快速计算的小几何。"""
    return replace(model.Geometry(d_model=16, layers=1, heads=2, d_head=8, vocab=32, max_seq_len=12), **over)


def data(tmp_path, rows=None):
    """写真实 JSONL，训练器与命令调用都能直接读取。"""
    path = tmp_path / "data.jsonl"  # 样本只属于当前测试
    rows = rows or [{"text": "train a tiny example"}, {"text": "another sentence here"}]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return [path]  # 与 Choice.files 保持一致


def test_heterogeneous_teacher_runs_and_stays_frozen(tmp_path):
    torch.manual_seed(11)  # 固定初始化，失败可复现
    student_geometry = geometry()
    teacher_geometry = geometry(d_model=32, layers=3, heads=4)
    teacher_path = tmp_path / "teacher"
    teacher = model.Model(teacher_geometry)
    teacher.save(teacher_path)
    teacher.write_config(teacher_path, "fp32")  # 几何由教师自己写出
    student = model.Model(student_geometry)
    initial = student.model.embed_tokens.weight.detach().clone()
    choice = trainer.Choice(out=tmp_path / "out", files=data(tmp_path), steps=2, teacher=teacher_path, task="distill")
    job = trainer.Trainer(student, student_geometry, choice)
    assert job.teacher.geo.layers == 3 and job.teacher.geo.d_model == 32  # 没有套用学生结构
    frozen = {name: tensor.clone() for name, tensor in job.teacher.state_dict().items()}
    job.step = 1
    loss = job._update()  # 真跑异构教师蒸馏反向
    assert np.isfinite(loss) and not torch.equal(student.model.embed_tokens.weight, initial)
    assert all(torch.equal(value, job.teacher.state_dict()[name]) for name, value in frozen.items())
    job.close()  # 释放冻结教师


def test_teacher_vocab_mismatch_has_clear_error(tmp_path):
    path = tmp_path / "teacher"
    teacher = model.Model(geometry(vocab=64))
    teacher.save(path)
    teacher.write_config(path, "fp32")
    choice = trainer.Choice(out=tmp_path / "out", files=data(tmp_path), task="distill", teacher=path)
    with pytest.raises(Failure, match="相同词表"):
        trainer.Trainer(model.Model(geometry()), geometry(), choice)  # 不允许错误 token 语义直接 KL


@pytest.mark.parametrize("variant", ["lora", "dora", "loha"])
def test_low_rank_variants_train_roundtrip_and_merge(tmp_path, variant):
    torch.manual_seed(31)
    original = model.Model(geometry())
    base = {name: value.detach().numpy().copy() for name, value in original.state_dict().items()}
    expected = original(torch.tensor([[1, 2, 3]])).detach()
    lora.attach(original, ["w_q"], r=2, alpha=4, dropout=0, variant=variant)
    torch.testing.assert_close(original(torch.tensor([[1, 2, 3]])), expected)  # 零增量起步不扰动基座
    layer = next(iter(lora.layers(original).values()))
    optimizer = torch.optim.Adam(lora.trainable(original), lr=0.05)
    x = torch.randn(2, 4, 16)
    target = torch.randn(2, 4, 16)
    for _ in range(3):
        optimizer.zero_grad()
        F.mse_loss(layer(x), target).backward()
        optimizer.step()  # 实际学习几步，让所有变体偏离初始状态
    assert layer.B.abs().sum() > 0  # LoHa 的两组全零死梯度会在这里暴露
    layer.eval()
    torch.testing.assert_close(layer(x), F.linear(x, layer.merged(), layer.base.bias), atol=1e-5, rtol=1e-5)
    folder = tmp_path / variant
    lora.save(original, folder)
    again = model.Model(geometry())
    again.load_state_dict({name: torch.from_numpy(value) for name, value in base.items()})
    lora.attach(again, ["w_q"], r=2, alpha=4, dropout=0, variant=variant)
    lora.load(again, folder)
    assert torch.equal(next(iter(lora.layers(again).values()))(x), layer(x))  # 所有额外矩阵/幅值都必须往返
    exported = merge.merge(base, [merge.Adapter(folder)])
    for name, value in lora.merge(original).items():
        np.testing.assert_allclose(exported[name], value.detach().numpy(), atol=2e-6, rtol=2e-6)  # forge 使用的 numpy 合并公式一致


def test_health_config_eval_drop_and_resume_state(tmp_path):
    geo = geometry()
    limits = {"nan_steps": 1, "norm_limit": 2.0, "norm_steps": 2, "eval_drop": 0.2, "eval_patience": 2}
    choice = trainer.Choice(out=tmp_path / "out", files=data(tmp_path), health_limits=limits)
    job = trainer.Trainer(model.Model(geo), geo, choice)
    assert not job._health(1, 1, eval_score=0.9)  # 首次建立最佳分
    assert not job._health(1, 1, eval_score=0.5)  # 第一次退化还不停止
    assert job._health(1, 1, eval_score=0.5) and "eval" in job.reason
    folder = job.save()  # 退化次数和最佳分要随断点保存
    resumed = trainer.Trainer(model.Model(geo), geo, choice)
    resumed.restore(folder)
    assert resumed.best_eval == 0.9 and resumed.bad_eval == 2
    resumed.close()
    job.close()


def test_real_eval_does_not_consume_training_data(tmp_path):
    geo = geometry()
    files = data(tmp_path)
    job = trainer.Trainer(model.Model(geo), geo, trainer.Choice(out=tmp_path / "out", files=files, eval_files=files, eval_every=1, steps=2))
    before = job.cursor
    score = job.evaluate()  # 真实模型逐 token 计算，不注入外部分数
    assert 0 <= score <= 1 and job.cursor == before and job.model.training
    assert job.run() == 2 and job.best_eval is not None


def test_difficulty_curriculum_is_not_length_order(tmp_path):
    rows = [{"text": "a" * 10, "difficulty": 0.0}, {"text": "bb", "difficulty": 1.0}, {"text": "ccccc", "difficulty": 0.5}]
    geo = geometry()
    job = trainer.Trainer(model.Model(geo), geo, trainer.Choice(out=tmp_path / "out", files=data(tmp_path, rows), curriculum=True, curriculum_by="difficulty", steps=9))
    job.step = 1
    assert len(job._next()) == 10  # 人工最容易的长样本先于最难的短样本
    job.step = 9
    assert len(job._next()) == 2  # 最后阶段可选到最难的短样本
    job.close()


@pytest.mark.parametrize("limits", [{"nan_steps": 0}, {"eval_patience": -1}, {"norm_limit": float("nan")}, {"eval_drop": 1.1}])
def test_health_limits_reject_invalid_values(limits):
    with pytest.raises(Failure):
        recipe.check({"train": {"health_limits": limits}})  # 不合法阈值在配方入口拒绝


def test_cross_batch_negatives_have_gradient_and_bounded_queue():
    contrast = model.ImageContrast(3, capacity=2, temperature=0.2)
    first = torch.tensor([[1.0, 0.0, 0.0]], requires_grad=True)
    contrast(first, first, torch.tensor([1])).backward()  # 第一批只有正例，先填入历史
    second = torch.tensor([[0.6, 0.8, 0.0]], requires_grad=True)
    loss = contrast(second, second, torch.tensor([2]))  # 当前只有一对，也能从上批得到负例
    loss.backward()
    assert float(loss.detach()) > 0 and second.grad.abs().sum() > 0
    assert contrast.keys.tolist() == [1, 2] and not contrast.images.requires_grad
    contrast(second.detach(), second.detach(), torch.tensor([3]))
    assert contrast.keys.tolist() == [2, 3] and len(contrast.images) == 2  # 最旧负例确实淘汰
    contrast.eval()
    before = contrast.keys.clone()
    contrast(second.detach(), second.detach(), torch.tensor([4]))
    assert torch.equal(before, contrast.keys)  # 验证不能污染训练历史


def test_conv_tower_gradient_retrieval_and_checkpoint(tmp_path):
    torch.manual_seed(9)
    geo = geometry(image_side=4, patch=2)
    net = model.Model(geo, align_weight=1.0, align_queue=4)
    ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    pixels = torch.randn(2, 4, 2, 2, 3)
    _, loss = net(ids, patches=pixels)
    loss.backward()  # 真实卷积参数必须接到对比学习梯度
    assert net.vision.conv_1.weight.grad.abs().sum() > 0
    assert net.vision.conv_2.weight.grad.abs().sum() > 0
    report = net.retrieval(ids, pixels)
    assert report["pairs"] == 2 and all(0 <= value <= 1 for key, value in report.items() if key != "pairs")
    state = net.save(tmp_path)
    assert "vision.conv_1.weight" in state and not any(name.startswith("contrast.") for name in state)
    again = model.Model(geo, align_weight=1.0)
    again.load(tmp_path)
    assert again.retrieval(ids, pixels) == report  # 推理产物完整保留视觉检索能力


@pytest.mark.parametrize("variant", ["lora", "dora", "loha"])
def test_bf16_adapter_export(variant, tmp_path):
    net = model.Model(geometry()).to(torch.bfloat16)  # 用 CPU 就能复现 BF16 safetensors 边界
    base = {name: value.float().numpy() for name, value in net.state_dict().items()}
    lora.attach(net, ["w_q"], 2, 4, 0, variant=variant)
    layer = next(iter(lora.layers(net).values()))
    with torch.no_grad():
        layer.B.fill_(0.05)  # 非零更新才会覆盖 BF16 解码的实际值
    lora.save(net, tmp_path)
    exported = merge.merge(base, [merge.Adapter(tmp_path)])
    for name, value in lora.merge(net).items():
        np.testing.assert_allclose(exported[name], value.numpy(), atol=2e-6, rtol=2e-6)


def test_integer_stored_base_keeps_floating_adapter():
    net = torch.nn.Sequential(torch.nn.Linear(4, 4))  # 模拟量化层的整数存储契约
    net[0].weight = torch.nn.Parameter(torch.ones(4, 4, dtype=torch.int8), requires_grad=False)
    layer = lora.LowRank(net[0], 2, 4, 0)  # 不能调用 Module.to(int8)
    assert layer.A.dtype == torch.float32 and layer.base.weight.dtype == torch.int8


def test_old_visual_weights_load_without_fake_random_tower(tmp_path):
    geo = geometry(image_side=4, patch=2, vision="linear")
    old = model.Model(geo).eval()
    old.save(tmp_path)
    new = model.Model(replace(geo, vision="conv")).eval()
    new.load(tmp_path)  # 明确识别旧结构并保留原先实际计算路径
    ids, pixels = torch.tensor([[1, 2]]), torch.randn(1, 4, 2, 2, 3)
    assert new.vision is None and new.geo.vision == "linear"
    torch.testing.assert_close(new(ids, pixels), old(ids, pixels))


def test_invalid_initial_eval_recovers_finite_baseline(tmp_path):
    geo = geometry()
    job = trainer.Trainer(model.Model(geo), geo, trainer.Choice(out=tmp_path / "out", files=data(tmp_path), health_limits={"eval_patience": 2}))
    assert not job._health(1, 1, eval_score=float("nan"))  # 无效验证不能成为最好分数
    assert job.best_eval is None and job.bad_eval == 1
    assert not job._health(1, 1, eval_score=0.7)  # 随后有限分数应重新建立有效基线
    assert job.best_eval == 0.7 and job.bad_eval == 0
    assert not job._health(1, 1, eval_score=float("nan"))
    assert job._health(1, 1, eval_score=float("nan")) and "eval" in job.reason
    job.close()
