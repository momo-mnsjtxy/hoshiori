"""训练的测试：loss 真在掉、存档结构对、续训接得上、LoRA 只动 A/B、分块交叉熵不跑偏。

约定：
  模型缩到 2 层 32 维 64 词表，数据是 10 条假样本，步数只给二十来步，CPU 上也只跑一两秒。
  词表 64 是故意比字节值小，顺带验字节取模那条路；序列压到 24，别让小矩阵撑大时间。
"""

import json
import shutil

import pytest
import torch
from torch.nn import functional as F

import main
from features import lora, model, trainer


@pytest.fixture
def geo():
    """一套能在 CPU 上秒过的小几何。"""
    return model.Geometry(d_model=32, layers=2, heads=2, d_head=16, vocab=64, max_seq_len=24)  # 2 × 16 正好 32


@pytest.fixture
def files(tmp_path):
    """10 条假样本，落成 bolt.jsonl。"""
    texts = [f"星织把散落的星点织成整匹布，第{i}条。" for i in range(10)]  # 十条各不相同的短句
    path = tmp_path / "bolt.jsonl"  # 跟 weave 的产物同名
    path.write_text("\n".join(json.dumps({"text": one}, ensure_ascii=False) for one in texts), encoding="utf-8")  # 一行一条
    return [path]


def short(geo, files, out, **over):
    """拼一个步数很少的参数包。"""
    base = dict(out=out, files=files, batch_size=2, steps=20, save_every=10, log_every=5, seed=7, lr=0.01)  # 学得动又不慢
    base.update(over)
    return trainer.Choice(**base)


# ---------- 训练本身 ----------

def test_loss_falls(geo, files, tmp_path):
    net = model.Model(geo)  # 随机起的模型
    job = trainer.Trainer(net, geo, short(geo, files, tmp_path / "run"))  # 二十步
    job.run()
    first = sum(job.history[:5]) / 5  # 开头五步的平均损失
    last = sum(job.history[-5:]) / 5  # 收尾五步的平均损失
    assert last < first  # 这二十步里确实学到了东西


def test_checkpoint_layout_and_resume(geo, files, tmp_path):
    out = tmp_path / "run"
    job = trainer.Trainer(model.Model(geo), geo, short(geo, files, out, steps=3, save_every=2))  # 三步，每两步存一档
    job.run()
    assert (out / "latest").read_text(encoding="utf-8") == "step-000003"  # 指针指向最后一档
    for step in (2, 3):  # 存过的两档
        folder = out / f"step-{step:06d}"
        assert (folder / "model.safetensors").is_file()  # 权重在
        assert (folder / "optimizer.pt").is_file()  # 优化器态在
        state = json.loads((folder / "state.json").read_text(encoding="utf-8"))  # 状态在
        assert state["step"] == step and state["cursor"] > 0 and state["rng"]  # 步数、游标、随机数都落了
    assert (out / "run.json").is_file() and (out / "config.json").is_file()  # 这一趟的设定和结构也在

    again = trainer.Trainer(model.Model(geo), geo, short(geo, files, out, steps=5, save_every=2, resume=out / "latest"))  # 再跑五步
    assert again.run() == 8  # 从第三步起再走五步，落在第八步
    assert (out / "step-000008" / "state.json").is_file()  # 新的一档落盘
    assert (out / "step-000006" / "state.json").is_file()  # 每两步那一档也在


def test_resume_keeps_cursor(geo, files, tmp_path):
    out = tmp_path / "run"
    trainer.Trainer(model.Model(geo), geo, short(geo, files, out, steps=4, save_every=4)).run()  # 先跑四步
    state = json.loads((out / "step-000004" / "state.json").read_text(encoding="utf-8"))  # 读回的游标
    assert state["cursor"] == 8  # 两步一批、四步共取八条
    job = trainer.Trainer(model.Model(geo), geo, short(geo, files, out, steps=5, save_every=5, resume=out / "step-000004"))  # 从断点接
    job.run()
    assert job.history and json.loads((out / "step-000005" / "state.json").read_text(encoding="utf-8"))["cursor"] == 10  # 接着往后取


# ---------- LoRA ----------

def test_lora_only_trains_ab(geo, files, tmp_path):
    net = model.Model(geo)  # 基座
    job = trainer.Trainer(net, geo, short(geo, files, tmp_path / "run", steps=3, save_every=3, lora={"r": 4, "alpha": 8, "dropout": 0.0, "target": ["w_q", "w_v"]}))  # 只挂两路注意力
    frozen = {name: one.detach().clone() for name, one in net.named_parameters() if not (name.endswith(".A") or name.endswith(".B"))}  # 冻住的那批，先照一份
    assert frozen and all(not one.requires_grad for name, one in net.named_parameters() if name in frozen)  # 除 A/B 外全不可训
    job.run()
    for name, one in net.named_parameters():  # 逐个对照
        if name in frozen:  # 基座参数
            assert torch.equal(one.detach(), frozen[name])  # 一步都没动
        else:  # 挂件的 A/B
            assert one.requires_grad  # 可训
    assert any(not torch.equal(one.detach(), torch.zeros_like(one)) for name, one in net.named_parameters() if name.endswith(".B"))  # B 从全零被推动了
    assert (tmp_path / "run" / "step-000003" / "adapter.safetensors").is_file()  # 挂件只存 A/B
    assert not (tmp_path / "run" / "step-000003" / "model.safetensors").exists()  # 不存全量


def test_lora_merge_and_reload(geo, tmp_path):
    net = model.Model(geo)
    lora.attach(net, ["w_q"], r=4, alpha=8, dropout=0.0)  # 挂一层
    base = {name.replace(".base.weight", ".weight"): one.detach().clone() for name, one in net.named_parameters() if one.requires_grad is False}  # 基座照一份，名字还原成不带 base 的
    with torch.no_grad():  # 手推一个非零增量
        next(iter(lora.layers(net).values())).B.add_(0.1)
    merged = lora.merge(net)  # 合并回基座
    assert set(merged) == set(base)  # 产物里没有 lora 键
    assert not torch.equal(merged["model.layers.0.self_attn.q_proj.weight"], base["model.layers.0.self_attn.q_proj.weight"])  # 那一层被加过增量
    net.save(tmp_path / "w")  # 存全量
    model.Model(geo).load(tmp_path / "w")  # 名字对得上就能读回


# ---------- 分块交叉熵 ----------

def test_chunked_matches_torch():
    torch.manual_seed(3)  # 固定卷子
    logits = torch.randn(2, 8, 64)  # 批 × 序列 × 词表
    labels = torch.randint(0, 64, (2, 8))  # 答案
    labels[0, :2] = -100  # 前两个位置算补位
    labels[1, 7] = -100  # 末尾也算补位
    mine = trainer.cross_entropy(logits, labels)  # 分块版
    plain = F.cross_entropy(logits.reshape(-1, 64), labels.reshape(-1), ignore_index=-100)  # 一次展开的官方版
    assert abs(float(mine) - float(plain)) < 1e-4  # 分块不能把数值带偏


# ---------- 命令接线 ----------

def test_spin_dry_run_unchanged(tmp_path):
    recipe = tmp_path / "s.warp.yaml"  # 一份最小配方
    recipe.write_text("version: 1\ntask: pretrain\nmodel:\n  geometry:\n    d_model: 32\n    layers: 1\n    heads: 2\n    d_head: 16\n    vocab: 64\ndata:\n  files: [bolt.jsonl]\n", encoding="utf-8")  # 只给几何和数据
    (tmp_path / "bolt.jsonl").write_text(json.dumps({"text": "一小段话"}, ensure_ascii=False), encoding="utf-8")  # 数据在
    assert main.run(["spin", str(recipe), "--dry-run"]) == 0  # 只出账
    assert not (tmp_path / "out").exists()  # 不落产物


def test_spin_runs_two_steps(tmp_path):
    recipe = tmp_path / "s.warp.yaml"  # 同上
    recipe.write_text("version: 1\ntask: pretrain\nmodel:\n  geometry:\n    d_model: 32\n    layers: 1\n    heads: 2\n    d_head: 16\n    vocab: 64\ndata:\n  files: [bolt.jsonl]\ntrain:\n  batch_size: 1\n  lr: 0.01\n", encoding="utf-8")
    (tmp_path / "bolt.jsonl").write_text(json.dumps({"text": "一小段话够训练"}, ensure_ascii=False), encoding="utf-8")  # 一条样本
    out = tmp_path / "out"
    assert main.run(["spin", str(recipe), "--steps", "2", "--save-every", "1", "--out", str(out)]) == 0  # 真跑两步
    assert (out / "step-000002" / "state.json").is_file()  # 收尾的档在
    assert (out / "config.json").is_file()  # 结构描述在


def test_ctrl_c_saves_before_exit(geo, files, tmp_path, monkeypatch):
    out = tmp_path / "run"
    job = trainer.Trainer(model.Model(geo), geo, short(geo, files, out, steps=10, save_every=0))  # 不定点存
    real = job._update  # 原本的一步
    calls = {"n": 0}  # 跑了几步

    def stop():  # 第三步手工掐断
        calls["n"] += 1
        if calls["n"] == 3:
            raise KeyboardInterrupt
        return real()

    monkeypatch.setattr(job, "_update", stop)  # 换掉一步
    assert job.run() == 3  # 停在第三步
    assert (out / "step-000003" / "state.json").is_file()  # 断点存下了


def test_spin_refuses_obvious_overflow(tmp_path):
    recipe = tmp_path / "s.warp.yaml"  # 六十亿参数还上 bf16，一眼放不下
    recipe.write_text("version: 1\ntask: pretrain\nmodel:\n  params_b: 60\ndata:\n  files: [bolt.jsonl]\ntrain:\n  precision: bf16\n", encoding="utf-8")
    (tmp_path / "bolt.jsonl").write_text(json.dumps({"text": "一小段话"}, ensure_ascii=False), encoding="utf-8")  # 数据在
    if shutil.which("nvidia-smi") is None:  # 没卡先卡在要卡那条，验不到显存
        pytest.skip("没有卡，轮不到显存那条")
    assert main.run(["spin", str(recipe), "--steps", "2", "--out", str(tmp_path / "out")]) == 2  # 按填错退回
