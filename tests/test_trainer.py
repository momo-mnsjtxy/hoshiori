"""训练的测试：loss 真在掉、存档结构对、续训接得上、LoRA 只动 A/B、分块交叉熵不跑偏。

约定：
  模型缩到 2 层 32 维 64 词表，数据是 10 条假样本，步数只给二十来步，CPU 上也只跑一两秒。
  词表 64 是故意比字节值小，顺带验字节取模那条路；序列压到 24，别让小矩阵撑大时间。
"""

import importlib.util
import json
import math
import shutil
import socket
import sys
import time
import types

import pytest
import torch
from torch.nn import functional as F

import main
from commands import spin
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


# ---------- 量化 ----------

def test_quantize_falls_back_on_cpu(geo, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)  # 当成没有卡
    net = model.Model(geo)
    assert net.quantize("nf4") is False  # 压不动就老实返回没压
    assert net.quantized is False and net.model.layers[0].mlp.up_proj.weight.dtype is torch.float32  # 还是 fp32 的普通线层
    assert net(torch.randint(0, geo.vocab, (1, 6))).shape == (1, 6, geo.vocab)  # 回退后照样能跑
    assert net.quantize("fp32") is False  # 本来就不是要压的精度


def test_quantize_swaps_layers_and_freezes(geo, monkeypatch):
    stub = _bnb_stub(monkeypatch)  # 拿一个假 bitsandbytes 顶上，机器上没装包也能走这条路径
    net = model.Model(geo)
    assert net.quantize("nf4", device=torch.device("cpu")) is True  # 真压了
    assert net.quantized
    assert isinstance(net.model.layers[0].self_attn.q_proj, stub.nn.Linear4bit)  # 权重线层被换成了 4bit 层
    assert not isinstance(net.model.norm, stub.nn.Linear4bit)  # 归一不动
    assert all(not one.requires_grad for one in net.parameters())  # 量化层自己没梯度，其余也冻上
    assert net(torch.randint(0, geo.vocab, (1, 6))).shape == (1, 6, geo.vocab)  # 假层就是普通线层，前向形状照旧


def test_quantized_only_trains_lora(geo, files, tmp_path, monkeypatch):
    _bnb_stub(monkeypatch)
    net = model.Model(geo)
    net.quantize("nf4", device=torch.device("cpu"))
    job = trainer.Trainer(net, geo, short(geo, files, tmp_path / "run", steps=2, save_every=0, precision="nf4", lora={"r": 4, "alpha": 8, "dropout": 0.0, "target": ["w_q"]}))  # 压过就只挂件可训
    assert job.params and all(name.endswith((".A", ".B")) for name, _ in net.named_parameters() if _.requires_grad)  # 能训的只剩 A/B
    assert job.run() == 2  # 真跑两步不炸
    assert (tmp_path / "run" / "step-000002" / "adapter.safetensors").is_file()  # 只落挂件


# ---------- 健康检查 ----------

def health_recipe(tmp_path, health="stop", steps=25):
    """一份会走到健康检查的配方；几何压到能秒过的尺寸。"""
    recipe = tmp_path / "s.warp.yaml"
    recipe.write_text(
        "version: 1\ntask: pretrain\nmodel:\n  geometry:\n    d_model: 32\n    layers: 1\n    heads: 2\n    d_head: 16\n    vocab: 64\n"
        f"data:\n  files: [bolt.jsonl]\ntrain:\n  batch_size: 1\n  lr: 0.01\n  max_steps: {steps}\n  health: {health}\n",
        encoding="utf-8",
    )
    (tmp_path / "bolt.jsonl").write_text(json.dumps({"text": "一小段话够训练"}, ensure_ascii=False), encoding="utf-8")  # 一条样本
    return recipe


def test_health_stops_on_nan(tmp_path, monkeypatch):
    recipe = health_recipe(tmp_path)  # 默认 stop
    out = tmp_path / "out"

    def fake(self):  # 三步 NaN
        self.last_tokens, self.last_norm = 1000, 0.5
        return float("nan")

    monkeypatch.setattr(trainer.Trainer, "_update", fake)
    assert main.run(["spin", str(recipe), "--out", str(out)]) == 1  # 坏了就退 1
    meta = json.loads((out / "run.json").read_text(encoding="utf-8"))  # 原因落在 run.json 里
    assert "NaN" in meta["health"]["reason"] and meta["health"]["step"] == 3  # 第三步踩线
    assert (out / "step-000003" / "state.json").is_file()  # 停之前存了一档


def test_health_stops_on_big_norm(tmp_path, monkeypatch):
    recipe = health_recipe(tmp_path)
    out = tmp_path / "out"

    def fake(self):  # 梯度范数一直炸
        self.last_tokens, self.last_norm = 1000, 2e3
        return 1.0

    monkeypatch.setattr(trainer.Trainer, "_update", fake)
    assert main.run(["spin", str(recipe), "--out", str(out)]) == 1
    meta = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert "梯度范数" in meta["health"]["reason"] and meta["health"]["step"] == 5  # 第五步才成立
    assert (out / "step-000005" / "state.json").is_file()


def test_health_stops_on_slow_throughput(tmp_path, monkeypatch):
    recipe = health_recipe(tmp_path)
    out = tmp_path / "out"

    def fake(self):  # 每步都慢一点点，第二十二步突然更慢
        self.last_tokens, self.last_norm = 1000, 0.5
        time.sleep(0.06 if self.step == 22 else 0.006)
        return 1.0

    monkeypatch.setattr(trainer.Trainer, "_update", fake)
    assert main.run(["spin", str(recipe), "--out", str(out)]) == 1
    meta = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert "吞吐" in meta["health"]["reason"] and meta["health"]["step"] == 22  # 攒够 20 步基线后第一步就踩线
    assert (out / "step-000022" / "state.json").is_file()


def test_health_warn_keeps_going(tmp_path, monkeypatch):
    recipe = health_recipe(tmp_path, health="warn", steps=5)
    out = tmp_path / "out"

    def fake(self):  # 一直 NaN 也不停
        self.last_tokens, self.last_norm = 1000, 0.5
        return float("nan")

    monkeypatch.setattr(trainer.Trainer, "_update", fake)
    assert main.run(["spin", str(recipe), "--out", str(out)]) == 0  # 只是喊一声，跑完为止
    assert (out / "step-000005" / "state.json").is_file()  # 照样跑满
    assert "health" not in json.loads((out / "run.json").read_text(encoding="utf-8"))  # 没有停机原因


# ---------- 分布式 ----------

def test_single_process_dist_branch_runs(geo, files, tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_RANK", "0")  # 冒充 torchrun 起的进程
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", str(_port()))  # 随便一个空口
    out = tmp_path / "run"
    job = trainer.Trainer(model.Model(geo), geo, short(geo, files, out, steps=2, save_every=0, strategy="ddp"))  # 单进程走 ddp 分支
    assert (job.rank, job.world, job.lead) == (0, 1, True)  # 单进程就是 0 号，存盘打日志都归它
    job.run()
    assert not torch.distributed.is_initialized()  # 收尾把组拆了
    assert (out / "step-000002" / "state.json").is_file()  # 产物照落


def test_grad_checkpoint_switches_forward(geo, files, tmp_path):
    net = model.Model(geo)
    job = trainer.Trainer(net, geo, short(geo, files, tmp_path / "run", steps=2, save_every=0, grad_checkpoint=True))
    assert net.grad_checkpoint is True  # 配方开关传到了模型上
    job.run()
    assert job.history and all(math.isfinite(one) for one in job.history)  # 重算那一版照样出有限 loss


def test_spin_launch_torchrun(tmp_path):
    if shutil.which("torchrun") is None:
        pytest.skip("没有 torchrun")
    recipe = tmp_path / "s.warp.yaml"  # 单卡也给 torchrun 起，验的是命令拼得对
    recipe.write_text(
        "version: 1\ntask: pretrain\nmodel:\n  geometry:\n    d_model: 32\n    layers: 1\n    heads: 2\n    d_head: 16\n    vocab: 64\n"
        "data:\n  files: [bolt.jsonl]\ntrain:\n  batch_size: 1\n  lr: 0.01\n  strategy: ddp\n  devices: [0]\n",
        encoding="utf-8",
    )
    (tmp_path / "bolt.jsonl").write_text(json.dumps({"text": "一小段话够训练"}, ensure_ascii=False), encoding="utf-8")
    out = tmp_path / "out"
    assert main.run(["spin", str(recipe), "--steps", "2", "--save-every", "0", "--out", str(out), "--launch", "torchrun"]) == 0  # 子进程跑通
    assert (out / "step-000002" / "state.json").is_file()  # 0 号进程存下了档


def _port() -> int:
    """挑一个空口给 rendezvous。"""
    with socket.socket() as one:
        one.bind(("127.0.0.1", 0))
        return one.getsockname()[1]


def _bnb_stub(monkeypatch):
    """一个假的 bitsandbytes：够让量化路径整条跑完，不做真压缩。"""
    stub = types.ModuleType("bitsandbytes")  # 假包

    class Params4bit(torch.nn.Parameter):  # 参数壳，量化层挂权重用
        def __new__(cls, data, requires_grad=False, quant_type="nf4", quant_storage=None):
            return super().__new__(cls, data, requires_grad)

    class Linear4bit(torch.nn.Linear):  # 形状和行为都照普通线层，够验换层
        def __init__(self, in_features, out_features, bias=True, compute_dtype=None, quant_type="nf4", compress_statistics=True):
            super().__init__(in_features, out_features, bias=bias)

    class Int8Params(torch.nn.Parameter):
        def __new__(cls, data, requires_grad=False, has_fp16_weights=False):
            return super().__new__(cls, data, requires_grad)

    class Linear8bitLt(torch.nn.Linear):
        def __init__(self, in_features, out_features, bias=True, has_fp16_weights=False):
            super().__init__(in_features, out_features, bias=bias)

    stub.nn = types.SimpleNamespace(Linear4bit=Linear4bit, Params4bit=Params4bit, Linear8bitLt=Linear8bitLt, Int8Params=Int8Params)
    monkeypatch.setitem(sys.modules, "bitsandbytes", stub)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)  # 假包里 compute_dtype 落到 fp16 也不炸
    return stub


def test_spin_quant_without_bnb_reports_missing(tmp_path, monkeypatch):
    recipe = health_recipe(tmp_path, steps=2)  # 几何和数据的架子复用
    text = recipe.read_text(encoding="utf-8").replace("health: stop", "health: stop\n  precision: nf4\n  lora: {r: 4, alpha: 8}")
    recipe.write_text(text, encoding="utf-8")
    monkeypatch.setattr(spin, "_has_cuda", lambda: True)  # 假装有卡
    real = importlib.util.find_spec  # 真找法留着，别的包照找
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *rest: None if name == "bitsandbytes" else real(name, *rest))
    assert main.run(["spin", str(recipe), "--steps", "2", "--out", str(tmp_path / "out")]) == 4  # 缺包按 4 退


def test_spin_quant_alone_refuses_without_lora(tmp_path, monkeypatch):
    recipe = health_recipe(tmp_path, steps=2)
    text = recipe.read_text(encoding="utf-8").replace("health: stop", "health: stop\n  precision: nf4")
    recipe.write_text(text, encoding="utf-8")
    monkeypatch.setattr(spin, "_has_cuda", lambda: True)  # 假装有卡
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *rest: object())  # 假装包都在
    monkeypatch.setattr(model.Model, "quantize", lambda self, precision, device=None: True)  # 假装真压了
    assert main.run(["spin", str(recipe), "--steps", "2", "--out", str(tmp_path / "out")]) == 2  # 量化层没梯度，没挂件就填错


# ---------- 任务族与词表 ----------

def _prefs(tmp_path):
    """造一份偏好对数据。"""
    path = tmp_path / "pref.jsonl"
    rows = [{"prompt": "织：", "chosen": f"好的第{i}条", "rejected": f"差的第{i}条"} for i in range(8)]
    path.write_text("\n".join(json.dumps(one, ensure_ascii=False) for one in rows), encoding="utf-8")
    return [path]


def _groups(tmp_path):
    """造一份带奖励的组数据。"""
    path = tmp_path / "groups.jsonl"
    rows = [{"prompt": "织：", "responses": [f"候选{i}{n}" for n in "甲乙丙丁"], "rewards": [1.0, 0.5, 0.2, 0.0]} for i in range(8)]
    path.write_text("\n".join(json.dumps(one, ensure_ascii=False) for one in rows), encoding="utf-8")
    return [path]


def test_vocab_sets_embedding_size(geo, files, tmp_path):
    from features import tokenizer

    vocab = tokenizer.Vocab.train(files, vocab_size=300, limit=20)  # 从样本学一个小词表
    small = model.Model(geo)  # 词表还是 64 的模型
    geo.vocab = vocab.size  # 有词表就以它为准
    net = model.Model(geo)
    assert sum(one.numel() for one in net.parameters()) > sum(one.numel() for one in small.parameters())  # 词表变大，词嵌入跟着长
    job = trainer.Trainer(net, geo, short(geo, files, tmp_path / "run", vocab=vocab, steps=3))
    assert job.run() == 3  # 按词表编码也能跑


def test_dpo_one_step(geo, files, tmp_path):
    net = model.Model(geo)
    job = trainer.Trainer(net, geo, short(geo, _prefs(tmp_path), tmp_path / "run", task="dpo", steps=2))
    assert job.run() == 2  # 偏好对这条路能走完
    assert math.isfinite(job.history[-1])  # 损失是个正常数
    assert job.ref_model is not None  # 参考模型建起来了


@pytest.mark.parametrize("task", ["grpo", "rloo", "ppo"])
def test_group_tasks_run(geo, tmp_path, task):
    net = model.Model(geo)
    job = trainer.Trainer(net, geo, short(geo, _groups(tmp_path), tmp_path / "run", task=task, steps=1, group=4))
    assert job.run() == 1  # 组相对这一路能走完


def test_distill_needs_teacher(geo, files, tmp_path):
    net = model.Model(geo)
    with pytest.raises(Exception):  # 没给教师就得当场报
        trainer.Trainer(net, geo, short(geo, files, tmp_path / "run", task="distill", steps=1))


def test_distill_runs_with_teacher(geo, files, tmp_path):
    teacher_dir = tmp_path / "teacher"
    teacher_dir.mkdir()
    model.Model(geo).save(teacher_dir)  # 拿一份权重当教师
    net = model.Model(geo)
    job = trainer.Trainer(net, geo, short(geo, files, tmp_path / "run", task="distill", steps=1, teacher=teacher_dir))
    assert job.run() == 1  # 蒸馏能跑
    assert job.teacher is not None  # 教师建起来了
