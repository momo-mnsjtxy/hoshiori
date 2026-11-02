"""训练的测试：文本与图片批次、存档和续训能跑通，参考模型与教师独立载入、冻结并按任务释放。

调用方式：在项目根目录用 pytest 运行本文件；模型缩到 2 层 32 维 64 词表，CPU 上也能验。

调用示例：python -m pytest tests/test_trainer.py -q

约定：
  模型缩到 2 层 32 维 64 词表，数据是 10 条假样本，步数只给二十来步，CPU 上也只跑一两秒。
  词表 64 是故意比字节值小，顺带验字节取模那条路；序列压到 24，别让小矩阵撑大时间。
"""

import importlib.util
import json
import math
import shutil
from pathlib import Path
import socket
import sys
import time
import types
import weakref  # 检查任务收尾后冻结模型确实已释放
from dataclasses import replace

import numpy as np
import pytest
import torch
from torch.nn import functional as F

import main
from commands import spin
from features import image, lora, model, trainer


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


@pytest.fixture
def varied_files(tmp_path):
    """10 条长度各异且内容不重复的正文，覆盖不能整除三档的边界。"""
    texts = [chr(65 + at) + "x" * size for at, size in enumerate(range(2, 12))]  # 编码后长度从 3 到 12
    path = tmp_path / "varied.jsonl"                                            # 单独一份变长训练集
    path.write_text("\n".join(json.dumps({"text": one}) for one in texts), encoding="utf-8")  # 一行一条
    return [path]                                                               # 与 Choice.files 的接口一致


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


# ---------- 课程采样 ----------

def test_curriculum_early_batch_is_shorter(geo, varied_files, tmp_path):
    job = trainer.Trainer(model.Model(geo), geo, short(geo, varied_files, tmp_path / "run", batch_size=2, steps=5, curriculum=True))  # 五步正好覆盖三段进度
    job.step = 1                                                               # 第一步仍在短样本段
    early = job._batch()[1]                                                    # 标签里的非忽略位就是每条真实长度减一
    job.step = job.total                                                       # 收尾一步已经放开长样本
    late = job._batch()[1]                                                     # 同样取一个两条的微批
    early_mean = (early != -100).sum(dim=1).float().mean()                     # 早期批的平均答案长度
    late_mean = (late != -100).sum(dim=1).float().mean()                       # 后期批的平均答案长度
    assert late_mean > early_mean                                              # 课程确实从短到长
    job.close()                                                                # 本测试不跑完整训练，显式收尾


def test_curriculum_boundaries_keep_every_sample(geo, varied_files, tmp_path):
    job = trainer.Trainer(model.Model(geo), geo, short(geo, varied_files, tmp_path / "run", batch_size=1, steps=10, curriculum=True))  # 十条不能被三档整除
    taken = []                                                                 # 收集一整轮实际取到的样本
    for step in range(1, len(job.samples) + 1):                                # 依次跨过 1/3、2/3 两道进度边界
        job.step = step                                                        # 档位由 step / total 当场推算
        taken.append(tuple(job._next()))                                       # 每次只取一条，方便核对身份
    assert len(taken) == len(job.samples)                                      # 跑完一轮，取到条数就是总条数
    assert sorted(taken) == sorted(tuple(one) for one in job.samples)           # 边界处没有重复，也没有漏掉任何样本
    job.close()                                                                # 显式释放任务资源


def test_curriculum_off_keeps_original_next_order(geo, files, tmp_path):
    job = trainer.Trainer(model.Model(geo), geo, short(geo, files, tmp_path / "run", batch_size=1, steps=10))  # 不写开关就是默认关闭
    gen = torch.Generator().manual_seed(job.choice.seed * 100003 + job.rank)    # 原实现首轮使用的同一颗种子
    order = torch.randperm(len(job.samples), generator=gen).tolist()            # 原实现整轮 randperm 的顺序
    expected = [job.samples[at] for at in order]                               # 旧版 _next 应逐条交出的样本
    job.step = job.total                                                       # 关闭时训练进度不应影响顺序
    taken = [job._next() for _ in job.samples]                                 # 真走一整轮
    assert job.choice.curriculum is False and taken == expected                # 默认值与逐条顺序都和原来一致
    job.close()                                                                # 显式释放任务资源


# ---------- 图片批次 ----------

def test_image_batch_shape_and_label_prefix(geo, tmp_path, monkeypatch):                  # 多图串块、右补零和标签前缀一起核对
    path = tmp_path / "pictures.jsonl"                                                   # 两条图片样本放在同一批
    rows = [                                                                             # 第一条一张图，第二条两张图
        {"text": "abcdef", "images": ["one.png"]},                                    # 五个答案位和四个真实图片块
        {"text": "abcdefgh", "images": ["two.png", "three.png"]},                    # 七个答案位和八个真实图片块
    ]
    path.write_text("\n".join(json.dumps(one) for one in rows), encoding="utf-8")        # 写成训练器实际读取的 JSONL
    pictured = replace(geo, image_side=4, patch=2)                                       # 每张图固定切成四块
    values = {"one.png": 1.0, "two.png": 2.0, "three.png": 3.0}                       # 不同图片用不同块值辨认顺序

    def prepare(path, side, patch):                                                      # 只隔离读图，批次拼法走真实实现
        blocks = np.full((4, patch, patch, 3), values[Path(path).name], dtype=np.float32)            # 一张图交四块
        return blocks, len(blocks)                                                       # 契约同时返回块数

    monkeypatch.setattr(image, "prepare", prepare)                                       # 不让文件编码细节干扰批次断言
    job = trainer.Trainer(model.Model(pictured), pictured, short(pictured, [path], tmp_path / "run", batch_size=2, steps=1))  # 真建训练任务
    samples = iter(job.samples)                                                         # 文件顺序就是本次批次顺序
    monkeypatch.setattr(job, "_next", lambda: next(samples))                            # 跳过随机洗牌，只考拼批
    ids, labels, patches = job._batch()                                                 # 取出图片微批
    assert ids.shape == (2, 7)                                                         # 最长八个号，输入留下前七个
    assert patches.shape == (2, 8, 2, 2, 3)                                           # 两张图决定批内最多八块
    assert labels.shape == (2, 15)                                                     # 八个图片位加七个文本位
    assert torch.all(labels[:, :8] == -100)                                            # 全部图片块都不参与语言损失
    assert torch.equal(labels[0, 8:13].cpu(), torch.tensor(job.samples[0][0][1:]))       # 文本答案紧接图片前缀
    assert torch.all(labels[0, 13:] == -100)                                            # 短文本右侧仍是忽略位
    assert torch.all(patches[0, :4] == 1) and torch.all(patches[0, 4:] == 0)             # 单图样本只在右侧补零块
    assert torch.all(patches[1, :4] == 2) and torch.all(patches[1, 4:] == 3)             # 多图按路径次序首尾拼接
    job.close()                                                                         # 只验批次，不运行的任务显式释放


def test_text_batch_keeps_original_forward(geo, files, tmp_path, monkeypatch):           # 没有 image_side 时接口和前向都不带图片参数
    net = model.Model(geo)                                                              # 原来的纯文本模型
    forward = net.forward                                                               # 留住真实文本前向

    def text_only(ids):                                                                 # 故意只接旧接口的一个参数
        return forward(ids)                                                             # 数值仍由真实模型计算

    monkeypatch.setattr(net, "forward", text_only)                                      # 若训练器多传 patches 会直接报 TypeError
    job = trainer.Trainer(net, geo, short(geo, files, tmp_path / "run", steps=1))        # 普通文本任务
    batch = job._batch()                                                                # 先核对旧批次接口
    assert len(batch) == 2 and batch[0].shape == batch[1].shape                         # 仍只返回 ids 和同形 labels
    assert all(isinstance(one, list) for one in job.samples)                            # 样本仍是一串编号，不改内部结构
    assert math.isfinite(job._update())                                                 # 旧签名前向真跑一步仍有有限损失
    job.close()                                                                         # 不走 run 时显式释放任务资源


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


# --- 参考模型的独立存储 ---

@pytest.mark.parametrize("loaded", [False, True])
def test_reference_has_separate_storage(geo, tmp_path, loaded):
    """随机拷贝和基座重载都不与学生共用参数内存。"""
    net = model.Model(geo)                                               # 先建学生，保留随机权重
    base = tmp_path / "base" if loaded else None                         # 两条构建路径都要验
    if loaded:                                                          # 有基座时由训练器从盘上重载
        net.save(base)                                                  # 落一份与学生相同的初始权重
    choice = short(geo, _prefs(tmp_path), tmp_path / "run", task="dpo", base=base)  # 走偏好任务构建参考模型
    job = trainer.Trainer(net, geo, choice)                              # 参考模型单独存在
    student = dict(net.named_parameters())                              # 按名字逐个对照学生参数
    assert job.ref_model is not net                                     # 模型对象也必须独立
    for name, one in job.ref_model.named_parameters():                   # 每个参数都检查，避免漏掉共享层
        assert one.untyped_storage().data_ptr() != student[name].untyped_storage().data_ptr()  # 底层存储没有重用
        assert torch.equal(one, student[name])                          # 初始权重仍与基座一致
    before = next(job.ref_model.parameters()).detach().clone()          # 留下参考参数的原值
    with torch.no_grad():                                               # 模拟学生更新
        next(net.parameters()).add_(1)                                 # 只修改学生的一块权重
    assert torch.equal(next(job.ref_model.parameters()), before)        # 参考权重不跟着学生变化
    job.close()                                                         # 没有运行的任务显式释放


# --- 参考模型的设备与冻结 ---

@pytest.mark.parametrize("task", ["dpo", "grpo", "ppo"])
def test_reference_stays_frozen_on_cpu(geo, tmp_path, task):
    """三种对齐任务的参考模型都在 CPU，输出回到学生设备且不建图。"""
    here = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 有卡时同时验跨设备前向
    net = model.Model(geo).to(here)                                      # 学生按实际环境放设备
    files = _prefs(tmp_path) if task == "dpo" else _groups(tmp_path)      # 各任务使用对应数据
    job = trainer.Trainer(net, geo, short(geo, files, tmp_path / "run", task=task))  # 随机起步也得把拷贝搬到 CPU
    assert all(one.device.type == "cpu" and not one.requires_grad for one in job.ref_model.parameters())  # 全部参数在 CPU 且冻结
    assert all(one.device.type == "cpu" for one in job.ref_model.buffers())  # 位置编码缓存也不能留在卡上
    assert all(not one.training for one in job.ref_model.modules())      # 每层都切到 eval
    ids = torch.randint(1, geo.vocab, (2, 8), device=here)                # 批次先在学生设备上
    logps = job._logp_ref(ids)                                          # 真跑一次参考前向
    assert logps.device == ids.device and not logps.requires_grad       # 概率回到学生设备且没有梯度图
    assert torch.isfinite(logps).all()                                  # 设备搬运后仍有有效数值
    job.close()                                                         # 验完就释放冻结模型


# --- 参考模型从基座载入 ---

def test_reference_loads_base_instead_of_copy(geo, tmp_path, monkeypatch):
    """给 model.path 对应的 base 时必须调用 Model.load，不许深拷贝学生。"""
    net = model.Model(geo)                                               # 先建学生
    base = tmp_path / "base"                                            # 命令层传给 Choice.base 的权重目录
    net.save(base)                                                      # 保存初始权重
    before = next(net.parameters()).detach().clone()                    # 磁盘上的第一块权重
    with torch.no_grad():                                               # 让内存中的学生与磁盘不同
        next(net.parameters()).add_(1)                                 # 深拷贝学生会读到这个改动
    calls = []                                                          # 只记载入路径，不持有模型对象
    original = model.Model.load                                         # 真载入仍然执行

    def load(self, source):                                             # 包一层给载入计数
        calls.append(source)                                            # 记录每次 Model.load 的来源
        return original(self, source)                                  # 灌回真实权重

    def copied(memo):                                                   # 只拦这个学生的深拷贝
        pytest.fail("已有 model.path 时不应深拷贝学生")                  # 载入前偷偷拷贝也不能通过

    monkeypatch.setattr(model.Model, "load", load)                       # 统计 Trainer 内部的载入调用
    monkeypatch.setattr(net, "__deepcopy__", copied, raising=False)       # 避免影响优化器等对象自己的拷贝
    choice = short(geo, _prefs(tmp_path), tmp_path / "run", task="dpo", base=base)  # 路径由现有 Choice 字段传入
    job = trainer.Trainer(net, geo, choice)                              # 独立构造参考模型并读取磁盘
    assert calls == [base]                                              # Model.load 恰好为参考模型调用一次
    assert torch.equal(next(job.ref_model.parameters()), before)        # 参考权重来自磁盘，没有学生的内存改动
    job.close()                                                         # 释放本次构造出的参考模型


# --- 偏好训练与释放 ---

def test_dpo_one_step(geo, files, tmp_path):
    net = model.Model(geo)                                              # 随机起步的学生
    job = trainer.Trainer(net, geo, short(geo, _prefs(tmp_path), tmp_path / "run", task="dpo", steps=2))  # 两步偏好训练
    reference = weakref.ref(job.ref_model)                              # 弱引用不会替任务留住模型
    assert reference() is not None                                     # 开始前参考模型已建好
    assert job.run() == 2                                               # 偏好对这条路能走完
    assert math.isfinite(job.history[-1])                               # 损失是个正常数
    assert job.ref_model is None and reference() is None                 # 收尾后对象确实释放


@pytest.mark.parametrize("task", ["grpo", "rloo", "ppo"])
def test_group_tasks_run(geo, tmp_path, task):
    net = model.Model(geo)
    job = trainer.Trainer(net, geo, short(geo, _groups(tmp_path), tmp_path / "run", task=task, steps=1, group=4))
    assert job.run() == 1  # 组相对这一路能走完
    assert job.ref_model is None  # 收尾不留参考模型，RLOO 本来就没有


def test_distill_needs_teacher(geo, files, tmp_path):
    net = model.Model(geo)
    with pytest.raises(Exception):  # 没给教师就得当场报
        trainer.Trainer(net, geo, short(geo, files, tmp_path / "run", task="distill", steps=1))


# --- 教师载入、训练与释放 ---

def test_distill_runs_with_teacher(geo, files, tmp_path):
    teacher_dir = tmp_path / "teacher"                                  # 教师给产物根目录，继续支持 latest
    model.Model(geo).save(teacher_dir / "step-000001")                    # 拿一份独立权重当教师
    (teacher_dir / "latest").write_text("step-000001", encoding="utf-8")   # 指向实际权重所在的档
    here = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # 有卡时也要能跨设备蒸馏
    net = model.Model(geo).to(here)                                       # 学生放到实际训练设备
    choice = short(geo, files, tmp_path / "run", task="distill", steps=1, teacher=teacher_dir)  # 教师路径单独传入
    job = trainer.Trainer(net, geo, choice)                               # 载入教师并冻结
    teacher = weakref.ref(job.teacher)                                   # 只观察存续，不额外持有模型
    assert job.teacher is not net and not job.teacher.training           # 教师是独立的 eval 模型
    assert all(one.device.type == "cpu" and not one.requires_grad for one in job.teacher.parameters())  # 全部教师参数在 CPU 且冻结
    assert all(left.data_ptr() != right.data_ptr() for left, right in zip(net.parameters(), job.teacher.parameters()))  # 参数内存没有共用
    assert job.run() == 1                                                # 蒸馏能跑
    assert math.isfinite(job.history[-1])                                # CPU 教师给出了有效损失
    assert job.teacher is None and teacher() is None                     # 任务结束后教师也释放
