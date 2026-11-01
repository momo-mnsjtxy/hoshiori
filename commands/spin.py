"""命令 spin：验配方、出账，然后把训练真跑起来。

调用示例：
  ori spin sft.warp.yaml --dry-run                  # 只出账，不跑
  ori spin sft.warp.yaml --steps 20 --out out/t     # 跑 20 步，产物落 out/t
  ori spin sft.warp.yaml --resume out/t/step-000020 --steps 20   # 从第 20 步起再跑 20 步
  ori spin sft.warp.yaml --resume auto --out out/t  # 接 out/t 里最新一档
  ori spin sft.warp.yaml --launch torchrun --out out/t   # 多卡：按配方的 train.devices 起 torchrun

检查只在开头这一遍：数据在不在、配方过没过、显存估不估得下、有没有 torch、要卡的精度有没有卡、
要量化有没有 bitsandbytes、要 torchrun 有没有 torchrun。
过了这遍，后面一律信任入参，内部不查类型、不判空。
配方里 model.path 指向带权重的目录就从那儿续训，指不到就从随机权重起，并在日志里说一声。
"""

import importlib.util
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import main
import torch
from features import model, plan, recipe, tokenizer
from utils import logs
from utils.fail import fail


def command(args) -> int:
    """验配方、查数据、出账，然后交给训练。"""
    data = recipe.load(Path(args.recipe))
    recipe.check(data)

    task = data.get("task") or "sft"  # 任务族
    vocab = None  # 词表，配方给了才载
    tok = (data.get("model") or {}).get("tokenizer")
    if tok:  # 给了词表就按配方旁边找，找不到再按当前目录
        tok_path = Path(tok) if Path(tok).exists() else Path(args.recipe).parent / tok
        vocab = tokenizer.Vocab.load(tok_path)  # 坏词表在这里就停
    teacher = None  # 教师权重路径，distill 才用
    if (data.get("model") or {}).get("teacher"):
        raw = Path(data["model"]["teacher"])  # 教师可能写在配方旁边
        teacher = raw if raw.exists() else Path(args.recipe).parent / raw

    files = []  # 解析出来的数据路径
    for one in (data.get("data") or {}).get("files") or []:  # 数据不在就别开跑
        found = Path(one)
        if not found.exists() and not found.is_absolute():  # 相对路径先按配方旁边找，换目录启动也找得到
            found = Path(args.recipe).parent / one
        if not found.exists():
            fail("input", f"训练数据不在: {one}", hint="先用 ori weave 出 bolt/，或把相对路径改成相对配方的")
        files.append(found)

    table = plan.build(data)
    print(table.show())
    print(f"\n配方哈希 {recipe.stamp(data)}")
    if args.dry_run:  # 只出账就停
        return 0

    precision = (data.get("train") or {}).get("precision", "fp32")  # 精度
    if importlib.util.find_spec("torch") is None:  # 真跑要 torch
        fail("missing", "实际开跑要 torch", hint="pip install torch")
    if table.needs_gpu and shutil.which("nvidia-smi") is None:  # 这套精度要卡
        fail("missing", "这套精度要 NVIDIA 卡", hint="换 precision: fp32，或装驱动")
    if any("爆显存" in note for note in table.notes):  # 估下来就放不下，别等跑到一半
        fail("input", "按这套配置估下来会爆显存", hint="压 batch_size 或 max_seq_len，或换 precision: nf4")
    if _has_cuda() and precision in model.QUANT and importlib.util.find_spec("bitsandbytes") is None:  # 压不动就是假量化
        fail("missing", f"{precision} 要真压下去，机器上没有 bitsandbytes", hint="pip install bitsandbytes")
    if args.launch == "torchrun" and shutil.which("torchrun") is None:  # 多卡得靠它起进程
        fail("missing", "多卡要用 torchrun 起进程，机器上没有", hint="pip install torch，会自带 torchrun")
    if (data.get("train") or {}).get("strategy") in {"ddp", "fsdp2"} and args.launch != "torchrun":  # 配了多卡却没起多进程
        logs.warn("spin", "配了多卡但没加 --launch torchrun，按单进程跑", strategy=(data.get("train") or {}).get("strategy"))
    if args.launch == "torchrun" and os.environ.get("LOCAL_RANK") is None:  # 已经是子进程就别再套一层
        return _relaunch(args, data)

    from features import trainer  # torch 拦过了才导

    geo = model.Geometry.from_recipe(data)  # 几何，头数和头宽对不上在这里就停
    if vocab is not None:  # 有词表就以词表大小为准，别让几何和词表对不上
        geo.vocab = vocab.size
    out = Path(args.out) if args.out else Path("out") / time.strftime("run-%Y%m%d-%H%M%S")  # 没给就按时间戳起一个
    if os.environ.get("RANK", "0") == "0" and out.exists() and any(out.iterdir()) and not args.resume:  # 目录里有东西又不续训，别混着写；只让 0 号查，免得别的进程被它刚写的文件绊住
        fail("input", f"产物目录不是空的: {out}", hint="换一个 --out，或加 --resume 接着写")
    out.mkdir(parents=True, exist_ok=True)  # 目录先建

    if not _has_cuda():  # 没卡就明说，跑到一半才发现更费劲
        logs.warn("spin", "没有可用的卡，按 CPU fp32 跑", precision=precision)
    net = model.Model(geo, precision)  # 搭架子
    base = (data.get("model") or {}).get("path")  # 基座权重
    loaded = None  # 真载进来的基座目录
    if base:
        base_path = Path(base) if Path(base).is_absolute() else Path(args.recipe).parent / base  # 相对路径按配方旁边找
        if base_path.exists():  # 有就接着训
            net.load(base_path)
            loaded = base_path  # 记下基座，挂件配置要写
            logs.done("spin", "载入基座", path=str(base_path))
        else:  # 指不到就从随机权重起
            logs.warn("spin", "基座不在，从随机权重起", path=str(base_path))
    here = torch.device("cuda") if _has_cuda() and torch.cuda.is_available() else torch.device("cpu")  # 真能上卡才上
    net.to(here)  # 先搬设备再量化，量化层建在目标设备上
    quant = net.quantize(precision)  # 量化只能发生在灌完权重之后、建优化器之前
    if quant:
        logs.done("spin", "量化完成", precision=precision)
    print(f"模型  {net.count()} 参数 · {geo.layers} 层 · {geo.d_model} 维 · 词表 {geo.vocab}")

    train = data.get("train") or {}  # 训练节
    low = train.get("lora") or None  # 挂件配置，没写 r 就是没挂
    if quant and not low:  # 量化层自己没梯度，全量训等于没东西可训
        fail("input", f"{precision} 只能训 LoRA，配方里要写 train.lora", where="train.lora", hint="例: lora: {r: 16, alpha: 32}")
    if low:
        low = {"r": int(low.get("r") or 16), "alpha": float(low.get("alpha") or 32), "dropout": float(low.get("dropout") or 0.0), "target": low.get("target")}
    choice = trainer.Choice(
        out=out,
        files=files,
        batch_size=int(train.get("batch_size") or 1),
        grad_accum=int(train.get("grad_accum") or 1),
        lr=float(train.get("lr") or 2e-4),
        warmup=int(train.get("warmup_steps") or 0),
        weight_decay=float(train.get("weight_decay") or 0.0),
        clip=float(train.get("max_grad_norm") or 1.0),
        steps=int(args.steps or train.get("max_steps") or 0),
        epochs=float(train.get("epochs") or 1),
        save_every=int(args.save_every if args.save_every is not None else train.get("save_every") or 50),
        log_every=int(train.get("log_every") or 1),
        seed=int(train.get("seed") or 42),
        precision=precision,
        resume=args.resume,
        base=loaded,
        lora=low,
        strategy=train.get("strategy") or "single",  # 单卡 single，多卡 ddp / fsdp2
        devices=train.get("devices"),  # 只记账，真分配在 torchrun
        grad_checkpoint=bool(train.get("grad_checkpoint")),  # 前向重算换显存
        health=train.get("health") or "stop",  # 指标坏了停还是喊
        task=task,  # 任务族，决定读什么数据、算哪种损失
        vocab=vocab,  # 词表，给了就按它编
        group=int(train.get("group") or 4),  # 对齐任务一组几条
        beta=float(train.get("beta") or 0.1),  # DPO 的 KL 强弱
        clip_ratio=float(train.get("clip_ratio") or 0.2),  # 裁剪区间
        kl_coef=float(train.get("kl_coef") or 0.01),  # KL 罚系数
        temperature=float(train.get("temperature") or 2.0),  # 蒸馏温度
        teacher=Path(teacher) if teacher else None,  # 教师权重
    )
    if os.environ.get("RANK", "0") == "0":  # 多进程时只让 0 号往 loom.jsonl 写
        logs.begin(out)  # 之后的事件同时落 out/loom.jsonl
    job = None  # 建不起来时下面不发问
    try:
        job = trainer.Trainer(net, geo, choice)  # 建 job，进程组也在这里建
        ran = job.run()  # 真跑
    finally:
        logs.end()  # 收尾关掉日志文件
    print(f"\n跑到第 {ran} 步，产物在 {out}")
    if job.reason:  # 健康检查叫停的，把原因说清再退 1
        print(f"健康停机：{job.reason}")
        return 1
    if low:  # 挂件训完提示一下怎么合
        print("合并用 features.lora.merge，或 ori forge --format safetensors")
    return 0


def _relaunch(args, data: dict) -> int:
    """按配方的 devices 拼 torchrun 命令重跑自己，子进程的退出码原样交回去。

    参数按 args 重新拼，不抄 sys.argv：从测试里 main.run(...) 进来时 sys.argv 是 pytest 的。
    """
    devices = [str(one) for one in (data.get("train") or {}).get("devices") or [0]]  # 用哪几张卡
    argv = [str(Path(args.recipe).resolve())]  # 配方原样带过去
    if args.out:
        argv += ["--out", str(Path(args.out).resolve())]  # 产物目录跟着走
    if args.steps:
        argv += ["--steps", str(args.steps)]
    if args.save_every is not None:
        argv += ["--save-every", str(args.save_every)]
    if args.resume:
        argv += ["--resume", str(args.resume)]
    argv += ["--launch", "torchrun"]  # 子进程里有 LOCAL_RANK，凭它跳过再起
    cmd = [shutil.which("torchrun"), "--nproc_per_node", str(len(devices)), "--master-port", str(_free_port()), str(Path(main.__file__).resolve()), "spin", *argv]  # 脚本由 torchrun 用自己那个解释器起，这里只给脚本路径
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(devices)}  # 子进程只看得到这几张卡，rank 号就落在它们身上
    logs.done("spin", "交给 torchrun", cards=len(devices), devices=",".join(devices))
    return subprocess.call(cmd, env=env)  # 子进程跑完才返回


def _free_port() -> int:
    """挑一个当下空闲的端口给 rendezvous，免得和别的任务抢默认口。"""
    with socket.socket() as one:
        one.bind(("127.0.0.1", 0))
        return one.getsockname()[1]


def _has_cuda() -> bool:
    """torch 看不看得见卡。"""
    import torch

    return torch.cuda.is_available()
