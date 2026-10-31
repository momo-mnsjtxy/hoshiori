"""命令 spin：验配方、出账，然后把训练真跑起来。

调用示例：
  ori spin sft.warp.yaml --dry-run                  # 只出账，不跑
  ori spin sft.warp.yaml --steps 20 --out out/t     # 跑 20 步，产物落 out/t
  ori spin sft.warp.yaml --resume out/t/step-000020 --steps 20   # 从第 20 步起再跑 20 步
  ori spin sft.warp.yaml --resume auto --out out/t  # 接 out/t 里最新一档

检查只在开头这一遍：数据在不在、配方过没过、显存估不估得下、有没有 torch、要卡的精度有没有卡。
过了这遍，后面一律信任入参，内部不查类型、不判空。
配方里 model.path 指向带权重的目录就从那儿续训，指不到就从随机权重起，并在日志里说一声。
"""

import importlib.util
import shutil
import time
from pathlib import Path

from features import model, plan, recipe
from utils import logs
from utils.fail import fail


def command(args) -> int:
    """验配方、查数据、出账，然后交给训练。"""
    data = recipe.load(Path(args.recipe))
    recipe.check(data)

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

    if importlib.util.find_spec("torch") is None:  # 真跑要 torch
        fail("missing", "实际开跑要 torch", hint="pip install torch")
    if table.needs_gpu and shutil.which("nvidia-smi") is None:  # 这套精度要卡
        fail("missing", "这套精度要 NVIDIA 卡", hint="换 precision: fp32，或装驱动")
    if any("爆显存" in note for note in table.notes):  # 估下来就放不下，别等跑到一半
        fail("input", "按这套配置估下来会爆显存", hint="压 batch_size 或 max_seq_len，或换 precision: nf4")

    from features import trainer  # torch 拦过了才导

    geo = model.Geometry.from_recipe(data)  # 几何，头数和头宽对不上在这里就停
    out = Path(args.out) if args.out else Path("out") / time.strftime("run-%Y%m%d-%H%M%S")  # 没给就按时间戳起一个
    if out.exists() and any(out.iterdir()) and not args.resume:  # 目录里有东西又不续训，别混着写
        fail("input", f"产物目录不是空的: {out}", hint="换一个 --out，或加 --resume 接着写")
    out.mkdir(parents=True, exist_ok=True)  # 目录先建

    precision = (data.get("train") or {}).get("precision", "fp32")  # 精度
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
    print(f"模型  {net.count()} 参数 · {geo.layers} 层 · {geo.d_model} 维 · 词表 {geo.vocab}")

    train = data.get("train") or {}  # 训练节
    low = train.get("lora") or None  # 挂件配置，没写 r 就是没挂
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
    )
    logs.begin(out)  # 之后的事件同时落 out/loom.jsonl
    try:
        ran = trainer.Trainer(net, geo, choice).run()  # 真跑
    finally:
        logs.end()  # 收尾关掉日志文件
    print(f"\n跑到第 {ran} 步，产物在 {out}")
    if low:  # 挂件训完提示一下怎么合
        print("合并用 features.lora.merge，或 ori forge --format safetensors")
    return 0


def _has_cuda() -> bool:
    """torch 看不看得见卡。"""
    import torch

    return torch.cuda.is_available()
