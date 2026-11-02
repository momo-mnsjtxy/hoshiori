"""两张不同 GPU 的 NCCL 续训验收：完整五步与三步存档再续两步逐权重对照。

运行：OMP_NUM_THREADS=1 python tests/benchmark_multigpu.py --out /tmp/ori-nccl-report
必须能看见两张物理 GPU；一张卡映射两次、CPU/gloo 都不能通过这份验收。
"""

import argparse  # 区分父进程与 torchrun 子进程
import json  # 各rank与最终对照证据
import os  # torchrun 提供 LOCAL_RANK/RANK
from pathlib import Path  # 数据、断点、报告位置
import socket  # 为本次任务选择空闲 rendezvous 端口
import subprocess  # 使用官方 torchrun 入口启动
import sys  # 独立运行时定位产品代码

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 保持测试入口可以独立调用
import torch  # 真 CUDA/NCCL 与权重差异
from safetensors.torch import load_file  # 对照最终产物，而非只看退出码
from features.model import Geometry, Model  # 正式模型
from features.trainer import Choice, Trainer  # 正式训练、存读、分布式路径


def worker(out: Path, stage: str) -> None:
    """torchrun 每个 rank 各占一张卡，执行指定的训练阶段。"""
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)  # 先认领设备，再分配模型，避免所有进程堆到 0 号卡
    torch.manual_seed(71)  # 独立启动的 full/first/resume 使用同一冻结初始化
    geometry = Geometry(d_model=32, layers=2, heads=2, d_head=16, vocab=64, max_seq_len=16)
    net = Model(geometry).cuda()
    folder = out / ("full" if stage == "full" else "split")
    choice = Choice(out=folder, files=[out / "data.jsonl"], steps=2 if stage == "resume" else 5, batch_size=2, lr=0.002, seed=71, strategy="ddp", devices=[0, 1], resume=folder if stage == "resume" else None)
    job = Trainer(net, geometry, choice)  # 在这里真正初始化 NCCL 并包装 DDP
    observed = {"rank": job.rank, "device": torch.cuda.current_device(), "backend": torch.distributed.get_backend(), "gpu": torch.cuda.get_device_name()}
    (out / f"{stage}-rank-{job.rank}.json").write_text(json.dumps(observed, indent=2), encoding="utf-8")
    if stage == "first":
        folder.mkdir(parents=True, exist_ok=True)
        for step in range(1, 4):
            job.step = step  # 原定总步数仍是五，余弦学习率不能误按三步重新规划
            loss = job._update()  # 正式训练一步，数据分片、梯度同步、优化器都由产品执行
            job.history.append(round(loss, 6))
        job.save()  # 所有rank一起存档，0号写出同一个可恢复断点
        torch.distributed.barrier()
        job.close()
    else:
        job.run()  # 完整训练或真实断点续训


def run(out: Path) -> dict:
    """确认两张物理 GPU，再启动三段 torchrun 并比较实际五步权重。"""
    if torch.cuda.device_count() < 2:
        raise RuntimeError("验收需要两张不同物理 GPU；当前不足两张")  # 单卡重复映射不能冒充完成
    properties = [torch.cuda.get_device_properties(index) for index in range(2)]
    identities = [str(getattr(one, "uuid", "")) for one in properties]
    if not all(identities) or len(set(identities)) != 2:
        raise RuntimeError("必须能核实两张不同 GPU 的 UUID")
    if out.exists() and any(out.iterdir()):
        raise RuntimeError("验收输出目录必须为空，避免旧断点污染结果")
    out.mkdir(parents=True, exist_ok=True)
    (out / "data.jsonl").write_text("\n".join(json.dumps({"text": f"NCCL resume example {number}"}) for number in range(12)), encoding="utf-8")
    for stage in ("full", "first", "resume"):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]  # 只为本次启动选端口，不复用固定全局端口
        command = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", "2", "--master-port", str(port), str(Path(__file__).resolve()), "--out", str(out.resolve()), "--worker", stage]
        subprocess.run(command, check=True)  # 任一rank失败就停止验收，不生成通过报告
    full = load_file(str(out / "full/step-000005/model.safetensors"))
    resumed = load_file(str(out / "split/step-000005/model.safetensors"))
    difference = max(float((full[name] - resumed[name]).abs().max()) for name in full)
    ranks = [json.loads((out / f"resume-rank-{index}.json").read_text()) for index in range(2)]
    if difference > 1e-6 or any(one["backend"] != "nccl" for one in ranks):
        raise RuntimeError(f"续训与连续训练不一致，最大差 {difference}")
    report = {"status": "passed", "gpus": identities, "ranks": ranks, "steps": {"full": 5, "before_resume": 3, "after_resume": 5}, "maximum_weight_difference": difference, "torch": torch.__version__}
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report  # 只有真实两卡、NCCL、权重一致才能得到 passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--worker", choices=["full", "first", "resume"])
    arguments = parser.parse_args()
    if arguments.worker:
        worker(arguments.out, arguments.worker)
    else:
        print(json.dumps(run(arguments.out), ensure_ascii=False, indent=2))
