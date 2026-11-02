"""真实 GPU 精度验收：运行前后损失、梯度、FP8 算子计数与断点续训。

运行：OMP_NUM_THREADS=1 python tests/benchmark_precision.py --precision fp8 --out docs/evidence/fp8.json
必须有 CUDA；不支持的设备明确失败，不能拿 CPU 降级或模拟结果作为 GPU 验收。
"""

import argparse  # 选择精度与输出报告
import json  # 真实运行结果
from pathlib import Path  # 临时训练集和断点
import sys  # 独立入口定位项目
import time  # 实测运行时间

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 直接运行 tests 脚本也能导入产品代码
import torch  # 真实 CUDA 与 profiler
from features import model, precision, trainer  # 使用正式训练路径


def run(kind: str, out: Path) -> dict:
    """在当前 GPU 训练三步再从断点续两步，保留算子证据与环境。"""
    if not torch.cuda.is_available():
        raise RuntimeError("GPU 验收需要可用 CUDA")  # 不会悄悄用 CPU 代替
    out.parent.mkdir(parents=True, exist_ok=True)
    work = out.parent / (out.stem + "-run")  # 报告与完整断点对应
    work.mkdir(parents=True, exist_ok=True)
    source = work / "data.jsonl"
    source.write_text("\n".join(json.dumps({"text": "Starfur checks actual floating point training " + str(index)}) for index in range(16)), encoding="utf-8")
    geometry = model.Geometry(d_model=64, layers=2, heads=4, d_head=16, vocab=256, max_seq_len=32)
    torch.manual_seed(23)
    net = model.Model(geometry, kind).cuda()  # 模型先搬到 CUDA 再启用真实精度
    base_path = work / "base"  # QLoRA 续训必须重载同一冻结基座，不能重新随机起一份
    net.save(base_path)
    net.write_config(base_path, kind)
    net.quantize(kind)
    low = {"r": 4, "alpha": 8, "dropout": 0.0} if kind in model.QUANT else None
    choice = trainer.Choice(out=work, files=[source], precision=kind, steps=3, batch_size=2, lr=0.002, save_every=0, lora=low, base=base_path)
    torch.cuda.reset_peak_memory_stats()
    began = time.perf_counter()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        job = trainer.Trainer(net, geometry, choice)
        job.run()  # 真正前向、反向、优化与落盘
    operators = {event.key: event.count for event in profile.key_averages() if "_scaled_mm" in event.key}  # bnb::int8_scaled_mm 与 aten::_scaled_mm 不是同一算子
    calls = operators.get("aten::_scaled_mm", 0)  # 只数真实 PyTorch FP8 算子，不能用名字子串
    resumed = model.Model(geometry, kind).cuda()
    resumed.load(base_path)  # 挂件断点不包含冻结权重，必须先恢复原基座
    resumed.quantize(kind)
    continuation = trainer.Trainer(resumed, geometry, trainer.Choice(out=work, files=[source], precision=kind, steps=2, batch_size=2, lr=0.002, resume=work / "latest", lora=low, base=base_path))
    continuation.run()  # 优化器与主权重从同一个实际断点恢复
    torch.cuda.synchronize()
    report = {"precision": kind, "gpu": torch.cuda.get_device_name(), "capability": list(torch.cuda.get_device_capability()), "torch": torch.__version__, "cuda": torch.version.cuda, "steps_before_resume": job.step, "steps_after_resume": continuation.step, "losses": continuation.history, "last_grad_norm": continuation.last_norm, "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "seconds": time.perf_counter() - began, "fp8_linear_layers": sum(isinstance(layer, precision.Linear) for layer in net.modules()), "scaled_mm_calls": calls, "scaled_operators": operators, "checkpoint": str(work / "latest")}
    if kind == "fp8" and (calls == 0 or report["fp8_linear_layers"] == 0):
        raise RuntimeError("FP8 验收没有观察到真实 scaled_mm 调用")
    if continuation.step != 5 or not all(torch.isfinite(torch.tensor(value)) for value in continuation.history):
        raise RuntimeError("精度训练或续训没有达到五步有限损失")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report  # 输出只在全部验收条件通过后落盘


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16", "fp8", "nf4", "int8"], default="fp8")
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.precision, arguments.out), ensure_ascii=False, indent=2))
