"""可复跑视觉检索验收：六种颜色、独立噪声测试图，真实卷积与跨批 InfoNCE 训练。

运行：OMP_NUM_THREADS=1 python tests/benchmark_vision.py --out docs/evidence/vision-cpu.json
这是受控的功能学习实验，不代表自然图文数据集上的语义检索性能。
"""

import argparse  # 指定报告输出位置
import json  # 机器可读结果
from pathlib import Path  # 模型产物和报告路径
import sys  # 允许直接从项目根运行本文件
import time  # 记录实际用时

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 从测试入口定位仓库
import torch  # CPU 真实训练，不注入检索分数
from features.model import Geometry, Model  # 产品模型和检索 API


def run(out: Path) -> dict:
    """训练颜色配对并在独立噪声图上检索，保存实际指标和模型。"""
    torch.manual_seed(41)  # 初始化、抽样与噪声均可重现
    torch.set_num_threads(1)  # 小模型多线程调度反而干扰计时
    geometry = Geometry(d_model=32, layers=1, heads=2, d_head=16, vocab=32, max_seq_len=8, image_side=8, patch=4)
    net = Model(geometry, align_weight=1.0, align_queue=12, align_temperature=0.1)
    colors = torch.tensor([[1, -1, -1], [-1, 1, -1], [-1, -1, 1], [1, 1, -1], [1, -1, 1], [-1, 1, 1]], dtype=torch.float32)
    blocks = colors[:, None, None, None, :].expand(-1, 4, 4, 4, -1).clone()  # 六个不同颜色配对
    ids = torch.tensor([[index + 1, index + 7, index + 13] for index in range(6)])  # 每种颜色独立文字描述编号
    heldout = (blocks + torch.randn_like(blocks) * 0.08).clamp(-1, 1)  # 测试噪声只造一次，不参与梯度
    before = net.retrieval(ids, heldout)  # 记录随机初始化基线
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.003)
    losses = []  # 记录真实对比目标走势
    began = time.perf_counter()
    for step in range(120):
        selected = torch.tensor([(step * 2) % 6, (step * 2 + 1) % 6])  # 每批只有两对，其他颜色依赖历史负例
        pixels = (blocks[selected] + torch.randn_like(blocks[selected]) * 0.08).clamp(-1, 1)
        optimizer.zero_grad(set_to_none=True)
        _, loss = net(ids[selected], patches=pixels)
        loss.backward()  # 卷积塔、投影和文本编码器共同学习
        optimizer.step()
        losses.append(float(loss.detach()))
    after = net.retrieval(ids, heldout)  # 不更改负例队列，直接计算两方向 Recall
    model_path = out.parent / "vision-cpu-model"
    net.save(model_path)  # 留下能够复查的真实权重
    net.write_config(model_path, "fp32")
    report = {"experiment": "controlled-color-retrieval", "device": "cpu", "torch": torch.__version__, "seed": 41, "steps": 120, "batch_size": 2, "train_pairs": 6, "heldout": "independent pixel noise; same six color classes", "vision": "two spatial Conv2d layers per patch", "negative_queue": net.contrast.capacity, "seconds": time.perf_counter() - began, "initial_loss": losses[0], "final_loss": losses[-1], "before": before, "after": after, "model": str(model_path)}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report  # 调用方可以用自己的阈值判断验收


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    print(json.dumps(run(parser.parse_args().out), ensure_ascii=False, indent=2))
