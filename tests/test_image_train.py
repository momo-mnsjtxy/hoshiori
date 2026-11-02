"""图片训练端到端测试：从配方和两张 PNG 出发，确认训练循环真把图片送进模型并产生有限 loss。

调用方式：测试在临时目录写配方、bolt.jsonl 和图片，再通过 main.run 跑两步；训练日志里的两次
loss 都必须存在且有限，避免只验证批次形状却没有经过反向更新。

调用示例：python -m pytest tests/test_image_train.py -q
"""

import json                                    # 写训练样本并读取训练事件
import math                                    # 判断日志里的 loss 是否有限

import numpy as np                             # 现造两张颜色不同的图片
from PIL import Image                          # 把像素真正编码成 PNG

import main                                    # 从命令入口跑完整训练链


# ---------- 两步图片训练 ----------

def test_two_pngs_train_for_two_steps(tmp_path):                                      # 配方、图片、批次、前向和反向整条走通
    pictures = []                                                                     # 两张图分别属于两条样本
    for number, value in enumerate((48, 208), start=1):                               # 明暗不同，确保不是同一份输入
        path = tmp_path / f"image-{number}.png"                                      # 图片与配方放在同一临时目录
        pixels = np.full((3, 5, 3), value, dtype=np.uint8)                            # 原尺寸故意不是模型边长
        Image.fromarray(pixels).save(path)                                            # 写出真实 PNG 给 Pillow 读取
        pictures.append(path)                                                        # 路径随后写进 bolt 样本

    bolt = tmp_path / "bolt.jsonl"                                                   # 训练器按一行一条读取
    rows = [                                                                         # 每条正文对应一张图片
        {"text": "first image training sample", "images": [str(pictures[0])]},      # 第一条使用暗图
        {"text": "second image training sample", "images": [str(pictures[1])]},     # 第二条使用亮图
    ]
    bolt.write_text("\n".join(json.dumps(one) for one in rows), encoding="utf-8")    # 保留图片路径数组

    recipe = tmp_path / "image.warp.yaml"                                            # 一份能在 CPU 上很快跑完的小配方
    recipe.write_text(                                                               # 图片边长四，按二切成四块
        "version: 1\n"
        "task: pretrain\n"
        "model:\n"
        "  geometry:\n"
        "    d_model: 16\n"
        "    layers: 1\n"
        "    heads: 2\n"
        "    d_head: 8\n"
        "    vocab: 64\n"
        "    max_seq_len: 16\n"
        "    image_side: 4\n"
        "    patch: 2\n"
        "data:\n"
        "  files: [bolt.jsonl]\n"
        "train:\n"
        "  batch_size: 2\n"
        "  max_steps: 2\n"
        "  save_every: 0\n"
        "  log_every: 1\n"
        "  lr: 0.01\n",
        encoding="utf-8",
    )
    out = tmp_path / "out"                                                         # 训练产物与日志落在这里
    assert main.run(["spin", str(recipe), "--out", str(out)]) == 0                  # 从真实命令入口跑满两步
    events = [json.loads(line) for line in (out / "loom.jsonl").read_text(encoding="utf-8").splitlines()]  # 读取真实进度事件
    losses = [event["loss"] for event in events if event.get("text") == "训练"]     # 每一步日志都会带 loss
    assert len(losses) == 2 and all(math.isfinite(value) for value in losses)         # 两步都真正算出有限损失
