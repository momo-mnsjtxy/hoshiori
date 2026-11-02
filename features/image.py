"""图片：用 Pillow 读 PNG/JPEG，按行切块并补齐边界，把像素归一到模型用的范围。

调用方式：load(path) 返回 (高, 宽, 3) 的 RGB float32，像素仍在 [0, 255]；灰度图复制成三通道。
patches(array, size) 保留通道，把右侧和底部补零，返回 (块数, size, size, 通道)；灰度数组没有通道维。
normalize(array) 把 [0, 255] 映射到 [-1, 1]，应先切块再归一，让补的黑边也变成 -1。
下游负责把原图尺寸与 geometry.image_side 对齐、统一同批块数，再把数组转成与模型同设备、同精度的张量。
本轮只接单图前缀；messages 的图片解析、批次拼装、文本标签和损失对齐由后续训练接入负责。

调用示例（photo.png 是 224×224 的图片）：
  import torch                                                                   # 把数组交给模型
  from features import image                                                     # 图片的三个操作
  from features.model import Geometry, Model                                      # 模型与几何
  geo = Geometry(image_side=224, patch=16)                                         # 一张图切成 196 块
  net = Model(geo)                                                               # 随机起一个接图的模型
  blocks = image.normalize(image.patches(image.load("photo.png"), geo.patch))      # 黑边补好再归一
  batch = torch.from_numpy(blocks)[None].to(net.model.embed_tokens.weight)         # 补批维并对齐设备、精度
  ids = torch.tensor([[1, 2]], device=batch.device)                                # 两个文本位置
  logits = net(ids, patches=batch)                                               # 返回 (1, 198, 512)
  text_logits = logits[:, batch.shape[1]:]                                        # 去掉图片位置，接文本标签
"""

import importlib.util             # 在读图入口确认 Pillow 已安装
from pathlib import Path          # 路径也可以直接传 Path

import numpy as np                # 存像素、补边和切块

from utils.fail import fail       # 缺包走项目统一的 missing 出口


# ---------- 读取图片 ----------

def load(path: str | Path) -> np.ndarray:                                           # 下游从这里读原图
    """读 PNG/JPEG 为 RGB float32 数组，形状 (高, 宽, 3)，像素范围 [0, 255]。"""
    if importlib.util.find_spec("PIL") is None:                                     # 可选依赖只在读图入口查一次
        fail("missing", "读取图片要 Pillow", hint="pip install pillow")              # 告诉调用方缺什么、怎么装
    from PIL import Image                                                         # 纯文本使用不要求安装 Pillow
    with Image.open(path) as source:                                              # 读完释放图片文件
        return np.asarray(source.convert("RGB"), dtype=np.float32)                 # 灰度和 RGB 都交成三通道


# ---------- 切成图片块 ----------

def patches(array: np.ndarray, size: int) -> np.ndarray:                            # 大图拆成等大的块
    """按行优先返回 size×size 块；size 为正整数，右侧、底部补零，保留原通道维。"""
    height, width = array.shape[:2]                                                # 原图的高和宽
    channels = array.shape[2:]                                                     # 灰度没有通道维，RGB 是三通道
    padding = [(0, -height % size), (0, -width % size)] + [(0, 0)] * len(channels)    # 只在右侧和底部补到整块
    padded = np.pad(array, padding, mode="constant")                               # 用黑像素补齐边界
    rows, columns = padded.shape[0] // size, padded.shape[1] // size                 # 每行、每列各有多少块
    grid = padded.reshape(rows, size, columns, size, *channels)                    # 块行与块内行先拆开
    return grid.swapaxes(1, 2).reshape(rows * columns, size, size, *channels)        # 先左到右，再上到下排块


# ---------- 归一像素 ----------

def normalize(array: np.ndarray) -> np.ndarray:                                    # 原始像素换成模型输入
    """把 [0, 255] 像素线性映射到 [-1, 1] 的 float32，保持数组形状。"""
    return array.astype(np.float32) / 127.5 - 1.0                                  # 黑为 -1，白为 1，不按单图亮度缩放
