"""图片：用 Pillow 读 PNG/JPEG，缩放、归一并按行切成模型使用的 RGB 图片块。

调用方式：prepare(path, side, patch) 一次返回 float32 图片块和块数；load、patches、normalize
保留为可单独调用的基础操作。读图失败会带着原路径抛出明确异常。

调用示例：
  import torch                                                              # 把图片块交给模型
  from features import image                                                # 图片准备入口
  blocks, count = image.prepare("photo.png", side=224, patch=16)             # 缩放后切成 196 块
  batch = torch.from_numpy(blocks)[None]                                     # 补上批维
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


# ---------- 一次准备完整图片 ----------

def prepare(path: str | Path, side: int, patch: int) -> tuple[np.ndarray, int]:     # 训练器从这里拿一张图的全部块
    """读图并缩放到 side×side，归一后返回 (块, patch, patch, 3) 与块数。"""
    try:                                                                            # 外部图片文件只在读取处接住一次
        pixels = load(path)                                                         # 统一读成 RGB float32
    except OSError as error:                                                        # 文件损坏、格式不认或途中读失败
        raise OSError(f"图片读取失败: {path}: {error}") from error                 # 路径和底层原因一起交给调用方
    from PIL import Image                                                           # load 已确认 Pillow 可用
    source = Image.fromarray(pixels.astype(np.uint8))                               # Pillow 缩放使用八位 RGB
    resized = np.asarray(source.resize((side, side)), dtype=np.float32)              # 所有图片统一成配方边长
    blocks = normalize(patches(resized, patch))                                     # 先切块再映射到模型范围
    return blocks, int(blocks.shape[0])                                              # 块数组和数量一次交给训练器
