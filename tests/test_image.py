"""图片的测试：读图、补边切块、归一，以及图片前缀进入模型后的形状、位置和影响。

约定：图片都在 tmp_path 现造；模型用 2 层 32 维、64 词表，5×5 原图按 2×2 切成 9 块。
纯文本与图片模型用相同种子起权重，逐值比较无图输出；改图只看文本打分，避免只改到前缀就算通过。

调用示例：
  python -m pytest tests/test_image.py -q
"""

import json                                      # 读回随模型保存的图片几何
from dataclasses import replace                  # 共用小几何，只改当前测试的开关

import numpy as np                               # 造出像素值可核对的图片
import pytest                                    # 临时目录、参数组合与缺包模拟
import torch                                     # 跑真实模型前向和反向
from PIL import Image                            # 写出 PNG/JPEG 测试文件

from features import image, model, recipe         # 图片、模型和配方的实际接口
from utils.fail import Failure                   # 核对缺包与配方错误的统一类型


# ---------- 小模型几何 ----------

@pytest.fixture                                                                                         # 多个模型测试共用尺寸
def geo():                                                                                              # 每次得到独立几何
    """原图故意不能整除块边长，最长文本也要放得下全部图片前缀。"""
    return model.Geometry(d_model=32, layers=2, heads=2, d_head=16, vocab=64, max_seq_len=24, image_side=5, patch=2)  # CPU 上直接跑


# ---------- 读取图片 ----------

@pytest.mark.parametrize("gray", [False, True])                                          # RGB 和灰度文件都要进得来
def test_load_png(tmp_path, gray):                                                       # PNG 可逐像素核对
    values = np.arange(15, dtype=np.uint8).reshape(3, 5)                                 # 每个位置的像素都不同
    pixels = values if gray else np.stack((values, values + 30, values + 60), axis=-1)   # RGB 各通道也不相同
    path = tmp_path / "photo.png"                                                       # 临时文件随测试清理
    Image.fromarray(pixels).save(path)                                                  # 真正编码成 PNG
    loaded = image.load(path)                                                          # 走实际读图入口
    expected = np.repeat(values[..., None], 3, axis=-1) if gray else pixels              # 灰度文件统一扩成 RGB
    assert loaded.shape == (3, 5, 3) and loaded.dtype == np.float32                      # 模型拿到固定三通道浮点数组
    np.testing.assert_array_equal(loaded, expected)                                     # PNG 无损，值应完全一致


@pytest.mark.parametrize("gray", [False, True])                                          # JPEG 也覆盖两种颜色模式
def test_load_jpeg(tmp_path, gray):                                                      # 用纯色避开压缩误差干扰断言
    pixels = np.full((3, 5) if gray else (3, 5, 3), 120, dtype=np.uint8)                 # 灰度与等值 RGB 都是同一块灰色
    path = tmp_path / "photo.jpg"                                                       # 不依赖仓库内的样图
    Image.fromarray(pixels).save(path)                                                  # 真正编码成 JPEG
    loaded = image.load(path)                                                          # Pillow 解码后转 RGB
    assert loaded.shape == (3, 5, 3) and loaded.dtype == np.float32                      # 输出契约与 PNG 一致
    np.testing.assert_allclose(loaded, 120, atol=1)                                      # JPEG 解码后仍保留原亮度


def test_load_reports_missing_pillow(tmp_path, monkeypatch):                            # 缺可选依赖时给出安装办法
    monkeypatch.setattr(image.importlib.util, "find_spec", lambda name: None)            # 模拟机器没装 Pillow
    with pytest.raises(Failure) as caught:                                              # 不把缺包伪装成图像损坏
        image.load(tmp_path / "photo.png")                                              # 入口先查依赖，尚未打开路径
    assert caught.value.kind == "missing"                                              # 项目统一缺包出口
    assert caught.value.hint == "pip install pillow"                                    # 可直接照着安装


# ---------- 切块与归一 ----------

@pytest.mark.parametrize("rgb", [False, True])                                           # 二维灰度数组与 RGB 都能切块
def test_patches_order_and_padding(rgb):                                                # 非方图同时考块数、行序和两条补边
    pixels = np.arange(1, 16, dtype=np.float32).reshape(3, 5)                            # 三行五列，切成两行三列的块
    expected = np.array([                                                              # 手写每块，避免拿另一遍切块当答案
        [[1, 2], [6, 7]], [[3, 4], [8, 9]], [[5, 0], [10, 0]],                         # 第一行三块，最后一块右边补零
        [[11, 12], [0, 0]], [[13, 14], [0, 0]], [[15, 0], [0, 0]],                     # 第二行三块，底边全部补零
    ], dtype=np.float32)                                                               # 补边不改变浮点类型
    if rgb:                                                                            # RGB 每个通道给不同数值
        pixels = np.stack((pixels, pixels * 2, pixels * 3), axis=-1)                    # 输入保持通道在最后
        expected = np.stack((expected, expected * 2, expected * 3), axis=-1)            # 期望块内也保留同一通道顺序
    blocks = image.patches(pixels, 2)                                                  # 实际按 2×2 切块
    assert blocks.shape == expected.shape and blocks.dtype == pixels.dtype             # 块数为六，灰度与 RGB 各自保留维度
    np.testing.assert_array_equal(blocks, expected)                                     # 逐像素核对行优先和边界补齐


def test_patches_exact_and_smaller_than_one_block():                                     # 整除时不多补，太小时仍返回一块
    exact = image.patches(np.ones((4, 4, 3), dtype=np.float32), 2)                       # 四块正好铺满原图
    assert exact.shape == (4, 2, 2, 3) and np.all(exact == 1)                            # 没有额外的黑边或空块
    small = image.patches(np.ones((1, 1, 3), dtype=np.float32), 2)                       # 原图比一块还小
    assert small.shape == (1, 2, 2, 3)                                                 # 仍返回一个完整大小的块
    np.testing.assert_array_equal(small[0, 0, 0], np.ones(3))                            # 左上角保留原像素
    assert np.count_nonzero(small) == 3                                                 # 其余三个像素都补零


def test_normalize_range_and_black_padding():                                           # 使用固定像素范围，不能按每张图拉伸
    pixels = np.array([[0, 127.5, 255]], dtype=np.float32)                               # 黑、半亮和白
    normal = image.normalize(pixels)                                                   # 一次线性归一
    np.testing.assert_array_equal(normal, [[-1, 0, 1]])                                 # 黑白两端与中点都准确
    assert normal.dtype == np.float32 and np.all((normal >= -1) & (normal <= 1))         # 输出在模型约定范围内
    blocks = image.normalize(image.patches(pixels, 2))                                  # 下游约定先补边再归一
    assert blocks[1, 0, 1] == -1 and blocks[0, 1, 0] == -1                               # 右侧和底部的黑边都变成 -1
    np.testing.assert_array_equal(pixels, [[0, 127.5, 255]])                             # 原数组没有被原地改写


# ---------- 模型图片前缀 ----------

@pytest.mark.parametrize("tie", [True, False])                                          # 共用词表和独立输出层都支持图片
def test_forward_shapes(geo, tie):                                                      # 最长文本加全部图片块也放得下
    net = model.Model(replace(geo, tie=tie))                                             # 起一个实际的小模型
    ids = torch.zeros((2, geo.max_seq_len), dtype=torch.long)                           # 两条最长文本
    pixels = np.arange(75, dtype=np.float32).reshape(5, 5, 3)                            # 一张 5×5 RGB 图
    blocks = torch.from_numpy(image.normalize(image.patches(pixels, geo.patch)))        # 走真实图片处理链
    batch = torch.stack((blocks, blocks))                                              # 下游把同样块数的图片拼成一批
    plain, pictured = net(ids), net(ids, patches=batch)                                 # 文字与带图两条前向
    assert plain.shape == (2, 24, 64)                                                  # 无图时长度仍与 ids 一致
    assert pictured.shape == (2, 33, 64)                                               # 九个图片位置加二十四个文本位置
    assert pictured[:, batch.shape[1]:].shape == plain.shape                            # 下游可直接切出文本打分
    assert torch.equal(pictured, net(ids, patches=batch.flatten(2)))                    # 预先展平的块也沿用同一接口


@pytest.mark.parametrize("tie", [True, False])                                          # 两种输出层都保持文本结果
@pytest.mark.parametrize("recompute", [False, True])                                    # 常规和重算模式都保持文本结果
def test_no_patches_keeps_text_forward(geo, tie, recompute):                             # 不是只比较两个默认参数写法
    torch.manual_seed(7)                                                              # 纯文本模型的初始种子
    plain = model.Model(replace(geo, image_side=0, tie=tie))                             # 不建图片参数的旧路径
    torch.manual_seed(7)                                                              # 开图后也必须保留文本初始化
    pictured = model.Model(replace(geo, tie=tie))                                       # 带投影层，但这次不给图片
    plain.grad_checkpoint = pictured.grad_checkpoint = recompute                       # 同一模式比较数值
    ids = torch.tensor([[1, 3, 5, 7], [2, 4, 6, 8]])                                   # 固定两条文本
    expected = plain(ids)                                                             # 纯文本几何的实际结果
    assert torch.equal(pictured(ids), expected)                                        # 开图片能力不会改变无图结果
    assert torch.equal(pictured(ids, patches=None), expected)                          # 显式不给图也逐值相同
    assert plain.image_proj is None and "image_proj.weight" not in plain.state_dict()   # 旧文本存档无需新增权重


def test_one_patch_changes_text_logits(geo):                                           # 真正验图片影响了文本计算
    torch.manual_seed(7)                                                              # 固定权重便于复现
    net = model.Model(geo).eval()                                                      # 使用实际注意力和前馈
    ids = torch.tensor([[1, 2, 3]])                                                    # 文本固定不变
    blocks = torch.zeros(1, 9, 2, 2, 3)                                               # 九个归一后的中灰图片块
    changed = blocks.clone()                                                          # 保留一份原图作对照
    changed[0, 4, 0, 0, 0] = 1                                                       # 只改第五块里的一个像素通道
    before, after = net(ids, patches=blocks), net(ids, patches=changed)                # 分别跑两次真实前向
    assert not torch.allclose(before[:, 9:], after[:, 9:])                             # 文本打分改变，证明真看到了图
    assert torch.equal(before[:, :4], after[:, :4])                                    # 更早的前缀不受未来块影响，因果性仍在


def test_rope_positions_continue_after_image(geo):                                     # 图片不单独使用一套位置编码
    net = model.Model(geo)                                                            # 一张共享的 RoPE 表
    seen = []                                                                         # 记录第一层实际拿到的位置分量
    hook = net.model.layers[0].register_forward_pre_hook(lambda block, args: seen.append(args[1:]))  # 只观察，不替换计算
    net(torch.tensor([[1, 2, 3]]), patches=torch.zeros(1, 9, 2, 2, 3))                  # 九块图片后接三个词
    hook.remove()                                                                     # 本次观察结束
    cos, sin = seen[0]                                                                # 这是实际传入 Block 的位置编码
    positions = torch.arange(12, dtype=torch.float32)                                  # 第一频率为 1，应连续走 0 到 11
    torch.testing.assert_close(cos[0, 0, :, 0], positions.cos(), rtol=0, atol=0)        # 文字从位置 9 接上，不能从 0 重来
    torch.testing.assert_close(sin[0, 0, :, 0], positions.sin(), rtol=0, atol=0)        # 正弦分量也走相同编号


def test_image_projection_gradient_with_checkpoint(geo):                               # 给后续训练接入留下可用的梯度链
    torch.manual_seed(7)                                                              # 固定模型和输入
    net = model.Model(geo)                                                            # 默认训练模式
    blocks = torch.linspace(-1, 1, 108).reshape(1, 9, 2, 2, 3)                        # 归一后的非零图片块
    gradients = []                                                                    # 比较普通前向与重算反向
    for recompute in (False, True):                                                   # 同一模型跑两种显存策略
        net.grad_checkpoint = recompute                                               # 切换已有的模型开关
        net.zero_grad(set_to_none=True)                                               # 每次只计当前梯度
        logits = net(torch.tensor([[1, 2, 3]]), patches=blocks)                       # 图片前缀进入真实计算
        logits[:, 9:].square().mean().backward()                                      # 只从文本位置反传
        gradients.append(net.image_proj.weight.grad.clone())                         # 记录图片投影实际拿到的梯度
    assert torch.isfinite(gradients[0]).all() and torch.count_nonzero(gradients[0]) > 0  # 图片层确实可训练
    torch.testing.assert_close(gradients[1], gradients[0])                             # 重算不会丢图或截断梯度


# ---------- 配方与存档 ----------

def test_recipe_accepts_image_geometry():                                             # 白名单与几何构造能完整接上
    data = {"model": {"geometry": {"image_side": 5, "patch": 2}}}                       # 其余几何保留默认值
    recipe.check(data)                                                               # 只在配方边界校验
    geo = model.Geometry.from_recipe(data)                                           # 模型照收已检查的字段
    assert geo.image_side == 5 and geo.patch == 2                                     # 两个值没有丢失
    assert model.Geometry.from_recipe({}).image_side == 0                            # 旧配方默认不接图


@pytest.mark.parametrize("field", ["image_side", "patch"])                              # 新字段只收整数
def test_recipe_rejects_image_geometry_text(field):                                   # 类型错误留在现有配方入口报
    with pytest.raises(Failure) as caught:                                           # 不在模型内部重复检查
        recipe.check({"model": {"geometry": {field: "2"}}})                           # 字符串不能冒充图片尺寸
    assert caught.value.kind == "schema"                                             # 沿用配方错误类型
    assert caught.value.where == f"model.geometry.{field}"                           # 指到具体字段


@pytest.mark.parametrize("tie", [True, False])                                         # 两种输出层的图片权重都能存回
def test_image_model_save_load_and_config(geo, tmp_path, tie):                         # 存档后图片能力仍完整
    net = model.Model(replace(geo, tie=tie)).eval()                                    # 随机起一份图片模型
    ids = torch.tensor([[1, 2, 3]])                                                  # 固定文本
    blocks = torch.linspace(-1, 1, 108).reshape(1, 9, 2, 2, 3)                        # 固定图片
    expected = net(ids, patches=blocks)                                              # 存盘前的输出
    state = net.save(tmp_path)                                                       # 用既有完整权重保存接口
    net.write_config(tmp_path, "fp32")                                               # 几何也照常落盘
    assert "image_proj.weight" in state                                              # 图片投影跟随模型保存
    config = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))        # 核对磁盘里的实际内容
    assert config["image_side"] == 5 and config["patch"] == 2                         # 重建图片层所需的两个字段在
    assert config["max_position_embeddings"] == 33                                   # 位置预算含九块图片和二十四个词
    again = model.Model(model.Geometry(**net.geo.as_dict())).eval()                   # 用落盘几何重建同一架子
    again.load(tmp_path)                                                             # 图片投影由既有同名张量加载路径灌回
    assert torch.equal(again(ids, patches=blocks), expected)                          # 图片前向逐值复现
