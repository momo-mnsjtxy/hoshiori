"""图片的测试：读图、补边切块、归一，以及图片前缀进入模型后的形状、位置、对齐和影响。

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

from features import image, model, recipe, sample # 图片、模型、配方和样本的实际接口
from utils.fail import Failure                   # 核对缺包与配方错误的统一类型


# ---------- 小模型几何 ----------

@pytest.fixture                                                                                         # 多个模型测试共用尺寸
def geo():                                                                                              # 每次得到独立几何
    """原图故意不能整除块边长，最长文本也要放得下全部图片前缀。"""
    return model.Geometry(d_model=32, layers=2, heads=2, d_head=16, vocab=64, max_seq_len=24, image_side=6, patch=2)  # CPU 上直接跑


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
    assert isinstance(pictured(ids), torch.Tensor)                                     # align_weight 默认零，接口仍只是 logits
    assert plain.image_proj is None and plain.image_align is None                      # 纯文本几何不建任何图片层
    assert not any(name.startswith(("image_proj.", "image_align.")) for name in plain.state_dict())  # 旧文本存档无需新增权重


def test_image_alignment_state_and_loss(geo):                                          # 对齐层名字、结构、公式和返回接口一起核对
    net = model.Model(geo, align_weight=0.25).eval()                                   # 正权重开启二元组返回
    ids = torch.tensor([[1, 2, 3], [4, 5, 6]])                                        # 两条文本与两张图片一一配对
    blocks = torch.linspace(-1, 1, 216).reshape(2, 9, 2, 2, 3)                         # 每张图片九块，内容彼此不同
    logits, loss = net(ids, patches=blocks)                                            # 真走投影、对齐层和主体
    names = set(net.state_dict())                                                      # HF 风格名字由注册模块自动给出
    expected_names = {                                                                # GELU 没参数，只有两层线性各自的权重和偏置
        "image_align.linear_1.weight", "image_align.linear_1.bias",
        "image_align.linear_2.weight", "image_align.linear_2.bias",
    }
    assert expected_names <= names                                                    # 对齐张量会跟随完整 state_dict 保存
    assert logits.shape == (2, 12, geo.vocab) and loss.ndim == 0                       # 二元组第二项必须是标量张量
    assert torch.isfinite(loss) and loss.requires_grad                                # 对齐目标可以直接参与反向
    with torch.no_grad():                                                             # 手算同一次确定性前向的损失
        image_hidden = net.image_align(net.image_proj(blocks.flatten(2)))              # 图片先投影再过两层对齐
        hidden = net.model(ids, image_hidden)                                          # 主体给出同批文本隐状态
        expected = model.align_loss(image_hidden, hidden[:, image_hidden.shape[1]:]) * net.align_weight  # 公式只缩放一次
    torch.testing.assert_close(loss, expected)                                         # forward 交回的就是约定公式


def test_changing_image_alignment_changes_text_logits(geo):                            # 对齐层不能只是挂名保存却绕过前向
    torch.manual_seed(7)                                                              # 固定模型与输入便于复现
    net = model.Model(geo).eval()                                                      # 默认返回单个 logits 张量
    ids = torch.tensor([[1, 2, 3]])                                                   # 文本保持不变
    blocks = torch.linspace(-1, 1, 108).reshape(1, 9, 2, 2, 3)                        # 给对齐层非零图片输入
    before = net(ids, patches=blocks)[:, 9:].clone()                                  # 只比较真正使用图片的文本位置
    with torch.no_grad():                                                             # 把对齐层改成恒为零的映射
        for parameter in net.image_align.parameters():                                # 两层权重和偏置都属于这层
            parameter.zero_()                                                         # 图片前缀随之全部归零
    after = net(ids, patches=blocks)[:, 9:]                                           # 用同一图文再次前向
    assert not torch.allclose(before, after)                                           # 文本打分改变，证明对齐层位于有效路径上


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


def test_padding_block_is_not_an_attention_key(geo):                                  # 改补位内容不能污染任何有效位置
    torch.manual_seed(7)                                                              # 固定权重与图片投影
    net = model.Model(geo, align_weight=0.25).eval()                                   # 同时观察主输出和模型边界的对齐损失
    ids = torch.tensor([[1, 2, 3]])                                                   # 三个文本位置
    blocks = torch.linspace(-1, 1, 36).reshape(1, 3, 2, 2, 3)                         # 三块里最后一块当批内补位
    block_mask = torch.tensor([[1, 1, 0]])                                             # 前两块有效，最后一块无效
    changed = blocks.clone()                                                          # 只改 mask 为 0 的那一块
    changed[:, 2] = 999                                                              # 若它进入键或残差，输出会明显变化
    before, before_loss = net(ids, patches=blocks, block_mask=block_mask)              # 补位块在入口归零且注意力键被遮
    after, after_loss = net(ids, patches=changed, block_mask=block_mask)                # 同一布局再次前向
    assert torch.equal(before, after)                                                  # 连补位位置自己的输出也不依赖伪像素
    assert torch.equal(before_loss, after_loss)                                        # 模型交回的对齐损失也完全忽略补位块


def test_alignment_loss_ignores_padding_blocks(geo):                                  # 对齐均值只能汇总真实图片块
    image_hidden = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [20.0, -30.0]]])             # 第三块故意与前两块差很远
    text_hidden = torch.tensor([[[1.0, 1.0], [1.0, 1.0]]])                             # 文本均值方向固定
    block_mask = torch.tensor([[1, 1, 0]])                                             # 最后一块只是补位
    masked = model.align_loss(image_hidden, text_hidden, block_mask)                   # 走带 mask 的实际公式
    expected = model.align_loss(image_hidden[:, :2], text_hidden)                      # 等价于根本没把补位块传进来
    changed = image_hidden.clone()                                                     # 再把补位表示改成另一个极端值
    changed[:, 2] = torch.tensor([-500.0, 800.0])                                     # 有效均值不应看见它
    assert torch.equal(masked, expected)                                               # 公式只平均前两块
    assert torch.equal(model.align_loss(changed, text_hidden, block_mask), masked)     # 补位值变化不影响损失


def test_patch_at_interleaves_blocks_and_default_stays_prefix(geo):                    # 指定位置与旧前缀行为都看真实层输入
    torch.manual_seed(7)                                                              # 固定嵌入和图片映射
    net = model.Model(geo).eval()                                                      # 第一层前观察拼好的隐藏序列
    ids = torch.tensor([[1, 2, 3, 4]])                                                # 四个文本位置
    blocks = torch.linspace(-1, 1, 24).reshape(1, 2, 2, 2, 3)                         # 两块图片分别插到不同位置
    patch_at = torch.tensor([[0, 2]])                                                  # 第一块跟在 ids[0] 后，第二块跟在 ids[2] 后
    seen = []                                                                         # 收集第一层真正收到的序列
    hook = net.model.layers[0].register_forward_pre_hook(lambda block, args: seen.append(args[0].detach().clone()))  # 只观察层输入
    net(ids, patches=blocks, patch_at=patch_at)                                        # 走交错布局
    net(ids, patches=blocks)                                                           # 不给位置走旧前缀布局
    hook.remove()                                                                      # 两次观察已经完成
    with torch.no_grad():                                                             # 手工只算拼接两边的原始表示
        text = net.model.embed_tokens(ids)                                             # 四个文字嵌入
        pictures = net.image_align(net.image_proj(blocks.flatten(2)))                  # 两个图片块表示
        interleaved = torch.stack((text[:, 0], pictures[:, 0], text[:, 1], text[:, 2], pictures[:, 1], text[:, 3]), dim=1)  # 指定次序
        prefixed = torch.cat((pictures, text), dim=1)                                  # 旧行为是整段图片在前
    torch.testing.assert_close(seen[0], interleaved)                                   # 两块准确落在各自指定文本之后
    torch.testing.assert_close(seen[1], prefixed)                                      # patch_at 缺省时行为不变


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
    data = {"model": {"geometry": {"image_side": 6, "patch": 2}}}                       # 其余几何保留默认值
    recipe.check(data)                                                               # 只在配方边界校验
    geo = model.Geometry.from_recipe(data)                                           # 模型照收已检查的字段
    assert geo.image_side == 6 and geo.patch == 2                                     # 两个值没有丢失
    assert model.Geometry.from_recipe({}).image_side == 0                            # 旧配方默认不接图


def test_sample_keeps_and_validates_image_at(tmp_path):                               # 样本边界一次验完图片位置
    first = tmp_path / "first.png"                                                    # 样本入口只要求路径存在
    second = tmp_path / "second.png"                                                  # 图片内容留给后面的 prepare
    first.write_bytes(b"first")                                                      # 两张占位文件足够通过路径检查
    second.write_bytes(b"second")                                                    # 顺序与 image_at 一一对应
    record = {"text": "abcd", "images": [str(first), str(second)], "image_at": [1, 4]}  # 末尾下标允许等于文本长度
    clean = sample.read(record, sample.Policy(scrub=False))                            # 不让脱敏改变本测试正文
    assert clean["image_at"] == [1, 4]                                               # 规范样本原样保留合法位置
    bad = ([2, 2], [3, 1], [1, 5], [-1, 2], [True, 2])                               # 重复、倒序、越界、负数和伪整数
    for positions in bad:                                                             # 每一种都在样本入口拒收
        with pytest.raises(sample.Refused) as caught:                                 # 不留到训练 scatter 才报错
            sample.read({**record, "image_at": positions}, sample.Policy(scrub=False))  # 只替换待测位置
        assert caught.value.code == "bad_image_at"                                    # 独立原因码方便 rejects 统计
    with pytest.raises(sample.Refused) as caught:                                     # 数量也必须与图片一一对应
        sample.read({**record, "image_at": [1]}, sample.Policy(scrub=False))          # 两图只给一个位置
    assert caught.value.code == "bad_image_at"                                        # 同属位置格式错误


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
    assert all(name in state for name in (                                            # 两层对齐张量也走同一保存路径
        "image_align.linear_1.weight", "image_align.linear_1.bias",
        "image_align.linear_2.weight", "image_align.linear_2.bias",
    ))
    config = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))        # 核对磁盘里的实际内容
    assert config["image_side"] == 6 and config["patch"] == 2                         # 重建图片层所需的两个字段在
    assert config["max_position_embeddings"] == 33                                   # 位置预算含九块图片和二十四个词
    again = model.Model(model.Geometry(**net.geo.as_dict())).eval()                   # 用落盘几何重建同一架子
    again.load(tmp_path)                                                             # 图片投影由既有同名张量加载路径灌回
    assert torch.equal(again(ids, patches=blocks), expected)                          # 图片前向逐值复现
