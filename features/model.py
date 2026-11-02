"""模型：一个从零搭的小型 decoder-only 因果 Transformer，几何从配方的 model.geometry 读。
每层是 RMSNorm → 带 RoPE 的因果注意力 → 残差 → RMSNorm → 前馈 → 残差，末尾再归一、算词表打分。

调用方式：Model(Geometry.from_recipe(recipe)) 建模；load 灌权重，save 存 model.safetensors，config 描述几何。
image_side=0 不建图片参数，旧文本存档照常载入；非零时按原图边长预留 ceil(image_side/patch)² 个位置。
max_seq_len 仍是文本长度预算，图片块额外占位置；实际图片尺寸和同批块数由下游统一，本层不缩放图片。
forward(ids, patches=None, block_mask=None, patch_at=None) 接收 (批, 文本长)；可选 patches 是归一后的 RGB，
形状 (批, 块数, patch, patch, 3)，也可预先展平成 (批, 块数, patch*patch*3)。block_mask 以 1/0 标出
真实块和批内补位块；patch_at 给每块所跟随的文本下标，不给时仍把整段图片放在文本前。图片与文字共用 RoPE，
按拼好后的序列从 0 连续编号。默认 vision="conv" 使用两层块内卷积，旧图片权重需 vision="linear"。
embeddings 与 retrieval 使用独立文本编码器前向，避免文本偷看图片；对比损失采用对称 InfoNCE 和有界跨批负样本队列。
按拼好后的序列从 0 连续编号。align_weight 默认 0，此时只返回 (批, 块数+文本长, 词表)；权重大于 0 时
返回 (logits, 对齐损失)，供调用方与语言损失相加。
量化顺序仍是 load → quantize → 建优化器；无卡时保持 fp32，压过后可训的只剩 LoRA 的 A/B。
文本权重沿用 HF 名字，tie 为真时只存一份词表；图片投影和 image_align.linear_1/linear_2 随完整权重保存、载入。

调用示例：
  from pathlib import Path                                                    # 存档位置
  import torch                                                                # 造一批模型输入
  from features.model import Geometry, Model                                   # 几何与模型
  geo = Geometry(image_side=32, patch=16, vision="conv")                                      # 四块图片前缀
  model = Model(geo)                                                          # 按几何搭架子，权重随机起
  ids = torch.tensor([[1, 2]])                                                # 两个文本位置
  patches = torch.zeros(1, 4, 16, 16, 3)                                      # 已归一的 RGB 图片块
  mask = torch.tensor([[1, 1, 1, 0]])                                         # 最后一块只是批内补位
  logits = model(ids, patches=patches, block_mask=mask)                       # 形状 (1, 6, 512)
  text_logits = logits[:, patches.shape[1]:]                                  # 两个文本位置的打分
  model.save(Path("out/1"))                                                   # 存成 model.safetensors
  model.load(Path("out/1"))                                                   # 几何一致就能完整灌回
"""

import json
import hashlib  # 稳定文本指纹用来排除队列中的重复负例
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from utils import logs
from utils.fail import fail

EPS = 1e-6  # RMSNorm 的分母下限，免得分母是 0 时除出无穷
DTYPES = {"fp8": torch.bfloat16, "fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16, "nf4": torch.float32, "int8": torch.float32}
QUANT = {"nf4", "int8"}  # 真要压下去的精度；其余是普通 dtype，压不压都不经过 bitsandbytes


# ---------- 模型几何 ----------

@dataclass                        # 几何字段可直接传给构造器
class Geometry:                   # 文本和图片共用一份几何
    """模型几何。默认值是一套能在 CPU 上几十秒跑完的演示尺寸。"""

    d_model: int = 128             # 隐藏维度
    layers: int = 4                # 层数
    heads: int = 4                 # 注意力头数
    d_head: int = 32               # 每个头的宽度，heads × d_head 要等于 d_model
    vocab: int = 512               # 词表大小
    max_seq_len: int = 128         # 最长文本序列，图片块额外预留位置
    rope_base: float = 10000.0     # RoPE 的底数
    tie: bool = True              # 输出层是否复用词表权重
    image_side: int = 0           # 原图边长，0 表示不接图
    vision: str = "conv"          # 新视觉模型默认两层卷积；旧图片存档显式用 linear
    patch: int = 16               # 图片块边长，接图时用正整数，右侧和底部可补齐

    @classmethod
    def from_recipe(cls, recipe: dict) -> "Geometry":
        """从配方的 model.geometry 节取几何，没写的就是默认值。"""
        given = (recipe.get("model") or {}).get("geometry") or {}  # 整节没写就当空
        geo = cls(**given)  # 白名单在 recipe.check 里已经过了一遍，这里照收
        if geo.heads * geo.d_head != geo.d_model:  # 头宽乘头数必须正好铺满隐藏维
            fail("schema", "heads × d_head 对不上 d_model", where="model.geometry.d_head", hint=f"{geo.heads} × {geo.d_head} 要等于 {geo.d_model}")
        return geo

    @classmethod
    def from_config(cls, data: dict) -> "Geometry":
        """从模型自己的 config.json 重建几何；旧视觉配置没有 vision 时按线性塔读取。"""
        fields = {"d_model": "hidden_size", "layers": "num_hidden_layers", "heads": "num_attention_heads", "d_head": "head_dim", "vocab": "vocab_size", "rope_base": "rope_theta", "tie": "tie_word_embeddings"}
        values = {name: data[key] for name, key in fields.items() if key in data}  # 不复用学生的隐藏维或层数
        values.update({name: data[name] for name in ("image_side", "patch") if name in data})
        values["vision"] = data.get("vision", "linear")                          # 旧图片权重没有卷积参数，明确保留旧结构
        blocks = math.ceil(values.get("image_side", 0) / values.get("patch", 16)) ** 2
        values["max_seq_len"] = data.get("text_max_seq_len", data.get("max_position_embeddings", 128) - blocks)
        return cls(**values)                                                      # 其余可选字段保持几何的标准默认值

    def dtype(self, precision: str) -> torch.dtype:
        """按精度挑权重类型。CPU 一律 fp32，nv 卡上才有半精度。"""
        return torch.float32 if not torch.cuda.is_available() else DTYPES[precision]

    def as_dict(self) -> dict:
        """落盘用的几何，续训时照着它把架子重新搭出来。"""
        return dict(self.__dict__)


class RMSNorm(nn.Module):
    """按最后一维的均方根做归一，再乘一个学出来的缩放。"""

    def __init__(self, size: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))  # 初始缩放是 1，归一后的值原样过

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """归一并缩放，形状不变。"""
        scale = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS)  # 均方根的倒数
        return x * scale * self.weight  # 乘回可学缩放


def whole(one: torch.Tensor) -> torch.Tensor:
    """把参数拼回整份、去梯度、落 CPU。fsdp2 下拿到的是分片，直接写盘只会得到一块。"""
    return (one.full_tensor() if hasattr(one, "full_tensor") else one).detach().cpu().contiguous()  # full_tensor 会跨进程把分片拼回来


def compute_dtype() -> torch.dtype:
    """量化层内部算账用的类型：卡认 bf16 就用它，不认就 fp16。"""
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def _quant(old: nn.Linear, precision: str, bnb, device: torch.device) -> nn.Module:
    """照原层的形状造一个量化层，把冻结的权重搬过去；落到卡上那一步才真压。"""
    if precision == "nf4":  # 4bit
        new = bnb.nn.Linear4bit(old.in_features, old.out_features, bias=old.bias is not None, compute_dtype=compute_dtype(), quant_type="nf4")  # 4bit 线层，算的时候临时升到 compute_dtype
        new.weight = bnb.nn.Params4bit(old.weight.detach(), requires_grad=False, quant_type="nf4")  # 先挂 fp 数据，交给后面的 .to 压成 4bit
    else:  # int8
        new = bnb.nn.Linear8bitLt(old.in_features, old.out_features, bias=old.bias is not None, has_fp16_weights=False)  # int8 线层
        new.weight = bnb.nn.Int8Params(old.weight.detach(), requires_grad=False, has_fp16_weights=False)  # 同样等落卡时压，第一次前向才真做
    if old.bias is not None:  # 有偏置就搬过去冻住
        new.bias = nn.Parameter(old.bias.detach().clone(), requires_grad=False)
    return new.to(device)  # 落卡即量化


def rope_cache(length: int, width: int, base: float, dtype: torch.dtype) -> tuple:
    """预先生成旋转位置编码的 cos/sin，形状 (1, 1, 序列, 头宽/2)。"""
    step = torch.arange(length, dtype=torch.float32)  # 每个位置一个标量
    pair = torch.arange(0, width, 2, dtype=torch.float32)  # 每两维一组，组内旋转
    freq = base ** (-pair / width)  # 组的角频率，越靠后转得越慢
    angle = step[:, None] * freq[None, :]  # 位置 × 频率，得到每组的旋转角
    return angle.cos().to(dtype)[None, None], angle.sin().to(dtype)[None, None]  # 补批和头两维，好跟张量对齐


def rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """把相邻两维当一对（x0, x1）转成（-x1, x0）再按角度混。"""
    pairs = x.view(*x.shape[:-1], -1, 2)  # 头宽拆成 组 × 组内两值
    left, right = pairs[..., 0], pairs[..., 1]  # 一组的两个分量
    turned = torch.stack((-right, left), dim=-1)  # 旋转 90 度
    return (pairs * cos.unsqueeze(-1) + turned * sin.unsqueeze(-1)).flatten(-2)  # 按角度加权后拼回头宽


def sequence_positions(text_steps: int, patch_at: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """算出交错序列里文字与图片块各自落在哪；patch_at 已按块顺序非递减。"""
    anchors = patch_at.clamp(min=-1, max=text_steps)                           # -1 是内部使用的前缀位，文本长度及以后都是后缀位
    text = torch.arange(text_steps, device=patch_at.device)                    # 原文本位置 0 到 text_steps-1
    before = (anchors.unsqueeze(-1) < text).sum(dim=1)                         # 每个文本位置之前已经插入了多少块
    text_at = text.unsqueeze(0) + before                                      # 原文本位置加上前面的块数就是新位置
    order = torch.arange(patch_at.shape[1], device=patch_at.device)            # 同一锚点的块仍按原顺序首尾相接
    block_at = (anchors + 1).clamp(min=0, max=text_steps) + order.unsqueeze(0)  # 图片紧跟锚点，前面已有的块也要让位
    return text_at.long(), block_at.long()                                    # scatter / gather 的下标统一用 long


class Attention(nn.Module):
    """多头因果注意力，权重名照 q_proj / k_proj / v_proj / o_proj 起。"""

    def __init__(self, geo: Geometry) -> None:
        super().__init__()
        self.heads = geo.heads  # 头数
        self.d_head = geo.d_head  # 头宽
        self.q_proj = nn.Linear(geo.d_model, geo.heads * geo.d_head, bias=False)  # 查询
        self.k_proj = nn.Linear(geo.d_model, geo.heads * geo.d_head, bias=False)  # 键
        self.v_proj = nn.Linear(geo.d_model, geo.heads * geo.d_head, bias=False)  # 值
        self.o_proj = nn.Linear(geo.heads * geo.d_head, geo.d_model, bias=False)  # 拼回隐藏维

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """前向：投影 → 分头 → 转位置 → 因果注意力 → 合头 → 投影回。"""
        batch, steps, _ = x.shape  # 批、序列、隐藏维
        split = lambda part: part.view(batch, steps, self.heads, self.d_head).transpose(1, 2)  # 拆成 (批, 头, 序列, 头宽)
        query, key, value = split(self.q_proj(x)), split(self.k_proj(x)), split(self.v_proj(x))  # 三路投影后分头
        query, key = rotate(query, cos, sin), rotate(key, cos, sin)  # 只转查询和键，值不动
        out = F.scaled_dot_product_attention(                                  # 缩放点积在分数上同时执行因果与补位遮罩
            query, key, value,                                                  # 查询、键和值已经分头并带上位置
            attn_mask=attention_mask,                                           # 有补位块时，假值表示这个键不能被任何查询看到
            is_causal=attention_mask is None,                                   # 没补位时保留原来的高效因果路径和数值行为
        )
        return self.o_proj(out.transpose(1, 2).reshape(batch, steps, -1))  # 头拼回去再投影


class Feed(nn.Module):
    """前馈：先升到 4 倍宽，用 SiLU 门控，再压回隐藏维。"""

    def __init__(self, geo: Geometry) -> None:
        super().__init__()
        wide = geo.d_model * 4  # 中间层宽度
        self.gate_proj = nn.Linear(geo.d_model, wide, bias=False)  # 门控那一路
        self.up_proj = nn.Linear(geo.d_model, wide, bias=False)  # 值那一路
        self.down_proj = nn.Linear(wide, geo.d_model, bias=False)  # 压回隐藏维

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """门控乘值，再压回去。"""
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))  # 两路相乘就是门控


class Block(nn.Module):
    """一层：两道 pre-norm 裹着注意力和前馈，各自带残差。"""

    def __init__(self, geo: Geometry) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(geo.d_model)  # 进注意力前先归一
        self.self_attn = Attention(geo)  # 注意力
        self.post_attention_layernorm = RMSNorm(geo.d_model)  # 进前馈前再归一
        self.mlp = Feed(geo)  # 前馈

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """残差两跳，平铺直叙。"""
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, attention_mask)  # 注意力这一跳，补位键在分数里被挡住
        return x + self.mlp(self.post_attention_layernorm(x))  # 前馈这一跳


class Core(nn.Module):
    """主体那一摞：词表、各层、收尾归一。名字前面挂 model. 就是为了对上 HF 的叫法。"""

    # ---------- 建主体 ----------

    def __init__(self, geo: Geometry, dtype: torch.dtype) -> None:                         # 按几何搭词表和各层
        super().__init__()                                                              # 注册模型模块
        self.embed_tokens = nn.Embedding(geo.vocab, geo.d_model)                          # 文本词表
        nn.init.normal_(self.embed_tokens.weight, std=0.02)                               # 小方差起，初始损失靠近 ln(词表)
        self.layers = nn.ModuleList([Block(geo) for _ in range(geo.layers)])               # 文字和图片走同一摞层
        self.norm = RMSNorm(geo.d_model)                                                 # 收尾归一
        image_steps = math.ceil(geo.image_side / geo.patch) ** 2 if geo.image_side else 0  # 非整除边长也要容下补边的块
        length = geo.max_seq_len + image_steps                                           # 保留原文本预算，再加图片位置
        cos, sin = rope_cache(length, geo.d_head, geo.rope_base, dtype)                    # 图片、文字共用一张位置表
        self.register_buffer("rope_cos", cos, persistent=False)                          # 不存盘，载入时重建
        self.register_buffer("rope_sin", sin, persistent=False)                          # 同一位置的正弦分量

    # ---------- 图片块与文本一起过各层 ----------

    def forward(
        self,
        ids: torch.Tensor,
        image_hidden: torch.Tensor = None,
        block_mask: torch.Tensor = None,
        patch_at: torch.Tensor = None,
    ) -> torch.Tensor:                                                                  # image_hidden 已投影并对齐到隐藏维
        """查词表 → 拼图片块 → 合成因果/补位遮罩 → 共用 RoPE 逐层计算 → 收尾归一。"""
        text = self.embed_tokens(ids)                                                   # 文本仍从同一张词表取嵌入
        x = text                                                                        # 没图时就是原来的纯文本序列
        key_mask = None                                                                 # 不给块遮罩时，所有键都按原行为有效
        if image_hidden is not None and patch_at is None:                               # 旧样本仍把所有图片块放在文本之前
            x = torch.cat((image_hidden, text), dim=1)                                  # 图片在前，所有文本都能看见图片
            if block_mask is not None:                                                 # 只在调用方明确给出真实块范围时造遮罩
                text_mask = torch.ones(ids.shape, dtype=torch.bool, device=ids.device)  # 每个文本键都有效
                key_mask = torch.cat((block_mask.bool(), text_mask), dim=1)             # 图片键按 1/0，文本键全放行
        elif image_hidden is not None:                                                  # patch_at 把每块放到指定文本位置之后
            text_at, block_at = sequence_positions(text.shape[1], patch_at)             # 两类位置共用唯一的布局算法
            steps = text.shape[1] + image_hidden.shape[1]                               # 拼完仍是文本数加块数
            x = text.new_empty((text.shape[0], steps, text.shape[2]))                   # 先留出完整交错序列
            x.scatter_(1, text_at.unsqueeze(-1).expand_as(text), text)                  # 文本逐项放回自己的新位置
            x.scatter_(1, block_at.unsqueeze(-1).expand_as(image_hidden), image_hidden) # 图片块紧跟各自锚点
            if block_mask is not None:                                                 # 交错位置下同样只遮图片补位键
                key_mask = torch.ones((text.shape[0], steps), dtype=torch.bool, device=text.device)  # 文本键先全放行
                key_mask.scatter_(1, block_at, block_mask.bool())                       # 图片位置写入真实块标记
        steps = x.shape[1]                                                              # 实际总长度包括全部图片位
        attention_mask = None                                                           # 无补位块时继续使用 SDPA 自带因果遮罩
        if key_mask is not None:                                                       # 有补位块时把两种限制合成一张分数遮罩
            causal = torch.ones((steps, steps), dtype=torch.bool, device=x.device).tril()  # 每个查询只能看自己和过去
            attention_mask = causal[None, None] & key_mask[:, None, None, :]            # 补位图片不能作为任何查询的键
        cos, sin = self.rope_cos[:, :, :steps], self.rope_sin[:, :, :steps]            # 按拼好后的顺序从 0 连续编号
        for block in self.layers:                                                       # 一层一层往下走
            if attention_mask is None:                                                 # 保留旧调用形状，观察钩子也不多一个空参数
                x = block(x, cos, sin)                                                 # 原来的纯因果注意力路径
            else:                                                                      # 有图片补位才传合成遮罩
                x = block(x, cos, sin, attention_mask)                                 # 每层都挡住同一批补位键
        return self.norm(x)                                                             # 收尾归一


class VisionTower(nn.Module):
    """两层真实卷积提取块内空间特征；块与块的上下文随后交给语言 Transformer。"""

    def __init__(self, patch: int, size: int) -> None:
        super().__init__()                                                       # 注册卷积权重，完整存档会自动收集
        self.patch = patch                                                       # 展平输入也按这份几何还原 RGB
        self.conv_1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)                  # 第一层学习局部颜色和边缘
        self.conv_2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)                 # 第二层组合局部视觉特征
        self.output = nn.Linear(64, size, bias=False)                            # 池化后投到语言隐藏维

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        """输入 (批, 块, RGB 展平)，输出 (批, 块, 隐藏维)。"""
        batch, blocks = pixels.shape[:2]                                         # 块独立编码，不把相邻两张图混成一张
        grid = pixels.reshape(batch * blocks, self.patch, self.patch, 3).permute(0, 3, 1, 2)
        hidden = F.gelu(self.conv_1(grid))                                       # 保留二维邻域的卷积特征
        hidden = F.gelu(self.conv_2(hidden)).mean(dim=(-2, -1))                   # 池化得到每块的视觉描述
        return self.output(hidden).reshape(batch, blocks, -1)                   # 恢复批与块两个维度


class ImageContrast(nn.Module):
    """对称 InfoNCE 与有界历史负样本；历史表示 detach，评估时绝不入队。"""

    def __init__(self, size: int, capacity: int = 1024, temperature: float = 0.07) -> None:
        super().__init__()                                                       # 队列属于本模型的训练状态
        self.capacity, self.temperature = capacity, temperature                  # 容量上限与温度都来自配方
        self.register_buffer("images", torch.empty(0, size), persistent=False)  # 只在训练断点单独保存，不污染推理权重
        self.register_buffer("texts", torch.empty(0, size), persistent=False)   # 图与文总是成对入队
        self.register_buffer("keys", torch.empty(0, dtype=torch.long), persistent=False)  # 排除重复文本的假负样本

    def forward(self, pictures: torch.Tensor, texts: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        """输入已归一化的图文向量，当前第 i 图与第 i 文是唯一标注正对。"""
        if not len(pictures):
            return (pictures.sum() + texts.sum()) * 0                            # 无图批仍保留可反传的零损失
        old_images = self.images.detach().clone().float()                        # 克隆避免本次入队改动反向所需的值
        old_texts = self.texts.detach().clone().float()                           # 历史负样本不建立反向链
        image_keys = torch.cat((pictures, old_images), dim=0)                    # 本批和历史都可成为候选
        text_keys = torch.cat((texts, old_texts), dim=0)
        all_keys = torch.cat((keys, self.keys), dim=0)                            # 同文本历史项不能充当负样本
        duplicate = keys[:, None] == all_keys[None, :]
        duplicate[:, :len(keys)].fill_diagonal_(False)                           # 自己对应的正例必须保留
        image_scores = (pictures @ text_keys.T / self.temperature).masked_fill(duplicate, -torch.inf)
        text_scores = (texts @ image_keys.T / self.temperature).masked_fill(duplicate, -torch.inf)
        labels = torch.arange(len(keys), device=keys.device)                     # 正例列位于当前批最前面
        loss = (F.cross_entropy(image_scores, labels) + F.cross_entropy(text_scores, labels)) / 2
        if self.training and self.capacity:
            self.images = torch.cat((old_images, pictures.detach()), dim=0)[-self.capacity:]
            self.texts = torch.cat((old_texts, texts.detach()), dim=0)[-self.capacity:]
            self.keys = torch.cat((self.keys, keys), dim=0)[-self.capacity:].detach()                       # 最旧对自动淘汰，显存不会随步数增长
        return loss                                                              # 温度只在相似度 softmax 前应用


class ImageAlign(nn.Module):
    """把图片投影映到文本隐藏空间；子层名沿用 HF 多模态投影器的 linear_1 / linear_2。"""

    def __init__(self, size: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(size, size)  # 第一层在 d_model 内学习图片特征组合
        self.act = nn.GELU()                   # 平滑非线性让两层不退化成单次线性投影
        self.linear_2 = nn.Linear(size, size)  # 输出仍是 d_model，可直接交给 Transformer

    def forward(self, image_hidden: torch.Tensor) -> torch.Tensor:
        """依次经过两层线性与中间 GELU，保持 (批, 块数, d_model)。"""
        return self.linear_2(self.act(self.linear_1(image_hidden)))  # 图片块逐个映到文本隐藏空间


def align_loss(image_hidden: torch.Tensor, text_hidden: torch.Tensor, block_mask: torch.Tensor = None) -> torch.Tensor:
    """返回同批图文表示的 1 - 平均余弦；补位块和整条无图样本都不进入平均。"""
    if block_mask is None:                                               # 旧调用没有补位概念，全部块照常取均值
        image_mean = image_hidden.float().mean(dim=1)                    # 每张图的全部块合成一个 fp32 表示
        samples = torch.ones(image_hidden.shape[0], device=image_hidden.device)  # 批内每条都有有效图片
    else:                                                               # 批次补齐后只汇总 mask 为 1 的真实块
        valid = block_mask.to(dtype=torch.float32).unsqueeze(-1)         # 变成 (批, 块, 1)，逐隐藏维广播
        counts = valid.sum(dim=1)                                       # 每条样本自己的真实块数
        image_mean = (image_hidden.float() * valid).sum(dim=1) / counts.clamp_min(1)  # 补位值完全不进图片均值
        samples = (counts.squeeze(-1) > 0).float()                       # 整条没有图片时也不制造一份虚假对齐损失
    text_mean = text_hidden.float().mean(dim=1)                          # 同批每段文本也合成一个表示
    similarity = F.cosine_similarity(image_mean, text_mean, dim=-1)      # 批内按样本一一配对，不跨样本混合
    losses = (1 - similarity) * samples                                 # 只留下至少有一个真实块的样本
    return losses.sum() / samples.sum().clamp_min(1)                     # 全批无图时返回可反传的标量零


class Model(nn.Module):
    """整摞模型：主体那摞加上输出层。"""

    # ---------- 建完整模型 ----------

    def __init__(self, geo: Geometry, precision: str = "fp32", align_weight: float = 0.0, align_queue: int = 1024, align_temperature: float = 0.07) -> None:       # 默认仍只返回文本打分
        super().__init__()                                                                           # 注册模型模块
        self.geo = geo                                                                               # 几何留在身上，存盘和报数都要用
        self.dtype = geo.dtype(precision)                                                            # 权重类型
        self.pad_id = 0  # Trainer 按实际词表覆盖；独立检索也可显式提供 text_mask
        self.align_weight = align_weight                                                             # 正数时把加权图文损失随 logits 一起交回
        self.grad_checkpoint = False                                                                # 训练器可开启前向重算，节省显存
        self.quantized = False                                                                      # 记下量化层是否已替换
        self.model = Core(geo, self.dtype)                                                           # 主体
        self.lm_head = None if geo.tie else nn.Linear(geo.d_model, geo.vocab, bias=False)                # 不共用就单独一层
        for name, one in self.named_parameters():                                                     # 残差输出按层数缩小，深堆时激活不炸
            if name.endswith("o_proj.weight") or name.endswith("down_proj.weight"):                    # 注意力与前馈各自的出口
                nn.init.normal_(one, std=0.02 / math.sqrt(2 * geo.layers))                             # 保持已有文本权重的初始化顺序
        self.image_proj = nn.Linear(geo.patch ** 2 * 3, geo.d_model, bias=False) if geo.image_side else None  # RGB 块投影；最后才建，不扰动文本初始化
        self.vision = VisionTower(geo.patch, geo.d_model) if geo.image_side and geo.vision == "conv" else None  # 真实卷积塔仅在接图时创建
        self.contrast = ImageContrast(geo.d_model, align_queue, align_temperature) if geo.image_side else None  # 有界跨批字典
        self.image_align = ImageAlign(geo.d_model) if geo.image_side else None                         # 两层可训映射把图片送进文本隐藏空间
        self.to(self.dtype)                                                                          # 图片层也一起落到目标精度

    # ---------- 前向打分 ----------

    def forward(
        self,
        ids: torch.Tensor,
        patches: torch.Tensor = None,
        block_mask: torch.Tensor = None,
        patch_at: torch.Tensor = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:                              # 下游显式传入图片块及其布局
        """返回 logits；align_weight 为正时再返回一个加权标量对齐损失。"""
        image_hidden = None                                                                          # 无图前向不增加前缀，也没有图文损失
        if patches is not None:                                                      # 图片编码只有这一条入口，检索与生成共用
            image_hidden = self.encode_image(patches, block_mask)                    # 两层卷积、像素旁路、图文投影与补位遮罩
        if self.grad_checkpoint and self.training:                                                    # 重算时也把图片布局和遮罩带上
            x = checkpoint(self.model, ids, image_hidden, block_mask, patch_at, use_reentrant=False)  # 保留图片投影与对齐层的梯度链
        else:                                                                                        # 常规一次过
            x = self.model(ids, image_hidden, block_mask, patch_at)                                  # 图片和文字按同一布局走完整摞
        if self.lm_head is None:                                                                      # 共用词表就直接拿词表权重做输出
            logits = x @ self.model.embed_tokens.weight.T                                            # 图片与文字位置都返回词表打分
        else:                                                                                        # 不共用词表时走独立输出层
            logits = self.lm_head(x)                                                                 # 同样保留全部图片与文本位置
        if self.align_weight <= 0:                                                                   # 默认接口仍只交回 logits
            return logits                                                                            # 旧调用方不需要拆元组
        if image_hidden is None:                                                                     # 这一批没有图片就没有可对齐的表示
            return logits, x.float().sum() * 0.0                                                     # 仍交回可反传的标量零，返回结构不变
        pictures, texts, valid = self.embeddings(ids, patches, block_mask, image_hidden=image_hidden)
        keys = torch.tensor([int.from_bytes(hashlib.blake2b(row[row != self.pad_id].numpy().tobytes(), digest_size=8).digest(), "little") & ((1 << 63) - 1) for row in ids.detach().cpu()], device=ids.device)  # 文本指纹仅用于排除重复负样本
        alignment = self.contrast(pictures[valid], texts[valid], keys[valid]) * self.align_weight
        return logits, alignment                                                     # 语言损失由训练器统一相加

    def encode_image(self, patches: torch.Tensor, block_mask: torch.Tensor = None) -> torch.Tensor:
        """图像生成与检索共用视觉编码；无效块在卷积前和投影后各清一次。"""
        pixels = patches.flatten(2)                                                  # 支持原来的五维和预展平输入
        if block_mask is not None:
            pixels = pixels * block_mask.unsqueeze(-1).to(pixels.dtype)               # 伪像素不能影响卷积或溢出
        hidden = self.image_proj(pixels)                                             # 保留像素旁路与旧权重名称
        if self.vision is not None:
            hidden = hidden + self.vision(pixels)                                    # 真卷积特征参与实际前向与反向
        hidden = self.image_align(hidden)                                            # 映射到语言隐藏空间
        if block_mask is not None:
            hidden = hidden * block_mask.unsqueeze(-1).to(hidden.dtype)               # 消掉投影偏置产生的伪块
        return hidden                                                                # 块顺序、数量与输入一致

    def embeddings(self, ids: torch.Tensor, patches: torch.Tensor, block_mask: torch.Tensor = None, text_mask: torch.Tensor = None, image_hidden: torch.Tensor = None) -> tuple:
        """返回归一化图、文向量与有图标记；文本单独编码，不能偷看待检索图片。"""
        hidden = self.encode_image(patches, block_mask) if image_hidden is None else image_hidden
        mask = torch.ones(hidden.shape[:2], device=hidden.device) if block_mask is None else block_mask.float()
        counts = mask.sum(dim=1, keepdim=True)                                        # 无图样本不会进入对比学习
        pictures = (hidden.float() * mask.unsqueeze(-1)).sum(dim=1) / counts.clamp_min(1)
        words = self.model(ids)                                                      # 独立文本前向，不使用图片前缀
        text_mask = (ids != self.pad_id) if text_mask is None else text_mask                    # 调用方可显式覆盖 pad=0 的默认约定
        texts = (words.float() * text_mask.unsqueeze(-1)).sum(dim=1) / text_mask.sum(dim=1, keepdim=True).clamp_min(1)
        return F.normalize(pictures, dim=-1), F.normalize(texts, dim=-1), counts[:, 0] > 0

    @torch.no_grad()
    def retrieval(self, ids: torch.Tensor, patches: torch.Tensor, block_mask: torch.Tensor = None, text_mask: torch.Tensor = None) -> dict:
        """评估成对图文检索 Recall@1/5；输入第 i 图和第 i 文必须是真实配对。"""
        training = self.training                                                     # 调用后恢复用户原来的模型状态
        self.eval()                                                                  # 不改变历史负样本队列
        pictures, texts, valid = self.embeddings(ids, patches, block_mask, text_mask)
        pictures, texts = pictures[valid], texts[valid]                               # 排除整条无图样本
        scores = pictures @ texts.T                                                  # 一次生成完整候选打分矩阵
        expected = torch.arange(len(scores), device=scores.device)                    # 对角线是标注正例
        result = {"pairs": len(scores)}                                              # 报清分母，避免空集伪装成满分
        for name, matrix in (("image_to_text", scores), ("text_to_image", scores.T)):
            for count in (1, 5):
                hit = (matrix.topk(min(count, len(scores)), dim=1).indices == expected[:, None]).any(dim=1)
                result[f"{name}_r{count}"] = float(hit.float().mean()) if len(scores) else 0.0
        self.train(training)                                                         # 检索不改变后续训练行为
        return result                                                                # 可直接写入 JSON 报告

    def count(self) -> int:
        """参数量，只含真正占参数的张量。"""
        return sum(one.numel() for one in self.parameters())  # 逐个累加

    def save(self, folder: Path, write: bool = True) -> dict:
        """把权重写成 folder/model.safetensors。共用的输出层不单独存。

        write 为假时只把整份张量拼出来不落盘：fsdp2 下拼分片是跨进程的，每个进程都得走一遍，
        但同一份文件只让 0 号写。返回拼好的那一份，调用方要用还能用。
        """
        state = {name.replace(".base.weight", ".weight"): whole(one) for name, one in self.state_dict().items()}  # 全拷到 CPU；挂了低秩时把 base 那一层去掉
        if self.geo.tie:  # 共用时 lm_head 就是词表的副本
            state.pop("lm_head.weight", None)  # 删掉副本，载入时会补回来
        if write:  # 只有 0 号真写文件
            folder.mkdir(parents=True, exist_ok=True)  # 目录没建就先建
            save_file(state, str(folder / "model.safetensors"))  # 整份写出去
        return state

    def load(self, source: Path) -> None:
        """从文件或目录里读同名张量灌回来。名字缺一个就报错，不静默跳过。"""
        configuration = (source if source.is_dir() else source.parent) / "config.json"  # 推理读取实际词表补位定义
        if configuration.is_file():
            self.pad_id = json.loads(configuration.read_text(encoding="utf-8")).get("pad_token_id", self.pad_id)
        files = sorted(source.glob("*.safetensors")) if source.is_dir() else [source]  # 目录就全收
        state = {}  # 攒起来的张量
        for one in files:  # 一个文件一个文件读
            state.update(load_file(str(one), device="cpu"))  # 并进同一份
        has_tower = any(name.startswith("vision.") for name in state)                # 存档本身决定有没有真实卷积参数
        if self.geo.image_side and self.vision is not None and not has_tower:
            self.vision = None                                                      # 旧视觉存档保留线性路径，不捏造未训练卷积
            self.geo.vision = "linear"                                              # 后续配置明确标出实际加载的结构
            logs.warn("model", "载入旧视觉存档，使用 linear 视觉路径")
        if self.geo.image_side and self.vision is None and has_tower:
            fail("input", "这是 conv 视觉存档，请使用 geometry.vision: conv", where=str(source))
        own = self.state_dict()  # 自己的整套名字
        with torch.no_grad():  # 参数是叶子又要梯度，原地灌得关掉记录
            for name, tensor in own.items():  # 按自己的名字逐个找
                found = state.get(name)  # 文件里有没有这一份
                if found is None:  # 确实缺
                    fail("input", f"权重里没有 {name}", where=str(source), hint="几何要和存盘那份一致")
                tensor.copy_(found.to(tensor.dtype))  # 灌回去

    def quantize(self, precision: str, device: torch.device = None) -> bool:
        """把所有权重线层换成 bitsandbytes 的量化层，返回真压了没有。

        device 不给就按当前那张卡；测试里在无卡机器上灌 cpu 也能走完这条路径。
        压完把自己身上所有参数冻住：量化层没有梯度，其余参数在 QLoRA 里也不该动。
        """
        if precision == "fp8":                                                      # FP8 是计算精度，主权重保持 BF16 可训练
            from features.precision import enable                                  # 不依赖 bitsandbytes
            count = enable(self)                                                   # CPU/不支持的 GPU 明确拒绝，不静默降级
            logs.done("spin", "FP8 线性前向已启用", layers=count, master="bf16", backward="bf16")
            return False                                                           # 不走 QLoRA 冻结语义，全参数仍可训练
        if precision not in QUANT:  # fp32 / bf16 / fp16 / fp8 不走这条路，本来就已经是真类型
            return False
        if device is None and not torch.cuda.is_available():  # 没卡就没地方放量化层
            logs.warn("spin", "量化要卡，按 fp32 跑", precision=precision)
            return False
        device = device or torch.device("cuda", torch.cuda.current_device())  # 默认落到当前那张卡
        import bitsandbytes as bnb  # 真走量化才导，没装也不影响 fp32 训练
        for name, one in list(self.named_modules()):  # 名字连着父模块，才知道换掉的是谁的哪个属性
            if not isinstance(one, nn.Linear):  # 只换权重线层，归一和词表不动
                continue
            parent = self.get_submodule(name.rpartition(".")[0]) if "." in name else self  # 父模块；lm_head 没有点就挂在模型上
            setattr(parent, name.rpartition(".")[2], _quant(one, precision, bnb, device))  # 就地换掉那一层
        self.to(device)  # 词表、归一这些没量的参数也搬上卡
        for one in self.parameters():  # 量化层自己没梯度，其余也一律冻上
            one.requires_grad_(False)
        self.quantized = True
        return True

    # ---------- 描述存档结构 ----------

    def config(self, precision: str) -> dict:                      # 配置跟权重一起交给下游
        """给导出和排查用的结构描述，字段名照 HF 的 config.json。"""
        return {                                                 # 直接读取模型正在使用的几何
            "model_type": "hoshiori",                             # 结构名
            "hidden_size": self.geo.d_model,                      # 隐藏维
            "num_hidden_layers": self.geo.layers,                 # 层数
            "num_attention_heads": self.geo.heads,                # 头数
            "head_dim": self.geo.d_head,                          # 头宽
            "vocab_size": self.geo.vocab,                         # 词表
            "max_position_embeddings": self.model.rope_cos.shape[2],  # 文本预算加图片块的总位置数
            "rope_theta": self.geo.rope_base,                     # RoPE 底数
            "tie_word_embeddings": self.geo.tie,                  # 输出层是否共用词表
            "pad_token_id": self.pad_id,                       # 检索池化不能把词表补位算进文本表示
            "vision": self.geo.vision,                         # 旧线性与新卷积权重可以明确重建
            "text_max_seq_len": self.geo.max_seq_len,            # 区别文字预算与含图总位置预算
            "image_side": self.geo.image_side,                   # 原图边长，0 表示不接图
            "patch": self.geo.patch,                             # 图片块边长，决定投影输入宽度
            "torch_dtype": precision,                            # 权重精度
        }                                                        # 图片结构也随配置落盘

    def write_config(self, folder: Path, precision: str) -> None:
        """把结构描述写到 folder/config.json。"""
        folder.mkdir(parents=True, exist_ok=True)  # 目录先建好
        (folder / "config.json").write_text(json.dumps(self.config(precision), ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":  # 手搓一下看维度对不对，不用装别的
    geo = Geometry(d_model=64, layers=2, heads=2, d_head=32, vocab=64, max_seq_len=16)  # 演示尺寸
    net = Model(geo)  # 搭一个
    out = net(torch.randint(0, geo.vocab, (2, 8)))  # 两批八步
    print(out.shape, round(math.log(net.count()), 3))  # 应该是 (2, 8, 64)
