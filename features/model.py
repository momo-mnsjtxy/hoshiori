"""模型：一个从零搭的小型 decoder-only 因果 Transformer，几何从配方的 model.geometry 读。
每层是 RMSNorm → 带 RoPE 的因果注意力 → 残差 → RMSNorm → 前馈 → 残差，末尾再归一、算词表打分。

调用方式：Model(Geometry.from_recipe(recipe)) 建模；load 灌权重，save 存 model.safetensors，config 描述几何。
image_side=0 不建图片参数，旧文本存档照常载入；非零时按原图边长预留 ceil(image_side/patch)² 个位置。
max_seq_len 仍是文本长度预算，图片前缀额外占位置；实际图片尺寸和同批块数由下游统一，本层不缩放图片。
forward(ids, patches=None) 接收 (批, 文本长)；可选 patches 是归一后的 RGB，形状 (批, 块数, patch, patch, 3)，
也可预先展平成 (批, 块数, patch*patch*3)，张量的设备、精度与模型一致。返回 (批, 块数+文本长, 词表)，
下游取 logits[:, 块数:] 后沿用文本标签和损失；图片与文字共用 RoPE，图片从 0 开始，文字接着编号。
量化顺序仍是 load → quantize → 建优化器；无卡时保持 fp32，压过后可训的只剩 LoRA 的 A/B。
文本权重沿用 HF 名字，tie 为真时只存一份词表；图片投影以 image_proj.weight 随完整权重保存、载入。

调用示例：
  from pathlib import Path                                                    # 存档位置
  import torch                                                                # 造一批模型输入
  from features.model import Geometry, Model                                   # 几何与模型
  geo = Geometry(image_side=32, patch=16)                                      # 四块图片前缀
  model = Model(geo)                                                          # 按几何搭架子，权重随机起
  ids = torch.tensor([[1, 2]])                                                # 两个文本位置
  patches = torch.zeros(1, 4, 16, 16, 3)                                      # 已归一的 RGB 图片块
  logits = model(ids, patches=patches)                                        # 形状 (1, 6, 512)
  text_logits = logits[:, patches.shape[1]:]                                  # 两个文本位置的打分
  model.save(Path("out/1"))                                                   # 存成 model.safetensors
  model.load(Path("out/1"))                                                   # 几何一致就能完整灌回
"""

import json
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
DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16, "nf4": torch.float32, "int8": torch.float32}
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
    max_seq_len: int = 128         # 最长文本序列，图片前缀额外预留位置
    rope_base: float = 10000.0     # RoPE 的底数
    tie: bool = True              # 输出层是否复用词表权重
    image_side: int = 0           # 原图边长，0 表示不接图
    patch: int = 16               # 图片块边长，接图时用正整数，右侧和底部可补齐

    @classmethod
    def from_recipe(cls, recipe: dict) -> "Geometry":
        """从配方的 model.geometry 节取几何，没写的就是默认值。"""
        given = (recipe.get("model") or {}).get("geometry") or {}  # 整节没写就当空
        geo = cls(**given)  # 白名单在 recipe.check 里已经过了一遍，这里照收
        if geo.heads * geo.d_head != geo.d_model:  # 头宽乘头数必须正好铺满隐藏维
            fail("schema", "heads × d_head 对不上 d_model", where="model.geometry.d_head", hint=f"{geo.heads} × {geo.d_head} 要等于 {geo.d_model}")
        return geo

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

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """前向：投影 → 分头 → 转位置 → 因果注意力 → 合头 → 投影回。"""
        batch, steps, _ = x.shape  # 批、序列、隐藏维
        split = lambda part: part.view(batch, steps, self.heads, self.d_head).transpose(1, 2)  # 拆成 (批, 头, 序列, 头宽)
        query, key, value = split(self.q_proj(x)), split(self.k_proj(x)), split(self.v_proj(x))  # 三路投影后分头
        query, key = rotate(query, cos, sin), rotate(key, cos, sin)  # 只转查询和键，值不动
        out = F.scaled_dot_product_attention(query, key, value, is_causal=True)  # 因果遮罩里的缩放点积
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

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """残差两跳，平铺直叙。"""
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)  # 注意力这一跳
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

    # ---------- 图片前缀与文本一起过各层 ----------

    def forward(self, ids: torch.Tensor, prefix: torch.Tensor = None) -> torch.Tensor:    # prefix 已投影到隐藏维
        """查词表 → 拼可选图片前缀 → 共用 RoPE 逐层计算 → 收尾归一，保留全部位置。"""
        x = self.embed_tokens(ids)                                                      # 文本仍从同一张词表取嵌入
        if prefix is not None:                                                          # 有图片才增加前缀位置
            x = torch.cat((prefix, x), dim=1)                                            # 图片在前，所有文本都能看见图片
        steps = x.shape[1]                                                              # 实际总长度包括图片块
        cos, sin = self.rope_cos[:, :, :steps], self.rope_sin[:, :, :steps]                # 从 0 连续编号，文字接在图片后
        for block in self.layers:                                                       # 一层一层往下走
            x = block(x, cos, sin)                                                      # 同一因果注意力处理全部位置
        return self.norm(x)                                                             # 收尾归一


class Model(nn.Module):
    """整摞模型：主体那摞加上输出层。"""

    # ---------- 建完整模型 ----------

    def __init__(self, geo: Geometry, precision: str = "fp32") -> None:                                  # 默认仍是纯文本模型
        super().__init__()                                                                           # 注册模型模块
        self.geo = geo                                                                               # 几何留在身上，存盘和报数都要用
        self.dtype = geo.dtype(precision)                                                            # 权重类型
        self.grad_checkpoint = False                                                                # 训练器可开启前向重算，节省显存
        self.quantized = False                                                                      # 记下量化层是否已替换
        self.model = Core(geo, self.dtype)                                                           # 主体
        self.lm_head = None if geo.tie else nn.Linear(geo.d_model, geo.vocab, bias=False)                # 不共用就单独一层
        for name, one in self.named_parameters():                                                     # 残差输出按层数缩小，深堆时激活不炸
            if name.endswith("o_proj.weight") or name.endswith("down_proj.weight"):                    # 注意力与前馈各自的出口
                nn.init.normal_(one, std=0.02 / math.sqrt(2 * geo.layers))                             # 保持已有文本权重的初始化顺序
        self.image_proj = nn.Linear(geo.patch ** 2 * 3, geo.d_model, bias=False) if geo.image_side else None  # RGB 块投影；最后才建，不扰动文本初始化
        self.to(self.dtype)                                                                          # 图片层也一起落到目标精度

    # ---------- 前向打分 ----------

    def forward(self, ids: torch.Tensor, patches: torch.Tensor = None) -> torch.Tensor:                # 下游显式传入图片块
        """返回 (批, 图片块数+文本长, 词表)；patches 为同设备、同精度的归一 RGB 块，可预先展平。"""
        prefix = None if patches is None else self.image_proj(patches.flatten(2))                     # 每块展平后投到 d_model
        if self.grad_checkpoint and self.training:                                                    # 重算时也把图片前缀带上
            x = checkpoint(self.model, ids, prefix, use_reentrant=False)                              # 保留图片投影的梯度链
        else:                                                                                        # 常规一次过
            x = self.model(ids, prefix)                                                              # 图片和文字走完整摞
        if self.lm_head is None:                                                                      # 共用词表就直接拿词表权重做输出
            return x @ self.model.embed_tokens.weight.T                                              # 图片与文字位置都返回词表打分
        return self.lm_head(x)                                                                       # 独立输出层

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
        files = sorted(source.glob("*.safetensors")) if source.is_dir() else [source]  # 目录就全收
        state = {}  # 攒起来的张量
        for one in files:  # 一个文件一个文件读
            state.update(load_file(str(one), device="cpu"))  # 并进同一份
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
            "max_position_embeddings": self.model.rope_cos.shape[2],  # 文本预算加图片前缀的总位置数
            "rope_theta": self.geo.rope_base,                     # RoPE 底数
            "tie_word_embeddings": self.geo.tie,                  # 输出层是否共用词表
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
