"""FP8 训练：线性层前向使用 E4M3 张量与 CUDA scaled_mm，主权重和梯度保留 BF16。

调用：enable(model) 在载入权重、搬到 CUDA 后替换线性层；随后照常建立优化器。
示例：net = Model(geo, "fp8").cuda(); enable(net); loss = net(ids).float().square().mean()
loss.backward() 使用直通梯度的 BF16 矩阵乘法，避免把 E4M3 有限范围当作主权重存储。
这不是所有算子都是 FP8，也不声称减半优化器显存；RMSNorm、注意力与词表保留 BF16。
实现依照 PyTorch 2.8 aten CUDA Blas.cpp 的 _scaled_mm：行主序左矩阵、列主序右矩阵、K/N 对齐 16。
"""

import torch  # FP8 dtype、CUDA scaled_mm 与自动微分
from torch import nn  # 兼容现有 nn.Linear 参数名
from torch.nn import functional as F  # 奇数尺寸按底层算子要求补边

from utils.fail import fail  # 不支持的硬件应明确失败，不能冒充 FP8


def _pack(tensor: torch.Tensor) -> tuple:
    """按本次张量动态缩放到 E4M3，返回量化张量和反量化比例。"""
    maximum = torch.finfo(torch.float8_e4m3fn).max  # E4M3 的有限最大值由实际 dtype 提供
    scale = tensor.detach().float().abs().amax().clamp_min(1e-12) / maximum
    packed = (tensor.float() / scale).clamp(-maximum, maximum).to(torch.float8_e4m3fn)
    return packed, scale.float()  # scaled_mm 需要 float32 的反量化比例


class Product(torch.autograd.Function):
    """FP8 前向与浮点直通反向；额外补位不改变外部形状。"""

    @staticmethod
    def forward(context, inputs, weight):
        context.save_for_backward(inputs, weight)  # 反向只使用原始浮点主权重和激活
        flat = inputs.reshape(-1, inputs.shape[-1])  # 批次和 token 合并成矩阵行
        left = F.pad(flat, (0, -flat.shape[1] % 16, 0, -flat.shape[0] % 16))
        right = F.pad(weight, (0, -weight.shape[1] % 16, 0, -weight.shape[0] % 16))
        left, left_scale = _pack(left)  # 激活独立动态缩放
        right, right_scale = _pack(right)  # 权重独立动态缩放，避免两者量级互相压制
        output = torch._scaled_mm(left, right.T, scale_a=left_scale, scale_b=right_scale, out_dtype=torch.bfloat16)
        output = output[:flat.shape[0], :weight.shape[0]]  # 丢掉硬件对齐产生的补位
        return output.reshape(*inputs.shape[:-1], weight.shape[0]).to(inputs.dtype)

    @staticmethod
    def backward(context, gradient):
        inputs, weight = context.saved_tensors  # 对量化算子使用直通估计
        flat_gradient = gradient.reshape(-1, weight.shape[0]).to(weight.dtype)
        flat_inputs = inputs.reshape(-1, inputs.shape[-1]).to(weight.dtype)
        input_gradient = (flat_gradient @ weight).reshape_as(inputs)  # 对输入的标准线性梯度
        weight_gradient = flat_gradient.T @ flat_inputs  # BF16 主权重可继续优化
        return input_gradient.to(inputs.dtype), weight_gradient.to(weight.dtype)


class Linear(nn.Linear):
    """保留 Linear 的 weight/bias 名称，让旧模型存档与 LoRA 仍可识别。"""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """使用真实 CUDA FP8 乘法；偏置仍按输出类型相加。"""
        output = Product.apply(inputs, self.weight)  # 自定义自动微分保留训练通路
        return output if self.bias is None else output + self.bias.to(output.dtype)


def enable(model: nn.Module) -> int:
    """替换 CUDA 模型的全部浮点线性层，返回真正启用 FP8 的层数。"""
    device = next(model.parameters()).device  # 已载入模型的真实设备决定是否可用
    if device.type != "cuda":
        fail("missing", "FP8 训练需要支持 E4M3 scaled_mm 的 NVIDIA GPU", where="train.precision")
    capability = torch.cuda.get_device_capability(device)
    if capability < (8, 9):
        fail("input", f"sm_{capability[0]}{capability[1]} 不支持这条 FP8 路径", where="train.precision", hint="使用 bf16/fp16/nf4/int8")
    count = 0  # 日志与验收需要知道真正替换了几层
    for name, layer in list(model.named_modules()):
        if not isinstance(layer, nn.Linear):
            continue  # 词表、归一和卷积保留浮点类型
        parent = model.get_submodule(name.rpartition(".")[0]) if "." in name else model
        replacement = Linear(layer.in_features, layer.out_features, bias=layer.bias is not None, device=device, dtype=layer.weight.dtype)
        replacement.weight = layer.weight  # 保留主权重，不重新初始化训练值
        replacement.bias = layer.bias  # 偏置也保持同一参数对象
        setattr(parent, name.rpartition(".")[2], replacement)
        count += 1
    return count  # 返回值为零意味着没有实际 FP8 线性计算
