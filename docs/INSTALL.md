# 装与验收

每个后端怎么装、怎么验。CI 上跑的就是这里的命令。

## 基础

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e .            # 只要 weave / validate / import / doctor
ori doctor                  # 缺什么这里会说
```

`ori --help` 在任何状态下都该在 300ms 内返回，它不 import torch、不联网。

## 依赖分层

| extras | 装什么 | 给谁 |
| --- | --- | --- |
| 无 | 标准库 | weave、validate、import、doctor 全部可用 |
| `speed` | numpy | 去重提速，缺了自动退回纯 Python |
| `dedup` | numpy、blake3 | 指纹换 blake3 |
| `full` | + torch、safetensors | 训练、合并、导出 |

```bash
pip install -e ".[full]"
```

## 后端矩阵

每格都要一份 CI 报告，跑的能力面固定为：一次 40 条的 weave、一次 validate、一次 spin dry-run、一次 200 步的 smoke 训练（只有权重不存）。

| 后端 | 前置 | 验收命令 | 状态 |
| --- | --- | --- | --- |
| CPU | 无 | `ori doctor && pytest -q` | 已验证 |
| CUDA 12.8 / sm_89 | 驱动 570+ | `ori doctor` 列到卡 → smoke | 待补报告 |
| CUDA 12.8 / sm_90 | 驱动 570+ | 同上，精度换 fp8 | 待补报告 |
| ROCm 6.2 | rocm-smi | `ori doctor` 认到 rocm-smi → smoke | 待补报告 |
| MPS | macOS 14+ | `ori doctor` → smoke，精度 fp32 | 待补报告 |
| TPU | torch_xla | `ori doctor` → smoke | 待补报告 |

```bash
# smoke 训练的固定配方，只验证链路通不通
ori spin examples/sft-lora.warp.yaml --dry-run
```

## 验收怎么算过

1. `pytest -q` 全绿，没有 skip 掉的后端用例
2. `ori doctor` 在目标后端上没 ✗
3. weave 出的卡片六字段齐、指纹能重算一致
4. 同一份配方 + 同一份数据指纹，两次跑的 loss 曲线偏差 <1%

## 常见卡点

- `numpy` 是可选件：判重在无 numpy 时走纯 Python，慢一个档但不报错，先用着。
- `safetensors` 缺失时只有导出会拦，训练和 weave 都不受影响。
- 端口被占：`ori serve` 与 `ori loom` 会直接报占用并退出 4，换 `--port`。
- Windows 上跑 `ori` 用 `python main.py` 打头，短横线的参数照写。
