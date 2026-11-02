# 训练深度与验收

本页对应缺口清单的 10、12、13、17、19、20、29。代码入口仍是 `commands/spin.py → features/trainer.py`；几何、视觉前向在 `features/model.py`，挂件在 `features/lora.py`，离线合并在 `features/merge.py`。

## 异构教师（10）

`model.teacher` 指向教师权重目录或带 `latest` 的运行目录。教师有 `config.json` 时用自己的层数、隐藏维、头数和上下文预算构建，独立放在 CPU 并冻结。没有结构配置的旧教师仍按学生几何载入。两边需要相同词表及 token 编号；不同词表不能直接做 KL。带图蒸馏还要求图片边长和切块布局一致。标签为 `-100` 的图片位、补位不进入蒸馏损失。

`tests/test_training_depth.py::test_heterogeneous_teacher_runs_and_stays_frozen` 实际用 3 层 32 维教师训练 1 层 16 维学生，验证学生权重变化、教师权重不变。这里的异构指星织模型的不同几何，不宣称支持任意外部架构或不同 tokenizer 的跨词表蒸馏。

## 视觉塔、跨批负例与检索（19）

新图片几何默认 `vision: conv`。每块 RGB 依次过两个 3×3 卷积、GELU、空间平均池化与隐藏维投影，再与原像素投影旁路相加，经图文投影送入文本 Transformer。卷积提取块内空间特征，块间上下文由后续 Transformer 建模。`vision: linear` 用于读取旧视觉结构；加载没有卷积张量的旧视觉存档时会明确记录这一路径，不使用随机卷积冒充旧模型能力。续训在建立优化器之前恢复旧视觉结构。

`train.align_weight > 0` 开启对称 InfoNCE。`align_queue`（默认 1024 对）存放此前批次的脱离梯度的图文表示，最旧对会淘汰；`align_temperature`（默认 0.07）控制 softmax 温度。重复文本的历史对不作为负例。纯文本/无图条目不进入对比目标，图片补位不参与池化。评估不会更新队列，训练断点将队列与优化器一起保存。分布式各 rank 使用本地队列，未声称跨 rank 负例全聚合。

文本检索表示通过独立的纯文本前向获得，不能看到待匹配图片。模型可直接调用：

```python
pictures, texts, valid = net.embeddings(ids, patches, block_mask, text_mask)
report = net.retrieval(ids, patches, block_mask, text_mask)
# report: pairs、image_to_text_r1/r5、text_to_image_r1/r5
```

`ids[i]` 与 `patches[i]` 必须是真配对；多个合理答案时需要调用方提供相应评价集，本接口按一一配对计算。`text_mask` 不给时按模型的 `pad_token_id` 排除补位。

复跑功能学习实验：

```bash
OMP_NUM_THREADS=1 python tests/benchmark_vision.py --out docs/evidence/vision-cpu.json
```

已运行的 [报告](evidence/vision-cpu.json)：六种颜色，batch 2、历史队列 12、120 步，测试图使用独立像素噪声。图到文 R@1 从 0.3333 到 1.0，文到图从 0.1667 到 1.0，对比 loss 从 0.6485 到 0.00637。模型保存在 `docs/evidence/vision-cpu-model/`。这是受控颜色配对的真实学习实验，不是自然图文检索基准或预训练视觉模型质量声明。

## 可配置健康检查（17）

旧 `train.health: stop|warn` 保持不变，默认阈值也保持。新增配置：

```yaml
train:
  health: stop
  health_limits:
    nan_steps: 3
    norm_steps: 5
    norm_limit: 1000.0
    speed_window: 20
    slow_ratio: 0.4
    eval_drop: 0.1
    eval_patience: 2
eval:
  files: [validation.jsonl]
  every: 50
```

验证 JSONL 使用 `text`，相对路径以配方目录为准。训练前先测基线，每隔 `every` 步通过真实模型前向计算非补位 token 准确率。低于历史最佳准确率的绝对差超过 `eval_drop`，连续 `eval_patience` 次才触发。非有限验证分数计失败，但不能污染最佳基线。验证不消费训练游标；最佳分和连续次数随断点恢复。这里的监控指标是 teacher-forced token 准确率，不把它称作通用任务评测分数。

## 难度课程（20）

`curriculum: true` 默认仍按长度分三档。设置 `curriculum_by: difficulty` 后，训练 JSONL 的 `difficulty`（0 到 1，越大越难）优先；未标注条目使用当前训练集的平均 token 稀有度，按全库规模归一化。这个自动代理独立于长度，不宣称等于人类感知难度。课程分位与排序由固定数据、种子和轮次重建，完整保留既有无放回取样与续训重放逻辑。

## DoRA 与 LoHa（29）

```yaml
train:
  lora:
    variant: dora # lora / dora / loha
    r: 8
    alpha: 16
    dropout: 0.0
    target: [w_q, w_v]
```

标准 LoRA 是 `W + (alpha/r) B@A`；LoHa 是 `W + (alpha/r) (B@A) ⊙ (D@C)`；DoRA 把低秩更新后的每个输出通道方向归一，再乘可训幅值。初始增量为零，DoRA 初始幅值取原权重行范数。DoRA 归一范数在反向时脱离梯度。对应依据是 [DoRA 论文](https://arxiv.org/abs/2402.09353) 与 [低秩 Hadamard 参数化论文](https://arxiv.org/abs/2108.06098)。

三种变体都能训练、保存、恢复、在模型内合并和经 `merge.Adapter` 离线合并。离线合并在 FP32 中进行，BF16 safetensors 复用导出模块的原始位解码。DoRA/LoHa 当前支持浮点基座；NF4/INT8 继续使用标准 LoRA，挂件只分配自己的浮点参数，不对整数存储的量化基座做整体 dtype 转换。

## FP8 与硬件矩阵（12）

`precision: fp8` 使用 `features/precision.py`：线性前向的激活与权重动态缩放至 E4M3，通过真正的 `aten::_scaled_mm` 计算。主权重、优化器与直通反向仍是 BF16；注意力、词表和归一仍是浮点。因此这不是全算子 FP8，也不能按每参数一字节估算主权重显存。布局约束参考 [PyTorch 2.8 CUDA BLAS 实现](https://github.com/pytorch/pytorch/blob/v2.8.0/aten/src/ATen/native/cuda/Blas.cpp)。CPU 和计算能力低于 8.9 的 GPU 明确拒绝此路径。

```bash
OMP_NUM_THREADS=1 python tests/benchmark_precision.py --precision fp8 --out /tmp/fp8.json
```

在 RTX 5090 D、sm_120、PyTorch 2.8.0+cu128 上已实际训练三步并恢复到第五步。FP8 验收必须同时观察到已替换的 FP8 线性层和精确名为 `aten::_scaled_mm` 的调用；`bitsandbytes::int8_scaled_mm` 单独记录，不算作 FP8 证据。

- [FP8 实测](evidence/fp8-5090d.json)：14 个 FP8 线性层；42 次 FP8 矩阵乘法；五步 loss 有限。
- [NF4 实测](evidence/nf4-5090d.json)：标准 LoRA 挂载、三步训练、恢复同一冻结基座后续到第五步。
- [INT8 实测](evidence/int8-5090d.json)：同样完成真实挂载与续训。

**未完成的硬件项：sm_86 与 sm_89 实机测试。** 当前证据仅证明上列 sm_120 环境。sm_86 不支持这里的 FP8 路径，但仍需要在该硬件验证其他精度及明确拒绝 FP8 的行为。

## 真两卡 NCCL 续训（13）

已修复模型分配前的设备选择：torchrun 子进程先按 `LOCAL_RANK` 设置 CUDA 设备，再建立/搬运模型；DDP 使用当前设备，不再把全局 rank 当本机卡号。取消超额 rank 对单卡取模，不让单卡重复映射冒充实际多卡。

准备好的严格验收命令：

```bash
OMP_NUM_THREADS=1 python tests/benchmark_multigpu.py --out /tmp/ori-nccl-report
```

脚本要求两张不同物理 GPU 的 UUID，运行完整五步、保留五步学习率计划但在三步存档、从该档再续两步。两个 rank 必须报告 NCCL，最终逐权重与连续训练比较，最大绝对差不超过 1e-6 才生成 `passed` 报告。

**当前未完成真两卡验收。** 已连接远端只有一张 5090 D，单机 CPU 分布式回归不能替代这条要求。

## 回归入口

```bash
OMP_NUM_THREADS=1 python -m pytest tests/test_training_depth.py tests/test_image.py tests/test_image_train.py tests/test_trainer.py tests/test_alignment.py tests/test_forge.py -q
```

新增深度用例覆盖真实更新、完整存读、BF16 导出、旧视觉存档、补位遮罩、跨批队列淘汰、非有限验证分恢复和难度分桶。测试只证明对应行为；上述未具备的真实硬件仍列为未完成。
