# 成本与显卡复用校准

训练账不再默认填写 2.5 元/卡时和 1000 token/s。缺价格、缺实测或配置不匹配时，对应时长/费用显示“未知”。原 `estimate(plan, price_per_hour, tokens_per_step)` 位置参数仍接受；要得到数值，还须显式给 `tokens_per_second` 或匹配的校准 profile。

## 取得公开价格

```bash
ori cost prices --gpu 'NVIDIA GeForce RTX 5090' --cloud secure --out price.json
```

实现通过 [Runpod 公共 GraphQL](https://graphql-spec.runpod.io/) 的 `gpuTypes` 抓取按需计算目录价。快照保存精确 GPU id、secure/community 类型、USD、GPU-hour、来源 URL、抓取 UTC 时间和原始返回项。报价为零或为空时拒绝生成价格，不视作免费资源。请求不创建云实例，也不会消费账户额度。

这是计算目录价，非库存保证；最终账单还会受运行时长、存储与税费等影响。[Runpod 计费说明](https://docs.runpod.io/pods/pricing) 是计费范围依据。快照不会自动刷新，重新运行命令即可获取当前价。

## 在真实 GPU 上校准

```bash
ori cost calibrate examples/calibration.warp.yaml --batches 1,2,4 --warmup 3 --steps 10 --out profile.json
ori cost estimate examples/calibration.warp.yaml --profile profile.json --price price.json
```

校准使用本项目真正的 `Trainer._update`：取批、前向、反向、梯度裁剪与 AdamW 更新都执行。每次更新后 `torch.cuda.synchronize` 才停止计时；预热不计入均值，显存计量包含已经分配的优化器状态。当前校准范围为单卡文本 SFT/pretrain；对齐、视觉、多卡与 FP8 不会用别的路径代跑。

这是满长合成文本、按配方几何随机初始化的训练计算短跑，不加载已有模型权重，也不计下载、分词、数据读取、评估、保存和启动时间。结果标记为 `synthetic_full_length_training`。随机初始化不会证明训练后质量，短跑波动也不能当长期吞吐保证。

profile 保存完整几何、精度、batch、梯度累积、LoRA 配置、检查点开关、任务、策略、GPU 名称、计算能力、总显存、开始时可用显存、Torch/CUDA 版本、逐步秒数、有效 token 数、PyTorch allocated/reserved 峰值与最终有限损失。显存数是本进程 PyTorch 分配器计数，未把 CUDA 上下文与其他进程占用混成模型占用。

推荐只在所有配置一致时比较已经成功跑过的 batch，并默认预留 15% 显存余量。不会从小模型推到 7B、从短序列推到长序列、从单卡推到多卡。`params_b` 标签若与实际构建的参数量明显不符，校准会报错。显式传入不同 GPU 名称时不返回复用推荐。

## 使用证据

配方可包含下面两个路径，相对路径以配方所在目录解析：

```yaml
cost:
  profile: profile.json
  price: price.json
```

`ori spin --dry-run` 和 `ori cost estimate` 都会展示来源与匹配结果。费用使用实测单步墙钟时间乘总更新步数，再乘卡数与单卡时价；不会偷偷换币种。配置相同但云价 GPU 与实测 GPU 不同，时长仍可显示，费用保持未知。

旧 `cost.recommend(vram_gb, params_b)` 仍保留三档经验提示，全部标为“未校准”；训练账不把这些经验提示当成真机复用推荐。

## 本次真实运行

2026-11-02 在 RTX 5090 D（31.36 GiB、sm_120、Torch 2.8.0+cu128）完成 3 个 batch 的短跑。模型实际 **4,327,680 参数**，4 层、256 维、词表 512、序列上限 128、BF16、AdamW、无 LoRA；每档预热 3 步、测量 10 步。

| batch | 平均秒/更新步 | 有效 token/s | PyTorch reserved 峰值 GiB |
| --- | ---: | ---: | ---: |
| 1 | 0.01834 | 6,924 | 0.06250 |
| 2 | 0.02729 | 9,307 | 0.06445 |
| 4 | 0.03013 | 16,859 | 0.08398 |

原始证据是 [cost-5090d-profile.json](evidence/cost-5090d-profile.json)，可用 [calibration.warp.yaml](../examples/calibration.warp.yaml) 复核配置匹配。此表只证明这个小模型与三个 batch，不能代表 7B 模型的容量或吞吐。

同日真实调用价格入口，Runpod Secure RTX 5090 返回 **USD 1.19/GPU-hour**，完整来源及 UTC 时间见 [runpod-5090-price.json](evidence/runpod-5090-price.json)。该价钱对应 RTX 5090，实测是 RTX 5090 D，因此 [cost-matched-estimate.json](evidence/cost-matched-estimate.json) 中费用保持 `null`。这是预期的跨卡保护，不能把两者凑成一笔“实测云成本”。

验证：`python -m pytest tests/test_cost.py tests/test_cost_evidence.py`，涵盖无证据未知、单价与吞吐线性计算、币种、配置不匹配、跨卡拒用、容量余量、相对路径和公共 API 响应边界。
