"""成本与复用卡：真实云价快照、训练短跑校准、严格按几何匹配的估算。

调用示例：
  price = prices("NVIDIA GeForce RTX 5090", cloud="secure")
  profile = calibrate(recipe, batches=[1, 2, 4], steps=10, warmup=3)
  quote = estimate(plan, profile=profile, configuration=configuration(recipe), price=price)

不提供价钱或实测吞吐就返回未知；不会补上占位单价/吞吐。原 recommend(vram_gb,
params_b) 仍提供明确标为“未校准”的经验提示；产品入口优先使用校准 profile。
"""

import copy                         # 各个实测 batch 从同一原始配方独立派生
import hashlib                      # 校准配置的稳定指纹
import json                         # profile 与价钱快照统一用 JSON
import math                         # 外部数值不能是 NaN 或无穷
import statistics                   # 单步计时汇总
import tempfile                     # 合成校准数据只存在临时目录
import time                         # CUDA 同步后的墙钟计时
import urllib.request               # 公共云价 API，不需要账户密钥
from datetime import datetime, timezone  # 价钱和实测都保留 UTC 时间
from pathlib import Path            # 用户明确指定的文件位置

NF4_GB_PER_B = 4.5                  # 仅旧显式 recommend 的未校准量级提示
BF16_GB_PER_B = 28.0                # 仅旧显式 recommend 的未校准量级提示
PRICE_URL = "https://api.runpod.io/graphql"


def estimate(plan, price_per_hour=None, tokens_per_step=1_000, *, tokens_per_second=None,
             profile=None, configuration=None, price=None) -> dict:
    """估算运行成本，旧位置参数保留；缺实测吞吐或单价时对应数值为 None。

    tokens_per_second 是调用者显式提供的单卡有效 token 吞吐；profile 使用同配置的
    实测更新步耗时，不能把小模型或另一张卡的数字套用过来。price 是 prices 的快照。
    """
    if tokens_per_step < 1:
        raise ValueError("tokens_per_step 必须为正数")
    cards = max(1, len(plan.devices))         # 费用按实际占用卡数，不做多卡线性加速假设
    seconds = None                          # 没有证据时未知，不回填固定 1000 token/s
    reason = "缺少实测吞吐或匹配的校准记录"
    run = matching_run(profile, configuration) if profile and configuration else None
    if run:
        seconds = run["seconds_per_step"]    # 同几何、精度、批量、累积、策略下的一步墙钟时间
        reason = "使用完全匹配配置的实测更新步耗时"
    elif tokens_per_second is not None:
        if not math.isfinite(tokens_per_second) or tokens_per_second <= 0:
            raise ValueError("实测 tokens_per_second 必须为有限正数")
        seconds = tokens_per_step / tokens_per_second
        reason = "使用调用者显式提供的单卡吞吐；多卡效率须由调用者实测"
    currency = "CNY"                        # 旧显式 price_per_hour 沿用人民币接口
    if price is not None:
        if price.get("unit") != "GPU-hour":
            raise ValueError("价格单位必须是 GPU-hour")
        currency = price["currency"]        # 不进行没有汇率依据的币种换算
        price_per_hour = price["price_per_hour"]
    if price_per_hour is not None and (not math.isfinite(price_per_hour) or price_per_hour < 0):
        raise ValueError("price_per_hour 必须为有限非负数")
    matched_hardware = not (profile and price) or profile["hardware"]["name"] == price["gpu"]
    hours = plan.steps * seconds / 3600 if seconds is not None and plan.steps > 0 else None
    total = hours * cards * price_per_hour if hours is not None and price_per_hour is not None and matched_hardware else None
    if not matched_hardware:
        reason += "；云价 GPU 与实测 GPU 不同，费用未知"
    return {"steps": plan.steps, "tokens_per_step": tokens_per_step, "seconds_per_step": seconds,
            "hours": hours, "cards": cards, "price_per_hour": price_per_hour, "total_price": total,
            "currency": currency, "reason": reason, "price_source": price, "hardware_match": matched_hardware,
            "measurement": {"hardware": profile["hardware"], "measured_at": profile.get("measured_at"),
                            "scope": profile.get("scope")} if run else None}


def render(quote: dict) -> str:
    """用明确的未知值展示成本，不输出看似精确的占位金额。"""
    timing = "未知（缺少匹配实测）" if quote["hours"] is None else f"≈{quote['hours']:.3f} 小时（{quote['seconds_per_step']:.3f} 秒/步）"
    amount = "未知（缺少价格、吞吐或硬件不匹配）" if quote["total_price"] is None else f"≈{quote['currency']} {quote['total_price']:.2f}"
    lines = ["成本估算", f"  工作量    {quote['steps']} 步 × {quote['tokens_per_step']} token/步",
             f"  时长      {timing}", f"  费用      {amount}", f"  依据      {quote['reason']}"]
    if quote.get("price_source"):
        price = quote["price_source"]             # 价格与硬件、来源、抓取时刻一起呈现
        lines.append(f"  云价      {price['gpu']} · {quote['currency']} {quote['price_per_hour']}/GPU-hour")
        lines.append(f"  价源      {price.get('source', '调用者提供')} · {price.get('fetched_at', '时间未提供')}")
    if quote.get("measurement"):
        evidence = quote["measurement"]           # 计时适用范围不能藏在 JSON 内
        lines.append(f"  实测      {evidence['hardware']['name']} · {evidence['measured_at']} · {evidence['scope']}")
    return "\n".join(lines)


def prices(gpu: str, cloud="secure", timeout=30) -> dict:
    """抓取 Runpod 指定 GPU 的按需 GPU-hour 公开价格；不代表实时库存或最终账单。"""
    if cloud not in {"secure", "community"} or not gpu.strip():
        raise ValueError("需要精确 GPU id 和 secure/community 云类型")
    query = "{ gpuTypes { id displayName memoryInGb securePrice communityPrice } }"
    request = urllib.request.Request(PRICE_URL, data=json.dumps({"query": query}).encode(),
                                     headers={"Content-Type": "application/json", "User-Agent": "hoshiori/0.1"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)           # 网络/API 错误交给命令入口，不能伪装成 0 元
    if payload.get("errors"):
        raise ValueError("云价 API 返回错误: " + json.dumps(payload["errors"], ensure_ascii=False))
    found = next((row for row in payload["data"]["gpuTypes"] if row["id"] == gpu), None)
    if found is None:
        raise ValueError("云价目录没有这个精确 GPU id: " + gpu)
    value = found.get(cloud + "Price")           # 0/null 代表没有可用报价，不作为免费实例
    if value is None or not math.isfinite(value) or value <= 0:
        raise ValueError("这个 GPU/云类型没有有效按需报价")
    return {"version": 1, "provider": "runpod", "gpu": gpu, "cloud": cloud, "currency": "USD",
            "unit": "GPU-hour", "price_per_hour": value, "fetched_at": datetime.now(timezone.utc).isoformat(),
            "source": PRICE_URL, "query": query, "raw_entry": found,
            "scope": "按需计算目录价；不含存储、税费、启动和空闲时间，库存未验证"}


def configuration(recipe: dict) -> dict:
    """提取影响训练吞吐/容量的完整配置，不把只有参数量标签相同当成同一个模型。"""
    from features.model import Geometry          # 延迟导入：只抓价格不要求 torch
    train = recipe.get("train") or {}
    low = train.get("lora")
    if low:
        low = {"r": low.get("r", 16), "alpha": low.get("alpha", 32), "dropout": low.get("dropout", 0.0),
               "target": low.get("target") or ["w_q", "w_k", "w_v", "w_o"], "variant": low.get("variant", "lora")}
    return {"geometry": Geometry.from_recipe(recipe).as_dict(), "task": recipe.get("task", "sft"),
            "precision": train.get("precision", "fp32"), "batch_size": train.get("batch_size", 1),
            "grad_accum": train.get("grad_accum", 1), "strategy": train.get("strategy", "single"),
            "cards": max(1, len(train.get("devices") or [])), "lora": low,
            "grad_checkpoint": bool(train.get("grad_checkpoint", False)), "optimizer": "AdamW",
            "declared_params_b": (recipe.get("model") or {}).get("params_b")}


def matching_run(profile: dict, config: dict):
    """只接受成功且配置完全相同的真实 CUDA 校准，不插值、不放大到更大模型。"""
    if profile.get("version") != 1 or profile.get("kind") != "cuda-training-calibration":
        raise ValueError("不是受支持的 CUDA 训练校准 profile")
    for row in profile["runs"]:                   # profile 是外部文件，拒绝伪装为测量值的 NaN/负数
        if row.get("status") != "ok":
            continue                             # OOM 记录没有性能数字，不参与估价
        if any(not math.isfinite(row[field]) or row[field] <= 0 for field in ("seconds_per_step", "tokens_per_second", "peak_reserved_gib")):
            raise ValueError("校准计时、吞吐与显存必须为有限正数")
    return next((run for run in profile["runs"] if run.get("status") == "ok" and run["configuration"] == config), None)


def calibrated_recommend(profile: dict, config: dict, vram_gb=None, gpu_name=None, headroom=0.15) -> list:
    """只推荐同 GPU、同模型配置中实际成功的 batch，预留可配置的显存余量。"""
    if not math.isfinite(headroom) or headroom < 0:
        raise ValueError("headroom 必须是有限非负数")
    matching_run(profile, config)               # 检查外部 profile 类型；不同配置允许没有精确行
    hardware = profile["hardware"]
    if gpu_name is not None and gpu_name != hardware["name"]:
        return []                              # 5090 D 的证据不能迁移到 3090 或另一型号
    capacity = hardware["free_gib_before"] if vram_gb is None else min(vram_gb, hardware["total_gib"])
    cards = []
    for run in profile["runs"]:
        candidate = dict(run["configuration"])
        candidate["batch_size"] = config["batch_size"]  # 只允许 batch 不同，其他维度逐项相同
        if run.get("status") != "ok" or candidate != config:
            continue
        if run["peak_reserved_gib"] * (1 + headroom) > capacity:
            continue                           # 成功但贴边的配置不建议直接复用
        cards.append({"strategy": config["strategy"], "precision": config["precision"],
                      "batch_size": run["configuration"]["batch_size"], "tokens_per_second": run["tokens_per_second"],
                      "why": f"{hardware['name']} 实测，峰值预留 {run['peak_reserved_gib']:.3f} GiB；仅适用本 profile 的几何与训练配置"})
    return sorted(cards, key=lambda row: row["tokens_per_second"], reverse=True)


def calibrate(recipe: dict, batches=(1, 2, 4), steps=10, warmup=3, device=0) -> dict:
    """在一张真实 CUDA 卡上短跑文本 SFT/pretrain，量每个 batch 的吞吐和峰值。

    使用真实 Trainer._update 的前向、反向、裁剪与 AdamW 更新；合成满长文本绕过
    下载语料和 tokenizer，结果明确标 synthetic_full_length，不能当端到端数据管线吞吐。
    """
    import gc                                  # 每个配置之间释放上一个模型和优化器
    import torch                               # 校准才加载训练依赖
    if not torch.cuda.is_available():
        raise ValueError("校准需要真实 CUDA GPU，CPU 结果不能冒充显卡证据")
    if steps < 2 or warmup < 1 or not batches or any(batch < 1 for batch in batches):
        raise ValueError("至少 1 步预热、2 步测量，batch 列表必须为正数")
    config = configuration(recipe)             # 归一化后检查适用范围，避免没测的路径也标成功
    if config["task"] not in {"sft", "pretrain"} or config["geometry"]["image_side"]:
        raise ValueError("当前校准只支持文本 SFT/pretrain，不适用于视觉或对齐任务")
    if config["strategy"] != "single" or config["cards"] != 1:
        raise ValueError("当前校准只支持真实单卡，不能推算多卡通信效率")
    if config["precision"] == "fp8":
        raise ValueError("FP8 必须由实际 FP8 训练后端独立校准，不能按其他 dtype 代跑")
    if config["precision"] in {"nf4", "int8"} and not config["lora"]:
        raise ValueError("量化训练校准必须配置 train.lora")
    torch.cuda.set_device(device)               # 明确使用哪一张真实卡
    torch.cuda.empty_cache()                    # 清掉当前进程以前留下的空闲池，不清其他进程
    free, total = torch.cuda.mem_get_info(device)
    info = torch.cuda.get_device_properties(device)
    profile = {"version": 1, "kind": "cuda-training-calibration", "scope": "synthetic_full_length_training",
               "measured_at": datetime.now(timezone.utc).isoformat(), "hardware": {"name": info.name,
               "capability": list(torch.cuda.get_device_capability(device)), "total_gib": total / 2**30,
               "free_gib_before": free / 2**30, "torch": torch.__version__, "cuda": torch.version.cuda}, "runs": []}
    for batch in batches:
        current = copy.deepcopy(recipe)        # batch 之外全部保持配方原样
        current.setdefault("train", {})["batch_size"] = batch
        try:
            result = _measure(current, steps, warmup, device)
        except torch.cuda.OutOfMemoryError:     # OOM 是校准的一个有效结果，不能伪造可用配置
            result = {"status": "oom", "configuration": configuration(current)}
        profile["runs"].append(result)
        gc.collect()                           # 先释放张量引用，再让空闲缓存交还设备
        torch.cuda.empty_cache()
    return profile


def _measure(recipe, steps, warmup, device):
    """一个配置完成预热、同步计时和显存计量；返回后模型随函数作用域释放。"""
    import torch                               # 内部训练动作使用已确认可用的 CUDA
    from features.model import Geometry, Model
    from features.trainer import Choice, Trainer
    config = configuration(recipe)             # profile 保存全部实际配置而不是简化参数量标签
    train = recipe.get("train") or {}
    geometry = Geometry.from_recipe(recipe)
    torch.manual_seed(42)                      # 各批量的初始化相同，便于重测比较
    net = Model(geometry, config["precision"]).to(torch.device("cuda", device))
    count = net.count()                        # 用真正建出的模型参数数目检查标签
    declared = config["declared_params_b"]
    if declared is not None and not 0.8 * declared * 1e9 <= count <= 1.2 * declared * 1e9:
        raise ValueError(f"params_b 标签与实际几何不符：实际 {count} 参数；不能用小模型校准大模型")
    net.quantize(config["precision"])          # 与 spin 一样在建优化器前转换真实量化权重
    with tempfile.TemporaryDirectory(prefix="ori-calibrate-") as folder:
        data = Path(folder) / "full-length.jsonl"
        row = json.dumps({"text": "\u0001" * geometry.max_seq_len})  # token 1 避开 pad=0
        data.write_text(row + "\n", encoding="utf-8")
        choice = Choice(out=Path(folder) / "run", files=[data], steps=steps + warmup,
                        batch_size=config["batch_size"], grad_accum=config["grad_accum"],
                        precision=config["precision"], lora=config["lora"], task=config["task"],
                        grad_checkpoint=config["grad_checkpoint"], lr=train.get("lr", 2e-4),
                        weight_decay=train.get("weight_decay", 0.0), clip=train.get("max_grad_norm", 1.0))
        runner = Trainer(net, geometry, choice)  # 真实训练器管理 LoRA、取批、反向与优化器
        for step in range(warmup):
            runner.step = step + 1
            runner._update()                   # 预热分配优化器状态，避免首步初始化污染均值
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        durations, tokens, losses = [], 0, []
        for step in range(steps):
            runner.step = warmup + step + 1
            started = time.perf_counter()
            loss = runner._update()
            torch.cuda.synchronize(device)     # 所有 kernel 完成后读钟，不只量 Python 发射速度
            durations.append(time.perf_counter() - started)
            tokens += runner.last_tokens       # 真实有效标签数，不把 pad token 算入吞吐
            losses.append(loss)
        if not all(math.isfinite(loss) for loss in losses):
            raise ValueError("校准训练出现非有限损失，不能作为复用证据")
        return {"status": "ok", "configuration": config, "parameters": count, "steps": steps,
                "warmup_steps": warmup, "seconds_per_step": statistics.mean(durations),
                "step_seconds": durations, "tokens": tokens, "tokens_per_second": tokens / sum(durations),
                "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
                "last_loss": losses[-1], "configuration_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()}


def recommend(vram_gb: float, params_b: float) -> list[dict]:
    """旧显式接口：返回三条未校准经验提示；可执行推荐请用 calibrated_recommend。"""
    quantized_need = params_b * NF4_GB_PER_B                               # 先看量化 LoRA 能否在单卡留出空间
    full_need = params_b * BF16_GB_PER_B                                   # 再看 BF16 全量训练是否有现实余量

    if vram_gb < 12 or vram_gb < quantized_need:                           # 小卡或连 NF4 量级都偏紧时优先保命
        return [
            {
                "strategy": "single",                                    # 单卡没有额外通信与副本
                "precision": "nf4",                                      # 四位权重最省显存
                "batch_size": 1,                                          # 最小微批先确保能开跑
                "why": "未校准经验提示：显存相对模型偏紧，先用 NF4 单卡小批量守住峰值。",
            },
            {
                "strategy": "fsdp2",                                     # 有多卡时把模型状态摊开
                "precision": "nf4",                                      # 仍保留最省显存的权重形式
                "batch_size": 1,                                          # 不用批大小吃掉分片省下的空间
                "why": "未校准经验提示：若能借到多卡，FSDP2 分片比复制整套权重更稳。",
            },
            {
                "strategy": "single",                                    # 仍给一条不依赖多卡的替代
                "precision": "int8",                                     # 八位精度用少量空间换更稳的数值
                "batch_size": 1,                                          # 小显存档不主动放大微批
                "why": "未校准经验提示：只在 NF4 数值不稳且实测有余量时退到 INT8，批量仍保持 1。",
            },
        ]

    if vram_gb < full_need:                                                # 量化能放下、全量 BF16 仍偏大的过渡档
        return [
            {
                "strategy": "single",                                    # 单卡路径最容易直接复用
                "precision": "nf4",                                      # 量化给批量留下余量
                "batch_size": 2,                                          # 过渡档可以试探性提高吞吐
                "why": "未校准经验提示：量化后已有余量，batch 2 通常比全量 BF16 更均衡。",
            },
            {
                "strategy": "single",                                    # 单卡不引入分布式复杂度
                "precision": "int8",                                     # 比 NF4 多保留一些数值精度
                "batch_size": 1,                                          # 精度换空间后把微批收回 1
                "why": "未校准经验提示：需要更稳数值时可用 INT8，但用 batch 1 控制显存峰值。",
            },
            {
                "strategy": "fsdp2",                                     # 多卡分片承接全量训练
                "precision": "bf16",                                     # 分片后尝试完整 BF16 权重
                "batch_size": 1,                                          # 先用最小批验证真实峰值
                "why": "未校准经验提示：单卡装不下全量状态时，用 FSDP2 分摊后再尝试 BF16。",
            },
        ]

    return [                                                               # 显存覆盖全量 BF16 量级，开始偏向吞吐
        {
            "strategy": "single",                                        # 一张大卡先走最短路径
            "precision": "bf16",                                         # 无需量化即可保留训练精度
            "batch_size": 4,                                              # 用较大微批吃掉充裕空间
            "why": "未校准经验提示：单卡显存覆盖 BF16 全量状态，可直接用 batch 4 提高利用率。",
        },
        {
            "strategy": "ddp",                                           # 多卡各存一份换并行吞吐
            "precision": "bf16",                                         # 每卡余量足够保留完整权重
            "batch_size": 4,                                              # 每卡批量不必为了副本缩小
            "why": "未校准经验提示：每卡都放得下完整模型时，DDP 适合用额外卡换训练速度。",
        },
        {
            "strategy": "fsdp2",                                         # 大模型扩展时仍可分片
            "precision": "bf16",                                         # 不牺牲权重精度
            "batch_size": 8,                                              # 分片余量优先转成更大微批
            "why": "未校准经验提示：继续放大模型或序列时，FSDP2 能把余量转成更大的 batch。",
        },
    ]
