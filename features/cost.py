"""训练成本与复用卡：把训练计划换成费用量级，并按显存给出可抄的配方。

调用示例：
  from features import cost
  quote = cost.estimate(plan, tokens_per_step=2048)
  print(cost.render(quote))
  cards = cost.recommend(vram_gb=24, params_b=7)

时长只是开跑前的量级估算：这里用固定 token 吞吐作锚点，真实账单应以同机型短跑实测替换。
"""

DEFAULT_PRICE_PER_HOUR = 2.5  # 占位单卡价（元/时），接入真实云厂商价格后应替换
REFERENCE_TOKENS_PER_SECOND = 1_000.0  # 粗略吞吐锚点；真实速度会随模型、卡型和序列长度变化
NF4_GB_PER_B = 4.5  # 量化 LoRA 每十亿参数的大致显存量级，含训练留量
BF16_GB_PER_B = 28.0  # BF16 全量训练每十亿参数的大致显存量级，含梯度与优化器态


def estimate(plan, price_per_hour: float = DEFAULT_PRICE_PER_HOUR, tokens_per_step: int = 1_000) -> dict:
    """按计划步数和每步 token 数估算运行小时数与总价。

    ``price_per_hour`` 是单卡每小时价格；多卡计划会按卡数累加费用。
    """
    cards = max(1, len(plan.devices))                                      # 没写设备仍按一张卡估价
    seconds_per_step = tokens_per_step / REFERENCE_TOKENS_PER_SECOND       # 把每步工作量换成粗略耗时
    hours = plan.steps * seconds_per_step / 3_600                          # 总步数乘每步耗时，再由秒换成小时
    total_price = hours * cards * price_per_hour                           # 每张卡都按同一小时单价计费
    return {                                                               # 留下表格需要的原始数，调用方也能继续计算
        "steps": plan.steps,                                               # 本次明确给出的总更新步数
        "tokens_per_step": tokens_per_step,                               # 单卡每个更新步处理的 token 上限
        "seconds_per_step": seconds_per_step,                             # 吞吐锚点下的一步耗时
        "hours": hours,                                                   # 预计墙钟小时数
        "cards": cards,                                                   # 同时占用的卡数
        "price_per_hour": price_per_hour,                                 # 本次采用的单卡时价
        "total_price": total_price,                                       # 全部卡合计的预计费用
    }


def render(quote: dict) -> str:
    """把 ``estimate`` 的结果排成几行人读的成本表。"""
    return "\n".join(                                                       # 保持纯文本，命令行可以直接打印
        [
            "成本估算",                                                     # 独立标题便于从整张训练账里扫到
            f"  工作量    {quote['steps']} 步 × {quote['tokens_per_step']} token/步",
            f"  时长      ≈{quote['hours']:.3f} 小时（{quote['seconds_per_step']:.2f} 秒/步）",
            f"  费用      ≈¥{quote['total_price']:.2f}（{quote['cards']} 卡 × ¥{quote['price_per_hour']:.2f}/时）",
        ]
    )


def recommend(vram_gb: float, params_b: float) -> list[dict]:
    """按单卡显存与模型参数量返回三条可复用配方，并说明选择原因。"""
    quantized_need = params_b * NF4_GB_PER_B                               # 先看量化 LoRA 能否在单卡留出空间
    full_need = params_b * BF16_GB_PER_B                                   # 再看 BF16 全量训练是否有现实余量

    if vram_gb < 12 or vram_gb < quantized_need:                           # 小卡或连 NF4 量级都偏紧时优先保命
        return [
            {
                "strategy": "single",                                    # 单卡没有额外通信与副本
                "precision": "nf4",                                      # 四位权重最省显存
                "batch_size": 1,                                          # 最小微批先确保能开跑
                "why": "显存相对模型偏紧，先用 NF4 单卡小批量守住峰值。",
            },
            {
                "strategy": "fsdp2",                                     # 有多卡时把模型状态摊开
                "precision": "nf4",                                      # 仍保留最省显存的权重形式
                "batch_size": 1,                                          # 不用批大小吃掉分片省下的空间
                "why": "若能借到多卡，FSDP2 分片比复制整套权重更稳。",
            },
            {
                "strategy": "single",                                    # 仍给一条不依赖多卡的替代
                "precision": "int8",                                     # 八位精度用少量空间换更稳的数值
                "batch_size": 1,                                          # 小显存档不主动放大微批
                "why": "只在 NF4 数值不稳且实测有余量时退到 INT8，批量仍保持 1。",
            },
        ]

    if vram_gb < full_need:                                                # 量化能放下、全量 BF16 仍偏大的过渡档
        return [
            {
                "strategy": "single",                                    # 单卡路径最容易直接复用
                "precision": "nf4",                                      # 量化给批量留下余量
                "batch_size": 2,                                          # 过渡档可以试探性提高吞吐
                "why": "量化后已有余量，batch 2 通常比全量 BF16 更均衡。",
            },
            {
                "strategy": "single",                                    # 单卡不引入分布式复杂度
                "precision": "int8",                                     # 比 NF4 多保留一些数值精度
                "batch_size": 1,                                          # 精度换空间后把微批收回 1
                "why": "需要更稳数值时可用 INT8，但用 batch 1 控制显存峰值。",
            },
            {
                "strategy": "fsdp2",                                     # 多卡分片承接全量训练
                "precision": "bf16",                                     # 分片后尝试完整 BF16 权重
                "batch_size": 1,                                          # 先用最小批验证真实峰值
                "why": "单卡装不下全量状态时，用 FSDP2 分摊后再尝试 BF16。",
            },
        ]

    return [                                                               # 显存覆盖全量 BF16 量级，开始偏向吞吐
        {
            "strategy": "single",                                        # 一张大卡先走最短路径
            "precision": "bf16",                                         # 无需量化即可保留训练精度
            "batch_size": 4,                                              # 用较大微批吃掉充裕空间
            "why": "单卡显存覆盖 BF16 全量状态，可直接用 batch 4 提高利用率。",
        },
        {
            "strategy": "ddp",                                           # 多卡各存一份换并行吞吐
            "precision": "bf16",                                         # 每卡余量足够保留完整权重
            "batch_size": 4,                                              # 每卡批量不必为了副本缩小
            "why": "每卡都放得下完整模型时，DDP 适合用额外卡换训练速度。",
        },
        {
            "strategy": "fsdp2",                                         # 大模型扩展时仍可分片
            "precision": "bf16",                                         # 不牺牲权重精度
            "batch_size": 8,                                              # 分片余量优先转成更大微批
            "why": "继续放大模型或序列时，FSDP2 能把余量转成更大的 batch。",
        },
    ]
