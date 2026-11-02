"""清洗：给正文脱敏、估质量，并把真实与合成样本按固定比例混在一起。

调用示例：
  from features import clean
  text, hits = clean.scrub("联系 fox@example.com，令牌 sk-abc123abc123abc123abc123abc123")
  quality = clean.score("这是一段完整、没有机械重复的正文。")
  rows = clean.blend(real_rows, fake_rows, 0.1)  # 最终批次约一成是合成样本

RULES 是唯一的脱敏规则表；加删规则只改这一处。blend 不做有放回抽样，避免少量合成
样本被复制许多遍；同样的输入始终得到同样的结果，数据指纹才可复现。
"""

import random                                               # 固定种子抽样，让混配可以复现
import re                                                   # 脱敏规则和空白 n-gram 都靠正则认


OCTET = r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"             # IPv4 的每段只能落在 0~255
RULES = [
    (
        "邮箱",
        re.compile(
            r"(?<![\w.+-])[\w.!#$%&'*+/=?^`{|}~-]+@"
            r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?"
            r"(?:\.[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?)+(?![\w.-])",
            re.IGNORECASE,
        ),
    ),
    ("手机号", re.compile(r"(?<!\d)(?:\+?86[- ]?)?1[3-9]\d(?:[- ]?\d{4}){2}(?!\d)")),
    ("身份证号", re.compile(r"(?<!\d)(?:\d{17}[\dXx]|\d{15})(?!\d)")),
    (
        "密钥",
        re.compile(
            r"(?<![A-Za-z0-9_+=-])"
            r"[A-Za-z0-9_+=-]{32,}(?![A-Za-z0-9_+=-])"
        ),
    ),
    ("IP", re.compile(rf"(?<![\d.]){OCTET}(?:\.{OCTET}){{3}}(?![\d.])")),
]

NGRAM = 4                                                   # 四字片段能抓机械循环，又不太误伤常用短语
IDEAL_MIN = 80                                              # 太短的正文通常缺少可学习的上下文
IDEAL_MAX = 4_000                                           # 这个区间内长度本身不扣分
USEFUL_MAX = 16_000                                         # 再长常是拼接稿，长度分线性降到零
LINE_ENDS = "。！？!?；;：:…」』】）)]}."                 # 正文、对话和列表常见的收尾符号
WEIGHTS = {"length": 0.4, "repeat": 0.4, "punctuation": 0.2}


def scrub(text: str) -> tuple[str, int]:
    """按 RULES 依次替换敏感片段，返回（脱敏正文，命中总数）。"""
    hits = 0                                                 # 所有规则的替换次数合成一个审计数字
    for name, pattern in RULES:                              # 顺序固定，重叠内容只由先命中的规则处理
        text, count = pattern.subn(f"«{name}»", text)        # 占位符保留原处的语义类别
        hits += count                                        # 一处替换算一次命中
    return text, hits                                        # 正文和命中数一起交回样本入口


def score(text: str) -> float:
    """按长度、重复四字片段和非空行收尾标点给正文打 0~1 分。"""
    body = text.strip()                                      # 首尾空白不应替正文贡献长度
    if not body:                                             # 空正文三项都没有有效信息
        return 0.0                                           # 明确给最低分，避免短串分支抬高它

    size = len(body)                                         # 长度按 Unicode 字符计，中文不吃亏
    if size < IDEAL_MIN:                                     # 短文本从零线性长到合适区间
        length = size / IDEAL_MIN                            # 越接近八十字，上下文越完整
    elif size <= IDEAL_MAX:                                  # 常规训练片段的长度不扣分
        length = 1.0                                         # 合适区间拿满长度分
    else:                                                    # 超长正文可能是多篇材料误拼在一起
        length = max(0.0, 1 - (size - IDEAL_MAX) / (USEFUL_MAX - IDEAL_MAX))

    compact = re.sub(r"\s+", "", body)                       # 换行与空格变化不该躲过重复检查
    grams = [compact[i : i + NGRAM] for i in range(max(1, len(compact) - NGRAM + 1))]
    repeat = len(set(grams)) / len(grams)                    # 唯一片段越多，机械循环越少

    lines = [line.rstrip() for line in body.splitlines() if line.strip()]
    ended = sum(line.endswith(tuple(LINE_ENDS)) for line in lines)
    punctuation = ended / len(lines)                         # 每条非空行都拥有同等权重

    total = (                                                # 三项都已归一化，直接按重要性合成
        length * WEIGHTS["length"]
        + repeat * WEIGHTS["repeat"]
        + punctuation * WEIGHTS["punctuation"]
    )
    return round(min(1.0, max(0.0, total)), 4)               # 浮点误差也不能越出公开边界


def blend(real: list, fake: list, ratio: float) -> list:
    """从两份列表无放回抽样；ratio 是返回结果中 fake 的目标占比。"""
    if not real:                                             # 没有真实样本时，留下可用的合成样本
        return list(fake)                                    # 返回副本，调用方改列表不会碰到输入
    if not fake or ratio <= 0:                               # 没有合成样本或明确要求零占比
        return list(real)                                    # 真实样本原样进入后续流程

    total = min(len(real), int(len(fake) / ratio))           # fake 不足就缩小整批，不复制凑数
    fake_count = min(len(fake), int(total * ratio + 0.5))    # 四舍五入到最接近的样本数
    real_count = total - fake_count                          # 剩余席位全部留给真实样本
    picker = random.Random(0)                                # 固定选择结果，保证数据指纹可复现
    rows = picker.sample(real, real_count)                   # 真实池也抽样，避免总取文件开头
    rows.extend(picker.sample(fake, fake_count))             # 合成池按目标占比补齐
    picker.shuffle(rows)                                     # 两类样本交错，别把合成数据堆在末尾
    return rows                                              # 交回一份全新的混合列表
