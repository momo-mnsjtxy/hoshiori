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
    ("姓名", re.compile(r"(?P<label>(?:姓名|联系人|收件人|收货人)[：:]\s*)(?P<value>[\u3400-\u9fff·]{2,8}|[A-Za-z][A-Za-z .\'-]{1,60})(?=$|[，,；;。\n\r\t])")),
    ("地址", re.compile(r"(?P<label>(?:家庭住址|收货地址|收件地址|联系地址|地址|住址)[：:]\s*)(?P<value>[^\n\r；;，,]{4,160})")),
    ("银行卡", re.compile(r"(?<![\d-])(?:[3-6]\d{12,18}|[3-6]\d{3}(?:[ -]\d{4}){2,3}(?:[ -]\d{1,3})?)(?![\d-])")),
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
        def replace(match):                                  # 有标签的字段保留字段名，方便核对用途
            nonlocal hits                                    # 只统计真正被替换的敏感值
            if name == "银行卡" and not bank_valid(match[0]): # 无标签长数字先过校验，减少误伤编号
                return match[0]                              # 无效校验位留给身份证等后续规则判断
            hits += 1                                        # 一次命中只记一笔
            label = match.groupdict().get("label", "")        # 姓名、地址保留冒号前的业务字段
            return label + f"«{name}»"                       # 不把原始敏感值写到审计记录
        text = pattern.sub(replace, text)                    # 重叠内容由先匹配的类别处理
    return text, hits                                        # 正文和命中数一起交回样本入口


def score(text: str, weights: dict = None) -> float:
    """按长度、重复四字片段和非空行收尾标点给正文打 0~1 分。"""
    parts = measures(text)                                   # 特征提取与标注校准使用同一个入口
    weights = WEIGHTS if weights is None else weights         # 没有人工档案时明确仍用旧启发式
    total = sum(parts[name] * weights[name] for name in WEIGHTS)  # 权重在档案边界检查，内部直接计算
    return round(min(1.0, max(0.0, total)), 4)                 # 对外分值保持四位小数


def bank_valid(text: str) -> bool:
    """判断 13–19 位卡号是否通过 Luhn；它只检查格式，不能确认账户存在。"""
    digits = [int(one) for one in text if one.isdigit()]       # 空格和连字符仅作分组，不参与校验
    if not 13 <= len(digits) <= 19:                           # 支付卡号长度范围在输入边界统一收窄
        return False                                         # 过短或过长不能当银行卡
    doubled = [value * 2 if at % 2 else value for at, value in enumerate(reversed(digits))]
    return sum(value - 9 if value > 9 else value for value in doubled) % 10 == 0  # 从校验位向左交替加倍


def measures(text: str) -> dict:
    """提取长度、非重复比例与标点完整度；三项都落在 0–1。"""
    body = text.strip()                                      # 首尾空白不应替正文贡献长度
    if not body:                                             # 空正文三项都没有有效信息
        return dict.fromkeys(WEIGHTS, 0.0)                    # 空正文不能从标点或重复率获得分数

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

    return {"length": length, "repeat": repeat, "punctuation": punctuation}  # 保留可解释的三项原始分



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


# --- 用人工标注校准权重 ---

def fit(rows: list, annotator: str, seed: int = 0) -> dict:
    """按唯一正文划分 80/20 训练/留出集，拟合非负、和为一的权重。

    rows 是 {text: 正文, score: 人工分数 0–1} 列表；annotator 记录标注来源，
    不能用模型生成的标签冒充人工。返回独立留出误差、旧权重误差及输入摘要。
    示例：profile = fit(labels, annotator="review-team/2026-11")。
    """
    import hashlib                                           # 只保存摘要，不把人工数据复制进档案
    import json                                              # 规范序列化保证输入摘要可复现
    import math                                              # 拒绝 NaN/Inf 标签
    from utils.fail import fail                              # 外部标注错误按项目入口格式报告

    if not isinstance(annotator, str) or not annotator.strip():  # 来源是能核查标注真实性的最低信息
        fail("input", "人工标注校准必须填写 annotator 来源")
    unique = {}                                              # 同一正文只能进一个集合，防止留出泄漏
    for number, row in enumerate(rows, 1):                   # JSONL 行号进入错误说明
        if not isinstance(row, dict) or not isinstance(row.get("text"), str):
            fail("input", f"第 {number} 条标注缺少 text 正文")
        text = row["text"].strip()                           # 空白变体不算独立样本
        value = row.get("score")                            # 分数必须是外部明确给出的数
        if not text or type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            fail("input", f"第 {number} 条标注需要非空正文与 0–1 有限 score")
        unique.setdefault(text, []).append(value)             # 多位标注者对同文打分在切分前取平均
    if len(unique) < 10:                                      # 小于十篇连两篇独立留出都不够
        fail("input", "校准至少需要 10 篇不同正文的人工标注")
    examples = sorted((text, sum(values) / len(values)) for text, values in unique.items())
    digest = hashlib.sha256(json.dumps(examples, ensure_ascii=False).encode()).hexdigest()
    random.Random(seed).shuffle(examples)                     # 固定种子，报告可复跑
    count = max(2, len(examples) // 5)                         # 留五分之一，仅用于最后报告
    held = [(measures(text), value) for text, value in examples[:count]]
    train = [(measures(text), value) for text, value in examples[count:]]

    def error(weights, data):                                # 三项特征已归一化，用均方误差衡量标注偏差
        return sum((sum(parts[key] * weights[key] for key in WEIGHTS) - label) ** 2 for parts, label in data) / len(data)

    candidates = [dict(zip(WEIGHTS, (a / 100, b / 100, (100-a-b) / 100)))
                  for a in range(101) for b in range(101-a)]  # 0.01 网格只有 5151 个组合，无额外数值依赖
    chosen = min(candidates, key=lambda weights: error(weights, train))  # 留出标签不参与选择权重
    return {                                                  # 档案明确分开拟合结果和泛化证据
        "version": 1, "weights": chosen, "annotator": annotator.strip(),
        "source_sha256": digest, "seed": seed,
        "train_count": len(train), "heldout_count": len(held),
        "train_mse": error(chosen, train), "heldout_mse": error(chosen, held),
        "default_heldout_mse": error(WEIGHTS, held),
    }


def profile(path) -> dict:
    """读取人工校准档案并返回三项权重；坏档在 weave 开跑前拒绝。"""
    import json                                              # 档案格式固定为 JSON
    import math                                              # 只接收有限权重
    from pathlib import Path                                 # 用户提供的档案路径
    from utils.fail import fail                              # 入口统一处理用户可修错误

    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))  # 只在外部文件边界捕获解析错误
    except (OSError, ValueError) as bad:
        fail("input", f"质量档案无法读取: {bad}", where=str(path))
    weights = data.get("weights") if isinstance(data, dict) else None
    if not isinstance(weights, dict) or set(weights) != set(WEIGHTS):
        fail("input", "质量档案需要 length / repeat / punctuation 三项权重")
    if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in weights.values()):
        fail("input", "质量权重必须是非负有限数")
    if not math.isclose(sum(weights.values()), 1, abs_tol=1e-6):
        fail("input", "质量权重之和必须等于 1")
    return weights                                           # 下游直接用，不重复验证
