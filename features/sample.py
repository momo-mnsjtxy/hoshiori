"""样本：从一条原始记录里认出正文和图片路径，按规矩收净、脱敏、评分并决定是否收下。

调用方式：read(record, policy) 返回规范样本；images 可写一个路径或路径数组，默认最多四张。
image_at 可给每张图在清理后正文里的插入下标，数量要与 images 相同，下标严格递增且不超过正文长度。
拒收是正常业务分支，每条原因码会进入 rejects.jsonl；正文、图片和插入位置只在这个入口检查一次。

调用示例：
  from features import sample
  policy = sample.Policy()                                                        # 默认脱敏，最多收四张图
  clean = sample.read({"text": " 一段话 ", "images": ["photo.png"], "image_at": [1]}, policy)  # 图片跟在正文下标 1 后
  sample.read({"text": ""}, policy)                                               # 抛 Refused("too_short")
"""

import math                                # 难度必须是有限数，NaN 无法排序
import re                                  # 清理控制符和多余空白
import unicodedata                         # 统一字符的 Unicode 表示
from dataclasses import dataclass          # 把样本规矩集中成一份数据
from pathlib import Path                   # 检查图片路径是否真有文件

from features import clean as text_clean   # 正文脱敏和质量评分共用一份规则

CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")  # 不可见控制符，粘着会污染分词
WS = re.compile(r"[ \t\u00a0\u3000]+")  # 各种空白，含全角空格
BLANKS = re.compile(r"\n{3,}")  # 连续三个以上换行，压成两行


class Refused(Exception):
    """这一条不要了。code 进 rejects 档，detail 补一句细节。"""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(code)
        self.code = code
        self.detail = detail


@dataclass
class Policy:
    """收样本的规矩。默认值就是常规做法，命令行按需改。"""

    fields: tuple = ("text", "content", "raw", "body")  # 按顺序找正文，先找到先用
    roles: tuple = ("system", "user", "assistant")  # 对话里认这三种角色
    form: str = "NFC"  # 统一字符表示；NFKC 会连带改字宽，默认不用
    min_chars: int = 1  # 短于此数算空样本
    max_chars: int = 0  # 0 表示不限长
    max_images: int = 4  # 一条样本最多带几张图片
    min_score: float = 0.0  # 质量分低于此线拒收；0 表示不筛
    scrub: bool = True  # 默认替换敏感片段
    quality_weights: dict = None  # 人工标注校准权重；None 保持启发式默认


def text_of(record: dict, policy: Policy) -> str:
    """认出正文：先按字段顺序找，再退到对话里的各轮内容。"""
    for key in policy.fields:  # 常见正文键，按顺序试
        value = record.get(key)
        if isinstance(value, str) and value.strip():  # 有字才算，纯空白继续往下找
            return value
    for key in policy.fields:  # 字段在但全是空白，交回空串让后面报 too_short
        if isinstance(record.get(key), str):
            return record[key]
    turns = record.get("messages")  # 对话样本：把各轮拼成一段，带上角色标记
    if isinstance(turns, list) and turns:
        parts = []
        for turn in turns:
            role = str(turn.get("role", ""))  # 角色名，拼进标记里
            body = turn.get("content")
            if not isinstance(body, str) or not body.strip():  # 空轮次直接跳
                continue
            if policy.roles and role not in policy.roles:  # 认不出的角色说明这份数据有别的来源
                raise Refused("unknown_role", role)
            parts.append(f"<|{role}|>\n{body}")  # 角色标记 + 正文，跟常见微调格式一致
        if parts:
            return "\n".join(parts)
    raise Refused("no_text_field", ",".join(policy.fields))  # 三种都试过，确实没有正文


def clean(text: str, policy: Policy) -> str:
    """按规矩收干净，顺手量长度并执行长短限制。"""
    text = CTRL.sub("", text)  # 先剔控制符，免得后面按空白折叠时被它卡住
    text = unicodedata.normalize(policy.form, text)  # 统一字符表示，组合字符和预组合字符归一种
    text = WS.sub(" ", text)  # 多种空白统一成一个半角空格
    text = BLANKS.sub("\n\n", text)  # 空行最多留一行
    text = text.strip()
    if len(text) < policy.min_chars:  # 收完还不够长，说明本来是空的
        raise Refused("too_short", str(len(text)))
    if policy.max_chars and len(text) > policy.max_chars:  # 超长样本会撑爆显存
        raise Refused("too_long", str(len(text)))
    return text


def read(record: dict, policy: Policy) -> dict:
    """主入口：一条原始记录进，一条规范样本出。"""
    text = clean(text_of(record, policy), policy)                              # 认正文再统一字符与空白
    if policy.scrub:                                                           # 命令行可以明确关闭脱敏
        text, _ = text_clean.scrub(text)                                       # 命中数由公开 scrub 接口按需取用
    quality = text_clean.score(text, policy.quality_weights)                                           # 对最终会训练的正文打分
    if quality < policy.min_score:                                             # 低质量样本和其他拒收走同一条路
        raise Refused("low_score", f"{quality:.4f} < {policy.min_score:.4f}")  # 拒收档留下实际分数与门槛
    given = record.get("images")                                               # 图片在正文之外单独读取
    images = [given] if isinstance(given, str) else list(given or [])           # 单路径和路径数组统一成列表
    if len(images) > policy.max_images:                                         # 超出单条样本的图片预算
        raise Refused("too_many_images", str(len(images)))                     # 原因码交给拒收档记账
    for path in images:                                                        # 每个路径只在样本入口检查一次
        if not Path(path).is_file():                                            # 找不到文件就不能留给训练阶段
            raise Refused("no_image", str(path))                               # 明说是哪张图不存在
    image_at = record.get("image_at")                                          # 不给位置就是旧的整段图片前缀
    if "image_at" in record:                                                   # 显式给了就必须是一图一个递增下标
        if not isinstance(image_at, list) or len(image_at) != len(images):      # 数量不齐时训练器无法知道哪张图放哪
            raise Refused("bad_image_at", "image_at 要与 images 等长")         # 在样本边界直接说明格式
        if any(type(at) is not int or at < 0 for at in image_at):               # bool 也不能冒充整数下标
            raise Refused("bad_image_at", "image_at 只能是非负整数下标")       # 不把坏值留到张量 scatter 时才炸
        if any(left >= right for left, right in zip(image_at, image_at[1:])):    # 每张图要沿正文方向依次出现
            raise Refused("bad_image_at", "image_at 必须严格递增")             # 相同或倒序都拒收
        if image_at and image_at[-1] > len(text):                               # 等于正文长度表示放到正文末尾
            raise Refused("bad_image_at", f"{image_at[-1]} > {len(text)}")      # 留下实际越界值和正文长度

    out = {"text": text, "chars": len(text)}                                  # 正文和长度交给下游
    if "difficulty" in record:                                               # 人工难度跟着 weave 产物进入课程训练
        difficulty = record["difficulty"]                                    # 仅接受明确的 0–1 数字
        if type(difficulty) not in (int, float) or not math.isfinite(difficulty) or not 0 <= difficulty <= 1:
            raise Refused("bad_difficulty", "difficulty 必须是 0–1 有限数")    # 在样本入口拒绝，训练内部无需猜测
        out["difficulty"] = float(difficulty)                                # 难度方向统一为越大越难
    out["source"] = record.get("source") or "unknown"                         # 记来源，混配时按来源称重
    if "images" in record:                                                     # 原记录带图时保留规范后的路径列表
        out["images"] = images                                                 # 训练器按原顺序准备每张图片
    if "image_at" in record:                                                   # 只给新样本增加字段，旧 bolt 结构不变
        out["image_at"] = image_at                                             # 训练器按同一顺序展开到每个图片块
    if isinstance(record.get("messages"), list):                               # 对话样本仍保留结构化轮次
        messages = record["messages"]                                         # 关闭脱敏时保持原来的对象与内容
        if policy.scrub:                                                       # 开启后不能让原文从旁路漏进产物
            messages = []                                                     # 每轮复制一份，不修改调用方的原记录
            for turn in record["messages"]:                                  # 角色和其他字段都原样保留
                safe = dict(turn)                                              # 只改这一轮自己的正文
                if isinstance(safe.get("content"), str):                       # 外部记录的 content 可能不是字符串
                    safe["content"], _ = text_clean.scrub(safe["content"])     # 使用正文同一套规则
                messages.append(safe)                                         # 脱敏后的轮次按原顺序收回
        out["messages"] = messages                                            # 训练或导出不会再看到敏感原文
    for key, value in (record.get("meta") or {}).items():                       # 上游附加信息逐项搬过来
        out.setdefault("meta", {})[key] = value                                # 第一个字段到来时再建 meta
    return out                                                                  # 交出一条可写入 bolt 的样本
