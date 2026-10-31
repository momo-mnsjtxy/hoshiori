"""样本：从一条原始记录里认出正文，按规矩收干净，不合格就拒收并说明原因。

调用示例：
  from features import sample
  policy = sample.Policy()                       # 默认规矩：NFC、折叠空白、最短 1 字
  clean = sample.read({"text": "  一段  话  "}, policy)   # 返回 {"text": "一段 话", ...}
  sample.read({"text": ""}, policy)              # 抛 Refused("too_short")

拒收是正常业务分支，不是故障：一次 weave 会拒掉很多条，每条都带原因码进 rejects.jsonl，
接手者看原因码就知道是数据脏还是规矩太严。规矩本身由调用方从命令行传进来。
"""

import re
import unicodedata
from dataclasses import dataclass, field

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
    text = clean(text_of(record, policy), policy)  # 认正文再收干净，任何一步不合规都拒
    out = {"text": text, "chars": len(text)}  # 只带下游真用得上的两样
    out["source"] = record.get("source") or "unknown"  # 记来源，混配时按来源称重
    if isinstance(record.get("messages"), list):  # 对话样本原样留一份，训练要用结构化那版
        out["messages"] = record["messages"]
    for key, value in (record.get("meta") or {}).items():  # 上游给的附加信息搬过来
        out.setdefault("meta", {})[key] = value
    return out
