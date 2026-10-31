"""回归门：拿新报告跟基线比总分，掉超过允许值就判不过，并列出掉分的题号。

调用示例：
  from features import gate
  out = gate.check(new_report, gate.load(Path("out/base/report.json")), min_delta=0.02)
  out.exit_code  # 掉过分是 1，没掉是 0
  out.dropped    # ["t3", "t7"]，分数比基线低的题号

比的是总分：总分掉过 min_delta 就拦，掉得少就当正常抖动放过去。
掉分题号单独列出来，是为了让人直接知道该去看哪几题，不用自己翻两份报告。
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

from utils.fail import fail

EPS = 1e-9  # 浮点误差，分差在这个量级以内不算掉


@dataclass
class Outcome:
    """一次过门的结论。"""

    delta: float                                 # 总分差，新减旧，负数是掉分
    allowed: float                               # 允许掉多少
    dropped: list = field(default_factory=list)  # 分数比基线低的题号

    @property
    def ok(self) -> bool:
        """总分没掉过允许值就放行。"""
        return self.delta >= -self.allowed - EPS  # 掉在允许值以内就算过

    @property
    def exit_code(self) -> int:
        """给命令层用的退出码：过 0，不过 1。"""
        return 0 if self.ok else 1  # 过就 0，不过就 1

    def say(self) -> str:
        """拼成给人看的结论。"""
        lines = [f"总分 {self.delta:+.2f}（允许掉 {self.allowed:.2f}）"]  # 先说分差和允许值
        lines.append("回归门：过" if self.ok else "回归门：不过")          # 再说结论
        if self.dropped:                                                  # 有掉分题就点名
            lines.append("掉分题: " + " ".join(self.dropped))             # 逐题号列出来
        return "\n".join(lines)                                          # 拼成给人看的几行


def load(path: Path) -> dict:
    """读一份基线报告。读不出、不是报告就在这停。"""
    if not path.exists():                                      # 没有基线就无从比
        fail("input", f"基线报告不在: {path}", hint="先跑一次不带 --baseline 的，把它的 report.json 留作基线")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))    # 基线是 json
    except json.JSONDecodeError as wrong:                      # 不是合法 json
        fail("input", f"基线不是合法 json: {wrong}", where=str(path))
    if "total" not in data or "tasks" not in data:             # 缺这两样就不是报告
        fail("input", str(path) + " 不像评测报告", hint="要的是 judge 出的 report.json")
    return data                                                # 交出去


def check(current: dict, baseline: dict, min_delta: float) -> Outcome:
    """比新报告和基线。返回结论，不含退出码。"""
    delta = round(current["total"] - baseline["total"], 4)                                                      # 总分差，负数是掉分
    before = {row["id"]: row["score"] for row in baseline["tasks"]}                                             # 基线里每题多少分
    dropped = sorted(row["id"] for row in current["tasks"] if row["score"] < before.get(row["id"], 0) - EPS)    # 比基线低的题号
    return Outcome(delta=delta, allowed=abs(min_delta), dropped=dropped)                                        # 结论交回去，退出码由调用方取
