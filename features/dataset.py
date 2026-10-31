"""数据集：一趟 weave 的产物目录由它建、由它写，别的文件不碰这些产物。

调用示例：
  from features.dataset import Dataset, Choice
  job = Dataset(Choice(out=Path("bolt"), name="zh-mix", license="cc0"))
  counts = job.make(["data/a.jsonl", "data/b.jsonl"])
  counts.summary()      # {"read": 207, "kept": 182, "duplicated": 24, …}

产物固定四样，全在 out 目录里：
  bolt.jsonl       收下的正文，一行一条，训练只吃这个
  rejects.jsonl    拒收的，带原因码和出处，方便回头查数据
  fingerprint.json 整份数据的指纹
  card.json        数据集卡：来源、许可、条数、指纹、去重参数

为什么产物写盘只放在这个文件：改产物格式时只改一处，接手者也知道产物谁负责。
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

from features import duplicate, fingerprint, source
from features.sample import Policy, Refused, read as read_sample
from utils import logs


@dataclass
class Choice:
    """这一趟怎么跑。由命令行填，ui 层不另存一份。"""

    out: Path = Path("bolt")  # 产物目录
    name: str = "weave"  # 数据集名，进卡片
    license: str = "unknown"  # 来源许可，进卡片；留 unknown 会被后续校验拦
    policy: Policy = field(default_factory=Policy)  # 收样本的规矩
    rule: duplicate.Rule = field(default_factory=duplicate.Rule)  # 判重的规矩
    algo: str = "sha256"  # 指纹算法
    keep_duplicates: bool = False  # 判掉的要不要也留进 bolt
    dry_run: bool = False  # 只跑流程，产物照样写，卡片里标注


@dataclass
class Counts:
    """这一趟的账。"""

    read: int = 0
    kept: int = 0
    duplicated: int = 0
    refused: dict = field(default_factory=dict)  # 原因码 → 条数

    def refuse(self, code: str) -> None:
        self.refused[code] = self.refused.get(code, 0) + 1

    def summary(self) -> dict:
        total = self.read or 1  # 防除零，读 0 条时按 1 算比例
        return {
            "read": self.read,
            "kept": self.kept,
            "duplicated": self.duplicated,
            "refused": dict(self.refused),
            "keep_rate": round(self.kept / total, 4),
            "dup_rate": round(self.duplicated / total, 4),
        }


class Dataset:
    """一趟 weave。一次 job 用一个实例。"""

    def __init__(self, choice: Choice) -> None:
        self.choice = choice
        self.counts = Counts()
        self._dup = duplicate.Duplicate(choice.rule)  # 判重索引，整趟共用
        self._stamp = fingerprint.Stamp(choice.algo)  # 指纹器，整趟共用
        self._kept_path = choice.out / "bolt.jsonl"
        self._refused_path = choice.out / "rejects.jsonl"

    def make(self, specs: list) -> Counts:
        """主流程：逐个来源读 → 收样本 → 判重 → 写盘 → 出指纹和卡。"""
        self._specs = specs  # 卡片要列来源，先存住
        self.choice.out.mkdir(parents=True, exist_ok=True)  # 产物目录先建好，后面只管写
        with self._kept_path.open("w", encoding="utf-8") as kept, self._refused_path.open("w", encoding="utf-8") as refused:
            for spec in specs:  # 一个来源一趟，来源之间互不影响
                logs.step("weave", "开始读", source=spec)
                for record, where in source.read(spec):  # 一条条来
                    self._one(record, where, kept, refused)
                logs.done("weave", "读完", source=spec)
        self._finish()
        return self.counts

    def _one(self, record: dict, where: str, kept, refused) -> None:
        """处理一条：收不下的进拒收档，收得下但重复的按开关处理。"""
        self.counts.read += 1
        try:
            sample = read_sample(record, self.choice.policy)  # 规范化，不合格抛 Refused
        except Refused as bad:
            self.counts.refuse(bad.code)  # 记原因码
            _put(refused, {"reason": bad.code, "detail": bad.detail, "origin": where})
            return
        hit = self._dup.check(sample["text"])  # 跟已收的比，重复返回对谁的下标
        if hit is not None:
            self.counts.duplicated += 1
            if not self.choice.keep_duplicates:  # 默认判掉就不留
                _put(refused, {"reason": "duplicate", "detail": str(hit), "origin": where})
                return
            sample["dup_of"] = hit  # 留的时候标上跟谁重，训练可以后处理
        sample["origin"] = where  # 出处跟着正文走，方便回查
        self._stamp.feed(sample["text"])  # 只有真正收下的才进指纹
        _put(kept, sample)
        self.counts.kept += 1
        if self.counts.kept % 1000 == 0:  # 每千条报一次进度，别静默太久
            logs.step("weave", "已收", kept=self.counts.kept, read=self.counts.read)

    def _finish(self) -> None:
        """写指纹和卡片。卡片缺字段会当场报错，不留半张。"""
        stamp = self._stamp.seal()
        (self.choice.out / "fingerprint.json").write_text(json.dumps(stamp, ensure_ascii=False, indent=2), encoding="utf-8")
        card = {
            "schema": "itoguchi/1",  # 卡片格式版本
            "name": self.choice.name,
            "license": self.choice.license,
            "sources": [{"uri": spec, "license": self.choice.license} for spec in self._specs],
            "counts": self.counts.summary(),
            "fingerprint": stamp["content"],
            "dedup": f"阈值 {self.choice.rule.threshold} / 窗口 {self.choice.rule.size} / 指纹 {self.choice.rule.count}",
            "dry_run": self.choice.dry_run,
        }
        if not card["sources"]:  # 没有来源卡就不是卡，别写出去
            raise ValueError("没有任何来源，卡片出不来")
        (self.choice.out / "card.json").write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")


def _put(handle, payload: dict) -> None:
    """往打开的档里写一行 JSON。"""
    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
