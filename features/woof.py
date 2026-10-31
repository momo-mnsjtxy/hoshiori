"""混配：一堆来源各占多少，按权重抽。同一个种子抽出来的比例固定，能复算。

调用示例：
  from features import woof
  plan = woof.read(Path("zh-mix.woof.yaml"))   # 读配方
  plan.split()                                 # {"wiki": 1200, "code": 300}
  plan.stamp()                                 # 配方哈希，进卡片

配方长这样：
  target: 100000
  temperature: 1.0
  sources: [{name: wiki, weight: 4}, {name: code, weight: 1}]

temperature 越小越均匀、越大越贴权重。take 是硬性条数，填了就优先满足它。
"""

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path

from utils import yaml_io
from utils.fail import fail


@dataclass
class Source:
    """一个来源要抽多少。"""

    name: str
    weight: float
    take: int = 0  # 硬性条数，0 表示按权重分


@dataclass
class Woof:
    """一份混配配方。"""

    sources: list = field(default_factory=list)
    seed: int = 0
    temperature: float = 1.0
    target: int = 0

    def split(self) -> dict:
        """算出每个来源抽多少条。"""
        if not self.sources:
            fail("schema", "混配配方里一个来源都没有", where="sources")
        if self.temperature <= 0:
            fail("schema", "temperature 要大于 0", where="temperature")
        fixed = sum(one.take for one in self.sources)  # 硬性条数先加起来，这部分不参与分配
        budget = self.target or fixed  # 没给总量就拿硬性条数当总量
        if not budget:  # 两头都没有，分不出来
            fail("schema", "target 和 take 至少要给一个", where="target")
        rest = max(0, budget - fixed)  # 扣掉硬性部分，剩下的才按权重分
        weights = [max(one.weight, 0) ** (1 / self.temperature) for one in self.sources]  # 温度改写权重
        whole = sum(weights) or 1
        out = {}
        for one, weight in zip(self.sources, weights):
            out[one.name] = min(int(round(rest * weight / whole)) + one.take, budget)  # 权重份额 + 硬性条数
        return out

    def stamp(self) -> str:
        """配方哈希。改权重、改种子都会变。"""
        body = json.dumps({"s": [[o.name, o.weight, o.take] for o in self.sources], "seed": self.seed, "t": self.temperature, "n": self.target}, sort_keys=True)
        return "sha256:" + hashlib.sha256(body.encode()).hexdigest()


def read(path: Path) -> Woof:
    """读一份混配配方。"""
    data = yaml_io.read(path)  # 读配方，坏 yaml 在这里停
    sources = []
    for item in data.get("sources", []):  # 每个来源一条，缺 name 或 weight 会直接报
        sources.append(Source(name=str(item["name"]), weight=float(item["weight"]), take=int(item.get("take", 0))))
    return Woof(sources=sources, seed=int(data.get("seed", 0)), temperature=float(data.get("temperature", 1.0)), target=int(data.get("target", 0)))
