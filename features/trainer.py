"""训练：真的把模型往前推。取批 → 前向 → 交叉熵 → 反向 → 累积 → 裁剪 → 一步更新。

调用示例：
  from features.trainer import Choice, Trainer
  job = Trainer(model, geo, Choice(out=Path("out/t"), files=[Path("bolt/bolt.jsonl")], steps=20))
  job.run()                       # 返回跑到第几步

产物（一次一档，不覆盖已有的）：
  step-000020/model.safetensors  全量权重；挂了 LoRA 时换成 adapter.safetensors
  step-000020/optimizer.pt       AdamW 的一阶二阶 + 梯度累积进度
  step-000020/state.json         步数、数据游标、随机数状态、最近几步的 loss
  latest                         最新一档的目录名，--resume auto 照它找
为什么三样分开：权重给导出，优化器和游标给续训，随机数状态给复现，少一样都接不上。

分词用字节：一个 UTF-8 字节一个号，对 vocab 取模就落进词表。不引分词器，换模型也不会因词表对不上崩。
交叉熵按 token 分块算，一次只把一块 token 的 V 维展开，词表上万也不炸内存。
"""

import contextlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn import functional as F

from features import lora
from utils import logs
from utils.fail import fail

CHUNK = 64  # 一次算多少 token 的交叉熵，越小越省内存、越慢
HISTORY = 20  # state.json 里留最近几步的 loss，续训后用来对照曲线


@dataclass
class Choice:
    """这一趟怎么跑。由命令层填，配方里的 train 节直接映射进来。"""

    out: Path  # 产物目录
    files: list  # 训练数据，weave 出来的 bolt.jsonl
    batch_size: int = 1  # 单步几条
    grad_accum: int = 1  # 攒几步再更新
    lr: float = 2e-4  # 学习率峰值
    warmup: int = 0  # 预热步数
    weight_decay: float = 0.0  # 权重衰减
    clip: float = 1.0  # 梯度裁剪阈值
    steps: int = 0  # 要跑的步数，0 表示按 epochs 换算；续训时是再跑多少步
    epochs: float = 1.0  # 过几遍数据
    save_every: int = 50  # 每多少步存一档，0 只在收尾存
    log_every: int = 1  # 每多少步报一次
    seed: int = 42  # 随机种子
    precision: str = "fp32"  # 精度，只有卡上才真生效
    resume: Path = None  # 从哪一档接着跑
    base: Path = None  # 基座目录，挂件存档要写上
    lora: dict = None  # 低秩配置，给了就只训挂件


class Trainer:
    """一次训练。一个 job 用一个实例。"""

    def __init__(self, model, geo, choice: Choice) -> None:
        self.model = model  # 要训的模型
        self.geo = geo  # 几何，编码和分块都要用
        self.choice = choice  # 这一趟的参数
        self.out = Path(choice.out)  # 产物目录
        self.samples = self._read()  # 编好的样本，一次读进来，别边跑边读盘
        self.total = choice.steps or max(1, math.ceil(choice.epochs * len(self.samples) / (choice.batch_size * choice.grad_accum)))  # 没给步数就按轮数换算
        if choice.lora:  # 挂低秩
            lora.attach(model, choice.lora.get("target") or list(lora.TARGETS)[:4], choice.lora["r"], choice.lora["alpha"], choice.lora["dropout"])
            params = lora.trainable(model)  # 只训挂件
        else:  # 全量训
            params = [one for one in model.parameters() if one.requires_grad]
        self.params = params  # 可训参数，裁剪时直接用
        self.optim = torch.optim.AdamW(params, lr=choice.lr, weight_decay=choice.weight_decay)  # 优化器只管可训的
        self.step = 0  # 已经跑了几步
        self.cursor = 0  # 数据游标，全局第几条
        self.history = []  # 最近的 loss
        self._perm = []  # 当前这轮的取样顺序
        self.rng = torch.get_rng_state()  # 随机数状态，存盘要

    # --- 数据 ---

    def _read(self) -> list:
        """把全部文件读成编号列表，短于两个号的丢掉。"""
        out = []  # 攒起来的样本
        for one in self.choice.files:  # 一个文件一趟
            for line in Path(one).read_text(encoding="utf-8").splitlines():  # 一行一条样本
                if not line.strip():  # 空行跳过
                    continue
                text = json.loads(line).get("text") or ""  # 正文，bolt.jsonl 固定有这个字段
                ids = [byte % self.geo.vocab for byte in text.encode("utf-8")[: self.geo.max_seq_len]]  # 字节取模落进词表，超长截断
                if len(ids) > 1:  # 至少要有输入和答案
                    out.append(ids)
        if not out:  # 一条都用不了说明数据是空的
            fail("input", "训练数据里没有能用的样本", where=", ".join(str(one) for one in self.choice.files), hint="先 ori weave 出一份 bolt/")
        return out

    def _shuffle(self, epoch: int) -> None:
        """按 种子×轮次 生成这一轮的顺序，续训时同一轮能原样重来。"""
        gen = torch.Generator().manual_seed(self.choice.seed * 100003 + epoch)  # 种子掺进轮号
        self._perm = torch.randperm(len(self.samples), generator=gen).tolist()  # 整轮打乱

    def _next(self) -> list:
        """取下一批，取完一轮自动换顺序。"""
        pos = self.cursor % len(self.samples)  # 这一轮里的位置
        if pos == 0:  # 走到轮首就重排
            self._shuffle(self.cursor // len(self.samples))
        self.cursor += 1  # 游标往后挪
        return self.samples[self._perm[pos]]  # 按顺序取一条

    def _batch(self) -> tuple:
        """凑一批，右侧补 0，答案位补 -100。"""
        rows = [self._next() for _ in range(self.choice.batch_size)]  # 取够一批
        widest = max(len(one) for one in rows) - 1  # 最长的输入长度
        ids = torch.zeros(len(rows), widest, dtype=torch.long)  # 输入，补 0
        labels = torch.full((len(rows), widest), -100, dtype=torch.long)  # 答案，补位不参与算损失
        for at, one in enumerate(rows):  # 一条条填
            ids[at, : len(one) - 1] = torch.tensor(one[:-1])  # 前 n-1 个当输入
            labels[at, : len(one) - 1] = torch.tensor(one[1:])  # 后 n-1 个当答案
        return ids, labels

    # --- 训练 ---

    def _context(self):
        """半精度只在卡上开，CPU 一律 fp32。"""
        if torch.cuda.is_available() and self.geo.dtype(self.choice.precision) in (torch.bfloat16, torch.float16):
            return torch.autocast("cuda", dtype=self.geo.dtype(self.choice.precision))  # 卡上开自动混合精度
        return contextlib.nullcontext()  # CPU 上什么都不套

    def _rate(self, step: int) -> float:
        """当前步的学习率：线性预热，然后余弦落到 0。"""
        if step < self.choice.warmup:  # 预热段
            return self.choice.lr * (step + 1) / self.choice.warmup
        left = max(1, self.total - self.choice.warmup)  # 剩下的步数
        return self.choice.lr * 0.5 * (1 + math.cos(math.pi * (step - self.choice.warmup) / left))  # 余弦下坡

    def _update(self) -> float:
        """跑一步：攒够累积再更新，返回这一步的平均损失。"""
        total = 0.0  # 这一步攒到的损失
        for _ in range(self.choice.grad_accum):  # 一个一个微批
            ids, labels = self._batch()  # 取微批
            with self._context():  # 半精度上下文
                loss = cross_entropy(self.model(ids), labels) / self.choice.grad_accum  # 除开累积数，梯度才不跟着放大
            loss.backward()  # 反向
            total += float(loss.detach())  # 累进账
        torch.nn.utils.clip_grad_norm_(self.params, self.choice.clip)  # 裁剪，防尖峰
        for group in self.optim.param_groups:  # 学习率按这一步重算
            group["lr"] = self._rate(self.step)
        self.optim.step()  # 更新
        self.optim.zero_grad(set_to_none=True)  # 清梯度
        self.rng = torch.get_rng_state()  # 记下随机数状态，续训要用
        return total

    def run(self) -> int:
        """跑到总步数，中途 Ctrl-C 存一档再退。"""
        self.out.mkdir(parents=True, exist_ok=True)  # 产物目录先建
        self.model.write_config(self.out, self.choice.precision)  # 结构描述留一份，导出时能用
        if self.choice.resume:  # 续训先把状态读回来
            self.restore(Path(self.choice.resume))
        self._write_run()  # 这一趟的设定留一份，回头知道跑的是什么
        try:
            while self.step < self.total:  # 一步一步推
                self.step += 1  # 先记步号，断在存盘前也对得上
                loss = self._update()  # 真跑一步
                self.history = (self.history + [round(loss, 6)])[-HISTORY:]  # 只留最近几步
                if self.step % self.choice.log_every == 0 or self.step == 1:  # 到点报一次
                    logs.step("spin", "训练", step=self.step, total=self.total, loss=round(loss, 4), lr=f"{self._rate(self.step):.2e}", gpu_gb=round(self._vram(), 2))
                if self.choice.save_every and self.step % self.choice.save_every == 0:  # 到点存一档
                    self.save()
                    logs.done("spin", "已存档", step=self.step)
        except KeyboardInterrupt:  # 手动掐断也不白跑
            self.save()
            logs.warn("spin", "已中断，断点存下了", step=self.step)
            return self.step
        self.save()  # 收尾必存
        logs.done("spin", "训练完成", step=self.step, total=self.total)
        return self.step

    def _vram(self) -> float:
        """卡上已用的峰值显存，GiB；没有卡就是 0。"""
        return torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0

    # --- 存读 ---

    def save(self) -> Path:
        """存一档：权重、优化器、状态。返回目录。"""
        folder = self.out / f"step-{self.step:06d}"  # 一档一个目录
        if self.choice.lora:  # 挂件只存 A/B
            lora.save(self.model, folder, self.choice.base)
        else:  # 全量训就存全量
            self.model.save(folder)
        self.model.write_config(folder, self.choice.precision)  # 结构描述跟着这一档走，导出时就在手边
        torch.save({"optim": self.optim.state_dict()}, folder / "optimizer.pt")  # 优化器态
        (folder / "state.json").write_text(json.dumps(self._state(), ensure_ascii=False, indent=2), encoding="utf-8")  # 步数、游标、随机数
        (self.out / "latest").write_text(folder.name, encoding="utf-8")  # 最新一档的指针
        return folder

    def _state(self) -> dict:
        """state.json 的内容。"""
        return {
            "step": self.step,  # 跑到第几步
            "cursor": self.cursor,  # 数据游标
            "total": self.total,  # 总步数
            "seed": self.choice.seed,  # 种子
            "rng": self.rng.numpy().tobytes().hex(),  # torch 随机数状态，原字节存成十六进制
            "cuda_rng": [one.numpy().tobytes().hex() for one in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [],  # 卡上的随机数状态
            "python_rng": random.getstate(),  # python 随机数状态
            "history": self.history,  # 最近几步的 loss
        }

    def restore(self, folder: Path) -> None:
        """从一档读回全部状态，接着往下跑。"""
        if folder.name == "auto" or folder.is_file():  # --resume auto 走 latest，给了 latest 文件也照它指
            name = (self.out / "latest").read_text(encoding="utf-8").strip() if folder.name == "auto" else folder.read_text(encoding="utf-8").strip()
            folder = self.out / name
        if not (folder / "state.json").exists():  # 给的是产物根目录时，顺手挑最新那一档
            folder = folder / (folder / "latest").read_text(encoding="utf-8").strip()
        state = json.loads((folder / "state.json").read_text(encoding="utf-8"))  # 状态先读
        if self.choice.lora:  # 挂件只灌 A/B
            lora.load(self.model, folder)
        else:  # 全量训就灌全量
            self.model.load(folder)
        self.optim.load_state_dict(torch.load(folder / "optimizer.pt", map_location="cpu", weights_only=False)["optim"])  # 优化器态
        self.step = state["step"]  # 从断点接着数
        self.cursor = state["cursor"]  # 从断点接着取数据
        self._shuffle(self.cursor // len(self.samples))  # 这一轮的顺序按轮号原样重放
        self.history = state["history"]  # 曲线接上
        self.total = self.step + self.choice.steps if self.choice.steps else state["total"]  # 给了 5 步就是再跑 5 步，没给就照旧档的总数
        self.total = max(self.total, self.step)  # 别把总步数改到断点之前
        self.rng = torch.frombuffer(bytearray.fromhex(state["rng"]), dtype=torch.uint8).clone()  # 随机数状态还原
        torch.set_rng_state(self.rng)
        if state["cuda_rng"]:  # 卡上跑的还要把卡上的状态放回去
            torch.cuda.set_rng_state_all([torch.frombuffer(bytearray.fromhex(one), dtype=torch.uint8) for one in state["cuda_rng"]])
        random.setstate(_state_tuple(state["python_rng"]))  # python 随机数状态还原
        logs.done("spin", "续训读档", step=self.step, cursor=self.cursor)

    def _write_run(self) -> None:
        """把这一趟的设定写进 out/run.json。"""
        meta = {
            "geometry": self.geo.as_dict(),  # 几何
            "train": {"steps": self.total, "batch_size": self.choice.batch_size, "grad_accum": self.choice.grad_accum, "lr": self.choice.lr, "seed": self.choice.seed, "precision": self.choice.precision},  # 训练超参
            "lora": self.choice.lora,  # 挂件配置，没挂是 null
            "files": [str(one) for one in self.choice.files],  # 数据来源
        }
        (self.out / "run.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def _state_tuple(raw: list) -> tuple:
    """JSON 读回来的 python 随机数状态还原成元组。"""
    return (raw[0], tuple(raw[1]), raw[2] if len(raw) > 2 else None)  # 三段原样拼回


def cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """按 token 分块算交叉熵，跳掉 -100 的补位。"""
    flat_logits = logits.reshape(-1, logits.shape[-1])  # 拍平成 (token, 词表)
    flat_labels = labels.reshape(-1)  # 答案也拍平
    total, count = torch.zeros((), dtype=torch.float32), 0  # 总和与有效 token 数
    for start in range(0, flat_labels.numel(), CHUNK):  # 一块一块来
        part = flat_labels[start : start + CHUNK]  # 这块的答案
        keep = part != -100  # 这块里哪些是真 token
        if not bool(keep.any()):  # 整块都是补位就跳过
            continue
        total += F.cross_entropy(flat_logits[start : start + CHUNK][keep], part[keep], reduction="sum")  # 只算真 token
        count += int(keep.sum())  # 累计有效数
    return total / max(1, count)  # 平均，一块都没有时按 1 除
