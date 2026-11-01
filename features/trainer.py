"""训练：真的把模型往前推。取批 → 前向 → 交叉熵 → 反向 → 累积 → 裁剪 → 一步更新。

调用示例：
  from features.trainer import Choice, Trainer
  job = Trainer(model, geo, Choice(out=Path("out/t"), files=[Path("bolt/bolt.jsonl")], steps=20))
  job.run()                       # 返回跑到第几步；健康检查叫停时 job.reason 非空，命令层据此退 1

多卡走 torchrun：每个进程自己读 LOCAL_RANK 建进程组，ddp 用 DistributedDataParallel 包一层（梯度在反向时
跨进程平均），fsdp2 用 fully_shard 逐层切（参数真摊开）。只有 0 号进程写盘和打日志，其余进程算完就等收尾，
否则同一份产物目录几路抢写。LOCAL_RANK 不在环境里就是单进程，一条分布式代码都不进。

健康检查盯三样：loss 是 NaN/Inf 连续 3 步、梯度范数超 1e3 连续 5 步、吞吐跌破前 20 步中位数的四成。
处置照配方 train.health：stop 存一档再退，退出码 1，原因写进 run.json 和日志；warn 只喊一声接着跑。

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
import os
import copy
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.distributed.fsdp import fully_shard
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from features import alignment, lora
from utils import logs
from utils.fail import fail

CHUNK = 64  # 一次算多少 token 的交叉熵，越小越省内存、越慢
HISTORY = 20  # state.json 里留最近几步的 loss，续训后用来对照曲线
SPEED = 21  # 吞吐窗口：当前一步 + 前 20 步的基线
NAN_STEPS = 3  # loss 连着几步是 NaN/Inf 就算坏了
NORM_STEPS = 5  # 梯度范数连着几步超线就算炸了
NORM_LIMIT = 1e3  # 梯度范数上限
SLOW_RATIO = 0.4  # 吞吐跌破基线中位数的这个比例就算卡住


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
    strategy: str = "single"  # single 单卡 / ddp 每卡一份 / fsdp2 摊开
    devices: list = None  # 用哪几张卡，只记账和进 run.json；真分配在命令层的 torchrun
    grad_checkpoint: bool = False  # 前向分段重算，用时间换显存
    health: str = "stop"  # 指标坏了怎么办：stop 存一档再退 / warn 只喊一声
    task: str = "sft"  # 任务族，决定读什么数据、算哪种损失
    vocab: object = None  # 词表，给了就按它编；不给按字节取模（老的兜底路子）
    group: int = 4  # 对齐任务一组几条候选
    beta: float = 0.1  # DPO 的 KL 强度
    clip_ratio: float = 0.2  # GRPO / PPO 的裁剪区间
    kl_coef: float = 0.01  # GRPO 的 KL 罚系数
    temperature: float = 2.0  # distill 的温度
    teacher: Path = None  # distill 的教师权重，必填


class Trainer:
    """一次训练。一个 job 用一个实例。"""

    def __init__(self, model, geo, choice: Choice) -> None:
        self.geo = geo  # 几何，编码和分块都要用
        self.choice = choice  # 这一趟的参数
        self.out = Path(choice.out)  # 产物目录
        self.rank, self.world = _setup()  # torchrun 起的进程有 LOCAL_RANK；自己起的单进程就是 0/1
        self.lead = self.rank == 0  # 只有 0 号进程存盘打日志，别的进程只算
        self.pad_id = getattr(choice.vocab, "pad_id", 0)  # 补位用词表的 pad；没词表还是老的 0
        self.ref_model = None  # 对齐任务的冻结参考模型，dpo / grpo 要它算 ref 概率
        self.teacher = None  # distill 的教师，冻结
        self.samples = self._read()  # 编好的样本，一次读进来，别边跑边读盘
        self.total = choice.steps or max(1, math.ceil(choice.epochs * len(self.samples) / (choice.batch_size * choice.grad_accum)))  # 没给步数就按轮数换算
        model.grad_checkpoint = choice.grad_checkpoint  # 前向按配方决定要不要重算
        if choice.lora:  # 挂低秩
            lora.attach(model, choice.lora.get("target") or list(lora.TARGETS)[:4], choice.lora["r"], choice.lora["alpha"], choice.lora["dropout"])  # 挂件以外全冻上
        if choice.lora:  # 挂件是新建的层，建在 CPU，得跟基座放一起
            model.to(_device_of(model))
        if choice.task in {"dpo", "grpo", "ppo"}:  # 这三类要一份不动的参考模型
            self.ref_model = copy.deepcopy(_unwrap(model)).eval()  # 深拷贝一份基线，挂件和量化都在上面
            for one in self.ref_model.parameters():
                one.requires_grad_(False)  # 参考模型只前向
        if choice.task == "distill":  # 蒸馏要教师
            self.teacher = _teacher(choice, geo).eval()  # 载入教师并冻上
        self.model = _wrap(model, choice.strategy, self.world, self.rank)  # 多卡才包，单卡原样
        self.params = [one for one in self.model.parameters() if one.requires_grad]  # 包完再取一遍：fsdp2 会把参数换成新的
        self.optim = torch.optim.AdamW(self.params, lr=choice.lr, weight_decay=choice.weight_decay)  # 优化器只管可训的
        self.step = 0  # 已经跑了几步
        self.cursor = 0  # 数据游标，全局第几条
        self.history = []  # 最近的 loss
        self._perm = []  # 当前这轮的取样顺序
        self.rng = torch.get_rng_state()  # 随机数状态，存盘要
        self.last_extra = {}  # 这一步的额外指标（准确率、平均奖励），只进日志
        self.reason = ""  # 健康检查叫停的原因，非空就是这一趟坏了
        self.bad_loss = 0  # 连着几步 loss 是 NaN/Inf
        self.big_norm = 0  # 连着几步梯度范数超线
        self.speeds = []  # 最近几步的吞吐（token/秒）
        self.last_norm = 0.0  # 这一步裁剪前的梯度范数
        self.last_tokens = 0  # 这一步过了多少真 token，算吞吐用

    # --- 数据 ---

    def _read(self) -> list:
        """按任务族读数据。对齐任务读偏好对或带奖励的组，其余读正文。"""
        if self.choice.task == "dpo":  # 偏好对
            self.rows = self._read_pairs()
        elif self.choice.task in {"grpo", "rloo", "ppo"}:  # 带奖励的组
            self.rows = self._read_groups()
        else:  # 语言模型目标，正文就行
            self.rows = self._read_text()
        if not self.rows:  # 一条都用不了说明数据是空的
            fail("input", "训练数据里没有能用的样本", where=", ".join(str(one) for one in self.choice.files), hint="先 ori weave 出一份 bolt/")
        return self.rows

    def _ids(self, text: str) -> list:
        """编码：有词表就按词表，没词表按字节取模兜底。超长截断到最大长度。"""
        if self.choice.vocab is not None:  # 有词表就走真分词
            return self.choice.vocab.ids_of(text, self.geo.max_seq_len)
        return [byte % self.geo.vocab for byte in text.encode("utf-8")[: self.geo.max_seq_len]]  # 兜底：字节取模

    def _read_text(self) -> list:
        """把正文读成编号列表，短于两个号的丢掉。"""
        out = []  # 攒起来的样本
        for one in self.choice.files:  # 一个文件一趟
            for line in Path(one).read_text(encoding="utf-8").splitlines():  # 一行一条样本
                if not line.strip():  # 空行跳过
                    continue
                text = json.loads(line).get("text") or ""  # 正文，bolt.jsonl 固定有这个字段
                ids = self._ids(text)  # 编码，超长已截断
                if len(ids) > 1:  # 至少要有输入和答案
                    out.append(ids)
        return out

    def _read_pairs(self) -> list:
        """读偏好对，每对编成两边各一串号。"""
        out = []  # 攒起来的对
        for chosen, rejected in alignment.pairs(self.choice.files):  # 一对拆两边
            good, bad = self._ids(chosen["text"]), self._ids(rejected["text"])  # 两边各编一遍
            if len(good) > 1 and len(bad) > 1:  # 两边都够长才算一对
                out.append((good, bad))
        return out

    def _read_groups(self) -> list:
        """读带奖励的组：一行一个题面加若干候选与分数，组大小按配方取。"""
        out = []  # 攒起来的组
        for one in self.choice.files:  # 一个文件一趟
            for line in Path(one).read_text(encoding="utf-8").splitlines():  # 一行一组
                if not line.strip():  # 空行跳过
                    continue
                item = json.loads(line)  # 题面、候选、分数
                prompt = item.get("prompt") or ""  # 题面前缀，拼在每条候选前
                answers = (item.get("responses") or [])[: self.choice.group]  # 候选，按组大小截
                rewards = (item.get("rewards") or [])[: len(answers)]  # 分数要对得上候选
                if len(answers) < 2 or len(rewards) != len(answers):  # 不足两条算不出组内优势
                    continue
                out.append(([self._ids(prompt + one) for one in answers], rewards))  # 编码后的候选与分数
        return out

    def _shuffle(self, epoch: int) -> None:
        """按 种子×轮次×进程号 生成这一轮的顺序，续训时同一轮能原样重来。"""
        gen = torch.Generator().manual_seed(self.choice.seed * 100003 + epoch * 7 + self.rank)  # 种子掺进轮号和进程号，几路进程才取到不同的条
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
        ids = torch.full((len(rows), widest), self.pad_id, dtype=torch.long)  # 输入，补词表的 pad
        labels = torch.full((len(rows), widest), -100, dtype=torch.long)  # 答案，补位不参与算损失
        for at, one in enumerate(rows):  # 一条条填
            ids[at, : len(one) - 1] = torch.tensor(one[:-1])  # 前 n-1 个当输入
            labels[at, : len(one) - 1] = torch.tensor(one[1:])  # 后 n-1 个当答案
        here = _device_of(self.model)  # 模型在哪张卡，数据就得在哪
        return ids.to(here), labels.to(here)

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

    def _stack(self, rows: list) -> torch.Tensor:
        """把几条不等长的号补成一批。"""
        widest = max(len(one) for one in rows)  # 最长那条决定宽度
        batch = torch.full((len(rows), widest), self.pad_id, dtype=torch.long)  # 先全补 pad
        for at, one in enumerate(rows):  # 一条条填进去
            batch[at, : len(one)] = torch.tensor(one)
        return batch.to(_device_of(self.model))  # 跟模型放一起

    def _logp(self, ids: torch.Tensor) -> torch.Tensor:
        """当前模型给一批序列的答案位对数概率。"""
        return self._logp_of(self.model, ids)

    def _logp_of(self, net, ids: torch.Tensor) -> torch.Tensor:
        """指定网络算对数概率（每条一个数），补位不算。"""
        ids = ids.to(next(self.model.parameters()).device)  # 跟模型放同一张卡
        logits = net(ids[:, :-1])  # 前向，最后一位没有答案
        picked = F.log_softmax(logits.float(), dim=-1).gather(2, ids[:, 1:].unsqueeze(-1)).squeeze(-1)  # 只取答案那一位
        keep = ids[:, 1:] != self.pad_id  # 补位不算
        return (picked * keep).sum(dim=1) / keep.sum(dim=1).clamp(min=1)  # 按真 token 平均

    def _align_loss(self) -> tuple:
        """对齐任务的一步损失，返回（损失, 真 token 数）。"""
        task = self.choice.task  # 哪个任务族
        if task == "dpo":  # 偏好对，一步取 batch_size 对
            rows = [self._next() for _ in range(self.choice.batch_size)]  # 一批对
            good = self._stack([one[0] for one in rows])  # 正例
            bad = self._stack([one[1] for one in rows])  # 负例
            logp_c, logp_r = self._logp(good), self._logp(bad)  # 当前策略
            if self.ref_model is not None:  # 有参考模型就真算参考概率
                ref_c, ref_r = self._logp_ref(good), self._logp_ref(bad)
            else:  # 没有就拿当前值 detach，第一步起步够用
                ref_c, ref_r = logp_c.detach(), logp_r.detach()
            out = alignment.dpo(logp_c, logp_r, ref_c, ref_r, self.choice.beta)  # DPO 损失
            margin = out["margin"].mean()  # 这一批的平均间隔
            self.last_extra = {"acc": float(out["accuracy"]), "margin": float(margin.detach() if hasattr(margin, "detach") else margin)}  # 记账，张量先摘图
            return out["loss"], int(good.numel() + bad.numel())  # 真 token 数

        rows = self._next()  # 一组：候选与分数
        ids, rewards = rows  # 拆开
        batch = self._stack(ids)  # 同一组的候选并成一批
        logps = self._logp(batch)  # 每条的对数概率
        if self.ref_model is not None:  # GRPO 的 KL 要参考概率
            refs = self._logp_ref(batch)
        else:
            refs = logps.detach()
        if task == "grpo":  # 组相对
            out = alignment.grpo(logps, rewards, clip=self.choice.clip_ratio, kl_coef=self.choice.kl_coef, ref_logps=refs)
        elif task == "rloo":  # 留一基线
            out = alignment.rloo(logps, rewards)
        else:  # ppo，旧概率取自 detach（单步）
            out = alignment.ppo(logps, logps.detach(), alignment.grpo(logps, rewards, clip=self.choice.clip_ratio, kl_coef=self.choice.kl_coef, ref_logps=refs)["advantages"], clip=self.choice.clip_ratio)
        self.last_extra = {"reward": float(sum(rewards) / len(rewards))}  # 平均奖励给日志
        return out["loss"], int(batch.numel())  # 真 token 数

    @torch.no_grad()
    def _logp_ref(self, ids: torch.Tensor) -> torch.Tensor:
        """参考模型的对数概率，不建图。"""
        return self._logp_of(self.ref_model, ids)

    def _align_update(self) -> float:
        """对齐任务的一步：攒够累积再更新，收尾与语言模型那条路一致。"""
        total = 0.0  # 这一步攒到的损失
        tokens = 0  # 真 token 数
        for _ in range(self.choice.grad_accum):  # 逐个微批
            with self._context():  # 半精度上下文
                loss, count = self._align_loss()  # 对齐损失
            (loss / self.choice.grad_accum).backward()  # 除开累积数再反向
            total += float(loss.detach())
            tokens += count
        self.last_norm = float(torch.nn.utils.clip_grad_norm_(self.params, self.choice.clip))  # 裁剪
        self.last_tokens = tokens  # 交给健康检查
        for group in self.optim.param_groups:  # 学习率按这一步重算
            group["lr"] = self._rate(self.step)
        self.optim.step()  # 更新
        self.optim.zero_grad(set_to_none=True)  # 清梯度
        self.rng = torch.get_rng_state()  # 存随机数状态
        return total

    def _update(self) -> float:
        """跑一步：攒够累积再更新，返回这一步的平均损失。"""
        if self.choice.task in {"dpo", "grpo", "rloo", "ppo"}:  # 对齐任务另一条路
            return self._align_update()
        total = 0.0  # 这一步攒到的损失
        tokens = 0  # 这一步过了多少真 token，算吞吐用
        for _ in range(self.choice.grad_accum):  # 一个一个微批
            ids, labels = self._batch()  # 取微批
            with self._context():  # 半精度上下文
                if self.choice.task == "distill":  # 蒸馏换成交叉熵之外的目标
                    with torch.no_grad():  # 教师不建图
                        teach = self.teacher(ids)
                    loss = alignment.distill(self.model(ids), teach, self.choice.temperature).mean() / self.choice.grad_accum
                else:  # 普通语言模型目标
                    loss = cross_entropy(self.model(ids), labels) / self.choice.grad_accum  # 除开累积数，梯度才不跟着放大
            loss.backward()  # 反向
            total += float(loss.detach())  # 累进账
            tokens += int((labels != -100).sum())  # 记下这块有多少真 token
        self.last_norm = float(torch.nn.utils.clip_grad_norm_(self.params, self.choice.clip))  # 裁剪，返回的是裁剪前的总范数
        self.last_tokens = tokens  # 交给健康检查算吞吐
        for group in self.optim.param_groups:  # 学习率按这一步重算
            group["lr"] = self._rate(self.step)
        self.optim.step()  # 更新
        self.optim.zero_grad(set_to_none=True)  # 清梯度
        self.rng = torch.get_rng_state()  # 记下随机数状态，续训要用
        return total

    # --- 健康 ---

    def _health(self, loss: float, spent: float) -> bool:
        """看 loss、梯度范数、吞吐，返回这一趟该不该停。

        三项各自计连续步数，踩线分别在第 3、5 步和第 21 步之后才成立。
        处置照 train.health：warn 喊一声清计数接着跑，stop 把原因留在 self.reason 上。
        """
        bad = ""  # 这次踩线的原因
        self.bad_loss = self.bad_loss + 1 if not math.isfinite(loss) else 0  # NaN/Inf 连着数
        if self.bad_loss >= NAN_STEPS:
            bad = f"loss 连续 {NAN_STEPS} 步是 NaN/Inf"
        self.big_norm = self.big_norm + 1 if self.last_norm > NORM_LIMIT else 0  # 梯度炸了连着数
        if not bad and self.big_norm >= NORM_STEPS:
            bad = f"梯度范数连续 {NORM_STEPS} 步超 {NORM_LIMIT:g}（最近 {self.last_norm:.3g}）"
        self.speeds = (self.speeds + [self.last_tokens / max(spent, 1e-9)])[-SPEED:]  # 只留最近这几步的吞吐
        if not bad and len(self.speeds) == SPEED:  # 攒够 20 步基线才判得出掉速
            base = statistics.median(self.speeds[:-1])  # 前 20 步的中位数
            if self.speeds[-1] < base * SLOW_RATIO:  # 跌破四成
                bad = f"吞吐掉到前 20 步中位数的四成以下（{self.speeds[-1]:.0f} < {base * SLOW_RATIO:.0f} token/s）"
        if not bad:  # 都没踩线
            return False
        if self.choice.health == "warn":  # 只喊一声，计数清零免得每步重复喊
            if self.lead:
                logs.warn("spin", "健康检查", step=self.step, why=bad)
            self.bad_loss = self.big_norm = 0  # 重新数
            self.speeds = []  # 吞吐基线重新攒
            return False
        self.reason = bad  # 停机原因，命令层和 run.json 都靠它
        if self.lead:
            logs.warn("spin", "健康检查停机", step=self.step, why=bad)
        return True

    def _agree(self, stop: bool) -> bool:
        """多进程时统一停不停：各卡取到的批不一样，结论必须由 0 号说了算。

        不统一的后果是有人退出循环、有人接着推，反向里的集合通信当场僵住。
        """
        if self.world == 1:  # 单进程，自己的结论就是全部
            return stop
        flag = torch.tensor([int(stop and self.lead)], device=_device())  # 只有 0 号的判断算数
        torch.distributed.broadcast(flag, src=0)  # 发给所有进程
        return bool(flag.item())

    def run(self) -> int:
        """跑到总步数；中途 Ctrl-C 或健康检查叫停，都是存一档再退。"""
        self.out.mkdir(parents=True, exist_ok=True)  # 产物目录先建
        if self.lead:  # 结构描述和这一趟的设定只留一份
            self.net.write_config(self.out, self.choice.precision)  # 导出时能用
        if self.choice.resume:  # 续训先把状态读回来
            self.restore(Path(self.choice.resume))
        if self.lead:
            self._write_run()  # 回头知道跑的是什么
        try:
            while self.step < self.total:  # 一步一步推
                self.step += 1  # 先记步号，断在存盘前也对得上
                began = time.perf_counter()  # 这一步的起点，算吞吐用
                loss = self._update()  # 真跑一步
                spent = time.perf_counter() - began  # 这一步花了多久
                self.history = (self.history + [round(loss, 6)])[-HISTORY:]  # 只留最近几步
                if self.lead and (self.step % self.choice.log_every == 0 or self.step == 1):  # 到点报一次
                    logs.step("spin", "训练", step=self.step, total=self.total, loss=round(loss, 4), lr=f"{self._rate(self.step):.2e}", gpu_gb=round(self._vram(), 2))
                if self._agree(self._health(loss, spent)):  # 三项指标有一项踩线到要停，多进程要口径一致
                    break
                if self.choice.save_every and self.step % self.choice.save_every == 0:  # 到点存一档
                    _barrier()  # 等齐再存，谁也不许先走
                    self.save()
                    _barrier()  # 存完再散，0 号写盘时别人还在
                    if self.lead:  # 只有 0 号有档可报
                        logs.done("spin", "已存档", step=self.step)
        except KeyboardInterrupt:  # 手动掐断也不白跑
            _barrier()
            self.save()
            _barrier()
            if self.lead:
                logs.warn("spin", "已中断，断点存下了", step=self.step)
            _done()  # 先存完再拆组，fsdp2 的存盘要用组
            return self.step
        _barrier()  # 存盘要跨进程拼分片，先等齐
        self.save()  # 收尾必存；拆组放在它后面，分片得先拼回来
        _barrier()
        if self.lead and self.reason:  # 健康检查叫停的，原因写进 run.json，命令层照它退 1
            self._note_health()
            logs.done("spin", "健康停机", step=self.step, why=self.reason)
        elif self.lead:  # 正常收尾
            logs.done("spin", "训练完成", step=self.step, total=self.total)
        _done()  # 进程组收尾，单进程调了也没事
        return self.step

    @property
    def net(self):
        """真正的那摞模型。DDP 把模型套了一层壳，存盘和写结构都要剥回去，否则权重名会多个 module. 前缀。"""
        return self.model.module if isinstance(self.model, DDP) else self.model  # 没套壳就是自己

    def _vram(self) -> float:
        """卡上已用的峰值显存，GiB；没有卡就是 0。"""
        return torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0

    # --- 存读 ---

    def save(self) -> Path:
        """存一档：权重、优化器、状态。返回目录。"""
        folder = self.out / f"step-{self.step:06d}"  # 一档一个目录
        if self.choice.lora:  # 挂件只存 A/B；分片要每个进程一起拼，落盘只让 0 号动手
            lora.save(self.net, folder, self.choice.base, write=self.lead)
        else:  # 全量训就存全量
            self.net.save(folder, write=self.lead)
        if not self.lead:  # 其余进程拼完就走
            return folder
        self.net.write_config(folder, self.choice.precision)  # 结构描述跟着这一档走，导出时就在手边
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
            lora.load(self.net, folder)
        else:  # 全量训就灌全量
            self.net.load(folder)
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
            "train": {"steps": self.total, "batch_size": self.choice.batch_size, "grad_accum": self.choice.grad_accum, "lr": self.choice.lr, "seed": self.choice.seed, "precision": self.choice.precision, "strategy": self.choice.strategy, "devices": self.choice.devices, "grad_checkpoint": self.choice.grad_checkpoint, "health": self.choice.health},  # 训练超参
            "lora": self.choice.lora,  # 挂件配置，没挂是 null
            "files": [str(one) for one in self.choice.files],  # 数据来源
        }
        (self.out / "run.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    def _note_health(self) -> None:
        """把健康停机的原因补进 run.json，回头一眼知道这一趟为什么停。"""
        path = self.out / "run.json"  # 这一趟的设定就在这
        meta = json.loads(path.read_text(encoding="utf-8"))  # 读回来补一笔
        meta["health"] = {"action": self.choice.health, "reason": self.reason, "step": self.step}  # 处置、原因、停在哪步
        path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")  # 原样写回


def _device_of(model):
    """模型实际待着的设备。量化层的参数属性可能还写着 cpu，所以先看有没有真上卡。"""
    if torch.cuda.is_available() and getattr(_unwrap(model), "quantized", False):  # 量化整套已经在卡上了
        return torch.device("cuda", torch.cuda.current_device())
    return next(model.parameters()).device


def _unwrap(model):
    """拿回没被分布式包过的模型本体。"""
    return model.module if hasattr(model, "module") else model


def _teacher(choice, geo):
    """按 model.teacher 载一份教师，几何要与学生一致。"""
    from features import model as models

    if choice.teacher is None:  # 没给教师就做不了蒸馏
        fail("input", "distill 要给 model.teacher 指教师权重", where="model.teacher")
    net = models.Model(geo)  # 同构的学生结构
    where = Path(choice.teacher)  # 教师权重目录，给了产物根目录就跟着 latest 走
    if where.is_dir() and (where / "latest").is_file():
        where = where / (where / "latest").read_text(encoding="utf-8").strip()
    net.load(where)  # 灌教师的权重
    return net


def _setup() -> tuple:
    """不是 torchrun 起的就返回 (0, 1)；是就建进程组，本进程认领自己那张卡。"""
    if "LOCAL_RANK" not in os.environ:  # 普通单进程，一条分布式代码都不进
        return 0, 1
    local = int(os.environ["LOCAL_RANK"])  # 本机第几张卡，torchrun 排好的
    if torch.cuda.is_available():  # 卡上每进程认领一张，默认 0 号会让所有进程挤在一张上
        torch.cuda.set_device(local)
    torch.distributed.init_process_group("nccl" if torch.cuda.is_available() else "gloo")  # 卡上用 nccl，CPU 上用 gloo；rank 和总数从环境变量读
    return int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])  # 全局序号和总进程数


def _wrap(net, strategy: str, world: int, local: int):
    """按策略套分布式外壳。单进程或 single 原样返回，走的就是原来那条路。"""
    if strategy == "single" or world == 1:  # 一张卡不用包，包了反而多一层
        return net
    if strategy == "ddp":  # 每卡一份完整模型，反向时自动把所有进程的梯度平均
        return DDP(net, device_ids=[local] if torch.cuda.is_available() else None)
    for block in net.model.layers:  # fsdp2 逐层切：一层算完只留分片，全量参数不会同时驻留
        fully_shard(block)
    fully_shard(net)  # 最外层也切一刀，残留的参数和梯度才一起摊开
    return net


def _device() -> torch.device:
    """集合通信用哪个设备：卡上用当前卡，CPU 上用 CPU。nccl 只认卡上的张量。"""
    return torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")


def _barrier() -> None:
    """进程间对齐一下。没建组（单进程）就什么也不做。"""
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


def _done() -> None:
    """进程组收尾。没建过（单进程）就什么也不做。"""
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def _state_tuple(raw: list) -> tuple:
    """JSON 读回来的 python 随机数状态还原成元组。"""
    return (raw[0], tuple(raw[1]), raw[2] if len(raw) > 2 else None)  # 三段原样拼回


def cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """按 token 分块算交叉熵，跳掉 -100 的补位。"""
    flat_logits = logits.reshape(-1, logits.shape[-1])  # 拍平成 (token, 词表)
    flat_labels = labels.reshape(-1)  # 答案也拍平
    total, count = logits.new_zeros((), dtype=torch.float32), 0  # 总和要求跟 logits 同设备，卡上才不会加不上
    for start in range(0, flat_labels.numel(), CHUNK):  # 一块一块来
        part = flat_labels[start : start + CHUNK]  # 这块的答案
        keep = part != -100  # 这块里哪些是真 token
        if not bool(keep.any()):  # 整块都是补位就跳过
            continue
        total += F.cross_entropy(flat_logits[start : start + CHUNK][keep], part[keep], reduction="sum")  # 只算真 token
        count += int(keep.sum())  # 累计有效数
    return total / max(1, count)  # 平均，一块都没有时按 1 除
