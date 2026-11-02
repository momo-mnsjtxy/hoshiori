"""训练：读取文本与可选图片，拼批、前向、反向，并管理存档和本次任务的冻结模型。

调用方式：Trainer(model, geo, choice).run() 跑完整趟；image_side 大于零时，bolt 样本的 images
会缩放切块，image_at 可把各图插进正文；没给位置仍作为文本前缀，图片位置不计语言损失。
同批补出的零块由 block_mask 标出并随前向传递。组相对任务给 Choice.reward 时，每条候选会在
沙箱里调用 reward({prompt, response, ...元数据})，结果覆盖数据里的 rewards。
多卡、续训、健康检查和参考模型仍由同一实例管理。

调用示例：
  from features.trainer import Choice, Trainer                                  # 训练参数和主入口
  choice = Choice(out=Path("out/t"), files=[Path("bolt/bolt.jsonl")], steps=20)  # 二十步训练
  job = Trainer(model, geo, choice)                                             # 读取并编码全部样本
  job.run()                                                                     # 返回最终步数并释放冻结模型
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

from features import alignment, image, lora, sandbox
from features.model import sequence_positions
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
    base: Path = None  # 已载入的 model.path，参考模型重载和挂件存档共用；随机起步为 None
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
    max_images: int = 4  # 一条样本最多取几张图片；位置不动以兼容旧的位置参数
    reward: Path = None  # 组相对任务的外部奖励文件；给了就覆盖数据里的 rewards
    curriculum: bool = False  # 先学短样本，再随训练进度放开中、长样本


class Trainer:
    """一次训练。一个 job 用一个实例。"""

    def __init__(self, model, geo, choice: Choice) -> None:
        self.geo = geo  # 几何，编码和分块都要用
        self.choice = choice  # 这一趟的参数
        self.out = Path(choice.out)  # 产物目录
        self.rank, self.world = _setup()  # torchrun 起的进程有 LOCAL_RANK；自己起的单进程就是 0/1
        self.lead = self.rank == 0  # 只有 0 号进程存盘打日志，别的进程只算
        self.pad_id = getattr(choice.vocab, "pad_id", 0)  # 补位用词表的 pad；没词表还是老的 0
        self.ref_model = None  # dpo / grpo / ppo 的参考模型，只在 CPU 前向
        self.teacher = None    # distill 的教师，同样留在 CPU
        self.samples = self._read()  # 编好的样本，一次读进来，别边跑边读盘
        self.total = choice.steps or max(1, math.ceil(choice.epochs * len(self.samples) / (choice.batch_size * choice.grad_accum)))  # 没给步数就按轮数换算
        model.grad_checkpoint = choice.grad_checkpoint  # 前向按配方决定要不要重算
        if choice.lora:  # 挂低秩
            lora.attach(model, choice.lora.get("target") or list(lora.TARGETS)[:4], choice.lora["r"], choice.lora["alpha"], choice.lora["dropout"])  # 挂件以外全冻上
        if choice.lora:  # 挂件是新建的层，建在 CPU，得跟基座放一起
            model.to(_device_of(model))
        if choice.task in {"dpo", "grpo", "ppo"}:                         # 这三类要一份不动的参考模型
            if choice.base is None:                                    # 命令层没载入基座，这一趟从随机权重起
                self.ref_model = copy.deepcopy(_unwrap(model)).cpu()    # 只在随机起步时拷贝学生，随后搬到 CPU
                if self.lead:                                          # 多进程只记一份日志
                    logs.warn("spin", "无已载入的 model.path，参考模型拷贝学生并放 CPU")  # 明说本次走了拷贝
            else:                                                      # 基座已经由命令层确认可用
                self.ref_model = _load_cpu(choice.base, geo)            # 按同一份几何重建，再从基座读权重
        if choice.task == "distill":                                    # 教师路径已由命令层检查
            self.teacher = _load_cpu(choice.teacher, geo)               # 教师也独立构建和载入
        for net in (self.ref_model, self.teacher):                      # 两类冻结模型共用同一套设置
            if net is not None:                                        # 普通训练没有额外模型
                net.eval()                                             # 关闭训练行为
                net.requires_grad_(False)                              # 所有参数都不参与反向
        self.model = _wrap(model, choice.strategy, self.world, self.rank)  # 多卡才包，单卡原样
        self.params = [one for one in self.model.parameters() if one.requires_grad]  # 包完再取一遍：fsdp2 会把参数换成新的
        self.optim = torch.optim.AdamW(self.params, lr=choice.lr, weight_decay=choice.weight_decay)  # 优化器只管可训的
        self.step = 0  # 已经跑了几步
        self.cursor = 0  # 数据游标，全局第几条
        self.history = []  # 最近的 loss
        self._perm = []  # 当前这轮的取样顺序
        self._buckets = []  # 课程采样的短、中、长三档；只活在内存，不进断点
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

    def _image_positions(self, text: str, positions: list) -> list:
        """把正文字符下标换成 ids 下标；真词表的 ids_of 会补齐，位置计算不能直接取它的长度。"""
        out = []                                                                  # 依图片顺序攒分词后的锚点
        for at in positions:                                                      # 样本入口已经保证字符下标严格递增
            prefix = text[: at + 1]                                               # image_at 表示跟在这个字符之后
            if self.choice.vocab is not None:                                     # 真词表先编码内容，不带 </s> 与 pad
                tokens = self.choice.vocab.encode(prefix)                         # ids_of 总是补到 max_seq_len，不能用于定位
                anchor = min(len(tokens) - 1, self.geo.max_seq_len - 2)           # 给 </s> 留出的最后一格不算正文位置
            else:                                                                # 无词表时一个 UTF-8 字节就是一个兜底 token
                size = min(len(prefix.encode("utf-8")), self.geo.max_seq_len)     # 与 _ids 的截断规则一致
                anchor = size - 1                                                 # 最后一个前缀 token 的下标
            if out:                                                              # BPE 加长前缀后可能重新合并成更少 token
                anchor = max(anchor, out[-1])                                     # 图片顺序仍按递增字符位置保持非递减
            out.append(anchor)                                                   # 同一 token 边界允许承载相邻两张图
        return out                                                               # _pictures 再按每图块数展开

    def _read_text(self) -> list:
        """把正文编码成编号；图片几何开启时，同条保存图片路径及可选的交错位置。"""
        out = []                                                                  # 攒起来的样本
        for one in self.choice.files:                                             # 一个文件一趟
            for line in Path(one).read_text(encoding="utf-8").splitlines():       # 一行一条样本
                if not line.strip():                                              # 空行没有训练内容
                    continue
                item = json.loads(line)                                           # bolt 里正文和图片同属一条记录
                text = item.get("text") or ""                                    # 图片字符位置与正文使用同一份来源
                ids = self._ids(text)                                             # 正文沿用唯一的编码入口
                if len(ids) <= 1:                                                 # 至少要有输入和答案
                    continue
                if self.geo.image_side:                                           # 图片模型还要留住对应路径
                    here = Path(one).parent                                          # 相对图片路径按数据文件旁边算
                    raw = (item.get("images") or [])[: self.choice.max_images]        # 上限之外的不拿
                    paths = [str(here / path) if not Path(path).is_absolute() else path for path in raw]  # 补齐成能直接打开的路径
                    image_at = None                                               # 没给位置时保留旧的整段图片前缀
                    if "image_at" in item:                                       # 样本入口已经保证一张图对应一个递增字符下标
                        chars = item["image_at"][: len(raw)]                     # 与 max_images 一起截，图片和位置不拆对
                        image_at = self._image_positions(text, chars)             # 字符下标换成实际分词后的 ids 下标
                    out.append((ids, paths, image_at))                            # 编号、路径和位置仍保持同条关系
                else:                                                            # 纯文本配方沿用原来的样本结构
                    out.append(ids)                                               # 下游继续拿一串编号
        return out                                                               # 一次读完，训练时不再碰数据文件

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
                if len(answers) < 2:  # 不足两条算不出组内优势，也不用白跑外部奖励
                    continue
                rewards = self._rewards(item, answers)  # 外部奖励给了就覆盖原列，否则照读数据
                if len(rewards) != len(answers):  # 数据自带奖励时，分数仍要与候选一一对应
                    continue
                out.append(([self._ids(prompt + one) for one in answers], rewards))  # 编码后的候选与分数
        return out

    def _rewards(self, item: dict, answers: list) -> list:
        """取得一组分数；外部奖励逐候选运行，payload 含 prompt、response 与该行其余元数据。"""
        if self.choice.reward is None:                                            # 没配外部代码就沿用数据里的奖励列
            return (item.get("rewards") or [])[: len(answers)]                    # 分数数量稍后与候选统一核对
        common = {key: value for key, value in item.items() if key not in {"responses", "rewards"}}  # 旧候选和旧分数不送入覆盖逻辑
        common["prompt"] = item.get("prompt") or ""                              # 没写题面也给奖励函数稳定字段
        scores = []                                                               # 按候选顺序攒新奖励
        for answer in answers:                                                    # 一条候选一次隔离进程，彼此也不共享状态
            payload = {**common, "response": answer}                              # 当前答案用单数字段交给 reward(payload)
            score, error = sandbox.run(self.choice.reward, payload)               # 外部异常、超时、超内存都变成 error
            if error:
                fail("input", error, where=str(self.choice.reward), hint="修正 train.reward 指向的 reward(payload) 后重试")
            scores.append(score)                                                  # 只有有限数字能走到这里
        return scores                                                             # 完整覆盖这一行原有 rewards

    def _length(self, sample) -> int:
        """一条训练样本的序列长度；成对或成组任务取其中最长的候选。"""
        if self.choice.task == "dpo":                                   # 偏好对的两边都要前向
            return max(len(one) for one in sample)                       # 较长一边代表这对的训练开销
        if self.choice.task in {"grpo", "rloo", "ppo"}:               # 一组候选一起参与相对目标
            return max(len(one) for one in sample[0])                    # 最长候选决定补齐宽度
        if self.geo.image_side:                                          # 图片样本是（文本编号，图片路径，交错位置）
            return len(sample[0])                                        # 分档只看正文序列长度
        return len(sample)                                                # 普通正文就是一串编号

    def _shuffle(self, epoch: int) -> None:
        """生成一轮顺序；课程模式按长度的 1/3、2/3 经验分位分成短、中、长三档。

        分位采用最近秩：排序后第 ceil(n/3) 与 ceil(2n/3) 条的长度是两道边界，
        等于边界的样本归较短档。这样依据只来自本轮样本长度，同种子续训可原样重建。
        """
        gen = torch.Generator().manual_seed(self.choice.seed * 100003 + epoch * 7 + self.rank)  # 种子掺进轮号和进程号，几路进程才取到不同的条
        if not self.choice.curriculum:                                    # 默认关闭时保留原来的整轮洗牌
            self._perm = torch.randperm(len(self.samples), generator=gen).tolist()  # 整轮打乱
            return                                                       # 不建立课程桶，旧顺序一位不变
        lengths = [self._length(one) for one in self.samples]            # 每条样本只算一次长度
        ordered = sorted(lengths)                                        # 经验分位按长度次序取最近秩
        short = ordered[math.ceil(len(ordered) / 3) - 1]                 # 至少三分之一落在短档边界内
        middle = ordered[math.ceil(len(ordered) * 2 / 3) - 1]            # 至少三分之二落在中档边界内
        buckets = [[], [], []]                                           # 依次装短、中、长样本的下标
        for index, length in enumerate(lengths):                         # 每条只进一个档，边界不会漏样本
            level = 0 if length <= short else 1 if length <= middle else 2  # 等于分位边界归较短档
            buckets[level].append(index)                                 # 留下原样本下标，不复制训练数据
        for level, bucket in enumerate(buckets):                         # 三档各自打乱，档内不按长度排队
            order = torch.randperm(len(bucket), generator=gen).tolist()  # 同一种子与轮次得到同一顺序
            buckets[level] = [bucket[at] for at in reversed(order)]      # 反放后可从末尾常数时间取下一条
        self._buckets = buckets                                          # 本轮消费状态只留在内存

    @staticmethod
    def _level(step: int, total: int) -> int:
        """按 step / total 把训练进度分成等宽的短、中、长三段。"""
        progress = step / max(1, total)                                  # 断点已有这两个数，不另存课程档位
        if progress <= 1 / 3:                                            # 前三分之一只把短档当上限档
            return 0                                                     # 短档
        if progress <= 2 / 3:                                            # 中间三分之一放开到中档
            return 1                                                     # 中档
        return 2                                                         # 最后三分之一放开到长档

    def _take(self, level: int) -> int:
        """从当前上限档取下标；该档空了先往短档找，允许档全空才向长档补齐本轮。"""
        levels = [*range(level, -1, -1), *range(level + 1, 3)]            # 上限内从长到短，再扫尚未开放的余档
        for one in levels:                                               # 最多看短、中、长三档
            if self._buckets[one]:                                       # 这一档还有本轮未取的样本
                return self._buckets[one].pop()                          # 每个下标只弹出一次，不会重复
        raise RuntimeError("课程采样桶提前耗尽")                         # cursor 未走满一轮时不该到这里

    def _next(self) -> list:
        """取下一批，取完一轮自动换顺序。"""
        pos = self.cursor % len(self.samples)  # 这一轮里的位置
        if pos == 0:  # 走到轮首就重排
            self._shuffle(self.cursor // len(self.samples))
        if not self.choice.curriculum:  # 默认关闭时走原来的取样路径
            self.cursor += 1  # 游标往后挪
            return self.samples[self._perm[pos]]  # 按原来的整轮顺序取一条
        level = self._level(self.step, self.total)                        # 档位只由当前进度推出来
        index = self._take(level)                                        # 从当前可学的上限档开始找
        self.cursor += 1                                                  # 成功取到以后再推进全局游标
        return self.samples[index]                                       # 交出原样本

    def _batch(self) -> tuple:
        """凑一批；图片批次再返回 patches、真实块 mask 与可选交错位置。"""
        rows = [self._next() for _ in range(self.choice.batch_size)]              # 取够一个微批
        if not self.geo.image_side:                                               # 没开图片几何就走原来的两张量路径
            widest = max(len(one) for one in rows) - 1                            # 最长的输入长度
            ids = torch.full((len(rows), widest), self.pad_id, dtype=torch.long)   # 输入右侧补词表 pad
            labels = torch.full((len(rows), widest), -100, dtype=torch.long)       # 答案补位不参与损失
            for at, one in enumerate(rows):                                       # 一条条填入文本
                ids[at, : len(one) - 1] = torch.tensor(one[:-1])                   # 前 n-1 个当输入
                labels[at, : len(one) - 1] = torch.tensor(one[1:])                 # 后 n-1 个当答案
            here = _device_of(self.model)                                         # 模型在哪张卡，文本就去哪
            return ids.to(here), labels.to(here)                                  # 保留旧接口的两个返回值

        widest = max(len(one[0]) for one in rows) - 1                              # 图片样本里最长的文本输入
        patches, block_mask, patch_at = self._pictures(rows, widest)              # 图片按批内最多块数补齐，并留下真实块范围
        image_steps = 0 if patches is None else patches.shape[1]                  # 整批无图时不增加任何图片位
        ids = torch.full((len(rows), widest), self.pad_id, dtype=torch.long)        # 文本输入仍只补文本位
        labels = torch.full((len(rows), image_steps + widest), -100, dtype=torch.long)  # 图片位无论前缀或交错都不算损失
        text_at = None if patch_at is None else sequence_positions(widest, patch_at)[0]  # 交错标签复用模型的唯一布局算法
        for at, (tokens, _, _) in enumerate(rows):                                 # 图片路径和位置已经交给上一步处理
            ids[at, : len(tokens) - 1] = torch.tensor(tokens[:-1])                 # 前 n-1 个文本号当输入
            start = image_steps if text_at is None else 0                          # 前缀标签是一段连续后缀，交错标签按位置散放
            places = slice(start, start + len(tokens) - 1) if text_at is None else text_at[at, : len(tokens) - 1]  # 当前文本各答案位
            labels[at, places] = torch.tensor(tokens[1:])                          # 图片位置保持 -100，文本答案顺序不变
        here = _device_of(self.model)                                              # 文本和标签跟随学生模型
        block_mask = None if block_mask is None else block_mask.to(here)            # 注意力遮罩跟图片放到同一设备
        patch_at = None if patch_at is None else patch_at.to(here)                  # 交错下标也跟随模型设备
        return ids.to(here), labels.to(here), patches, block_mask, patch_at          # 五项直接交给模型前向

    def _pictures(self, rows: list, text_steps: int) -> tuple:
        """切好一批图片，右补零块，并给出真实块 mask 与按块展开的插入位置。"""
        prepared = []                                                              # 每条样本各自串好的图片块
        counts = []                                                                # 每条实际有多少块
        positions = []                                                             # 每张图的位置按它的块数展开
        interleaved = False                                                        # 整批都没 image_at 时保留旧前缀路径
        for _, paths, image_at in rows:                                            # 文本已在同条里，此处只管图片
            pieces = []                                                            # 一条样本可能带多张图片
            block_at = []                                                          # 同一张图的全部块共享一个文本锚点
            count = 0                                                              # 多张图的块数累加
            interleaved = interleaved or image_at is not None                      # 混合批次也要为带位置的样本启用交错
            for index, path in enumerate(paths):                                   # 路径顺序就是图片顺序
                blocks, size = image.prepare(path, self.geo.image_side, self.geo.patch)  # 一张图缩放、归一、切块
                pieces.append(torch.from_numpy(blocks))                            # float32 数组先转 CPU 张量
                count += size                                                      # 记下该样本的真实图片位
                anchor = -1 if image_at is None else image_at[index]               # 旧图片用内部 -1 表示仍在文本前
                block_at.extend([anchor] * size)                                   # 每块都带着所属图片的同一位置
            joined = torch.cat(pieces) if pieces else torch.empty((0, self.geo.patch, self.geo.patch, 3))  # 多图首尾相接
            prepared.append(joined)                                                # 留待按批内最大值补齐
            counts.append(count)                                                   # 右侧补多少由这个数决定
            positions.append(block_at)                                             # 与 joined 的块顺序逐项对应
        widest = max(counts)                                                       # 同批模型前缀必须等长
        if widest == 0:                                                            # 整批都没图片时不增加任何前缀
            return None, None, None                                                # 模型仍走无图前向
        batch = torch.zeros((len(rows), widest, self.geo.patch, self.geo.patch, 3), dtype=torch.float32)  # 零块铺底
        block_mask = torch.zeros((len(rows), widest), dtype=torch.bool)             # 1 是真实块，0 是右侧补位块
        patch_at = torch.full((len(rows), widest), text_steps, dtype=torch.long) if interleaved else None  # 补位块放到文本后，不挪动文本位置
        for at, blocks in enumerate(prepared):                                     # 每条从左边放真实图片块
            batch[at, : counts[at]] = blocks                                      # 剩下的右侧保持零块
            block_mask[at, : counts[at]] = True                                   # 同一段标成可参与注意力和对齐的真实块
            if patch_at is not None and counts[at]:                               # 交错批次把每个真实块放回样本指定位置
                patch_at[at, : counts[at]] = torch.tensor(positions[at])           # 旧前缀样本的真实块在这里写成 -1
        here = _device_of(self.model)                                              # 图片跟学生模型放在同一设备
        dtype = self.geo.dtype(self.choice.precision)                              # 图片投影使用模型精度
        return batch.to(device=here, dtype=dtype), block_mask, patch_at             # mask 和位置由 _batch 随文本一起搬设备

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
        """在指定网络的设备上算对数概率，结果回到输入设备；每条一个数，补位不算。"""
        here = ids.device                                         # 留住学生批次的设备，结果要搬回来
        ids = ids.to(next(net.parameters()).device)                # 批次跟实际前向的网络走，参考模型就在 CPU
        logits = net(ids[:, :-1])                                  # 前向，最后一位没有答案
        picked = F.log_softmax(logits.float(), dim=-1).gather(2, ids[:, 1:].unsqueeze(-1)).squeeze(-1)  # 只取答案那一位
        keep = ids[:, 1:] != self.pad_id                           # 补位不算
        logps = (picked * keep).sum(dim=1) / keep.sum(dim=1).clamp(min=1)  # 按真 token 平均
        return logps.to(here)                                      # 只传每条的概率，不把整份 logits 搬上卡

    def _align_loss(self) -> tuple:
        """对齐任务的一步损失，返回（损失, 真 token 数）。"""
        task = self.choice.task  # 哪个任务族
        if task == "dpo":  # 偏好对，一步取 batch_size 对
            rows = [self._next() for _ in range(self.choice.batch_size)]  # 一批对
            good = self._stack([one[0] for one in rows])  # 正例
            bad = self._stack([one[1] for one in rows])  # 负例
            logp_c, logp_r = self._logp(good), self._logp(bad)  # 当前策略
            ref_c, ref_r = self._logp_ref(good), self._logp_ref(bad)  # 构建时已有冻结参考模型，概率回到学生设备
            out = alignment.dpo(logp_c, logp_r, ref_c, ref_r, self.choice.beta)  # DPO 损失
            margin = out["margin"].mean()  # 这一批的平均间隔
            self.last_extra = {"acc": float(out["accuracy"]), "margin": float(margin.detach())}  # 记账，张量先摘图
            return out["loss"], int(good.numel() + bad.numel())  # 真 token 数

        rows = self._next()  # 一组：候选与分数
        ids, rewards = rows  # 拆开
        batch = self._stack(ids)  # 同一组的候选并成一批
        logps = self._logp(batch)  # 每条的对数概率
        refs = logps.detach() if task == "rloo" else self._logp_ref(batch)  # RLOO 不要参考模型，其余取 CPU 算回的概率
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
        """CPU 参考模型不建图，概率回到学生批次所在设备。"""
        return self._logp_of(self.ref_model, ids)  # 输入和结果的搬运统一在概率计算里完成

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
            if self.geo.image_side:                                               # 图片几何的批次还带五维图片块
                ids, labels, patches, block_mask, patch_at = self._batch()         # 图片布局和真实块范围已经反映在 labels 里
            else:                                                                # 纯文本保留原来的两张量接口
                ids, labels = self._batch()                                       # 取文本微批
            with self._context():  # 半精度上下文
                if self.choice.task == "distill":  # 蒸馏换成交叉熵之外的目标
                    with torch.no_grad():  # 教师不建图
                        if self.geo.image_side:                                   # 教师也要看到同一批图片及其布局
                            teacher_patches = None if patches is None else patches.float().cpu()  # CPU 教师使用 fp32 图片
                            teacher_mask = None if block_mask is None else block_mask.cpu()  # 补位范围原样搬到 CPU
                            teacher_at = None if patch_at is None else patch_at.cpu()         # 交错位置也与学生完全相同
                            teach = self.teacher(ids.cpu(), patches=teacher_patches, block_mask=teacher_mask, patch_at=teacher_at).to(ids.device)  # 打分送回学生设备
                        else:                                                    # 纯文本蒸馏保持原来的调用
                            teach = self.teacher(ids.cpu()).to(ids.device)         # 教师只在 CPU 前向
                    if self.geo.image_side:                                       # 学生的图片块已经对齐设备和精度
                        scores = self.model(ids, patches=patches, block_mask=block_mask, patch_at=patch_at)  # 图片与文本按同一布局产生蒸馏打分
                    else:                                                        # 没有图片几何不传新增参数
                        scores = self.model(ids)                                  # 保留纯文本前向
                    loss = alignment.distill(scores, teach, self.choice.temperature).mean() / self.choice.grad_accum  # 按累积数缩放
                else:  # 普通语言模型目标
                    if self.geo.image_side:                                       # 图片块可作前缀，也可按样本位置交错
                        scores = self.model(ids, patches=patches, block_mask=block_mask, patch_at=patch_at)  # 返回图片位和文本位的全部打分
                    else:                                                        # 纯文本仍调用原接口
                        scores = self.model(ids)                                  # 输出长度与 ids 相同
                    aux = 0.0                                                     # 图文对齐那一路，不开就是 0
                    if isinstance(scores, tuple):                                 # align_weight 大于 0 时模型多交回一项
                        scores, aux = scores                                      # 拆开：打分与对齐损失
                    loss = (cross_entropy(scores, labels) + aux) / self.choice.grad_accum  # -100 图片位和补位都会跳过
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

    # --- 运行与收尾 ---

    def run(self) -> int:
        """跑到总步数；正常结束、中断或出错都释放本次任务的冻结模型。"""
        try:                      # 训练与存档先走完，再收尾
            return self._body()   # Ctrl-C 和健康停机仍由训练主线存档
        finally:                  # 出错也释放资源，错误原样交回入口
            self.close()          # 不把参考模型或教师留到下一次任务

    def close(self) -> None:
        """释放参考模型、教师和进程组；只构建而不运行的任务也可直接调用。"""
        del self.ref_model, self.teacher           # 去掉任务持有的冻结模型
        self.ref_model = self.teacher = None       # 留下已释放状态，重复收尾也无需检查
        torch.cuda.empty_cache()                  # 归还已不用的 CUDA 缓存，CPU 环境下无操作
        _done()                                   # 存档已结束，分片用的进程组可以拆掉

    def _body(self) -> int:
        """训练主线；中途 Ctrl-C 或健康检查叫停，都是存一档再退。"""
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
            return self.step
        _barrier()  # 存盘要跨进程拼分片，先等齐
        self.save()  # 收尾必存；拆组放在它后面，分片得先拼回来
        _barrier()
        if self.lead and self.reason:  # 健康检查叫停的，原因写进 run.json，命令层照它退 1
            self._note_health()
            logs.done("spin", "健康停机", step=self.step, why=self.reason)
        elif self.lead:  # 正常收尾
            logs.done("spin", "训练完成", step=self.step, total=self.total)
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
        self.history = state["history"]  # 曲线接上
        self.total = self.step + self.choice.steps if self.choice.steps else state["total"]  # 给了 5 步就是再跑 5 步，没给就照旧档的总数
        self.total = max(self.total, self.step)  # 别把总步数改到断点之前
        self._shuffle(self.cursor // len(self.samples))                    # 这一轮的随机顺序按轮号原样重建
        used = self.cursor % len(self.samples)                             # 只需重放当前轮已经取过的部分
        if self.choice.curriculum and used:                               # 课程桶还要恢复各档消费到哪里
            per_step = self.choice.grad_accum                             # 对齐任务每个累积块取一组
            if self.choice.task not in {"dpo", "grpo", "rloo", "ppo"}:  # 语言模型每个累积块还含一个微批
                per_step *= self.choice.batch_size                        # 一步实际取 batch_size × grad_accum 条
            first = self.cursor - used                                    # 当前轮第一条的全局游标
            for at in range(first, self.cursor):                          # 用旧总步数重演断点前的课程选择
                old_step = at // per_step + 1                             # 第 at 条属于哪一个已完成训练步
                self._take(self._level(old_step, state["total"]))        # 只弹下标，不重复读取或训练
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
            "train": {"steps": self.total, "batch_size": self.choice.batch_size, "grad_accum": self.choice.grad_accum, "lr": self.choice.lr, "seed": self.choice.seed, "precision": self.choice.precision, "strategy": self.choice.strategy, "devices": self.choice.devices, "grad_checkpoint": self.choice.grad_checkpoint, "health": self.choice.health, "curriculum": self.choice.curriculum},  # 训练超参
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


# --- CPU 权重 ---

def _load_cpu(source, geo):
    """按学生几何独立构建 CPU 模型，从基座或教师路径载入；产物根目录跟随 latest。"""
    from features.model import Model                                        # 基座和教师用同一种模型结构

    where = Path(source)                                                    # 路径由命令层整理好，内部直接使用
    if where.is_dir() and (where / "latest").is_file():                       # 产物根目录指向最近一档
        where = where / (where / "latest").read_text(encoding="utf-8").strip()  # 跟着已有存档指针找到权重
    with torch.device("cpu"):                                               # 构造时就落 CPU，避免临时占卡
        net = Model(geo)                                                    # 默认 fp32，不复制学生的量化层或挂件
    net.load(where)                                                         # 从盘上逐份载入，不接触学生参数
    return net                                                              # 冻结和 eval 由任务统一设置


def _setup() -> tuple:
    """不是 torchrun 起的就返回 (0, 1)；是就建进程组，本进程认领自己那张卡。"""
    if "LOCAL_RANK" not in os.environ:  # 普通单进程，一条分布式代码都不进
        return 0, 1
    local = int(os.environ["LOCAL_RANK"])  # 本机第几张卡，torchrun 排好的
    if torch.cuda.is_available():  # 卡上每进程认领一张，默认 0 号会让所有进程挤在一张上
        cards = torch.cuda.device_count()  # 几张卡
        torch.cuda.set_device(local % cards if cards else 0)  # 进程比卡多时绕回来（单机用 devices:[0,0] 验分布式就是这种）
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
