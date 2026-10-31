"""配方：训练要什么全写在一份 yaml 里。这个文件负责读它、合并父配方、按白名单验它。

调用示例：
  from features import recipe
  data = recipe.load(Path("sft.warp.yaml"))   # 读并展开 extends 链，循环会当场报错
  recipe.check(data)                          # 验字段名、类型、必备项
  recipe.stamp(data)                          # 拿一个能锁死的哈希

配方长这样（只列几项，全部字段见 FIELDS）：
  task: sft
  model: {path: ../base}
  train: {strategy: fsdp2, devices: [0, 1], lora: {r: 16}}

为什么字段名要白名单：拼错的字段会被静默忽略，跑完才发现那行没生效。
白名单里查不到就报错，并从最近似的几个名字里挑候选提示，改起来一眼就知道改成什么。
"""

import difflib
import hashlib
import json
from pathlib import Path

from utils import yaml_io
from utils.fail import fail

TASKS = {"pretrain", "sft", "distill", "dpo", "grpo", "ppo", "rloo"}  # 认的任务
STRATEGIES = {"single", "ddp", "fsdp2"}  # 认的多卡方式
PRECISIONS = {"bf16", "fp16", "fp32", "fp8", "nf4", "int8"}  # 认的精度
FORMATS = {"safetensors", "gguf", "onnx", "mlx"}  # 认的导出格式

# 允许出现的字段，按节写。(类型, 允许取值) 表示枚举；写 dict 表示还有下一层。
FIELDS = {
    "version": int,
    "extends": str,
    "name": str,
    "task": (str, TASKS),
    "model": {"path": str, "dtype": str, "params_b": float},
    "data": {"files": list, "warp": str, "fingerprint": str, "fields": list, "min_chars": int, "max_chars": int},
    "train": {
        "strategy": (str, STRATEGIES),
        "devices": list,
        "precision": (str, PRECISIONS),
        "batch_size": int,
        "grad_accum": int,
        "epochs": (int, float),
        "lr": float,
        "warmup_steps": int,
        "weight_decay": float,
        "max_grad_norm": float,
        "seed": int,
        "grad_checkpoint": bool,
        "lora": {"r": int, "alpha": int, "dropout": float, "target": (list, ())},
    },
    "eval": {"suites": list, "gate": {"min_delta": float}, "limit": int},
    "forge": {"formats": (list, FORMATS), "quant": list, "out": str},
}

FINE_TUNE = {"sft", "distill", "dpo", "grpo", "ppo", "rloo"}  # 这几类都要基座


def load(path: Path, seen: tuple = ()) -> dict:
    """读配方并展开 extends。子覆盖父，映射逐层并，列表整体替换。"""
    path = path.resolve()  # 统一成绝对路径，去重和比较都靠它
    if path in seen:  # 绕回读过的文件，说明 extends 成圈了
        fail("schema", "extends 绕成圈了", where=" → ".join(str(p.name) for p in (*seen, path)))
    data = yaml_io.read(path)  # 读这一份，坏 yaml 在这里就停
    parent = data.get("extends")  # 有没有父配方
    if not parent:  # 没有就是根，直接交出去
        return data
    parent_path = (path.parent / parent).resolve()  # 按当前文件的位置找父配方，换目录启动也找得到
    if not parent_path.exists():
        fail("input", f"extends 指的父配方不存在: {parent}", where=str(path), hint=f"放到 {parent_path}，或�� extends")
    return _merge(load(parent_path, (*seen, path)), data)


def _merge(base: dict, over: dict) -> dict:
    """父配方打底，子的覆盖上去。"""
    out = dict(base)  # 拷一份，不动传进来的父配方
    for key, value in over.items():
        if key == "extends":  # 已经展开过，不再往下传
            continue
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)  # 两边都是映射就继续往里并
        else:
            out[key] = value  # 其余整体替换，列表不拼
    return out


def check(data: dict) -> None:
    """验一份配方。先验字段，再验跨字段的必备关系。"""
    _walk(data, FIELDS)  # 字段名和类型先过一遍
    if data.get("task") in FINE_TUNE and not (data.get("model") or {}).get("path"):
        fail("schema", "微调类任务要 model.path 指基座", where="model.path", hint="例: ../merged_model")
    train = data.get("train") or {}  # 没写就当空，下面逐项看
    if train.get("strategy") in {"ddp", "fsdp2"} and not train.get("devices"):
        fail("schema", "多卡不给 devices 就不知道该用哪几张", where="train.devices", hint="例: [0, 1]")
    if (data.get("forge") or {}).get("formats") and not (data.get("forge") or {}).get("out"):
        fail("schema", "给了导出格式就要给导出目录", where="forge.out", hint="例: out/gguf")


def _walk(node: dict, spec: dict, where: str = "") -> None:
    """按白名单逐层走。字段不认识、类型不对都报错并指出位置。"""
    if not isinstance(node, dict):  # 这一节写成了一串或一个数
        fail("schema", "这一节要写成 键: 值", where=where or "<顶层>")
    unknown = [key for key in node if key not in spec]  # 白名单里没有的字段
    if unknown:
        near = difflib.get_close_matches(unknown[0], list(spec), n=3, cutoff=0.5)
        fail("schema", f"不认识的字段 {unknown[0]}", where=where or "<顶层>", hint="候选: " + " / ".join(near) if near else "可用: " + " / ".join(sorted(spec)[:8]))
    for key, value in node.items():  # 逐字段走
        here = f"{where}.{key}" if where else key  # 位置串，报错直接定位到行
        if isinstance(spec[key], dict):  # 还有下一层就继续往下
            _walk(value, spec[key], here)
            continue
        _expect(value, spec[key], here)  # 到底了，验值


def _expect(value, want, where: str) -> None:
    """验一个值的类型。写了允许取值的再逐个验取值。

    want 有四种写法：单类型 int、(类型, 取值集合)、(类型, 另一个类型) 表示两种都收。
    """
    name = where.split(".")[-1]  # 只报字段名，前面那串看 where 那行
    if not isinstance(want, tuple):  # 单类型
        if not isinstance(value, want):
            fail("schema", f"{name} 类型不对", where=where, hint=f"要 {want.__name__}")
        return

    kind, allowed = want  # 前半是类型，后半要么是取值集合，要么是另一个类型
    if isinstance(allowed, type):  # (int, float) 这种写法，两种都收
        if not isinstance(value, (kind, allowed)):  # 命中一个就算过
            fail("schema", f"{name} 类型不对", where=where, hint=f"要 {kind.__name__} 或 {allowed.__name__}")
        return
    if not isinstance(value, kind):  # 类型先过，不是这个类型直接停
        fail("schema", f"{name} 类型不对", where=where, hint=f"要 {kind.__name__}")
    if not allowed:  # 没给取值清单，类型对了就算过
        return
    picks = value if isinstance(value, list) else [value]  # 列表型逐个查，标量包成一项
    bad = [one for one in picks if one not in allowed]  # 挑出不在清单里的
    if bad:  # 报第一个就够，改完再验会发现下一个
        fail("schema", f"{name} 的取值不认识: {bad[0]}", where=where, hint="可用: " + " / ".join(sorted(allowed)))


def stamp(data: dict) -> str:
    """配方的哈希。同一份内容换个键序也一样，改一个值就变。"""
    body = json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(body.encode()).hexdigest()
