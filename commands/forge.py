"""命令 forge：把权重导成能带走的产物，safetensors 或 GGUF，可带量化档。

调用示例：
  ori forge sft.warp.yaml --out out/gguf --format gguf --quant q8_0
  ori forge sft.warp.yaml --out out/full.gguf --format gguf            # 不量化，fp16
  ori forge sft.warp.yaml --out out --format gguf --quant q5_k_m --quant q8_0
  ori forge sft.warp.yaml --out out/merged --format safetensors

检查只在这里做一次：格式认不认、量化档认不认、--out 给没给、权重在不在。
剩下的转换全交给 features.forge，命令层不碰字节。
onnx 和 mlx 要额外的包，缺了就按缺包报 exit 4，说清缺哪个，别硬转出来骗人。
配方的 model.path 指向带 adapter_config.json 的目录时，先合并再导，合并交给 features.merge。
"""

import importlib.util
import json
from pathlib import Path

from features import recipe
from utils.fail import fail

FORMATS = {"safetensors", "gguf", "onnx", "mlx"}  # 认的四类产物
NEED = (("numpy", "读写权重数组"), ("safetensors", "读写 safetensors"))  # 转换必备的两个包
SUFFIX = {"safetensors": ".safetensors", "gguf": ".gguf"}  # 能真转的两类各自的扩展名
BACKEND = {"onnx": "onnx", "mlx": "mlx"}  # 转不了的两类各自缺的包


def command(args) -> int:
    """验配方、定目标、合并 adapter、逐个格式导。"""
    data = recipe.load(Path(args.recipe))  # 读配方，坏 yaml 在这里停
    recipe.check(data)  # 字段和必备项先过
    for need, why in NEED:  # 缺包先拦，别在半路 import 崩
        if importlib.util.find_spec(need) is None:
            fail("missing", f"导出要 {need}", hint=f"pip install {need}")
    from features import forge, merge  # 现在才导，缺包上面已经拦过

    formats = args.formats or (data.get("forge") or {}).get("formats") or []  # 命令行优先，配方兜底
    if not formats:  # 两头都没给就无从导
        fail("input", "没给导出格式", hint="--format gguf 或 --format safetensors")
    unknown = [one for one in formats if one not in FORMATS]  # 认不出的格式
    if unknown:  # 报出来别硬跑
        fail("input", f"不认识的格式: {', '.join(unknown)}", hint="可用 " + " / ".join(sorted(FORMATS)))

    for one in formats:  # 后端缺包先拦，别导完一半才炸
        need = BACKEND.get(one)
        if not need:  # 能真转的两类跳过
            continue
        if importlib.util.find_spec(need) is None:  # 包不在就按缺包报，说清装哪个
            fail("missing", f"转 {one} 要 {need} 包", hint=f"pip install {need}")
        fail("missing", f"{one} 的转换后端还没接上", hint=f"装好 {need} 也还是这一步，先导 gguf")

    quants = args.quant or (data.get("forge") or {}).get("quant") or ["f16"]  # 量化档，默认不量化
    bad = [one for one in quants if one not in forge.QUANTS]  # 认不出的档
    if bad:
        fail("input", f"不认识的量化档: {', '.join(bad)}", hint="可用 " + " / ".join(sorted(forge.QUANTS)))

    source = weights(data, Path(args.recipe))  # 基座权重在哪
    tensors, real = apply_adapters(source)  # 有 adapter 就先合并，real 是它指向的基座
    here = real or source  # 结构配置跟着真正要导的权重走
    config = here.parent / "config.json" if here.is_file() else here / "config.json"  # 结构配置

    targets = plan_out(Path(args.out), formats, quants)  # 产物路径一个个定下来
    for fmt, quant, at in targets:  # 一份产物一趟
        job = forge.Choice(  # 这一份怎么导
            source=source,  # 权重来自哪
            out=at,  # 写哪个文件
            format=fmt,  # 目标格式
            quant=quant,  # 量化档
            config=config if config.exists() else None,  # 结构配置，没有就不写结构参数
            name=data.get("name") or at.stem,  # 名字进 general.name
            tensors=tensors,  # 合并过就直接给数组，没合并就是 None
        )
        report = forge.Export(job).run()  # 转换交给 features
        print(f"{fmt} {quant} → {at}（{report['bytes']} 字节，{report['tensors']} 个张量）")
        for one in report["dropped"]:  # 丢掉的东西一条条摆出来
            print(f"  丢: {one}")
    return 0


def weights(data: dict, recipe_path: Path) -> Path:
    """按配方找基座权重，相对路径按配方所在目录算。"""
    path = (data.get("model") or {}).get("path")  # 配方里的权重位置
    if not path:  # 没指基座
        fail("input", "配方没给 model.path，不知道导什么", hint="model: {path: ../ckpt}")
    found = Path(path)
    if not found.exists() and not found.is_absolute():  # 当前目录找不到再按配方旁边找，和 spin 一致
        found = recipe_path.parent / path
    if not found.exists():  # 不在就停
        fail("input", f"权重不在: {found}", hint="核对 model.path，或先把训练产物放过去")
    return found


def apply_adapters(source: Path):
    """权重旁边有 adapter_config.json 就先合并进基座，返回（合并后的字典, 基座路径）；没有就（None, None）。"""
    from features import forge, merge  # 缺包在 command 开头已经拦过

    folder = source.parent if source.is_file() else source  # adapter 和权重放同一处
    config_path = folder / "adapter_config.json"
    if not config_path.exists():  # 不是 adapter 目录，直接读盘
        return None, None
    config = json.loads(config_path.read_text(encoding="utf-8"))  # adapter 说明
    base = Path(config["base_model_name_or_path"])  # 基座在哪
    if not base.is_absolute():  # 相对路径按 adapter 目录算
        base = folder / base
    if not base.exists():
        fail("input", f"adapter 指向的基座不在: {base}", hint="核对 adapter_config.json 的 base_model_name_or_path")
    adapter = merge.Adapter(folder, alpha=config.get("lora_alpha"), r=config.get("r"))  # 用配置里的缩放
    print(f"先合并 adapter {folder}（alpha/r = {adapter.alpha:g}/{adapter.r}）")
    return merge.merge(forge.load(base), [adapter]), base  # 合并完再导


def plan_out(out: Path, formats: list, quants: list) -> list:
    """定产物路径。只导一份且 --out 带对扩展名就是文件，其余都当目录。"""
    if len(formats) == 1 and len(quants) == 1 and out.suffix == SUFFIX[formats[0]]:  # 明确指到一个文件
        return [(formats[0], quants[0], out)]
    out.mkdir(parents=True, exist_ok=True)  # 当目录用，先建好
    targets = []
    for fmt in formats:  # 每个格式一份
        for quant in (quants if fmt == "gguf" else ["f16"]):  # 量化只对 gguf 有意义
            name = "model" if len(quants) == 1 else f"model.{quant}"  # 多档时档位写进文件名
            targets.append((fmt, quant, out / f"{name}{SUFFIX[fmt]}"))
    return targets
