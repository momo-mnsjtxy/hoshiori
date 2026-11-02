"""命令 forge：把权重导成能带走的产物，safetensors 或 GGUF，可带量化档。

调用示例：
  ori forge sft.warp.yaml --out out/gguf --format gguf --quant q8_0
  ori forge sft.warp.yaml --out out/full.gguf --format gguf            # 不量化，fp16
  ori forge sft.warp.yaml --out out --format gguf --quant q5_k_m --quant q8_0
  ori forge sft.warp.yaml --out out/merged --format safetensors
  ori forge sft.warp.yaml --out out/m.onnx --format onnx     # 整张因果图 + 旁边同名 .safetensors 权重
  ori forge sft.warp.yaml --out out/mlx --format mlx         # 产物是目录，交给 mlx_lm.convert
  ori forge sft.warp.yaml --out out --format mlx --mlx-print-only --verbose   # 只打印要跑的命令，在 macOS 上照抄
  ori forge sft.warp.yaml --out out/m.gguf --format gguf --quant q8_0 --self-check
  ori forge sft.warp.yaml --out out/m.gguf --format gguf --quant q5_k_m --self-check --suites zh-lite format-lite

检查只在这里做一次：自评套件在不在、格式认不认、量化档认不认、--out 给没给、权重在不在、后端包在不在。
剩下的转换全交给 features.forge，命令层不碰字节。
onnx 要 torch 和 onnx，mlx 要 mlx_lm，缺了就按缺包报 exit 4，说清缺哪个，别硬转出来骗人；
--mlx-print-only 只打印命令，所以走这条路时 mlx_lm 缺了也照过。
--self-check 要 PATH 里有 llama-cli；基线和量化档都走它，避免两个运行时的差异混进退化值。
配方的 model.path 指向带 adapter_config.json 的目录时，先合并再导，合并交给 features.merge。
"""

import importlib.util
import json
from pathlib import Path

from features import recipe, suite
from utils.fail import fail

FORMATS = {"safetensors", "gguf", "onnx", "mlx"}  # 认的四类产物
DEFAULT_SUITES = ("zh-lite", "format-lite")  # 没点名时自评就跑这两套内置题
DEFAULT_THRESHOLD = 0.05  # 总分退化超过 5 个百分点就拦
NEED = (("numpy", "读写权重数组"), ("safetensors", "读写 safetensors"))  # 转换必备的两个包
SUFFIX = {"safetensors": ".safetensors", "gguf": ".gguf", "onnx": ".onnx", "mlx": ""}  # 各类产物的后缀，mlx 是目录
BACKEND = {"onnx": ("torch", "onnx"), "mlx": ("mlx_lm",)}  # 转这两类各自缺的包


def command(args) -> int:
    """验配方、定目标、合并 adapter、逐个格式导。"""
    data = recipe.load(Path(args.recipe))  # 读配方，坏 yaml 在这里停
    recipe.check(data)  # 字段和必备项先过
    checked_suites = []  # 不自评时保持空，普通导出不碰套件
    threshold = DEFAULT_THRESHOLD  # 命令最后拿这个数过门
    if args.self_check:  # 自评的全部用户输入先在开头验完
        names = suite_names(args.suites)  # 逗号和空格两种写法收成一列
        if not names:  # 显式给了空串也不能悄悄跑零题
            fail("input", "自评至少要一套基准", hint="例: --suites zh-lite format-lite")
        checked_suites = [suite.locate(suite.folder(), name) for name in names]  # 缺任一套就停
        for path in checked_suites:  # 文件存在还不够，题目形状也要在导出前过
            suite.read(path)  # 后面的 benchmark 直接信任这些套件
        threshold = self_check_threshold(data, args.self_check_threshold)  # 命令行优先，配方其次，默认 0.05
        if args.no_tokenizer:  # 没词表的 GGUF 无法把题面送进模型
            fail("input", "--self-check 不能和 --no-tokenizer 一起用", hint="保留词表再自评")
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
    if args.self_check and "gguf" not in formats:  # 自评比较的是 f16 GGUF 与量化 GGUF
        fail("input", "--self-check 要同时导出 gguf", hint="加 --format gguf 和一个 --quant 档")

    for one in formats:  # 后端缺包先拦，别导完一半才炸
        if one == "mlx" and args.mlx_print_only:  # 只打印时不用真跑 mlx_lm
            continue
        for need in BACKEND.get(one, ()):  # 这个格式要的包一个个看
            if importlib.util.find_spec(need) is None:  # 缺哪个就报哪个
                fail("missing", f"转 {one} 要 {need} 包", hint=f"pip install {need}")

    quants = args.quant or (data.get("forge") or {}).get("quant") or ["f16"]  # 量化档，默认不量化
    bad = [one for one in quants if one not in forge.QUANTS]  # 认不出的档
    if bad:
        fail("input", f"不认识的量化档: {', '.join(bad)}", hint="可用 " + " / ".join(sorted(forge.QUANTS)))
    if args.self_check and (len(quants) != 1 or quants[0] in {"f16", "f32"}):  # 一份固定名报告只对应一个真量化产物
        fail("input", "--self-check 要给一个量化档", hint="例: --quant q8_0；多档请分开运行")

    source = weights(data, Path(args.recipe))  # 基座权重在哪
    tensors, real = apply_adapters(source)  # 有 adapter 就先合并，real 是它指向的基座
    here = real or source  # 结构配置跟着真正要导的权重走
    config = here.parent / "config.json" if here.is_file() else here / "config.json"  # 结构配置

    targets = plan_out(Path(args.out), formats, quants)  # 产物路径一个个定下来
    exit_code = 0  # 任一量化档退化过线，整条命令最后返回 1
    for fmt, quant, at in targets:  # 一份产物一趟
        job = forge.Choice(  # 这一份怎么导
            source=source,  # 权重来自哪
            out=at,  # 写哪个文件
            format=fmt,  # 目标格式
            quant=quant,  # 量化档
            config=config if config.exists() else None,  # 结构配置，没有就不写结构参数
            tokenizer=here if here.is_dir() else None,  # 词表跟着要导的基座走
            with_tokenizer=not args.no_tokenizer,  # --no-tokenizer 时只出权重
            name=data.get("name") or at.stem,  # 名字进 general.name
            tensors=tensors,  # 合并过就直接给数组，没合并就是 None
            verbose=args.verbose,  # --verbose 把命令、参数、目录布局打出来
            mlx_print_only=args.mlx_print_only,  # --mlx-print-only 只打印不执行
            self_check=args.self_check and fmt == "gguf",  # 只有 GGUF 有量化前后的可比运行时
            suites=checked_suites,  # 命令开头已经逐份验过
        )
        report = forge.Export(job).run()  # 转换交给 features
        print(f"{fmt} {quant} → {at}（{report['bytes']} 字节，{report['tensors']} 个张量）")
        for one in report["dropped"]:  # 丢掉的东西一条条摆出来
            print(f"  丢: {one}")
        if "self_check" in report:  # 自评产物多报分差、报告与门控结论
            exit_code = max(exit_code, show_self_check(report["self_check"], threshold))  # 一档不过就整条不过
    return exit_code


def suite_names(given) -> list:
    """把 --suites 的空格项与逗号项摊成干净名字；没给就用两套默认基准。"""
    groups = given or DEFAULT_SUITES  # parser 不重复保存默认值，来源只在本文件这一处
    return [name.strip() for group in groups for name in str(group).split(",") if name.strip()]  # 两种分隔写法都收


def self_check_threshold(data: dict, given) -> float:
    """自评允许退化多少：命令行优先，配方 eval.gate.min_delta 其次，默认 0.05。"""
    configured = ((data.get("eval") or {}).get("gate") or {}).get("min_delta")  # 沿用 judge 已有的门配置
    value = float(given if given is not None else configured if configured is not None else DEFAULT_THRESHOLD)  # 三层优先级
    if not 0 <= value <= 1:  # 分数落在 0~1，退化阈值也只认这个范围
        fail("input", "自评阈值要落在 0 到 1 之间", hint="常规用 0.05")
    return value


def show_self_check(data: dict, threshold: float) -> int:
    """打印自评结论；总分退化严格超过阈值时列出掉最多的三项并返回 1。"""
    degradation = float(data["degradation"])  # 正数表示量化后掉分
    print(f"  自评: 基线 {data['baseline']['total']:.4f} → 量化 {data['quantized']['total']:.4f}，退化 {degradation:+.4f}")
    print(f"  报告: {data['files']['markdown']} · {data['files']['json']}")
    if degradation <= threshold:  # 等于阈值仍在允许范围内
        print(f"  自评门：过（允许退化 {threshold:.4f}）")
        return 0
    print(f"  自评门：不过（允许退化 {threshold:.4f}）")
    dropped = [one for one in data["items"] if one["degradation"] > 0][:3]  # 只报真掉分的前三项
    for one in dropped:  # 一项一行，日志里能直接定位
        name = f"{one['suite']}/{one['id']}" if one["suite"] else one["id"]  # 没套件名的旧报告也能读
        print(f"    {name}: {one['baseline']:.4f} → {one['quantized']:.4f}（退化 {one['degradation']:.4f}）")
    return 1


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
            targets.append((fmt, quant, out / f"{name}{SUFFIX[fmt]}"))  # mlx 没有后缀，落成一个目录
    return targets
