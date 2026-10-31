"""命令 forge：把训练好的权重导成能带走的东西。

调用示例：
  ori forge sft.warp.yaml --format gguf --format safetensors

眼下先核对参数并把要导的格式报一遍，转换接上后就在 formats 那一层铺开。
"""

from pathlib import Path

from features import recipe

FORMATS = {"safetensors", "gguf", "onnx", "mlx"}  # 认的四类产物


def command(args) -> int:
    """核对配方和格式，再把计划报一遍。"""
    data = recipe.load(Path(args.recipe))  # 读配方，坏 yaml 在这里停
    recipe.check(data)  # 字段和必备项先过
    wanted = args.formats or (data.get("forge") or {}).get("formats") or []  # 命令行优先，配方兜底
    if not wanted:  # 两头都没给就无从导
        print("没给导出格式，配方里也没有 forge.formats")
        return 0
    unknown = [one for one in wanted if one not in FORMATS]  # 认不出的格式
    if unknown:  # 报出来别硬跑
        print(f"不认识的格式: {', '.join(unknown)}，可用 {' / '.join(sorted(FORMATS))}")
        return 2
    print(f"要导 {args.recipe} → {', '.join(wanted)}")
    print("这一步还没接转换后端")
    return 0
