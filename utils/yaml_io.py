"""读写 yaml 文件，配方和混配都用它。解析失败统一报成 input 错。

调用示例：
  from utils import yaml_io
  data = yaml_io.read(path)        # 顶层不是映射就抛错
  yaml_io.write(path, recipe)
"""

from pathlib import Path

from utils.fail import fail


def read(path: Path) -> dict:
    """读一个 yaml 映射。文件不在、解析失败都抛 input 错。"""
    if not path.exists():
        fail("input", f"找不到 {path}", hint="核对路径")
    import yaml  # 放在函数里：ori --help 不需要 yaml 也能跑

    text = path.read_text(encoding="utf-8")
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as wrong:
        fail("input", f"yaml 解析失败: {wrong}", where=str(path))
    if not isinstance(data, dict):
        fail("input", "yaml 顶层要是映射（键: 值 那种）", where=str(path))
    return data


def write(path: Path, data: dict, head: str = "") -> None:
    """写 yaml，head 是文件最上面的说明行。"""
    import yaml

    path.write_text(head + yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
