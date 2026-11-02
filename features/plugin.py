"""插件：扫描四类目录，导入外部作者放进来的单文件能力。

调用方式：
  from features import plugin
  plugin.kinds()                              # 四类目录名
  plugin.scan("metrics")                     # 这一类里可用的模块
  metric = plugin.load("metrics", "char_hit")

调用示例：
  metric.run(answer="星织", reference="星绒")

每个文件必须定义 NAME、SUMMARY、run，run 的结果要能转成 JSON。四类参数约定：
  sources.run(path)；rewards.run(answer, limit, rate)；
  metrics.run(answer, reference)；exporters.run(report, path)。
目录就是清单：放入文件便出现，删掉文件便消失，不另设登记表。
"""

import importlib.util                                    # 按文件位置导入插件
import warnings                                          # 单个坏插件只发警告
from pathlib import Path                                 # 插件目录跟着项目位置走

from utils.fail import fail                              # 未知插件按用户输入错误返回

ROOT = Path(__file__).resolve().parents[1] / "plugins"   # 四类插件共同的根目录
KINDS = ("sources", "rewards", "metrics", "exporters")  # 固定的四类插件位置
FIELDS = ("NAME", "SUMMARY", "run")                    # 每个插件必须交出的三样东西


def kinds() -> tuple:                                    # 给命令层列出四类位置
    """返回四类插件目录名，顺序也是命令行展示顺序。"""
    return KINDS                                         # 调用方只读，不需要复制


def scan(kind: str) -> list:                             # 一类目录就是一份清单
    """扫描一类目录并导入可用插件；坏文件警告后跳过。"""
    found = []                                           # 只收完整、能导入的插件
    for path in sorted((ROOT / kind).glob("*.py")):      # 文件名排序，让输出每次一致
        if path.name.startswith("_"):                   # __init__.py 不是插件
            continue                                     # 留给 Python 自己使用
        try:                                             # 外部插件彼此隔离
            found.append(_read(kind, path))              # 导入并核对统一契约
        except Exception as bad:                          # 只隔离当前这个外部文件
            warnings.warn(f"跳过坏插件 {path}: {bad}", UserWarning, stacklevel=2)  # 警告后继续下一份
    return found                                         # 扫描结果不缓存，删除立刻生效


def load(kind: str, name: str):                          # 按插件自己声明的名字找
    """按 NAME 加载一个插件；找不到时抛出 input Failure。"""
    found = scan(kind)                                   # 每次从目录现状重新找
    for item in found:                                   # 名字由插件文件自己声明
        if item.NAME == name:                            # 找到用户点名的那一份
            return item                                  # 模块本身带 SUMMARY 与 run
    names = ", ".join(item.NAME for item in found) or "（没有可用插件）"  # 给出能选的名字
    fail("input", f"{kind} 里找不到插件 {name}", hint=f"可用插件: {names}")  # 用户可以改名字重试


def _read(kind: str, path: Path):                        # 导入一个扫描到的文件
    """从一个文件导入模块，并核对三项插件契约。"""
    module_name = f"hoshiori_plugin_{kind}_{path.stem}"  # 每类同名文件也互不冲突
    spec = importlib.util.spec_from_file_location(module_name, path)  # 直接从扫描到的文件导入
    item = importlib.util.module_from_spec(spec)          # 建一个不进中央登记表的模块
    spec.loader.exec_module(item)                        # 执行文件，拿到它公开的能力
    for field in FIELDS:                                 # 三项缺一项就算坏插件
        getattr(item, field)                             # 缺字段会抛错，由 scan 隔离
    if not callable(item.run):                           # run 必须真的能调用
        raise TypeError("run 不是可调用函数")             # 让 scan 用统一警告跳过
    return item                                          # 完整模块交给列表或调用方
