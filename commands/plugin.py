"""命令 plugin：列出目录里的插件，或用一份 JSON 参数调用其中一个。

调用方式：
  ori plugin list
  ori plugin run <类> <名字> --args '<JSON 对象>'

调用示例：
  ori plugin run metrics char_hit --args '{"answer":"星织","reference":"星绒"}'

检查只在命令开头一次：类别必须是四类之一，名字不能空，--args 必须是 JSON 对象。
插件内部异常照常暴露；只有扫描和导入异常由 features.plugin 隔离。
"""

import json                                               # 把命令行 JSON 变成 run 的关键字参数

from features import plugin                              # 扫描、加载并调用插件
from utils.fail import fail                              # 输入问题统一交给入口翻译


def command(args) -> int:                                 # 命令行只从这一处进入
    """列出插件，或按类别与名字运行一次。"""
    if args.action == "list":                            # list 没有用户参数要检查
        return show()                                     # 四类逐一扫描并展示

    kind = args.kind.strip()                              # 先整理用户输入
    name = args.name.strip()                              # 空白名字也算没给
    raw = args.args.strip()                               # JSON 两边空白不影响解析
    if kind not in plugin.kinds():                        # 只允许固定四类目录
        fail("input", f"不认识插件类别 {kind}", hint="可用: " + ", ".join(plugin.kinds()))
    if not name:                                          # 名字为空无从加载
        fail("input", "插件名字不能为空")                # argparse 管不了纯空白参数
    try:                                                  # JSON 是外部输入，只在这里解析一次
        values = json.loads(raw)                          # 对象的键会展开成 run 的参数
    except json.JSONDecodeError as bad:                    # 解析错误属于用户输入问题
        fail("input", "--args 不是合法 JSON", where=str(bad))  # 解析失败由用户改输入
    if not isinstance(values, dict):                      # 只有对象才能展开成关键字参数
        fail("input", "--args 必须是 JSON 对象", hint='例: {"path":"data.txt"}')

    item = plugin.load(kind, name)                        # 从当前目录现状找到插件
    result = item.run(**values)                           # 四类都按自己的参数契约运行
    print(json.dumps(result, ensure_ascii=False, indent=2))  # 结果统一打成机器可读 JSON
    return 0                                              # 插件顺利跑完


def show() -> int:                                        # list 动作的完整流程
    """列出四类目录里当前可用的插件。"""
    for kind in plugin.kinds():                           # 四类按固定顺序展示
        print(f"{kind}:")                                # 先打印目录类别
        found = plugin.scan(kind)                         # 目录现状就是这一类清单
        if not found:                                     # 空目录也明确说一声
            print("  （无）")                             # 用户知道扫描确实跑过
        for item in found:                                # 每个可用插件占一行
            print(f"  {item.NAME}  {item.SUMMARY}")      # 名字可用于 run，摘要说用途
    return 0                                              # 四类都列完
