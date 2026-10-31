"""命令 doctor：看看这台机器缺什么。

调用示例：
  ori doctor

每一项单独查，全查完再一起报数，缺几样返回 4。
GPU 那几行只列存在与否，驱动信息直接从 nvidia-smi 打出来看。
"""

import importlib.util
import platform
import shutil
import sys

NEED = (
    ("yaml", "配方解析要它（装 pyyaml），必有"),
    ("numpy", "去重加速，缺了就慢，还有纯 Python 顶上"),
    ("blake3", "指纹换 blake3 算法时才要"),
    ("torch", "训练和导出"),
    ("safetensors", "读写权重"),
)

TOOLS = ("nvidia-smi", "rocm-smi", "nvidia-ctk")  # 有就说明装了哪家驱动


def command(args) -> int:
    """跑完全部检查并汇总。"""
    print("星织 doctor\n")  # 先报环境底子，接手者一眼知道在哪台机器上跑
    print(f"  python  {platform.python_version()} @ {sys.executable}")
    print(f"  系统    {platform.system()} {platform.release()} {platform.machine()}\n")

    missing = 0  # 缺了几样，最后按它给退出码
    for name, why in NEED:
        found = importlib.util.find_spec(name) is not None  # 能导到就算有，认模块名不认发行名
        print(f"  {'✓' if found else '✗'} {name:<12} {why}")  # 缺的标叉，说明为什么需要它
        if not found:
            missing += 1

    for tool in TOOLS:  # 驱动工具，缺了不一定算失败，看用哪家卡
        found = shutil.which(tool)
        print(f"  {'✓' if found else '-'} {tool}")
    if shutil.which("nvidia-smi"):  # 有卡就顺手把型号和显存打出来
        import subprocess

        print()
        print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], capture_output=True, text=True).stdout)

    print()
    if missing:  # 缺东西就按缺东西的退出码
        print(f"缺 {missing} 样，按上面的名字装")
        return 4
    print("基础面齐了，后端矩阵见 docs/INSTALL.md")
    return 0
