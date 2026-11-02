"""外部命令参数：先解析可信模板，再把外部文本一次性填入参数，不经过系统 shell。

调用：argv('python backend.py --prompt "{prompt}"', prompt='题面含空格与换行')
返回 ['python', 'backend.py', '--prompt', '题面含空格与换行']，可交 subprocess.run(..., shell=False)。
模板内的管道、重定向、变量展开不具有 shell 语义；复杂流程应写成可信脚本，接收独立参数。
本工具不会审计后端脚本，也不能阻止脚本主动 eval 参数；不要把占位符放进程序源码。
"""

import re  # 单次替换，插入的外部文本不再被识别为另一个占位符
import shlex  # 只解析运营者配置的可信模板，不解析题面
from pathlib import Path  # 识别直接调用的 Python 或 shell 的已知代码参数

PLACE = re.compile(r"\{(prompt|model)\}")  # 两个现有入口共用的占位名


def argv(template: str, **values: str) -> list[str]:
    """返回独立参数；空模板、动态可执行文件和已知直接代码插值会明确报 ValueError。"""
    parts = shlex.split(template)  # 先固定参数边界，题面里的引号不参与解析
    if not parts:
        raise ValueError("命令模板不能为空")
    if PLACE.search(parts[0]):
        raise ValueError("可执行文件必须固定，不能用题面或模型占位符")
    program = Path(parts[0]).name  # 这里只识别直接 Python/shell 的 -c，不宣称覆盖任意执行器
    python = re.fullmatch(r"python(?:\d+(?:\.\d+)*)?", program) is not None
    shell = program in {"sh", "bash", "dash", "zsh", "ksh"}
    if python or shell:
        for index, part in enumerate(parts[1:], start=1):
            if part == "--":
                break  # 后续已经是脚本的数据参数，不再作为解释器开关检查
            if ((python and part.startswith("-c") and len(part) > 2) or part.startswith("--command=")) and PLACE.search(part):
                raise ValueError("不要把占位符嵌入解释器代码；请使用固定脚本和独立参数")
            if not re.fullmatch(r"-[A-Za-z]*c[A-Za-z]*", part) and part != "--command":
                continue  # 固定 Python/shell 程序保持可用
            if index + 1 < len(parts) and PLACE.search(parts[index + 1]):
                raise ValueError("不要把占位符嵌入 -c/--command 程序；请把题面作为独立参数传给固定脚本")
            break  # -c 之后的其余参数属于固定程序，不再被解释器当开关
    return [PLACE.sub(lambda match: values.get(match.group(1), match.group(0)), part) for part in parts]  # 每项只替换一次
