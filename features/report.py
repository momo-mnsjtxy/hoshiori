"""评测报告：一次评测的结论落成 markdown、json、csv 三份，内容同源。

调用示例：
  from features import report
  rows = [{"suite": "zh-lite", "id": "units", "score": 1.0, "reason": "答案里有全部关键词", "answer": "12"}]
  data = report.make(model="zh-mix", backend="echo", suite="zh-lite", recipe="sha256:…", rows=rows)
  paths = report.save(data, Path("out/r"))
  paths["json"]  # out/r/report.json，回归门就吃这一份

三份同源：markdown 给人看，json 给门和机器读，csv 拖进表格算。
json 里固定带模型名、后端、套件名、逐题得分、总分、配方哈希、硬件信息。
"""

import csv
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path

FILES = {"json": "report.json", "markdown": "report.md", "csv": "report.csv"}  # 三份产物的固定名


def hardware() -> dict:
    """这台机器的底子：CPU、核数、内存、显卡。写进报告，复现问题时有据可查。"""
    return {                                              # 一份硬件底子
        "cpu": cpu_name(),                                # CPU 型号
        "cores": platform.machine() + f" x{_cores()}",    # 架构与核数
        "mem_gb": _mem_gb(),                              # 总内存，读不到给 0
        "gpu": _gpu(),                                    # 显卡清单，没卡就是空
        "os": platform.platform(),                        # 系统版本
    }


def cpu_name() -> str:
    """CPU 型号。先查 /proc/cpuinfo，非 Linux 退回 platform。"""
    if Path("/proc/cpuinfo").exists():                                                                   # Linux 才有这一份
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace").splitlines():    # 一行一个字段
            if line.startswith("model name"):                                                            # 就是型号那一行
                return line.split(":", 1)[1].strip()                                                     # 冒号后面就是型号
    return platform.processor() or "unknown"                                                             # 别的系统只能问 platform


def make(model: str, backend: str, suite: str, recipe: str, rows: list) -> dict:
    """把逐题结果收成一份报告的数据，三份产物都由它派生。"""
    total = round(sum(row["score"] for row in rows) / len(rows), 4) if rows else 0.0    # 平均分就是总分
    return {                                                                            # 一份报告
        "schema": "hoshiori/judge/1",                                                   # 报告格式版本，门按这个认
        "model": model,                                                                 # 被评的模型或配方名
        "backend": backend,                                                             # 答案从哪来：echo 或外部命令
        "suite": suite,                                                                 # 跑了哪几套，逗号分开
        "recipe": recipe,                                                               # 配方哈希，配方一改这里就变
        "hardware": hardware(),                                                         # 跑评测的机器
        "count": len(rows),                                                             # 实际跑了几题
        "total": total,                                                                 # 总分
        "tasks": rows,                                                                  # 逐题结果
    }


def save(data: dict, out: Path) -> dict:
    """把一份报告落成三份文件，返回各自路径。目录不存在就先建。"""
    out.mkdir(parents=True, exist_ok=True)                                                            # 报告目录先建好
    paths = {kind: out / name for kind, name in FILES.items()}                                        # 三份的落点
    paths["json"].write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")        # 机器读的那份
    paths["markdown"].write_text(markdown(data), encoding="utf-8")                                    # 人读的那份
    with paths["csv"].open("w", encoding="utf-8", newline="") as handle:                              # 表格那份
        writer = csv.writer(handle)                                                                   # 交给 csv 处理引号和换行
        writer.writerow(["id", "suite", "score", "reason", "answer"])                                 # 表头
        for row in data["tasks"]:                                                                     # 逐题一行
            writer.writerow([row["id"], row["suite"], row["score"], row["reason"], row["answer"]])    # 一行一题
    return paths                                                                                      # 三份都写完了，路径交回去


def markdown(data: dict) -> str:
    """把报告排成给人看的一份。表格里的换行压成空格，免得撑破行。"""
    head = data["hardware"]                                             # 硬件那一节
    lines = [                                                           # 报告的头几行
        f"# 评测报告 {data['suite']}",
        "",
        f"- 模型: {data['model']}",
        f"- 后端: {data['backend']}",
        f"- 配方: {data['recipe']}",
        f"- 机器: {head['cpu']} / {head['cores']} / {head['mem_gb']} GiB / {head['gpu'] or '无 GPU'}",
        f"- 总分: {data['total']}（{data['count']} 题）",
        "",
        "| 题号 | 得分 | 判定 |",
        "|---|---|---|",
    ]
    for row in data["tasks"]:                                           # 一题一行
        reason = row["reason"].replace("\n", " ")                       # 换行会破坏表格
        lines.append(f"| {row['id']} | {row['score']} | {reason} |")    # 表格里加一行
    return "\n".join(lines) + "\n"                                      # 拼成一整份


def _cores() -> int:
    """核数，读不到按 0 记。"""
    return os.cpu_count() or 0  # 读不到按 0 记


def _mem_gb() -> float:
    """总内存（GiB），从 /proc/meminfo 读，读不到给 0。"""
    info = Path("/proc/meminfo")                                   # Linux 的内存信息
    if not info.exists():                                          # 别的系统没有
        return 0.0                                                 # 按 0 记
    for line in info.read_text(encoding="utf-8").splitlines():     # 一行一个字段
        if line.startswith("MemTotal:"):                           # 就是总量那一行，单位 kB
            return round(int(line.split()[1]) / 1024 / 1024, 1)    # kB 换成 GiB
    return 0.0                                                     # 没读到


def _gpu() -> list:
    """显卡型号清单。没有 nvidia-smi、或驱动没装就是空。"""
    if shutil.which("nvidia-smi") is None:                                                         # 没这工具就是没 NVIDIA 卡
        return []                                                                                  # 没卡
    try:
        done = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=20)    # 一行一张卡
    except (OSError, subprocess.SubprocessError):                                                  # 驱动坏了也不该拖垮报告
        return []                                                                                  # 有工具没驱动
    if done.returncode:                                                                            # 只有工具没驱动时输出是一句错误话，不是卡名
        return []                                                                                  # 那就当没卡
    return [line.split("(")[0].strip() for line in done.stdout.splitlines() if "GPU" in line]      # 拆掉后面的 UUID 括号
