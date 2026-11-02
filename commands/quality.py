"""质量校准：从人工标注 JSONL 拟合清洗权重，保留独立留出集误差。

调用示例：ori quality fit labels.jsonl --annotator team-a --out quality.json
每行是 {"text": "正文", "score": 0.8}。原文不会复制进质量档案。
"""

import json                                                  # 读取逐行人工分数和写档案
from pathlib import Path                                     # 输入输出由命令明确指定
from features import clean                                   # 特征和拟合都由清洗主体维护
from utils.fail import fail                                  # 错误交给统一命令入口


def command(args) -> int:
    """校验文件格式、拟合权重、保存报告，并打印留出误差。"""
    try:
        lines = Path(args.labels).read_text(encoding="utf-8").splitlines()  # 一条 JSON 对应一份人工打分
        rows = [json.loads(line) for line in lines if line.strip()]        # 空行不算标注
    except (OSError, ValueError) as bad:
        fail("input", f"标注文件读取失败: {bad}", where=args.labels)
    result = clean.fit(rows, args.annotator, args.seed)         # 所有标注语义检查集中在拟合入口
    target = Path(args.out)                                   # 用户产物不落到包的安装目录
    target.parent.mkdir(parents=True, exist_ok=True)           # 第一次校准准备目录
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))    # 同时交回机器可读的误差和来源
    return 0                                                 # 拟合成功不等同于留出误差一定改善
