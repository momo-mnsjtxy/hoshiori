"""下载 OASST1 公开人工 quality 标注，跑清洗权重校准并保存可复核来源。

调用：python tests/calibrate_quality.py --out .work/quality --pages 10
只选非合成、未删除、至少两票的 assistant 消息，每个对话树只取一条。
输出 labels.jsonl、原始分页和 profile.json；人工身份来自数据集标注，不由本脚本打分。
"""
import argparse                                              # 独立验收脚本的参数入口
import hashlib                                               # 给原始分页留摘要便于核对
import json                                                  # HF Viewer 的分页格式
import sys                                                   # 支持从 tests 路径直接启动
from pathlib import Path                                     # 所有下载产物放用户指定目录
from urllib.request import urlopen                           # 公开只读接口不需要凭据
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # 使用当前工作树的清洗实现
from features import clean                                   # 调用产品中同一套拟合入口


def run(out: Path, pages: int) -> dict:
    """采集固定验证集前若干页；每个对话只留一条，减少训练/留出之间的内容重叠。"""
    out.mkdir(parents=True, exist_ok=True)                    # 下载与校准证据有固定位置
    rows, provenance, trees = [], [], set()                  # 标注样本、来源、已选对话分别记录
    for page in range(pages):
        url = f"https://datasets-server.huggingface.co/rows?dataset=OpenAssistant%2Foasst1&config=default&split=validation&offset={page*100}&length=100"
        cached = out / f"page-{page:02d}.json"                 # 已下载页可恢复，网络超时不需要重取全部
        if cached.exists():
            raw = cached.read_bytes()                        # 缓存本身进入下面的哈希与结构解析
        else:
            with urlopen(url, timeout=60) as response:
                raw = response.read()                        # 外部失败如实退出，下次从缓存处继续
        cached.write_bytes(raw)       # 只放验收目录，不把原文发布到榜单
        provenance.append({"url": url, "sha256": hashlib.sha256(raw).hexdigest()})
        print(f"page {page + 1}/{pages}", flush=True)          # 便于确认活进程进度，不靠猜测重启
        for item in json.loads(raw)["rows"]:
            row = item["row"]                               # Viewer 的展示包装不参与拟合
            if row["synthetic"] or row["deleted"] or row["role"] != "assistant":
                continue                                     # 只用真实人工对助手答案的评价
            labels = row.get("labels") or {}                # 无人标注的消息不冒充零分
            names = labels.get("name", [])
            if "quality" not in names or row["message_tree_id"] in trees:
                continue                                     # 一棵对话树只占一个独立样本
            index = names.index("quality")
            if labels["count"][index] < 2:
                continue                                     # 至少两票，减少单次误操作影响
            rows.append({"text": row["text"], "score": labels["value"][index], "id": row["message_id"], "lang": row["lang"]})
            trees.add(row["message_tree_id"])                # 锁定已经选过的对话树
    (out / "labels.jsonl").write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    result = clean.fit(rows, "OpenAssistant/oasst1 human quality votes; validation split", seed=0)
    result["provenance"] = {"dataset": "https://huggingface.co/datasets/OpenAssistant/oasst1", "license": "Apache-2.0", "pages": provenance, "message_ids": [row["id"] for row in rows], "sampling": "first eligible assistant per conversation; >=2 quality votes; non-synthetic"}
    (out / "profile.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result                                            # 权重和真实留出误差一起交回


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pages", type=int, default=10)
    args = parser.parse_args()
    result = run(args.out, args.pages)
    print(json.dumps({key: value for key, value in result.items() if key != "provenance"}, indent=2))
