#!/usr/bin/env bash
# 空白环境里按 README 走一遍：建 venv、只装项目依赖、从零收数据到导出产物，每步断言产物。
# 用法：scripts/cleanroom.sh [工作目录]
# 退出码：0 全通；非 0 是第几步断了（步骤序号就是退出码）。
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"      # 仓库根，跑之前先记住
WORK="${1:-$HOME/hoshiori-cleanroom}"          # 默认放仓库外：这台机器 /tmp 会被清，放那儿复用不了 venv
SAMPLES="$WORK/samples.jsonl"                 # 自造的语料，不依赖仓库里的示例
export HOME="$WORK/home"                      # 空 HOME，别读到本机配置
export PATH="/usr/local/bin:/usr/bin:/bin"     # 只留系统路径，收紧一点
mkdir -p "$HOME" "$WORK"

step() { echo; echo "=== $1 ==="; }            # 打步骤名，录像里看得出走到哪
fail() { echo "✗ $1"; exit "$2"; }             # 断了就带步号退出

step "1/9 建 venv 装依赖"
cd "$WORK"
if [ "${ORI_REUSE_VENV:-0}" = 1 ] && [ -x .venv/bin/ori ]; then  # 重录时复用已装好的环境，省掉几分钟网络
    echo "（复用已装好的 venv）"
else
    python -m venv .venv || fail "venv 建不起来" 1
    . .venv/bin/activate
    python -m pip install -q --upgrade pip || fail "pip 升级失败" 1
    python -m pip install -q -e "$HERE[full]" || fail "项目装不起来" 1
fi
. .venv/bin/activate
ori --version || fail "装完没有 ori" 1

step "2/9 自检"
ori doctor || echo "（doctor 报缺是可选项，继续）"

step "3/9 造 40 条语料"
python - <<'PY' || fail "造语料失败" 3
import json, random
random.seed(3)
words = "断线 张力 校准 布面 打纬 综框 真布 毛边 纱支 捻度".split()
rows = [{"text": "".join(random.choice(words) for _ in range(12)) + f"。第{i}条"} for i in range(40)]
open("samples.jsonl", "w", encoding="utf-8").write("\n".join(json.dumps(r, ensure_ascii=False) for r in rows))
PY

if [ "${ORI_REUSE_VENV:-0}" = 1 ]; then  # 复用环境时上一趟的产物要清掉，否则训练嫌目录不空
    rm -rf "$WORK/bolt" "$WORK/run" "$WORK/judge" "$WORK/gguf" "$WORK/vocab.json"
fi

step "4/9 学词表"
ori vocab --in "$SAMPLES" --out "$WORK/vocab.json" --size 320 --limit 40 || fail "词表学不出来" 4

step "5/9 收数据"
ori weave --in "$SAMPLES" --out "$WORK/bolt" --name cleanroom --license cc0 || fail "weave 失败" 5
test -s "$WORK/bolt/bolt.jsonl" || fail "bolt 是空的" 5
test -s "$WORK/bolt/card.json" || fail "卡片没出" 5

step "6/9 写配方并校验"
cat > "$WORK/run.warp.yaml" <<YAML
version: 1
task: sft
model:
  path: $WORK/base
  geometry: {d_model: 32, layers: 2, heads: 2, d_head: 16, vocab: 320, max_seq_len: 24, tie: true}
  tokenizer: vocab.json
train:
  batch_size: 2
  lr: 0.005
  lora: {r: 4, alpha: 8, dropout: 0.0, target: [w_q, w_v]}
data:
  files: [bolt/bolt.jsonl]
YAML
ori validate "$WORK/run.warp.yaml" || fail "配方没过校验" 6

step "7/9 训练二十步"
ori spin "$WORK/run.warp.yaml" --steps 20 --out "$WORK/run" || fail "训练失败" 7
test -s "$WORK/run/latest" || fail "没有存档指针" 7

step "8/9 评测"
ori judge "$WORK/run.warp.yaml" --suite format-lite --answer-cmd "echo false" --limit 2 --out "$WORK/judge" || fail "评测失败" 8
test -s "$WORK/judge/report.json" || fail "评测报告没出" 8

step "9/9 导出"
LATEST="$(cat "$WORK/run/latest")"                       # 训练存下的最后一档
sed "s|^  path: .*|  path: $WORK/run/$LATEST|" "$WORK/run.warp.yaml" > "$WORK/export.warp.yaml"  # 导出认存档，不认原配方
ori forge "$WORK/export.warp.yaml" --out "$WORK/gguf" --format gguf --quant q8_0 || fail "导出失败" 9
test -s "$WORK/gguf"/*.gguf || fail "gguf 没出" 9

echo
echo "✓ 全通，产物在 $WORK"
