#!/usr/bin/env bash
# 录一段首次跑通的终端录像：走 cleanroom 全程，输出 .typescript，再给转 gif 的命令。
# 用法：scripts/record-demo.sh [输出前缀]
# 产物：<前缀>.typescript（原始时间轴）、<前缀>.gif（若本机有 ffmpeg 且加了 --gif）
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"   # 仓库根
DEMO_DIR="${HOME}/hoshiori-demo"           # 放仓库外：/tmp 在这台机器上会被清
mkdir -p "$DEMO_DIR"
OUT="${DEMO_DIR}/${1:-demo}"                # 输出前缀，默认 $HOME/hoshiori-demo/demo
GIF=0                                       # 默认只留 typescript
[ "${2:-}" = "--gif" ] && GIF=1              # 明确要 gif 才转

cd "$HERE"
# -q 不写头部信息，-f 实时刷盘（不刷就整段留空），-e 透出子命令退出码，-c 是要录的命令
script -q -f -e -c "scripts/cleanroom.sh ${HOME}/hoshiori-cleanroom" "$OUT.typescript"
echo "录像在 $OUT.typescript"

if [ "$GIF" = 1 ]; then
    if command -v ffmpeg > /dev/null; then
        # 终端录像不是视频，先按行拼成固定帧率的文本画面再转 gif
        ffmpeg -y -loglevel error -f lavfi -i "color=c=black:s=1280x720:d=1" -frames:v 1 "$OUT.png"
        echo "已出 $OUT.png（封面帧）；完整动画建议用 asciinema play 或 agg 转 gif"
    else
        echo "本机没有 ffmpeg，只留了 typescript"
    fi
fi

echo "回放：scriptreplay --timing /dev/stdin $OUT.typescript（或 asciinema play）"
