#!/usr/bin/env bash
# 云服务器上的一键入口。
#
#   ./trajectory/run.sh /path/to/game.mp4 outputs/run1 7
#
# 参数：
#   $1  输入视频          （必填）
#   $2  输出目录          （默认 outputs/run1）
#   $3  要画的球衣号      （不填就画所有球员）
#
# 需要的环境变量：
#   BASKETBALL_V2   basketball-v2 仓库路径（不设就靠自动查找，见 bridge.py）

set -euo pipefail

VIDEO="${1:?用法: run.sh <video> [out_dir] [jersey]}"
OUT="${2:-outputs/run1}"
JERSEY="${3:-}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(dirname "$HERE")"
cd "$PROJECT"

# 让 `python -m trajectory.cli` 能找到 trajectory/ 和 calibration/
export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"
# trajectory now lives inside the basketball-v2 repository itself.
export BASKETBALL_V2="${BASKETBALL_V2:-$PROJECT}"
# 中文路径下打印不炸
export PYTHONIOENCODING=utf-8
if [ -n "${PYTHON_BIN:-}" ]; then
  :
elif [ -x "$PROJECT/.venv/bin/python" ]; then
  PYTHON_BIN="$PROJECT/.venv/bin/python"
elif [ -x "$PROJECT/venv/bin/python" ]; then
  PYTHON_BIN="$PROJECT/venv/bin/python"
else
  PYTHON_BIN="python3"
fi

echo "=============================================================="
echo "project : $PROJECT"
echo "python  : $PYTHON_BIN"
echo "video   : $VIDEO"
echo "out     : $OUT"
echo "=============================================================="

# 0) 自检。环境不对就别浪费几小时的 GPU 时间
"$PYTHON_BIN" -m trajectory.cli check

# 1) 只抽取开头一分钟。annotated.mp4 和所有后续产物共用这份 60 秒数据。
"$PYTHON_BIN" -m trajectory.cli extract \
  --video "$VIDEO" \
  --out "$OUT" \
  --duration-sec 60

# 2) 有哪些球员
"$PYTHON_BIN" -m trajectory.cli players --csv "$OUT/tracks.csv"

# 3) 出图
if [ -n "$JERSEY" ]; then
  "$PYTHON_BIN" -m trajectory.cli plot --csv "$OUT/tracks.csv" --jersey "$JERSEY" --video
else
  "$PYTHON_BIN" -m trajectory.cli plot --csv "$OUT/tracks.csv" --all
fi

echo
echo "完成。产出在 $OUT/"
ls -la "$OUT"
