#!/usr/bin/env bash
# 位置无关的启动器。适配服务器布局：
#   <PROJECT>/{src,configs,calibration,trajectory,jersey_train,.venv}
# 目前 PROJECT = /tmp/s1143700/basketball-v2
#
# ⚠ 路径不写死。PROJECT 是从本脚本自己的位置推导的（dirname 的 dirname），
#   所以整个项目挪到任何地方、换任何用户名，这个脚本都不用改。
#
#   chmod +x jersey_train/jersey_train.sh
#   ./jersey_train/jersey_train.sh check
#   ./jersey_train/jersey_train.sh fetch --source soccernet --out data/soccernet
#   ./jersey_train/jersey_train.sh mine  --video game.mp4 --out data/mined --end-frame 9000
#   ./jersey_train/jersey_train.sh label --src data/mined --out data/labelled
#   ./jersey_train/jersey_train.sh train --data data/soccernet --data data/labelled
#   ./jersey_train/jersey_train.sh verify --ckpt train/jersey_classifier.pt --numbers 0,6,12

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(dirname "$HERE")"

# 让 `python -m jersey_train.cli` 能找到 jersey_train/ 和同级的 trajectory/
export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"
# jersey_train 和 trajectory 都在 basketball-v2 仓库里面，所以项目根就是仓库根
export BASKETBALL_V2="${BASKETBALL_V2:-$PROJECT}"
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

exec "$PYTHON_BIN" -m jersey_train.cli "$@"
