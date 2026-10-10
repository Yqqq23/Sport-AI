#!/usr/bin/env bash
# 位置无关的启动器。适配服务器布局：
#   <PROJECT>/{src,configs,calibration,trajectory,jersey_train,.venv}
# 目前 PROJECT = /tmp/s1143700/basketball-v2
#
# ⚠ 路径不写死：PROJECT 由本脚本自己的位置推导。项目挪到任何地方、
#   换任何用户名，这个脚本都不用改。

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(dirname "$HERE")"

export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"
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

exec "$PYTHON_BIN" -m trajectory.cli "$@"
