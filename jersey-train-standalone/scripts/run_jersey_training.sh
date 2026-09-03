#!/usr/bin/env bash
# Run the jersey training pipeline on the GPU server, detached.
#
#   bash scripts/run_jersey_training.sh stage1      # SVHN pretraining
#   bash scripts/run_jersey_training.sh synthetic   # generate 50,500 crops
#   bash scripts/run_jersey_training.sh stage3      # SoccerNet fine-tune
#   bash scripts/run_jersey_training.sh all
#
# Each stage detaches with nohup and writes to logs/, so an SSH drop cannot
# kill a 30-epoch run. Follow along with:
#
#   tail -f logs/stage1_svhn.log
#
# setsid is used so the job is not merely orphaned but leaves the SSH session's
# process group entirely - a nohup'd child in the same group can still die to a
# SIGHUP the terminal sends on the way out.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
mkdir -p logs checkpoints data

# Keeps the ImageNet download inside the checkout so later runs need no
# network. Must match jersey_net._use_project_torch_home.
export TORCH_HOME="$REPO/.torch_home"

PYTHON="${PYTHON:-python}"

launch() {
  local name="$1"; shift
  local log="logs/${name}.log"

  if pgrep -f "$1 $2" >/dev/null 2>&1; then
    echo "REFUSING: something matching '$1 $2' is already running."
    echo "Two runs writing one checkpoint corrupt each other silently."
    exit 1
  fi

  echo "launching $name -> $log"
  setsid nohup "$PYTHON" "$@" >>"$log" 2>&1 < /dev/null &
  echo "  pid $!   follow with: tail -f $REPO/$log"
}

stage1() {
  # --download is a no-op once the data is present, so this stays re-runnable.
  launch stage1_svhn train/pretrain_svhn.py --download
}

synthetic() {
  # Foreground on purpose: this is the C4 review gate. It ends by writing
  # data/debug/synthetic_sample.png, and a human has to look at that sheet
  # before stage 3 trains on 50,500 images derived from these settings.
  echo "generating synthetic data in the FOREGROUND - this is the review gate"
  "$PYTHON" scripts/gen_synthetic.py --workers "$(nproc)"
  echo
  echo "Now look at data/debug/synthetic_sample.png before running stage3."
}

stage3() {
  if [ ! -f checkpoints/stage1_svhn.pt ]; then
    echo "missing checkpoints/stage1_svhn.pt - run stage1 first"; exit 1
  fi
  if [ ! -f data/synthetic/manifest.json ]; then
    echo "missing data/synthetic/manifest.json - run synthetic first"; exit 1
  fi
  launch stage3_soccernet train/finetune_soccernet.py
}

case "${1:-all}" in
  stage1)    stage1 ;;
  synthetic) synthetic ;;
  stage3)    stage3 ;;
  all)
    stage1
    echo
    echo "stage1 is running in the background. When it finishes:"
    echo "  bash scripts/run_jersey_training.sh synthetic"
    echo "  # review data/debug/synthetic_sample.png"
    echo "  bash scripts/run_jersey_training.sh stage3"
    ;;
  *) echo "usage: $0 {stage1|synthetic|stage3|all}"; exit 2 ;;
esac
