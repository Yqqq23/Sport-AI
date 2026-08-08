#!/usr/bin/env bash
set -euo pipefail

echo "[CHECK] python3 version"
python3 --version || true

echo "[CHECK] node version"
node --version || true

echo "[CHECK] npm version"
npm --version || true

echo "[CHECK] GPU"
nvidia-smi || true

echo "[CHECK] disk usage"
df -h || true

echo "[CHECK] server IP"
hostname -I || true

echo "[CHECK] port 8001"
ss -ltnp | grep 8001 || true

echo "[CHECK] port 5173"
ss -ltnp | grep 5173 || true
