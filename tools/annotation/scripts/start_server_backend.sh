#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BACKEND_DIR="$ROOT_DIR/backend"

cd "$BACKEND_DIR"

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

if [ ! -f ".env" ] && [ -f ".env.server.example" ]; then
  cp .env.server.example .env
  echo "[INFO] Created backend/.env from .env.server.example. Please edit it before production use."
fi

python -m uvicorn app.main:app --host 0.0.0.0 --port 8001
