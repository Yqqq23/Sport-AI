#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
FRONTEND_DIR="$ROOT_DIR/frontend"

cd "$FRONTEND_DIR"

if [ ! -f ".env" ] && [ -f ".env.server.example" ]; then
  cp .env.server.example .env
  echo "[INFO] Created frontend/.env from .env.server.example. Please edit it before production use."
fi

echo "[INFO] Frontend dev server will proxy API requests to \
${VITE_PROXY_TARGET:-$(grep '^VITE_PROXY_TARGET=' .env 2>/dev/null | cut -d '=' -f2- || echo 'http://127.0.0.1:8001')}"

npm install
npm run dev -- --host 0.0.0.0 --port 5173
