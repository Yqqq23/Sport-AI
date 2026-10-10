#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Starting backend..."
cd "$ROOT_DIR/backend"
if [ ! -d .venv ]; then
    python3 -m venv .venv
    .venv/bin/pip install --upgrade pip
    .venv/bin/pip install -r requirements.txt
fi
nohup .venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8001 > backend.log 2>&1 &
BACKEND_PID=$!
echo "Backend PID: $BACKEND_PID"

echo "Starting frontend..."
cd "$ROOT_DIR/frontend"
npm install --silent
nohup npm run dev > frontend.log 2>&1 &
FRONTEND_PID=$!
echo "Frontend PID: $FRONTEND_PID"

echo "Done. Backend=$BACKEND_PID Frontend=$FRONTEND_PID"
echo "Run 'kill $BACKEND_PID $FRONTEND_PID' to stop."
