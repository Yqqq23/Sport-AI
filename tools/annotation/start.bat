@echo off
setlocal

set ROOT_DIR=%~dp0

echo Starting backend...
start "jersey-backend" cmd /k "cd /d %ROOT_DIR%backend && if not exist .venv python -m venv .venv && .venv\Scripts\python.exe -m pip install --upgrade pip && .venv\Scripts\python.exe -m pip install -r requirements.txt && .venv\Scripts\python.exe -m uvicorn app.main:app --reload"

echo Starting frontend...
start "jersey-frontend" cmd /k "cd /d %ROOT_DIR%frontend && npm install && npm run dev"

echo Done.
