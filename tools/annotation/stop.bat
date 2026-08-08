@echo off
echo Stopping uvicorn / vite / node processes on this machine...
taskkill /F /IM python.exe /FI "WINDOWTITLE eq jersey-backend*" >nul 2>nul
taskkill /F /IM node.exe /FI "WINDOWTITLE eq jersey-frontend*" >nul 2>nul
echo Done. If some windows remain open, close them manually.
