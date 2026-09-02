@echo off
setlocal

if "%~1"=="" goto :usage

set "FRAME_IDX=%~2"
if "%FRAME_IDX%"=="" set "FRAME_IDX=100"

python "%~dp0scripts\calibrate_court_launcher.py" ^
  --video "%~f1" ^
  --frame-idx "%FRAME_IDX%"

exit /b %ERRORLEVEL%

:usage
echo Usage:
echo   calibrate_court.bat "D:\path\to\3x3_pitch_competition.mp4" [frame_index]
echo.
echo Example:
echo   calibrate_court.bat "D:\videos\3x3_pitch_competition.mp4" 100
exit /b 2
