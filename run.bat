@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo 먼저 setup.bat 을 실행하세요.
    pause
    exit /b 1
)
rem 추가 인자 예:  run.bat --list-devices   /   run.bat --file sample.mp4   /   run.bat --input --device "CABLE"
".venv\Scripts\python.exe" -m live_to_subtitle %*
pause
