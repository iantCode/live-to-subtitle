@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

rem ---- Python 찾기 (3.12 권장). 없으면 uv 로 가상환경을 만듦 ----
set "PY="
set "UV="
where py >nul 2>nul && py -3.12 -c "import sys" >nul 2>nul && set "PY=py -3.12"
if not defined PY (
    rem Microsoft Store 의 가짜 python.exe 를 거르기 위해 실제로 실행해 봄
    python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul && set "PY=python"
)
if not defined PY (
    where uv >nul 2>nul && set "UV=1"
)
if not defined PY if not defined UV (
    echo [오류] Python 3.10 이상도, uv 도 찾을 수 없습니다.
    echo.
    echo   방법 1 ^(권장^): uv 설치 - PowerShell 에 아래 한 줄을 붙여 넣고, 새 창에서 setup.bat 을 다시 실행
    echo       powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    echo   방법 2: https://www.python.org 에서 Python 3.12 설치 ^(설치 화면에서 "Add python.exe to PATH" 체크^)
    goto :fail
)
if defined UV (
    echo [1/4] Python 이 없어 uv 로 진행합니다 ^(Python 3.12 를 자동으로 받음^)
) else (
    echo [1/4] Python: %PY%
)

rem ---- 가상환경 ----
if exist ".venv\Scripts\python.exe" (
    echo [2/4] 기존 가상환경 사용
) else (
    echo [2/4] 가상환경 생성 중...
    if defined UV (
        uv venv --seed --python 3.12 .venv
    ) else (
        %PY% -m venv .venv
    )
    if not exist ".venv\Scripts\python.exe" goto :fail
)

rem ---- 패키지 설치 ----
echo [3/4] 패키지 설치 중... 처음에는 수 분 걸립니다
".venv\Scripts\python.exe" -m pip install --upgrade pip >nul
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :fail

if not exist "config.toml" (
    copy /y "config.example.toml" "config.toml" >nul
    echo       config.toml 생성됨
)

rem ---- 모델 다운로드 + GPU 점검 ----
echo [4/4] Whisper 모델 다운로드 및 GPU 점검... 첫 실행은 약 1.6GB 다운로드
".venv\Scripts\python.exe" -m live_to_subtitle --check
if errorlevel 1 (
    echo [경고] 점검 실패. 위 메시지를 확인하세요.
    goto :fail
)

echo.
echo 설치 완료!
echo   1. run.bat 실행 ^(처음 실행하면 OpenAI API 키를 묻습니다^)
echo   2. OBS 브라우저 소스에 http://127.0.0.1:8765/ 추가
pause
exit /b 0

:fail
echo.
echo 설치 중 문제가 발생했습니다. 위 메시지를 확인하거나 README 의 '설치가 어렵다면' 을 참고하세요.
pause
exit /b 1
