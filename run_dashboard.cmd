@echo off
setlocal
cd /d "%~dp0"
set "DASHBOARD_PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%DASHBOARD_PYTHON%" set "DASHBOARD_PYTHON=C:\ml\aiethics\.venv\Scripts\python.exe"
if not exist "%DASHBOARD_PYTHON%" (
    echo Run setup_dashboard.cmd once, then run this file again.
    pause
    exit /b 1
)
set "PYTHONUTF8=1"
set "AIETHICS_DASHBOARD_SNAPSHOT=1"
"%DASHBOARD_PYTHON%" -m streamlit run "%~dp0app\app.py" --server.address 127.0.0.1 --server.port 8501 --server.headless false
if errorlevel 1 pause
