@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" goto install
where py >nul 2>nul
if errorlevel 1 goto use_python
py -3.12 -m venv .venv
if errorlevel 1 goto failed
goto install
:use_python
python -m venv .venv
if errorlevel 1 goto failed
:install
".venv\Scripts\python.exe" -m pip install -r requirements-dashboard.txt
if errorlevel 1 goto failed
echo Setup complete. Run run_dashboard.cmd to open http://localhost:8501
pause
exit /b 0
:failed
echo Setup failed. Install Python 3.12 and check the error above.
pause
exit /b 1
