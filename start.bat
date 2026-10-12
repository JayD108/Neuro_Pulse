@echo off
cd /d "%~dp0"
title APALA Launcher
echo ===================================================
echo   APALA - Pre-ictal EEG Monitoring
echo   Starting the research dashboard...
echo ===================================================
echo.

:: Clear any stale processes holding ports 8000 or 3000
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8000" ^| findstr "LISTENING"') do taskkill /F /PID %%a /T >nul 2>&1
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":3000" ^| findstr "LISTENING"') do taskkill /F /PID %%a /T >nul 2>&1

echo [1/3] Starting AI Backend (FastAPI)...
start "APALA Backend" cmd /k "cd /d "%~dp0" && python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000"

echo [2/3] Starting Clinical Dashboard (Next.js)...
start "APALA Frontend" cmd /k "cd /d "%~dp0frontend" && npm run dev"

echo [3/3] Waiting for services to boot...
timeout /t 6 /nobreak > NUL

echo.
echo Launching Dashboard in your default browser...
start http://localhost:3000

echo.
echo ===================================================
echo  APALA is now running. 
echo  The backend and frontend servers are running.
echo.
echo  Press any key to exit this launcher window.
echo  (The server windows will stay running).
echo.
echo  To shut down cleanly later, run stop.bat
echo ===================================================
pause >nul
exit /b 0
