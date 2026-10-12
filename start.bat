@echo off
cd /d "%~dp0"
title NeuroPulse Launcher
echo ===================================================
powershell -Command "Write-Host ' ____  _             _   _             ' -ForegroundColor Green"
powershell -Command "Write-Host '/ ___|| |_ __ _ _ __| |_(_)_ __   __ _ ' -ForegroundColor Green"
powershell -Command "Write-Host '\___ \| __/ _` | ''__| __| | ''_ \ / _` |' -ForegroundColor Green"
powershell -Command "Write-Host ' ___) | || (_| | |  | |_| | | | | (_| |' -ForegroundColor Green"
powershell -Command "Write-Host '|____/ \__\__,_|_|   \__|_|_| |_|\__, |' -ForegroundColor Green"
powershell -Command "Write-Host '                                 |___/ ' -ForegroundColor Green"
powershell -Command "Write-Host ' _   _                      ____        _          ' -ForegroundColor Green"
powershell -Command "Write-Host '| \ | | ___ _   _ _ __ ___ |  _ \ _   _| |___  ___ ' -ForegroundColor Green"
powershell -Command "Write-Host '|  \| |/ _ \ | | | ''__/ _ \| |_) | | | | / __|/ _ \ ' -ForegroundColor Green"
powershell -Command "Write-Host '| |\  |  __/ |_| | | | (_) |  __/| |_| | \__ \  __/' -ForegroundColor Green"
powershell -Command "Write-Host '|_| \_|\___|\__,_|_|  \___/|_|    \__,_|_|___/\___|' -ForegroundColor Green"
echo ===================================================
echo.

:: Clear any stale processes holding port 8000
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8000" ^| findstr "LISTENING"') do taskkill /F /PID %%a /T >nul 2>&1

echo [1/3] Starting AI Backend (FastAPI)...
start "NeuroPulse Backend" cmd /k "cd /d "%~dp0" && python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000"

echo [2/3] Starting Clinical Dashboard (Next.js)...
start "NeuroPulse Frontend" cmd /k "cd /d "%~dp0frontend" && npm run dev"

echo [3/3] Waiting for services to boot...
timeout /t 6 /nobreak > NUL

echo.
echo Launching Dashboard in your default browser...
start http://localhost:3000

echo.
echo ===================================================
echo  NeuroPulse is now LIVE! 
echo  The backend and frontend servers are running.
echo.
echo  Press any key to exit this launcher window.
echo  (The server windows will stay running).
echo.
echo  To shut down cleanly later, run stop.bat
echo ===================================================
pause >nul
exit /b 0
