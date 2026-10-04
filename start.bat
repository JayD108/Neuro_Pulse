@echo off
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

echo [1/3] Starting AI Backend (FastAPI)...
start "NeuroPulse Backend" cmd /c "uvicorn backend.app.main:app --host 0.0.0.0 --port 8000"

echo [2/3] Starting Clinical Dashboard (Next.js)...
start "NeuroPulse Frontend" cmd /c "cd frontend && npm run dev"

echo [3/3] Waiting for services to boot...
timeout /t 5 /nobreak > NUL

echo.
echo Launching Dashboard in your default browser...
start http://localhost:3000

echo.
echo ===================================================
echo  NeuroPulse is now LIVE! 
echo  You can close this window, the servers will 
echo  keep running in their own windows.
echo.
echo  To shut down cleanly, run stop.bat
echo ===================================================
pause
