@echo off
cd /d "%~dp0"
title NeuroPulse Shutdown
echo ===================================================
powershell -Command "Write-Host ' ____                  ____ _           _             ' -ForegroundColor Green"
powershell -Command "Write-Host '| __ ) _   _  ___     / ___| | ___  ___(_)_ __   __ _ ' -ForegroundColor Green"
powershell -Command "Write-Host '|  _ \| | | |/ _ \   | |   | |/ _ \/ __| | ''_ \ / _` |' -ForegroundColor Green"
powershell -Command "Write-Host '| |_) | |_| |  __/_  | |___| | (_) \__ \ | | | | (_| |' -ForegroundColor Green"
powershell -Command "Write-Host '|____/ \__, |\___( )  \____|_|\___/|___/_|_| |_|\__, |' -ForegroundColor Green"
powershell -Command "Write-Host '       |___/     |/                             |___/ ' -ForegroundColor Green"
powershell -Command "Write-Host ' _   _                      ____        _          ' -ForegroundColor Green"
powershell -Command "Write-Host '| \ | | ___ _   _ _ __ ___ |  _ \ _   _| |___  ___ ' -ForegroundColor Green"
powershell -Command "Write-Host '|  \| |/ _ \ | | | ''__/ _ \| |_) | | | | / __|/ _ \ ' -ForegroundColor Green"
powershell -Command "Write-Host '| |\  |  __/ |_| | | | (_) |  __/| |_| | \__ \  __/' -ForegroundColor Green"
powershell -Command "Write-Host '|_| \_|\___|\__,_|_|  \___/|_|    \__,_|_|___/\___|' -ForegroundColor Green"
echo ===================================================
echo.

echo Stopping AI Backend...
taskkill /F /FI "WINDOWTITLE eq NeuroPulse Backend*" /T >nul 2>&1
:: Fallback just in case port 8000 is still held
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8000" ^| findstr "LISTENING"') do taskkill /F /PID %%a /T >nul 2>&1

echo Stopping Clinical Dashboard...
taskkill /F /FI "WINDOWTITLE eq NeuroPulse Frontend*" /T >nul 2>&1
:: Fallback just in case port 3000 is still held
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":3000" ^| findstr "LISTENING"') do taskkill /F /PID %%a /T >nul 2>&1

echo.
echo ===================================================
echo  NeuroPulse has been successfully shut down.
echo ===================================================
pause
