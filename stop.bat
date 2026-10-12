@echo off
cd /d "%~dp0"
title APALA Shutdown
echo ===================================================
echo   APALA - Pre-ictal EEG Monitoring
echo   Stopping the research dashboard...
echo ===================================================
echo.

echo Stopping AI Backend...
taskkill /F /FI "WINDOWTITLE eq APALA Backend*" /T >nul 2>&1
:: Fallback just in case port 8000 is still held
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8000" ^| findstr "LISTENING"') do taskkill /F /PID %%a /T >nul 2>&1

echo Stopping Clinical Dashboard...
taskkill /F /FI "WINDOWTITLE eq APALA Frontend*" /T >nul 2>&1
:: Fallback just in case port 3000 is still held
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":3000" ^| findstr "LISTENING"') do taskkill /F /PID %%a /T >nul 2>&1

echo.
echo ===================================================
echo  APALA has been shut down.
echo ===================================================
pause
