@echo off
echo ===================================================
echo     STARTING OFFICE ATTENDANCE SYSTEM
echo ===================================================
echo.

echo [1/2] Launching Dashboard Server...
start "Dashboard Server" cmd /k "python dashboard_only.py"

echo Waiting a moment for server to initialize...
timeout /t 2 /nobreak >nul

echo [2/2] Launching Camera Monitor...
start "Camera Monitor" cmd /k "python monitor_only.py"

echo.
echo ===================================================
echo   ALL SYSTEMS LAUNCHED SUCCESSFULLY!
echo   Dashboard: http://localhost:5000
echo ===================================================
echo.
echo Two new terminal windows have been opened for the Background Tasks.
echo You can now close this launcher window.
pause
