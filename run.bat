@echo off
chcp 65001 >nul 2>&1
echo ========================================
echo  AirControl - Gesture Control System
echo ========================================
echo.

cd /d "%~dp0"

echo [1/2] Checking Python environment...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.8+
    pause
    exit /b 1
)
echo Python OK

echo.
echo [2/2] Launching AirControl GUI...
echo Press Ctrl+C to exit at any time
echo.
python -m app.main_ui

if errorlevel 1 (
    echo.
    echo Program exited with error code: %errorlevel%
    echo Please check if the camera is occupied by another program
    pause
)
