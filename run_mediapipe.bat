@echo off
cd /d "%~dp0"

echo ============================================
echo   AirControl - MediaPipe Engine (CPU)
echo ============================================
echo.

:: Force MediaPipe engine (env var overrides config.json)
set AIRCONTROL_ENGINE=mediapipe
python -m app.main_ui

if errorlevel 1 (
    echo.
    echo Exit code: %errorlevel%
    pause
)
