@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0"

:: ============================================================
::  AirControl - Auto Install Dependencies
:: ============================================================

echo.
echo ============================================
echo   AirControl - Installing Dependencies
echo ============================================
echo.

:: --- Check Python ---
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found in PATH.
    echo Please install Python 3.10+ from https://python.org
    echo Make sure to check "Add Python to PATH" during install.
    echo.
    pause
    exit /b 1
)
python --version
echo.

:: --- Step 1: Install requirements ---
echo [1/1] Installing dependencies...
pip install opencv-python mediapipe pywin32 PyQt6 numpy psutil sounddevice sherpa-onnx --quiet
if errorlevel 1 (
    echo [ERROR] Package installation failed.
    pause
    exit /b 1
)
echo [1/1] Dependencies OK

:: --- Verify ---
echo.
echo ============================================
echo   Checking installation...
echo ============================================
echo.

python -c "import cv2;                print('  opencv-python  OK')" 2>nul || echo "  opencv-python  FAILED"
python -c "import mediapipe;          print('  mediapipe      OK')" 2>nul || echo "  mediapipe      FAILED"
python -c "import PyQt6;              print('  PyQt6          OK')" 2>nul || echo "  PyQt6          FAILED"
python -c "import win32api;           print('  pywin32        OK')" 2>nul || echo "  pywin32        FAILED"
python -c "import numpy;              print('  numpy          OK')" 2>nul || echo "  numpy          FAILED"
python -c "import sherpa_onnx;        print('  sherpa-onnx    OK')" 2>nul || echo "  sherpa-onnx    FAILED"
python -c "import sounddevice;        print('  sounddevice    OK')" 2>nul || echo "  sounddevice    FAILED"

echo.
echo ============================================
echo   Installation complete!
echo   Now run: run.bat
echo ============================================
echo.
pause
