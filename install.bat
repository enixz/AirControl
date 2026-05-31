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

:: --- Step 1: Core packages ---
echo [1/2] Installing core packages...
pip install opencv-python mediapipe pywin32 PyQt6 numpy psutil sounddevice --quiet
if errorlevel 1 (
    echo [ERROR] Core packages install failed
    pause
    exit /b 1
)
echo [1/2] Core packages OK

:: --- Step 2: Torch + YOLO ---
echo.
echo [2/2] Installing torch + ultralytics (GPU support)...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --quiet
if errorlevel 1 (
    echo [WARN] GPU torch failed, trying CPU version...
    pip install torch torchvision --quiet
)
pip install ultralytics --quiet
echo [2/2] Torch + ultralytics OK

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
python -c "import torch;              print('  torch          OK  CUDA='+str(torch.cuda.is_available()))" 2>nul || echo "  torch          FAILED"
python -c "from ultralytics import YOLO; model=YOLO('models\\hand_yolo11s-pose.pt'); print('  ultralytics    OK')" 2>nul || echo "  ultralytics    FAILED"

echo.
echo ============================================
echo   Installation complete!
echo   Now run: run_yolo.bat
echo ============================================
echo.
pause
