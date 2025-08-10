@echo off
title Eye Tracker Launcher

echo -----------------------------------------------
echo Project: EyeMouse
echo Created by Evan O
echo Warning! This program accesses your camera.
echo It will install required Python packages if needed.
echo -----------------------------------------------
timeout /t 5 /nobreak >nul

echo Installing dependencies...
python -m pip install --user mediapipe opencv-python pyautogui numpy scikit-learn

echo Please choose the eye tracking method:
echo 1. EyeMouseMoreAccurate.py (slower but more accurate)
echo 2. EyeMouseFast.py (faster but less accurate)
set /p choice="Enter your choice (1 or 2): "

if "%choice%"=="1" (
    echo Starting EyeMouseMoreAccurate...
    python EyeMouse\EyeMouseMoreAccurate.py
) else if "%choice%"=="2" (
    echo Starting EyeMouseFast...
    python EyeMouse\EyeMouseFast.py
) else (
    echo Invalid choice. Exiting...
)

pause
