# EyeMouse – Webcam-Based Eye Tracking for Mouse Control

**Author:** Evan O  

**Description:**  
EyeMouse is a Python-based system that lets you control your mouse cursor using **eye movements** detected via a standard webcam. It uses **MediaPipe FaceMesh** to track your irises, applies calibration, smoothing, and optional head pose compensation, and maps your gaze to screen coordinates.

---

## Features
- **Webcam-based tracking** — no extra hardware needed.  
- **Nine-point calibration** for improved accuracy.  
- **Adaptive smoothing** for natural pointer movement.  
- **Head pose compensation** to reduce errors when you tilt or turn your head.  
- **Optional batch launcher** for easy start-up and dependency installation.  

---

## Requirements
- **Python:** 3.8+  
- **Webcam**  
- **OS:** Windows (batch launcher optional), Linux, or macOS (manual run only)  

---

## Install Dependencies
```bash
pip install mediapipe opencv-python pyautogui numpy scikit-learn --user
