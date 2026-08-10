# 🎯 AirControl - Gesture + Voice Control System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-Windows%2010%2F11-0078d4?logo=windows&logoColor=white)
![Release](https://img.shields.io/badge/Release-v1.4.0-6f42c1)
![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)
![Stars](https://img.shields.io/github/stars/enixz/AirControl?style=social)

**🚀 Free your hands, control your computer remotely**

*Gesture Control · Voice Assistant · Multimodal Interaction*

[中文](README.md) | [English](#quick-start)

</div>

---

## ✨ One-line Introduction

> **AirControl** is a Windows air controller based on MediaPipe + voice recognition. Control PPT presentations, mouse cursor, screen annotation, and even launch voice assistants with just **gestures** and **voice** - no keyboard or mouse needed!

---

## 🎬 Feature Highlights

<table>
<tr>
<td width="50%">

### 🖐️ Gesture Control
- **Presentation Mode**: Wave to change slides, start/stop playback
- **Mouse Mode**: Air mouse, pinch to click, scissor to scroll
- **Drawing Mode**: Finger writing, fist to clear, shape correction

</td>
<td width="50%">

### 🎤 Voice Assistant
- **Offline Wake-up**: Sherpa-ONNX keyword detection (privacy-first)
- **Online Recognition**: Tencent Cloud ASR for free text input
- **Mode-aware**: Auto-switch voice commands by mode

</td>
</tr>
<tr>
<td>

### 🧠 Smart Recognition
- **Kalman Filter**: Dual smoothing for 21 hand landmarks
- **Shape Correction**: Auto-detect and correct hand-drawn shapes
- **Edge Acceleration**: Cursor speeds up near screen edges

</td>
<td>

### ⚡ High Performance
- **Async Inference**: MediaPipe runs in background thread
- **30 FPS**: Real-time smooth hand tracking
- **Low Latency**: < 50ms response time

</td>
</tr>
</table>

---

## 🚀 Quick Start

### 1️⃣ Clone the Project

```bash
git clone https://github.com/enixz/AirControl.git
cd AirControl
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Program

```bash
# GUI version (recommended)
python -m app.main_ui

# Command-line version
python -m app.main
```

> 💡 **Note**: On first run, MediaPipe model files (~16MB) will be downloaded automatically. Please ensure a stable network connection.

---

## 📖 Detailed Features

### 🎭 Three Interaction Modes

Switch modes by **double-fist** gesture (both hands):

#### 1. Presentation Mode 📊

Control PowerPoint or WPS presentations.

| Gesture | Action | Shortcut |
|---------|--------|----------|
| 👋 Wave right | Next slide | → |
| 👋 Wave left | Previous slide | ← |
| 👋 Wave up | Start presentation | F5 |
| 👋 Wave down | End presentation | Esc |
| 👍 Thumb up | Switch to target app | - |
| ✊ Fist | Customizable mapping | - |

> 🛡️ **Anti-misfire Design**: Spreading all five fingers and moving freely won't trigger any action; actions won't fire within the first 0.3 seconds of hand entering the frame.

#### 2. Mouse Mode 🖱️

Transform your hand into an air mouse with a large circular cursor on screen.

| Gesture | Action | Visual Feedback |
|---------|--------|-----------------|
| ☝️ Middle fingertip move | Control cursor position | White circular cursor |
| 🤏 Thumb + index pinch | Left click | Blue ripple animation |
| 🤏 Thumb + middle pinch | Right click | Green ripple animation |
| ✌️ Scissor hand move | Scroll wheel | Yellow arrow indicator |
| 🤏 Pinch & hold | Left button drag | Pulse animation |

**Edge Acceleration**: When the cursor approaches screen edges, movement speed automatically increases 2-3x for easy access to taskbar and screen corners.

#### 3. Drawing Mode ✏️

Handwriting annotation on screen, perfect for teaching or presentation explanations.

| Gesture | Action |
|---------|--------|
| ☝️ Index finger only | Write/Draw |
| 👌 Thumb + index pinch | Hover (no drawing) |
| ✊ Fist | Clear canvas |
| ✊✊ Double fist | Toggle shape correction |

**Smart Shape Correction**: When enabled, hand-drawn lines, triangles, rectangles, and ellipses are automatically corrected to standard geometric shapes.

---

### 🎤 Voice Assistant

AirControl integrates a dual-engine voice system:

#### Offline Keyword Spotting (KWS)
- **Engine**: Sherpa-ONNX (fully offline, privacy-first)
- **Model**: `kws-zh-wenetspeech` (Chinese, ~18MB lightweight model)
- **Commands**: A fixed set of Chinese phrases mapped per mode (presentation/mouse/draw)
- **Latency**: ~100ms keyword detection
- **Cooldown**: 1.0s anti-bounce

#### Offline Dictation (ASR)
- **Engine**: SenseVoice-Small (Alibaba DAMO Academy, loaded via sherpa-onnx)
- **Scenario**: Drawing mode "speak-to-screen" — say "开始板书" to start recording,
  say "结束板书" to stop and type the recognized text onto the canvas
- **Languages**: Chinese / English / Japanese / Korean / Cantonese (auto-detect)
- **Privacy**: 100% local, no network calls
- **Disk footprint**: ~234 MB (int8 quantized)

##### Model download

```bash
# Download SenseVoice-Small from the sherpa-onnx releases page:
# https://github.com/k2-fsa/sherpa-onnx/releases  (tag: asr-models)
# File: sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2

# Extract and move the directory to AirControl/models/sense-voice/
# Expected layout:
#   models/sense-voice/
#     ├── model.int8.onnx
#     └── tokens.txt
```

If the model directory is absent, dictation is silently disabled; KWS keywords keep
working. The model file is gitignored due to GitHub's 100 MB single-file limit.

#### Mode Awareness
Voice commands automatically switch based on current mode:
- **Presentation Mode**: 开始播放 / 结束播放 / 下一页 / 上一页
- **Mouse Mode**: 点一下 / 双击 / 右键
- **Drawing Mode**: 清屏 / 开始板书 / 结束板书 / 图形修正

#### YOLO Hand Detection Model (Optional)

At long range (3-5 m), the default MediaPipe detector has low recall.
AirControl supports an optional HaGRID YOLO hand detection engine
(`hagrid_yolo`) and engine auto-switching (`engine_auto_switch`) that
falls back to YOLO capture when the hand is lost at distance.

Because the model's ONNX metadata carries an **AGPL-3.0** licence that
conflicts with this project's Apache-2.0, it is **not bundled in
releases**. Users who need long-range enhancement must download it
themselves:

```bash
# Option 1: Official Ultralytics YOLOv8n (recommended)
# 1. Download the .pt weights
#    https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt
# 2. Install ultralytics and export to ONNX
pip install ultralytics
yolo export model=yolov8n.pt format=onnx opset=13 simplify imgsz=640
# 3. Rename to hand_yolov8n.onnx and place in the models/ directory

# Option 2: HaGRID v2 pre-trained hand detector
# 1. Download the .pt weights
#    https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid_v2/models/YOLOv10n_hands.pt
# 2. Export to ONNX (same as above), place in the models/ directory
```

> Without the model, the `hagrid_yolo` engine and `engine_auto_switch`
> long-range auto-switching are unavailable. The default `mediapipe`
> engine is unaffected. See [Model Provenance and Release Gate](MODEL_PROVENANCE.md).

---

## 🏗️ Technical Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  AirControl Architecture                 │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │
│  │  Camera     │  │  Voice      │  │  User       │      │
│  │  Capture    │  │  Input      │  │  Config     │      │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘      │
│         │                │                │              │
│         ▼                ▼                ▼              │
│  ┌─────────────────────────────────────────────────┐    │
│  │              InferenceWorker (QThread)           │    │
│  │  ┌─────────────┐  ┌─────────────┐               │    │
│  │  │ MediaPipe   │  │ Kalman      │               │    │
│  │  │ HandLandmark│  │ + EMA       │               │    │
│  │  └─────────────┘  └─────────────┘               │    │
│  └─────────────────────────────────────────────────┘    │
│         │                                                │
│         ▼                                                │
│  ┌─────────────────────────────────────────────────┐    │
│  │              ModeManager                         │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐      │    │
│  │  │Presentat.│  │  Mouse   │  │ Drawing  │      │    │
│  │  └──────────┘  └──────────┘  └──────────┘      │    │
│  └─────────────────────────────────────────────────┘    │
│         │                                                │
│         ▼                                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │
│  │  PPT Ctrl   │  │  Mouse Ctrl │  │  Screen     │      │
│  │             │  │             │  │  Drawing     │      │
│  └─────────────┘  └─────────────┘  └─────────────┘      │
└─────────────────────────────────────────────────────────┘
```

### Core Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Hand Detection | MediaPipe HandLandmarker | Real-time 21 landmark detection |
| Gesture Recognition | ML model + rule fallback | Gesture classification (fist, open, scissor, etc.) |
| Position Smoothing | Kalman filter + EMA | Eliminate jitter, predict on loss |
| Shape Correction | OpenCV geometry analysis | Auto-correct hand-drawn shapes |
| GUI | PyQt6 | Floating window, settings panel, overlays |
| Mouse Control | Win32 API | SetCursorPos, mouse_event |
| Voice KWS | Sherpa-ONNX | Offline keyword detection |
| Voice ASR | Tencent Cloud | Online speech recognition |
| Audio Capture | sounddevice | Real-time audio streaming |

---

## ⚙️ Configuration Options

Click the ⚙️ settings button on the floating window to adjust:

| Setting | Description | Default |
|---------|-------------|---------|
| Target App | PowerPoint or WPS | WPS |
| Model Precision | Lite (faster) / Heavy (more accurate) | Heavy |
| Interaction Mode | presentation / mouse / draw | mouse |
| Gesture Cooldown | Minimum interval between gestures | 1000 ms |
| Mouse Sensitivity | Tracking sensitivity in mouse mode | 40% |
| Pen Width | Stroke width in drawing mode | 20 px |
| Edge Acceleration | Auto-speed boost near screen edges | Enabled |
| Voice Assistant | Select voice assistant app | Doubao |
| Action Mapping | Gesture-to-action mapping | See `config.json` |

Settings are automatically saved to `config.json`.

---

## 📁 Project Structure

```
AirControl/
├── app/
│   ├── main.py                    # CLI entry (OpenCV window)
│   ├── main_ui.py                 # GUI entry (PyQt6 floating window)
│   ├── config_manager.py          # Config file read/write
│   ├── mode_manager.py            # Mode manager (double-fist switch)
│   ├── drawing_overlay.py         # Drawing mode fullscreen canvas
│   ├── draw_toolbar.py            # Drawing toolbar
│   ├── mouse_cursor_overlay.py    # Mouse cursor overlay
│   ├── modes/
│   │   ├── base.py                # Mode base class (strategy pattern)
│   │   ├── presentation.py        # Presentation mode
│   │   ├── mouse_mode.py          # Mouse mode
│   │   └── draw_mode.py           # Drawing mode
│   ├── services/
│   │   ├── camera.py              # Camera service
│   │   ├── hand_tracker.py        # Hand landmark tracking (MediaPipe + Kalman)
│   │   ├── gesture_recognizer.py  # Gesture recognition & swipe detection
│   │   ├── inference_worker.py    # Async inference worker thread
│   │   ├── mouse_controller.py    # Mouse control (Win32 API)
│   │   ├── ppt_controller.py      # PPT/WPS control
│   │   ├── shape_recognizer.py    # Shape recognizer
│   │   ├── voice_assistant.py     # Voice assistant service
│   │   └── voice_command.py       # Voice command processing
│   └── voice_keywords/            # Voice keyword configs
├── models/
│   └── kws-zh-wenetspeech/       # Voice wake-up model
├── tests/                         # Unit tests
├── config.json                    # User configuration
├── requirements.txt               # Python dependencies
├── build.py                       # PyInstaller build script
├── AirControl.spec                # PyInstaller config
├── hand_landmarker.task           # MediaPipe hand model (7.8MB)
└── gesture_recognizer.task        # Gesture recognition model (8.4MB)
```

---

## 🧪 Testing

Run unit tests:

```bash
# Run all tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_edge_map.py
```

Test coverage:
- ✅ Configuration boundary checks
- ✅ Edge mapping algorithms
- ✅ Mouse controller compatibility
- ✅ Cursor overlay
- ✅ Voice assistant integration
- ✅ UI integration tests

---

## 📦 Build Executable

The project is configured with PyInstaller for one-click packaging:

```bash
python build.py
```

Output will be in the `dist/` directory with all dependencies and core model
files included. `hand_yolov8n.onnx` (AGPL-3.0) is not bundled in the release;
users who need the long-range YOLO engine should follow the download
instructions above. See [Model Provenance and Release Gate](MODEL_PROVENANCE.md).
Before release, run the hardware-free package self-test. Exit code `0` means
the models and native runtimes loaded successfully:

```powershell
$p = Start-Process .\dist\AirControl\AirControl.exe -ArgumentList "--self-test" -Wait -PassThru
$p.ExitCode
```

---

## 🤝 Contributing

Contributions are welcome! Whether it's bug reports, feature requests, or code contributions.

1. Fork the project
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit changes: `git commit -m 'feat: add your feature'`
4. Push to branch: `git push origin feature/your-feature`
5. Create a Pull Request

### Development Setup

```bash
# Clone project
git clone https://github.com/enixz/AirControl.git
cd AirControl

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run tests
python -m pytest tests/
```

### Code Standards

- Follow PEP 8 coding standards
- Add type annotations
- Write unit tests
- Update documentation

---

## 🙏 Acknowledgments

- [MediaPipe](https://mediapipe.dev/) - Hand detection and landmark tracking
- [Sherpa-ONNX](https://github.com/k2-fsa/sherpa-onnx) - Offline speech recognition engine
- [PyQt6](https://riverbankcomputing.com/software/pyqt/) - GUI framework
- [OpenCV](https://opencv.org/) - Computer vision library

---

## 📄 License

The repository **code** is licensed under the [Apache License 2.0](LICENSE).
The `hand_yolov8n.onnx` model (AGPL-3.0) is **not bundled** in releases;
users who need it download it separately. The release package therefore
contains only Apache-2.0 code and permissively licensed dependencies. See
[Model Provenance and Release Gate](MODEL_PROVENANCE.md) and the
[official Ultralytics licensing guidance](https://www.ultralytics.com/license)
for details.

```
Copyright 2026 AirControl

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

---

<div align="center">

**If you find this useful, please give us a ⭐ Star!**

[![Star History Chart](https://api.star-history.com/svg?repos=enixz/AirControl&type=Date)](https://star-history.com/#enixz/AirControl&Date)

</div>

---

<div align="center">

[⬆ Back to Top](#-aircontrol---gesture--voice-control-system)

</div>
