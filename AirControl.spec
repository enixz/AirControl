# -*- mode: python ; coding: utf-8 -*-
import os
import sys

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

ROOT = os.path.abspath(SPECPATH)
CONSOLE_BUILD = os.environ.get("AIRCONTROL_BUILD_CONSOLE") == "1"

# build.py 从 app/version.py 生成合法的 Windows VSVersionInfo 临时文件。
# 显式要求该文件，避免把普通版本字符串误当成资源文件路径。
VERSION_FILE = os.environ.get("AIRCONTROL_VERSION_FILE")
if not VERSION_FILE or not os.path.isfile(VERSION_FILE):
    raise RuntimeError("请通过 python build.py 构建，以生成 Windows 版本资源")


def add_data_if_present(items, source, destination):
    path = os.path.join(ROOT, source)
    if os.path.exists(path):
        items.append((path, destination))


datas = []
binaries = []

for package in ("mediapipe", "sherpa_onnx"):
    datas += collect_data_files(package)
    binaries += collect_dynamic_libs(package)

add_data_if_present(datas, "config.json", ".")
add_data_if_present(datas, "gesture_recognizer.task", ".")
add_data_if_present(datas, os.path.join("models", "hand_landmarker.task"), "models")
# hagrid_yolo 引擎的 HaGRID YOLO 手部检测器（随仓库分发，打包进 models/）。
add_data_if_present(datas, os.path.join("models", "hand_yolov8n.onnx"), "models")
add_data_if_present(datas, os.path.join("models", "model_manifest.json"), "models")
for kws_file in (
    "encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx",
    "decoder-epoch-12-avg-2-chunk-16-left-64.onnx",
    "joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx",
    "tokens.txt",
):
    add_data_if_present(
        datas,
        os.path.join("models", "kws-zh-wenetspeech", kws_file),
        os.path.join("models", "kws-zh-wenetspeech"),
    )
add_data_if_present(
    datas,
    os.path.join("app", "voice_keywords", "keywords.txt"),
    os.path.join("app", "voice_keywords"),
)

# Optional assets are packaged when available in the build workspace.
add_data_if_present(datas, os.path.join("models", "sense-voice"), "models/sense-voice")
add_data_if_present(datas, "ESPCN_x2.pb", ".")
add_data_if_present(datas, "Real-ESRGAN_x2plus.onnx", ".")

a = Analysis(
    [os.path.join(ROOT, "app", "main_ui.py")],
    pathex=[os.path.join(ROOT, "app")],
    binaries=binaries,
    datas=datas,
    hiddenimports=[
        "mediapipe.tasks.python.vision",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "IPython",
        "jupyter",
        "pandas",
        "pytest",
        "scipy",
        "torch",
        "torchvision",
        "ultralytics",
    ],
    noarchive=False,
)

# A machine with both opencv-python and opencv-contrib-python installed can
# leave multiple versioned FFmpeg DLLs in cv2/. Keep only the newest one.
opencv_ffmpeg = [
    entry for entry in a.binaries
    if os.path.basename(entry[0]).lower().startswith("opencv_videoio_ffmpeg")
]
if len(opencv_ffmpeg) > 1:
    newest_ffmpeg = max(opencv_ffmpeg, key=lambda entry: os.path.basename(entry[0]))
    a.binaries = [
        entry for entry in a.binaries
        if entry not in opencv_ffmpeg or entry == newest_ffmpeg
    ]

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="AirControl",
    version=VERSION_FILE,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=CONSOLE_BUILD,
    disable_windowed_traceback=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="AirControl",
)
