import os
import sys
import shutil
import subprocess

def build():
    print("开始打包 AirControl...")
    
    # 清理旧的构建目录
    for dir_name in ['build', 'dist']:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)

    # MediaPipe 需要包含其内部的二进制和模型数据
    import mediapipe
    mediapipe_dir = os.path.dirname(mediapipe.__file__)
    
    # 构建 PyInstaller 命令
    # 注意：我们使用 --noconsole 隐藏控制台窗口
    # 使用 --add-data 将本地模型文件和 mediapipe 的内部资源打包进去
    cmd = [
        sys.executable,
        "-m", "PyInstaller",
        "--noconfirm",
        "--onedir",          # 使用单目录模式，比单文件模式启动快很多
        "--windowed",        # 隐藏命令行窗口
        "--name", "AirControl",
        "--add-data", f"{mediapipe_dir};mediapipe",
    ]
    
    # gesture_recognizer.task
    if os.path.exists("gesture_recognizer.task"):
        cmd.extend(["--add-data", "gesture_recognizer.task;."])
    elif os.path.exists(os.path.join("models", "gesture_recognizer.task")):
        cmd.extend(["--add-data", "models/gesture_recognizer.task;."])
    
    # 动态查找 python3x.dll，兼容不同 Python 版本
    python_dir = os.path.dirname(sys.executable)
    python_ver = f"python{sys.version_info.major}{sys.version_info.minor}.dll"
    python_dll = os.path.join(python_dir, python_ver)
    if not os.path.exists(python_dll):
        # fallback: 尝试查找任意 python3*.dll
        import glob
        candidates = glob.glob(os.path.join(python_dir, "python3*.dll"))
        if candidates:
            python_dll = candidates[0]
    if os.path.exists(python_dll):
        cmd.extend(["--add-binary", f"{python_dll};."])
    
    for model in ["hand_landmarker.task", "hand_landmarker_heavy.task", "hand_landmarker_full.task"]:
        if os.path.exists(model):
            cmd.extend(["--add-data", f"{model};."])
        elif os.path.exists(os.path.join("models", model)):
            cmd.extend(["--add-data", f"models/{model};."])
            
    cmd.append(os.path.join("app", "main_ui.py"))
    
    # 执行打包
    subprocess.run(cmd, check=True)
    
    print("\n打包完成！")
    print("你的可执行文件在: dist/AirControl/AirControl.exe")

if __name__ == "__main__":
    build()