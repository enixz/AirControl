import shutil
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from packaging.version import Version


ROOT = Path(__file__).resolve().parent
MIN_PYINSTALLER_VERSION = Version("6.10")


def _check_build_tools():
    try:
        installed = Version(version("pyinstaller"))
    except PackageNotFoundError as exc:
        raise RuntimeError(
            "PyInstaller 未安装，请先执行: python -m pip install -r requirements-dev.txt"
        ) from exc
    if installed < MIN_PYINSTALLER_VERSION:
        raise RuntimeError(
            f"PyInstaller {installed} 过旧，至少需要 {MIN_PYINSTALLER_VERSION}；"
            "请执行: python -m pip install -r requirements-dev.txt"
        )


def build():
    _check_build_tools()
    print("开始打包 AirControl...")
    for directory in ("build", "dist"):
        path = ROOT / directory
        if path.exists():
            shutil.rmtree(path)

    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        str(ROOT / "AirControl.spec"),
    ]
    subprocess.run(cmd, check=True, cwd=str(ROOT))
    print("\n打包完成: dist/AirControl/AirControl.exe")


if __name__ == "__main__":
    build()
