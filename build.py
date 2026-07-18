import os
import re
import shutil
import subprocess
import sys
import tempfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from packaging.version import Version

ROOT = Path(__file__).resolve().parent
MIN_PYINSTALLER_VERSION = Version("6.10")


def _get_app_version():
    """从 app/version.py 读取版本号（单一来源）。"""
    sys.path.insert(0, str(ROOT / "app"))
    from version import __version__
    return __version__


def _render_windows_version_info(app_version):
    """Render the PyInstaller VSVersionInfo resource from the app version."""
    parsed = Version(app_version)
    numeric = tuple(parsed.release[:4])
    file_version = numeric + (0,) * (4 - len(numeric))
    dotted_version = ".".join(str(part) for part in parsed.release)
    return f"""# UTF-8
VSVersionInfo(
  ffi=FixedFileInfo(
    filevers={file_version!r},
    prodvers={file_version!r},
    mask=0x3f,
    flags=0x0,
    OS=0x40004,
    fileType=0x1,
    subtype=0x0,
    date=(0, 0)
  ),
  kids=[
    StringFileInfo([
      StringTable(
        u'040904B0',
        [
          StringStruct(u'CompanyName', u'AirControl'),
          StringStruct(u'FileDescription', u'AirControl Gesture and Voice Control'),
          StringStruct(u'FileVersion', u'{dotted_version}'),
          StringStruct(u'InternalName', u'AirControl'),
          StringStruct(u'OriginalFilename', u'AirControl.exe'),
          StringStruct(u'ProductName', u'AirControl'),
          StringStruct(u'ProductVersion', u'{dotted_version}')
        ]
      )
    ]),
    VarFileInfo([VarStruct(u'Translation', [1033, 1200])])
  ]
)
"""


def _check_version_consistency():
    """检查 README badge 与 version.py 是否一致，不一致则警告。"""
    app_ver = _get_app_version()
    for readme_name in ("README.md", "README_EN.md"):
        readme_path = ROOT / readme_name
        if not readme_path.exists():
            continue
        content = readme_path.read_text(encoding="utf-8")
        # 匹配 badge 中的 v1.3.0 格式
        match = re.search(r"Release-v(\d+\.\d+\.\d+)", content)
        if match and match.group(1) != app_ver:
            print(
                f"[WARN] {readme_name} badge 版本({match.group(1)}) "
                f"与 version.py({app_ver}) 不一致，请更新"
            )


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
    _check_version_consistency()
    print(f"开始打包 AirControl v{_get_app_version()}...")
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
    with tempfile.TemporaryDirectory(prefix="aircontrol-version-") as temp_dir:
        version_file = Path(temp_dir) / "version_info.txt"
        version_file.write_text(
            _render_windows_version_info(_get_app_version()),
            encoding="utf-8",
        )
        env = os.environ.copy()
        env["AIRCONTROL_VERSION_FILE"] = str(version_file)
        subprocess.run(cmd, check=True, cwd=str(ROOT), env=env)
    print("\n打包完成: dist/AirControl/AirControl.exe")


if __name__ == "__main__":
    build()
