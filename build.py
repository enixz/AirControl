import argparse
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


def _check_release_model_excluded():
    """Verify that the AGPL-licensed YOLO model is NOT packaged into the release.

    hand_yolov8n.onnx carries an AGPL-3.0 licence string in its ONNX metadata,
    which conflicts with the repository's Apache-2.0 licence. The model is
    intentionally excluded from AirControl.spec. This check reads the spec
    file and fails the build if the model path reappears, preventing
    accidental redistribution without a provenance audit.
    """
    spec_path = ROOT / "AirControl.spec"
    spec_text = spec_path.read_text(encoding="utf-8")
    # Match the specific datas entry that packages the YOLO model.
    if "hand_yolov8n.onnx" in spec_text and 'add_data_if_present' in spec_text:
        # Check if it's in an active add_data_if_present call (not just a comment)
        for line in spec_text.splitlines():
            stripped = line.strip()
            if (
                stripped.startswith("add_data_if_present")
                and "hand_yolov8n.onnx" in stripped
            ):
                raise RuntimeError(
                    "AirControl.spec 尝试打包 hand_yolov8n.onnx，但该模型的 "
                    "AGPL-3.0 许可证与仓库 Apache-2.0 不兼容。如需分发，请先"
                    "完成 MODEL_PROVENANCE.md 中的来源审计与许可审批。当前"
                    "策略：模型不打包，用户按 README 指引自行下载。"
                )


def build(development=False):
    _check_build_tools()
    _check_version_consistency()
    _check_release_model_excluded()
    if development:
        print("[INFO] 开发构建：仅供本地测试，不可分发。")
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


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Build the AirControl Windows bundle")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--release",
        action="store_true",
        help="build a distributable bundle (the default; AGPL model is excluded)",
    )
    mode.add_argument(
        "--development",
        action="store_true",
        help="build a local-only bundle (label only; same output as --release)",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    build(development=_parse_args().development)
