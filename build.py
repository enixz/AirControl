import argparse
import hashlib
import json
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
MODEL_MANIFEST_PATH = ROOT / "models" / "model_manifest.json"


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


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_release_model_provenance(allow_unapproved=False):
    """Validate known bundled-model evidence before a release build.

    Packaging is release-safe by default. Local/CI development packaging must
    opt in explicitly, and its output is not approved for redistribution.
    """
    try:
        manifest = json.loads(MODEL_MANIFEST_PATH.read_text(encoding="utf-8"))
        model = manifest["models"]["hand_yolov8n.onnx"]
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"无法读取模型发布清单: {MODEL_MANIFEST_PATH}"
        ) from exc

    model_path = ROOT / "models" / "hand_yolov8n.onnx"
    if not model_path.is_file():
        raise RuntimeError(f"缺少受清单保护的模型: {model_path}")
    if _sha256(model_path) != model["sha256"]:
        raise RuntimeError(
            "hand_yolov8n.onnx 的 SHA-256 与 models/model_manifest.json 不一致；"
            "请重新完成来源、许可证和分发审批审计后更新清单。"
        )

    approved = model.get("redistribution_approved") is True
    has_evidence = bool(model.get("source_url")) and bool(
        model.get("approval_reference")
    )
    if approved and has_evidence:
        return

    message = (
        "YOLO 模型尚无可验证的分发许可。详见 MODEL_PROVENANCE.md；"
        "必须记录 source_url、approval_reference，并明确将 "
        "redistribution_approved 设为 true。"
    )
    if not allow_unapproved:
        raise RuntimeError(f"拒绝发布构建：{message}")
    print(f"[WARN] 不可分发的开发构建：{message}")


def build(development=False):
    _check_build_tools()
    _check_version_consistency()
    _check_release_model_provenance(allow_unapproved=development)
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
        help="build a distributable bundle (the default; provenance is enforced)",
    )
    mode.add_argument(
        "--development",
        action="store_true",
        help="build a local-only bundle even when model redistribution is unapproved",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    build(development=_parse_args().development)
