from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from PyInstaller.utils.win32.versioninfo import load_version_info_from_text_file

ROOT = Path(__file__).resolve().parents[1]


def _load_build_module():
    spec = spec_from_file_location("aircontrol_build", ROOT / "build.py")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rendered_windows_version_info_is_parseable(tmp_path):
    build = _load_build_module()
    version_file = tmp_path / "version_info.txt"
    version_file.write_text(
        build._render_windows_version_info("1.4.0"),
        encoding="utf-8",
    )

    info = load_version_info_from_text_file(version_file)

    assert info.ffi.fileVersionMS == (1 << 16) | 4
    assert info.ffi.fileVersionLS == 0
    assert info.ffi.productVersionMS == (1 << 16) | 4
    assert info.ffi.productVersionLS == 0


def test_build_version_comes_from_app_version():
    build = _load_build_module()

    assert build._get_app_version() == "1.4.0"
