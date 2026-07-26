from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
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


def test_release_build_rejects_unapproved_yolo_model():
    build = _load_build_module()

    with pytest.raises(RuntimeError, match="拒绝发布构建"):
        build._check_release_model_provenance()


def test_default_cli_mode_is_release_safe():
    build = _load_build_module()

    args = build._parse_args([])

    assert not args.development


def test_development_build_requires_explicit_flag():
    build = _load_build_module()

    args = build._parse_args(["--development"])

    assert args.development


def test_explicit_development_override_is_loud(capsys):
    build = _load_build_module()

    build._check_release_model_provenance(allow_unapproved=True)

    assert "不可分发的开发构建" in capsys.readouterr().out


def test_ci_checks_release_gate_before_explicit_development_build():
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )

    release_check = workflow.index("python build.py --release")
    development_build = workflow.index("python build.py --development")
    assert release_check < development_build
    assert "$global:LASTEXITCODE = 0" in workflow
