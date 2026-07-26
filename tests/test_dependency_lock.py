from pathlib import Path

from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[1]


def _locked_requirements():
    requirements = []
    for raw_line in (ROOT / "requirements.lock").read_text(
        encoding="utf-8"
    ).splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#"):
            line = line.split(" #", 1)[0].rstrip()
            requirements.append(Requirement(line))
    return requirements


def _version_for(package_name, python_version):
    environment = {"python_version": python_version}
    matches = [
        requirement
        for requirement in _locked_requirements()
        if requirement.name.lower() == package_name.lower()
        and (
            requirement.marker is None
            or requirement.marker.evaluate(environment)
        )
    ]
    assert len(matches) == 1
    return str(matches[0].specifier)


def test_lock_selects_python_310_compatible_numpy():
    assert _version_for("numpy", "3.10") == "==2.2.6"


def test_lock_selects_current_numpy_for_python_311_plus():
    assert _version_for("numpy", "3.11") == "==2.3.0"
    assert _version_for("numpy", "3.12") == "==2.3.0"


def test_lock_selects_python_310_compatible_directml_runtime():
    assert _version_for("onnxruntime-directml", "3.10") == "==1.23.0"
    assert _version_for("onnxruntime-directml", "3.11") == "==1.24.4"
    assert _version_for("onnxruntime-directml", "3.12") == "==1.24.4"


def test_lock_declares_utf8_for_legacy_windows_pip():
    first_line = (ROOT / "requirements.lock").read_bytes().splitlines()[0]
    assert first_line == b"# -*- coding: utf-8 -*-"


def test_ci_development_requirements_extend_the_release_lock():
    first_directive = next(
        line.strip()
        for line in (ROOT / "requirements-dev.txt").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    )
    assert first_directive == "-r requirements.lock"
