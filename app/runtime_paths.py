"""Shared resource and writable-data paths for source and packaged builds."""

import hashlib
import os
import sys
import tempfile
from pathlib import Path

APP_NAME = "AirControl"


def project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resource_root():
    if getattr(sys, "frozen", False):
        return getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    return project_root()


def resource_path(*parts):
    return os.path.join(resource_root(), *parts)


def writable_data_dir():
    override = os.environ.get("AIRCONTROL_DATA_DIR")
    if override:
        root = os.path.abspath(os.path.expanduser(override))
    elif getattr(sys, "frozen", False):
        appdata = (
            os.environ.get("APPDATA")
            or os.environ.get("LOCALAPPDATA")
            or os.path.expanduser("~")
        )
        root = os.path.join(appdata, APP_NAME)
    else:
        root = project_root()
    os.makedirs(root, exist_ok=True)
    return root


def data_path(*parts):
    path = os.path.join(writable_data_dir(), *parts)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return path


def native_asset_path(path):
    """Return a native-loader-safe path for a read-only model or token asset.

    Some Windows C++ dependencies decode Unicode filenames using the ANSI
    codepage. Cache only affected files under an ASCII path, keyed by content.
    Configuration and logs remain in their original user data directory.
    """
    path = os.fspath(path)
    if sys.platform != "win32" or path.isascii():
        return path
    content = Path(path).read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    suffix = Path(path).suffix
    if not suffix.isascii():
        suffix = ".bin"
    candidates = [tempfile.gettempdir(), os.environ.get("PUBLIC", "")]
    for base in candidates:
        if not base or not os.path.abspath(base).isascii():
            continue
        cache = Path(base) / "AirControlNativeAssets"
        temporary = None
        try:
            cache.mkdir(parents=True, exist_ok=True)
            target = cache / (digest + suffix)
            if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == digest:
                return str(target)
            with tempfile.NamedTemporaryFile(dir=cache, delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(content)
            os.replace(temporary, target)
            return str(target)
        except OSError:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    raise OSError("Unable to create an ASCII model cache; use an ASCII extraction path")

