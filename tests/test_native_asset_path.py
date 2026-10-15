import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import runtime_paths


def test_unicode_asset_cache_tracks_content_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime_paths.sys, "platform", "win32")
    monkeypatch.setattr(runtime_paths.tempfile, "gettempdir", lambda: str(tmp_path))
    original = tmp_path / "中文模型.onnx"
    original.write_bytes(b"first-model")
    first = Path(runtime_paths.native_asset_path(original))
    assert str(first).isascii()
    assert first.read_bytes() == original.read_bytes()
    original.write_bytes(b"new-model")
    second = Path(runtime_paths.native_asset_path(original))
    assert second != first
    assert second.read_bytes() == b"new-model"
    assert first.read_bytes() == b"first-model"


def test_ascii_path_needs_no_copy():
    path = "C:/AirControl/models/absent.onnx"
    assert runtime_paths.native_asset_path(path) == path
