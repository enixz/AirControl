import json
from pathlib import Path

import pytest

from .historical_baseline import before_fixes
from .application_replay import application_time
from .replay import ROOT, ReplayClock
from .run_replay import read_metadata


def test_capture_gaps_keep_original_time_and_index(tmp_path):
    path = tmp_path / "meta.jsonl"
    rows = [{"i": i, "t": t} for i, t in [(0, 100.), (2, 100.2), (5, 100.5)]]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    assert read_metadata(path, allow_capture_gaps=True) == rows
    with pytest.raises(ValueError, match="Non-contiguous"):
        read_metadata(path)
    rows[2]["i"] = 2
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="Non-increasing capture"):
        read_metadata(path, allow_capture_gaps=True)


def test_historical_method_clock_and_exception_restore():
    snapshot = ROOT / "artifacts/small_hand_tracking/app_far_development_20261008"
    if not snapshot.exists():
        pytest.skip("Local source snapshot not available")
    from modes.draw_mode import DrawMode
    from services.base_hand_tracker import BaseHandTracker
    from modes.mouse_mode import MouseMode
    original_draw, original_assign = DrawMode.handle, BaseHandTracker._assign_smoother_keys
    original_mouse = MouseMode.handle
    clock = ReplayClock()
    clock.set(1234.)
    with pytest.raises(RuntimeError, match="sentinel"):
        with before_fixes(snapshot):
            with application_time(clock):
                assert DrawMode.handle.__globals__["time"].time() == 1234.
                assert BaseHandTracker._assign_smoother_keys is not original_assign
                assert MouseMode.handle is not original_mouse
                raise RuntimeError("sentinel")
    assert DrawMode.handle is original_draw
    assert BaseHandTracker._assign_smoother_keys is original_assign
    assert MouseMode.handle is original_mouse
