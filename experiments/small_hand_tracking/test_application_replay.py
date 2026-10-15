"""Verify real application wiring with deterministic detection, no native models."""
import time
from copy import deepcopy

import numpy as np
import pytest

from .application_replay import ApplicationReplay, application_time
from .replay import ReplayClock


@pytest.fixture
def application_fixture(monkeypatch, tmp_path):
    import orchestrator
    from config_manager import ConfigManager
    from PyQt6.QtCore import QCoreApplication
    from services.base_hand_tracker import BaseHandTracker

    qt = QCoreApplication.instance() or QCoreApplication([])
    config = ConfigManager(str(tmp_path / "config.json")).config
    config.update(interaction_mode="mouse", engine_auto_switch=True, dominant_hand="Auto",
                  long_range_enabled=False, hand_prediction_enabled=False, zoom_sr_engine="none",
                  voice_command_enabled=False, draw_record_trace=False)

    class DeterministicTracker(BaseHandTracker):
        @property
        def engine_name(self):
            return self._test_engine

        def _detect(self, frame):
            points = [[i, 100. + (i % 5) * 5, 100. + (i // 5) * 5, 0.] for i in range(21)]
            return [points], [dict(label="OTHER", ml_label="None", handedness="Right", score=1., bbox_area=400.)], [None]

        def _detect_crop_zoom(self, frame, center, size):
            return self._detect(frame)

        def close(self):
            pass

    def factory(engine="mediapipe", **kwargs):
        tracker = DeterministicTracker(config=kwargs["config"], dominant_hand=kwargs["dominant_hand"])
        tracker._test_engine = engine
        return tracker

    monkeypatch.setattr(orchestrator, "create_hand_tracker", factory)
    yield config, qt


def semantic_rows(config, delay=0):
    replay = ApplicationReplay(config, 1000.)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    rows = []
    try:
        for i in range(8):
            if delay:
                time.sleep(delay)
            row = replay.infer(frame.copy(), 1000. + i * .2)
            rows.append({key: row[key] for key in ("source_timestamp", "hands", "fsm_before", "fsm_state", "events", "mode_gesture", "mode_state")})
        return rows
    finally:
        replay.close()


def test_real_worker_handoff_preserves_hint_and_primary(application_fixture):
    config, _ = application_fixture
    original = deepcopy(config)
    rows = semantic_rows(config)
    assert rows[0]["fsm_state"] == "near"
    assert rows[3]["fsm_state"] == "far_track"
    swap = [event for row in rows for event in row["events"] if event["kind"] == "tracker_swap"]
    assert len(swap) == 1
    assert swap[0]["seeded_zoom"] is True
    assert rows[4]["hands"][0]["track_id"] == rows[3]["hands"][0]["track_id"]
    assert config == original


@pytest.mark.parametrize("mode", ["mouse", "draw"])
def test_mode_and_fsm_semantics_do_not_follow_wall_clock(application_fixture, mode):
    config, _ = application_fixture
    config["interaction_mode"] = mode
    assert semantic_rows(config) == semantic_rows(config, .002)


def test_application_clock_restored_on_exception():
    import orchestrator
    from services import inference_worker
    clock = ReplayClock()
    clock.set(123.)
    original = orchestrator.time
    with pytest.raises(RuntimeError):
        with application_time(clock):
            assert orchestrator.time.time() == 123.
            assert inference_worker.time is time
            raise RuntimeError()
    assert orchestrator.time is original


def test_private_config_mode_switch_never_saves_input(application_fixture):
    config, _ = application_fixture
    original = deepcopy(config)
    replay = ApplicationReplay(config, 1000.)
    try:
        with application_time(replay.clock):
            replay.app.mode_manager.cycle_mode()
        assert replay.config.get("interaction_mode") != original["interaction_mode"]
        assert config == original
    finally:
        replay.close()


def test_responsive_crop_changes_center_but_preserves_size(application_fixture):
    from contextlib import ExitStack

    import orchestrator

    from .responsive_crop import install_responsive_crop
    config, _ = application_fixture
    tracker = orchestrator.create_hand_tracker(config=config, dominant_hand="Auto")
    tracker._crop_zoom_mode = True
    tracker._current_crop_center = (200., 200.)
    tracker._current_crop_size = 200.
    tracker._last_hint_center = (300., 200.)
    tracker._last_hint_size = 60.
    with ExitStack() as patches:
        install_responsive_crop(tracker, patches)
        tracker._compute_crop_viewport(640, 480)
    assert tracker._current_crop_center[0] == pytest.approx(255.)
    assert tracker._current_crop_size == pytest.approx(.85 * 200 + .15 * 150)
    assert not tracker.last_gestures


@pytest.mark.parametrize("separation,retired", [(400., True), (15., False)])
def test_identity_retires_alias_only_with_clear_separation(application_fixture, separation, retired):
    from contextlib import ExitStack

    import orchestrator

    from .identity_continuity import install_identity_continuity
    config, _ = application_fixture
    tracker = orchestrator.create_hand_tracker(config=config, dominant_hand="Auto")
    tracker._prev_wrist_keys = [((100., 100.), "Primary")]
    tracker._active_handedness = {"Primary"}
    hands = [[[i, 102. + separation * index + (i % 5) * 5, 100. + (i // 5) * 5, 0.] for i in range(21)] for index in range(2)]
    with ExitStack() as patches:
        install_identity_continuity(tracker, patches)
        keys = tracker._assign_smoother_keys(hands, [dict(handedness="Left"), dict(handedness="Right")], 640, 480)
    assert keys == ["Left", "Right"]
    assert len(set(keys)) == 2
    assert tracker._prev_wrist_keys[0][1] == "Left"
    assert ("Primary" not in tracker._active_handedness) is retired


def test_identity_does_not_promote_secondary_after_primary_loss(application_fixture):
    from contextlib import ExitStack

    import orchestrator

    from .identity_continuity import install_identity_continuity
    config, _ = application_fixture
    tracker = orchestrator.create_hand_tracker(config=config, dominant_hand="Auto")
    tracker._prev_wrist_keys = [((100., 100.), "Primary"), ((500., 100.), "Right")]
    hand = [[i, 500. + (i % 5) * 5, 100. + (i // 5) * 5, 0.] for i in range(21)]
    with ExitStack() as patches:
        install_identity_continuity(tracker, patches)
        assert tracker._assign_smoother_keys([hand], [dict(handedness="Right")], 640, 480) == ["Right"]
