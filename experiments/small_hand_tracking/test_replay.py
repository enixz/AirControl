"""Regression tests for claims made by the offline evaluator."""
import json
import time

import numpy as np
import pytest

from experiments.small_hand_tracking.replay import (
    ReplayClock, TrackerReplay, algorithm_time, encode_hands, summarize,
)
from experiments.small_hand_tracking.run_replay import read_metadata, variant_config


def hand(x=100):
    return [[i, float(x + i * 2), float(150 + i * 3), 0.] for i in range(21)]


def test_prediction_and_primary_identity_are_not_invented():
    encoded = encode_hands([hand(), hand(200)], [
        {"track_key": "Right", "predicted": True, "ml_label": "Victory"},
        {"track_key": "Left", "predicted": False, "ml_label": "Thumb_Up"},
    ])
    assert encoded[0]["detected_fresh"] is False
    assert encoded[1]["detected_fresh"] is True
    assert encoded[0]["track_id"] == "Right"
    assert encoded[0]["gesture_label"] == "Victory"


@pytest.mark.parametrize("gestures", [[], [{}], [{"predicted": "false"}], [{"predicted": False}, {}]])
def test_missing_or_misaligned_metadata_is_unknown(gestures):
    encoded = encode_hands([hand()], gestures)
    assert encoded[0]["detected_fresh"] is None
    assert encoded[0]["freshness"] == "unknown"


def test_model_calls_are_monotonic_without_advancing_source_time():
    clock = ReplayClock()
    clock.set(1000.)
    assert [clock.model_timestamp_ms() for _ in range(3)] == [1000000, 1000001, 1000002]
    assert clock.time() == 1000.
    clock.set(1000.04)
    assert clock.model_timestamp_ms() == 1000040
    with pytest.raises(ValueError):
        clock.set(1000.04)
    with pytest.raises(ValueError):
        clock.set(float("nan"))


def test_real_smoothing_independent_of_wall_clock_and_clock_restored():
    from services import smoothers
    original = smoothers.time
    real_perf = time.perf_counter

    def run(delay):
        clock = ReplayClock()
        smoother = smoothers.OneEuroSmoother()
        values = []
        for i in range(5):
            time.sleep(delay)
            clock.set(1000 + i / 25)
            with algorithm_time(clock):
                assert time.perf_counter is real_perf
                values.append(smoother.update(hand(100 + i * 12)))
        return values

    assert run(0) == run(.003)
    assert smoothers.time is original
    clock = ReplayClock()
    clock.set(1000)
    with pytest.raises(RuntimeError), algorithm_time(clock):
        raise RuntimeError("inference failure")
    assert smoothers.time is original


def row(t, hands):
    return {"source_timestamp": t, "hands": hands, "raw_hands": [], "used_crop_zoom": False,
            "events": [], "upscalers": [], "tracker_ms": 10, "model_calls": [], "sr_calls": []}


def test_secondary_fresh_hand_does_not_mask_primary_gap():
    fresh = encode_hands([hand()], [{"predicted": False}])
    mixed = encode_hands([hand(), hand()], [{"predicted": True}, {"predicted": False}])
    rows = [row(0., fresh), row(.1, mixed), row(.3, []), row(.5, fresh), row(.6, [])]
    summary = summarize(rows)
    assert summary["fresh_primary_output_rate"] == .4
    assert summary["fresh_any_output_rate"] == .6
    assert summary["no_fresh_primary_intervals"][0]["duration_ms"] == pytest.approx(400)
    assert summary["no_fresh_primary_intervals"][-1]["right_censored"] is True
    assert summary["reacquisition_ms"] is None
    assert summary["visible_hand_recall"] is None


def test_empty_or_unlabelled_output_cannot_claim_accuracy():
    assert summarize([])["fresh_primary_output_rate"] is None
    result = summarize([row(0, []), row(.1, [])])
    assert result["fresh_primary_output_rate"] == 0
    assert result["visible_hand_recall"] is None
    assert result["click_success_rate"] is None


@pytest.mark.parametrize("rows", [[], [{"i": 0}], [{"i": 0, "t": 1}, {"i": 1, "t": 1}], [{"i": 3, "t": 1}]])
def test_invalid_metadata_fails_instead_of_guessing_timestamps(tmp_path, rows):
    path = tmp_path / "meta.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    with pytest.raises(ValueError):
        read_metadata(path)


def test_ablation_overrides_do_not_modify_base_config():
    config = {"engine_auto_switch": True, "hand_prediction_enabled": True, "nested": {"x": 1}}
    result = variant_config(config, "B")
    assert result["zoom_sr_engine"] == "none"  # literal string, not JSON null
    assert result["hand_prediction_enabled"] is False
    assert config["hand_prediction_enabled"] is True
    assert variant_config(config, "current_tracker") == config


@pytest.fixture
def fake_tracker_factory(monkeypatch):
    from services.base_hand_tracker import BaseHandTracker
    from services import hand_tracker_factory

    class FakeTracker(BaseHandTracker):
        engine_name = "fake"

        def close(self):
            self._sr.release()

        def _next_mp_timestamp(self):
            return 0

        def _run_recognizer(self, frame):
            self._next_mp_timestamp()
            return None

        def _detect(self, frame):
            self._run_recognizer(frame)
            if frame[0, 0, 0] == 0:
                return [], [], []
            return [hand(int(frame[0, 0, 0]))], [{"label": "OTHER", "score": 1., "handedness": "Right", "bbox_area": 2400.}], []

        def _detect_crop_zoom(self, frame, center, size):
            return self._detect(frame)

    monkeypatch.setattr(hand_tracker_factory, "create_hand_tracker", lambda **kwargs: FakeTracker(config=kwargs["config"]))


def test_tracker_observation_marks_ghosts_and_keeps_raw_separate(fake_tracker_factory):
    replay = TrackerReplay({"long_range_enabled": False, "hand_prediction_enabled": True})
    try:
        found = replay.infer(np.full((480, 640, 3), 100, dtype=np.uint8), 1000.)
        lost = replay.infer(np.zeros((480, 640, 3), dtype=np.uint8), 1000.04)
        assert found["raw_hands"][0]["detected_fresh"] is True
        assert found["hands"][0]["detected_fresh"] is True
        assert lost["raw_hands"] == []
        assert lost["hands"][0]["predicted"] is True
        assert lost["hands"][0]["track_id"] == found["hands"][0]["track_id"]
        assert lost["model_calls"][0]["model_timestamp_ms"] == 1000040
        assert lost["auto_switch_executed"] is False
        assert lost["operation_events"] is None
    finally:
        replay.close()


@pytest.mark.parametrize("grace", [False, True])
def test_search_grace_retains_only_search_and_expires(fake_tracker_factory, grace):
    replay = TrackerReplay({"long_range_enabled": True, "hand_prediction_enabled": False,
                            "experiment_search_grace": grace})
    try:
        replay.infer(np.full((480, 640, 3), 100, dtype=np.uint8), 1000.)
        replay.tracker._crop_zoom_mode = True
        replay.tracker._current_crop_center = (120., 180.)
        replay.tracker._current_crop_size = 200.
        lost = replay.infer(np.zeros((480, 640, 3), dtype=np.uint8), 1000.04)
        assert lost["hands"] == []
        assert lost["raw_hands"] == []
        assert lost["after"]["primary_track_id"] is None
        assert lost["after"]["zoom_on"] is grace
        # Time deadline must work even when smoothers already got reset.
        expired = replay.infer(np.zeros((480, 640, 3), dtype=np.uint8), 1000.5)
        assert expired["after"]["zoom_on"] is False
        assert expired["hands"] == []
    finally:
        replay.close()


def test_search_grace_still_obeys_miss_frame_limit(fake_tracker_factory):
    replay = TrackerReplay({"long_range_enabled": True, "hand_prediction_enabled": False,
                            "experiment_search_grace": True, "zoom_miss_frames": 3})
    try:
        replay.infer(np.full((480, 640, 3), 100, dtype=np.uint8), 1000.)
        replay.tracker._crop_zoom_mode = True
        replay.tracker._current_crop_center = (120., 180.)
        replay.tracker._current_crop_size = 200.
        results = [replay.infer(np.zeros((480, 640, 3), dtype=np.uint8), 1000. + i * .03) for i in range(1, 4)]
        assert [r["after"]["zoom_on"] for r in results] == [True, True, False]
        assert all(r["hands"] == [] for r in results)
    finally:
        replay.close()


def test_native_retry_only_on_miss_rate_limited_and_restores_width():
    from types import SimpleNamespace
    from experiments.small_hand_tracking.native_rescue import install_native_rescue
    calls = []
    tracker = SimpleNamespace(_inference_max_width=720)
    clock = ReplayClock()
    def original(frame):
        calls.append(tracker._inference_max_width)
        return ([hand()], [], []) if tracker._inference_max_width == 0 else ([], [], [])
    tracker._detect = original
    replay = SimpleNamespace(tracker=tracker, clock=clock, frame_log={},
                             _replace=lambda obj, name, func: setattr(obj, name, func))
    install_native_rescue(replay)
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    clock.set(1000.)
    assert tracker._detect(frame)[0]
    assert tracker._inference_max_width == 720
    clock.set(1000.1)
    assert tracker._detect(frame)[0] == []
    clock.set(1000.3)
    assert tracker._detect(frame)[0]
    assert calls == [720, 0, 720, 720, 0]
    assert len(replay.frame_log["native_rescue"]) == 2


def test_native_retry_exception_cannot_change_following_frame_resolution():
    from types import SimpleNamespace
    from experiments.small_hand_tracking.native_rescue import install_native_rescue
    tracker = SimpleNamespace(_inference_max_width=720)
    def original(frame):
        if tracker._inference_max_width == 0:
            raise RuntimeError("model failed")
        return [], [], []
    tracker._detect = original
    clock = ReplayClock()
    clock.set(1000.)
    replay = SimpleNamespace(tracker=tracker, clock=clock, frame_log={},
                             _replace=lambda obj, name, func: setattr(obj, name, func))
    install_native_rescue(replay)
    with pytest.raises(RuntimeError):
        tracker._detect(np.zeros((1080, 1920, 3), dtype=np.uint8))
    assert tracker._inference_max_width == 720


@pytest.mark.parametrize("width_limit,frame_width,found", [(720, 1920, True), (720, 640, False), (0, 1920, False)])
def test_native_retry_avoids_redundant_inference(width_limit, frame_width, found):
    from types import SimpleNamespace
    from experiments.small_hand_tracking.native_rescue import install_native_rescue
    calls = []
    tracker = SimpleNamespace(_inference_max_width=width_limit)
    def original(frame):
        calls.append(tracker._inference_max_width)
        return ([hand()] if found else []), [], []
    tracker._detect = original
    clock = ReplayClock()
    clock.set(1000.)
    replay = SimpleNamespace(tracker=tracker, clock=clock, frame_log={},
                             _replace=lambda obj, name, func: setattr(obj, name, func))
    install_native_rescue(replay)
    tracker._detect(np.zeros((480, frame_width, 3), dtype=np.uint8))
    assert calls == [width_limit]
    assert replay.frame_log == {}
