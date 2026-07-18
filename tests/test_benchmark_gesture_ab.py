import pytest

from benchmark_gesture_ab import _compute_freeze_observation


def test_release_frame_is_excluded_from_freeze_drift():
    metrics = _compute_freeze_observation(
        [False, True, False],
        [(0.0, 0.0), (10.0, 10.0), (500.0, 500.0)],
    )

    assert metrics["freeze_events"] == 1
    assert metrics["freeze_evaluable_events"] == 0
    assert metrics["freeze_observed_frames"] == 0
    assert metrics["freeze_event_max_drift_mean_px"] == 0.0


def test_only_continuing_pinch_frames_contribute_to_drift():
    metrics = _compute_freeze_observation(
        [False, True, True, True, False],
        [(0.0, 0.0), (10.0, 10.0), (13.0, 14.0), (16.0, 18.0), (900.0, 900.0)],
    )

    assert metrics["freeze_events"] == 1
    assert metrics["freeze_evaluable_events"] == 1
    assert metrics["freeze_observed_frames"] == 2
    assert metrics["freeze_event_max_drift_mean_px"] == pytest.approx(10.0)
    assert metrics["freeze_event_max_drift_p95_px"] == pytest.approx(10.0)
