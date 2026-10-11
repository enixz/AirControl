"""Real tracker output must not contain a duplicate primary after rekeying."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from services.base_hand_tracker import BaseHandTracker


def hand(x, y=400., handedness="Right"):
    points = [[i, x + i % 5 * 5., y + i // 5 * 6., 0.] for i in range(21)]
    return points, dict(handedness=handedness, handedness_score=.9,
                        label="OTHER", ml_label="None", score=.9, bbox_area=480.)


class ScriptedTracker(BaseHandTracker):
    engine_name = "scripted"

    def __init__(self):
        super().__init__(dominant_hand="Right", config={"hand_prediction_enabled": True})
        self.detections = []

    def _detect(self, frame):
        return ([h[0] for h in self.detections], [dict(h[1]) for h in self.detections], [])

    def _detect_crop_zoom(self, *args):
        raise AssertionError("This identity regression must not depend on zoom")

    def feed(self, *hands):
        self.detections = hands
        return self.find_hands(np.zeros((1080, 1920, 3), dtype=np.uint8), draw=False)[1:]


@pytest.mark.parametrize("locations,keep_ghost", [((405., 1400.), False), ((405., 415.), True), ((1000., 1400.), True), ((1400., 405.), True)])
def test_rekeying_only_retires_an_unambiguous_duplicate(locations, keep_ghost):
    tracker = ScriptedTracker()
    tracker.feed(hand(400.))
    old_smoother = tracker.smoothers["Primary"]
    old_points = [p[:] for p in old_smoother.last_landmarks]
    _, gestures = tracker.feed(*(hand(x) for x in locations))
    assert sum(not g["predicted"] for g in gestures) == 2
    assert any(g["track_key"] == "Primary" and g["predicted"] for g in gestures) is keep_ghost
    if not keep_ghost:
        assert gestures[0]["predicted"] is False
        assert gestures[0].get("previous_track_key") == "Primary"
        assert "previous_track_key" not in gestures[1]
        # No new numbering and no destructive reset of the old smoother.
        assert {g["track_key"] for g in gestures} == {"Left", "Right"}
        assert old_smoother.initialized
        assert old_smoother.last_landmarks == old_points
        _, next_gestures = tracker.feed(*(hand(x) for x in locations))
        assert all("previous_track_key" not in g for g in next_gestures)
    else:
        assert all("previous_track_key" not in g for g in gestures)


def test_a_real_primary_dropout_keeps_primary_and_secondary_separate():
    tracker = ScriptedTracker()
    tracker.feed(hand(400., handedness="Right"), hand(1400., handedness="Left"))
    assert tracker._primary_track_key == "Right"
    _, gestures = tracker.feed(hand(1400., handedness="Left"))
    assert gestures[0]["track_key"] == "Right"
    assert gestures[0]["predicted"] is True
    assert gestures[1]["track_key"] == "Left"
    assert gestures[1]["predicted"] is False


def test_single_hand_disappearance_still_uses_prediction():
    tracker = ScriptedTracker()
    tracker.feed(hand(400.))
    _, gestures = tracker.feed()
    assert len(gestures) == 1
    assert gestures[0]["track_key"] == "Primary"
    assert gestures[0]["predicted"] is True
