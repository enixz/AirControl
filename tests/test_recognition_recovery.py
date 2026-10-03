"""Regressions for small hands, fresh-primary rescue and bounded pinch release."""
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from services.engine_auto_switcher import (
    STATE_CAPTURE,
    STATE_FAR_TRACK,
    EngineAutoSwitcher,
    primary_hand_evidence,
)
from services.gesture_recognizer import GestureRecognizer
from services.hand_tracker import HandTracker
from test_hand_precision import DummyTracker
from test_pinch_freeze import _features_for, _make_landmarks, _make_mouse_mode
from test_pinch_hysteresis import make_hand_with_pinch_ratio


@pytest.mark.parametrize("width,height", [(640, 480), (1280, 720), (1920, 1080)])
def test_small_valid_hand_survives_at_all_resolutions(width, height):
    tracker = object.__new__(HandTracker)
    tracker._is_pure_landmarker = True
    points = [SimpleNamespace(x=.4 + i * .001, y=.3 + i * .001, z=0.) for i in range(21)]
    result = SimpleNamespace(hand_landmarks=[points], handedness=[])
    hands, gestures, _ = tracker._extract_results(result, width, height)
    assert len(hands) == 1
    assert 0 < gestures[0]["bbox_area"] < width * height * .005
    assert tracker._extract_results(result, width, height, min_area_ratio=.005)[0] == []


@pytest.mark.parametrize("bad", ["short", "flat", "nan", "inf"])
def test_invalid_model_geometry_is_rejected(bad):
    tracker = object.__new__(HandTracker)
    tracker._is_pure_landmarker = True
    points = [SimpleNamespace(x=i * .001, y=i * .001, z=0.) for i in range(21)]
    if bad == "short":
        points.pop()
    elif bad == "flat":
        for point in points:
            point.y = .3
    else:
        points[4].x = float(bad)
    result = SimpleNamespace(hand_landmarks=[points], handedness=[])
    assert tracker._extract_results(result, 1280, 720)[0] == []


def test_secondary_hand_does_not_mask_predicted_primary():
    hand = _make_landmarks()
    count, ratio, valid = primary_hand_evidence(
        [hand, hand], [{"predicted": True}, {"predicted": False}], 640 * 480,
    )
    assert (count, ratio, valid) == (1, 0., False)


@pytest.mark.parametrize("fps", [10, 30, 60])
def test_fleeting_reacquisition_does_not_postpone_rescue(fps):
    switcher = EngineAutoSwitcher(enabled=True)
    transitions = []
    for frame in range(2 * fps):
        now = frame / fps
        valid = frame % max(3, fps // 3) == 1
        target = switcher.update(int(valid), .01, now=now, primary_valid=valid)
        if target:
            transitions.append((now, target))
    assert len(transitions) == 1
    assert transitions[0][1] == STATE_CAPTURE
    assert 1.2 <= transitions[0][0] <= 1.2 + 1 / fps


def test_stable_primary_clears_loss_and_small_primary_gets_zoom():
    switcher = EngineAutoSwitcher(enabled=True)
    switcher.update(0, now=0, primary_valid=False)
    for i in range(1, 21):
        assert switcher.update(1, .02, now=i / 10, primary_valid=True) is None
    for i in range(6):
        target = switcher.update(1, .002, now=3 + i / 10, primary_valid=True)
    assert target == STATE_FAR_TRACK


def test_secondary_hand_does_not_block_capture_handoff():
    switcher = EngineAutoSwitcher(enabled=True, hand_frames=3)
    switcher.reset(STATE_CAPTURE)
    for i in range(3):
        target = switcher.update(2, .002, now=i / 10, primary_valid=True)
    assert target == STATE_FAR_TRACK


def test_prediction_metadata_stays_aligned_and_marked():
    tracker = DummyTracker()
    tracker._hand_prediction_enabled = True
    tracker.smoothers["Right"].update(_make_landmarks())
    tracker.smoothers["Left"].update(_make_landmarks())
    tracker._active_handedness = {"Left", "Right"}
    tracker.last_gestures = [{"track_key": "Right", "label": "R"},
                             {"track_key": "Left", "label": "L"}]
    # One track expires: its old metadata must not be paired with the other.
    with patch.object(tracker.smoothers["Right"], "predict", return_value=None):
        _, hands, gestures = tracker._handle_all_lost(
            np.zeros((480, 640, 3), dtype=np.uint8), False, False, 640, 480,
        )
    assert len(hands) == len(gestures) == 1
    assert gestures[0]["label"] == "L"
    assert gestures[0]["predicted"] is True


def test_pinch_bridges_jitter_but_releases_on_timeout_and_clear_open():
    recognizer = GestureRecognizer()
    recognizer.pinch_exit_hysteresis_enabled = True
    def sample(ratio, t):
        with patch("services.gesture_recognizer.time.monotonic", return_value=t):
            return recognizer.get_hand_features(make_hand_with_pinch_ratio(ratio))["thumb_index_pinch"]
    assert not sample(.42, 0)  # no wider entry threshold
    assert sample(.2, .1)
    assert sample(.44, .13)
    assert sample(.3, .16)  # short boundary excursion is bridged
    assert sample(.44, .2)
    assert not sample(.44, .29)  # sustained opening cannot stick
    assert sample(.2, .3)
    assert not sample(.6, .31)  # clear opening releases immediately
    recognizer.reset_hand_features()
    assert not sample(.38, .32)  # reacquisition cannot inherit hold hysteresis


def test_mouse_prediction_cannot_start_click_or_hold_indefinitely():
    mode, mouse, recognizer = _make_mouse_mode()
    mode.on_enter()
    try:
        hand = _make_landmarks(True)
        recognizer.get_hand_features.return_value = _features_for(True)
        with patch("modes.mouse_mode.time.monotonic", return_value=1.):
            mode.handle([hand], [{"predicted": True}], 640, 480)
        mouse.left_down.assert_not_called()
        with patch("modes.mouse_mode.time.monotonic", return_value=1.1):
            mode.handle([hand], [{}], 640, 480)
        mouse.left_down.assert_called_once()
        with patch("modes.mouse_mode.time.monotonic", return_value=1.2):
            mode.handle([hand], [{"predicted": True}], 640, 480)
        assert mode._is_left_holding
        with patch("modes.mouse_mode.time.monotonic", return_value=1.33):
            mode.handle([hand], [{"predicted": True}], 640, 480)
        assert not mode._is_left_holding
        mouse.left_up.assert_called_once()
        # Telemetry must not evaluate stateful geometry twice on frame 15.
        mode._frame_count = 14
        recognizer.get_hand_features.reset_mock()
        mode.handle([hand], [{}], 640, 480)
        recognizer.get_hand_features.assert_called_once()
    finally:
        mode.on_exit()


def test_distant_click_requires_fresh_confirmation_but_near_click_does_not():
    mode, mouse, recognizer = _make_mouse_mode()
    mode.on_enter()
    try:
        features = dict(_features_for(True), hand_width=20.)
        recognizer.get_hand_features.return_value = features
        hand = _make_landmarks(True)
        with patch("modes.mouse_mode.time.monotonic", return_value=1.):
            mode.handle([hand], [{}], 1280, 720)
        mouse.left_down.assert_not_called()
        with patch("modes.mouse_mode.time.monotonic", return_value=1.04):
            mode.handle([hand], [{}], 1280, 720)
        mouse.left_down.assert_not_called()
        with patch("modes.mouse_mode.time.monotonic", return_value=1.07):
            mode.handle([hand], [{}], 1280, 720)
        mouse.left_down.assert_called_once()
        mode.handle([], [], 1280, 720)
        features["hand_width"] = 80.
        # 丢手强制释放后：残留捏合不能立即重建第二次点击（第 3 步规则 5），
        # 需要一次明确张开的帧解锁
        mode.handle([hand], [{}], 1280, 720)
        assert mouse.left_down.call_count == 1
        recognizer.get_hand_features.return_value = dict(_features_for(False), hand_width=80.)
        mode.handle([_make_landmarks(False)], [{}], 1280, 720)
        recognizer.get_hand_features.return_value = features
        mode.handle([hand], [{}], 1280, 720)
        assert mouse.left_down.call_count == 2
    finally:
        mode.on_exit()
