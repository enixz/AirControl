"""A proven same-hand rekey preserves evidence, but fresh release still wins."""
from unittest.mock import patch

import pytest
from test_pinch_freeze import _features_for, _make_landmarks, _make_mouse_mode


@pytest.fixture
def mode_parts():
    mode, mouse, recognizer = _make_mouse_mode()
    mode.on_enter()
    mode._watchdog_stop.set()
    mode._watchdog_thread.join()
    yield mode, mouse, recognizer
    mode.on_exit()


def feed(parts, now, key, pinch=True, previous=None, predicted=False, distant=False):
    mode, _, recognizer = parts
    features = _features_for(pinch)
    if distant:
        features["hand_width"] = 40.
    recognizer.get_hand_features.return_value = features
    gesture = dict(track_key=key, predicted=predicted, ml_label="None")
    if previous is not None:
        gesture["previous_track_key"] = previous
    with patch("modes.mouse_mode.time.monotonic", return_value=now):
        mode.handle([_make_landmarks(pinch)], [gesture], 1920, 1080)


def test_same_hand_rekey_does_not_interrupt_drag(mode_parts):
    mode, mouse, recognizer = mode_parts
    feed(mode_parts, 1., "Primary")
    recognizer.reset_hand_features.reset_mock()
    feed(mode_parts, 1.04, "Left", previous="Primary")
    assert mode._is_left_holding
    assert mode._left_fsm.last_transition_reason == "press_confirmed"
    mouse.left_down.assert_called_once()
    mouse.left_up.assert_not_called()
    recognizer.reset_hand_features.assert_not_called()
    assert mode._feature_track_key == "Left"


def test_same_hand_rekey_keeps_distant_press_confirmation(mode_parts):
    mode, mouse, _ = mode_parts
    feed(mode_parts, 1., "Primary", distant=True)
    mouse.left_down.assert_not_called()
    feed(mode_parts, 1.06, "Left", previous="Primary", distant=True)
    mouse.left_down.assert_called_once()
    assert mode._is_left_holding


def test_fresh_open_releases_even_on_same_hand_rekey(mode_parts):
    mode, mouse, _ = mode_parts
    feed(mode_parts, 1., "Primary")
    feed(mode_parts, 1.04, "Left", pinch=False, previous="Primary")
    mouse.left_up.assert_called_once()
    assert not mode._is_left_holding
    assert mode._left_fsm.last_transition_reason == "release_clear_open"


def test_distant_new_hand_cannot_inherit_the_old_hands_press_evidence(mode_parts):
    mode, mouse, _ = mode_parts
    feed(mode_parts, 1., "Primary", distant=True)
    feed(mode_parts, 1.06, "Primary", distant=True)
    mouse.left_down.assert_called_once()
    feed(mode_parts, 1.10, "Left", distant=True)
    mouse.left_up.assert_called_once()
    assert not mode._is_left_holding
    assert mode._left_fsm.press_evidence_since == 1.10
    feed(mode_parts, 1.16, "Left", distant=True)
    assert mouse.left_down.call_count == 2


def test_same_hand_rekey_keeps_right_click_latched_without_repeating_it(mode_parts):
    mode, mouse, recognizer = mode_parts
    features = _features_for(False)
    features["thumb_middle_pinch"] = True
    recognizer.get_hand_features.return_value = features
    for now, gesture in ((1., dict(track_key="Primary", predicted=False)),
                         (1.04, dict(track_key="Left", previous_track_key="Primary", predicted=False))):
        with patch("modes.mouse_mode.time.monotonic", return_value=now):
            mode.handle([_make_landmarks(False)], [gesture], 1920, 1080)
    mouse.right_click.assert_called_once()
    assert mode._right_fsm.state == "holding"


def test_same_hand_rekey_keeps_the_block_after_forced_release(mode_parts):
    mode, mouse, _ = mode_parts
    feed(mode_parts, 1., "Primary")
    mode._left_fsm.force_release("watchdog")
    mode._is_left_holding = False
    feed(mode_parts, 1.04, "Left", previous="Primary")
    mouse.left_down.assert_called_once()
    assert mode._left_fsm.blocked_until_open
    assert not mode._is_left_holding


@pytest.mark.parametrize("previous,predicted", [(None, False), ("Right", False), ("Primary", True)])
def test_real_switch_or_unproven_alias_clears_old_hold(mode_parts, previous, predicted):
    mode, mouse, recognizer = mode_parts
    feed(mode_parts, 1., "Primary")
    recognizer.reset_hand_features.reset_mock()
    # An open new hand cannot inherit the old hand's drag. A predicted alias
    # cannot certify a same-hand transition either.
    feed(mode_parts, 1.04, "Left", pinch=False, previous=previous, predicted=predicted)
    recognizer.reset_hand_features.assert_called_once()
    mouse.left_up.assert_called_once()
    assert not mode._is_left_holding
    assert mode._left_fsm.last_transition_reason == "reset_identity"
