"""Evidence must reject ghosts, identity changes and inconsistent geometry."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "app"))
from .capture_handoff import CaptureHandoffEvidence


def sample(x=400., key="Primary", predicted=False, scale=1.):
    points = [[i, x + scale * (i % 5) * 5., 200. + scale * (i // 5) * 6.] for i in range(21)]
    return [points], [dict(track_key=key, predicted=predicted)]


def observe(evidence, now, *args, **kwargs):
    hands, gestures = sample(*args, **kwargs)
    return evidence.observe(hands, gestures, 1920 * 1080, now, 0.)


@pytest.mark.parametrize("fps", [10, 30, 60])
def test_consistent_primary_handoffs_by_time_instead_of_thirty_frames(fps):
    evidence = CaptureHandoffEvidence()
    times = [i / fps for i in range(1, fps + 1)]
    ready = next(t for t in times if observe(evidence, t))
    assert .5 <= ready <= .5 + 1 / fps


@pytest.mark.parametrize("bad", ["prediction", "loss", "other_key", "position_jump", "scale_jump", "gap", "nearby_other"])
def test_invalid_or_ambiguous_samples_restart_confirmation(bad):
    evidence = CaptureHandoffEvidence()
    for t in (.1, .2, .3, .4):
        assert not observe(evidence, t)
    hands, gestures = sample()
    now = .5
    if bad == "prediction":
        gestures[0]["predicted"] = True
    elif bad == "loss":
        hands, gestures = [], []
    elif bad == "other_key":
        gestures[0]["track_key"] = "Left"
    elif bad == "position_jump":
        hands, gestures = sample(x=1000.)
    elif bad == "scale_jump":
        hands, gestures = sample(scale=3.)
    elif bad == "gap":
        now = 1.
    else:
        extra_hands, extra_gestures = sample(x=410., key="Left")
        hands += extra_hands
        gestures += extra_gestures
    assert not evidence.observe(hands, gestures, 1920 * 1080, now, 0.)
    assert not observe(evidence, now + .1)


def test_certified_rekey_and_distant_secondary_do_not_restart_evidence():
    evidence = CaptureHandoffEvidence()
    for t in (.1, .2, .3, .4):
        observe(evidence, t)
    hands, gestures = sample(key="Left")
    gestures[0]["previous_track_key"] = "Primary"
    other_hands, other_gestures = sample(x=1400., key="Right")
    assert evidence.observe(hands + other_hands, gestures + other_gestures, 1920 * 1080, .5, 0.)


def test_a_new_capture_episode_discards_previous_evidence():
    evidence = CaptureHandoffEvidence()
    for t in (.1, .2, .3, .4):
        observe(evidence, t)
    hands, gestures = sample()
    assert not evidence.observe(hands, gestures, 1920 * 1080, .5, .45)
    assert evidence.frames == 1
