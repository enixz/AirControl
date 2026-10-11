"""Candidate: bypass capture cooldown only after consistent fresh-primary evidence.

This changes only CAPTURE -> FAR_TRACK in the application replay. All other
production transitions, models, crop logic and SR remain untouched.
"""
import math
from unittest.mock import patch

from services.engine_auto_switcher import STATE_CAPTURE, STATE_FAR_TRACK, primary_hand_evidence


class CaptureHandoffEvidence:
    MIN_FRESH_FRAMES = 5
    MIN_STABLE_SEC = .3
    MIN_CAPTURE_SEC = .5
    MAX_SAMPLE_GAP_SEC = .25

    def __init__(self):
        self.capture_since = None
        self.clear()

    def clear(self):
        self.since = None
        self.last_time = None
        self.wrist = None
        self.span = None
        self.key = None
        self.frames = 0

    def observe(self, hands, gestures, frame_area, now, capture_since):
        if capture_since != self.capture_since:
            self.clear()
            self.capture_since = capture_since
        _, ratio, valid = primary_hand_evidence(hands, gestures, frame_area)
        if not valid or ratio <= 0:
            self.clear()
            return False
        points = hands[0]
        wrist = (float(points[0][1]), float(points[0][2]))
        xs, ys = zip(*((p[1], p[2]) for p in points))
        span = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
        key = gestures[0].get("track_key")
        if key is None:
            self.clear()
            return False
        # Two nearby fresh wrists are ambiguous. Predictions and distant
        # resting hands cannot supply or block primary evidence.
        for other, gesture in zip(hands[1:], gestures[1:]):
            if gesture.get("predicted", False):
                continue
            if math.hypot(other[0][1] - wrist[0], other[0][2] - wrist[1]) <= 1.5 * span:
                self.clear()
                return False
        compatible_key = (key == self.key or gestures[0].get("previous_track_key") == self.key)
        continuous = (self.last_time is not None and 0 < now - self.last_time <= self.MAX_SAMPLE_GAP_SEC
                      and compatible_key
                      and math.dist(wrist, self.wrist) <= .75 * max(span, self.span)
                      and max(span, self.span) <= 2 * min(span, self.span))
        if not continuous:
            self.clear()
            self.since = now
        self.frames += 1
        self.last_time, self.wrist, self.span, self.key = now, wrist, span, key
        return (self.frames >= self.MIN_FRESH_FRAMES
                and now - self.since >= self.MIN_STABLE_SEC
                and now - capture_since >= self.MIN_CAPTURE_SEC)


def install_capture_handoff(patches):
    import os
    from orchestrator import AirControlOrchestrator

    original = AirControlOrchestrator._maybe_auto_switch_engine

    def update(owner, hands, hand_bbox_ratio=0., hands_gestures=None, frame_area=1):
        switcher = owner._engine_switcher
        if (switcher.enabled and switcher.state == STATE_CAPTURE
                and not owner._engine_switch_pending
                and getattr(owner, "_tracker_config_signature", None)
                and not os.environ.get("AIRCONTROL_ENGINE") and hands_gestures is not None):
            if not hasattr(owner, "_experiment_capture_handoff"):
                owner._experiment_capture_handoff = CaptureHandoffEvidence()
            now = switcher._clock()
            capture_since = switcher._last_switch_time
            if capture_since is None:
                capture_since = now
            evidence = owner._experiment_capture_handoff
            if evidence.observe(hands or [], hands_gestures, frame_area, now, capture_since):
                target = switcher._mark_switch(STATE_FAR_TRACK, now,
                    "实验：新鲜主手位置连续，提前交给 ZOOM")
                evidence.clear()
                owner._apply_fsm_state(target)
                return
        else:
            owner._experiment_capture_handoff = CaptureHandoffEvidence()
        return original(owner, hands, hand_bbox_ratio, hands_gestures, frame_area)

    patches.enter_context(patch.object(AirControlOrchestrator, "_maybe_auto_switch_engine", update))
