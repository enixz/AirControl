"""Retire a duplicate ghost after an unambiguous one-to-two key reassignment.

Keep the existing fresh-hand keys and smoother histories. Rewriting their keys
changed later matching and introduced new losses in the first experiment.
"""
import math
from unittest.mock import patch


def install_identity_continuity(tracker, patches):
    original = tracker._assign_smoother_keys

    def assign(hands, gestures, frame_w=1920, frame_h=1080):
        previous = list(tracker._prev_wrist_keys)
        keys = original(hands, gestures, frame_w, frame_h)
        if len(previous) != 1 or len(hands) != 2:
            return keys
        previous_wrist, previous_key = previous[0]
        wrists = [(float(hand[0][1]), float(hand[0][2])) for hand in hands]
        distances = [math.hypot(w[0] - previous_wrist[0], w[1] - previous_wrist[1]) for w in wrists]
        nearest = 0 if distances[0] <= distances[1] else 1
        if nearest != 0:
            return keys
        other = 1 - nearest
        span = max(1., tracker._hand_span(hands[nearest]))
        # Require both a close match and strong separation from the other hand.
        # The larger .75*span motion allowance remains bounded by that margin.
        if distances[nearest] > max(.75 * span, 30.) or distances[other] <= max(2 * span, distances[nearest] + 1.5 * span):
            return keys
        if previous_key not in tracker._active_handedness or previous_key in keys:
            return keys
        # That physical hand has a fresh replacement under a different key.
        # Do not append its old alias as a predicted primary for five frames.
        # Keep both real detections and their existing keys/smoothers intact.
        tracker._active_handedness.discard(previous_key)
        return keys

    patches.enter_context(patch.object(tracker, "_assign_smoother_keys", assign))
