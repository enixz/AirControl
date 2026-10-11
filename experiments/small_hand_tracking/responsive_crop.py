"""Experimental faster search-center response; never changes hand outputs."""
import math


def install_responsive_crop(tracker, patches):
    from unittest.mock import patch
    original = tracker._compute_crop_viewport

    def compute(width, height):
        before = tracker._current_crop_center
        target = tracker._last_hint_center
        side = tracker._current_crop_size
        eligible = tracker._crop_zoom_mode and target is not None and before is not None and side is not None and side > 0
        original(width, height)
        if eligible:
            error = math.hypot(target[0] - before[0], target[1] - before[1]) / side
            # Retain ordinary smoothing for tiny changes. Ramp up only when
            # the target has moved a material fraction of the search square.
            alpha = max(tracker._zoom_alpha, min(.60, tracker._zoom_alpha + max(0., error - .10)))
            tracker._current_crop_center = tuple((1 - alpha) * before[i] + alpha * target[i] for i in range(2))
        # Crop size keeps the existing EMA, so this experiment changes one
        # aspect of viewport behavior rather than its magnification as well.

    patches.enter_context(patch.object(tracker, "_compute_crop_viewport", compute))
