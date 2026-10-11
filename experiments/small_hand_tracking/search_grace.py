"""Experimental bounded search retention; no production defaults are changed."""
from copy import deepcopy


def install_search_grace(replay, grace_seconds=.4):
    tracker = replay.tracker
    original_lost = tracker._handle_all_lost
    original_present = tracker._handle_hands_present
    loss_started = None
    fields = ("_crop_zoom_mode", "_last_hint_center", "_last_hint_size",
              "_current_crop_center", "_current_crop_size", "_zoom_miss_streak")

    def present(*args, **kwargs):
        nonlocal loss_started
        loss_started = None
        return original_present(*args, **kwargs)

    def lost(frame, use_crop_zoom, draw, width, height):
        nonlocal loss_started
        now = replay.clock.monotonic()
        eligible = (tracker._long_range_enabled and use_crop_zoom and tracker._crop_zoom_mode
                    and not tracker._hand_prediction_enabled and tracker._last_hint_center is not None
                    and 0 < tracker._zoom_miss_streak < tracker._zoom_miss_threshold)
        if eligible and loss_started is None:
            loss_started = now
        preserve = eligible and now - loss_started < grace_seconds
        saved = {field: deepcopy(getattr(tracker, field)) for field in fields} if preserve else None
        sr_state = (tracker._sr._auto_sr_enabled, tracker._sr._last_sr_tier)
        result = original_lost(frame, use_crop_zoom, draw, width, height)
        if preserve:
            for field, value in saved.items():
                setattr(tracker, field, value)
            tracker._sr._auto_sr_enabled, tracker._sr._last_sr_tier = sr_state
        elif eligible:
            # Smoothers were cleared on the first loss, so original_lost may no
            # longer reset the viewport on subsequent misses. Enforce deadline.
            tracker._crop_zoom_mode = False
            tracker._last_hint_center = None
            tracker._last_hint_size = 0
            tracker._current_crop_center = None
            tracker._current_crop_size = None
            tracker._zoom_miss_streak = 0
            tracker._sr.reset_tier()
        if not eligible or not preserve:
            loss_started = None
        replay.frame_log["search_grace"] = {"preserved_search": bool(preserve), "limit_seconds": grace_seconds}
        return result  # No old landmarks, identity, or gesture labels restored.

    replay._replace(tracker, "_handle_all_lost", lost)
    replay._replace(tracker, "_handle_hands_present", present)
