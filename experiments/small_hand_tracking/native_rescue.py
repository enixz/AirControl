"""Experiment: rate-limited native-resolution retry after a full-frame miss."""


def install_native_rescue(replay, interval_seconds=.2):
    tracker = replay.tracker
    original = tracker._detect
    last_attempt = None

    def detect(frame):
        nonlocal last_attempt
        result = original(frame)
        width_limit = tracker._inference_max_width
        now = replay.clock.monotonic()
        if (result[0] or width_limit <= 0 or frame.shape[1] <= width_limit
                or (last_attempt is not None and now - last_attempt < interval_seconds)):
            return result
        last_attempt = now
        try:
            tracker._inference_max_width = 0
            rescued = original(frame)
        finally:
            tracker._inference_max_width = width_limit
        replay.frame_log.setdefault("native_rescue", []).append({
            "accepted_hand_count": len(rescued[0]), "interval_seconds": interval_seconds,
            "input_size": [frame.shape[1], frame.shape[0]],
        })
        return rescued

    replay._replace(tracker, "_detect", detect)
