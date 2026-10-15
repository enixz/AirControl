"""Source-time replay and observation of the real tracker, without desktop I/O.

Run in a dedicated single-threaded evaluation process. Module-local clock
bindings are replaced only during inference and always restored afterwards;
the standard library time module and performance measurements stay real.
"""
from __future__ import annotations

import math
import sys
import time
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))


class ReplayClock:
    def __init__(self):
        self.value = None
        self.origin = None
        self.last_model_ms = -1

    def set(self, timestamp):
        timestamp = float(timestamp)
        if not math.isfinite(timestamp):
            raise ValueError("Source timestamp must be finite")
        if self.value is not None and timestamp <= self.value:
            raise ValueError("Source timestamps must be strictly increasing")
        if self.origin is None:
            self.origin = timestamp
        self.value = timestamp

    def time(self):
        if self.value is None:
            raise RuntimeError("Set a source timestamp before inference")
        return self.value

    def perf_counter(self):
        return 1.0 + self.time() - self.origin

    monotonic = perf_counter

    def model_timestamp_ms(self):
        # Multiple crop/full-frame attempts can occur for the same source frame.
        self.last_model_ms = max(round(self.time() * 1000), self.last_model_ms + 1)
        return self.last_model_ms


@contextmanager
def algorithm_time(clock):
    # Do not patch time.perf_counter globally: that corrupts timing and logging.
    names = ("services.base_hand_tracker", "services.hand_tracker", "services.smoothers")
    bindings = []
    try:
        for name in names:
            module = sys.modules.get(name)
            if module is not None and hasattr(module, "time"):
                bindings.append((module, module.time))
                module.time = clock
        yield
    finally:
        for module, original in reversed(bindings):
            module.time = original


def encode_hands(hands, gestures, *, raw=False):
    """Preserve identity/label provenance; never infer freshness from coordinates."""
    aligned = gestures is not None and len(hands) == len(gestures)
    result = []
    for index, lm in enumerate(hands):
        g = gestures[index] if aligned and isinstance(gestures[index], dict) else {}
        predicted = g.get("predicted")
        if raw:
            predicted = False  # Observed at _run_detection, before prediction.
        elif type(predicted) is not bool:
            predicted = None
        points = [[float(p[1]), float(p[2])] for p in lm]
        xs, ys = zip(*points, strict=True)
        result.append({
            "track_id": g.get("track_key"),
            "is_primary_output": index == 0 and not raw,
            "handedness": g.get("handedness"),
            "handedness_score": g.get("handedness_score"),
            "bbox_xyxy": [min(xs), min(ys), max(xs), max(ys)],
            "keypoints_2d_21x2": points,
            "keypoint_scores_21_or_null": None,
            "keypoints_3d_or_null": None,
            "detected_fresh": None if predicted is None else not predicted,
            "predicted": predicted,
            "freshness": "unknown" if predicted is None else ("predicted" if predicted else "fresh"),
            "gesture_label": g.get("ml_label"),
            "internal_label": g.get("label"),
            "gesture_score": g.get("score"),
            "label_source": "mediapipe_gesture_recognizer" if g.get("ml_label") not in (None, "None") else None,
            "skipped": g.get("skipped", False),
        })
    return result


def tracker_state(tracker):
    center = tracker._current_crop_center
    return {
        "engine": tracker.engine_name,
        "long_range_enabled": bool(tracker._long_range_enabled),
        "zoom_on": bool(tracker._crop_zoom_mode),
        "crop_center": list(center) if center is not None else None,
        "crop_size": tracker._current_crop_size,
        "primary_track_id": tracker._primary_track_key,
        "zoom_miss_streak": tracker._zoom_miss_streak,
    }


class TrackerReplay:
    """Fixed-engine tracker observation. Does not emulate orchestrator or UI."""
    def __init__(self, config):
        from services.hand_tracker_factory import create_hand_tracker

        self.config = deepcopy(config)
        self.clock = ReplayClock()
        self.index = 0
        self.patches = []
        self.frame_log = None
        started = time.perf_counter()
        self.tracker = create_hand_tracker(
            engine="mediapipe",
            max_num_hands=config.get("max_num_hands", 2),
            min_detection_confidence=config.get("hand_detection_confidence", 0.5),
            min_presence_confidence=config.get("hand_presence_confidence", 0.5),
            min_tracking_confidence=config.get("hand_tracking_confidence", 0.5),
            preferred_model_type=config.get("model_type", "Heavy"),
            dominant_hand=config.get("dominant_hand", "Right"),
            config=config,
        )
        self.initialization_ms = (time.perf_counter() - started) * 1000
        if config.get("experiment_search_grace", False):
            from .search_grace import install_search_grace
            install_search_grace(self)
        if config.get("experiment_native_rescue", False):
            from .native_rescue import install_native_rescue
            install_native_rescue(self)
        self._instrument()

    def _replace(self, owner, name, replacement):
        p = patch.object(owner, name, replacement)
        p.start()
        self.patches.append(p)

    def _instrument(self):
        tracker = self.tracker
        self._replace(tracker, "_next_mp_timestamp", self.clock.model_timestamp_ms)
        original = tracker._run_recognizer

        def run_model(frame):
            started = time.perf_counter()
            result = original(frame)
            self.frame_log["model_calls"].append({
                "input_size": [frame.shape[1], frame.shape[0]],
                "model_timestamp_ms": self.clock.last_model_ms,
                "duration_ms": (time.perf_counter() - started) * 1000,
            })
            return result

        self._replace(tracker, "_run_recognizer", run_model)
        original_detection = tracker._run_detection

        def run_detection(*args):
            result = original_detection(*args)
            self.frame_log["raw_hands"] = encode_hands(result[0], result[1], raw=True)
            self.frame_log["used_crop_zoom"] = bool(result[2])
            return result

        self._replace(tracker, "_run_detection", run_detection)
        for name in ("_detect", "_detect_crop_zoom"):
            self._wrap_attempt(name)
        for name in ("_compute_crop_viewport", "_update_zoom_mode", "_handle_all_lost", "_run_detection"):
            self._wrap_transition(name)
        for name in ("espcn", "realesrgan"):
            original_sr = getattr(tracker._sr, name)

            def run_sr(*args, _original=original_sr, _name=name, **kwargs):
                started = time.perf_counter()
                out = _original(*args, **kwargs)
                self.frame_log["sr_calls"].append({
                    "engine": _name, "succeeded": out is not None,
                    "duration_ms": (time.perf_counter() - started) * 1000,
                })
                return out

            self._replace(tracker._sr, name, run_sr)
        original_log_tier = tracker._sr.log_tier

        def log_tier(tier, crop_size, target):
            self.frame_log["upscalers"].append({"effective_tier": tier, "crop_size": crop_size, "target": target})
            return original_log_tier(tier, crop_size, target)

        self._replace(tracker._sr, "log_tier", log_tier)

    def _wrap_attempt(self, name):
        original = getattr(self.tracker, name)

        def attempt(frame, *args, **kwargs):
            entry = {"method": name, "frame_size": [frame.shape[1], frame.shape[0]]}
            if name == "_detect_crop_zoom":
                entry["requested_crop_center"] = list(args[0])
                entry["requested_crop_size"] = float(args[1])
                size = min(max(round(args[1]), self.tracker._crop_min_size), min(frame.shape[:2]))
                x = max(0, min(round(args[0][0] - size / 2), frame.shape[1] - size))
                y = max(0, min(round(args[0][1] - size / 2), frame.shape[0] - size))
                entry["crop_xyxy"] = [x, y, x + size, y + size]
            started = time.perf_counter()
            result = original(frame, *args, **kwargs)
            entry["duration_ms"] = (time.perf_counter() - started) * 1000
            entry["accepted_hand_count"] = len(result[0])
            self.frame_log["attempts"].append(entry)
            return result

        self._replace(self.tracker, name, attempt)

    def _wrap_transition(self, name):
        original = getattr(self.tracker, name)

        def observed(*args, **kwargs):
            before = tracker_state(self.tracker)
            event_count = len(self.frame_log["events"])
            result = original(*args, **kwargs)
            after = tracker_state(self.tracker)
            if before["zoom_on"] != after["zoom_on"] and len(self.frame_log["events"]) == event_count:
                self.frame_log["events"].append({
                    "kind": "zoom_transition", "observed_in": name,
                    "from": before["zoom_on"], "to": after["zoom_on"],
                    "before": before, "after": after,
                })
            return result

        self._replace(self.tracker, name, observed)

    def infer(self, frame, timestamp):
        self.clock.set(timestamp)
        before = tracker_state(self.tracker)
        self.frame_log = {"model_calls": [], "sr_calls": [], "upscalers": [],
                          "raw_hands": [], "attempts": [], "events": [], "used_crop_zoom": False}
        started = time.perf_counter()
        with algorithm_time(self.clock):
            _, hands, gestures = self.tracker.find_hands(frame, draw=False)
        ended = time.perf_counter()
        encoded = encode_hands(hands, gestures)
        row = {
            "schema_version": 2, "pipeline": "tracker_only", "frame_index": self.index,
            "source_timestamp": float(timestamp), "source_size": [frame.shape[1], frame.shape[0]],
            "coordinate_space": "source_frame_pixels", "actual_engine": self.tracker.engine_name,
            "auto_switch_executed": False, "fsm_state": None, "operation_events": None,
            "sr_requested": self.config.get("zoom_sr_engine", "auto"),
            "before": before, "after": tracker_state(self.tracker), "hands": encoded,
            "inference_start_ms": started * 1000, "inference_end_ms": ended * 1000,
            "tracker_ms": (ended - started) * 1000, "application_frame_age_ms": None,
            **self.frame_log,
        }
        for event in row["events"]:
            event.update(frame_index=self.index, source_timestamp=float(timestamp))
        self.index += 1
        return row

    def backend_info(self):
        sr = self.tracker._sr
        sessions = {}
        for name in ("_realesrgan_cpu_session", "_realesrgan_gpu_session"):
            session = getattr(sr, name, None)
            if session is not None:
                sessions[name] = session.get_providers()
        return {"mediapipe": "CPU task runner (default BaseOptions)",
                "sr_session_providers": sessions,
                "note": "Session provider list does not prove every node ran on GPU"}

    def close(self):
        for p in reversed(self.patches):
            p.stop()
        self.patches.clear()
        self.tracker.close()


def summarize(rows):
    """Unlabelled observations only. No accuracy or visible-hand recall claims."""
    import numpy as np

    count = len(rows)
    primary = [bool(r["hands"] and r["hands"][0]["detected_fresh"] is True) for r in rows]
    intervals = []
    start = None
    for i, fresh in enumerate(primary):
        if not fresh and start is None:
            start = i
        if fresh and start is not None:
            intervals.append({"start_frame": start, "end_frame_exclusive": i,
                              "duration_ms": 1000 * (rows[i]["source_timestamp"] - rows[start]["source_timestamp"]),
                              "left_censored": start == 0, "right_censored": False})
            start = None
    if start is not None:
        intervals.append({"start_frame": start, "end_frame_exclusive": count,
                          "duration_ms": 1000 * (rows[-1]["source_timestamp"] - rows[start]["source_timestamp"]),
                          "left_censored": start == 0, "right_censored": True})

    def rate(n):
        return n / count if count else None

    def distribution(values):
        return {"count": len(values), "p50": float(np.percentile(values, 50)) if values else None,
                "p95": float(np.percentile(values, 95)) if values else None}

    tiers = {}
    for row in rows:
        for up in row["upscalers"]:
            tier = up["effective_tier"]
            tiers[tier] = tiers.get(tier, 0) + 1
    return {
        "schema_version": 2, "pipeline": "tracker_only", "frames": count,
        "source_span_seconds": rows[-1]["source_timestamp"] - rows[0]["source_timestamp"] if count > 1 else None,
        "fresh_any_output_rate": rate(sum(any(h["detected_fresh"] is True for h in r["hands"]) for r in rows)),
        "fresh_primary_output_rate": rate(sum(primary)),
        "output_any_rate": rate(sum(bool(r["hands"]) for r in rows)),
        "prediction_only_frames": sum(bool(r["hands"]) and all(h["predicted"] is True for h in r["hands"]) for r in rows),
        "unknown_freshness_frames": sum(any(h["detected_fresh"] is None for h in r["hands"]) for r in rows),
        "raw_detection_frames": sum(bool(r["raw_hands"]) for r in rows),
        "zoom_used_frames": sum(r["used_crop_zoom"] for r in rows),
        "zoom_events": sum(len(r["events"]) for r in rows),
        "native_rescue_calls": sum(len(r.get("native_rescue", [])) for r in rows),
        "native_rescue_acceptances": sum(x["accepted_hand_count"] > 0 for r in rows for x in r.get("native_rescue", [])),
        "upscaler_call_counts": tiers,
        "tracker_ms": distribution([r["tracker_ms"] for r in rows]),
        "model_ms": distribution([sum(x["duration_ms"] for x in r["model_calls"]) for r in rows]),
        "sr_ms": distribution([x["duration_ms"] for r in rows for x in r["sr_calls"]]),
        "no_fresh_primary_intervals": intervals,
        "no_fresh_primary_observed_ms": sum(x["duration_ms"] for x in intervals),
        "visible_hand_recall": None, "keypoint_accuracy": None, "reacquisition_ms": None,
        "click_success_rate": None, "application_frame_age_ms": None,
        "limits": ["No human visibility/identity labels: output coverage is not recall or accuracy",
                   "No-fresh intervals include out-of-frame/occluded time; final interval is censored",
                   "Tracker-only; no orchestrator FSM, desktop actions, UI queues or real-time FPS",
                   "Tracker timing includes observation overhead; cold/lazy initialization may appear"],
    }
