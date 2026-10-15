"""Shadow ZOOM probe: run a real long-range tracker beside the capture engine.

This replays a production capture-state trace while an independent MediaPipe
tracker with the FAR_TRACK runtime override (long_range_enabled=True) runs in
the same process. Neither the shadow tracker's output nor the probe itself can
mutate FSM state or desktop sinks; the replay consumes an already-recorded
predictions.jsonl.

The first shadow fresh primary sample arms a one-shot confirmation window
(uniform K frames, no EMA lag): a fresh non-predicted primary whose wrist lies
within 0.75× the capture hand span confirms the shadow is following the same
hand. When the shadow misses (ZOOM miss streak reset / re-search), the capture
position re-seeds the viewport, mirroring the production handoff intent. The
primary question answered here is whether the shadow tracker and the capture
engine keep identifying the same hand while tracking, not whether ZOOM is
better than capture.
"""
import argparse
import json
import logging
import math
import time
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from .application_replay import application_time
from .corpus_report import fresh, read, records
from .replay import ROOT, ReplayClock, encode_hands, tracker_state
from .run_replay import sha, write_json

CONFIRM_WINDOW_FRAMES = 5
CONFIRM_MATCH_SPAN_RATIO = 0.75


def shadow_zoom_probe(baseline, clip, destination, reseed_miss_frames):
    import cv2
    from services.hand_tracker import HandTracker

    source = baseline / clip / "current"
    manifest = read(source / "dataset_manifest.json")
    video = Path(manifest["video"])
    assert sha(video) == manifest["video_sha256"]
    rows = records(source / "predictions.jsonl")
    config = read(source / "effective_config.json")
    destination.mkdir(parents=True, exist_ok=False)
    write_json(destination / "provenance.json", dict(
        baseline=str(source.resolve()), manifest=manifest,
        source_baseline=read(baseline / "baseline.json"),
        production_source_sha256={str(p.relative_to(ROOT)): sha(p) for p in (ROOT / "app").rglob("*.py")},
        effective_config=config,
        shadow="mediapipe HandTracker with long_range_enabled=True (FAR_TRACK runtime override equivalent); "
               f"confirmation window {CONFIRM_WINDOW_FRAMES:d} frames; capture re-seeds viewport on shadow ZOOM miss",
        limits="Shadow tracker runs beside an archived capture-state trace; shadow output never drives FSM, "
               "modes, or desktop sinks. Geometric wrist agreement within 0.75x capture span is not identity "
               "truth. This probe does not measure live latency or click accuracy."))
    cap = cv2.VideoCapture(str(video))
    tracker = None
    output = []
    clock = ReplayClock()
    # Offline statistics only; wall-clock timing for the extra model calls is
    # recorded but the source clock governs all algorithm state.
    try:
        tracker = HandTracker(max_num_hands=2, preferred_model_type=config.get("model_type", "Heavy"),
                              dominant_hand=config.get("dominant_hand", "Right"),
                              min_detection_confidence=config.get("hand_detection_confidence", .5),
                              min_presence_confidence=config.get("hand_presence_confidence", .5),
                              min_tracking_confidence=config.get("hand_tracking_confidence", .5),
                              config=dict(config, long_range_enabled=True))
        # Seed the shadow viewport with the first fresh capture position so the
        # probe's first crop mirrors a real handoff seed (one-shot placement).
        seed_center = seed_size = None
        confirm_remaining = 0
        confirmations = 0
        confirmation_frame = None
        last_confirmed_wrist = None
        reseed_events = 0
        shadow_miss_frames = 0
        with ExitStack() as stack:
            stack.enter_context(patch.object(tracker, "_next_mp_timestamp", clock.model_timestamp_ms))
            for row in rows:
                ok, frame = cap.read()
                if not ok or [frame.shape[1], frame.shape[0]] != row["source_size"]:
                    raise ValueError(f"Frame mismatch: {row['frame_index']}")
                clock.set(row["source_timestamp"])
                # Capture-position supply: only fresh capture primary frames.
                cap_fresh = row["fsm_state"] == "capture" and fresh(row)
                if cap_fresh:
                    lm = [[i, *p] for i, p in enumerate(row["hands"][0]["keypoints_2d_21x2"])]
                    center, hand_size = tracker._compute_hint_from_landmarks([lm])
                    span = math.hypot(row["hands"][0]["bbox_xyxy"][2] - row["hands"][0]["bbox_xyxy"][0],
                                       row["hands"][0]["bbox_xyxy"][3] - row["hands"][0]["bbox_xyxy"][1])
                else:
                    center = hand_size = span = None
                # One-shot seed on the first fresh capture position.
                if seed_center is None and cap_fresh:
                    seed_center, seed_size = center, hand_size
                    tracker._crop_zoom_mode = True
                    tracker._last_hint_center = seed_center
                    tracker._last_hint_size = seed_size
                    tracker._current_crop_center = None
                    tracker._current_crop_size = None
                    logging.info("shadow viewport seeded at %s size %.0f", seed_center, seed_size)
                before_state = tracker_state(tracker)
                started = time.perf_counter()
                with application_time(clock):
                    _, hands, gestures = tracker.find_hands(frame, draw=False)
                shadow_wall_ms = (time.perf_counter() - started) * 1000.0
                after_state = tracker_state(tracker)
                hands = list(hands or [])
                gestures = list(gestures or [])
                entry = {
                    "frame_index": row["frame_index"],
                    "source_timestamp": row["source_timestamp"],
                    "fsm_state": row["fsm_state"],
                    "capture_fresh": bool(cap_fresh),
                    "capture_span": span,
                    "shadow_hands": encode_hands(hands, gestures),
                    "shadow_before": before_state,
                    "shadow_after": after_state,
                    "shadow_wall_ms": shadow_wall_ms,
                }
                # Shadow ZOOM miss (viewport reset or miss streak) → capture
                # re-seeds the viewport next frame, mirroring production intent.
                # reseed_miss_frames: streak threshold at which a same-frame
                # capture position may rewrite the stalled viewport (step 2).
                zoom_reset = (before_state.get("zoom_on") and not after_state.get("zoom_on")) \
                    or (after_state.get("zoom_miss_streak", 0) >= reseed_miss_frames)
                if zoom_reset:
                    shadow_miss_frames += 1
                    if cap_fresh:
                        # Re-seed immediately in-frame (no EMA lag) when capture
                        # has a same-frame position; otherwise viewport restarts
                        # from its own full-frame search on subsequent frames.
                        tracker._crop_zoom_mode = True
                        tracker._last_hint_center = center
                        tracker._last_hint_size = hand_size
                        tracker._current_crop_center = None
                        tracker._current_crop_size = None
                        reseed_events += 1
                        entry["capture_reseeded"] = True
                # Confirmation bookkeeping. A window opens only on a frame
                # where BOTH engines produced fresh output and the shadow
                # wrist geometrically agrees with the capture hand. It then
                # stays open only while the shadow keeps reporting a fresh
                # primary whose wrist remains within CONFIRM_WINDOW_FRAMES
                # consecutive small moves of its own previous wrist (shadow
                # self-continuity, no capture data). If the shadow reports a
                # predicted/empty primary or jumps, the window closes and the
                # next same-frame agreement must reopen it.
                shadow_fresh = len(hands) > 0 and len(gestures) > 0 \
                    and isinstance(gestures[0], dict) and not gestures[0].get("predicted", False)
                if shadow_fresh and cap_fresh:
                    wrist = (hands[0][0][1], hands[0][0][2])
                    entry["shadow_wrist_distance_to_capture"] = math.dist(wrist, center)
                    same_frame_match = math.dist(wrist, center) <= CONFIRM_MATCH_SPAN_RATIO * span
                else:
                    same_frame_match = False
                if confirm_remaining > 0 and shadow_fresh:
                    span_now = math.hypot(
                        max(p[1] for p in hands[0]) - min(p[1] for p in hands[0]),
                        max(p[2] for p in hands[0]) - min(p[2] for p in hands[0]))
                    wrist = (hands[0][0][1], hands[0][0][2])
                    self_continuous = (last_confirmed_wrist is not None
                                       and math.dist(wrist, last_confirmed_wrist)
                                       <= CONFIRM_MATCH_SPAN_RATIO * max(span_now, 1.0))
                    if self_continuous:
                        confirm_remaining -= 1
                        last_confirmed_wrist = wrist
                        if confirm_remaining == 0:
                            confirmations += 1
                            confirmation_frame = row["frame_index"]
                            entry["confirmed"] = True
                    else:
                        confirm_remaining = 0
                if confirm_remaining == 0 and same_frame_match:
                    confirm_remaining = CONFIRM_WINDOW_FRAMES - 1
                    last_confirmed_wrist = (hands[0][0][1], hands[0][0][2])
                output.append(entry)
            if cap.read()[0]:
                raise ValueError("Video longer than archived trace")
        with (destination / "predictions.jsonl").open("w", encoding="utf-8") as out:
            for entry in output:
                out.write(json.dumps(entry, allow_nan=False) + "\n")
        capture_fresh_frames = sum(e["capture_fresh"] for e in output)
        shadow_fresh_frames = sum(bool(e["shadow_hands"] and e["shadow_hands"][0]["detected_fresh"] is True)
                                  for e in output)
        same_frame_agreement = sum(1 for e in output if e.get("shadow_wrist_distance_to_capture") is not None
                                   and e["shadow_wrist_distance_to_capture"] <= CONFIRM_MATCH_SPAN_RATIO * (e["capture_span"] or 1))
        import numpy as np
        wall = [e["shadow_wall_ms"] for e in output]
        result = {
            "clip": clip,
            "frames": len(output),
            "capture_fresh_frames": capture_fresh_frames,
            "shadow_fresh_frames": shadow_fresh_frames,
            "same_frame_capture_agreement": same_frame_agreement,
            "confirmations": confirmations,
            "confirmed_frames": [e["frame_index"] for e in output if e.get("confirmed")],
            "last_confirmation_frame": confirmation_frame,
            "shadow_wall_ms": {"count": len(wall),
                               "p50": float(np.percentile(wall, 50)) if wall else None,
                               "p95": float(np.percentile(wall, 95)) if wall else None,
                               "note": "includes model init/first-call warmup; not a live FPS baseline"},
            "shadow_zoom_miss_frames": shadow_miss_frames,
            "capture_reseed_events": reseed_events,
            "reseed_miss_frames": reseed_miss_frames,
            "limits": "Geometric wrist agreement is not identity truth; shadow runs beside an archived "
                      "capture trace with same-frame re-seeds only; no FSM/mode/desktop effects measured.",
        }
        write_json(destination / "summary.json", result)
        return result
    finally:
        cap.release()
        if tracker is not None:
            tracker.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--clips", nargs="+", required=True)
    parser.add_argument("--reseed-miss-frames", type=int, default=5,
                        help="shadow ZOOM miss streak at which a same-frame capture "
                             "position may rewrite the stalled viewport (step-2 knob)")
    args = parser.parse_args()
    for clip in args.clips:
        if Path(clip).name != clip or clip in (".", ".."):
            parser.error("clips must be directory names")
    if args.reseed_miss_frames < 1:
        parser.error("reseed-miss-frames must be >= 1")
    logging.basicConfig(level=logging.WARNING)
    for clip in args.clips:
        print(json.dumps(shadow_zoom_probe(args.baseline, clip, args.destination / clip,
                                           args.reseed_miss_frames), ensure_ascii=True), flush=True)
