"""Diagnostic: test MP+existing SR on the fresh capture hand's current crop.

This supplies a same-frame crop from archived primary coordinates on every
probe. It is neither a real handoff nor a tracking/operation success metric.
"""
import argparse
import json
import math
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from .application_replay import application_time
from .corpus_report import fresh, read, records
from .replay import ROOT, ReplayClock, encode_hands
from .run_replay import sha, write_json


def probe(baseline, clip, destination):
    import cv2
    from services.hand_tracker import HandTracker

    source = baseline / clip / "current"
    manifest = read(source / "dataset_manifest.json")
    video = Path(manifest["video"])
    assert sha(video) == manifest["video_sha256"]
    rows = records(source / "predictions.jsonl")
    config = read(source / "effective_config.json")
    destination.mkdir(parents=True, exist_ok=False)
    write_json(destination / "provenance.json", dict(baseline=str(source.resolve()), manifest=manifest,
        source_baseline=read(baseline / "baseline.json"),
        production_source_sha256={str(p.relative_to(ROOT)): sha(p) for p in (ROOT / "app").rglob("*.py")},
        effective_config=config, crop="current fresh capture primary; padding and minimum size from production, no center EMA"))
    cap = cv2.VideoCapture(str(video))
    tracker = None
    output = []
    clock = ReplayClock()
    try:
        tracker = HandTracker(max_num_hands=2, preferred_model_type=config.get("model_type", "Heavy"),
                              dominant_hand=config.get("dominant_hand", "Right"),
                              min_detection_confidence=config.get("hand_detection_confidence", .5),
                              min_presence_confidence=config.get("hand_presence_confidence", .5),
                              min_tracking_confidence=config.get("hand_tracking_confidence", .5), config=config)
        with ExitStack() as stack:
            stack.enter_context(patch.object(tracker, "_next_mp_timestamp", clock.model_timestamp_ms))
            for row in rows:
                ok, frame = cap.read()
                if not ok or [frame.shape[1], frame.shape[0]] != row["source_size"]:
                    raise ValueError(f"Frame mismatch: {row['frame_index']}")
                if row["fsm_state"] != "capture" or not fresh(row):
                    continue
                lm = [[i, *p] for i, p in enumerate(row["hands"][0]["keypoints_2d_21x2"])]
                center, hand_size = tracker._compute_hint_from_landmarks([lm])
                crop_size = min(max(int(hand_size * tracker._crop_padding_ratio), tracker._crop_min_size), min(frame.shape[:2]))
                clock.set(row["source_timestamp"])
                with application_time(clock):
                    hands, gestures, _ = tracker._detect_crop_zoom(frame, center, crop_size)
                span = math.hypot(*[row["hands"][0]["bbox_xyxy"][i + 2] - row["hands"][0]["bbox_xyxy"][i] for i in (0, 1)])
                distances = [math.hypot(h[0][1] - lm[0][1], h[0][2] - lm[0][2]) for h in hands]
                matching = bool(distances) and min(distances) <= .75 * span
                output.append(dict(frame_index=row["frame_index"], source_timestamp=row["source_timestamp"],
                                   crop_center=center, crop_size=crop_size, capture_span=span,
                                   wrist_distances=distances, geometrically_matches_capture_primary=matching,
                                   probe_hands=encode_hands(hands, gestures, raw=True)))
            if cap.read()[0]:
                raise ValueError("Video longer than archived trace")
        with (destination / "predictions.jsonl").open("w", encoding="utf-8") as out:
            for row in output:
                out.write(json.dumps(row, allow_nan=False) + "\n")
        result = dict(clip=clip, probe_frames=len(output),
                      matched_crop_outputs=sum(r["geometrically_matches_capture_primary"] for r in output),
                      unmatched_frames=[r["frame_index"] for r in output if not r["geometrically_matches_capture_primary"]],
                      limits="Same-frame capture-supplied crops; geometric wrist agreement is not identity truth, detection recall, or independent MP tracking stability.")
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
    args = parser.parse_args()
    for clip in args.clips:
        if Path(clip).name != clip or clip in (".", ".."):
            parser.error("clips must be directory names")
    for clip in args.clips:
        print(json.dumps(probe(args.baseline, clip, args.destination / clip), ensure_ascii=True), flush=True)
