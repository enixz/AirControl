"""Run deterministic application semantics; desktop/camera/queue timing excluded."""
import argparse
from collections import Counter
from copy import deepcopy
from contextlib import nullcontext
import json
import logging
from pathlib import Path
import os

from .replay import ROOT, summarize
from .run_replay import load_config, read_metadata, sha, snapshot, write_json
from .application_replay import ApplicationReplay


def application_summary(rows):
    result = summarize(rows)
    result.update(pipeline="application_semantic_replay", model_ms=None, sr_ms=None,
                  zoom_events=None, fsm_transitions=[dict(frame_index=r["frame_index"], source_timestamp=r["source_timestamp"], **e)
                                                    for r in rows for e in r["events"]],
                  state_frames=dict(Counter(r["fsm_state"] for r in rows)),
                  mode_gestures=dict(Counter(r["mode_gesture"] for r in rows)),
                  fresh_primary_mode_gestures=dict(Counter(r["mode_gesture"] for r in rows if r["hands"] and r["hands"][0]["detected_fresh"] is True)),
                  operation_counts=dict(Counter(f'{e["sink"]}.{e["method"]}' for r in rows for e in r["operation_events"])),
                  missing_primary_stages=dict(Counter("no_raw_detection" if not r["raw_hands"] else "raw_present_primary_predicted"
                                                       for r in rows if not r["hands"] or r["hands"][0]["detected_fresh"] is not True)),
                  limits=["Actual models, worker swap/migration, orchestrator FSM and mode handling; inert desktop sinks",
                          "Source timestamps govern algorithm time; construction synchronous between frames, prewarming excluded",
                          "Camera scheduling, Qt queued delivery, native UI and mouse watchdog excluded: no live frame age or FPS",
                          "No visibility/identity or action intent labels: output coverage is not detection recall or operation success"])
    return result


def run_clip(rec_dir, destination, config, max_frames):
    import cv2
    meta = read_metadata(rec_dir / "meta.jsonl", allow_capture_gaps=True)
    videos = [rec_dir / name for name in ("frames.mkv", "frames.mp4", "frames.avi") if (rec_dir / name).exists()]
    if len(videos) != 1:
        raise ValueError(f"Expected one original frames video in {rec_dir}, found {videos}")
    video = videos[0]
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video}")
    destination.mkdir(parents=True, exist_ok=False)
    write_json(destination / "effective_config.json", config)
    write_json(destination / "dataset_manifest.json", {"video": str(video), "video_sha256": sha(video),
                                                      "meta_sha256": sha(rec_dir / "meta.jsonl"), "ground_truth_used": False,
                                                      "metadata_frames": len(meta),
                                                      "capture_index_gaps": sum(b.get("i", i + 1) - a.get("i", i) - 1 for i, (a, b) in enumerate(zip(meta, meta[1:]))),
                                                      "alignment": "video decode order to metadata row order; original timestamps retained; no missing frames synthesized"})
    replay = None
    rows = []
    try:
        replay = ApplicationReplay(config, meta[0]["t"])
        count = min(max_frames or len(meta), len(meta))
        with (destination / "predictions.jsonl").open("w", encoding="utf-8") as out:
            for index in range(count):
                ok, frame = cap.read()
                if not ok or (frame.shape[1], frame.shape[0]) != (meta[index]["w"], meta[index]["h"]):
                    raise ValueError(f"Video alignment failure at frame {index}")
                row = replay.infer(frame, meta[index]["t"])
                row["capture_frame_index"] = meta[index].get("i", index)
                rows.append(row)
                out.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                if (index + 1) % 250 == 0:
                    print(f"  {destination.name}: {index + 1}/{count}", flush=True)
            if count == len(meta) and cap.read()[0]:
                raise ValueError("Video longer than metadata")
        result = application_summary(rows)
        result["construction_calls"] = replay.initialization
        write_json(destination / "summary.json", result)
        return result
    finally:
        cap.release()
        if replay:
            replay.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--clips", nargs="+", required=True)
    parser.add_argument("--variants", nargs="+", choices=("current", "before_fixes", "no_sr", "always_zoom", "responsive_crop", "responsive_auto", "identity_continuity", "capture_handoff", "shadow_zoom"), default=["current", "no_sr"])
    parser.add_argument("--baseline-snapshot", help="Verified run directory with the pre-fix application methods")
    parser.add_argument("--mode", choices=("mouse", "draw", "presentation"), required=True)
    parser.add_argument("--recordings-root", default="D:/AC-trae/raw_capture")
    parser.add_argument("--config", default=str(ROOT / "config.json"))
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()
    if "before_fixes" in args.variants:
        if not args.baseline_snapshot:
            parser.error("before_fixes requires --baseline-snapshot")
        from .historical_baseline import inspect_snapshot
        inspect_snapshot(args.baseline_snapshot)
    for name in [args.run_id, *args.clips]:
        if not name or name in (".", "..") or any(x in name for x in ("/", "\\", ":")):
            parser.error("run-id and clips must be directory names")
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("max-frames must be positive")
    if os.environ.get("AIRCONTROL_ENGINE"):
        parser.error("Unset AIRCONTROL_ENGINE to exercise actual automatic switching")
    import cv2
    from PyQt6.QtCore import QCoreApplication
    cv2.setNumThreads(1)
    qt_app = QCoreApplication.instance() or QCoreApplication([])
    logging.basicConfig(level=logging.WARNING)
    destination = ROOT / "artifacts/small_hand_tracking" / args.run_id
    destination.mkdir(parents=True, exist_ok=False)
    source = Path(args.config).resolve()
    original_hash = sha(source)
    config = load_config(source, destination)
    snapshot(destination, source, config, args)
    if "before_fixes" in args.variants:
        import shutil
        from .historical_baseline import METHODS
        origin = Path(args.baseline_snapshot).resolve()
        for relative, *_ in METHODS:
            copied = destination / "before_fixes_source" / relative
            copied.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(origin / "source" / relative, copied)
        write_json(destination / "before_fixes_provenance.json", {
            "snapshot": str(origin), "methods": [list(item[:1] + item[2:]) for item in METHODS],
            "source_sha256": {relative: sha(origin / "source" / relative) for relative, *_ in METHODS},
            "execution": "Only the listed verified historical methods are replaced in memory; all other app code, models, and config identical"})
    baseline_path = destination / "baseline.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline["pipeline"] = "application_semantic_replay"
    baseline["scheduling"] = "synchronous construction between source frames; prewarming/Qt queue timing excluded"
    for path in (ROOT / "models").glob("*.onnx"):
        baseline["models_sha256"][str(path.relative_to(ROOT))] = sha(path)
    write_json(baseline_path, baseline)
    comparisons = []
    try:
        for clip in args.clips:
            for variant in args.variants:
                effective = deepcopy(config)
                effective.update(interaction_mode=args.mode, draw_record_trace=False)
                if variant == "no_sr":
                    effective["zoom_sr_engine"] = "none"
                elif variant == "always_zoom":
                    effective.update(long_range_enabled=True, zoom_sr_engine="none")
                elif variant == "responsive_crop":
                    effective.update(zoom_sr_engine="none", experiment_responsive_crop=True)
                elif variant == "responsive_auto":
                    effective["experiment_responsive_crop"] = True
                elif variant == "identity_continuity":
                    effective["experiment_identity_continuity"] = True
                elif variant == "capture_handoff":
                    effective["experiment_capture_handoff"] = True
                elif variant == "shadow_zoom":
                    effective.update(shadow_zoom_handoff_enabled=True)
                print(f"START {clip}/{variant} mode={args.mode}", flush=True)
                context = nullcontext()
                if variant == "before_fixes":
                    from .historical_baseline import before_fixes
                    context = before_fixes(args.baseline_snapshot)
                with context:
                    result = run_clip(Path(args.recordings_root) / clip, destination / clip / variant, effective, args.max_frames)
                comparisons.append(dict(clip=clip, variant=variant, initial_mode=args.mode, **result))
                write_json(destination / "comparison.json", comparisons)
                print(f"DONE {clip}/{variant}: fresh_primary={result['fresh_primary_output_rate']:.4f}; gestures={result['mode_gestures']}", flush=True)
    finally:
        write_json(destination / "config_preservation.json", {"unchanged": sha(source) == original_hash, "before": original_hash, "after": sha(source)})
        if sha(source) != original_hash:
            raise RuntimeError("Source config changed during replay")


if __name__ == "__main__":
    main()
