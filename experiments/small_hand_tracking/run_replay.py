"""Run reproducible tracker ablations on existing recordings, without config writes.

python -m experiments.small_hand_tracking.run_replay --clips 20260722_145659 \
    --variants A B C D --run-id first_pass --max-frames 200
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
from importlib.metadata import version, PackageNotFoundError
import json
import logging
import math
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

from .replay import ROOT, TrackerReplay, summarize

VARIANTS = {
    "current_tracker": {},
    "A": {"long_range_enabled": False, "inference_max_width": 720},
    "A2": {"long_range_enabled": False, "inference_max_width": 0},
    "B": {"long_range_enabled": True, "zoom_sr_engine": "none"},
    "C": {"long_range_enabled": True, "zoom_sr_engine": "espcn"},
    "D": {"long_range_enabled": True, "zoom_sr_engine": "realesrgan_gpu"},
    "Z_auto": {"long_range_enabled": True, "zoom_sr_engine": "auto"},
    "B_grace": {"long_range_enabled": True, "zoom_sr_engine": "none", "experiment_search_grace": True},
    "B_native": {"long_range_enabled": True, "zoom_sr_engine": "none", "experiment_native_rescue": True},
}


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read_metadata(path, *, allow_capture_gaps=False):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    previous = None
    previous_index = -1
    for index, row in enumerate(rows):
        timestamp = row.get("t")
        if not isinstance(timestamp, (float, int)) or isinstance(timestamp, bool) or not math.isfinite(timestamp):
            raise ValueError(f"Missing/non-finite source timestamp at {path}:{index + 1}")
        if previous is not None and timestamp <= previous:
            raise ValueError(f"Non-increasing source timestamp at {path}:{index + 1}")
        source_index = row.get("i", index)
        if allow_capture_gaps and (type(source_index) is not int or source_index <= previous_index):
            raise ValueError(f"Non-increasing capture index at {path}:{index + 1}")
        if not allow_capture_gaps and source_index != index:
            raise ValueError(f"Non-contiguous frame index at {path}:{index + 1}")
        previous_index = source_index
        previous = timestamp
    if not rows:
        raise ValueError(f"Empty metadata: {path}")
    return rows


def load_config(config_path, destination):
    # ConfigManager can repair/write invalid files: operate on a private copy.
    from config_manager import ConfigManager
    json.loads(config_path.read_text(encoding="utf-8"))
    copied = destination / "config_input.json"
    shutil.copyfile(config_path, copied)
    manager = ConfigManager(str(copied))
    return deepcopy(manager.config)


def variant_config(config, variant):
    result = deepcopy(config)
    if variant != "current_tracker":
        result.update(engine_auto_switch=False, hand_prediction_enabled=False, adaptive_skip_enabled=False)
    result.update(VARIANTS[variant])
    return result


def git_output(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True, encoding="utf-8")


def snapshot(destination, config_path, config, args):
    import cv2
    import onnxruntime as ort
    files = list((ROOT / "app").rglob("*.py")) + list(Path(__file__).parent.glob("*.py"))
    files += [config_path, Path(__file__).with_name("protocol_v2.json")]
    hashes = {}
    for path in files:
        relative = path.relative_to(ROOT) if path.is_relative_to(ROOT) else Path("external_config.json")
        target = destination / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        hashes[str(relative)] = sha(path)
    models = {}
    for name in ("gesture_recognizer.task", "models/hand_landmarker.task", "ESPCN_x2.pb", "Real-ESRGAN_x2plus.onnx"):
        path = ROOT / name
        models[name] = sha(path) if path.exists() else None
    packages = {}
    for name in ("mediapipe", "opencv-contrib-python", "numpy", "onnxruntime-directml"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    (destination / "working_tree.patch").write_text(git_output("diff", "HEAD", "--", "app", "config.json"), encoding="utf-8")
    write_json(destination / "baseline.json", {
        "schema_version": 2, "git_head": git_output("rev-parse", "HEAD").strip(),
        "git_status": git_output("status", "--short"), "source_sha256": hashes, "models_sha256": models,
        "config_input_sha256": sha(config_path), "effective_config": config,
        "python": sys.version, "platform": platform.platform(), "packages": packages,
        "available_ort_providers": ort.get_available_providers(), "cv2_threads": cv2.getNumThreads(),
        "arguments": vars(args), "pipeline": "tracker_only",
        "algorithm_clock": "source timestamps; module-local bindings scoped to each infer call",
        "performance_clock": "unmodified time.perf_counter",
    })


def run_clip(rec_dir, destination, config, max_frames, delay_ms):
    import cv2

    meta = read_metadata(rec_dir / "meta.jsonl")
    video = next((rec_dir / name for name in ("frames.mkv", "frames.mp4", "frames.avi") if (rec_dir / name).exists()), None)
    if video is None:
        raise ValueError(f"No frames video in {rec_dir}")
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot decode {video}")
    destination.mkdir(parents=True, exist_ok=False)
    write_json(destination / "effective_config.json", config)
    manifest = {"video": str(video), "video_sha256": sha(video), "meta_sha256": sha(rec_dir / "meta.jsonl"),
                "metadata_frames": len(meta), "container_fps": cap.get(cv2.CAP_PROP_FPS),
                "metadata_span_seconds": meta[-1]["t"] - meta[0]["t"],
                "frames_requested": min(max_frames, len(meta)) if max_frames else len(meta),
                "timestamp_is_synthetic": False, "ground_truth_used": False}
    write_json(destination / "dataset_manifest.json", manifest)
    tracker = None
    rows = []
    try:
        tracker = TrackerReplay(config)
        with (destination / "predictions.jsonl").open("w", encoding="utf-8") as out, (destination / "events.jsonl").open("w", encoding="utf-8") as events:
            count = manifest["frames_requested"]
            for index in range(count):
                ok, frame = cap.read()
                if not ok or frame is None:
                    raise RuntimeError(f"Video ended at {index}; metadata requested {count}")
                if (meta[index].get("w", frame.shape[1]), meta[index].get("h", frame.shape[0])) != (frame.shape[1], frame.shape[0]):
                    raise ValueError(f"Frame dimensions mismatch at {index}")
                if delay_ms:
                    time.sleep(delay_ms / 1000)
                row = tracker.infer(frame, meta[index]["t"])
                rows.append(row)
                out.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                for event in row["events"]:
                    events.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
                if (index + 1) % 250 == 0:
                    print(f"  {destination.name}: {index + 1}/{count}", flush=True)
            if count == len(meta) and cap.read()[0]:
                raise ValueError("Video has more frames than metadata; alignment must be resolved")
        summary = summarize(rows)
        summary.update(initialization_ms=tracker.initialization_ms, backend=tracker.backend_info(),
                       warmup_performed=False, ground_truth_used=False)
        write_json(destination / "summary.json", summary)
        return summary
    finally:
        cap.release()
        if tracker is not None:
            tracker.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recordings-root", default="D:/AC-trae/raw_capture")
    parser.add_argument("--config", default=str(ROOT / "config.json"))
    parser.add_argument("--clips", nargs="+", required=True)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=["A", "B", "C", "D"])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--delay-ms", type=float, default=0)
    args = parser.parse_args()
    if args.rounds < 1 or args.delay_ms < 0 or (args.max_frames is not None and args.max_frames < 1):
        parser.error("rounds/max-frames must be positive and delay-ms nonnegative")
    if not args.run_id or any(x in args.run_id for x in ("/", "\\", ":")) or args.run_id in (".", ".."):
        parser.error("run-id must be a directory name")
    for clip in args.clips:
        if not clip or any(x in clip for x in ("/", "\\", ":")) or clip in (".", ".."):
            parser.error("clips must be directory names")
    import cv2
    cv2.setNumThreads(1)
    logging.basicConfig(level=logging.WARNING)
    destination = ROOT / "artifacts" / "small_hand_tracking" / args.run_id
    destination.mkdir(parents=True, exist_ok=False)
    config_path = Path(args.config).resolve()
    config_hash = sha(config_path)
    config = load_config(config_path, destination)
    snapshot(destination, config_path, config, args)
    comparisons = []
    try:
        for round_index in range(args.rounds):
            variants = args.variants[round_index % len(args.variants):] + args.variants[:round_index % len(args.variants)]
            for clip in args.clips:
                for variant in variants:
                    print(f"START round={round_index + 1} clip={clip} variant={variant}", flush=True)
                    result = run_clip(Path(args.recordings_root) / clip,
                                      destination / f"round_{round_index + 1}" / clip / variant,
                                      variant_config(config, variant), args.max_frames, args.delay_ms)
                    compact = {key: value for key, value in result.items() if key != "no_fresh_primary_intervals"}
                    comparisons.append(dict(round=round_index + 1, clip=clip, variant=variant, **compact))
                    write_json(destination / "comparison.json", comparisons)
                    print(f"DONE {clip}/{variant}: fresh_primary={result['fresh_primary_output_rate']:.4f}, p95={result['tracker_ms']['p95']:.1f}ms", flush=True)
    finally:
        unchanged = sha(config_path) == config_hash
        write_json(destination / "config_preservation.json", {"unchanged": unchanged, "before": config_hash, "after": sha(config_path)})
        if not unchanged:
            raise RuntimeError("Source config changed during run; inspect before drawing conclusions")


if __name__ == "__main__":
    main()
