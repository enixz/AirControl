"""Compare native tracker outputs at two wall-clock replay speeds."""
import argparse
import json
from pathlib import Path

from .replay import ROOT
from .run_replay import load_config, run_clip, sha, snapshot, write_json


def semantics(row):
    keys = ("source_timestamp", "source_size", "raw_hands", "hands", "before", "after",
            "events", "upscalers", "used_crop_zoom")
    result = {key: row[key] for key in keys}
    result["model_calls"] = [{"input_size": x["input_size"], "model_timestamp_ms": x["model_timestamp_ms"]}
                             for x in row["model_calls"]]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", default="20260722_164055")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--frames", type=int, default=120)
    args = parser.parse_args()
    for value in (args.run_id, args.clip):
        if not value or any(x in value for x in ("/", "\\", ":")) or value in (".", ".."):
            parser.error("clip and run-id must be directory names")
    if args.frames < 1:
        parser.error("frames must be positive")
    import cv2
    cv2.setNumThreads(1)
    destination = ROOT / "artifacts" / "small_hand_tracking" / args.run_id
    destination.mkdir(parents=True, exist_ok=False)
    config_path = ROOT / "config.json"
    before = sha(config_path)
    config = load_config(config_path, destination)
    snapshot(destination, config_path, config, args)
    rec = Path("D:/AC-trae/raw_capture") / args.clip
    for name, delay in (("no_delay", 0), ("delayed", 15)):
        run_clip(rec, destination / name, config, args.frames, delay)
    load = lambda name: [semantics(json.loads(x)) for x in (destination / name / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    first, second = load("no_delay"), load("delayed")
    differing = [i for i, (a, b) in enumerate(zip(first, second)) if a != b]
    result = {"frames": len(first), "exact_semantic_match": first == second, "differing_frames": differing,
              "delay_per_frame_ms": 15, "config_unchanged": before == sha(config_path),
              "compared": "Raw/processed landmarks, freshness, identity, source/model times, crop state, SR tier and transitions; performance durations excluded"}
    write_json(destination / "clock_verification.json", result)
    print(json.dumps(result, ensure_ascii=False))
    if not result["exact_semantic_match"] or not result["config_unchanged"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
