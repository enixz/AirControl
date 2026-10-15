"""Compare a fresh-only clear fix against the preceding native draw replay."""
import argparse
import json
from pathlib import Path

from .replay import ROOT
from .run_replay import sha, write_json


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify(before_dir, after_dir, clip):
    relative = Path(clip) / "current"
    before_snapshot, after_snapshot = (read(p / "baseline.json") for p in (before_dir, after_dir))
    before, after = ([json.loads(line) for line in (p / relative / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
                     for p in (before_dir, after_dir))
    assert len(before) == len(after) and before
    assert [r["source_timestamp"] for r in before] == [r["source_timestamp"] for r in after]
    assert read(before_dir / relative / "effective_config.json") == read(after_dir / relative / "effective_config.json")
    assert before_snapshot["models_sha256"] == after_snapshot["models_sha256"]
    changed = [k for k, value in before_snapshot["source_sha256"].items()
               if k.startswith("app") and after_snapshot["source_sha256"].get(k) != value]
    assert changed == ["app\\modes\\draw_mode.py"], changed

    def clear_frames(rows):
        return [r["frame_index"] for r in rows if r["operation_counts"]["overlay"].get("clear_canvas", 0)]

    def predicted_clear_frames(rows):
        return [r["frame_index"] for r in rows if r["operation_counts"]["overlay"].get("clear_canvas", 0)
                and (not r["hands"] or r["hands"][0]["detected_fresh"] is not True)]

    def comparable_events(row):
        # Older event lists omitted these two methods; compare their counts separately.
        return [e for e in row["operation_events"] if e["method"] not in ("clear_canvas", "toggle_shape_correction")]

    result = {
        "pipeline": "application_semantic_replay", "frames": len(after),
        "baseline": str(before_dir.resolve()), "current": str(after_dir.resolve()),
        "changed_application_sources": changed, "models_and_config_identical": True,
        "different_frame_counts": {k: sum(a[k] != b[k] for a, b in zip(before, after, strict=True))
                                   for k in ("raw_hands", "hands", "mode_state", "mode_gesture")},
        "stroke_and_mouse_event_changed_frames": sum(comparable_events(a) != comparable_events(b) for a, b in zip(before, after, strict=True)),
        "shape_toggle_changed_frames": sum(a["operation_counts"]["overlay"].get("toggle_shape_correction", 0) != b["operation_counts"]["overlay"].get("toggle_shape_correction", 0)
                                          for a, b in zip(before, after, strict=True)),
        "clear_frames_before": clear_frames(before), "clear_frames_after": clear_frames(after),
        "nonfresh_clear_frames_before": predicted_clear_frames(before),
        "nonfresh_clear_frames_after": predicted_clear_frames(after),
        "config_preserved": read(after_dir / "config_preservation.json")["unchanged"]
                            and sha(ROOT / "config.json") == after_snapshot["config_input_sha256"],
        "current_app_matches_snapshot": all(sha(ROOT / k) == v for k, v in after_snapshot["source_sha256"].items() if k.startswith("app")),
        "limits": "Development recording and inert sinks; fresh-only clears verified, but action intent and live responsiveness remain unverified",
    }
    result["passed"] = (
        all(result["different_frame_counts"][k] == 0 for k in ("raw_hands", "hands", "mode_state"))
        and result["stroke_and_mouse_event_changed_frames"] == 0
        and result["shape_toggle_changed_frames"] == 0
        and not result["nonfresh_clear_frames_after"]
        and result["config_preserved"] and result["current_app_matches_snapshot"]
    )
    write_json(after_dir / "clear_verification.json", result)
    if not result["passed"]:
        raise AssertionError("Inspect clear_verification.json for regressions")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--clip", default="20260722_145659")
    args = parser.parse_args()
    print(json.dumps(verify(args.before, args.after, args.clip), ensure_ascii=False, indent=2))
