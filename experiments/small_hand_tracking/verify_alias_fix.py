"""Compare the production alias fix with the frozen pre-fix full recordings."""
import json
from pathlib import Path

from .replay import ROOT
from .run_replay import sha, write_json


def verify(run_dir):
    root = ROOT / "artifacts/small_hand_tracking"
    run_dir = Path(run_dir)
    cases = {"20260722_145659": "app_far_mouse_20261008", "20260722_164055": "app_near_development_20261008"}
    baseline = json.loads((run_dir / "baseline.json").read_text(encoding="utf-8"))
    result = {"pipeline": "application_semantic_replay", "recordings": {}}
    for clip, previous_run in cases.items():
        before_dir = root / previous_run
        before_baseline = json.loads((before_dir / "baseline.json").read_text(encoding="utf-8"))
        before_config = json.loads((before_dir / clip / "current/effective_config.json").read_text(encoding="utf-8"))
        after_config = json.loads((run_dir / clip / "current/effective_config.json").read_text(encoding="utf-8"))
        changed_app = [k for k, value in before_baseline["source_sha256"].items()
                       if k.startswith("app") and baseline["source_sha256"].get(k) != value]
        assert changed_app == ["app\\services\\base_hand_tracker.py"], changed_app
        assert before_baseline["models_sha256"] == baseline["models_sha256"]
        assert before_config == after_config
        before = [json.loads(x) for x in (before_dir / clip / "current/predictions.jsonl").read_text(encoding="utf-8").splitlines()]
        after = [json.loads(x) for x in (run_dir / clip / "current/predictions.jsonl").read_text(encoding="utf-8").splitlines()]
        assert len(before) == len(after)
        assert [r["source_timestamp"] for r in before] == [r["source_timestamp"] for r in after]

        def fresh(row):
            return bool(row["hands"] and row["hands"][0]["detected_fresh"] is True)

        case = {"frames": len(before), "baseline": str(before_dir),
                "recovered_primary_frames": [i for i, (a, b) in enumerate(zip(before, after)) if not fresh(a) and fresh(b)],
                "new_missing_primary_frames": [i for i, (a, b) in enumerate(zip(before, after)) if fresh(a) and not fresh(b)],
                "different_frame_counts": {k: sum(a[k] != b[k] for a, b in zip(before, after))
                                           for k in ("raw_hands", "hands", "mode_gesture", "operation_events")},
                "left_holding_changed_frames": sum(a["mode_state"]["left_holding"] != b["mode_state"]["left_holding"] for a, b in zip(before, after)),
                "effective_config_identical": True, "models_identical": True,
                "changed_application_sources": changed_app}
        result["recordings"][clip] = case
    result["config_preserved"] = json.loads((run_dir / "config_preservation.json").read_text())["unchanged"]
    result["current_config_matches"] = sha(ROOT / "config.json") == baseline["config_input_sha256"]
    result["current_app_matches_snapshot"] = all(sha(ROOT / k) == v for k, v in baseline["source_sha256"].items() if k.startswith("app"))
    result["passed"] = all(not r["new_missing_primary_frames"]
                           and r["different_frame_counts"]["raw_hands"] == 0
                           and r["different_frame_counts"]["operation_events"] == 0
                           and r["left_holding_changed_frames"] == 0 for r in result["recordings"].values())
    result["passed"] &= result["config_preserved"] and result["current_config_matches"] and result["current_app_matches_snapshot"]
    result["limits"] = "Two development recordings, no independent action/identity truth or live camera/Qt latency measurement"
    write_json(run_dir / "fix_verification.json", result)
    if not result["passed"]:
        raise AssertionError("Alias fix introduced a regression; inspect fix_verification.json")
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    print(json.dumps(verify(parser.parse_args().run_dir), ensure_ascii=False, indent=2))
