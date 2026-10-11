"""Compare the rekey fix to the previous final tracker/clear implementation."""
import argparse
import ast
import json
from pathlib import Path

from .corpus_report import compare, read, records
from .historical_baseline import extract
from .replay import ROOT
from .run_replay import sha, write_json


ALLOWED = {"app/services/base_hand_tracker.py": ("BaseHandTracker", "_assign_smoother_keys"),
           "app/modes/mouse_mode.py": ("MouseMode", "handle")}


def verify_sources(before, after):
    a, b = read(before / "baseline.json"), read(after / "baseline.json")
    assert a["models_sha256"] == b["models_sha256"]
    assert a["config_input_sha256"] == b["config_input_sha256"]
    changed = []
    for name, digest in b["source_sha256"].items():
        relative = Path(name)
        canonical = relative.as_posix()
        if relative.parts[0] != "app":
            continue
        assert sha(after / "source" / name) == digest
        assert sha(ROOT / name) == digest, f"Workspace has moved past run: {name}"
        assert sha(before / "source" / name) == a["source_sha256"][name]
        if digest == a["source_sha256"][name]:
            continue
        assert canonical in ALLOWED, f"Unrelated application change: {name}"
        cls, method = ALLOWED[canonical]
        old = ast.parse((before / "source" / name).read_text(encoding="utf-8-sig"))
        new = ast.parse((after / "source" / name).read_text(encoding="utf-8-sig"))
        extract(old, cls, method).body = [ast.Pass()]
        extract(new, cls, method).body = [ast.Pass()]
        assert ast.dump(old) == ast.dump(new), f"Change outside {cls}.{method}"
        changed.append(canonical)
    assert set(changed) == set(ALLOWED)
    assert read(after / "config_preservation.json")["unchanged"]
    return changed


def verify(before_draw, before_mouse, after_draw, after_mouse, destination):
    cases, changes = [], {}
    for mode, before, after in (("draw", before_draw, after_draw), ("mouse", before_mouse, after_mouse)):
        changes[mode] = verify_sources(before, after)
        for clip in read(after / "baseline.json")["arguments"]["clips"]:
            bd, ad = before / clip / "current", after / clip / "current"
            case = compare(clip, mode, bd, ad)
            before_rows = records(bd / "predictions.jsonl")
            rows = records(ad / "predictions.jsonl")
            case["controller_changed_frames"] = [i for i, (x, y) in enumerate(zip(before_rows, rows))
                                                   if x["operation_events"] != y["operation_events"]]
            case["certified_rekeys"] = [dict(frame_index=r["frame_index"], **e)
                                       for r in rows for e in r["events"] if e["kind"] == "same_hand_rekey"]
            cases.append(case)
    assert len(cases) == 34 and len({c["clip"] for c in cases}) == 17
    assert all(c["raw_changed_frames"] == c["hands_changed_frames"] == 0 for c in cases)
    assert all(not c["new_missing_frames"] and not c["stroke_changed_frames"] for c in cases)
    assert all(not c["mouse_changed_frames"] for c in cases), "Mouse events changed; review required"
    assert all(not c["controller_changed_frames"] for c in cases), "Controller events changed; review required"
    focus = after_mouse / "20260722_155625/current"
    row = records(focus / "predictions.jsonl")[1594]
    assert row["mode_state"]["_left_fsm"]["last_transition_reason"] == "release_clear_open"
    assert any(e["kind"] == "same_hand_rekey" and e["from"] == "Primary" and e["to"] == "Left" for e in row["events"])
    assert row["last_fresh_mouse_features"]["thumb_index_pinch"] is False
    assert row["last_fresh_mouse_features"]["geometry_reliable"] is True
    points = row["hands"][0]["keypoints_2d_21x2"]
    import math
    ratio = math.dist(points[4], points[8]) / row["last_fresh_mouse_features"]["hand_width"]
    assert ratio >= .55
    result = dict(completed_pairs=len(cases), recordings=17,
                  replay_frames=sum(c["frames"] for c in cases),
                  native_detection_and_hand_outputs_unchanged=True,
                  drawing_and_mouse_calls_unchanged=True, source_changes=changes,
                  all_controller_events_unchanged=True,
                  rekey_cause_fixed=True, full_product_acceptance=False,
                  release_review=dict(frame=1594, reason="release_clear_open", pinch_ratio=ratio,
                                      features=row["last_fresh_mouse_features"],
                                      limit="Fresh geometry releases immediately; no action intent truth or live drag success established."),
                  cases=cases)
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / "verification.json", result)
    (destination / "report.md").write_text(
        "# 同手改编号修复复测\n\n"
        f"17 段录像、两种初始模式共 34 组本次回放，{result['replay_frames']} 帧。"
        "相对上一版，原始检测、处理后手部输出、绘制坐标和鼠标调用逐帧一致。\n\n"
        "新鲜且无歧义匹配到原主手时，检测器发出仅当帧有效的改编号证明，鼠标保留同手的按下/释放历史。"
        "预测、其他手和不匹配的旧编号仍执行身份重置。\n\n"
        f"问题帧 1594 的松开原因由 reset_identity 变为 release_clear_open，捏合比值 {ratio:.3f} ≥ 0.55。"
        "释放时刻仍为 1594：新鲜几何已判为明确松开，不延迟到旧预测超时。"
        "这证明改编号不再错误重置；不能据此宣称拖拽成功或恢复原意图。\n\n"
        "输入配置、模型及其他生产代码保持一致。这里只验证离线逻辑，完整产品验收仍未完成。\n",
        encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("before_draw", "before_mouse", "after_draw", "after_mouse", "destination"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    r = verify(**vars(args))
    print(json.dumps({k: v for k, v in r.items() if k != "cases"}, ensure_ascii=True, indent=2))
