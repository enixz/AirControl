"""Shadow-ZOOM handoff A/B comparison: current vs shadow_zoom variants.

Compares two variants of the same application-semantics replay run where the
only intended config difference is shadow_zoom_handoff_enabled. Unlike
corpus_report, this tool asserts the config difference is exactly that flag,
then reports per-clip coverage, FSM time-in-state, promotion events, and
operation differences. Coverage denominators are all frames (including
hand-out-of-frame segments); this is not detection recall or click accuracy.
"""
import argparse
import json
import math
from collections import Counter
from pathlib import Path

from .run_replay import write_json


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def fresh(row):
    return bool(row["hands"] and row["hands"][0]["detected_fresh"] is True)


def ranges(indices):
    result = []
    for index in indices:
        if result and result[-1][1] == index:
            result[-1][1] = index + 1
        else:
            result.append([index, index + 1])
    return result


def compare(clip, mode, base_dir, shadow_dir):
    base_cfg, shadow_cfg = read(base_dir / "effective_config.json"), read(shadow_dir / "effective_config.json")
    diff_keys = {k for k in set(base_cfg) | set(shadow_cfg) if base_cfg.get(k) != shadow_cfg.get(k)}
    assert diff_keys == {"shadow_zoom_handoff_enabled"}, f"unexpected config diff: {diff_keys}"
    assert read(base_dir / "dataset_manifest.json") == read(shadow_dir / "dataset_manifest.json")
    base, shadow = records(base_dir / "predictions.jsonl"), records(shadow_dir / "predictions.jsonl")
    assert len(base) == len(shadow)
    assert [r["source_timestamp"] for r in base] == [r["source_timestamp"] for r in shadow]

    recovered = [i for i, (a, b) in enumerate(zip(base, shadow, strict=True)) if not fresh(a) and fresh(b)]
    lost = [i for i, (a, b) in enumerate(zip(base, shadow, strict=True)) if fresh(a) and not fresh(b)]

    def promotions(rows):
        events = []
        for r in rows:
            for e in r["events"]:
                if e.get("kind") == "fsm_transition" and e.get("to") == "far_track":
                    events.append(dict(frame_index=r["frame_index"], reason=e.get("reason", "")))
        return events

    def counts(rows):
        return dict(Counter(f'{e["sink"]}.{e["method"]}' for r in rows for e in r["operation_events"]))

    def mouse_events(r):
        return [e for e in r["operation_events"] if e["sink"] == "mouse"]

    base_ops, shadow_ops = counts(base), counts(shadow)
    mouse_changed = [i for i, (a, b) in enumerate(zip(base, shadow, strict=True)) if mouse_events(a) != mouse_events(b)]

    # Post-promotion continuity: frames after the first promotion in the
    # shadow variant, checking wrist jumps (> 0.5 * frame diagonal) between
    # consecutive fresh frames as a coarse wrong-hand / teleport indicator.
    promo_frames = [p["frame_index"] for p in promotions(shadow)]
    post_promo = [r for r in shadow if promo_frames and r["frame_index"] >= promo_frames[0]]
    jumps = []
    prev_wrist = None
    for r in post_promo:
        if r["hands"] and r["hands"][0]["detected_fresh"] is True:
            w = r["hands"][0]["keypoints_2d_21x2"][0]
            if prev_wrist is not None and math.dist(w, prev_wrist) > 960:
                jumps.append(r["frame_index"])
            prev_wrist = w
        else:
            prev_wrist = None

    return dict(clip=clip, initial_mode=mode, frames=len(shadow),
                base_fresh=sum(map(fresh, base)), shadow_fresh=sum(map(fresh, shadow)),
                recovered_frames=recovered, new_missing_frames=lost,
                recovered_intervals=ranges(recovered), new_missing_intervals=ranges(lost),
                base_state_frames=dict(Counter(r["fsm_state"] for r in base)),
                shadow_state_frames=dict(Counter(r["fsm_state"] for r in shadow)),
                base_promotions=promotions(base), shadow_promotions=promotions(shadow),
                operations_base=base_ops, operations_shadow=shadow_ops,
                mouse_changed_frames=mouse_changed,
                post_promotion_wrist_jump_frames=jumps,
                limits="Coverage over all frames is not accuracy; geometric agreement and "
                       "continuity are not identity truth; operation changes need intent review; "
                       "no live latency measured.")


def render(run, destination):
    baseline = read(run / "baseline.json")
    mode = baseline["arguments"]["mode"]
    clips = baseline["arguments"]["clips"]
    cases, pending = [], []
    for clip in clips:
        base_dir, shadow_dir = run / clip / "current", run / clip / "shadow_zoom"
        if not all((p / "summary.json").exists() for p in (base_dir, shadow_dir)):
            pending.append(clip)
            continue
        cases.append(compare(clip, mode, base_dir, shadow_dir))
    totals = {"completed_pairs": len(cases), "pending_pairs": len(pending),
              "paired_source_frames": sum(c["frames"] for c in cases),
              "recovered_primary_frames": sum(len(c["recovered_frames"]) for c in cases),
              "new_missing_primary_frames": sum(len(c["new_missing_frames"]) for c in cases),
              "clips_with_promotion": sum(1 for c in cases if c["shadow_promotions"]),
              "clips_with_new_missing": [c["clip"] for c in cases if c["new_missing_frames"]],
              "clips_with_mouse_event_changes": [c["clip"] for c in cases if c["mouse_changed_frames"]],
              "clips_with_post_promotion_jumps": [c["clip"] for c in cases if c["post_promotion_wrist_jump_frames"]]}
    destination.mkdir(parents=True, exist_ok=True)
    lines = ["# 影子 ZOOM 确认晋升 A/B 对照（current vs shadow_zoom）", "",
             f"已完成 {len(cases)} 段对照（初始模式 {mode}），待完成 {len(pending)} 段。",
             "两侧唯一配置差异为 shadow_zoom_handoff_enabled。覆盖统计分母为全部帧（含收手/出画段），不是准确率；",
             "几何一致与轨迹连续不是身份真值；操作调用变化需意图核对；未测实机时延。",
             "预期边界：近距大目标与双手重叠段 YOLO 强于 MP（第 3 步全量已记录同类差异）；",
             "晋升后（FAR_TRACK 用 MP）此类段的新增缺失不构成远距交接回退，near_frames 判据最终弹回 NEAR。",
             "| 录像 | 帧数 | 新鲜主手 current | 新鲜主手 shadow | 恢复帧 | 新增缺失帧 | 晋升次数 | 晋升后跳变帧 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for c in cases:
        lines.append(f"| {c['clip']} | {c['frames']} | {c['base_fresh']} | {c['shadow_fresh']} | "
                     f"{len(c['recovered_frames'])} | {len(c['new_missing_frames'])} | "
                     f"{len(c['shadow_promotions'])} | {len(c['post_promotion_wrist_jump_frames'])} |")
    lines += ["", "## 每段细节", ""]
    for c in cases:
        lines.append(f"- {c['clip']}：FSM current {c['base_state_frames']} → shadow {c['shadow_state_frames']}")
        for p in c["shadow_promotions"]:
            lines.append(f"  - 晋升 帧{p['frame_index']}：{p['reason']}")
        if c["recovered_intervals"]:
            lines.append(f"  - 恢复区间 {c['recovered_intervals']}")
        if c["new_missing_intervals"]:
            lines.append(f"  - 新增缺失区间 {c['new_missing_intervals']}")
        if c["mouse_changed_frames"]:
            lines.append(f"  - 鼠标事件差异帧 {c['mouse_changed_frames']}")
        if c["post_promotion_wrist_jump_frames"]:
            lines.append(f"  - 晋升后手腕跳变帧 {c['post_promotion_wrist_jump_frames']}（粗指标，需人工复核）")
    lines += ["", "逐帧差异见同目录 comparison.json。", ""]
    (destination / "report.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(destination / "comparison.json", dict(totals=totals, cases=cases, pending=pending,
                                                    limits="A/B over replay semantics only; see per-case limits."))
    return totals


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="run_application 输出目录（含各 clip 的 current 与 shadow_zoom）")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(render(args.run, args.output), ensure_ascii=True, indent=2))
