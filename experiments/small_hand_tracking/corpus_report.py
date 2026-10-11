"""Aggregate completed before/current pairs without calling coverage accuracy."""
import argparse
from collections import Counter
import json
import math
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


def compare(clip, mode, before_dir, after_dir):
    assert read(before_dir / "effective_config.json") == read(after_dir / "effective_config.json")
    assert read(before_dir / "dataset_manifest.json") == read(after_dir / "dataset_manifest.json")
    before, after = records(before_dir / "predictions.jsonl"), records(after_dir / "predictions.jsonl")
    assert len(before) == len(after)
    assert [r["source_timestamp"] for r in before] == [r["source_timestamp"] for r in after]
    recovered = [i for i, (a, b) in enumerate(zip(before, after)) if not fresh(a) and fresh(b)]
    lost = [i for i, (a, b) in enumerate(zip(before, after)) if fresh(a) and not fresh(b)]
    counts = lambda rows: dict(Counter(f'{e["sink"]}.{e["method"]}' for r in rows for e in r["operation_events"]))
    strokes = lambda r: [e for e in r["operation_events"] if e["method"] in ("draw_to", "force_lift_pen")]
    mouse = lambda r: [e for e in r["operation_events"] if e["sink"] == "mouse"]
    nonfresh_clears = lambda rows: [r["frame_index"] for r in rows if not fresh(r) and any(e["method"] == "clear_canvas" for e in r["operation_events"])]
    structure_changed, coordinate_delta = [], 0.0
    for i, (a, b) in enumerate(zip(before, after)):
        sa, sb = strokes(a), strokes(b)
        if [e["method"] for e in sa] != [e["method"] for e in sb]:
            structure_changed.append(i)
            continue
        for ea, eb in zip(sa, sb):
            if ea["method"] == "draw_to":
                coordinate_delta = max(coordinate_delta, math.dist(ea["args"][:2], eb["args"][:2]))
    return dict(clip=clip, initial_mode=mode, frames=len(after),
                before_fresh=sum(map(fresh, before)), after_fresh=sum(map(fresh, after)),
                recovered_frames=recovered, new_missing_frames=lost,
                recovered_intervals=ranges(recovered), new_missing_intervals=ranges(lost),
                raw_changed_frames=sum(a["raw_hands"] != b["raw_hands"] for a, b in zip(before, after)),
                hands_changed_frames=sum(a["hands"] != b["hands"] for a, b in zip(before, after)),
                stroke_changed_frames=[i for i, (a, b) in enumerate(zip(before, after)) if strokes(a) != strokes(b)],
                stroke_structure_changed_frames=structure_changed,
                max_draw_coordinate_delta_px=coordinate_delta,
                mouse_changed_frames=[i for i, (a, b) in enumerate(zip(before, after)) if mouse(a) != mouse(b)],
                mode_changed_frames=sum(a["mode_gesture"] != b["mode_gesture"] for a, b in zip(before, after)),
                actual_modes_before=dict(Counter(r["mode"] for r in before)),
                actual_modes_after=dict(Counter(r["mode"] for r in after)),
                nonfresh_clear_frames_before=nonfresh_clears(before), nonfresh_clear_frames_after=nonfresh_clears(after),
                operations_before=counts(before), operations_after=counts(after))


def render(runs, destination, current_runs=()):
    cases, pending = [], []
    replacements = {read(p / "baseline.json")["arguments"]["mode"]: p for p in current_runs}
    for run in runs:
        baseline = read(run / "baseline.json")
        mode = baseline["arguments"]["mode"]
        current_run = replacements.get(mode, run)
        if current_run != run:
            current_baseline = read(current_run / "baseline.json")
            assert baseline["models_sha256"] == current_baseline["models_sha256"]
            assert baseline["config_input_sha256"] == current_baseline["config_input_sha256"]
        for clip in baseline["arguments"]["clips"]:
            before, after = run / clip / "before_fixes", current_run / clip / "current"
            if not all((p / "summary.json").exists() for p in (before, after)):
                pending.append(dict(run=str(run), clip=clip, mode=mode))
                continue
            case = compare(clip, mode, before, after)
            case.update(before_directory=str(before.resolve()), current_directory=str(after.resolve()))
            cases.append(case)
    totals = {"completed_pairs": len(cases), "pending_pairs": len(pending),
              "unique_recordings": len({c['clip'] for c in cases}),
              "paired_source_frames": sum(c['frames'] for c in cases),
              "native_frames_executed_in_completed_pairs": 2 * sum(c['frames'] for c in cases),
              "recovered_primary_frames": sum(len(c['recovered_frames']) for c in cases),
              "new_missing_primary_frames": sum(len(c['new_missing_frames']) for c in cases),
              "unique_recovered_source_frames": len({(c['clip'], i) for c in cases for i in c['recovered_frames']}),
              "unique_new_missing_source_frames": len({(c['clip'], i) for c in cases for i in c['new_missing_frames']}),
              "nonfresh_clear_calls_before": sum(len(c['nonfresh_clear_frames_before']) for c in cases),
              "nonfresh_clear_calls_after": sum(len(c['nonfresh_clear_frames_after']) for c in cases)}
    result = {"totals": totals, "cases": cases, "pending": pending,
              "limits": "Coverage over all frames is not accuracy; action changes need video/intent review. Initial mode may change via actual gestures. No live latency measured."}
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / "comparison.json", result)
    lines = ["# 全部现有录像回放对照", "", f"已完成 {len(cases)} 组修复前后对照，覆盖 {totals['unique_recordings']} 段录像；待完成 {len(pending)} 组。",
             "每组使用相同录像、原时间戳、模型和配置。历史方法的范围见运行目录 before_fixes_provenance.json；复用旧结果时，以各自源码快照为准。",
             "覆盖统计包含收手和出画，不能称为准确率；动作调用变化不等于正确操作增减。初始模式可随真实手势切换。", "",
             "| 录像 | 初始模式 | 帧数 | 修复前新鲜主手 | 修复后新鲜主手 | 恢复帧 | 新增缺失帧 | 笔画变化帧 | 鼠标事件变化帧 | 预测清屏 前→后 |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for c in cases:
        lines.append(f"| {c['clip']} | {c['initial_mode']} | {c['frames']} | {c['before_fresh']} | {c['after_fresh']} | {len(c['recovered_frames'])} | {len(c['new_missing_frames'])} | {len(c['stroke_changed_frames'])} | {len(c['mouse_changed_frames'])} | {len(c['nonfresh_clear_frames_before'])}→{len(c['nonfresh_clear_frames_after'])} |")
    lines += ["", "## 操作差异", ""]
    for c in cases:
        if c['stroke_changed_frames']:
            lines.append(f"- {c['clip']}，{c['initial_mode']}：笔画差异帧 {c['stroke_changed_frames']}；绘制/抬笔事件序列变化 {len(c['stroke_structure_changed_frames'])} 帧。可逐项对应的绘制坐标最大差异 {c['max_draw_coordinate_delta_px']:.3f} 像素，不能当作精度改善指标。")
        if c['mouse_changed_frames']:
            lines.append(f"- {c['clip']}，{c['initial_mode']}：鼠标事件在帧 {c['mouse_changed_frames']} 不一致。即使总调用次数相同，也不能判为操作完全一致；需核对时刻、身份和意图。")
    lines += ["", "逐帧差异和实际模式分布见同目录 comparison.json。原视频采样图和 inventory.json 用于复核，不替代完整动作意图标注。", ""]
    (destination / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--current-runs", nargs="*", type=Path, default=[], help="Replacement final-version current runs, matched by initial mode")
    args = parser.parse_args()
    print(json.dumps(render(args.runs, args.output, args.current_runs)["totals"], indent=2))
