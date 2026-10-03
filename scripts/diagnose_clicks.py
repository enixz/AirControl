"""逐帧诊断回放：把远距点击异常归因到具体原因。

基座与 artifacts/recognition_optimization_20261003/replay_mouse.py 完全一致
（真实 MouseMode 跑在缓存检测结果上，inert controllers），因此指标可与其
mouse_optimized.json 直接对照；在此之上为每个异常片段叠加逐帧证据。

用法：
    python scripts/diagnose_clicks.py --recording 20260722_145659 \
        [--data artifacts/recognition_optimization_20261003/results/optimized_controlled] \
        [--source artifacts/recognition_optimization_20261003/sources/mouse_final] \
        [--truth D:/AC-trae/raw_capture/<rec>/truth_events.jsonl] \
        [--out artifacts/click_diagnosis]

输出（out 目录）：
    <rec>/events.json        事件级归因清单（每条含分类与证据摘要）
    <rec>/frames.csv         异常片段 ±2s 的逐帧明细
    <rec>/clips/<idx>/f%05d.jpg  叠加诊断信息的视频帧（--frames 才写）
    <rec>/summary.json       与 comparison.json 对齐的指标复核

归因分类（每个未匹配按下 / 拖拽中断）：
    hand_lost          回放中该片段内手输出连续丢失（无 fresh 且无预测）
    pinch_geometry     有手但 idx 比值越过阈值（几何误判 / 实际松手，细分见 cause）
    primary_switch     主手 track_key 变化（跨手继承）
    prediction_gap     预测补帧窗口内按下被 0.12s 上限掐断
    hold_time_cap      保持期超时（当前代码无此机制，占位）
    unclear            证据不足
"""
from __future__ import annotations

import argparse
import bisect
import csv
import gzip
import importlib.util
import json
import logging
import sys
import time as real_time
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[1]
OPT = ROOT / "artifacts" / "recognition_optimization_20261003"
EVAL = ROOT / "artifacts" / "recognition_eval_20261003"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fmt_clock(t):
    return datetime.fromtimestamp(t).strftime("%H:%M:%S.%f")[:-3]


def load_rows(data_dir: Path, rec: str):
    with gzip.open(data_dir / f"{rec}.jsonl.gz", "rt", encoding="utf8") as stream:
        return [json.loads(line) for line in stream]


def classify_gap(frames, pinch_states, states, i0, i1, det_events):
    """对 [i0, i1) 的失活片段归因。frames 为回放中用到的行切片。"""
    segment = frames[i0:i1]
    if not segment:
        return "unclear", {}
    no_output = [r for r in segment if not r["hands"]]
    pred_only = [r for r in segment if r["hands"] and all(
        g.get("predicted") for g in r["gestures"])]
    fresh = [r for r in segment if r["fresh_hands"]]
    # 全程无手输出 → 检测丢失
    if len(no_output) == len(segment):
        return "hand_lost", {"no_output_frames": len(no_output)}
    # 有手输出的帧全部是预测补帧 → 预测窗口
    if not fresh and pred_only:
        return "prediction_gap", {"pred_frames": len(pred_only)}
    # 有 fresh 手但 pinch 为 False：几何原因（含实际松手），细分看比值轨迹
    ratios = []
    for r in fresh:
        f = r["features"]
        hw = f.get("hand_width")
        if hw:
            lm = r["hands"][0]
            ti = ((lm[4][1] - lm[8][1]) ** 2 + (lm[4][2] - lm[8][2]) ** 2) ** 0.5
            ratios.append(ti / hw)
    if ratios:
        med = sorted(ratios)[len(ratios) // 2]
        if med >= 0.55:
            cause = "real_release"
        elif med >= 0.40:
            cause = "boundary_jitter"
        else:
            cause = "geometry_miss"
        return "pinch_geometry", {"median_ratio": round(med, 3),
                                  "ratio_min": round(min(ratios), 3),
                                  "ratio_max": round(max(ratios), 3),
                                  "sub": cause}
    return "unclear", {}


def classify_onset(frames_row):
    """对一次新按下的帧归因（误触来源）。"""
    lm = frames_row["hands"][0] if frames_row["hands"] else None
    g = frames_row["gestures"][0] if frames_row["gestures"] else {}
    if lm is None:
        return "unclear", {}
    hw = frames_row["features"].get("hand_width") or 40.0
    ti = ((lm[4][1] - lm[8][1]) ** 2 + (lm[4][2] - lm[8][2]) ** 2) ** 0.5
    ratio = ti / hw if hw else None
    evid = {
        "track_key": g.get("track_key"),
        "predicted": g.get("predicted", False),
        "ml_label": g.get("ml_label"),
        "hand_width": round(hw, 1),
        "idx_ratio": round(ratio, 3) if ratio else None,
    }
    label = g.get("ml_label")
    if label == "Closed_Fist":
        return "fist_misread", evid
    if ratio is not None and ratio < 0.35 and label in (None, "None", "OTHER"):
        return "small_hand_geometry", evid
    return "pinch_confirmed", evid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recording", default="20260722_145659")
    parser.add_argument("--data", type=Path,
                        default=OPT / "results" / "optimized_controlled")
    parser.add_argument("--source", type=Path,
                        default=OPT / "sources" / "mouse_final")
    parser.add_argument("--truth", type=Path, default=None)
    parser.add_argument("--out", type=Path,
                        default=ROOT / "artifacts" / "click_diagnosis")
    parser.add_argument("--window", type=float, default=2.0,
                        help="异常片段前后保留秒数（逐帧明细与截图）")
    parser.add_argument("--frames", action="store_true",
                        help="额外输出叠加诊断信息的 JPG 帧")
    parser.add_argument("--video", type=Path, default=None,
                        help="原始视频（--frames 时叠加用）")
    args = parser.parse_args()

    rec = args.recording
    truth_path = args.truth or Path(rf"D:\AC-trae\raw_capture\{rec}\truth_events.jsonl")
    out_dir = args.out / rec
    out_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(args.source / "app"))
    from modes.mouse_mode import MouseMode          # noqa: E402
    from services.gesture_recognizer import GestureRecognizer  # noqa: E402
    from config_manager import ConfigManager        # noqa: E402

    evaluation = load_module("evaluation", EVAL / "run_eval.py")
    metrics = load_module("metrics", EVAL / "aggregate_eval.py")

    logging.disable(logging.CRITICAL)
    rows = load_rows(args.data, rec)
    truth = metrics.load_truth(truth_path)

    clock = evaluation.ReplayClock()
    for name, module in list(sys.modules.items()):
        if name.startswith(("services.", "modes.")) and getattr(module, "time", None) is real_time:
            module.time = clock

    config = ConfigManager(str(args.source / "config.json"))
    # 与 replay_mouse.py 的 optimized 变体一致
    recognizer = GestureRecognizer()
    recognizer.pinch_exit_hysteresis_enabled = True
    clock.value = clock.origin = rows[0]["t"]
    mouse = MagicMock()
    mouse.move_to_normalized.return_value = (500, 400)
    mode = MouseMode(config, recognizer, mouse, MagicMock(), MagicMock(), MagicMock(), MagicMock())
    mode.on_enter()
    mode._watchdog_stop.set()
    mode._watchdog_thread.join()

    states, times, trace = [], [], []
    left_calls = []
    try:
        for row in rows:
            clock.value = row["t"]
            before = mouse.left_down.call_count
            mode.handle(row["hands"], row["gestures"], 1920, 1080)
            holding = bool(mode._is_left_holding)
            states.append(holding)
            times.append(row["t"])
            left_calls.append(mouse.left_down.call_count > before)
            g = row["gestures"][0] if row["gestures"] else {}
            trace.append({
                "t": row["t"], "frame": row["frame"], "holding": holding,
                "left_down": left_calls[-1],
                "n_hands": len(row["hands"]),
                "n_fresh": len(row["fresh_hands"]),
                "predicted": bool(g.get("predicted")),
                "track_key": g.get("track_key"),
                "ml_label": g.get("ml_label"),
                "features": row["features"],
                "hands": row["hands"],
                "gestures": row["gestures"],
                "fresh_hands": row["fresh_hands"],
            })
    finally:
        mode.on_exit()

    det_events = metrics.states_to_events(states, times)
    em = metrics.match_events(truth, det_events, 0.5)
    tw = metrics.weighted_truth(states, times, truth)

    # ---- 事件级归因 ----
    events_report = []
    for i, (s, e) in enumerate(det_events):
        i0 = bisect.bisect_left(times, s)
        i1 = bisect.bisect_right(times, e)
        onset_row = trace[i0] if i0 < len(trace) else {}
        cause, evid = classify_onset(onset_row) if onset_row else ("unclear", {})
        overlaps = any(s < iv[1] and iv[0] < e for iv in truth)
        events_report.append({
            "det_index": i, "start": s, "end": e, "dur": round(e - s, 3),
            "overlaps_truth": overlaps,
            "onset_cause": cause, "onset_evidence": evid,
        })

    gaps_report = []
    for pt in tw.get("per_truth", []):
        for gi, gap in enumerate(pt.get("gaps", [])):
            i0 = bisect.bisect_right(times, gap["start"]) - 1
            i1 = bisect.bisect_right(times, gap["end"])
            internal = gap.get("start", 0) > pt["truth"][0] and gap.get("end", 0) < pt["truth"][1]
            cause, evid = classify_gap(trace, states, times, max(0, i0), min(len(trace), i1), det_events)
            gaps_report.append({
                "truth_index": pt["truth_index"], "gap_index": gi,
                "start": gap["start"], "end": gap["end"],
                "dur": round(gap["end"] - gap["start"], 3),
                "internal_interruption": internal,
                "cause": cause, "evidence": evid,
            })

    # ---- 异常片段清单（去重合并窗口）----
    anomaly_windows = []
    for ev in events_report:
        if not ev["overlaps_truth"]:
            anomaly_windows.append((ev["start"] - args.window, ev["end"] + args.window,
                                    f"false_press[{ev['det_index']}]", ev))
    for gap in gaps_report:
        if gap["internal_interruption"]:
            anomaly_windows.append((gap["start"] - args.window, gap["end"] + args.window,
                                    f"drag_interrupt[t{gap['truth_index']}]", gap))
    missed = [i for i in em.get("miss_truth_indices", [])]
    for ti in missed:
        a, b, _k = truth[ti]
        anomaly_windows.append((a - args.window, b + args.window,
                                f"missed_intent[{ti}]", {"truth": [a, b]}))
    anomaly_windows.sort()

    # ---- 逐帧 CSV ----
    csv_path = out_dir / "frames.csv"
    with open(csv_path, "w", newline="", encoding="utf8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "clock", "t", "frame", "holding", "left_down",
            "n_hands", "n_fresh", "predicted", "track_key", "ml_label",
            "hand_width", "idx_dist", "idx_ratio", "thr_ratio",
            "pinch_feat", "zoom", "in_truth",
        ])
        truth_union = metrics.merge_intervals([(iv[0], iv[1]) for iv in truth])
        for tr in trace:
            lm = tr["hands"][0] if tr["hands"] else None
            hw = (tr["features"] or {}).get("hand_width")
            idx_ratio = thr = ""
            if lm is not None and hw:
                ti = ((lm[4][1] - lm[8][1]) ** 2 + (lm[4][2] - lm[8][2]) ** 2) ** 0.5
                idx_ratio = round(ti / hw, 3)
                # 与 recognizer 语义一致：exit-only 滞回的进入 0.35 / 退出 0.40
                thr = 0.40 if tr["holding"] else 0.35
            in_truth = any(tr["t"] < b and a <= tr["t"] for a, b in truth_union)
            writer.writerow([
                fmt_clock(tr["t"]), round(tr["t"], 3), tr["frame"],
                int(tr["holding"]), int(tr["left_down"]),
                tr["n_hands"], tr["n_fresh"], int(tr["predicted"]),
                tr["track_key"], tr["ml_label"],
                round(hw, 1) if hw else "", idx_ratio, thr,
                tr["features"].get("thumb_index_pinch"),
                "", int(in_truth),
            ])

    summary = {
        "recording": rec,
        "data": str(args.data),
        "source": str(args.source),
        "pipeline": "actual MouseMode on cached landmarks (same as replay_mouse optimized)",
        "event_matching_0.5": {k: em[k] for k in
                               ("truth_events", "detected_events", "hits", "misses",
                                "false_alarms", "false_detected_indices", "miss_truth_indices")},
        "time_weighted": {k: tw.get(k) for k in
                          ("truth_sec", "tp_sec", "fp_sec", "truth_pinch_coverage",
                           "time_precision", "missing_runs", "internal_interruptions",
                           "longest_missing_sec", "longest_internal_gap_sec")},
        "events": events_report,
        "gaps": gaps_report,
        "anomaly_windows": [
            {"start": a, "end": b, "kind": k} for a, b, k, _payload in anomaly_windows
        ],
    }
    (out_dir / "events.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf8")

    print(json.dumps({k: summary[k] for k in
                      ("recording", "event_matching_0.5", "time_weighted")},
                     indent=2, ensure_ascii=False))
    print("\n=== 事件归因 ===")
    for ev in events_report:
        mark = "HIT " if ev["overlaps_truth"] else "FA  "
        print(f"{mark} det[{ev['det_index']:2d}] {fmt_clock(ev['start'])} dur={ev['dur']:.2f}s "
              f"cause={ev['onset_cause']} evid={ev['onset_evidence']}")
    print("\n=== 拖拽中断 / 缺失归因 ===")
    for gap in gaps_report:
        kind = "中断" if gap["internal_interruption"] else "缺失"
        print(f"{kind} t{gap['truth_index']} {fmt_clock(gap['start'])} dur={gap['dur']:.2f}s "
              f"cause={gap['cause']} evid={gap['evidence']}")
    print(f"\n逐帧明细: {csv_path}")


if __name__ == "__main__":
    main()
