"""手势特性 A/B 对比脚本（实施方案 Phase 3.1/3.2/3.3 验证）

用本地录像离线对比三个新特性开关前后的**手势指标**：
  - pinch_hysteresis（Phase 3.2）：pinch 状态翻转次数、滞回带帧数
  - thumb_perp_ratio（Phase 3.3）：thumb_extended 翻转次数、perp_ratio 分布
  - pinch_freeze（Phase 3.1）：pinch 持续期间混合指针的漂移风险（观察性指标）

设计原则（参考项目 memory 教训）：
  - 同一批 landmarks 喂给 4 个独立 recognizer（各自维护滞回状态），公平对比
  - 检测阶段只跑一次（find_hands 依赖 smoother 状态，重跑会变）
  - freeze 指标不把松手帧计入，并使用 MouseMode 的混合指针而非单一食指尖
  - 无人工点击/拖拽真值时只报告观察结果，不自动建议默认开启

带真值模式（评估报告 P1-1）：若录像目录存在 truth_events.jsonl（录制时
用另一只手点按/按住空格标记意图点击/拖拽所生成，见
services/truth_event_logger.py），额外输出检出率/漏检率/误报/onset 延迟
等量化指标，并据此给出"默认开/关"的建议。

用法：
  python benchmark_gesture_ab.py raw_capture/20260705_174137
  python benchmark_gesture_ab.py raw_capture/20260705_174137 --max-frames 300
"""
import argparse
import json
import logging
import math
import os
import statistics
import sys
import time

import cv2

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def _load_config():
    with open(os.path.join(PROJECT_ROOT, "config.json"), encoding="utf-8") as f:
        return json.load(f)


def _iter_frames(rec_dir):
    """按顺序产出 frame（BGR）。优先 frames.mkv，其次 frames.mp4。"""
    for name in ("frames.mkv", "frames.mp4"):
        p = os.path.join(rec_dir, name)
        if os.path.exists(p):
            cap = cv2.VideoCapture(p)
            try:
                while True:
                    ok, fr = cap.read()
                    if not ok or fr is None:
                        break
                    yield fr
            finally:
                cap.release()
            return
    raise FileNotFoundError(f"未找到 frames.mkv 或 frames.mp4 in {rec_dir}")


def _cache_landmarks(rec_dir, cfg, max_frames=None):
    """跑一遍 find_hands，缓存每帧的 hands_landmarks。

    返回 list of (hands_landmarks | None)，None=无手。
    只跑一次，避免 smoother 状态导致重跑结果不一致。
    """
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "app"))
    from services.hand_tracker_factory import create_hand_tracker

    tracker = create_hand_tracker(
        engine=cfg.get("detection_engine", "mediapipe"),
        max_num_hands=2,
        min_detection_confidence=cfg.get("hand_detection_confidence") or 0.6,
        min_presence_confidence=cfg.get("hand_presence_confidence") or 0.5,
        min_tracking_confidence=cfg.get("hand_tracking_confidence") or 0.5,
        preferred_model_type=cfg.get("model_type"),
        dominant_hand=cfg.get("dominant_hand") or "Right",
        config=cfg,
    )

    # 静音 gesture logger（避免 zoom 日志刷屏）
    glog = logging.getLogger("gesture")
    prev_level = glog.level
    glog.setLevel(logging.WARNING)

    cached = []
    t_start = time.perf_counter()
    try:
        for i, frame in enumerate(_iter_frames(rec_dir)):
            if max_frames is not None and i >= max_frames:
                break
            _, hands, _ = tracker.find_hands(frame, draw=False)
            cached.append(hands if hands else None)
            if (i + 1) % 200 == 0:
                print(f"  缓存进度: {i+1} 帧...")
    finally:
        tracker.close()
        glog.setLevel(prev_level)

    elapsed = time.perf_counter() - t_start
    with_hand = sum(1 for h in cached if h)
    print(f"  缓存完成: {len(cached)} 帧, 有手 {with_hand} 帧, 耗时 {elapsed:.1f}s")
    return cached


def _load_truth_context(rec_dir, n_frames):
    """加载逐帧时间戳与真值区间；缺 meta.jsonl 时按 30fps 合成时间戳并告警。"""
    from services.truth_ab import load_frame_times, load_truth_intervals

    meta_path = os.path.join(rec_dir, "meta.jsonl")
    if os.path.exists(meta_path):
        times = load_frame_times(meta_path)
    else:
        print("  [WARN] 未找到 meta.jsonl，按 30fps 合成帧时间戳（延迟指标准确性下降）")
        times = [i / 30.0 for i in range(n_frames)]
    if len(times) > n_frames:
        times = times[:n_frames]

    truth_path = os.path.join(rec_dir, "truth_events.jsonl")
    truth = load_truth_intervals(truth_path) if os.path.exists(truth_path) else []
    return times, truth


def _count_flips(states):
    """统计 bool 序列的翻转次数。"""
    flips = 0
    prev = None
    for s in states:
        if prev is not None and s != prev:
            flips += 1
        prev = s
    return flips


def _p95(values):
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]


def _compute_freeze_observation(pinch_states, pointer_positions, grace_frames=9):
    """Measure pointer drift only while pinch remains active.

    The release frame is deliberately excluded: MouseMode clears freeze as soon as
    pinch is released, so counting release motion would overstate freeze benefit.
    Events with no active continuation frame are reported as unevaluable.
    """
    rising_events = 0
    event_start = None
    pointer_at_rise = None
    current_drifts = []
    event_max_drifts = []
    observed_frames = 0
    previous = False

    def finish_event():
        nonlocal event_start, pointer_at_rise, current_drifts
        if current_drifts:
            event_max_drifts.append(max(current_drifts))
        event_start = None
        pointer_at_rise = None
        current_drifts = []

    for frame_index, (is_pinch, pointer) in enumerate(
        zip(pinch_states, pointer_positions, strict=True)
    ):
        if is_pinch and not previous and pointer is not None:
            finish_event()
            rising_events += 1
            event_start = frame_index
            pointer_at_rise = pointer
        elif event_start is not None:
            within_grace = frame_index - event_start <= grace_frames
            if is_pinch and pointer is not None and within_grace:
                dx = pointer[0] - pointer_at_rise[0]
                dy = pointer[1] - pointer_at_rise[1]
                current_drifts.append(math.hypot(dx, dy))
                observed_frames += 1
            else:
                finish_event()
        previous = is_pinch

    finish_event()
    return {
        "freeze_events": rising_events,
        "freeze_evaluable_events": len(event_max_drifts),
        "freeze_observed_frames": observed_frames,
        "freeze_event_max_drift_mean_px": (
            statistics.mean(event_max_drifts) if event_max_drifts else 0.0
        ),
        "freeze_event_max_drift_p95_px": _p95(event_max_drifts),
    }


def _analyze_config(cached_landmarks, pinch_hyst, thumb_perp):
    """用指定配置跑所有帧，返回手势指标。

    pinch_hyst: pinch_hysteresis_enabled
    thumb_perp: thumb_perp_ratio_enabled
    """
    from modes.mouse_mode import MouseMode
    from services.gesture_recognizer import GestureRecognizer
    from services.mouse_controller import blended_landmark_point

    gr = GestureRecognizer()
    gr.pinch_hysteresis_enabled = pinch_hyst
    gr.thumb_perp_ratio_enabled = thumb_perp

    pinch_states = []
    thumb_ext_states = []
    idx_ratios = []
    perp_ratios = []
    pointer_positions = []

    for _fi, hands in enumerate(cached_landmarks):
        if not hands:
            pinch_states.append(False)
            thumb_ext_states.append(False)
            pointer_positions.append(None)
            gr._reset_state()
            continue

        lm = hands[0]  # 主手
        features = gr.get_hand_features(lm)

        is_pinch = features.get("thumb_index_pinch", False)
        pinch_states.append(is_pinch)
        thumb_ext_states.append(features.get("thumb_extended", False))
        pointer_positions.append(blended_landmark_point(lm, MouseMode.POINTER_WEIGHTS))

        # 记录比值（用于标定）
        hw = features.get("hand_width", 0) or 1
        thumb_idx_dist = math.hypot(lm[4][1] - lm[8][1], lm[4][2] - lm[8][2])
        idx_ratios.append(thumb_idx_dist / hw)
        perp_ratios.append(features.get("thumb_perp_ratio", 0.0))

    # 滞回带帧数：idx_ratio 在 0.30-0.40 之间
    in_band = sum(1 for r in idx_ratios if 0.30 <= r <= 0.40)
    result = {
        "pinch_flips": _count_flips(pinch_states),
        "pinch_frames": sum(pinch_states),
        "pinch_in_hysteresis_band": in_band,
        "pinch_changed_frames_vs_baseline": 0,
        "thumb_ext_flips": _count_flips(thumb_ext_states),
        "idx_ratio_mean": statistics.mean(idx_ratios) if idx_ratios else 0,
        "idx_ratio_std": statistics.stdev(idx_ratios) if len(idx_ratios) > 1 else 0,
        "perp_ratio_mean": statistics.mean(perp_ratios) if perp_ratios else 0,
        "perp_ratio_std": statistics.stdev(perp_ratios) if len(perp_ratios) > 1 else 0,
        "_pinch_states": pinch_states,
    }
    result.update(_compute_freeze_observation(pinch_states, pointer_positions))
    return result


def _fmt_pct(value):
    return "-" if value is None else f"{value:.1%}"


def _fmt_ms(value):
    return "-" if value is None else f"{value:.0f}ms"


def _print_compare_table(results, labels):
    """打印 4 种配置的对比表。"""
    keys = [
        ("pinch_flips", "{:.0f}", "pinch 状态翻转次数（↓好）"),
        ("pinch_frames", "{:.0f}", "pinch 帧数"),
        ("pinch_in_hysteresis_band", "{:.0f}", "滞回带候选帧数 0.30-0.40"),
        ("pinch_changed_frames_vs_baseline", "{:.0f}", "相对基线实际改变的帧数"),
        ("thumb_ext_flips", "{:.0f}", "thumb_extended 翻转次数（↓好）"),
        ("idx_ratio_mean", "{:.3f}", "idx_ratio 均值（标定用）"),
        ("idx_ratio_std", "{:.3f}", "idx_ratio 标准差"),
        ("perp_ratio_mean", "{:.3f}", "perp_ratio 均值（标定用）"),
        ("perp_ratio_std", "{:.3f}", "perp_ratio 标准差"),
        ("freeze_events", "{:.0f}", "pinch 上升沿次数"),
        ("freeze_evaluable_events", "{:.0f}", "可评估持续 pinch 事件数"),
        ("freeze_observed_frames", "{:.0f}", "grace 内持续 pinch 帧数"),
        ("freeze_event_max_drift_mean_px", "{:.1f}", "事件最大混合指针漂移均值"),
        ("freeze_event_max_drift_p95_px", "{:.1f}", "事件最大混合指针漂移 P95"),
    ]

    w = 14
    print()
    print("=" * 100)
    print(f"{'指标':<32} {'说明':<28} {labels[0]:>{w}} {labels[1]:>{w}} {labels[2]:>{w}} {labels[3]:>{w}}")
    print("=" * 100)
    for key, fmt, desc in keys:
        vals = [fmt.format(results[i].get(key, 0)) for i in range(4)]
        print(f"{key:<32} {desc:<28} {vals[0]:>{w}} {vals[1]:>{w}} {vals[2]:>{w}} {vals[3]:>{w}}")

    if results[0].get("truth"):
        print("-" * 100)
        print("真值量化指标（意图点击/拖拽 vs 检测 pinch 事件）")
        truth_rows = [
            ("truth_recall", "检出率 recall（↑好）", "recall", _fmt_pct),
            ("truth_miss_rate", "漏检率（↓好）", "miss_rate", _fmt_pct),
            ("truth_precision", "准确率 precision（↑好）", "precision", _fmt_pct),
            ("truth_false_alarms", "误报次数（↓好）", "false_alarms", str),
            ("truth_onset_mean", "onset 延迟均值", "onset_delay_mean_ms", _fmt_ms),
            ("truth_onset_p95", "onset 延迟 P95", "onset_delay_p95_ms", _fmt_ms),
        ]
        for label, desc, mkey, fmt in truth_rows:
            vals = [fmt(results[i]["truth"][mkey]) for i in range(4)]
            print(f"{label:<32} {desc:<28} {vals[0]:>{w}} {vals[1]:>{w}} {vals[2]:>{w}} {vals[3]:>{w}}")
    print("=" * 100)


def _truth_line(metrics):
    return (
        f"检出 {_fmt_pct(metrics['recall'])}, 漏检 {_fmt_pct(metrics['miss_rate'])}, "
        f"误报 {metrics['false_alarms']}, onsetP95 {_fmt_ms(metrics['onset_delay_p95_ms'])}"
    )


def _print_truth_recommendation(results):
    """基于真值指标给出默认开/关的量化建议。

    判据（保守）：漏检率不升（容差 1pp）、误报不增、onset P95 增幅 < 50ms，
    且至少一项核心指标改善（漏检降 / 误报减 / pinch 翻转降 ≥30%）。
    """
    print("\n[真值量化结论] pinch 点击/拖拽（意图真值 vs 检测事件）:")
    base = results[0]["truth"]
    print(f"  baseline(全关): {_truth_line(base)}")
    for idx, name in ((1, "+hyst"), (2, "+perp"), (3, "+both")):
        print(f"  {name:<16}: {_truth_line(results[idx]['truth'])}")

    base_flips = results[0]["pinch_flips"]
    for idx, name in ((1, "+hyst"), (3, "+both")):
        m = results[idx]["truth"]
        flip_cut = (
            (base_flips - results[idx]["pinch_flips"]) / base_flips
            if base_flips else 0.0
        )
        miss_ok = (m["miss_rate"] or 0) <= (base["miss_rate"] or 0) + 0.01
        fa_ok = m["false_alarms"] <= base["false_alarms"]
        lat_ok = (m["onset_delay_p95_ms"] or 0) <= (base["onset_delay_p95_ms"] or 0) + 50
        improved = (
            (m["miss_rate"] or 0) < (base["miss_rate"] or 0)
            or m["false_alarms"] < base["false_alarms"]
            or flip_cut >= 0.3
        )
        ok = miss_ok and fa_ok and lat_ok and improved
        verdict = "建议默认开启" if ok else "建议继续关闭"
        detail = (
            f"漏检{'OK' if miss_ok else 'X'}, 误报{'OK' if fa_ok else 'X'}, "
            f"延迟{'OK' if lat_ok else 'X'}, 有改善{'OK' if improved else 'X'}"
        )
        print(f"  {name}: {verdict}（{detail}）")


def _print_conclusions(results, has_truth=False):
    """打印结论分析。has_truth=True 时附真值量化建议。"""
    print("\n" + "=" * 100)
    print("结论分析")
    print("=" * 100)

    base = results[0]  # baseline

    # Phase 3.2: pinch 滞回
    hyst = results[1]
    flip_reduction = base["pinch_flips"] - hyst["pinch_flips"]
    flip_pct = (flip_reduction / base["pinch_flips"] * 100) if base["pinch_flips"] else 0
    print("\n[Phase 3.2] pinch 双阈值滞回:")
    print(f"  pinch 翻转次数: {base['pinch_flips']} → {hyst['pinch_flips']}  "
          f"(减少 {flip_reduction} 次, -{flip_pct:.1f}%)")
    print(f"  滞回带候选帧数: {base['pinch_in_hysteresis_band']} 帧")
    print(f"  相对基线实际改变: {hyst['pinch_changed_frames_vs_baseline']} 帧")
    if has_truth:
        print("  → 翻转次数仅供参考，默认开/关建议见下方真值量化结论")
    else:
        print("  → 无人工点击真值，翻转减少不等同于准确率提高，不能据此自动开启默认值")

    # Phase 3.3: thumb_perp 旋转不变
    perp = results[2]
    ext_flip_reduction = base["thumb_ext_flips"] - perp["thumb_ext_flips"]
    ext_flip_pct = (ext_flip_reduction / base["thumb_ext_flips"] * 100) if base["thumb_ext_flips"] else 0
    print("\n[Phase 3.3] thumb_extended 旋转不变判定:")
    print(f"  thumb_extended 翻转次数: {base['thumb_ext_flips']} → {perp['thumb_ext_flips']}  "
          f"(减少 {ext_flip_reduction} 次, -{ext_flip_pct:.1f}%)")
    print(f"  perp_ratio 分布: 均值 {perp['perp_ratio_mean']:.3f}, 标准差 {perp['perp_ratio_std']:.3f}")
    print("  → 样本没有 thumb-tucked/extended 标签，均值不能用于选阈值；继续默认关闭")

    # Phase 3.1: freeze 收益
    print("\n[Phase 3.1] freeze-on-pinch 收益评估:")
    print(f"  pinch 上升沿次数: {base['freeze_events']}")
    print(f"  可评估持续 pinch 事件: {base['freeze_evaluable_events']}")
    if base["freeze_evaluable_events"]:
        print(
            "  grace 内事件最大混合指针漂移: "
            f"均值 {base['freeze_event_max_drift_mean_px']:.1f}px, "
            f"P95 {base['freeze_event_max_drift_p95_px']:.1f}px"
        )
        if has_truth:
            print("  → 漂移观察 + 真值量化结论（见下）共同决定 freeze 默认值")
        else:
            print("  → 这是录像像素的观察指标；仍需实际屏幕点击/拖拽真值决定默认值")
    else:
        print("  → 没有持续至少两个帧的 pinch 事件，无法评估 freeze")

    # idx_ratio 标定
    print("\n[标定参考] pinch 比值分布:")
    print(f"  idx_ratio 均值 {base['idx_ratio_mean']:.3f}, 标准差 {base['idx_ratio_std']:.3f}")
    print("  → 真实捏合 idx_ratio 应 < 0.30（ENTER 阈值），")
    print("     握拳 idx_ratio 应 > 0.40（EXIT 阈值）。若均值落在 0.30-0.40 说明录像多为边界状态")

    if has_truth:
        _print_truth_recommendation(results)


def main():
    ap = argparse.ArgumentParser(description="手势特性 A/B 对比（实施方案 Phase 3.1/3.2/3.3 验证）")
    ap.add_argument("rec_dir", help="录制目录 raw_capture/<时间戳>")
    ap.add_argument("--max-frames", type=int, default=None, help="最多处理帧数（调试用）")
    args = ap.parse_args()

    if not os.path.isdir(args.rec_dir):
        ap.error(f"录制目录不存在: {args.rec_dir}")

    logging.basicConfig(level=logging.WARNING)

    print("=" * 100)
    print("手势特性 A/B 对比测试（实施方案 Phase 3.1/3.2/3.3）")
    print("=" * 100)
    print(f"录像: {args.rec_dir}")

    cfg = _load_config()
    print(f"引擎: {cfg.get('detection_engine', 'mediapipe')}")

    print("\n[1/2] 缓存录像帧 landmarks（只跑一次检测）...")
    cached = _cache_landmarks(args.rec_dir, cfg, max_frames=args.max_frames)

    # 真值上下文（可选）：逐帧时间戳 + 意图点击/拖拽区间
    frame_times, truth_intervals = _load_truth_context(args.rec_dir, len(cached))
    if truth_intervals:
        print(f"  真值事件 {len(truth_intervals)} 个 → 输出检出率/漏检率/延迟量化指标")
    else:
        print("  无 truth_events.jsonl → 仅观察性指标（录制时按住空格可采集真值）")

    print("\n[2/2] 对比 4 种配置的手势指标...")
    configs = [
        (False, False, "baseline(全关)"),  # = v1.3.6 行为
        (True, False, "+hyst"),            # Phase 3.2
        (False, True, "+perp"),           # Phase 3.3
        (True, True, "+both"),             # 全开
    ]
    results = []
    for pinch_hyst, thumb_perp, label in configs:
        print(f"  跑配置: {label} ...")
        r = _analyze_config(cached, pinch_hyst, thumb_perp)
        r["config"] = label
        results.append(r)

    state_sequences = [result.pop("_pinch_states") for result in results]
    baseline_states = state_sequences[0]
    for result, states in zip(results[1:], state_sequences[1:], strict=True):
        result["pinch_changed_frames_vs_baseline"] = sum(
            old != new for old, new in zip(baseline_states, states, strict=True)
        )

    if truth_intervals:
        from services.truth_ab import compute_metrics, states_to_intervals

        for result, states in zip(results, state_sequences, strict=True):
            detected = states_to_intervals(states, frame_times)
            result["truth"] = compute_metrics(truth_intervals, detected)

    labels = [c[2] for c in configs]
    _print_compare_table(results, labels)
    _print_conclusions(results, has_truth=bool(truth_intervals))

    # 保存结果
    out_path = os.path.join(args.rec_dir, "gesture_ab_result.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "schema_version": 2,
                "methodology": {
                    "freeze_pointer": "MouseMode blended pointer in source-video pixels",
                    "release_frame_excluded": True,
                    "has_click_drag_ground_truth": bool(truth_intervals),
                    "truth_onset_tolerance_sec": 0.5,
                },
                "configs": labels,
                "results": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\n结果已保存: {out_path}")


if __name__ == "__main__":
    main()
