"""手势特性 A/B 对比脚本（实施方案 Phase 3.1/3.2/3.3 验证）

用本地录像离线对比三个新特性开关前后的**手势指标**：
  - pinch_hysteresis（Phase 3.2）：pinch 状态翻转次数、滞回带帧数
  - thumb_perp_ratio（Phase 3.3）：thumb_extended 翻转次数、perp_ratio 分布
  - pinch_freeze（Phase 3.1）：pinch 上升沿后 grace 期内手部漂移（收益评估）

设计原则（参考项目 memory 教训）：
  - 同一批 landmarks 喂给 4 个独立 recognizer（各自维护滞回状态），公平对比
  - 检测阶段只跑一次（find_hands 依赖 smoother 状态，重跑会变）
  - 单元测试验证过逻辑本身；本脚本测"实际录像场景下的指标差异"

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


def _count_flips(states):
    """统计 bool 序列的翻转次数。"""
    flips = 0
    prev = None
    for s in states:
        if prev is not None and s != prev:
            flips += 1
        prev = s
    return flips


def _analyze_config(cached_landmarks, pinch_hyst, thumb_perp):
    """用指定配置跑所有帧，返回手势指标。

    pinch_hyst: pinch_hysteresis_enabled
    thumb_perp: thumb_perp_ratio_enabled
    """
    from services.gesture_recognizer import GestureRecognizer

    gr = GestureRecognizer()
    gr.pinch_hysteresis_enabled = pinch_hyst
    gr.thumb_perp_ratio_enabled = thumb_perp

    pinch_states = []
    thumb_ext_states = []
    idx_ratios = []
    mid_ratios = []
    perp_ratios = []
    # freeze 收益：记录每次 pinch 上升沿后 grace 期内食指尖漂移
    freeze_drifts = []
    pinch_rising_frame = None  # pinch 上升沿的帧索引
    idx_tip_at_rise = None     # 上升沿时食指尖位置
    GRACE_FRAMES = 9  # ~0.3s @ 30fps

    for fi, hands in enumerate(cached_landmarks):
        if not hands:
            pinch_states.append(False)
            thumb_ext_states.append(False)
            # 手丢失 → pinch 释放，结束当前 freeze 窗口
            pinch_rising_frame = None
            idx_tip_at_rise = None
            gr._reset_state()
            continue

        lm = hands[0]  # 主手
        features = gr.get_hand_features(lm)

        is_pinch = features.get("thumb_index_pinch", False)
        pinch_states.append(is_pinch)
        thumb_ext_states.append(features.get("thumb_extended", False))

        # 记录比值（用于标定）
        hw = features.get("hand_width", 0) or 1
        thumb_idx_dist = math.hypot(lm[4][1] - lm[8][1], lm[4][2] - lm[8][2])
        idx_ratios.append(thumb_idx_dist / hw)
        perp_ratios.append(features.get("thumb_perp_ratio", 0.0))

        # freeze 收益分析：检测 pinch 上升沿，记录后续 grace 期内食指尖漂移
        if (len(pinch_states) >= 2 and is_pinch and not pinch_states[-2]):
            # 上升沿
            pinch_rising_frame = fi
            idx_tip_at_rise = (lm[8][1], lm[8][2])
        elif (pinch_rising_frame is not None
              and fi - pinch_rising_frame <= GRACE_FRAMES
              and idx_tip_at_rise is not None):
            # grace 期内，计算食指尖相对上升沿的漂移
            dx = lm[8][1] - idx_tip_at_rise[0]
            dy = lm[8][2] - idx_tip_at_rise[1]
            freeze_drifts.append(math.hypot(dx, dy))
        elif is_pinch and pinch_rising_frame is not None:
            # grace 结束或 pinch 释放
            if fi - pinch_rising_frame > GRACE_FRAMES:
                pinch_rising_frame = None
                idx_tip_at_rise = None
        if not is_pinch:
            pinch_rising_frame = None
            idx_tip_at_rise = None

    # 滞回带帧数：idx_ratio 在 0.30-0.40 之间
    in_band = sum(1 for r in idx_ratios if 0.30 <= r <= 0.40)

    return {
        "pinch_flips": _count_flips(pinch_states),
        "pinch_frames": sum(pinch_states),
        "pinch_in_hysteresis_band": in_band,
        "thumb_ext_flips": _count_flips(thumb_ext_states),
        "idx_ratio_mean": statistics.mean(idx_ratios) if idx_ratios else 0,
        "idx_ratio_std": statistics.stdev(idx_ratios) if len(idx_ratios) > 1 else 0,
        "perp_ratio_mean": statistics.mean(perp_ratios) if perp_ratios else 0,
        "perp_ratio_std": statistics.stdev(perp_ratios) if len(perp_ratios) > 1 else 0,
        "freeze_drift_mean_px": statistics.mean(freeze_drifts) if freeze_drifts else 0,
        "freeze_drift_p95_px": (sorted(freeze_drifts)[min(len(freeze_drifts) - 1, int(len(freeze_drifts) * 0.95))]
                                if freeze_drifts else 0),
        "freeze_events": len([1 for i in range(1, len(pinch_states))
                              if pinch_states[i] and not pinch_states[i-1]]),
    }


def _print_compare_table(results, labels):
    """打印 4 种配置的对比表。"""
    keys = [
        ("pinch_flips", "{:.0f}", "pinch 状态翻转次数（↓好）"),
        ("pinch_frames", "{:.0f}", "pinch 帧数"),
        ("pinch_in_hysteresis_band", "{:.0f}", "滞回带内帧数 0.30-0.40"),
        ("thumb_ext_flips", "{:.0f}", "thumb_extended 翻转次数（↓好）"),
        ("idx_ratio_mean", "{:.3f}", "idx_ratio 均值（标定用）"),
        ("idx_ratio_std", "{:.3f}", "idx_ratio 标准差"),
        ("perp_ratio_mean", "{:.3f}", "perp_ratio 均值（标定用）"),
        ("perp_ratio_std", "{:.3f}", "perp_ratio 标准差"),
        ("freeze_events", "{:.0f}", "pinch 上升沿次数"),
        ("freeze_drift_mean_px", "{:.1f}", "grace 期漂移均值 px（freeze 收益）"),
        ("freeze_drift_p95_px", "{:.1f}", "grace 期漂移 P95 px"),
    ]

    w = 14
    print()
    print("=" * 100)
    print(f"{'指标':<32} {'说明':<28} {labels[0]:>{w}} {labels[1]:>{w}} {labels[2]:>{w}} {labels[3]:>{w}}")
    print("=" * 100)
    for key, fmt, desc in keys:
        vals = [fmt.format(results[i].get(key, 0)) for i in range(4)]
        print(f"{key:<32} {desc:<28} {vals[0]:>{w}} {vals[1]:>{w}} {vals[2]:>{w}} {vals[3]:>{w}}")
    print("=" * 100)


def _print_conclusions(results, labels):
    """打印结论分析。"""
    print("\n" + "=" * 100)
    print("结论分析")
    print("=" * 100)

    base = results[0]  # baseline

    # Phase 3.2: pinch 滞回
    hyst = results[1]
    flip_reduction = base["pinch_flips"] - hyst["pinch_flips"]
    flip_pct = (flip_reduction / base["pinch_flips"] * 100) if base["pinch_flips"] else 0
    print(f"\n[Phase 3.2] pinch 双阈值滞回:")
    print(f"  pinch 翻转次数: {base['pinch_flips']} → {hyst['pinch_flips']}  "
          f"(减少 {flip_reduction} 次, -{flip_pct:.1f}%)")
    print(f"  滞回带内帧数: {base['pinch_in_hysteresis_band']} 帧 "
          f"(这些帧在旧版会抖动，新版滞回保持稳定)")
    if base["pinch_in_hysteresis_band"] > 0:
        print(f"  → 滞回对 {base['pinch_in_hysteresis_band']} 帧有效（占 pinch 帧的 "
              f"{base['pinch_in_hysteresis_band']/max(base['pinch_frames'],1)*100:.1f}%）")

    # Phase 3.3: thumb_perp 旋转不变
    perp = results[2]
    ext_flip_reduction = base["thumb_ext_flips"] - perp["thumb_ext_flips"]
    ext_flip_pct = (ext_flip_reduction / base["thumb_ext_flips"] * 100) if base["thumb_ext_flips"] else 0
    print(f"\n[Phase 3.3] thumb_extended 旋转不变判定:")
    print(f"  thumb_extended 翻转次数: {base['thumb_ext_flips']} → {perp['thumb_ext_flips']}  "
          f"(减少 {ext_flip_reduction} 次, -{ext_flip_pct:.1f}%)")
    print(f"  perp_ratio 分布: 均值 {perp['perp_ratio_mean']:.3f}, 标准差 {perp['perp_ratio_std']:.3f}")
    print(f"  → 标定建议: 阈值 {perp['perp_ratio_mean']:.2f} 附近（当前 0.50）")

    # Phase 3.1: freeze 收益
    print(f"\n[Phase 3.1] freeze-on-pinch 收益评估:")
    print(f"  pinch 上升沿次数: {base['freeze_events']}")
    if base["freeze_drift_mean_px"] > 0:
        print(f"  grace 期内食指尖漂移: 均值 {base['freeze_drift_mean_px']:.1f}px, "
              f"P95 {base['freeze_drift_p95_px']:.1f}px")
        if base["freeze_drift_mean_px"] > 5:
            print(f"  → 漂移 >5px，freeze 能显著消除点击漂移（建议开启 pinch_freeze_enabled）")
        else:
            print(f"  → 漂移较小，freeze 收益有限（可暂不开启）")
    else:
        print(f"  → 录像中无完整 pinch 上升沿，无法评估 freeze 收益")

    # idx_ratio 标定
    print(f"\n[标定参考] pinch 比值分布:")
    print(f"  idx_ratio 均值 {base['idx_ratio_mean']:.3f}, 标准差 {base['idx_ratio_std']:.3f}")
    print(f"  → 真实捏合 idx_ratio 应 < 0.30（ENTER 阈值），")
    print(f"     握拳 idx_ratio 应 > 0.40（EXIT 阈值）。若均值落在 0.30-0.40 说明录像多为边界状态")


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

    labels = [c[2] for c in configs]
    _print_compare_table(results, labels)
    _print_conclusions(results, labels)

    # 保存结果
    out_path = os.path.join(args.rec_dir, "gesture_ab_result.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"configs": labels, "results": results}, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {out_path}")


if __name__ == "__main__":
    main()
