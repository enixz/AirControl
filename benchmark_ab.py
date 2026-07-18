"""MediaPipe vs HaGRID YOLO 手部检测器 A/B 对比脚本。

在相同视频上分别用两种引擎跑 find_hands，输出关键指标对比。

用法：
    python benchmark_ab.py raw_capture/20260705_174137
"""
import json
import os
import sys
import time

import cv2

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "app"))


def run_engine(engine_name, video_path, cfg):
    """用指定引擎跑完整视频，返回指标。"""
    from services.hand_tracker_factory import create_hand_tracker

    tracker = create_hand_tracker(
        engine=engine_name,
        max_num_hands=2,
        min_detection_confidence=cfg.get("hand_detection_confidence", 0.5),
        min_presence_confidence=cfg.get("hand_presence_confidence", 0.5),
        min_tracking_confidence=cfg.get("hand_tracking_confidence", 0.5),
        preferred_model_type=cfg.get("model_type"),
        dominant_hand=cfg.get("dominant_hand", "Right"),
        config=cfg,
    )

    cap = cv2.VideoCapture(video_path)
    total = 0
    detected = 0
    multi_hand = 0
    jerks = []
    prev_wrist = None
    timestamps = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        total += 1
        t0 = time.perf_counter()
        _, hands, gestures = tracker.find_hands(frame, draw=False)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        timestamps.append(elapsed_ms)

        n = len(hands)
        if n > 0:
            detected += 1
        if n >= 2:
            multi_hand += 1

        # wrist jerk
        if n > 0:
            wrist = hands[0][0]
            wx, wy = float(wrist[1]), float(wrist[2])
            if prev_wrist is not None:
                dist = ((wx - prev_wrist[0]) ** 2 + (wy - prev_wrist[1]) ** 2) ** 0.5
                jerks.append(dist)
            prev_wrist = (wx, wy)
        else:
            prev_wrist = None

    cap.release()
    tracker.close()

    # 计算指标
    jerks_sorted = sorted(jerks) if jerks else [0]
    jerk_mean = sum(jerks) / len(jerks) if jerks else 0
    jerk_p95 = jerks_sorted[min(len(jerks_sorted) - 1, int(len(jerks_sorted) * 0.95))]
    ts_sorted = sorted(timestamps) if timestamps else [0]
    latency_mean = sum(timestamps) / len(timestamps) if timestamps else 0
    latency_p95 = ts_sorted[min(len(ts_sorted) - 1, int(len(ts_sorted) * 0.95))]

    return {
        "engine": engine_name,
        "total_frames": total,
        "detected_frames": detected,
        "detect_rate": detected / max(total, 1),
        "multi_hand_frames": multi_hand,
        "multi_hand_rate": multi_hand / max(total, 1),
        "jerk_mean_px": jerk_mean,
        "jerk_p95_px": jerk_p95,
        "latency_mean_ms": latency_mean,
        "latency_p95_ms": latency_p95,
        "fps": total / max(sum(timestamps) / 1000, 0.001),
    }


def main():
    if len(sys.argv) < 2:
        print("用法: python benchmark_ab.py <rec_dir>")
        sys.exit(1)

    rec_dir = sys.argv[1]
    mkv = os.path.join(rec_dir, "frames.mkv")
    mp4 = os.path.join(rec_dir, "frames.mp4")
    video = mkv if os.path.exists(mkv) else mp4
    if not os.path.exists(video):
        print(f"未找到视频: {mkv} 或 {mp4}")
        sys.exit(1)

    with open(os.path.join(PROJECT_ROOT, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)

    print(f"=== A/B 对比测试 ===")
    print(f"视频: {video}")
    print()

    results = {}
    for engine in ["mediapipe", "hagrid_yolo"]:
        print(f"--- 运行引擎: {engine} ---")
        result = run_engine(engine, video, cfg)
        results[engine] = result
        print(f"  完成: {result['total_frames']} 帧, 检出 {result['detected_frames']} ({result['detect_rate']:.1%})")
        print()

    # 对比表
    print("=" * 70)
    print(f"{'指标':<25} {'MediaPipe':>20} {'HaGRID YOLO':>20}")
    print("=" * 70)

    mp = results["mediapipe"]
    yo = results["hagrid_yolo"]

    rows = [
        ("总帧数", mp["total_frames"], yo["total_frames"], ""),
        ("检出帧数", mp["detected_frames"], yo["detected_frames"], ""),
        ("检出率", f"{mp['detect_rate']:.1%}", f"{yo['detect_rate']:.1%}", "↑" if yo["detect_rate"] > mp["detect_rate"] else "↓"),
        ("多手帧数", mp["multi_hand_frames"], yo["multi_hand_frames"], ""),
        ("多手率", f"{mp['multi_hand_rate']:.1%}", f"{yo['multi_hand_rate']:.1%}", ""),
        ("抖动均值(px)", f"{mp['jerk_mean_px']:.1f}", f"{yo['jerk_mean_px']:.1f}", "↓好" if yo["jerk_mean_px"] < mp["jerk_mean_px"] else "↑差"),
        ("抖动P95(px)", f"{mp['jerk_p95_px']:.1f}", f"{yo['jerk_p95_px']:.1f}", "↓好" if yo["jerk_p95_px"] < mp["jerk_p95_px"] else "↑差"),
        ("延迟均值(ms)", f"{mp['latency_mean_ms']:.1f}", f"{yo['latency_mean_ms']:.1f}", "↓好" if yo["latency_mean_ms"] < mp["latency_mean_ms"] else "↑差"),
        ("延迟P95(ms)", f"{mp['latency_p95_ms']:.1f}", f"{yo['latency_p95_ms']:.1f}", "↓好" if yo["latency_p95_ms"] < mp["latency_p95_ms"] else "↑差"),
        ("等效FPS", f"{mp['fps']:.1f}", f"{yo['fps']:.1f}", "↑好" if yo["fps"] > mp["fps"] else "↓差"),
    ]

    for label, mp_val, yo_val, flag in rows:
        print(f"  {label:<23} {str(mp_val):>20} {str(yo_val):>20}  {flag}")

    print("=" * 70)

    # 一句话结论
    detect_diff = (yo["detect_rate"] - mp["detect_rate"]) * 100
    latency_diff = yo["latency_p95_ms"] - mp["latency_p95_ms"]
    print(f"\n结论: HaGRID YOLO 检出率{'高' if detect_diff > 0 else '低'} {abs(detect_diff):.1f}%, "
          f"延迟P95 {'快' if latency_diff < 0 else '慢'} {abs(latency_diff):.1f}ms")


if __name__ == "__main__":
    main()
