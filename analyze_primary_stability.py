"""主手稳定性分析 — 测量 replay_video.py 没覆盖的"识别点拉扯"指标。

针对用户报告"双手同时出现时识别点拉扯跳动，飞在两手中间"：
  - primary_switches: 主手切换次数
  - wrist_jerk: 识别点(Primary wrist) 相邻帧位移均值
  - wrist_jumps: 识别点相邻帧位移 > 阈值的次数（跳变）
  - mid_air_frames: 识别点落在两手中间的帧数（距两手 wrist 都远）
  - switch_jerk: 主手切换前后 3 帧的识别点位移（切换瞬间跳变幅度）

两种数据源：
  --from-meta（推荐）: 直接读录制时写入 meta.jsonl 的 primary_wrist/wrists，
        反映**原始运行时**的识别点行为（含 smoother 状态、主手切换过程）。
        需要新版 inference_worker 录制（2026-07-05 起）。
  默认（重新检测）: 用当前代码对录制帧重新跑 find_hands，反映**当前代码**在原始
        帧上的表现，不是原始运行时行为。仅用于旧录像或代码 A/B 对比。

用法：
  python analyze_primary_stability.py raw_capture/20260705_095523 --from-meta
  python analyze_primary_stability.py raw_capture/20260705_095523
  python analyze_primary_stability.py raw_capture/20260705_095523 --render-overlay
"""
import argparse
import json
import logging
import os
import sys

import cv2

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


# ----------------------------------------------------------- meta 模式 ---

def run_from_meta(rec_dir, render_overlay=False, overlay_out=None):
    """读 meta.jsonl 重建原始运行时识别点轨迹并算指标。

    meta.jsonl 每行由 FrameRecorder 写入，含 i/t/w/h 基础字段 + 可选的
    primary_wrist/wrists/primary_switched/hands/zoom_on/crop_center/crop_size。
    """
    meta_path = os.path.join(rec_dir, "meta.jsonl")
    if not os.path.exists(meta_path):
        print(f"[错误] 未找到 {meta_path}，无法用 --from-meta 模式")
        print("提示：该录像可能是旧版 inference_worker 录制的，没有 primary_wrist 字段。")
        print("      去掉 --from-meta 用重新检测模式，或用新版代码重新录制。")
        return None

    rows = []
    with open(meta_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not rows:
        print("[错误] meta.jsonl 为空")
        return None

    # 检查是否有 primary_wrist 字段（新版录制）
    has_meta = any("primary_wrist" in r for r in rows)
    if not has_meta:
        print("[错误] meta.jsonl 不含 primary_wrist 字段（旧版录制）")
        print("      去掉 --from-meta 用重新检测模式，或用新版代码重新录制。")
        return None

    w = rows[0].get("w", 1920)
    h = rows[0].get("h", 1080)

    # 1. primary_switches: primary_switched=True 的帧数
    primary_switches = sum(1 for r in rows if r.get("primary_switched"))

    # 2. wrist_jerk / wrist_jumps: 相邻帧 primary_wrist 位移
    jerks = []
    jumps = 0
    JUMP_THRESHOLD = max(50.0, 0.05 * max(w, h))
    prev_pw = None
    for r in rows:
        pw = r.get("primary_wrist")
        if pw is None:
            prev_pw = None
            continue
        if prev_pw is not None:
            dist = ((pw[0] - prev_pw[0]) ** 2 + (pw[1] - prev_pw[1]) ** 2) ** 0.5
            jerks.append(dist)
            if dist > JUMP_THRESHOLD:
                jumps += 1
        prev_pw = pw

    # 3. mid_air_frames: primary_wrist 离所有手 wrist 都 > 100px 且在 x 范围内
    mid_air = 0
    mid_air_frames = []
    multi_frames = 0
    for idx, r in enumerate(rows):
        wrists = r.get("wrists")
        pw = r.get("primary_wrist")
        if not wrists or len(wrists) < 2 or pw is None:
            continue
        multi_frames += 1
        dists = [
            ((pw[0] - wx) ** 2 + (pw[1] - wy) ** 2) ** 0.5
            for wx, wy in wrists
        ]
        min_dist = min(dists)
        if min_dist > 100.0:
            xs = [wx for wx, _ in wrists]
            if min(xs) <= pw[0] <= max(xs):
                mid_air += 1
                mid_air_frames.append(idx)

    # 4. switch_jerk: 主手切换前后 3 帧的识别点位移
    switch_jerks = []
    for idx, r in enumerate(rows):
        if not r.get("primary_switched"):
            continue
        # 取 idx-3 到 idx+3 的 primary_wrist 位移
        window = []
        for j in range(max(0, idx - 3), min(len(rows), idx + 4)):
            pw_j = rows[j].get("primary_wrist")
            if pw_j is not None:
                window.append(pw_j)
        if len(window) >= 2:
            total = sum(
                ((window[i+1][0] - window[i][0]) ** 2 +
                 (window[i+1][1] - window[i][1]) ** 2) ** 0.5
                for i in range(len(window) - 1)
            )
            switch_jerks.append(total)

    jerk_mean = sum(jerks) / len(jerks) if jerks else 0.0
    jerk_p95 = sorted(jerks)[min(len(jerks) - 1, int(len(jerks) * 0.95))] if jerks else 0.0
    switch_jerk_mean = sum(switch_jerks) / len(switch_jerks) if switch_jerks else 0.0

    print(f"\n=== 主手稳定性分析 [meta 模式] ({rec_dir}) ===")
    print("  数据源              meta.jsonl（原始运行时识别点轨迹）")
    print(f"  frames              {len(rows)}")
    print(f"  multi_hand_frames   {multi_frames} ({multi_frames/len(rows):.1%})")
    print(f"  primary_switches    {primary_switches}")
    print(f"  wrist_jerk_mean     {jerk_mean:.1f} px")
    print(f"  wrist_jerk_p95      {jerk_p95:.1f} px")
    print(f"  wrist_jumps(>{JUMP_THRESHOLD:.0f}px)  {jumps}")
    print(f"  mid_air_frames      {mid_air}  (识别点飞在两手中间)")
    if mid_air_frames:
        print(f"  mid_air_at          {mid_air_frames[:20]}{'...' if len(mid_air_frames)>20 else ''}")
    print(f"  switch_jerk_mean    {switch_jerk_mean:.1f} px (切换前后3帧总位移)")

    if render_overlay:
        _render_overlay_video(rec_dir, rows, overlay_out, w, h)

    return primary_switches, jerk_mean, jumps, mid_air


def _render_overlay_video(rec_dir, rows, overlay_out, w, h):
    """基于 meta.jsonl 把原始识别点轨迹叠加到原始帧视频上，输出可看的 mp4。"""
    if overlay_out is None:
        overlay_out = os.path.join(rec_dir, "overlay_with_meta.mp4")
    mkv = os.path.join(rec_dir, "frames.mkv")
    mp4 = os.path.join(rec_dir, "frames.mp4")
    src = mkv if os.path.exists(mkv) else mp4
    if not os.path.exists(src):
        print(f"[警告] 未找到原始帧视频 {src}，跳过 overlay 渲染")
        return

    cap = cv2.VideoCapture(src)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(overlay_out, fourcc, 30.0, (int(w), int(h)))
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if frame_idx < len(rows):
            r = rows[frame_idx]
            pw = r.get("primary_wrist")
            wrists = r.get("wrists") or []
            switched = r.get("primary_switched")
            # 画所有手 wrist（黄色小圈）
            for wx, wy in wrists:
                cv2.circle(frame, (int(wx), int(wy)), 8, (0, 255, 255), 2)
            # 画 Primary wrist（红色大圈，识别点）
            if pw is not None:
                color = (0, 0, 255) if not switched else (0, 0, 200)
                cv2.circle(frame, (int(pw[0]), int(pw[1])), 14, color, -1)
                cv2.circle(frame, (int(pw[0]), int(pw[1])), 18, color, 2)
            # 切换帧标注
            if switched:
                cv2.putText(frame, "SWITCH", (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 200), 2)
            # 帧号 + 手数
            cv2.putText(frame, f"#{frame_idx} hands={len(wrists)}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        writer.write(frame)
        frame_idx += 1
    cap.release()
    writer.release()
    print(f"  overlay_video        {overlay_out}")


# ------------------------------------------------------- 重新检测模式 ---

class PrimarySwitchCounter(logging.Handler):
    """计 [PRIMARY] 主手切换日志次数。"""
    def __init__(self):
        super().__init__()
        self.count = 0
        self.frame_indices = []

    def emit(self, record):
        try:
            msg = record.getMessage()
        except Exception:
            return
        if "[PRIMARY]" in msg and "主手切换" in msg:
            self.count += 1


def run_redetect(rec_dir):
    """用当前代码对录制帧重新跑 find_hands，算指标。

    注意：这只反映**当前代码**在原始帧上的表现，不是原始运行时行为。
    原始运行时行为请用 run_from_meta（--from-meta）。
    """
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "app"))
    from services.hand_tracker_factory import create_hand_tracker

    with open(os.path.join(PROJECT_ROOT, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)

    counter = PrimarySwitchCounter()
    glog = logging.getLogger("gesture")
    glog.setLevel(logging.INFO)
    glog.addHandler(counter)
    glog.propagate = False

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

    mkv = os.path.join(rec_dir, "frames.mkv")
    mp4 = os.path.join(rec_dir, "frames.mp4")
    src = mkv if os.path.exists(mkv) else mp4
    if not os.path.exists(src):
        print(f"[错误] 未找到原始帧视频 {mkv} 或 {mp4}")
        return None
    cap = cv2.VideoCapture(src)

    primary_wrist_history = []
    all_wrists_history = []
    frame_idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            _, hands_landmarks, _ = tracker.find_hands(frame, draw=False)

            if hands_landmarks:
                px = float(hands_landmarks[0][0][1])
                py = float(hands_landmarks[0][0][2])
                primary_wrist_history.append((frame_idx, (px, py)))
                all_wrists = [
                    (float(h[0][1]), float(h[0][2])) for h in hands_landmarks
                ]
                all_wrists_history.append((frame_idx, all_wrists))
            else:
                primary_wrist_history.append((frame_idx, None))
                all_wrists_history.append((frame_idx, []))
            frame_idx += 1
    finally:
        cap.release()
        glog.removeHandler(counter)
        glog.propagate = True

    h, w = 1080, 1920
    cap2 = cv2.VideoCapture(src)
    ok, first = cap2.read()
    if ok and first is not None:
        h, w = first.shape[:2]
    cap2.release()

    jerks = []
    jumps = 0
    JUMP_THRESHOLD = max(50.0, 0.05 * max(w, h))
    prev = None
    for _idx, pw in primary_wrist_history:
        if pw is None:
            prev = None
            continue
        if prev is not None:
            dist = ((pw[0] - prev[0]) ** 2 + (pw[1] - prev[1]) ** 2) ** 0.5
            jerks.append(dist)
            if dist > JUMP_THRESHOLD:
                jumps += 1
        prev = pw

    mid_air = 0
    mid_air_frames = []
    for idx, (pw, all_wrists) in enumerate(
        zip(primary_wrist_history, all_wrists_history, strict=True)
    ):
        if pw is None or len(all_wrists[1]) < 2:
            continue
        _, primary = pw
        wrists = all_wrists[1]
        dists = [
            ((primary[0] - wx) ** 2 + (primary[1] - wy) ** 2) ** 0.5
            for wx, wy in wrists
        ]
        min_dist = min(dists)
        if min_dist > 100.0:
            xs = [wx for wx, _ in wrists]
            if min(xs) <= primary[0] <= max(xs):
                mid_air += 1
                mid_air_frames.append(idx)

    jerk_mean = sum(jerks) / len(jerks) if jerks else 0.0
    jerk_p95 = sorted(jerks)[min(len(jerks) - 1, int(len(jerks) * 0.95))] if jerks else 0.0

    multi_frames = sum(1 for _, ws in all_wrists_history if len(ws) >= 2)

    print(f"\n=== 主手稳定性分析 [重新检测模式] ({rec_dir}) ===")
    print("  数据源              当前代码重新跑 find_hands（非原始运行时行为）")
    print(f"  frames              {frame_idx}")
    print(f"  multi_hand_frames   {multi_frames} ({multi_frames/frame_idx:.1%})")
    print(f"  primary_switches    {counter.count}")
    print(f"  wrist_jerk_mean     {jerk_mean:.1f} px")
    print(f"  wrist_jerk_p95      {jerk_p95:.1f} px")
    print(f"  wrist_jumps(>{JUMP_THRESHOLD:.0f}px)  {jumps}")
    print(f"  mid_air_frames      {mid_air}  (识别点飞在两手中间)")
    if mid_air_frames:
        print(f"  mid_air_at          {mid_air_frames[:20]}{'...' if len(mid_air_frames)>20 else ''}")
    print(f"  smoother_keys       {tracker.HAND_KEYS}")
    return counter.count, jerk_mean, jumps, mid_air


def main():
    ap = argparse.ArgumentParser(description="主手稳定性分析 — 测量识别点拉扯指标")
    ap.add_argument("rec_dir", help="录制目录 raw_capture/<时间戳>")
    ap.add_argument("--from-meta", action="store_true",
                    help="读 meta.jsonl 重建原始运行时识别点轨迹（推荐）")
    ap.add_argument("--render-overlay", action="store_true",
                    help="（仅 --from-meta）把识别点轨迹叠加到原始帧视频，输出 mp4")
    ap.add_argument("--overlay-out", help="overlay 视频输出路径（默认 <rec_dir>/overlay_with_meta.mp4）")
    args = ap.parse_args()

    if not os.path.isdir(args.rec_dir):
        ap.error(f"录制目录不存在: {args.rec_dir}")

    if args.from_meta:
        run_from_meta(args.rec_dir, render_overlay=args.render_overlay,
                      overlay_out=args.overlay_out)
    else:
        run_redetect(args.rec_dir)


if __name__ == "__main__":
    main()
