"""离线评测:超分对 ZOOM crop-zoom 检测到底有没有用(P0,零风险)。

方法:
  Pass 0 — 全帧检测(MediaPipe IMAGE 模式)收集"小手"帧:crop = bbox×2.5
           (生产视口公式),仅保留 crop < 345px(SR 会介入)的帧。
  Pass 1 — 对同一批 crop 分别用 none(双线性)/ESPCN/Real-ESRGAN 放大到
           384px 后跑检测,比较检出率、handedness 分数、关键点稳定性、耗时。

只读录像与模型;输出只写 artifacts/sr_effectiveness_p0/,不改生产配置。
"""

import json
import os
import sys
import time

import cv2
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "app"))
sys.path.insert(0, REPO)

VIDEOS = [
    r"D:/AIRvideo/演示模式.mp4",
    r"D:/AIRvideo/鼠标模式.mp4",
    r"D:/AIRvideo/板书模式.mp4",
]
OUT_DIR = os.path.join(REPO, "artifacts", "sr_effectiveness_p0")

CROP_PADDING_RATIO = 2.5   # 生产 _crop_padding_ratio
CROP_MIN_SIZE = 32         # 生产 _crop_min_size
CROP_TARGET_SIZE = 384     # 生产 _crop_target_size
SR_ENGAGE_MAX = int(CROP_TARGET_SIZE * 0.9)  # auto 滞回下沿,crop<345 才吃 SR
MAX_EVAL_PER_VIDEO = 150   # 每视频最多评测帧数(均匀抽样),控制墙钟
TIERS = ("none", "espcn", "realesrgan")


def build_viewports(video, conf):
    """全帧检测 → 收集 (帧号, 中心, crop尺寸)。"""
    from services.hand_tracker import HandTracker

    from app.config_manager import ConfigManager

    tracker = HandTracker(
        static_image_mode=True,
        min_detection_confidence=conf,
        min_tracking_confidence=conf,
        config=ConfigManager(),
    )
    cap = cv2.VideoCapture(video)
    viewports = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        h, w = frame.shape[:2]
        res = tracker._run_recognizer(frame)
        if not (res and res.hand_landmarks):
            continue
        lm = res.hand_landmarks[0]
        xs = [p.x * w for p in lm]
        ys = [p.y * h for p in lm]
        size = max(max(xs) - min(xs), max(ys) - min(ys))
        crop = int(size * CROP_PADDING_RATIO)
        crop = max(crop, CROP_MIN_SIZE)
        crop = min(crop, min(w, h))
        if crop >= SR_ENGAGE_MAX:
            continue  # 生产 auto 档此时为 none,不评
        cx = (min(xs) + max(xs)) / 2.0
        cy = (min(ys) + max(ys)) / 2.0
        viewports.append({"frame": idx, "cx": cx, "cy": cy, "size": crop})
    cap.release()
    return viewports


def crop_region(frame, cx, cy, size):
    h, w = frame.shape[:2]
    size = min(size, min(w, h))
    x0 = max(0, min(int(round(cx - size / 2)), w - size))
    y0 = max(0, min(int(round(cy - size / 2)), h - size))
    return frame[y0:y0 + size, x0:x0 + size].copy()


def upscale(tier, crop, sr):
    if tier == "none":
        return cv2.resize(crop, (CROP_TARGET_SIZE, CROP_TARGET_SIZE),
                          interpolation=cv2.INTER_LINEAR)
    if tier == "espcn":
        out = sr.espcn(crop, CROP_TARGET_SIZE)
    else:
        out = sr.realesrgan(crop, CROP_TARGET_SIZE, prefer_gpu=True)
    if out is None:
        return cv2.resize(crop, (CROP_TARGET_SIZE, CROP_TARGET_SIZE),
                          interpolation=cv2.INTER_LINEAR)
    return out


def eval_tier(tier, viewports, video, conf):
    from services.hand_tracker import HandTracker
    from services.sr_engine import SREngine

    from app.config_manager import ConfigManager

    tracker = HandTracker(
        static_image_mode=True,
        min_detection_confidence=conf,
        min_tracking_confidence=conf,
        config=ConfigManager(),
    )
    sr = SREngine()
    sr.init()
    cap = cv2.VideoCapture(video)

    wanted = {v["frame"]: v for v in viewports}
    n = detect = 0
    scores = []
    jitters = []
    ms_list = []
    prev = None  # (frame_idx, kp Nx2)

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        vp = wanted.get(idx)
        if vp is None:
            continue
        n += 1
        crop = crop_region(frame, vp["cx"], vp["cy"], vp["size"])

        t0 = time.perf_counter()
        zoomed = upscale(tier, crop, sr)
        ms_list.append((time.perf_counter() - t0) * 1000.0)

        res = tracker._run_recognizer(zoomed)
        if res and res.hand_landmarks:
            detect += 1
            try:
                scores.append(float(res.handedness[0][0].score))
            except (IndexError, AttributeError, TypeError):
                pass
            kp = np.array([[p.x, p.y] for p in res.hand_landmarks[0]])
            if prev is not None and idx - prev[0] <= 5 and prev[1].shape == kp.shape:
                d = (kp - kp[0]) - (prev[1] - prev[1][0])  # 腕对齐去平移
                jitters.append(float(np.mean(np.linalg.norm(d, axis=1))))
            prev = (idx, kp)
        else:
            prev = None

    cap.release()
    return {
        "tier": tier,
        "frames": n,
        "detections": detect,
        "detect_rate": round(detect / n, 4) if n else None,
        "mean_score": round(float(np.mean(scores)), 4) if scores else None,
        "mean_jitter": round(float(np.mean(jitters)), 5) if jitters else None,
        "mean_ms": round(float(np.mean(ms_list)), 1) if ms_list else None,
        "p50_ms": round(float(np.percentile(ms_list, 50)), 1) if ms_list else None,
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    all_results = []

    for video in VIDEOS:
        name = os.path.basename(video)
        print(f"=== {name} ===", flush=True)

        print("[pass0] conf=0.6 收集小手帧 ...", flush=True)
        viewports = build_viewports(video, 0.6)
        used_conf = 0.6
        if len(viewports) < 5:
            print("[pass0] 检出不足,降 0.35 重试 ...", flush=True)
            viewports = build_viewports(video, 0.35)
            used_conf = 0.35

        if not viewports:
            print("[pass0] 仍无小手帧,跳过该视频", flush=True)
            continue

        sizes = [v["size"] for v in viewports]
        print(f"[pass0] conf={used_conf}: {len(viewports)} 帧, "
              f"crop min={min(sizes)} med={int(np.median(sizes))} max={max(sizes)}", flush=True)

        if len(viewports) > MAX_EVAL_PER_VIDEO:
            step = len(viewports) / float(MAX_EVAL_PER_VIDEO)
            viewports = [viewports[int(i * step)] for i in range(MAX_EVAL_PER_VIDEO)]
            print(f"[pass0] 抽样至 {len(viewports)} 帧", flush=True)

        for tier in TIERS:
            r = eval_tier(tier, viewports, video, used_conf)
            r["video"] = name
            r["conf"] = used_conf
            all_results.append(r)
            print(f"  {tier:>10}: 检出 {r['detections']}/{r['frames']}"
                  f" ({(r['detect_rate'] or 0)*100:.1f}%)"
                  f" score={r['mean_score']} jitter={r['mean_jitter']}"
                  f" ms={r['mean_ms']}(p50 {r['p50_ms']})", flush=True)

    with open(os.path.join(OUT_DIR, "results.json"), "w", encoding="utf-8") as f:
        json.dump({"videos": VIDEOS, "crop_target": CROP_TARGET_SIZE,
                   "results": all_results}, f, ensure_ascii=False, indent=2)
    print(f"[done] {os.path.join(OUT_DIR, 'results.json')}", flush=True)


if __name__ == "__main__":
    main()
