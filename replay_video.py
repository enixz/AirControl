"""原始视频回放：用一段真实录制画面离线、客观地对比检测质量。

把 raw_capture/<时间戳>/ 里录制的真实相机帧逐帧喂回**真实的** find_hands，
读取检测结果与缩放状态，汇总成客观指标。任何检测/缩放/超分/参数改动都能拿
同一段画面跑，看数字升降，而不必每次真人摆姿势实测。

唯一测不了的是相机格式/采集帧率（MJPG vs YUY2）——那是实时采集属性，录像一旦
录下就定死了。但「压缩对识别的影响」这半可以用 --jpeg-quality 离线测（对录制的
无损帧做 JPEG 往返再喂检测）。

用法：
  python replay_video.py raw_capture/20260614_xxxx
  python replay_video.py <dir> --sr none              # 关超分看检出/帧率变化
  python replay_video.py <dir> --far-threshold 0.005  # 调 ZOOM 触发阈值
  python replay_video.py <dir> --jpeg-quality 70      # 测压缩对识别的影响
  python replay_video.py <dir> --compare --sr none    # 基线 vs 改动 并排对比

指标：
  detect_rate     检出率（有手帧占比）
  zoom_active     ZOOM 开启帧占比
  zoom_toggles    ZOOM 开/关翻转次数（越少越稳，拉风箱的客观量）
  viewport_jumps  视口大幅跳变次数（裁剪框相邻帧突变）
  lock_top_rate   多手帧里锁定「最高那只手」的比例（union→取最高手 改动的客观量）
  acquire         人脸引导重捕次数（越多=越在丢手重抓）
  ms_mean/p95     单帧处理耗时；fps_proc 本机处理帧率（相对值，用于 A/B）
"""
import argparse
import glob
import json
import logging
import os
import sys
import time
from importlib import import_module

import cv2

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


# ----------------------------------------------------------------- 指标聚合 --

def compute_metrics(records):
    """纯函数：把逐帧记录汇总成客观指标。不依赖摄像头/mediapipe，便于单测。

    records: list[dict]，每帧含
        hands:int  top_is_highest:bool|None  zoom_on:bool
        crop_center:(x,y)|None  crop_size:float|None  ms:float
    """
    n = len(records)
    if n == 0:
        return {}

    detect = sum(1 for r in records if r["hands"] > 0)
    zoom_active = sum(1 for r in records if r["zoom_on"])

    toggles = 0
    prev_on = None
    for r in records:
        if prev_on is not None and r["zoom_on"] != prev_on:
            toggles += 1
        prev_on = r["zoom_on"]

    jumps = 0
    jerk = 0.0
    pc = ps = None
    for r in records:
        c, s = r["crop_center"], r["crop_size"]
        if c is not None and s and pc is not None and ps:
            d = ((c[0] - pc[0]) ** 2 + (c[1] - pc[1]) ** 2) ** 0.5 + abs(s - ps)
            jerk += d
            if d > 0.15 * ps:          # 相邻帧裁剪框变化 >15% 记一次跳变
                jumps += 1
        if c is not None:
            pc, ps = c, s

    multi = [r for r in records if r["top_is_highest"] is not None]
    lock_top = sum(1 for r in multi if r["top_is_highest"])

    ms_sorted = sorted(r["ms"] for r in records)
    ms_mean = sum(ms_sorted) / n
    p95 = ms_sorted[min(n - 1, int(n * 0.95))]

    return {
        "frames": n,
        "detect_rate": detect / n,
        "zoom_active": zoom_active / n,
        "zoom_toggles": toggles,
        "viewport_jumps": jumps,
        "viewport_jerk": jerk / n,
        "multi_hand_frames": len(multi),
        "lock_top_rate": (lock_top / len(multi)) if multi else None,
        "ms_mean": ms_mean,
        "ms_p95": p95,
        "fps_proc": (1000.0 / ms_mean) if ms_mean else 0.0,
    }


# ----------------------------------------------------------------- 帧读取 --

def iter_frames(rec_dir):
    """按顺序产出 frame（BGR）。支持录制器输出的视频或 PNG 帧序列。"""
    video_path = next(
        (
            path
            for path in (
                os.path.join(rec_dir, "frames.mkv"),
                os.path.join(rec_dir, "frames.mp4"),
                os.path.join(rec_dir, "frames.avi"),
            )
            if os.path.exists(path)
        ),
        None,
    )
    if video_path is not None:
        cap = cv2.VideoCapture(video_path)
        try:
            while True:
                ok, fr = cap.read()
                if not ok or fr is None:
                    break
                yield fr
        finally:
            cap.release()
        return
    for p in sorted(glob.glob(os.path.join(rec_dir, "frames", "*.png"))):
        fr = cv2.imread(p)
        if fr is not None:
            yield fr


# --------------------------------------------------------------- zoom 计数 --

class _GestureEventCounter(logging.Handler):
    """挂到 'gesture' logger，按子串数 ZOOM ON/OFF/ACQUIRE/SR 切换。"""

    def __init__(self):
        super().__init__()
        # 注意：不能用 self.acquire，会覆盖 logging.Handler.acquire（线程锁方法）
        self.zoom_on = self.zoom_off = self.acquire_count = self.sr_switch = 0

    def emit(self, record):
        try:
            msg = record.getMessage()
        except Exception:
            return
        if "ZOOM ON" in msg:
            self.zoom_on += 1
        elif "ZOOM OFF" in msg:
            self.zoom_off += 1
        elif "ACQUIRE" in msg:
            self.acquire_count += 1
        elif "[SR]" in msg and "upscaler" in msg:
            self.sr_switch += 1


# ------------------------------------------------------------------- 回放 --

def _load_config(overrides):
    with open(os.path.join(PROJECT_ROOT, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg


def run(rec_dir, overrides=None, jpeg_quality=None):
    """跑一遍：建真 tracker、逐帧喂真 find_hands、读状态、返回 (指标, 事件计数)。"""
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "app"))
    create_hand_tracker = import_module(
        "services.hand_tracker_factory"
    ).create_hand_tracker

    cfg = _load_config(overrides or {})

    counter = _GestureEventCounter()
    glog = logging.getLogger("gesture")
    glog.setLevel(logging.INFO)
    glog.addHandler(counter)
    prev_propagate = glog.propagate
    glog.propagate = False  # 回放期间别把 zoom 日志刷到控制台

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

    records = []
    try:
        for frame in iter_frames(rec_dir):
            if jpeg_quality is not None:
                ok, buf = cv2.imencode(
                    ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)]
                )
                if ok:
                    frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)

            t0 = time.perf_counter()
            _, hands_landmarks, _ = tracker.find_hands(frame, draw=False)
            ms = (time.perf_counter() - t0) * 1000.0

            top_is_highest = None
            if len(hands_landmarks) >= 2:
                wrist_ys = [h[0][2] for h in hands_landmarks]
                top_is_highest = abs(hands_landmarks[0][0][2] - min(wrist_ys)) < 1e-6

            records.append({
                "hands": len(hands_landmarks),
                "top_is_highest": top_is_highest,
                "zoom_on": bool(getattr(tracker, "_crop_zoom_mode", False)),
                "crop_center": getattr(tracker, "_current_crop_center", None),
                "crop_size": getattr(tracker, "_current_crop_size", None),
                "ms": ms,
            })
    finally:
        close = getattr(tracker, "close", None)
        if callable(close):
            close()
        glog.removeHandler(counter)
        glog.propagate = prev_propagate

    return compute_metrics(records), counter


# -------------------------------------------------------------------- 打印 --

_FMT = {
    "frames": "{:.0f}", "detect_rate": "{:.1%}", "zoom_active": "{:.1%}",
    "zoom_toggles": "{:.0f}", "viewport_jumps": "{:.0f}", "viewport_jerk": "{:.2f}",
    "multi_hand_frames": "{:.0f}", "lock_top_rate": "{:.1%}",
    "ms_mean": "{:.1f}", "ms_p95": "{:.1f}", "fps_proc": "{:.1f}",
}


def _fmt(key, val):
    if val is None:
        return "n/a"
    return _FMT.get(key, "{}").format(val)


def _print_single(metrics, counter, label):
    print(f"\n=== {label} ===")
    for k in _FMT:
        print(f"  {k:<18} {_fmt(k, metrics.get(k))}")
    print(f"  {'acquire':<18} {counter.acquire_count}")
    print(f"  {'sr_switch':<18} {counter.sr_switch}")


def _print_compare(base, base_c, var, var_c, var_label):
    keys = list(_FMT) + ["acquire", "sr_switch"]
    base = dict(base)
    base["acquire"] = base_c.acquire_count
    base["sr_switch"] = base_c.sr_switch
    var = dict(var)
    var["acquire"] = var_c.acquire_count
    var["sr_switch"] = var_c.sr_switch
    print(f"\n{'metric':<18} {'baseline':>12} {var_label:>20} {'delta':>12}")
    print("-" * 66)
    for k in keys:
        b, v = base.get(k), var.get(k)
        delta = ""
        if isinstance(b, (int, float)) and isinstance(v, (int, float)):
            delta = _fmt(k, v - b) if k not in ("detect_rate", "zoom_active", "lock_top_rate") \
                else f"{(v - b):+.1%}"
        print(f"{k:<18} {_fmt(k, b):>12} {_fmt(k, v):>20} {delta:>12}")


def main():
    ap = argparse.ArgumentParser(description="原始视频回放 — 离线客观对比检测质量")
    ap.add_argument("rec_dir", help="录制目录 raw_capture/<时间戳>")
    ap.add_argument("--sr", dest="zoom_sr_engine", help="超分引擎 auto/espcn/none")
    ap.add_argument("--far-threshold", dest="zoom_far_threshold", type=float)
    ap.add_argument("--near-threshold", dest="zoom_near_threshold", type=float)
    ap.add_argument("--engine", dest="detection_engine")
    ap.add_argument("--jpeg-quality", type=int, help="先对每帧做 JPEG 往返再检测（测压缩影响）")
    ap.add_argument("--compare", action="store_true",
                    help="基线(config.json) vs 应用上述覆盖项，并排对比")
    ap.add_argument("--label", default="run", help="单跑模式的标签")
    args = ap.parse_args()

    if not os.path.isdir(args.rec_dir):
        ap.error(f"录制目录不存在: {args.rec_dir}")

    overrides = {
        k: getattr(args, k) for k in
        ("zoom_sr_engine", "zoom_far_threshold", "zoom_near_threshold", "detection_engine")
    }
    overrides = {k: v for k, v in overrides.items() if v is not None}

    logging.basicConfig(level=logging.WARNING)

    if args.compare:
        base_m, base_c = run(args.rec_dir, overrides=None, jpeg_quality=None)
        var_m, var_c = run(args.rec_dir, overrides=overrides, jpeg_quality=args.jpeg_quality)
        label = "+".join(f"{k}={v}" for k, v in overrides.items()) or "variant"
        if args.jpeg_quality is not None:
            label += f"+jpeg{args.jpeg_quality}"
        _print_compare(base_m, base_c, var_m, var_c, label[:20])
    else:
        m, c = run(args.rec_dir, overrides=overrides, jpeg_quality=args.jpeg_quality)
        _print_single(m, c, args.label)


if __name__ == "__main__":
    main()
