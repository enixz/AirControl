"""统一运行器（T0）：把适配器输出落盘为 predictions/<route>/<clip>.jsonl。

用法：
  python run_adapter.py <rec_dir> --route MP_FULL --out <run_dir>

rec_dir: D:/AC-trae/raw_capture/<时间戳>
run_dir: artifacts/hand_feasibility/<run_id>/
"""
import argparse
import importlib.util
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
ADAPTERS_DIR = os.path.join(os.path.dirname(HERE), "adapters")
sys.path.insert(0, HERE)
sys.path.insert(0, ADAPTERS_DIR)


def load_adapter(route_id):
    """按 route_id 加载适配器类（简单注册表，避免包依赖）。"""
    registry = {
        "MP_FULL": ("mediapipe_full_adapter", "MediaPipeFullAdapter"),
        "R1B_DISTILL": ("rtmpose_distill_adapter", "RTMPoseDistillAdapter"),
        "R3_YOLO26": ("yolo26_hand_adapter", "Yolo26HandAdapter"),
    }
    if route_id not in registry:
        raise SystemExit(f"未知 route_id: {route_id}；已注册: {list(registry)}")
    mod_name, cls_name = registry[route_id]
    spec = importlib.util.spec_from_file_location(
        mod_name, os.path.join(ADAPTERS_DIR, f"{mod_name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, cls_name)


def iter_frames(rec_dir):
    import cv2
    for name in ("frames.mkv", "frames.mp4", "frames.avi"):
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
    raise FileNotFoundError(f"未找到视频 in {rec_dir}")


def load_frame_times(rec_dir):
    """meta.jsonl 每帧时间戳；缺失时 None（延迟指标标估计）。"""
    meta = os.path.join(rec_dir, "meta.jsonl")
    if not os.path.exists(meta):
        return None
    times = []
    with open(meta, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                if "t" in row:
                    times.append(float(row["t"]))
    return times or None


def main():
    ap = argparse.ArgumentParser(description="可行性论证统一适配器运行器")
    ap.add_argument("rec_dir")
    ap.add_argument("--route", default="MP_FULL")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()

    clip_id = os.path.basename(os.path.normpath(args.rec_dir))
    run_id = args.run_id or f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir = os.path.join(ROOT, "artifacts", "hand_feasibility", run_id)
    os.makedirs(os.path.join(out_dir, "predictions", args.route), exist_ok=True)
    out_path = os.path.join(out_dir, "predictions", args.route, f"{clip_id}.jsonl")

    cfg = {}
    config_path = os.path.join(ROOT, "config.json")
    if os.path.exists(config_path):
        with open(config_path, encoding="utf-8") as f:
            cfg = json.load(f)
    cfg["run_id"] = run_id
    cfg["clip_id"] = clip_id

    times = load_frame_times(args.rec_dir)

    adapter_cls = load_adapter(args.route)
    adapter = adapter_cls(config=cfg)

    n_frames = 0
    t_start = time.perf_counter()
    with open(out_path, "w", encoding="utf-8") as out:
        for i, frame in enumerate(iter_frames(args.rec_dir)):
            if args.max_frames is not None and i >= args.max_frames:
                break
            ts = times[i] if times and i < len(times) else None
            if ts is None:
                # 缺 meta 时退回帧序号/25fps 合成并注明，不冒充真实采集时间
                ts = i / 25.0
            result = adapter.infer(frame, ts)
            result["timestamp_is_synthetic"] = times is None or i >= len(times)
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            n_frames += 1
            if (i + 1) % 200 == 0:
                print(f"  {i+1} 帧...")
    adapter.close()
    elapsed = time.perf_counter() - t_start

    print(f"完成: {n_frames} 帧, 耗时 {elapsed:.1f}s, 输出 {out_path}")


if __name__ == "__main__":
    main()
