"""B0 当前体验基线回放（方案 §6.1）：完整生产链 + 每帧实际引擎记录。

B0 要求：固定当前有效配置和完整生产链（自动切换、平滑、预测、模式逻辑），
记录每帧实际引擎，不得把切换后的输出全部叫 MediaPipe。

用法：
  python run_b0_baseline.py <rec_dir> --run-id p2_b0_baseline
"""
import argparse
import json
import logging
import os
import sys
import time

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "adapters"))


def load_adapter(route_id):
    registry = {
        "MP_FULL": ("mediapipe_full_adapter", "MediaPipeFullAdapter"),
    }
    if route_id not in registry:
        raise SystemExit(f"未知 route_id: {route_id}")
    mod_name, cls_name = registry[route_id]
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        mod_name, os.path.join(os.path.dirname(HERE), "adapters", f"{mod_name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, cls_name)


def main():
    ap = argparse.ArgumentParser(description="B0 当前体验基线回放")
    ap.add_argument("rec_dir")
    ap.add_argument("--route", default="MP_FULL")
    ap.add_argument("--run-id", default="p2_b0_baseline")
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args()

    clip_id = os.path.basename(os.path.normpath(args.rec_dir))
    out_dir = os.path.join(ROOT, "artifacts", "hand_feasibility", args.run_id)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{clip_id}_b0.jsonl")

    cfg = {}
    config_path = os.path.join(ROOT, "config.json")
    if os.path.exists(config_path):
        with open(config_path, encoding="utf-8") as f:
            cfg = json.load(f)

    # 读逐帧时间戳
    meta = os.path.join(args.rec_dir, "meta.jsonl")
    times = []
    if os.path.exists(meta):
        with open(meta, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    times.append(json.loads(line).get("t"))

    adapter_cls = load_adapter(args.route)
    adapter = adapter_cls(config=cfg)

    # 引擎切换记录（B0 必须记录每帧实际引擎；auto_switch 会换 hagrid_yolo）
    engine_log = logging.getLogger("gesture")
    engine_events = []

    class _EngineHandler(logging.Handler):
        def emit(self, record):
            try:
                msg = record.getMessage()
            except Exception:
                return
            if "引擎" in msg or "engine" in msg.lower() or "ZOOM" in msg:
                engine_events.append(msg)

    handler = _EngineHandler()
    engine_log.addHandler(handler)
    prev_level = engine_log.level
    engine_log.setLevel(logging.INFO)

    n = 0
    t_start = time.perf_counter()
    with open(out_path, "w", encoding="utf-8") as out:
        import importlib
        cap = cv2.VideoCapture(next(
            (os.path.join(args.rec_dir, n_) for n_ in ("frames.mkv", "frames.mp4", "frames.avi")
             if os.path.exists(os.path.join(args.rec_dir, n_))), None))
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.max_frames is not None and n >= args.max_frames:
                break
            ts = times[n] if n < len(times) else n / 25.0
            result = adapter.infer(frame, ts)
            result["timestamp_is_synthetic"] = n >= len(times)
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            n += 1
            if (n) % 200 == 0:
                print(f"  {n} 帧...")
        cap.release()
    adapter.close()
    elapsed = time.perf_counter() - t_start
    engine_log.removeHandler(handler)
    engine_log.setLevel(prev_level)

    print(f"B0 完成: {n} 帧, 耗时 {elapsed:.1f}s, 输出 {out_path}")
    if engine_events:
        print(f"引擎/缩放事件 {len(engine_events)} 条（前 5 条）:")
        for e in engine_events[:5]:
            print(f"  {e}")


if __name__ == "__main__":
    main()
