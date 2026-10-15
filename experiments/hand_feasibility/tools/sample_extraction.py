"""P2 最小筛选样本抽取（方案 §4 阶段表：每路线 50 张样本 + 一段 60 秒视频）。

从 raw_capture 会话中按场景抽帧：
  - 困难集：pose_yaw045/090/135_near（近距侧手，MediaPipe 已知弱项）
  - 常规集：20260722_145659（自然点击会话）
  - 远距集：pose_yaw*_far

输出：
  artifacts/hand_feasibility/p2_screening/sample_frames/<clip>/frame_XXXXX.png
  artifacts/hand_feasibility/p2_screening/sample_manifest.json（帧清单+来源+时间戳）
"""
import argparse
import json
import os

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
RAW = r"D:\AC-trae\raw_capture"

# 每场景抽帧数（合计 50）
DEFAULT_PLAN = [
    ("pose_yaw045_near", 10, "difficulty"),
    ("pose_yaw090_near", 10, "difficulty"),
    ("pose_yaw135_near", 10, "difficulty"),
    ("pose_yaw180_near", 10, "difficulty"),
    ("20260722_145659", 10, "natural"),
]


def load_times(rec_dir):
    meta = os.path.join(rec_dir, "meta.jsonl")
    times = []
    with open(meta, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                times.append(json.loads(line).get("t"))
    return times


def extract(rec_dir, clip_id, n_samples, tier, out_root, manifest):
    video = next(
        (os.path.join(rec_dir, n) for n in ("frames.mkv", "frames.mp4", "frames.avi")
         if os.path.exists(os.path.join(rec_dir, n))), None)
    if video is None:
        print(f"  [SKIP] {clip_id}: 无视频")
        return
    times = load_times(rec_dir)
    cap = cv2.VideoCapture(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        print(f"  [SKIP] {clip_id}: 帧数未知")
        return
    # 均匀抽样（避开首尾 5%）
    margin = max(1, int(total * 0.05))
    step = max(1, (total - 2 * margin) // max(n_samples, 1))
    indices = list(range(margin, total - margin, step))[:n_samples]

    out_dir = os.path.join(out_root, clip_id)
    os.makedirs(out_dir, exist_ok=True)
    idx_set = set(indices)
    saved = 0
    fi = 0
    while saved < len(indices):
        ok, frame = cap.read()
        if not ok:
            break
        if fi in idx_set:
            name = f"frame_{fi:05d}.png"
            cv2.imwrite(os.path.join(out_dir, name), frame)
            manifest.append({
                "clip_id": clip_id, "tier": tier, "frame_index": fi,
                "source_timestamp": times[fi] if fi < len(times) else None,
                "file": os.path.join(out_dir, name),
            })
            saved += 1
        fi += 1
    cap.release()
    print(f"  {clip_id}: 抽 {saved}/{total} 帧")


def main():
    ap = argparse.ArgumentParser(description="P2 样本抽取")
    ap.add_argument("--out", default=os.path.join(
        ROOT, "artifacts", "hand_feasibility", "p2_screening"))
    ap.add_argument("--per-clip", type=int, default=10)
    args = ap.parse_args()

    out_root = os.path.join(args.out, "sample_frames")
    os.makedirs(out_root, exist_ok=True)
    manifest = []
    for clip_id, n, tier in DEFAULT_PLAN:
        extract(os.path.join(RAW, clip_id), clip_id, n, tier, out_root, manifest)

    with open(os.path.join(args.out, "sample_manifest.json"), "w", encoding="utf-8") as f:
        json.dump({
            "schema_version": 1,
            "created_by": "tools/sample_extraction.py",
            "source_root": RAW,
            "n_samples": len(manifest),
            "samples": manifest,
        }, f, ensure_ascii=False, indent=2)
    print(f"清单: {os.path.join(args.out, 'sample_manifest.json')}（{len(manifest)} 帧）")


if __name__ == "__main__":
    main()
