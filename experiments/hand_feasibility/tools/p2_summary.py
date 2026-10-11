"""P2 筛选汇总：多路线 × 多目录的四维对照。

四维（方案 §10.1 P2 门槛）：
  1. 坐标正确性（同帧手腕/食指尖中位距离，vs MP_FULL 参照）
  2. 检出率（真实检出，R3 无 GT 仅为输出覆盖）
  3. 置信度分布（R1B/R3 有逐点分数）
  4. 速度（单帧模型耗时中位数）

输出 artifacts/hand_feasibility/p2_screening/summary.json
"""
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))

CLIPS = [
    "pose_yaw045_near", "pose_yaw090_near", "pose_yaw135_near",
    "pose_yaw180_near", "20260722_145659",
]
ROUTES = ["MP_FULL", "R1B_DISTILL", "R3_YOLO26"]
RUN_ID = "p2_screening_all"
REF = "MP_FULL"


def load(run_id, route, clip):
    p = os.path.join(ROOT, "artifacts", "hand_feasibility", run_id,
                     "predictions", route, f"{clip}.jsonl")
    if not os.path.exists(p):
        return None
    rows = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def summarize(rows):
    n = len(rows)
    with_hands = sum(1 for r in rows if r["hands"])
    ms_key = next((k for k in ("model_inference_ms", "simcc_inference_ms") if rows and k in rows[0]), None)
    ms = [r[ms_key] for r in rows if ms_key and r.get(ms_key) is not None] if ms_key else []
    return {
        "frames": n,
        "frames_with_hands": with_hands,
        "output_coverage": with_hands / n if n else None,
        "model_ms_median": statistics.median(ms) if ms else None,
    }


def cross_compare(rows_a, rows_b):
    """最近邻一对一匹配后的手腕(0)/食指尖(8)中位距离（有手帧）。

    不能按输出顺序配对：不同路线的双手检出差异大（实测远距录像 MP 双手帧
    1271/2000 vs R1B 300/2000），同序号配对会比较不同的手（实测同序号
    371px vs 最近邻 11.2px）。
    """
    wrists, tips = [], []
    for ra, rb in zip(rows_a, rows_b):
        if not (ra["hands"] and rb["hands"]):
            continue
        for ha in ra["hands"]:
            wa = ha["keypoints_2d_21x2"][0]
            best_j, best_d = None, None
            for j, hb in enumerate(rb["hands"]):
                wb = hb["keypoints_2d_21x2"][0]
                d = ((wa[0]-wb[0])**2 + (wa[1]-wb[1])**2) ** 0.5
                if best_d is None or d < best_d:
                    best_j, best_d = j, d
            if best_d is not None:
                wrists.append(best_d)
                ta = ha["keypoints_2d_21x2"][8]
                tb = rb["hands"][best_j]["keypoints_2d_21x2"][8]
                tips.append(((ta[0]-tb[0])**2 + (ta[1]-tb[1])**2) ** 0.5)
    return {
        "n_compared": len(wrists),
        "wrist_dist_median_px": statistics.median(wrists) if wrists else None,
        "wrist_dist_p95_px": sorted(wrists)[int(len(wrists)*0.95)] if wrists else None,
        "fingertip8_dist_median_px": statistics.median(tips) if tips else None,
        "matching": "nearest_neighbor_per_frame",
    }


def main():
    out = {}
    for clip in CLIPS:
        clip_res = {}
        ref_rows = load(RUN_ID, REF, clip)
        for route in ROUTES:
            rows = load(RUN_ID, route, clip)
            if rows is None:
                clip_res[route] = "未测"
                continue
            s = summarize(rows)
            if route != REF and ref_rows:
                s["vs_ref"] = cross_compare(rows, ref_rows)
            clip_res[route] = s
        out[clip] = clip_res

    out_path = os.path.join(ROOT, "artifacts", "hand_feasibility",
                            "p2_screening", "summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"写入 {out_path}")

    # 控制台摘要
    for clip, res in out.items():
        print(f"\n== {clip} ==")
        for route, s in res.items():
            if s == "未测":
                print(f"  {route}: 未测")
                continue
            line = f"  {route}: 覆盖 {s['output_coverage']:.0%}, {s['model_ms_median'] and round(s['model_ms_median'],1)}ms"
            if "vs_ref" in s and s["vs_ref"]["n_compared"]:
                v = s["vs_ref"]
                line += f", 手腕差 {v['wrist_dist_median_px'] and round(v['wrist_dist_median_px'],1)}px"
            print(line)


if __name__ == "__main__":
    main()
