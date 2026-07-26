"""侧位姿态矩阵 A/B：逐段录像 × 多引擎，汇总成一张对比表。

用法（先录后跑）：
  1. 录制：运行 app（python -m app.main_ui），F8 开始/停止录制。
     每个"姿态段"录一个目录，命名建议带角度标签，例如：
       raw_capture/pose_yaw000_near/   手正对镜头，近距
       raw_capture/pose_yaw045_near/   手转 45°
       raw_capture/pose_yaw090_near/   手侧对（手背/侧面）
       raw_capture/pose_yaw135_near/
       raw_capture/pose_yaw180_near/   手完全背对镜头
       ... 以及 _far 一组（3–5m）
     录制时每段保持该角度 ~5–10 秒即可。
  2. 跑矩阵：
       python benchmark_pose_matrix.py raw_capture/pose_yaw000_near raw_capture/pose_yaw045_near ...
     或一次性给根目录（自动发现其子目录）：
       python benchmark_pose_matrix.py --glob "raw_capture/pose_yaw*"

输出：每个目录 × 每个引擎一行（检出率/多手率/抖动/延迟），并写
  <第一段所在目录>/pose_matrix_result.json。

默认对比三引擎：mediapipe / hagrid_yolo / person_pose_hand。
"""

import argparse
import glob
import json
import os
import sys

# 复用 benchmark_ab 的单引擎跑测
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchmark_ab


def run_one(rec_dir, engine, cfg):
    video = None
    for name in ("frames.mkv", "frames.mp4"):
        cand = os.path.join(rec_dir, name)
        if os.path.isfile(cand):
            video = cand
            break
    if video is None:
        return {"engine": engine, "error": "no frames.mp4/mkv"}
    try:
        return benchmark_ab.run_engine(engine, video, cfg)
    except Exception as e:  # 单段失败不拖垮整个矩阵
        return {"engine": engine, "error": str(e)}


def label_of(rec_dir):
    return os.path.basename(os.path.normpath(rec_dir))


def main(argv=None):
    ap = argparse.ArgumentParser(description="侧位姿态矩阵 A/B")
    ap.add_argument("rec_dirs", nargs="*", help="录像目录列表")
    ap.add_argument("--glob", dest="glob_pat", help="自动发现目录的 glob，如 'raw_capture/pose_yaw*'")
    ap.add_argument("--engines", default="mediapipe,hagrid_yolo,person_pose_hand",
                    help="逗号分隔引擎列表")
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    help="config 覆盖 key=value（可多次）")
    ap.add_argument("--out", default=None, help="结果 JSON 输出路径")
    args = ap.parse_args(argv)

    rec_dirs = list(args.rec_dirs)
    if args.glob_pat:
        rec_dirs.extend(sorted(d for d in glob.glob(args.glob_pat) if os.path.isdir(d)))
    # 去重保序
    seen = set()
    uniq = []
    for d in rec_dirs:
        if d not in seen:
            seen.add(d)
            uniq.append(d)
    rec_dirs = [d for d in uniq if os.path.isdir(d)]
    if not rec_dirs:
        print("未找到录像目录。先按 docstring 录制，或用 --glob 指定。")
        return 2

    cfg = benchmark_ab._parse_overrides(args.overrides)
    engines = [e.strip() for e in args.engines.split(",") if e.strip()]

    all_rows = []
    for rec in rec_dirs:
        label = label_of(rec)
        print(f"\n=== 姿态段: {label} ===")
        for eng in engines:
            print(f"  [{eng}] 跑测中 ...", flush=True)
            metrics = run_one(rec, eng, cfg)
            metrics["_label"] = label
            all_rows.append(metrics)
            if "error" in metrics:
                print(f"    错误: {metrics['error']}")
            else:
                print(f"    检出 {metrics.get('detect_rate', 0):.1%}  "
                      f"多手 {metrics.get('multi_hand_rate', 0):.1%}  "
                      f"延迟P95 {metrics.get('latency_p95_ms', 0):.0f}ms")

    # 汇总表：行=姿态段，列=引擎检出率
    print("\n=== 侧位对比汇总（检出率）===")
    labels = []
    for r in all_rows:
        if r["_label"] not in labels:
            labels.append(r["_label"])
    header = f"{'姿态段':<22}" + "".join(f"{e:>18}" for e in engines)
    print(header)
    print("-" * len(header))
    for lab in labels:
        row = f"{lab:<22}"
        for eng in engines:
            m = next((r for r in all_rows if r["_label"] == lab and r["engine"] == eng), None)
            if m is None or "error" in m:
                row += f"{'—':>18}"
            else:
                row += f"{m.get('detect_rate', 0):>17.1%} "
        print(row)

    out_path = args.out or os.path.join(rec_dirs[0], "pose_matrix_result.json")
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"engines": engines, "rows": all_rows}, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存: {out_path}")
    except Exception as e:
        print(f"保存结果失败: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
