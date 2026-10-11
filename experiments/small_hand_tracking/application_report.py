"""Create application observation reports and review strips from original video."""
import argparse
from collections import Counter
import json
from pathlib import Path


def render(run_dir):
    rows = json.loads((run_dir / "comparison.json").read_text(encoding="utf-8"))
    lines = ["# 小手完整逻辑回放结果", "",
             "运行实际检测模型、自动切换、工作线程交接和模式判断；鼠标及画布使用记录器。"
             "模型构建在两帧之间同步完成，未复现摄像头、后台预热、Qt 排队和窗口交互延迟。"
             "这些结果描述离线逻辑，不能换算实机帧率或操作成功率。", "",
             "| 会话 | 初始模式 | 组别 | 帧数 | 新鲜主手输出 | 任意手输出 | 仅预测帧 | 裁剪跟踪帧 | 跟踪耗时 P95 ms |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['clip']} | {row['initial_mode']} | {row['variant']} | {row['frames']} | {row['fresh_primary_output_rate']:.2%} | {row['output_any_rate']:.2%} | {row['prediction_only_frames']} | {row['zoom_used_frames']} | {row['tracker_ms']['p95']:.1f} |")
    lines += ["", "current 保留当前参数；no_sr 只关闭 SR；always_zoom 在 no_sr 上从开始启用远距搜索及放大。"
              "responsive_crop 在 no_sr 上仅加快搜索框中心响应，responsive_auto 在 current 上加同样改动；缩放平滑和手部输出规则保持一致。"
              "identity_continuity 为身份处理实验，其确切实现以运行目录 source 快照为准；当前版本清除重复预测别名，已接入正式代码。"
              "所有组均运行实际自动切换并保留预测配置。新鲜主手输出的分母是所有帧，包括未抬手时间；未人工验证身份及坐标正确性。", "",
              "## 切换和操作记录", ""]
    for row in rows:
        lines += [f"### {row['clip']} {row['variant']}", "",
                  f"模式输出：`{row['mode_gestures']}`。实际放大方式：`{row['upscaler_call_counts']}`。", "",
                  f"控制器调用：`{row['operation_counts']}`。调用次数不等于正确操作次数。", ""]
        for e in row["fsm_transitions"]:
            lines.append(f"- 帧 {e['frame_index']}：{json.dumps(e, ensure_ascii=False)}")
        lines += [""]
    lines += ["## 后续复核", "", "应逐段复核无新鲜主手输出的区间，检查手是否可见、是否为主控手，以及模式判断是否符合动作意图。"
              "SR 对指尖位置的影响需要人工标点和固定裁剪对照。", ""]
    (run_dir / "application_report.md").write_text("\n".join(lines), encoding="utf-8")


def review_strips(run_dir, clip, variant="current", count=20):
    import cv2
    import numpy as np
    directory = run_dir / clip / variant
    rows = [json.loads(line) for line in (directory / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((directory / "dataset_manifest.json").read_text(encoding="utf-8"))
    # Include short gaps to reach representative samples; sampled images are
    # observations, not interval-wide visibility truth.
    gaps = sorted(summary["no_fresh_primary_intervals"], key=lambda g: -g["duration_ms"])[:count]
    gaps.sort(key=lambda g: g["start_frame"])
    cap = cv2.VideoCapture(manifest["video"])
    destination = run_dir / "review" / clip / variant
    destination.mkdir(parents=True, exist_ok=True)
    reviews, panels = [], []
    try:
        for number, gap in enumerate(gaps):
            start, end = gap["start_frame"], gap["end_frame_exclusive"]
            indices = [max(0, start - 1), start, (start + end - 1) // 2, min(len(rows) - 1, end)]
            prior = rows[max(0, start - 1)]["hands"]
            box = prior[0]["bbox_xyxy"] if prior else [700, 350, 1200, 850]
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            size = max(350, 4 * max(box[2] - box[0], box[3] - box[1]))
            tiles = []
            for index in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(f"Cannot decode review frame {index}")
                height, width = frame.shape[:2]
                side = int(min(size, height, width))
                x = int(max(0, min(cx - side / 2, width - side)))
                y = int(max(0, min(cy - side / 2, height - side)))
                zoom = cv2.resize(frame[y:y + side, x:x + side], (270, 270))
                full = cv2.resize(frame, (480, 270))
                fresh = bool(rows[index]["hands"] and rows[index]["hands"][0]["detected_fresh"])
                label = f"#{number + 1} f{index} {'fresh' if fresh else 'no fresh'}"
                cv2.putText(full, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .58, (0, 255, 0) if fresh else (0, 80, 255), 2)
                tiles.append(np.hstack([full, zoom]))
            panels.append(np.hstack(tiles))
            reviews.append(dict(sample_id=number + 1, **gap, sampled_frames=indices,
                                sampled_visibility="unreviewed", target_identity="unreviewed",
                                cause="unreviewed", note="Four snapshots do not label the entire interval"))
        for page in range(0, len(panels), 4):
            cv2.imwrite(str(destination / f"page_{page // 4 + 1}.jpg"), np.vstack(panels[page:page + 4]))
        (destination / "review_samples.json").write_text(json.dumps(reviews, ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
        cap.release()
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--review-clip")
    args = parser.parse_args()
    render(args.run_dir)
    if args.review_clip:
        print(review_strips(args.run_dir, args.review_clip))
