"""Render the development observations without relabelling them as accuracy."""
import argparse
import csv
import json
from pathlib import Path


def render(run_dir):
    run_dir = Path(run_dir)
    rows = json.loads((run_dir / "comparison.json").read_text(encoding="utf-8"))
    baseline = json.loads((run_dir / "baseline.json").read_text(encoding="utf-8"))
    lines = ["# 小手跟踪开发集首轮观察", "",
             "这是固定 MediaPipe tracker 的离线对照，未运行应用自动切换、鼠标或板书逻辑。"
             "覆盖率的分母为全部处理帧，包含无手和遮挡时间；新鲜输出表示当前帧产生了结果，不能证明结果正确。", "",
             f"源码版本：`{baseline['git_head']}`。完整配置、模型哈希、源码副本和每帧预测保存在本目录。", "",
             "| 会话 | 组别 | 帧数 | 新鲜主手输出 | 任意手输出 | 仅预测帧 | 裁剪跟踪分支帧 | tracker P95 ms |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['clip']} | {row['variant']} | {row['frames']} | "
                     f"{row['fresh_primary_output_rate']:.2%} | {row['output_any_rate']:.2%} | "
                     f"{row['prediction_only_frames']} | {row['zoom_used_frames']} | {row['tracker_ms']['p95']:.1f} |")
    lines += ["", "组别：current_tracker 保留当前 tracker 配置，但不执行自动引擎切换；"
              "A 为全图 720 宽预缩放，A2 为全图原尺寸，B 为 ZOOM 加普通插值，C 为 ZOOM 加 ESPCN，D 为 ZOOM 加 Real-ESRGAN。"
              "A 至 D 均关闭预测补帧、主动跳帧及自动引擎切换，其他参数一致。", "",
              "裁剪跟踪分支帧不包含全图失败后的人脸引导裁剪；实际放大调用在下表单列。"
              "B_grace 若出现，表示在 B 上增加最多 400ms 的短失锁搜索框保留实验。"
              "B_native 表示在 B 上增加原分辨率补搜，补搜间隔至少 200ms。"
              "C/D 显式指定引擎，可能对无需放大的大裁剪仍执行 SR；Z_auto 若出现则使用当前 auto 尺寸滞回与后端选择。", "",
              "## 实际放大方式", "",
              "| 会话 | 组别 | 实际档位及调用数 |", "|---|---|---|"]
    for row in rows:
        tiers = "; ".join(f"{k}: {v}" for k, v in row["upscaler_call_counts"].items()) or "无裁剪放大调用"
        lines.append(f"| {row['clip']} | {row['variant']} | {tiers} |")
    lines += ["", "## 状态变化", "",
              "下表记录导致 ZOOM 状态变化的方法，供排查静默退出。方法名是观察位置，不自动等于根因。", "",
              "| 会话 | 组别 | 状态变化位置与次数 |", "|---|---|---|"]
    for row in rows:
        counts = {}
        path = run_dir / f"round_{row['round']}" / row["clip"] / row["variant"] / "events.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            key = f"{event['observed_in']} {'ON' if event['to'] else 'OFF'}"
            counts[key] = counts.get(key, 0) + 1
        lines.append(f"| {row['clip']} | {row['variant']} | " + ("; ".join(f"{k}: {v}" for k, v in counts.items()) or "无") + " |")
    lines += ["", "## 结论边界", "",
              "- 当前仅为开发集首轮筛查；旧录像不能作独立验收集。",
              "- tracker 耗时含观察开销及冷启动或惰性模型加载，未完成预热和交错三轮性能测试；不得换算为实机 FPS。",
              "- 无人工可见性、身份、关键点和动作真值：真实检出率、准确度、重捕时延、操作成功率均未测。",
              "- B 与 A 同时改变裁剪和远距搜索；C、D 与 B 的在线裁剪轨迹也可能不同。纯 SR 收益仍需相同人工裁剪对照。",
              "- 有效 provider 列表保存在 summary.json；不代表所有节点均在 GPU 执行。",
              "- 原始配置是否保持一致见 config_preservation.json。", ""]
    output = run_dir / "ablation_report.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    with (run_dir / "failure_candidates.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        fields = ["clip", "variant", "round", "start_frame", "end_frame_exclusive", "duration_ms",
                  "left_censored", "right_censored", "visibility", "confirmed_tracking_failure", "review_note"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            path = run_dir / f"round_{row['round']}" / row["clip"] / row["variant"] / "summary.json"
            summary = json.loads(path.read_text(encoding="utf-8"))
            for gap in summary["no_fresh_primary_intervals"]:
                if gap["duration_ms"] >= 200 or gap["right_censored"]:
                    writer.writerow({"clip": row["clip"], "variant": row["variant"], "round": row["round"],
                                     **gap, "visibility": "unknown", "confirmed_tracking_failure": "unknown",
                                     "review_note": "需复核目标是否可见、身份是否正确；此表不能直接算丢手率"})
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    print(render(parser.parse_args().run_dir))
