"""Review the capture handoff candidate against unchanged production models."""
import argparse
from collections import Counter
import json
from pathlib import Path

from .corpus_report import fresh, ranges, read, records
from .run_replay import write_json


def episodes(rows):
    result = []
    start = None
    for i, row in enumerate(rows):
        if row["fsm_state"] == "capture" and start is None:
            start = i
        elif row["fsm_state"] != "capture" and start is not None:
            result.append(dict(start_frame=start, end_frame=i, next_state=row["fsm_state"],
                               duration_sec=row["source_timestamp"] - rows[start]["source_timestamp"]))
            start = None
    if start is not None:
        result.append(dict(start_frame=start, end_frame=None, next_state=None,
                           duration_sec=rows[-1]["source_timestamp"] - rows[start]["source_timestamp"]))
    return result


def render(baseline, candidate, destination):
    a, b = read(baseline / "baseline.json"), read(candidate / "baseline.json")
    assert a["models_sha256"] == b["models_sha256"]
    assert a["config_input_sha256"] == b["config_input_sha256"]
    assert a["arguments"]["mode"] == b["arguments"]["mode"]
    assert {k: v for k, v in a["source_sha256"].items() if Path(k).parts[0] == "app"} == {
        k: v for k, v in b["source_sha256"].items() if Path(k).parts[0] == "app"}
    assert all(read(p / "config_preservation.json")["unchanged"] for p in (baseline, candidate))
    cases = []
    for clip in b["arguments"]["clips"]:
        old, new = baseline / clip / "current", candidate / clip / "capture_handoff"
        ca, cb = read(old / "effective_config.json"), read(new / "effective_config.json")
        assert cb.pop("experiment_capture_handoff") is True
        assert ca == cb
        assert read(old / "dataset_manifest.json") == read(new / "dataset_manifest.json")
        ra, rb = records(old / "predictions.jsonl"), records(new / "predictions.jsonl")
        assert len(ra) == len(rb)
        assert [r["source_timestamp"] for r in ra] == [r["source_timestamp"] for r in rb]
        lost = [i for i, (x, y) in enumerate(zip(ra, rb)) if fresh(x) and not fresh(y)]
        gained = [i for i, (x, y) in enumerate(zip(ra, rb)) if not fresh(x) and fresh(y)]
        counts = lambda rows: dict(Counter(f"{e['sink']}.{e['method']}" for r in rows for e in r["operation_events"]))
        zoom = lambda rows: [r["frame_index"] for r in rows if r.get("used_crop_zoom")]
        mouse = lambda r: [e for e in r["operation_events"] if e["sink"] == "mouse"]
        cases.append(dict(clip=clip, frames=len(rb), before_fresh=sum(map(fresh, ra)), after_fresh=sum(map(fresh, rb)),
                          recovered=gained, new_missing=lost, new_missing_ranges=ranges(lost),
                          zoom_frames_before=len(zoom(ra)), zoom_frames_after=len(zoom(rb)),
                          first_zoom_before=next(iter(zoom(ra)), None), first_zoom_after=next(iter(zoom(rb)), None),
                          capture_before=episodes(ra), capture_after=episodes(rb),
                          operations_before=counts(ra), operations_after=counts(rb),
                          mouse_changed_frames=[i for i, (x, y) in enumerate(zip(ra, rb)) if mouse(x) != mouse(y)]))
    result = dict(candidate="capture_handoff", adopted=False,
                  reason="Earlier handoff enables ZOOM but introduces missing-primary intervals and changes actions.",
                  recordings=len(cases), frames=sum(c["frames"] for c in cases),
                  recovered_source_frames=sum(len(c["recovered"]) for c in cases),
                  new_missing_source_frames=sum(len(c["new_missing"]) for c in cases),
                  model_and_sr_unchanged=True, source_config_unchanged=True, cases=cases,
                  limits="Fresh-primary output is not detection recall; no identity or action intent labels, no live latency measured.")
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / "comparison.json", result)
    lines = ["# 提前交给 ZOOM 的实验结果", "",
             "候选只改 CAPTURE→FAR_TRACK：至少 5 个新鲜主手样本、位置和编号连续 ≥0.3 秒，捕获态驻留 ≥0.5 秒才提前交接；相邻双手、预测、位置跳变和间隔过长重置证据。其他切换、模型和 SR 不变。",
             "", "| 录像 | 新鲜输出 前→后 | 恢复帧 | 新增缺失帧 | ZOOM 帧 前→后 |",
             "|---|---:|---:|---:|---:|"]
    for c in cases:
        lines.append(f"| {c['clip']} | {c['before_fresh']}→{c['after_fresh']} | {len(c['recovered'])} | {len(c['new_missing'])} | {c['zoom_frames_before']}→{c['zoom_frames_after']} |")
    lines += ["", "候选未进入生产。短录像能更早进入 ZOOM，但总新鲜输出增多仍掩盖新增缺失；长录像也出现净减少。动作调用随检测变化，不能算成功率提升。已有反例足以拒绝这一版，不继续扩大同一候选的测试。",
              "", "下一步需验证跟踪引擎实际接住捕获到的同一只手，再完成交接；几何连续的捕获结果不足以证明另一引擎能持续识别。待验证的方向包括试跟踪确认和交接失败回退，当前未修改生产冷却或 SR 参数。", ""]
    (destination / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "candidate", "destination"):
        parser.add_argument(name, type=Path)
    result = render(**vars(parser.parse_args()))
    print(json.dumps({k: v for k, v in result.items() if k != "cases"}, indent=2))
