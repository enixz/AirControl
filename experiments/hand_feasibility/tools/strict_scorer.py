"""严格评分器（T0，方案 §7.2/§8.2）：修正现有 truth_ab.py 的宽松口径。

与 truth_ab.py 的区别（旧口径结果标 legacy）：
  1. 匹配窗口收紧为 [真值起点-100ms, +500ms]，且以「窗口内距真值起点最近」
     取代「区间重叠 OR gap<=0.5s」——区间重叠允许起点相差 0.5s 以上，
     短促点击的准确率被系统性高估。
  2. 区分左右键与控制手：truth 区间带 key，检测区间也带 key，同键才配。
  3. 无真值事件时率指标返回 null，附暴露分钟数与 3/T 上界（不填 0）。
  4. 负延迟单独报告（检测起点早于真值起点 = 标注误差或预测）。

纯函数模块，供实验评分与单元测试使用；不修改 services/truth_ab.py。
"""
import statistics

PRESS_WINDOW_BEFORE_SEC = 0.1
PRESS_WINDOW_AFTER_SEC = 0.5


def load_frame_times(meta_path):
    """同 services.truth_ab.load_frame_times：meta.jsonl 每帧 epoch 时间戳。"""
    import json
    times = []
    with open(meta_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "t" in row:
                times.append(float(row["t"]))
    return times


def load_truth_intervals(events_path):
    """同 services.truth_ab.load_truth_intervals：[(start, end, key)]。"""
    import json
    opens = {}
    intervals = []
    last_t = None
    with open(events_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            t = row.get("t")
            if t is not None:
                last_t = float(t)
            if row.get("type") in ("header", "footer"):
                continue
            key = row.get("key")
            event = row.get("event")
            if t is None or key is None:
                continue
            if event == "down":
                if key in opens:
                    intervals.append((opens.pop(key), float(t), key))
                opens[key] = float(t)
            elif event == "up" and key in opens:
                intervals.append((opens.pop(key), float(t), key))
    for key, start in opens.items():
        if last_t is not None and last_t > start:
            intervals.append((start, last_t, key))
    intervals.sort(key=lambda iv: iv[0])
    return intervals


def states_to_keyed_intervals(states, times, keys=None):
    """逐帧 (bool, key) 序列转 [(start_t, end_t, key)]。

    keys 为 None 时所有帧同键（"left"）。逐帧 key 用于左右键分列的输出。
    """
    if keys is None:
        keys = ["left"] * len(states)
    intervals = []
    start_idx = None
    for i, flag in enumerate(states):
        if i >= len(times):
            break
        if flag and start_idx is None:
            start_idx = i
        elif not flag and start_idx is not None:
            intervals.append((times[start_idx], times[i - 1], keys[i - 1]))
            start_idx = None
    if start_idx is not None:
        last = min(len(states), len(times)) - 1
        intervals.append((times[start_idx], times[last], keys[last]))
    return intervals


def match_events_strict(truth, detected,
                        before=PRESS_WINDOW_BEFORE_SEC,
                        after=PRESS_WINDOW_AFTER_SEC):
    """严格一对一匹配：检测起点 ∈ [真值起点-before, 真值起点+after]。

    同键才配（truth[2] vs detected[2]；无键的输入按 "left" 处理）。
    一个真值至多配一个检测，取窗口内 |起点差| 最小者；窗口外不命中，
    即使区间有重叠（旧口径缺陷）。
    """
    def _key(iv):
        return iv[2] if len(iv) > 2 else "left"

    hits = []
    used = set()
    for t_iv in truth:
        t_start = t_iv[0]
        best_j = None
        best_gap = None
        for j, d_iv in enumerate(detected):
            if j in used:
                continue
            if _key(t_iv) != _key(d_iv):
                continue
            d_start = d_iv[0]
            if -before <= (d_start - t_start) <= after:
                gap = abs(d_start - t_start)
                if best_gap is None or gap < best_gap:
                    best_j, best_gap = j, gap
        if best_j is not None:
            used.add(best_j)
            hits.append((t_iv, detected[best_j]))
    hit_truth_ids = {id(t) for t, _ in hits}
    misses = [t for t in truth if id(t) not in hit_truth_ids]
    false_alarms = [d for j, d in enumerate(detected) if j not in used]
    return hits, misses, false_alarms


def false_alarm_upper_bound_per_min(n_false_alarms, exposure_minutes):
    """零/少误报时每分钟误报 95% 上界 ≈ 3/T（poisson 近似）。"""
    if exposure_minutes is None or exposure_minutes <= 0:
        return None
    if n_false_alarms == 0:
        return 3.0 / exposure_minutes
    # 有误报时用点估计+上界：粗略 poisson 上界 (n+3)/T
    return (n_false_alarms + 3.0) / exposure_minutes


def compute_metrics_strict(truth, detected, exposure_seconds=None):
    """由真值/检测区间计算严格口径指标（延迟毫秒）。

    exposure_seconds: 无真值事件时的暴露时长，用于误报上界。
    无真值时 recall/miss_rate 返回 null（不填 0）。
    """
    hits, misses, false_alarms = match_events_strict(truth, detected)
    onset_ms = [(d[0] - t[0]) * 1000.0 for t, d in hits]
    offset_ms = [(d[1] - t[1]) * 1000.0 for t, d in hits]
    n_truth = len(truth)
    n_det = len(detected)
    n_hits = len(hits)
    negative_onset = sum(1 for v in onset_ms if v < 0)
    exposure_minutes = (
        exposure_seconds / 60.0 if exposure_seconds is not None else None
    )
    out = {
        "scoring_version": "strict-1",
        "truth_events": n_truth,
        "detected_events": n_det,
        "hits": n_hits,
        "misses": len(misses),
        "false_alarms": len(false_alarms),
        "recall": (n_hits / n_truth) if n_truth else None,
        "miss_rate": (len(misses) / n_truth) if n_truth else None,
        "precision": (n_hits / n_det) if n_det else None,
        "negative_onset_count": negative_onset,
        "onset_delay_mean_ms": statistics.mean(onset_ms) if onset_ms else None,
        "onset_delay_p95_ms": _p95(onset_ms) if onset_ms else None,
        "offset_delay_mean_ms": statistics.mean(offset_ms) if offset_ms else None,
        "offset_delay_p95_ms": _p95(offset_ms) if offset_ms else None,
    }
    if n_truth == 0:
        out["exposure_minutes"] = exposure_minutes
        out["false_alarm_upper_bound_per_min"] = false_alarm_upper_bound_per_min(
            len(false_alarms), exposure_minutes
        )
    # 左右键分列（truth key 相同时跳过）
    truth_keys = {t[2] if len(t) > 2 else "left" for t in truth}
    if len(truth_keys) > 1:
        out["by_key"] = {}
        for k in sorted(truth_keys):
            t_k = [t for t in truth if (t[2] if len(t) > 2 else "left") == k]
            d_k = [d for d in detected if (d[2] if len(d) > 2 else "left") == k]
            sub = compute_metrics_strict(t_k, d_k, exposure_seconds)
            sub.pop("by_key", None)
            out["by_key"][k] = sub
    return out


def _p95(values):
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]
