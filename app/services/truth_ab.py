"""带真值的点击/拖拽 A/B 指标（纯函数模块）。

供 benchmark_gesture_ab.py 离线回放与单元测试使用：
把检测到的 pinch 状态序列（逐帧 bool）与 truth_events.jsonl 记录的真值
区间（意图点击/拖拽）按时间对齐，输出检出率、漏检率、准确率、误报数与
onset/offset 延迟统计——替代原先的纯观察性指标。

时间基准：meta.jsonl 的逐帧 "t" 与 truth_events.jsonl 的 "t" 同为录制进程
的 time.time() epoch 秒，天然对齐。
"""
import json
import statistics

# 检测事件起点与真值起点允许的最大偏差（秒）。真值是"边捏合边按键"，
# 检测器天然有几帧响应延迟，短促点击可能因此与真值区间无重叠。
DEFAULT_ONSET_TOLERANCE = 0.5


def load_frame_times(meta_path):
    """读取 meta.jsonl，按行序返回每帧 epoch 时间戳（第 k 行 ↔ 视频第 k 帧）。

    录制器丢帧时 meta 行与视频帧同步缺失，按顺序对齐即可，row["i"] 仅作
    信息字段不参与对齐。
    """
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
    """解析 truth_events.jsonl，返回 [(start_t, end_t, key)] 真值区间列表。

    down/up 按键配对；缺失 up 时用 footer 时间或最后一条事件时间闭合。
    返回按 start_t 升序。
    """
    opens = {}      # key -> start_t
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
                # 同键重复 down（异常）：先闭合旧区间再开新区间
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


def states_to_intervals(states, times):
    """把逐帧 bool 序列转成 [(start_t, end_t)] 事件区间（上升沿→最后一个 True 帧）。"""
    intervals = []
    start_idx = None
    for i, flag in enumerate(states):
        t = times[i] if i < len(times) else None
        if t is None:
            break
        if flag and start_idx is None:
            start_idx = i
        elif not flag and start_idx is not None:
            intervals.append((times[start_idx], times[i - 1]))
            start_idx = None
    if start_idx is not None:
        last = min(len(states), len(times)) - 1
        intervals.append((times[start_idx], times[last]))
    return intervals


def match_events(truth, detected, onset_tolerance=DEFAULT_ONSET_TOLERANCE):
    """真值区间与检测区间一对一贪心匹配。

    判定命中：时间区间有重叠，或检测起点距真值起点 ≤ onset_tolerance
    （覆盖短促点击"真值已结束、检测才到来"的情况）。每个检测事件至多
    匹配一个真值，取起始时间差最小者。

    Returns:
        (hits, misses, false_alarms)
        hits: [(truth_iv, det_iv)]；misses: 未命中的真值；false_alarms: 未匹配的检测。
    """
    hits = []
    used = set()
    for t_iv in truth:
        t_start, t_end = t_iv[0], t_iv[1]
        best_j = None
        best_gap = None
        for j, d_iv in enumerate(detected):
            if j in used:
                continue
            d_start, d_end = d_iv[0], d_iv[1]
            overlap = d_start <= t_end and t_start <= d_end
            gap = abs(d_start - t_start)
            if overlap or gap <= onset_tolerance:
                if best_gap is None or gap < best_gap:
                    best_j, best_gap = j, gap
        if best_j is not None:
            used.add(best_j)
            hits.append((t_iv, detected[best_j]))
    hit_truth_ids = {id(t) for t, _ in hits}
    misses = [t for t in truth if id(t) not in hit_truth_ids]
    false_alarms = [d for j, d in enumerate(detected) if j not in used]
    return hits, misses, false_alarms


def _p95(values):
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]


def compute_metrics(truth, detected, onset_tolerance=DEFAULT_ONSET_TOLERANCE):
    """由真值/检测区间计算量化指标（延迟单位毫秒）。"""
    hits, misses, false_alarms = match_events(truth, detected, onset_tolerance)
    onset_ms = [(d[0] - t[0]) * 1000.0 for t, d in hits]
    offset_ms = [(d[1] - t[1]) * 1000.0 for t, d in hits]
    n_truth = len(truth)
    n_det = len(detected)
    n_hits = len(hits)
    return {
        "truth_events": n_truth,
        "detected_events": n_det,
        "hits": n_hits,
        "misses": len(misses),
        "recall": (n_hits / n_truth) if n_truth else None,
        "miss_rate": (len(misses) / n_truth) if n_truth else None,
        "false_alarms": len(false_alarms),
        "precision": (n_hits / n_det) if n_det else None,
        "onset_delay_mean_ms": statistics.mean(onset_ms) if onset_ms else None,
        "onset_delay_p95_ms": _p95(onset_ms) if onset_ms else None,
        "offset_delay_mean_ms": statistics.mean(offset_ms) if offset_ms else None,
        "offset_delay_p95_ms": _p95(offset_ms) if offset_ms else None,
    }
