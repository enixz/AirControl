"""严格评分器自检（方案 §12 评测代码必要自检）。

覆盖：匹配窗口语义、一对一匹配、同键配对、负延迟计数、
无真值返回 null 而非 0、误报上界公式。旧口径（truth_ab.py）对照标 legacy。
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.strict_scorer import (
    compute_metrics_strict,
    false_alarm_upper_bound_per_min,
    match_events_strict,
    states_to_keyed_intervals,
)


def test_window_match_within():
    # 检测起点晚真值 300ms → 命中（+500ms 内）
    truth = [(10.0, 10.2, "left")]
    det = [(10.3, 10.4, "left")]
    hits, misses, fa = match_events_strict(truth, det)
    assert len(hits) == 1 and not misses and not fa


def test_window_match_outside_overlap():
    # 检测起点早真值 200ms，区间重叠（10.0-10.3 与 10.0-10.2 重叠）
    # 旧口径会命中（重叠），严格口径不命中（起点在 -100ms 窗外）
    truth = [(10.0, 10.2, "left")]
    det = [(9.8, 10.3, "left")]
    hits, misses, fa = match_events_strict(truth, det)
    assert len(hits) == 0 and len(misses) == 1 and len(fa) == 1


def test_window_match_too_late():
    truth = [(10.0, 10.2, "left")]
    det = [(10.6, 10.9, "left")]  # 起点 +600ms 窗外
    hits, misses, fa = match_events_strict(truth, det)
    assert len(hits) == 0 and len(misses) == 1 and len(fa) == 1


def test_one_to_one_matching():
    # 两个检测事件 vs 一个真值：只有一个命中，另一个是误报
    truth = [(10.0, 10.2, "left")]
    det = [(10.05, 10.15, "left"), (10.3, 10.4, "left")]
    hits, misses, fa = match_events_strict(truth, det)
    assert len(hits) == 1 and len(misses) == 0 and len(fa) == 1
    # 取距真值起点最近者
    assert hits[0][1][0] == 10.05


def test_key_separation():
    # 左键真值 + 右键检测：不同键不配
    truth = [(10.0, 10.2, "left")]
    det = [(10.1, 10.3, "right")]
    hits, misses, fa = match_events_strict(truth, det)
    assert len(hits) == 0 and len(misses) == 1 and len(fa) == 1


def test_states_to_keyed_intervals():
    states = [True, True, False, True, False]
    times = [0.0, 0.1, 0.2, 0.3, 0.4]
    keys = ["left", "left", "left", "right", "right"]
    ivs = states_to_keyed_intervals(states, times, keys)
    assert ivs == [(0.0, 0.1, "left"), (0.3, 0.3, "right")]


def test_no_truth_returns_null_not_zero():
    # 无真值事件：recall/miss_rate 为 null，误报上界按 (n+3)/T
    truth = []
    det = [(10.0, 10.1, "left"), (20.0, 20.1, "left")]
    m = compute_metrics_strict(truth, det, exposure_seconds=1800.0)
    assert m["recall"] is None
    assert m["miss_rate"] is None
    assert m["hits"] == 0
    assert m["false_alarms"] == 2
    # 1800s = 30min，2 误报 → (2+3)/30 ≈ 0.167
    assert math.isclose(m["false_alarm_upper_bound_per_min"], 5 / 30)


def test_zero_fa_upper_bound():
    # 零误报：3/T = 3/30 = 0.1
    m = compute_metrics_strict([], [], exposure_seconds=1800.0)
    assert m["false_alarms"] == 0
    assert math.isclose(m["false_alarm_upper_bound_per_min"], 0.1)


def test_negative_onset_counted():
    truth = [(10.0, 10.2, "left")]
    det = [(9.95, 10.3, "left")]  # 起点 -50ms 窗内
    m = compute_metrics_strict(truth, det)
    assert m["hits"] == 1
    assert m["negative_onset_count"] == 1
    assert math.isclose(m["onset_delay_mean_ms"], -50.0)


def test_fa_upper_bound_with_events():
    assert math.isclose(
        false_alarm_upper_bound_per_min(2, 10.0), 0.5
    )
    assert false_alarm_upper_bound_per_min(0, None) is None


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"{len(fns)} 自检全部通过")
