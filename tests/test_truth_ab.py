"""带真值 A/B 指标纯函数测试（services/truth_ab.py）。"""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from services.truth_ab import (
    compute_metrics,
    load_frame_times,
    load_truth_intervals,
    match_events,
    states_to_intervals,
)


class TestStatesToIntervals(unittest.TestCase):
    def test_basic(self):
        times = [0.0, 0.1, 0.2, 0.3, 0.4]
        states = [False, True, True, False, True]
        self.assertEqual(
            states_to_intervals(states, times), [(0.1, 0.2), (0.4, 0.4)],
        )

    def test_ends_while_true(self):
        times = [0.0, 0.1, 0.2]
        states = [False, True, True]
        self.assertEqual(states_to_intervals(states, times), [(0.1, 0.2)])

    def test_all_false(self):
        self.assertEqual(states_to_intervals([False, False], [0.0, 0.1]), [])


class TestLoadTruthIntervals(unittest.TestCase):
    def _write(self, d, rows):
        path = os.path.join(d, "truth_events.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        return path

    def test_pairing_and_sort(self):
        with tempfile.TemporaryDirectory() as d:
            path = self._write(d, [
                {"type": "header"},
                {"t": 2.0, "key": "space", "event": "down"},
                {"t": 2.3, "key": "space", "event": "up"},
                {"t": 1.0, "key": "space", "event": "down"},
                {"t": 1.2, "key": "space", "event": "up"},
                {"type": "footer", "t": 3.0},
            ])
            intervals = load_truth_intervals(path)
        self.assertEqual(
            [(s, e) for s, e, _ in intervals], [(1.0, 1.2), (2.0, 2.3)],
        )

    def test_missing_up_closed_by_footer(self):
        with tempfile.TemporaryDirectory() as d:
            path = self._write(d, [
                {"t": 1.0, "key": "space", "event": "down"},
                {"type": "footer", "t": 5.0},
            ])
            intervals = load_truth_intervals(path)
        self.assertEqual(intervals, [(1.0, 5.0, "space")])


class TestLoadFrameTimes(unittest.TestCase):
    def test_reads_times_in_row_order(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "meta.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                f.write('{"i":0,"t":10.0}\n')
                f.write('{"i":1,"t":10.05}\n')
                f.write("\n")  # 空行跳过
                f.write('{"i":3,"t":10.11}\n')  # i 跳号不影响顺序对齐
            self.assertEqual(load_frame_times(path), [10.0, 10.05, 10.11])


class TestMatchEvents(unittest.TestCase):
    def test_overlap_hit(self):
        hits, misses, false_alarms = match_events([(1.0, 1.5)], [(1.2, 1.6)])
        self.assertEqual(len(hits), 1)
        self.assertEqual(misses, [])
        self.assertEqual(false_alarms, [])

    def test_gap_within_tolerance_hit(self):
        hits, _, _ = match_events([(1.0, 1.1)], [(1.4, 1.6)], onset_tolerance=0.5)
        self.assertEqual(len(hits), 1)

    def test_gap_beyond_tolerance_miss_and_false_alarm(self):
        hits, misses, false_alarms = match_events(
            [(1.0, 1.1)], [(2.0, 2.2)], onset_tolerance=0.5,
        )
        self.assertEqual(hits, [])
        self.assertEqual(len(misses), 1)
        self.assertEqual(len(false_alarms), 1)

    def test_matching_is_one_to_one(self):
        hits, misses, _ = match_events([(1.0, 1.2), (1.1, 1.3)], [(1.05, 1.25)])
        self.assertEqual(len(hits), 1)
        self.assertEqual(len(misses), 1)


class TestComputeMetrics(unittest.TestCase):
    def test_hand_computed_example(self):
        truth = [(1.0, 1.5), (3.0, 3.4), (5.0, 5.2)]
        detected = [(1.1, 1.45), (8.0, 8.3)]  # 命中1 / 漏检2 / 误报1
        m = compute_metrics(truth, detected)
        self.assertEqual(m["truth_events"], 3)
        self.assertEqual(m["detected_events"], 2)
        self.assertEqual(m["hits"], 1)
        self.assertEqual(m["misses"], 2)
        self.assertAlmostEqual(m["recall"], 1 / 3)
        self.assertAlmostEqual(m["miss_rate"], 2 / 3)
        self.assertAlmostEqual(m["precision"], 0.5)
        self.assertEqual(m["false_alarms"], 1)
        self.assertAlmostEqual(m["onset_delay_mean_ms"], 100.0)
        self.assertAlmostEqual(m["offset_delay_mean_ms"], -50.0)

    def test_no_truth_gives_none_rates(self):
        m = compute_metrics([], [(1.0, 1.2)])
        self.assertIsNone(m["recall"])
        self.assertIsNone(m["miss_rate"])
        self.assertEqual(m["false_alarms"], 1)
        self.assertEqual(m["precision"], 0.0)

    def test_no_detected_gives_none_precision(self):
        m = compute_metrics([(1.0, 1.2)], [])
        self.assertIsNone(m["precision"])
        self.assertEqual(m["recall"], 0.0)
        self.assertEqual(m["misses"], 1)


if __name__ == "__main__":
    unittest.main()
