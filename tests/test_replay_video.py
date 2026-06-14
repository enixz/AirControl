"""replay_video.compute_metrics 的纯函数单测。

只测指标聚合逻辑——不依赖摄像头/mediapipe/真实录像（create_hand_tracker 在
replay_video.run() 里延迟导入，所以 import replay_video 不会拉起 mediapipe）。
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from replay_video import compute_metrics


def rec(hands=1, top=None, zoom=False, center=(100.0, 100.0), size=300.0, ms=10.0):
    return {
        "hands": hands, "top_is_highest": top, "zoom_on": zoom,
        "crop_center": center, "crop_size": size, "ms": ms,
    }


class TestComputeMetrics(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(compute_metrics([]), {})

    def test_detect_rate(self):
        recs = [rec(hands=0), rec(hands=1), rec(hands=2), rec(hands=1)]
        m = compute_metrics(recs)
        self.assertEqual(m["frames"], 4)
        self.assertAlmostEqual(m["detect_rate"], 0.75)

    def test_zoom_toggles_and_active(self):
        pattern = [False, True, True, False, True]
        recs = [rec(zoom=z) for z in pattern]
        m = compute_metrics(recs)
        # F->T, T->T, T->F, F->T => 3 次翻转
        self.assertEqual(m["zoom_toggles"], 3)
        self.assertAlmostEqual(m["zoom_active"], 3 / 5)

    def test_viewport_jump_on_big_size_change(self):
        recs = [
            rec(center=(100.0, 100.0), size=300.0),
            rec(center=(100.0, 100.0), size=300.0),   # 无变化，不跳
            rec(center=(100.0, 100.0), size=400.0),   # Δ=100 > 15%*300=45，跳一次
        ]
        m = compute_metrics(recs)
        self.assertEqual(m["viewport_jumps"], 1)

    def test_no_jump_when_stable(self):
        recs = [rec(center=(100.0, 100.0), size=300.0) for _ in range(5)]
        m = compute_metrics(recs)
        self.assertEqual(m["viewport_jumps"], 0)
        self.assertAlmostEqual(m["viewport_jerk"], 0.0)

    def test_lock_top_rate_excludes_single_hand(self):
        recs = [rec(top=True), rec(top=True), rec(top=False), rec(top=None)]
        m = compute_metrics(recs)
        self.assertEqual(m["multi_hand_frames"], 3)   # None（单手）不计
        self.assertAlmostEqual(m["lock_top_rate"], 2 / 3)

    def test_lock_top_rate_none_when_no_multihand(self):
        recs = [rec(top=None), rec(top=None)]
        m = compute_metrics(recs)
        self.assertEqual(m["multi_hand_frames"], 0)
        self.assertIsNone(m["lock_top_rate"])

    def test_timing(self):
        recs = [rec(ms=10.0), rec(ms=20.0), rec(ms=30.0), rec(ms=40.0)]
        m = compute_metrics(recs)
        self.assertAlmostEqual(m["ms_mean"], 25.0)
        self.assertAlmostEqual(m["ms_p95"], 40.0)        # idx=min(3,int(4*0.95)=3)=3
        self.assertAlmostEqual(m["fps_proc"], 1000.0 / 25.0)


if __name__ == "__main__":
    unittest.main()
