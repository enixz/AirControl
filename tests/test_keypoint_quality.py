"""T-Keypoint-Quality: 关键点质量提升测试

验证：
  1. OneEuroSmoother 置信度加权平滑（帧间位移大的点更强平滑）
  2. GeometricConstraintFilter 几何约束（骨骼长度突变修正）
  3. TemporalGestureVoter 抖动自适应窗口
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'app'))

import numpy as np
from services.base_hand_tracker import GeometricConstraintFilter, OneEuroSmoother
from services.temporal_voter import MAX_WINDOW, MIN_WINDOW, TemporalGestureVoter


def _make_landmarks(pts):
    """把 (x, y) 列表转为 [idx, x, y] 格式。"""
    return [[i, float(x), float(y)] for i, (x, y) in enumerate(pts)]


class TestConfidenceWeightedSmoothing(unittest.TestCase):
    """测试 OneEuroSmoother 的置信度加权平滑"""

    def test_stable_point_keeps_original_cutoff(self):
        """稳定点（小位移）保持原 min_cutoff"""
        smoother = OneEuroSmoother(min_cutoff=1.0, beta=0.01)
        pts = _make_landmarks([(100 + i * 0.1, 100) for i in range(21)])
        smoother.update(pts)
        # 第二帧位移很小，cutoff 不应降低
        pts2 = _make_landmarks([(100.1 + i * 0.1, 100.1) for i in range(21)])
        smoother.update(pts2)
        cutoff = smoother._compute_effective_cutoff(0)
        self.assertAlmostEqual(cutoff, 1.0, places=2)

    def test_jittery_point_lowers_cutoff(self):
        """抖动点（大位移）降低 min_cutoff"""
        smoother = OneEuroSmoother(min_cutoff=1.0, beta=0.01, jitter_threshold=3.0)
        pts = _make_landmarks([(100, 100) for _ in range(21)])
        smoother.update(pts)
        # 第二帧大幅跳变
        pts2 = _make_landmarks([(100, 100) for _ in range(21)])
        pts2[8] = [8, 200.0, 200.0]  # 指尖跳变 100px
        smoother.update(pts2)
        # 关键点 8 的 jitter EMA 应该很大
        jitter = smoother._jitter_ema[8]
        self.assertGreater(jitter, 3.0)
        # cutoff 应该降低
        cutoff = smoother._compute_effective_cutoff(8)
        self.assertLess(cutoff, 1.0)

    def test_jitter_stats_available(self):
        """get_jitter_stats 返回 21 个关键点的抖动 EMA"""
        smoother = OneEuroSmoother()
        stats = smoother.get_jitter_stats()
        self.assertEqual(len(stats), 21)
        self.assertTrue(np.all(stats == 0.0))  # 初始全 0

    def test_reset_clears_jitter(self):
        """reset 清除抖动状态"""
        smoother = OneEuroSmoother()
        pts = _make_landmarks([(100, 100) for _ in range(21)])
        smoother.update(pts)
        pts[8] = [8, 200.0, 200.0]
        smoother.update(pts)
        self.assertGreater(smoother._jitter_ema[8], 0)
        smoother.reset()
        self.assertTrue(np.all(smoother._jitter_ema == 0.0))
        self.assertIsNone(smoother._prev_raw)


class TestGeometricConstraintFilter(unittest.TestCase):
    """测试几何约束后处理"""

    def test_first_frame_no_correction(self):
        """首帧不做修正"""
        gf = GeometricConstraintFilter()
        pts = _make_landmarks([(i * 10, i * 10) for i in range(21)])
        result = gf.apply(pts)
        self.assertEqual(result, pts)

    def test_normal_motion_no_correction(self):
        """正常运动（骨骼长度不变）不做修正"""
        gf = GeometricConstraintFilter()
        pts1 = _make_landmarks([(i * 10, i * 10) for i in range(21)])
        gf.apply(pts1)
        # 整体平移，骨骼长度不变
        pts2 = _make_landmarks([(i * 10 + 5, i * 10 + 5) for i in range(21)])
        result = gf.apply(pts2)
        self.assertEqual(len(result), 21)
        # 应该和输入一致（无修正）
        for i in range(21):
            self.assertAlmostEqual(result[i][1], pts2[i][1], places=4)
            self.assertAlmostEqual(result[i][2], pts2[i][2], places=4)

    def test_bone_length_violation_corrected(self):
        """骨骼长度突变时修正关键点"""
        gf = GeometricConstraintFilter(max_bone_length_change=0.3)
        # 正常手部关键点
        pts1 = _make_landmarks([
            (0, 0),    # 0 wrist
            (10, 0),   # 1 thumb_cmc
            (20, 0),   # 2 thumb_mcp
            (30, 0),   # 3 thumb_ip
            (40, 0),   # 4 thumb_tip
        ] + [(50 + i * 10, 0) for i in range(16)])
        gf.apply(pts1)
        # 第二帧：拇指指尖突然跳到很远的地方（骨骼 3-4 长度从 10 变成 100）
        pts2 = _make_landmarks([
            (0, 0), (10, 0), (20, 0), (30, 0),
            (130, 0),  # 4 thumb_tip 跳变
        ] + [(50 + i * 10, 0) for i in range(16)])
        result = gf.apply(pts2)
        # 关键点 4 应该被修正回上一帧位置
        self.assertAlmostEqual(result[4][1], 40.0, places=1)

    def test_reset_clears_state(self):
        """reset 清除状态"""
        gf = GeometricConstraintFilter()
        pts = _make_landmarks([(i * 10, i * 10) for i in range(21)])
        gf.apply(pts)
        gf.reset()
        self.assertIsNone(gf._prev_bone_lengths)
        self.assertIsNone(gf._prev_landmarks)

    def test_non_21_landmarks_passthrough(self):
        """非 21 个关键点直接返回不处理"""
        gf = GeometricConstraintFilter()
        pts = _make_landmarks([(0, 0), (10, 10), (20, 20)])
        result = gf.apply(pts)
        self.assertEqual(result, pts)

    def test_z_occlusion_reverts_outlier_keypoint(self):
        """P0-3: 单点 z 突跳（遮挡/翻面）应回退到上一帧位置。"""
        gf = GeometricConstraintFilter(z_occlusion_threshold=0.06)
        # 第一帧：所有点 z=0.0（3 元组 + z）
        pts1 = [[i, float(i * 10), float(i * 10), 0.0] for i in range(21)]
        gf.apply(pts1)
        # 第二帧：整体 z 漂移到 0.02，但关键点 8(食指尖) z 突跳到 0.20（遮挡）
        pts2 = [[i, float(i * 10), float(i * 10), 0.02] for i in range(21)]
        pts2[8] = [8, 80.0, 80.0, 0.20]
        result = gf.apply(pts2)
        # 关键点 8 的 z 应回退到上一帧（0.0），而非保留突跳值 0.20
        self.assertAlmostEqual(result[8][3], 0.0, places=6)
        # 其余点 z 不被回退（跟随整体漂移 0.02）
        self.assertAlmostEqual(result[0][3], 0.02, places=6)

    def test_z_preserved_in_output_when_present(self):
        """带 z 的输入，输出应保留 z 维度（4 元组）。"""
        gf = GeometricConstraintFilter()
        pts1 = [[i, float(i * 10), float(i * 10), 0.0] for i in range(21)]
        result = gf.apply(pts1)
        self.assertEqual(len(result[0]), 4)

    def test_3tuple_input_stays_3tuple(self):
        """无 z 的 3 元组输入应保持 3 元组输出（向后兼容）。"""
        gf = GeometricConstraintFilter()
        pts1 = _make_landmarks([(i * 10, i * 10) for i in range(21)])
        pts2 = _make_landmarks([(i * 10 + 1, i * 10 + 1) for i in range(21)])
        gf.apply(pts1)
        result = gf.apply(pts2)
        self.assertEqual(len(result[0]), 3)


class TestJitterAdaptiveWindow(unittest.TestCase):
    """测试抖动自适应窗口"""

    def test_zero_jitter_uses_min_window(self):
        """无抖动时使用最小窗口"""
        voter = TemporalGestureVoter()
        # 近距离 + 无抖动
        window = voter._adaptive_window(hand_width=90.0, jitter=0.0)
        self.assertEqual(window, MIN_WINDOW)

    def test_high_jitter_uses_max_window(self):
        """高抖动时使用最大窗口"""
        voter = TemporalGestureVoter()
        # 近距离但高抖动
        window = voter._adaptive_window(hand_width=90.0, jitter=15.0)
        self.assertEqual(window, MAX_WINDOW)

    def test_jitter_monotonic_with_window(self):
        """窗口长度随抖动单调递增"""
        voter = TemporalGestureVoter()
        w0 = voter._adaptive_window(90.0, jitter=0.0)
        w5 = voter._adaptive_window(90.0, jitter=5.0)
        w10 = voter._adaptive_window(90.0, jitter=10.0)
        self.assertLessEqual(w0, w5)
        self.assertLessEqual(w5, w10)

    def test_jitter_and_distance_combined(self):
        """距离 + 抖动两个因子取最大值"""
        voter = TemporalGestureVoter()
        # 远距离 + 低抖动
        w_far_low = voter._adaptive_window(30.0, jitter=0.0)
        # 近距离 + 高抖动
        w_near_high = voter._adaptive_window(90.0, jitter=15.0)
        # 两者都应该拉长窗口
        self.assertGreater(w_far_low, MIN_WINDOW)
        self.assertEqual(w_near_high, MAX_WINDOW)

    def test_update_accepts_jitter(self):
        """update 方法接受 jitter 参数"""
        voter = TemporalGestureVoter()
        gestures = [{"ml_label": "Closed_Fist", "score": 0.9, "label": "FIST"}]
        # 不应报错
        result = voter.update(gestures, hand_width=90.0, jitter=5.0)
        self.assertIsInstance(result, str)


if __name__ == "__main__":
    unittest.main(verbosity=2)
