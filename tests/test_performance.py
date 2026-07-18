"""性能回归门禁 — 纯算法热路径组件的单帧延迟基准。

PLAN.md:207 列了 test_performance.py 计划但一直没做。本测试补这个缺口：
对每帧都跑的纯 CPU 算法组件设延迟上限，如果后续改动引入 O(n²) 退化或意外
昂贵操作（如误加载模型、误加 IO），中位数超阈值即测试失败。

只测不依赖摄像头/模型/硬件的纯算法层，保证可复现。阈值留 ~4x 余量防 CI 慢机
误报，同时能抓数量级退化。
"""
import math
import os
import sys
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'app'))

from services.base_hand_tracker import (  # noqa: E402
    GeometricConstraintFilter,
    KalmanSmoother,
    OneEuroSmoother,
)
from services.geometric_classifier import WeightedVoteClassifier  # noqa: E402

# 一只摊开的手的归一化关键点（21 个），作为抖动基准
_BASE_PTS = [
    (0.50, 0.70), (0.48, 0.66), (0.46, 0.62), (0.44, 0.60), (0.42, 0.58),
    (0.52, 0.60), (0.53, 0.54), (0.54, 0.50), (0.55, 0.46),
    (0.56, 0.60), (0.57, 0.53), (0.58, 0.49), (0.59, 0.45),
    (0.60, 0.60), (0.61, 0.54), (0.62, 0.50), (0.63, 0.46),
    (0.64, 0.60), (0.65, 0.55), (0.66, 0.51), (0.67, 0.47),
]


def _jittered_landmarks(frame_idx, with_z=True):
    """生成带微小抖动的 21 关键点，模拟真实逐帧输入。"""
    lm = []
    for i, (bx, by) in enumerate(_BASE_PTS):
        x = bx + 0.002 * math.sin(frame_idx * 0.13 + i)
        y = by + 0.0015 * math.cos(frame_idx * 0.11 + i)
        entry = [i, x, y]
        if with_z:
            entry.append(0.01 * math.sin(frame_idx * 0.09 + i))
        lm.append(entry)
    return lm


def _median_us_per_call(callable_, n_warmup=50, n_measure=2000):
    """跑 n_warmup 预热 + n_measure 计时，返回中位数微秒/次。"""
    for f in range(n_warmup):
        callable_(f)
    samples = []
    for f in range(n_warmup, n_warmup + n_measure):
        t0 = time.perf_counter()
        callable_(f)
        samples.append((time.perf_counter() - t0) * 1e6)
    samples.sort()
    return samples[len(samples) // 2]


class TestSmootherPerformance(unittest.TestCase):
    """平滑器每帧调用，是手→光标延迟链路的热路径。"""

    def test_one_euro_smoother_under_threshold(self):
        smoother = OneEuroSmoother()
        # 首帧初始化走不同路径，预热已覆盖
        us = _median_us_per_call(lambda f: smoother.update(_jittered_landmarks(f)))
        # 实测 ~0.1ms，阈值 0.5ms 留 ~5x 余量防 CI 慢机，能抓数量级退化
        self.assertLess(us, 500.0,
                        f"OneEuroSmoother.update 中位 {us:.0f}µs > 500µs 阈值，疑似退化")

    def test_kalman_smoother_under_threshold(self):
        smoother = KalmanSmoother()
        us = _median_us_per_call(lambda f: smoother.update(_jittered_landmarks(f)))
        # Kalman 每点 predict+correct，实测 ~0.27ms，阈值 1.5ms 留 ~5x 余量
        self.assertLess(us, 1500.0,
                        f"KalmanSmoother.update 中位 {us:.0f}µs > 1500µs 阈值，疑似退化")


class TestGeometricConstraintPerformance(unittest.TestCase):
    """几何约束每帧 apply，远距离增强层开启时是热路径。"""

    def test_geometric_constraint_filter_under_threshold(self):
        gf = GeometricConstraintFilter()
        us = _median_us_per_call(lambda f: gf.apply(_jittered_landmarks(f)))
        # 实测 ~0.1-0.2ms（21 点骨长 + z 漂移），阈值 1.0ms 留 5x 余量
        self.assertLess(us, 1000.0,
                        f"GeometricConstraintFilter.apply 中位 {us:.0f}µs > 1000µs 阈值，疑似退化")


class TestClassifierPerformance(unittest.TestCase):
    """加权投票分类器在 _stabilize_gestures 每帧调用。"""

    def test_weighted_vote_classify_under_threshold(self):
        clf = WeightedVoteClassifier()
        lm = _jittered_landmarks(0)
        us = _median_us_per_call(
            lambda f: clf.classify(lm, ml_label="Open_Palm", ml_score=0.8)
        )
        # 30+ 维特征提取，实测 ~0.3-0.6ms，阈值 2.0ms 留 3x 余量
        self.assertLess(us, 2000.0,
                        f"WeightedVoteClassifier.classify 中位 {us:.0f}µs > 2000µs 阈值，疑似退化")


if __name__ == '__main__':
    unittest.main()
