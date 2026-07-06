"""ActiveRegionMapper 自适应活动区映射测试。

验证：远距离时手只扫过画面一小块也能映射到全屏；静止时映射到中心、不漂移；
输出始终在 [0,1]。

所有 map() 调用传入 dt=1/30 模拟 30fps，确保收缩量确定性（contract 按秒计算）。
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'app'))

from services.mouse_controller import ActiveRegionMapper

_DT = 1.0 / 30.0  # 模拟 30fps 的帧间隔


class TestActiveRegionMapper(unittest.TestCase):
    def _calibrate(self, m, lo, hi, axis='x', iters=200):
        """反复在 [lo, hi] 间扫动，让活动区收敛。"""
        for i in range(iters):
            v = lo if i % 2 == 0 else hi
            if axis == 'x':
                m.map(v, 0.5, dt=_DT)
            else:
                m.map(0.5, v, dt=_DT)

    def test_output_always_in_unit_range(self):
        m = ActiveRegionMapper()
        for v in (0.0, 0.1, 0.37, 0.5, 0.63, 0.9, 1.0):
            x, y = m.map(v, 1.0 - v, dt=_DT)
            self.assertGreaterEqual(x, 0.0)
            self.assertLessEqual(x, 1.0)
            self.assertGreaterEqual(y, 0.0)
            self.assertLessEqual(y, 1.0)

    def test_small_offcenter_sweep_reaches_full_screen(self):
        """远距离：手只在画面 [0.55, 0.75] 偏右扫动，也应能写到屏幕两端。"""
        m = ActiveRegionMapper()
        self._calibrate(m, 0.55, 0.75, axis='x')
        x_lo, _ = m.map(0.55, 0.5, dt=_DT)
        x_hi, _ = m.map(0.75, 0.5, dt=_DT)
        self.assertLess(x_lo, 0.1)       # 扫动左端 → 屏幕左侧
        self.assertGreater(x_hi, 0.9)    # 扫动右端 → 屏幕右侧

    def test_midpoint_maps_to_center(self):
        m = ActiveRegionMapper()
        self._calibrate(m, 0.3, 0.7, axis='x')
        x_mid, _ = m.map(0.5, 0.5, dt=_DT)
        self.assertAlmostEqual(x_mid, 0.5, delta=0.08)

    def test_stationary_point_maps_to_center_no_drift(self):
        """静止手（始终在 0.8）应稳定映射到中心附近，且不随时间漂移。"""
        m = ActiveRegionMapper()
        last = None
        for _ in range(120):
            x, _ = m.map(0.8, 0.2, dt=_DT)
            last = x
        self.assertAlmostEqual(last, 0.5, delta=0.08)

    def test_freeze_keeps_region_stable_during_writing(self):
        """书写中（update=False）活动区冻结：超出当前区域的点不再改变传递函数。"""
        m = ActiveRegionMapper()
        self._calibrate(m, 0.3, 0.7, axis='x')
        lo_before, hi_before = list(m._lo), list(m._hi)
        # 冻结：喂一个远超区域的点，区域不变
        m.map(0.99, 0.5, update=False, dt=_DT)
        self.assertEqual(m._lo, lo_before)
        self.assertEqual(m._hi, hi_before)
        # 非冻结：同样的越界点应扩张区域
        m.map(0.99, 0.5, update=True, dt=_DT)
        self.assertGreater(m._hi[0], hi_before[0])

    def test_reset_clears_state(self):
        m = ActiveRegionMapper()
        self._calibrate(m, 0.2, 0.4, axis='x')
        m.reset()
        # reset 后第一个点视为新的起点 → 映射到中心
        x, _ = m.map(0.9, 0.5, dt=_DT)
        self.assertAlmostEqual(x, 0.5, delta=0.08)

    def test_contraction_is_time_based(self):
        """收缩量应按时间计算：相同迭代次数下，dt 越大收缩越快。"""
        # 用小 dt（慢收缩）
        m_slow = ActiveRegionMapper()
        for i in range(60):
            v = 0.3 if i % 2 == 0 else 0.7
            m_slow.map(v, 0.5, dt=1.0 / 60.0)  # 60fps：每帧 dt 小
        span_slow = m_slow._hi[0] - m_slow._lo[0]

        # 用大 dt（快收缩）
        m_fast = ActiveRegionMapper()
        for i in range(60):
            v = 0.3 if i % 2 == 0 else 0.7
            m_fast.map(v, 0.5, dt=1.0 / 15.0)  # 15fps：每帧 dt 大
        span_fast = m_fast._hi[0] - m_fast._lo[0]

        # 大 dt（低帧率）→ 同样迭代次数下总时间更长 → 收缩更多 → 活动区更小
        self.assertLess(span_fast, span_slow,
                        "低帧率（大 dt）应在相同迭代次数下收缩更多")


if __name__ == "__main__":
    unittest.main(verbosity=2)
