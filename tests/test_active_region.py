"""ActiveRegionMapper 自适应活动区映射测试。

验证：远距离时手只扫过画面一小块也能映射到全屏；静止时映射到中心、不漂移；
输出始终在 [0,1]。
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'app'))

from services.mouse_controller import ActiveRegionMapper


class TestActiveRegionMapper(unittest.TestCase):
    def _calibrate(self, m, lo, hi, axis='x', iters=200):
        """反复在 [lo, hi] 间扫动，让活动区收敛。"""
        for i in range(iters):
            v = lo if i % 2 == 0 else hi
            if axis == 'x':
                m.map(v, 0.5)
            else:
                m.map(0.5, v)

    def test_output_always_in_unit_range(self):
        m = ActiveRegionMapper()
        for v in (0.0, 0.1, 0.37, 0.5, 0.63, 0.9, 1.0):
            x, y = m.map(v, 1.0 - v)
            self.assertGreaterEqual(x, 0.0)
            self.assertLessEqual(x, 1.0)
            self.assertGreaterEqual(y, 0.0)
            self.assertLessEqual(y, 1.0)

    def test_small_offcenter_sweep_reaches_full_screen(self):
        """远距离：手只在画面 [0.55, 0.75] 偏右扫动，也应能写到屏幕两端。"""
        m = ActiveRegionMapper()
        self._calibrate(m, 0.55, 0.75, axis='x')
        x_lo, _ = m.map(0.55, 0.5)
        x_hi, _ = m.map(0.75, 0.5)
        self.assertLess(x_lo, 0.1)       # 扫动左端 → 屏幕左侧
        self.assertGreater(x_hi, 0.9)    # 扫动右端 → 屏幕右侧

    def test_midpoint_maps_to_center(self):
        m = ActiveRegionMapper()
        self._calibrate(m, 0.3, 0.7, axis='x')
        x_mid, _ = m.map(0.5, 0.5)
        self.assertAlmostEqual(x_mid, 0.5, delta=0.08)

    def test_stationary_point_maps_to_center_no_drift(self):
        """静止手（始终在 0.8）应稳定映射到中心附近，且不随时间漂移。"""
        m = ActiveRegionMapper()
        last = None
        for _ in range(120):
            x, _ = m.map(0.8, 0.2)
            last = x
        self.assertAlmostEqual(last, 0.5, delta=0.08)

    def test_freeze_keeps_region_stable_during_writing(self):
        """书写中（update=False）活动区冻结：超出当前区域的点不再改变传递函数。"""
        m = ActiveRegionMapper()
        self._calibrate(m, 0.3, 0.7, axis='x')
        lo_before, hi_before = list(m._lo), list(m._hi)
        # 冻结：喂一个远超区域的点，区域不变
        m.map(0.99, 0.5, update=False)
        self.assertEqual(m._lo, lo_before)
        self.assertEqual(m._hi, hi_before)
        # 非冻结：同样的越界点应扩张区域
        m.map(0.99, 0.5, update=True)
        self.assertGreater(m._hi[0], hi_before[0])

    def test_reset_clears_state(self):
        m = ActiveRegionMapper()
        self._calibrate(m, 0.2, 0.4, axis='x')
        m.reset()
        # reset 后第一个点视为新的起点 → 映射到中心
        x, _ = m.map(0.9, 0.5)
        self.assertAlmostEqual(x, 0.5, delta=0.08)


if __name__ == "__main__":
    unittest.main(verbosity=2)
