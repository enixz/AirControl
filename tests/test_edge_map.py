"""
T1: _edge_map 数学正确性单元测试
验证非线性边缘加速映射函数的数学性质。
"""
import sys
import types
import unittest
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from unittest.mock import patch

# 在唯一模块名下导入被测文件，mock 仅在导入上下文内存在，不污染 pytest collection。
mock_ctypes = types.ModuleType("ctypes")
mock_ctypes.wintypes = types.ModuleType("ctypes.wintypes")
mock_windll = types.ModuleType("ctypes.windll")
mock_user32 = types.ModuleType("user32")
mock_windll.user32 = mock_user32
mock_ctypes.windll = mock_windll
mock_ctypes.byref = lambda value: value
mock_ctypes.wintypes.POINT = lambda: type("POINT", (), {"x": 0, "y": 0})()

_module_path = Path(__file__).resolve().parents[1] / "app" / "services" / "mouse_controller.py"
_spec = spec_from_file_location("edge_map_mouse_controller_under_test", _module_path)
_module = module_from_spec(_spec)
with patch.dict(
    sys.modules,
    {"ctypes": mock_ctypes, "ctypes.wintypes": mock_ctypes.wintypes},
):
    _spec.loader.exec_module(_module)

_edge_map = _module._edge_map


class TestEdgeMapMath(unittest.TestCase):
    """验证 _edge_map 的数学性质。"""

    def test_boundary_preservation(self):
        """边界保持：edge_map(0, s) == 0，edge_map(1, s) == 1"""
        for s in [0.3, 0.5, 1.0]:
            with self.subTest(s=s):
                self.assertAlmostEqual(_edge_map(0.0, s), 0.0, places=10)
                self.assertAlmostEqual(_edge_map(1.0, s), 1.0, places=10)

    def test_center_invariance(self):
        """中心不变：edge_map(0.5, s) == 0.5"""
        for s in [0.0, 0.1, 0.3, 0.5, 0.8, 1.0]:
            with self.subTest(s=s):
                self.assertAlmostEqual(_edge_map(0.5, s), 0.5, places=10)

    def test_monotonicity(self):
        """单调递增：当 t1 < t2 时，edge_map(t1, s) <= edge_map(t2, s)；
        在非 clamp 区域内部严格递增。"""
        for s in [0.0, 0.1, 0.3, 0.5, 1.0]:
            with self.subTest(s=s):
                steps = [i / 100.0 for i in range(101)]
                for i in range(len(steps) - 1):
                    v1 = _edge_map(steps[i], s)
                    v2 = _edge_map(steps[i + 1], s)
                    # clamping may cause equality near boundaries, so use <=
                    self.assertLessEqual(v1, v2,
                        f"monotonicity broken at {steps[i]} vs {steps[i+1]} with s={s}")
                # In the interior [0.2, 0.8], function is not clamped and should be strictly increasing
                interior_steps = [i / 100.0 for i in range(20, 81)]
                for i in range(len(interior_steps) - 1):
                    v1 = _edge_map(interior_steps[i], s)
                    v2 = _edge_map(interior_steps[i + 1], s)
                    self.assertLess(v1, v2,
                        f"strict monotonicity broken in interior at {interior_steps[i]} vs {interior_steps[i+1]} with s={s}")

    def test_symmetry(self):
        """对称性：edge_map(0.5+u, s) + edge_map(0.5-u, s) == 1"""
        for s in [0.0, 0.1, 0.3, 0.5, 1.0]:
            with self.subTest(s=s):
                for i in range(1, 51):
                    u = i / 100.0
                    a = _edge_map(0.5 + u, s)
                    b = _edge_map(0.5 - u, s)
                    self.assertAlmostEqual(a + b, 1.0, places=10,
                        msg=f"symmetry broken at u={u}, s={s}")

    def test_strength_zero_is_linear(self):
        """strength=0 等价线性：edge_map(t, 0) == t"""
        for i in range(101):
            t = i / 100.0
            with self.subTest(t=t):
                self.assertAlmostEqual(_edge_map(t, 0.0), t, places=10)

    def test_edge_acceleration_effect(self):
        """边缘加速效果：strength=0.3 时，edge_map(0.8, 0.3) > 0.8，edge_map(0.2, 0.3) < 0.2"""
        s = 0.3
        self.assertGreater(_edge_map(0.8, s), 0.8)
        self.assertLess(_edge_map(0.2, s), 0.2)

    def test_boundary_clamp(self):
        """边界裁剪：strength=1.0 时，edge_map(0.1, 1.0) 应被 clamp 到 >= 0"""
        val = _edge_map(0.1, 1.0)
        self.assertGreaterEqual(val, 0.0)
        self.assertLessEqual(val, 1.0)

    def test_clamp_extreme_strength(self):
        """极端 strength 下仍保证输出在 [0,1]"""
        for s in [0.0, 0.5, 1.0]:
            with self.subTest(s=s):
                for t in [0.0, 0.01, 0.05, 0.1, 0.9, 0.95, 0.99, 1.0]:
                    val = _edge_map(t, s)
                    self.assertGreaterEqual(val, 0.0,
                        f"clamp lower failed at t={t}, s={s}")
                    self.assertLessEqual(val, 1.0,
                        f"clamp upper failed at t={t}, s={s}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
