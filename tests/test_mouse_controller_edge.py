"""
T2: MouseController 向后兼容性单元测试
验证默认构造、旧式调用、to_screen 行为、热更新、move_to_normalized 委托。
"""
import sys
import types
import unittest
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from unittest.mock import patch

# Mock ctypes / Windows API before importing mouse_controller
mock_ctypes = types.ModuleType('ctypes')
mock_ctypes.wintypes = types.ModuleType('ctypes.wintypes')
mock_windll = types.ModuleType('ctypes.windll')
mock_user32 = types.ModuleType('user32')
FAKE_SCREEN_W = 1920
FAKE_SCREEN_H = 1080

# SM_CXVIRTUALSCREEN=78, SM_CYVIRTUALSCREEN=79 也需要 mock（多显示器支持）
_VIRTUAL_METRICS = {0: FAKE_SCREEN_W, 1: FAKE_SCREEN_H, 78: FAKE_SCREEN_W, 79: FAKE_SCREEN_H}
mock_user32.GetSystemMetrics = lambda idx: _VIRTUAL_METRICS.get(idx, FAKE_SCREEN_H)
mock_user32.GetCursorPos = lambda pt: None
mock_user32.SetCursorPos = lambda x, y: None
mock_user32.mouse_event = lambda *a: None
mock_windll.user32 = mock_user32
mock_ctypes.windll = mock_windll
mock_ctypes.byref = lambda x: x
mock_ctypes.wintypes.POINT = lambda: type('POINT', (), {'x': 0, 'y': 0})()

_module_path = Path(__file__).resolve().parents[1] / "app" / "services" / "mouse_controller.py"
_spec = spec_from_file_location("edge_mouse_controller_under_test", _module_path)
_mc_module = module_from_spec(_spec)
with patch.dict(
    sys.modules,
    {"ctypes": mock_ctypes, "ctypes.wintypes": mock_ctypes.wintypes},
):
    _spec.loader.exec_module(_mc_module)

MouseController = _mc_module.MouseController


class TestMouseControllerEdge(unittest.TestCase):
    """验证 MouseController 的向后兼容性与边缘加速功能。"""

    def test_default_construction(self):
        """默认参数构造不报错，edge_enabled=False，edge_strength=30/100=0.3"""
        mc = MouseController()
        self.assertFalse(mc.edge_enabled)
        self.assertEqual(mc.edge_strength, 0.3)

    def test_old_style_construction(self):
        """旧式调用兼容：MouseController(sensitivity=50) 仍可正常构造"""
        mc = MouseController(sensitivity=50)
        self.assertFalse(mc.edge_enabled)
        self.assertEqual(mc.sensitivity, 0.5)

    def test_to_screen_disabled(self):
        """to_screen 关闭时行为：mc.to_screen(0.5, 0.5) 返回 (screen_w//2, screen_h//2)（近似）"""
        mc = MouseController(edge_enabled=False)
        x, y = mc.to_screen(0.5, 0.5)
        self.assertEqual(x, FAKE_SCREEN_W // 2)
        self.assertEqual(y, FAKE_SCREEN_H // 2)

    def test_to_screen_enabled(self):
        """to_screen 开启时行为：to_screen(0.8, 0.8) 返回值应大于 (0.8*screen_w, 0.8*screen_h)"""
        mc = MouseController(edge_enabled=True, edge_strength=50)
        x, y = mc.to_screen(0.8, 0.8)
        base_x = int(0.8 * FAKE_SCREEN_W)
        base_y = int(0.8 * FAKE_SCREEN_H)
        self.assertGreater(x, base_x, f"expected > {base_x}, got {x}")
        self.assertGreater(y, base_y, f"expected > {base_y}, got {y}")

    def test_set_edge_acceleration_hot_update(self):
        """set_edge_acceleration 热更新：运行时切换 enabled/strength 后，to_screen 结果相应变化"""
        mc = MouseController(edge_enabled=False)
        x_off, y_off = mc.to_screen(0.8, 0.8)
        base_x = int(0.8 * FAKE_SCREEN_W)
        self.assertEqual(x_off, base_x)

        mc.set_edge_acceleration(True, 50)
        x_on, y_on = mc.to_screen(0.8, 0.8)
        self.assertGreater(x_on, base_x)

        mc.set_edge_acceleration(False, 50)
        x_off2, _ = mc.to_screen(0.8, 0.8)
        self.assertEqual(x_off2, base_x)

    def test_move_to_normalized_delegates_to_screen(self):
        """move_to_normalized 不再重复计算：检查其内部是否调用了 to_screen()"""
        mc = MouseController(edge_enabled=True, edge_strength=50)
        # Patch to_screen to record calls
        original_to_screen = mc.to_screen
        calls = []

        def patched_to_screen(x, y):
            calls.append((x, y))
            return original_to_screen(x, y)

        mc.to_screen = patched_to_screen
        mc.move_to_normalized(0.3, 0.4)
        self.assertEqual(len(calls), 1, f"to_screen should be called exactly once, called {len(calls)} times")
        self.assertEqual(calls[0], (0.3, 0.4))

    def test_edge_strength_clamping(self):
        """edge_strength 超出范围应被 clamp 到 [0,100]"""
        mc = MouseController(edge_enabled=True, edge_strength=-10)
        self.assertEqual(mc.edge_strength, 0.0)
        mc.set_edge_acceleration(True, 150)
        self.assertEqual(mc.edge_strength, 1.0)

    def test_to_screen_edge_values(self):
        """to_screen 对边界值处理正确"""
        mc = MouseController(edge_enabled=True, edge_strength=100)
        self.assertEqual(mc.to_screen(0.0, 0.0), (0, 0))
        self.assertEqual(mc.to_screen(1.0, 1.0), (FAKE_SCREEN_W, FAKE_SCREEN_H))


if __name__ == "__main__":
    unittest.main(verbosity=2)
