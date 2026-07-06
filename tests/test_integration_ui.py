"""
T4: 集成回归测试 — py_compile 验证全部 app 模块语法。
T5: 行为测试 — 验证配置校验、模式循环、动作分发等核心行为。
"""
import os
import py_compile
import unittest

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
APP_DIR = os.path.join(BASE_DIR, 'app')


class T4IntegrationRegression(unittest.TestCase):
    """集成回归测试：验证全部 app 模块无语法错误。"""

    def test_py_compile_all_app_modules(self):
        """编译 app/ 目录下所有 .py 文件，确保无语法错误。"""
        app_files = []
        for root, _dirs, files in os.walk(APP_DIR):
            for fname in files:
                if fname.endswith('.py'):
                    app_files.append(os.path.join(root, fname))
        self.assertGreater(len(app_files), 5, "应至少有 5 个 app 模块文件")
        for full in app_files:
            rel = os.path.relpath(full, BASE_DIR)
            with self.subTest(file=rel):
                try:
                    py_compile.compile(full, doraise=True)
                except py_compile.PyCompileError as e:
                    self.fail(f"语法错误 {rel}: {e}")


class T5ConfigValidationBehavior(unittest.TestCase):
    """行为测试：ConfigManager 的 interaction_mode 校验。"""

    @classmethod
    def setUpClass(cls):
        import sys
        sys.path.insert(0, APP_DIR)
        from config_manager import ConfigManager, _validate_config
        from modes import MODE_NAMES
        cls.ConfigManager = ConfigManager
        cls._validate_config = staticmethod(_validate_config)
        cls.MODE_NAMES = MODE_NAMES

    def test_valid_modes_accepted(self):
        """所有 MODE_NAMES 中的模式名应通过校验。"""
        for mode in self.MODE_NAMES:
            with self.subTest(mode=mode):
                cfg = {"interaction_mode": mode}
                cfg, warnings = self._validate_config(cfg)
                self.assertEqual(cfg["interaction_mode"], mode)
                self.assertEqual(warnings, [])

    def test_invalid_mode_rejected(self):
        """无效模式名应被回退为默认值 "mouse"。"""
        cfg = {"interaction_mode": "invalid_mode_xyz"}
        cfg, warnings = self._validate_config(cfg)
        self.assertEqual(cfg["interaction_mode"], "mouse")
        self.assertTrue(any("interaction_mode" in w for w in warnings),
                        f"应产生校验警告: {warnings}")


class T5ModeCyclingBehavior(unittest.TestCase):
    """行为测试：ModeManager.cycle_mode() 的模式循环。"""

    @classmethod
    def setUpClass(cls):
        import sys
        from unittest.mock import MagicMock
        sys.path.insert(0, APP_DIR)
        from modes import MODE_NAMES
        cls.MODE_NAMES = MODE_NAMES
        cls.MagicMock = MagicMock

    def test_cycle_visits_all_modes_in_order(self):
        """连续 cycle_mode 应按 MODE_NAMES 顺序遍历所有模式。"""
        from mode_manager import ModeManager

        modes = {name: self.MagicMock() for name in self.MODE_NAMES}
        config = self.MagicMock()
        config.get.return_value = 1.0
        config.batch_update.return_value.__enter__ = self.MagicMock()
        config.batch_update.return_value.__exit__ = self.MagicMock(return_value=False)

        mm = ModeManager(modes, config, recognizer=self.MagicMock())
        mm.switch_to(self.MODE_NAMES[0])

        visited = []
        for _ in range(len(self.MODE_NAMES)):
            mm.cycle_mode()
            visited.append(mm.current_mode_name)

        expected = list(self.MODE_NAMES[1:]) + [self.MODE_NAMES[0]]
        self.assertEqual(visited, expected,
                         f"循环切换应按顺序遍历: {expected}, 实际: {visited}")


class T5ExecuteActionBehavior(unittest.TestCase):
    """行为测试：Orchestrator.execute_action() 的动作分发。"""

    @classmethod
    def setUpClass(cls):
        import sys
        sys.path.insert(0, APP_DIR)
        # 快照整个 sys.modules，tearDownClass 中精确还原，避免本类对 orchestrator/
        # mode_manager 的重导入与 win32 mock 泄漏到后续测试（如 test_mode_switch_ily
        # 依赖 mode_manager 模块身份做 time patch）。
        cls._sys_modules_before = dict(sys.modules)
        # 清除可能被其他测试 mock 的 PyQt6
        for _m in list(sys.modules.keys()):
            if _m == 'PyQt6' or _m.startswith('PyQt6.'):
                del sys.modules[_m]
        for _m in ['orchestrator', 'mode_manager', 'modes', 'modes.base']:
            sys.modules.pop(_m, None)

        import types
        from unittest.mock import MagicMock
        mock_win32con = types.ModuleType('win32con')
        mock_win32con.WS_MINIMIZE = 0x20000000
        mock_win32con.GWL_STYLE = -16
        mock_win32con.SW_RESTORE = 9
        mock_win32con.SW_SHOW = 5
        mock_win32con.VK_MENU = 0x12
        mock_win32con.VK_ESCAPE = 0x1B
        mock_win32con.KEYEVENTF_KEYUP = 0x0002
        sys.modules['win32con'] = mock_win32con
        for mod_name in ['win32api', 'win32gui', 'win32process', 'psutil', 'winreg', 'winsound']:
            sys.modules[mod_name] = MagicMock()

        from modes import MODE_NAMES
        from orchestrator import AirControlOrchestrator
        cls.Orchestrator = AirControlOrchestrator
        cls.MODE_NAMES = MODE_NAMES

    @classmethod
    def tearDownClass(cls):
        """精确还原 setUpClass 之前的 sys.modules，防止重导入模块/win32 mock 泄漏。"""
        import sys
        snapshot = cls._sys_modules_before
        for key in list(sys.modules.keys()):
            if key not in snapshot:
                del sys.modules[key]
        sys.modules.update(snapshot)
        super().tearDownClass()

    def test_switch_to_mode_actions(self):
        """switch_to_* 动作应正确切换到对应模式。"""
        from unittest.mock import MagicMock, patch

        with patch.object(self.Orchestrator, 'init_services'), \
             patch.object(self.Orchestrator, '_init_modes'), \
             patch.object(self.Orchestrator, 'set_mode') as mock_set_mode, \
             patch('orchestrator.ConfigManager'), \
             patch('orchestrator.MouseController'):
            orch = self.Orchestrator(MagicMock(), MagicMock(), MagicMock())

            action_to_mode = {
                "switch_to_presentation": "presentation",
                "switch_to_mouse": "mouse",
                "switch_to_draw": "draw",
            }
            for action, expected_mode in action_to_mode.items():
                with self.subTest(action=action):
                    mock_set_mode.reset_mock()
                    orch.execute_action(action)
                    mock_set_mode.assert_called_once_with(expected_mode)

    def test_ppt_actions_dispatched(self):
        """PPT 动作应正确委托给 ppt 控制器。"""
        from unittest.mock import MagicMock, patch

        with patch.object(self.Orchestrator, 'init_services'), \
             patch.object(self.Orchestrator, '_init_modes'), \
             patch.object(self.Orchestrator, 'set_mode'), \
             patch('orchestrator.ConfigManager'), \
             patch('orchestrator.MouseController'):
            orch = self.Orchestrator(MagicMock(), MagicMock(), MagicMock())
            orch.ppt = MagicMock()

            ppt_actions = {
                "next_slide": "next_slide",
                "prev_slide": "prev_slide",
                "start_presentation": "start_presentation",
                "end_presentation": "end_presentation",
                "switch_app": "switch_app",
            }
            for action, method_name in ppt_actions.items():
                with self.subTest(action=action):
                    getattr(orch.ppt, method_name).reset_mock()
                    orch.execute_action(action)
                    getattr(orch.ppt, method_name).assert_called_once()

    def test_mouse_actions_dispatched(self):
        """鼠标动作应正确委托给 mouse 控制器。"""
        from unittest.mock import MagicMock, patch

        with patch.object(self.Orchestrator, 'init_services'), \
             patch.object(self.Orchestrator, '_init_modes'), \
             patch.object(self.Orchestrator, 'set_mode'), \
             patch('orchestrator.ConfigManager'), \
             patch('orchestrator.MouseController'):
            orch = self.Orchestrator(MagicMock(), MagicMock(), MagicMock())
            orch.mouse = MagicMock()

            mouse_actions = {
                "left_click": "left_click",
                "double_click": "double_click",
                "right_click": "right_click",
            }
            for action, method_name in mouse_actions.items():
                with self.subTest(action=action):
                    getattr(orch.mouse, method_name).reset_mock()
                    orch.execute_action(action)
                    getattr(orch.mouse, method_name).assert_called_once()

    def test_unknown_action_is_noop(self):
        """未知动作名不应抛异常（静默忽略）。"""
        from unittest.mock import MagicMock, patch

        with patch.object(self.Orchestrator, 'init_services'), \
             patch.object(self.Orchestrator, '_init_modes'), \
             patch.object(self.Orchestrator, 'set_mode'), \
             patch('orchestrator.ConfigManager'), \
             patch('orchestrator.MouseController'):
            orch = self.Orchestrator(MagicMock(), MagicMock(), MagicMock())
            # 不应抛异常
            orch.execute_action("nonexistent_action")


if __name__ == "__main__":
    unittest.main(verbosity=2)
