"""
Tests for AirControlOrchestrator dictation and caption signals.
"""
import os
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

# Ensure app is on sys.path
_app_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'app')
sys.path.insert(0, _app_dir)

# 无条件清除被其他测试 mock 的 PyQt6 模块，确保本测试拿到真实的 Qt 绑定
# （AirControlOrchestrator 继承 QObject，需要真实的 pyqtSignal 才能工作）
for _m in list(sys.modules.keys()):
    if _m == 'PyQt6' or _m.startswith('PyQt6.'):
        del sys.modules[_m]

# 清除可能被前序测试污染的 orchestrator 模块缓存（导入后会还原 app 子模块，见下方）
_app_module_names = ['orchestrator', 'mode_manager', 'modes', 'modes.base']
_saved_app_modules = {_m: sys.modules.get(_m) for _m in _app_module_names}
for _m in _app_module_names:
    sys.modules.pop(_m, None)

# Mock win32 模块（orchestrator.py 的服务依赖在导入时需要这些）
_mock_win32con = types.ModuleType('win32con')
_mock_win32con.WS_MINIMIZE = 0x20000000
_mock_win32con.GWL_STYLE = -16
_mock_win32con.SW_RESTORE = 9
_mock_win32con.SW_SHOW = 5
_mock_win32con.VK_MENU = 0x12
_mock_win32con.VK_ESCAPE = 0x1B
_mock_win32con.KEYEVENTF_KEYUP = 0x0002

# 保存被替换的原始模块，tearDownClass 中恢复
_saved_modules = {}
_win32_mocks = {
    'win32con': _mock_win32con,
    'win32api': MagicMock(),
    'win32gui': MagicMock(),
    'win32process': MagicMock(),
    'psutil': MagicMock(),
    'winreg': MagicMock(),
    'winsound': MagicMock(),
}
for _name, _mock in _win32_mocks.items():
    _saved_modules[_name] = sys.modules.get(_name)
    sys.modules[_name] = _mock

from orchestrator import AirControlOrchestrator

# 关键：pytest 先 collection 所有测试模块再执行。上面在 collection 期把 mode_manager
# 重导入为新模块对象，若不还原，会令 test_mode_switch_ily（其 ModeManager 绑定自原
# 模块）的 mock.patch("mode_manager.time") 落空。这里把 mode_manager/modes 还原为其它
# 测试 collection 期绑定的原模块；仅保留本次重导入的 orchestrator 作为本模块 patch 目标。
for _name in ('mode_manager', 'modes', 'modes.base'):
    _orig = _saved_app_modules.get(_name)
    if _orig is not None:
        sys.modules[_name] = _orig
    else:
        sys.modules.pop(_name, None)


class TestOrchestratorDictation(unittest.TestCase):
    @patch('orchestrator.AirControlOrchestrator.init_services')
    @patch('orchestrator.AirControlOrchestrator._init_modes')
    @patch('orchestrator.AirControlOrchestrator.set_mode')
    @patch('orchestrator.ConfigManager')
    @patch('orchestrator.MouseController')
    def test_dictation_signals(self, mock_mouse_cls, mock_config_cls, mock_set_mode, mock_init_modes, mock_init_services):
        # Create dummy overlay, cursor_overlay, toolbar
        overlay = MagicMock()
        cursor_overlay = MagicMock()
        toolbar = MagicMock()

        # Instantiate orchestrator
        orchestrator = AirControlOrchestrator(overlay, cursor_overlay, toolbar)

        # Verify public signals are defined
        self.assertTrue(hasattr(orchestrator, 'dictation_status_signal'))
        self.assertTrue(hasattr(orchestrator, 'dictation_text_signal'))
        self.assertTrue(hasattr(orchestrator, 'dictation_partial_signal'))
        self.assertTrue(hasattr(orchestrator, 'caption_full'))

        # Set up signal spies
        status_spy = MagicMock()
        text_spy = MagicMock()
        partial_spy = MagicMock()
        caption_full_spy = MagicMock()

        orchestrator.dictation_status_signal.connect(status_spy)
        orchestrator.dictation_text_signal.connect(text_spy)
        orchestrator.dictation_partial_signal.connect(partial_spy)
        orchestrator.caption_full.connect(caption_full_spy)

        # Emit internal signals and check if public signals are emitted with correct arguments
        orchestrator._dictation_status_signal.emit("started", "payload")
        status_spy.assert_called_once_with("started", "payload")

        orchestrator._dictation_text_signal.emit("final text", (100, 200))
        text_spy.assert_called_once_with("final text", (100, 200))

        orchestrator._dictation_partial_signal.emit("partial text")
        partial_spy.assert_called_once_with("partial text")

        # Test caption_full signal emission via _on_caption_full slot
        orchestrator.voice_command = MagicMock()
        orchestrator.voice_command.is_dictating = True
        orchestrator._on_caption_full()
        orchestrator.voice_command.stop_dictation.assert_called_once()
        caption_full_spy.assert_called_once()

    @classmethod
    def tearDownClass(cls):
        """恢复被本模块 mock 的 win32 模块，避免污染后续测试。"""
        for name, original in _saved_modules.items():
            if original is not None:
                sys.modules[name] = original
            else:
                sys.modules.pop(name, None)
        super().tearDownClass()


if __name__ == "__main__":
    unittest.main(verbosity=2)
