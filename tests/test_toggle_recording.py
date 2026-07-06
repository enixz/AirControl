"""F5 热切换录制开关的单元测试。

验证 orchestrator.toggle_recording() 的状态机：
  - 未录 → 开始录制（创建 FrameRecorder，挂到 inference_worker）
  - 录中 → 停止录制（close 旧 recorder，清空引用）
  - 反复切换不崩、不漏关

FrameRecorder 用 MagicMock 替代，避免真实写盘。
"""
import os
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

_app_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'app')
sys.path.insert(0, _app_dir)

# 清除可能被前序测试污染的 PyQt6 / orchestrator 缓存（与 test_orchestrator_dictation 同套路）
for _m in list(sys.modules.keys()):
    if _m == 'PyQt6' or _m.startswith('PyQt6.'):
        del sys.modules[_m]

_app_module_names = ['orchestrator']
_saved_app_modules = {_m: sys.modules.get(_m) for _m in _app_module_names}
for _m in _app_module_names:
    sys.modules.pop(_m, None)

# Mock win32 / 系统模块（orchestrator 导入期需要）
_mock_win32con = types.ModuleType('win32con')
_mock_win32con.WS_MINIMIZE = 0x20000000
_mock_win32con.GWL_STYLE = -16
_mock_win32con.SW_RESTORE = 9
_mock_win32con.SW_SHOW = 5
_mock_win32con.VK_MENU = 0x12
_mock_win32con.VK_ESCAPE = 0x1B
_mock_win32con.KEYEVENTF_KEYUP = 0x0002

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


def _make_orchestrator():
    """构造一个跳过 init_services 的 orchestrator，配好 mock inference_worker。"""
    with patch('orchestrator.AirControlOrchestrator.init_services'), \
         patch('orchestrator.AirControlOrchestrator._init_modes'), \
         patch('orchestrator.AirControlOrchestrator.set_mode'), \
         patch('orchestrator.ConfigManager'), \
         patch('orchestrator.MouseController'):
        orch = AirControlOrchestrator(MagicMock(), MagicMock(), MagicMock())
    # 配置 mock：toggle_recording 读 config.get(...) 取 max_frames/max_seconds
    orch.config = MagicMock()
    orch.config.get = lambda k, d=None: {
        "record_raw_max_frames": 100,
        "record_raw_max_seconds": 10.0,
    }.get(k, d if d is not None else False)
    # mock inference_worker：热切换通过 set_frame_recorder 传给它
    orch.inference_worker = MagicMock()
    orch.inference_worker.set_frame_recorder = MagicMock()
    return orch


class TestToggleRecording(unittest.TestCase):
    """F5 录制开关状态机。"""

    @patch('services.frame_recorder.FrameRecorder')
    def test_start_when_not_recording(self, mock_fr_cls):
        """未录 → toggle 后开始录制，返回 True；recorder 挂到 inference_worker。"""
        orch = _make_orchestrator()
        self.assertFalse(orch.is_recording())

        mock_recorder = MagicMock()
        mock_recorder.dir = "/tmp/test_capture"
        mock_fr_cls.return_value = mock_recorder

        now_recording, path = orch.toggle_recording()

        self.assertTrue(now_recording)
        self.assertEqual(path, "/tmp/test_capture")
        self.assertTrue(orch.is_recording())
        # recorder 已挂到 worker
        orch.inference_worker.set_frame_recorder.assert_called_once_with(mock_recorder)
        # FrameRecorder 已实例化
        mock_fr_cls.assert_called_once()

    @patch('services.frame_recorder.FrameRecorder')
    def test_stop_when_recording(self, mock_fr_cls):
        """录中 → toggle 后停止，close 被调用，引用清空，worker 收到 None。"""
        orch = _make_orchestrator()
        mock_recorder = MagicMock()
        mock_recorder.dir = "/tmp/test_capture"
        mock_fr_cls.return_value = mock_recorder

        # 先开始
        orch.toggle_recording()
        self.assertTrue(orch.is_recording())

        # 再停止
        now_recording, path = orch.toggle_recording()

        self.assertFalse(now_recording)
        self.assertEqual(path, "/tmp/test_capture")
        self.assertFalse(orch.is_recording())
        mock_recorder.close.assert_called_once()
        # worker 第二次收到 None
        self.assertEqual(orch.inference_worker.set_frame_recorder.call_count, 2)
        orch.inference_worker.set_frame_recorder.assert_called_with(None)

    @patch('services.frame_recorder.FrameRecorder')
    def test_toggle_cycle_multiple_times(self, mock_fr_cls):
        """反复按 F5 多次：开始→停止→开始→停止，不崩、状态正确。"""
        orch = _make_orchestrator()
        mock_recorder = MagicMock()
        mock_recorder.dir = "/tmp/test_capture"
        mock_fr_cls.return_value = mock_recorder

        for i in range(3):
            # 开始
            now_recording, _ = orch.toggle_recording()
            self.assertTrue(now_recording, f"第{i+1}次开始应返回 True")
            self.assertTrue(orch.is_recording())
            # 停止
            now_recording, _ = orch.toggle_recording()
            self.assertFalse(now_recording, f"第{i+1}次停止应返回 False")
            self.assertFalse(orch.is_recording())

        # 6 次 toggle = 3 次创建 + 3 次 close
        self.assertEqual(mock_fr_cls.call_count, 3)

    @patch('services.frame_recorder.FrameRecorder', side_effect=Exception("disk full"))
    def test_start_failure_returns_false(self, mock_fr_cls):
        """FrameRecorder 构造失败 → 返回 (False, None)，is_recording 保持 False。"""
        orch = _make_orchestrator()
        self.assertFalse(orch.is_recording())

        now_recording, path = orch.toggle_recording()

        self.assertFalse(now_recording)
        self.assertIsNone(path)
        self.assertFalse(orch.is_recording())
        # 失败时不应该调 worker.set_frame_recorder
        orch.inference_worker.set_frame_recorder.assert_not_called()

    @patch('services.frame_recorder.FrameRecorder')
    def test_stop_failure_still_clears_reference(self, mock_fr_cls):
        """停止时 close() 抛异常 → 仍清空引用，避免卡在录制状态。"""
        orch = _make_orchestrator()
        mock_recorder = MagicMock()
        mock_recorder.dir = "/tmp/test_capture"
        mock_recorder.close.side_effect = Exception("io error")
        mock_fr_cls.return_value = mock_recorder

        orch.toggle_recording()
        self.assertTrue(orch.is_recording())

        # close 抛异常，但 toggle 应吞掉并清状态
        now_recording, path = orch.toggle_recording()

        self.assertFalse(now_recording)
        self.assertFalse(orch.is_recording())
        # 仍尝试通知 worker
        orch.inference_worker.set_frame_recorder.assert_called_with(None)


if __name__ == "__main__":
    unittest.main()
