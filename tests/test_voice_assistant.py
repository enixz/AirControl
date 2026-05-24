"""
Unit tests for VoiceAssistantService focusing on the focus restoration bug fix.
Tests the _restore_aircontrol_focus method and the try/finally patterns in activate() and hang_up().
"""
import unittest
from unittest.mock import patch, MagicMock, call
import sys
import os

# Add app directory to path
_app_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'app')
sys.path.insert(0, _app_dir)
sys.path.insert(0, os.path.join(_app_dir, 'services'))

# Mock ctypes / Windows API before importing mouse_controller
import types
mock_ctypes = types.ModuleType('ctypes')
mock_ctypes.wintypes = types.ModuleType('ctypes.wintypes')
mock_windll = types.ModuleType('ctypes.windll')
mock_user32 = types.ModuleType('user32')
mock_user32.GetSystemMetrics = lambda idx: 1920 if idx in (0, 78) else 1080
mock_user32.GetCursorPos = lambda pt: None
mock_user32.SetCursorPos = lambda x, y: None
mock_user32.mouse_event = lambda *a: None
mock_windll.user32 = mock_user32
mock_ctypes.windll = mock_windll
mock_ctypes.byref = lambda x: x
mock_ctypes.wintypes.POINT = lambda: type('POINT', (), {'x': 0, 'y': 0})()

sys.modules['ctypes'] = mock_ctypes
sys.modules['ctypes.wintypes'] = mock_ctypes.wintypes

# Create a proper win32con mock with integer constants
mock_win32con = types.ModuleType('win32con')
mock_win32con.WS_MINIMIZE = 0x20000000
mock_win32con.GWL_STYLE = -16
mock_win32con.SW_RESTORE = 9
mock_win32con.SW_SHOW = 5
mock_win32con.VK_MENU = 0x12
mock_win32con.VK_ESCAPE = 0x1B
mock_win32con.KEYEVENTF_KEYUP = 0x0002
sys.modules['win32con'] = mock_win32con

for mod_name in ['win32api', 'win32gui', 'win32process', 'psutil', 'winreg']:
    sys.modules[mod_name] = MagicMock()

from voice_assistant import VoiceAssistantService, _bring_to_front


class TestVoiceAssistantService(unittest.TestCase):
    """Test VoiceAssistantService with focus on the bug fix."""

    def test_init_sets_aircontrol_hwnd_none(self):
        """__init__ should initialize aircontrol_hwnd to None."""
        svc = VoiceAssistantService()
        self.assertIsNone(svc.aircontrol_hwnd)

    def test_init_with_assistant(self):
        """__init__ should accept an assistant parameter."""
        svc = VoiceAssistantService(assistant="qianwen")
        self.assertEqual(svc.assistant, "qianwen")
        self.assertIsNone(svc.aircontrol_hwnd)

    def test_restore_aircontrol_focus_when_hwnd_set(self):
        """_restore_aircontrol_focus should call _bring_to_front with the hwnd."""
        svc = VoiceAssistantService()
        svc.aircontrol_hwnd = 12345

        with patch('voice_assistant._bring_to_front') as mock_bring:
            svc._restore_aircontrol_focus()
            mock_bring.assert_called_once_with(12345)

    def test_restore_aircontrol_focus_when_hwnd_none(self):
        """_restore_aircontrol_focus should do nothing if aircontrol_hwnd is None."""
        svc = VoiceAssistantService()
        svc.aircontrol_hwnd = None

        with patch('voice_assistant._bring_to_front') as mock_bring:
            svc._restore_aircontrol_focus()
            mock_bring.assert_not_called()

    def test_restore_aircontrol_focus_when_hwnd_zero(self):
        """_restore_aircontrol_focus should do nothing if aircontrol_hwnd is 0 (falsy)."""
        svc = VoiceAssistantService()
        svc.aircontrol_hwnd = 0

        with patch('voice_assistant._bring_to_front') as mock_bring:
            svc._restore_aircontrol_focus()
            mock_bring.assert_not_called()

    def test_restore_aircontrol_focus_handles_exception(self):
        """_restore_aircontrol_focus should not raise if _bring_to_front raises."""
        svc = VoiceAssistantService()
        svc.aircontrol_hwnd = 12345

        with patch('voice_assistant._bring_to_front', side_effect=Exception("win32 error")):
            # Should not raise
            svc._restore_aircontrol_focus()

    def test_activate_calls_restore_focus_on_success(self):
        """activate() should call _restore_aircontrol_focus in finally block even on success."""
        svc = VoiceAssistantService()
        svc.aircontrol_hwnd = 12345

        with patch('voice_assistant._find_and_focus', return_value=True), \
             patch('voice_assistant._press_esc'), \
             patch('voice_assistant._send_hotkey'), \
             patch('voice_assistant.time'), \
             patch.object(svc, '_restore_aircontrol_focus') as mock_restore:
            svc.activate()
            mock_restore.assert_called_once()

    def test_activate_calls_restore_focus_on_failure(self):
        """activate() should call _restore_aircontrol_focus in finally block even when finding fails."""
        svc = VoiceAssistantService()
        svc.aircontrol_hwnd = 12345

        with patch('voice_assistant._find_and_focus', return_value=False), \
             patch.object(svc, '_restore_aircontrol_focus') as mock_restore:
            result = svc.activate()
            self.assertFalse(result)
            mock_restore.assert_called_once()

    def test_hang_up_calls_restore_focus_on_success(self):
        """hang_up() should call _restore_aircontrol_focus in finally block on success."""
        svc = VoiceAssistantService()
        svc.aircontrol_hwnd = 12345

        import win32gui
        win32gui.IsWindow.return_value = True

        with patch('voice_assistant._is_process_running', return_value=True), \
             patch('voice_assistant._enum_all_process_windows', return_value=[(100, "Test Window")]), \
             patch('voice_assistant._bring_to_front'), \
             patch('voice_assistant._send_hotkey'), \
             patch('voice_assistant._minimize_window'), \
             patch('voice_assistant.time'), \
             patch.object(svc, '_restore_aircontrol_focus') as mock_restore:
            svc.hang_up()
            mock_restore.assert_called_once()

    def test_hang_up_calls_restore_focus_when_not_running(self):
        """hang_up() should call _restore_aircontrol_focus in finally block even if not running."""
        svc = VoiceAssistantService()
        svc.aircontrol_hwnd = 12345

        with patch('voice_assistant._is_process_running', return_value=False), \
             patch.object(svc, '_restore_aircontrol_focus') as mock_restore:
            result = svc.hang_up()
            self.assertFalse(result)
            mock_restore.assert_called_once()

    def test_hang_up_calls_restore_focus_when_no_windows(self):
        """hang_up() should call _restore_aircontrol_focus in finally block even when no windows found."""
        svc = VoiceAssistantService()
        svc.aircontrol_hwnd = 12345

        with patch('voice_assistant._is_process_running', return_value=True), \
             patch('voice_assistant._enum_all_process_windows', return_value=[]), \
             patch.object(svc, '_restore_aircontrol_focus') as mock_restore:
            result = svc.hang_up()
            self.assertFalse(result)
            mock_restore.assert_called_once()

    def test_activate_try_finally_structure(self):
        """Verify activate() has proper try/finally by checking _restore is called on exception."""
        svc = VoiceAssistantService()
        svc.aircontrol_hwnd = 12345

        with patch('voice_assistant._find_and_focus', side_effect=Exception("unexpected")), \
             patch.object(svc, '_restore_aircontrol_focus') as mock_restore:
            with self.assertRaises(Exception):
                svc.activate()
            mock_restore.assert_called_once()

    def test_hang_up_try_finally_structure(self):
        """Verify hang_up() has proper try/finally by checking _restore is called on exception."""
        svc = VoiceAssistantService()
        svc.aircontrol_hwnd = 12345

        with patch('voice_assistant._is_process_running', side_effect=Exception("unexpected")), \
             patch.object(svc, '_restore_aircontrol_focus') as mock_restore:
            with self.assertRaises(Exception):
                svc.hang_up()
            mock_restore.assert_called_once()

    def test_activate_returns_true_on_success(self):
        """activate() should return True on successful activation."""
        svc = VoiceAssistantService()
        svc.aircontrol_hwnd = 12345

        with patch('voice_assistant._find_and_focus', return_value=True), \
             patch('voice_assistant._press_esc'), \
             patch('voice_assistant._send_hotkey'), \
             patch('voice_assistant.time'), \
             patch.object(svc, '_restore_aircontrol_focus'):
            result = svc.activate()
            self.assertTrue(result)

    def test_activate_returns_false_when_exe_not_found(self):
        """activate() should return False when assistant exe is not found."""
        svc = VoiceAssistantService()
        svc.aircontrol_hwnd = 12345

        with patch('voice_assistant._find_and_focus', return_value=None), \
             patch('voice_assistant._find_exe', return_value=None), \
             patch('voice_assistant.time'), \
             patch.object(svc, '_restore_aircontrol_focus'):
            result = svc.activate()
            self.assertFalse(result)

    def test_hang_up_returns_true_on_success(self):
        """hang_up() should return True on successful hang up."""
        svc = VoiceAssistantService()
        svc.aircontrol_hwnd = 12345

        import win32gui
        win32gui.IsWindow.return_value = True

        with patch('voice_assistant._is_process_running', return_value=True), \
             patch('voice_assistant._enum_all_process_windows', return_value=[(100, "Test Window")]), \
             patch('voice_assistant._bring_to_front'), \
             patch('voice_assistant._send_hotkey'), \
             patch('voice_assistant._minimize_window'), \
             patch('voice_assistant.time'), \
             patch.object(svc, '_restore_aircontrol_focus'):
            result = svc.hang_up()
            self.assertTrue(result)

    def test_restore_focus_called_regardless_of_activate_outcome(self):
        """_restore_aircontrol_focus must always be called in activate(), regardless of exit path."""
        svc = VoiceAssistantService()
        svc.aircontrol_hwnd = 12345

        # Case 1: result=True (success)
        with patch('voice_assistant._find_and_focus', return_value=True), \
             patch('voice_assistant._press_esc'), \
             patch('voice_assistant._send_hotkey'), \
             patch('voice_assistant.time'), \
             patch.object(svc, '_restore_aircontrol_focus') as mock_restore:
            svc.activate()
            mock_restore.assert_called_once()

        # Case 2: result=False (failure)
        with patch('voice_assistant._find_and_focus', return_value=False), \
             patch.object(svc, '_restore_aircontrol_focus') as mock_restore:
            svc.activate()
            mock_restore.assert_called_once()

        # Case 3: Exception raised
        with patch('voice_assistant._find_and_focus', side_effect=RuntimeError("boom")), \
             patch.object(svc, '_restore_aircontrol_focus') as mock_restore:
            with self.assertRaises(RuntimeError):
                svc.activate()
            mock_restore.assert_called_once()

    def test_restore_focus_called_regardless_of_hang_up_outcome(self):
        """_restore_aircontrol_focus must always be called in hang_up(), regardless of exit path."""
        svc = VoiceAssistantService()
        svc.aircontrol_hwnd = 12345

        # Case 1: not running
        with patch('voice_assistant._is_process_running', return_value=False), \
             patch.object(svc, '_restore_aircontrol_focus') as mock_restore:
            svc.hang_up()
            mock_restore.assert_called_once()

        # Case 2: no windows
        with patch('voice_assistant._is_process_running', return_value=True), \
             patch('voice_assistant._enum_all_process_windows', return_value=[]), \
             patch.object(svc, '_restore_aircontrol_focus') as mock_restore:
            svc.hang_up()
            mock_restore.assert_called_once()

        # Case 3: Exception raised
        with patch('voice_assistant._is_process_running', side_effect=RuntimeError("boom")), \
             patch.object(svc, '_restore_aircontrol_focus') as mock_restore:
            with self.assertRaises(RuntimeError):
                svc.hang_up()
            mock_restore.assert_called_once()


class TestBringToFront(unittest.TestCase):
    """Test _bring_to_front helper function."""

    @patch('voice_assistant.win32gui')
    @patch('voice_assistant.win32con')
    def test_bring_to_front_restores_minimized(self, mock_win32con, mock_win32gui):
        """Should call SW_RESTORE for minimized windows."""
        mock_win32con.WS_MINIMIZE = 0x20000000
        mock_win32con.GWL_STYLE = -16
        mock_win32con.SW_RESTORE = 9
        mock_win32gui.GetWindowLong.return_value = 0x20000000  # WS_MINIMIZE flag set

        _bring_to_front(12345)

        mock_win32gui.ShowWindow.assert_called_with(12345, mock_win32con.SW_RESTORE)
        mock_win32gui.SetForegroundWindow.assert_called_with(12345)

    @patch('voice_assistant.win32gui')
    @patch('voice_assistant.win32con')
    def test_bring_to_front_shows_invisible(self, mock_win32con, mock_win32gui):
        """Should call SW_SHOW for invisible, non-minimized windows."""
        mock_win32con.WS_MINIMIZE = 0x20000000
        mock_win32con.GWL_STYLE = -16
        mock_win32con.SW_SHOW = 5
        mock_win32gui.GetWindowLong.return_value = 0  # Not minimized
        mock_win32gui.IsWindowVisible.return_value = False

        _bring_to_front(12345)

        mock_win32gui.ShowWindow.assert_called_with(12345, mock_win32con.SW_SHOW)
        mock_win32gui.SetForegroundWindow.assert_called_with(12345)

    @patch('voice_assistant.win32gui')
    @patch('voice_assistant.win32con')
    def test_bring_to_front_skips_show_for_visible_window(self, mock_win32con, mock_win32gui):
        """Visible, non-minimized windows should only call SetForegroundWindow."""
        mock_win32con.WS_MINIMIZE = 0x20000000
        mock_win32con.GWL_STYLE = -16
        mock_win32gui.GetWindowLong.return_value = 0  # Not minimized
        mock_win32gui.IsWindowVisible.return_value = True

        _bring_to_front(12345)

        mock_win32gui.ShowWindow.assert_not_called()
        mock_win32gui.SetForegroundWindow.assert_called_with(12345)

    @patch('voice_assistant.win32gui')
    @patch('voice_assistant.win32con')
    def test_bring_to_front_handles_exception(self, mock_win32con, mock_win32gui):
        """Should not raise on exception."""
        mock_win32gui.GetWindowLong.side_effect = Exception("access denied")
        # Should not raise
        _bring_to_front(12345)


if __name__ == "__main__":
    unittest.main(verbosity=2)
