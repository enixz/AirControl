"""
Tests for AirControlOrchestrator dictation and caption signals.
"""
import sys
import os
import unittest
from unittest.mock import MagicMock, patch

# Ensure app is on sys.path
_app_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'app')
sys.path.insert(0, _app_dir)

# Clean up any mocked PyQt6 modules in sys.modules left by other tests
for m in list(sys.modules.keys()):
    if m == 'PyQt6' or m.startswith('PyQt6.'):
        mod = sys.modules[m]
        # If the module has no __file__ attribute or is a mock, remove it
        if not hasattr(mod, '__file__') or 'mock' in str(type(mod)).lower() or 'MagicMock' in str(mod):
            del sys.modules[m]

if 'orchestrator' in sys.modules:
    del sys.modules['orchestrator']

# Mock win32 modules needed for imports
import types
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

from orchestrator import AirControlOrchestrator

class TestOrchestratorDictation(unittest.TestCase):
    @patch('orchestrator.AirControlOrchestrator.init_services')
    @patch('orchestrator.AirControlOrchestrator._init_modes')
    @patch('orchestrator.AirControlOrchestrator._set_mode')
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

if __name__ == "__main__":
    unittest.main(verbosity=2)
