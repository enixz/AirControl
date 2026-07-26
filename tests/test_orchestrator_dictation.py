"""
Tests for AirControlOrchestrator dictation and caption signals.
"""
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Ensure app is on sys.path
_app_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'app')
sys.path.insert(0, _app_dir)

from orchestrator import AirControlOrchestrator


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

    @patch('orchestrator.AirControlOrchestrator.init_services')
    @patch('orchestrator.AirControlOrchestrator._init_modes')
    @patch('orchestrator.AirControlOrchestrator.set_mode')
    @patch('orchestrator.ConfigManager')
    @patch('orchestrator.MouseController')
    def test_apply_config_refreshes_gesture_feature_switches(
        self,
        mock_mouse_cls,
        mock_config_cls,
        mock_set_mode,
        mock_init_modes,
        mock_init_services,
    ):
        orchestrator = AirControlOrchestrator(MagicMock(), MagicMock(), MagicMock())
        values = {
            "cooldown": 1.0,
            "pinch_hysteresis_enabled": True,
            "pinch_exit_hysteresis_enabled": True,
            "thumb_perp_ratio_enabled": True,
            "interaction_mode": "mouse",
        }
        orchestrator.config = MagicMock()
        orchestrator.config.get.side_effect = lambda key, default=None: values.get(key, default)
        orchestrator.recognizer = MagicMock()
        orchestrator.ppt = MagicMock()
        orchestrator.mouse = MagicMock()
        orchestrator.overlay = MagicMock()
        orchestrator.voice_assistant = MagicMock()
        orchestrator.voice_command = MagicMock()
        orchestrator.mode_manager = MagicMock(current_mode_name="mouse")
        orchestrator._tracker_signature = MagicMock(return_value="same")
        orchestrator._tracker_config_signature = "same"
        orchestrator._current_voice_kws_signature = MagicMock(return_value="same")
        orchestrator._voice_kws_signature = "same"

        orchestrator.apply_config()

        self.assertTrue(orchestrator.recognizer.pinch_hysteresis_enabled)
        self.assertTrue(orchestrator.recognizer.pinch_exit_hysteresis_enabled)
        self.assertTrue(orchestrator.recognizer.thumb_perp_ratio_enabled)

if __name__ == "__main__":
    unittest.main(verbosity=2)
