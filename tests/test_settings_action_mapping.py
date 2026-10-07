"""Exercise the real Qt dialog in isolation, without camera probing."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("action", ["cancel_llm_command", "hang_up_voice_assistant"])
def test_saving_unrelated_settings_preserves_cancel_action(tmp_path, action):
    root = Path(__file__).resolve().parents[1]
    script = """
import os
import sys
from unittest.mock import patch
sys.path.insert(0, os.path.abspath('app'))
from PyQt6.QtWidgets import QApplication
from config_manager import ConfigManager
from main_ui import SettingsDialog
app = QApplication([])
config = ConfigManager(os.path.join(sys.argv[1], 'config.json'))
config.set_mapping('THUMB_DOWN', sys.argv[2])
config.set_mapping('SWIPE_LEFT', 'hang_up_voice_assistant')
config.set_mapping('FIST', 'custom_action_from_newer_build')
with patch.object(SettingsDialog, '_start_camera_worker', return_value=None):
    dialog = SettingsDialog(config)
assert dialog.thumb_down_combo.currentText() == 'cancel_llm_command'
assert dialog.left_combo.currentText() == 'cancel_llm_command'
dialog.pen_spin.setValue(17)
dialog.save_settings()
assert config.get_mapping('THUMB_DOWN') == 'cancel_llm_command'
assert config.get_mapping('SWIPE_LEFT') == 'cancel_llm_command'
assert config.get_mapping('FIST') == 'custom_action_from_newer_build'
assert ConfigManager(config.config_file).get_mapping('THUMB_DOWN') == 'cancel_llm_command'
dialog.close()
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), action],
        cwd=root,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen", "PYTHONIOENCODING": "utf-8"},
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
