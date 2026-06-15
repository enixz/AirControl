import json
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from config_manager import ConfigManager


class TestConfigRuntime(unittest.TestCase):
    def test_extended_schema_rejects_invalid_runtime_values(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "config.json")
            with open(path, "w", encoding="utf-8") as stream:
                json.dump(
                    {
                        "camera_force_mjpeg": "yes",
                        "draw_vote_ratio": 2.0,
                        "mode_switch_release_sec": 0.0,
                        "record_raw_max_frames": -1,
                    },
                    stream,
                )
            config = ConfigManager(path)
            self.assertIs(config.get("camera_force_mjpeg"), True)
            self.assertEqual(config.get("draw_vote_ratio"), 0.60)
            self.assertEqual(config.get("mode_switch_release_sec"), 0.25)
            self.assertEqual(config.get("record_raw_max_frames"), 2000)

    def test_save_is_atomic_and_leaves_no_temp_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "config.json")
            config = ConfigManager(path)
            config.set("mouse_sensitivity", 77)
            self.assertFalse(os.path.exists(path + ".tmp"))
            with open(path, encoding="utf-8") as stream:
                self.assertEqual(json.load(stream)["mouse_sensitivity"], 77)

    def test_nested_batches_save_once(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ConfigManager(os.path.join(temp_dir, "config.json"))
            with mock.patch.object(
                config, "_do_save", wraps=config._do_save
            ) as save:
                with config.batch_update():
                    config.set("pen_width", 10)
                    with config.batch_update():
                        config.set("mouse_sensitivity", 50)
                self.assertEqual(save.call_count, 1)


if __name__ == "__main__":
    unittest.main()
