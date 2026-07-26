import json
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from config_manager import ConfigManager, ConfigSaveError


class TestConfigRuntime(unittest.TestCase):
    def test_missing_config_deep_copies_nested_defaults(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "config.json")
            config = ConfigManager(path)
            original = config.default_config["gesture_mapping"]["SWIPE_RIGHT"]

            self.assertIsNot(
                config.config["gesture_mapping"],
                config.default_config["gesture_mapping"],
            )
            config.set_mapping("SWIPE_RIGHT", "none")
            self.assertEqual(
                config.default_config["gesture_mapping"]["SWIPE_RIGHT"],
                original,
            )

    def test_frozen_first_run_seeds_bundled_published_defaults(self):
        with tempfile.TemporaryDirectory() as resource_dir:
            with tempfile.TemporaryDirectory() as data_dir:
                template = {
                    "engine_auto_switch": True,
                    "camera_min_fps": 20,
                    "record_truth_marker": "space,mbutton",
                }
                with open(
                    os.path.join(resource_dir, "config.json"),
                    "w",
                    encoding="utf-8",
                ) as stream:
                    json.dump(template, stream)

                with (
                    mock.patch.object(sys, "frozen", True, create=True),
                    mock.patch.object(
                        sys,
                        "_MEIPASS",
                        resource_dir,
                        create=True,
                    ),
                    mock.patch.dict(
                        os.environ,
                        {"AIRCONTROL_DATA_DIR": data_dir},
                    ),
                ):
                    config = ConfigManager()

                self.assertEqual(
                    config.config_file,
                    os.path.join(data_dir, "config.json"),
                )
                self.assertTrue(os.path.isfile(config.config_file))
                self.assertTrue(config.get("engine_auto_switch"))
                self.assertEqual(config.get("camera_min_fps"), 20)
                self.assertEqual(
                    config.get("record_truth_marker"),
                    "space,mbutton",
                )

    def test_extended_schema_rejects_invalid_runtime_values(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "config.json")
            with open(path, "w", encoding="utf-8") as stream:
                json.dump(
                    {
                        "camera_force_mjpeg": "yes",
                        "stability_profile": "turbo",
                        "draw_vote_ratio": 2.0,
                        "mode_switch_release_sec": 0.0,
                        "record_raw_max_frames": -1,
                    },
                    stream,
                )
            config = ConfigManager(path)
            self.assertIs(config.get("camera_force_mjpeg"), True)
            self.assertEqual(config.get("stability_profile"), "stable")
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

    def test_set_raises_and_rolls_back_memory_when_atomic_save_fails(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ConfigManager(os.path.join(temp_dir, "config.json"))
            original = config.get("mouse_sensitivity")
            config.last_save_error = OSError("disk full")

            with (
                mock.patch.object(config, "_do_save", return_value=False),
                self.assertRaises(ConfigSaveError),
            ):
                config.set("mouse_sensitivity", 99)

            self.assertEqual(config.get("mouse_sensitivity"), original)

    def test_failed_batch_restores_all_values_and_existing_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "config.json")
            config = ConfigManager(path)
            with open(path, encoding="utf-8") as stream:
                before_file = stream.read()
            before_memory = json.loads(json.dumps(config.config))
            config.last_save_error = OSError("read only")

            with (
                mock.patch.object(config, "_do_save", return_value=False),
                self.assertRaises(ConfigSaveError),
            ):
                with config.batch_update():
                    config.set("pen_width", 33)
                    config.set_mapping("FIST", "next_slide")

            self.assertEqual(config.config, before_memory)
            with open(path, encoding="utf-8") as stream:
                self.assertEqual(stream.read(), before_file)

    def test_batch_body_exception_rolls_back_without_saving(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ConfigManager(os.path.join(temp_dir, "config.json"))
            original = config.get("pen_width")

            with (
                mock.patch.object(config, "_do_save", wraps=config._do_save) as save,
                self.assertRaisesRegex(RuntimeError, "cancel"),
            ):
                with config.batch_update():
                    config.set("pen_width", 44)
                    raise RuntimeError("cancel")

            self.assertEqual(config.get("pen_width"), original)
            save.assert_not_called()

    def test_corrupt_config_is_backed_up_before_defaults_replace_it(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "config.json")
            corrupt_bytes = b'{"mouse_sensitivity": invalid}'
            with open(path, "wb") as stream:
                stream.write(corrupt_bytes)

            config = ConfigManager(path)

            backups = [
                name for name in os.listdir(temp_dir)
                if name.startswith("config.json.corrupt-") and name.endswith(".bak")
            ]
            self.assertEqual(len(backups), 1)
            with open(os.path.join(temp_dir, backups[0]), "rb") as stream:
                self.assertEqual(stream.read(), corrupt_bytes)
            with open(path, encoding="utf-8") as stream:
                self.assertEqual(json.load(stream), config.config)

    def test_corrupt_config_is_not_overwritten_when_backup_fails(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "config.json")
            corrupt_bytes = b"not-json"
            with open(path, "wb") as stream:
                stream.write(corrupt_bytes)

            with (
                mock.patch(
                    "config_manager.shutil.copy2",
                    side_effect=PermissionError("denied"),
                ),
                self.assertRaises(ConfigSaveError),
            ):
                ConfigManager(path)

            with open(path, "rb") as stream:
                self.assertEqual(stream.read(), corrupt_bytes)


if __name__ == "__main__":
    unittest.main()
