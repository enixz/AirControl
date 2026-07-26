import os
import sys
import threading
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from main_ui import SettingsDialog, camera_config_values
from orchestrator import AirControlOrchestrator, choose_startup_resolution


def _make_orchestrator():
    with (
        patch("orchestrator.AirControlOrchestrator.init_services"),
        patch("orchestrator.AirControlOrchestrator._init_modes"),
        patch("orchestrator.AirControlOrchestrator.set_mode"),
        patch("orchestrator.ConfigManager"),
        patch("orchestrator.MouseController"),
    ):
        orchestrator = AirControlOrchestrator(
            MagicMock(),
            MagicMock(),
            MagicMock(),
        )
    values = {
        "camera_width": None,
        "camera_height": None,
        "camera_force_mjpeg": True,
        "camera_min_fps": 20,
    }
    orchestrator.config = MagicMock()
    orchestrator.config.get.side_effect = (
        lambda key, default=None: values.get(key, default)
    )
    orchestrator.camera = MagicMock(
        camera_index=0,
        width=1280,
        height=720,
        _backend=700,
    )
    orchestrator.inference_worker = MagicMock()
    orchestrator._restart_inference_worker = MagicMock()
    orchestrator._find_available_cameras = MagicMock()
    return orchestrator


class TestCameraSwitchPreprobe(unittest.TestCase):
    def test_incomplete_current_camera_item_preserves_saved_resolution(self):
        self.assertEqual(
            camera_config_values({"index": 0, "current": True}),
            {"camera_index": 0},
        )
        self.assertEqual(
            camera_config_values({"index": 1, "width": 1920, "height": 1080}),
            {"camera_index": 1, "camera_width": 1920, "camera_height": 1080},
        )

    def test_startup_resolution_uses_fast_default_until_user_probe(self):
        self.assertEqual(choose_startup_resolution(None, None), (1280, 720))
        self.assertEqual(choose_startup_resolution(1920, 1080), (1920, 1080))

    @patch("orchestrator.CameraService")
    def test_switch_reuses_background_probe_without_enumerating_ui_thread(
        self,
        camera_service_class,
    ):
        orchestrator = _make_orchestrator()
        new_camera = MagicMock(width=1920, height=1080)
        camera_service_class.return_value = new_camera
        camera_info = {
            "index": 1,
            "backend": 1400,
            "width": 1920,
            "height": 1080,
        }

        self.assertTrue(orchestrator.switch_camera(1, camera_info))

        orchestrator._find_available_cameras.assert_not_called()
        camera_service_class.assert_called_once_with(
            camera_index=1,
            width=1920,
            height=1080,
            force_mjpeg=True,
            min_fps=20,
            preferred_backend=1400,
        )
        new_camera.start.assert_called_once_with()
        orchestrator._restart_inference_worker.assert_called_once_with()

    @patch("orchestrator.CameraService")
    def test_failed_switch_rolls_back_with_previous_backend_and_resolution(
        self,
        camera_service_class,
    ):
        orchestrator = _make_orchestrator()
        failed_camera = MagicMock()
        failed_camera.start.side_effect = RuntimeError("busy")
        rollback_camera = MagicMock()
        camera_service_class.side_effect = [failed_camera, rollback_camera]

        self.assertFalse(
            orchestrator.switch_camera(
                1,
                {
                    "index": 1,
                    "backend": 1400,
                    "width": 1920,
                    "height": 1080,
                },
            )
        )

        failed_camera.release.assert_called_once_with()
        rollback_kwargs = camera_service_class.call_args_list[1].kwargs
        self.assertEqual(rollback_kwargs["camera_index"], 0)
        self.assertEqual(rollback_kwargs["width"], 1280)
        self.assertEqual(rollback_kwargs["height"], 720)
        self.assertEqual(rollback_kwargs["preferred_backend"], 700)
        rollback_camera.start.assert_called_once_with()

    def test_settings_enumeration_does_not_probe_every_candidate(self):
        fake_dialog = MagicMock()
        fake_dialog._camera_worker_cancel = MagicMock()
        fake_dialog._camera_worker_cancel.is_set.return_value = False
        fake_dialog.config.get.side_effect = lambda key, default=None: {
            "camera_index": 0,
            "camera_min_fps": 20,
            "camera_force_mjpeg": True,
        }.get(key, default)
        cameras = [
            {
                "index": 0,
                "name": "摄像头 0（当前）",
                "backend": None,
            },
            {
                "index": 1,
                "name": "摄像头 1",
                "backend": 1400,
            },
        ]
        with (
            patch("main_ui.list_available_cameras", return_value=cameras),
            patch(
                "main_ui.probe_max_resolution",
                return_value=(1920, 1080),
            ) as probe,
        ):
            SettingsDialog._enumerate_cameras_worker(fake_dialog)

        probe.assert_not_called()
        emitted = fake_dialog._cameras_enumerated.emit.call_args.args[0]
        self.assertTrue(emitted[0]["current"])
        self.assertNotIn("width", emitted[1])
        self.assertNotIn("height", emitted[1])

    def test_only_selected_camera_is_probed_in_background(self):
        fake_dialog = MagicMock()
        fake_dialog.config.get.side_effect = lambda key, default=None: {
            "camera_min_fps": 20,
            "camera_force_mjpeg": True,
        }.get(key, default)
        fake_dialog._camera_probe_lock = threading.Lock()
        fake_dialog._camera_probe_threads = {1: threading.current_thread()}
        fake_dialog._camera_worker_cancel = MagicMock()
        fake_dialog._camera_worker_cancel.is_set.return_value = False
        camera = {
            "index": 1,
            "name": "摄像头 1",
            "backend": 1400,
        }
        with (
            patch(
                "main_ui.probe_max_resolution",
                return_value=(1920, 1080),
            ) as probe,
        ):
            SettingsDialog._probe_camera_worker(fake_dialog, camera)

        probe.assert_called_once_with(
            1,
            min_fps=20,
            force_mjpeg=True,
            preferred_backend=1400,
        )
        fake_dialog._camera_worker_cancel.wait.assert_called_once_with(0.5)
        fake_dialog._camera_probed.emit.assert_called_once_with(
            1,
            (1920, 1080),
            "",
        )


if __name__ == "__main__":
    unittest.main()
