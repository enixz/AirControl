import os
import sys
import threading
import time
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from orchestrator import AirControlOrchestrator


def _make_orchestrator():
    with (
        mock.patch("orchestrator.AirControlOrchestrator.init_services"),
        mock.patch("orchestrator.AirControlOrchestrator._init_modes"),
        mock.patch("orchestrator.AirControlOrchestrator.set_mode"),
        mock.patch("orchestrator.ConfigManager"),
        mock.patch("orchestrator.MouseController"),
    ):
        orchestrator = AirControlOrchestrator(
            mock.MagicMock(),
            mock.MagicMock(),
            mock.MagicMock(),
        )
    orchestrator.mode_manager = mock.MagicMock()
    orchestrator.mode_manager.current_mode = None
    orchestrator.inference_worker = mock.MagicMock()
    orchestrator.voice_command = mock.MagicMock()
    orchestrator.voice_dictation = mock.MagicMock()
    orchestrator.voice_assistant = mock.MagicMock()
    orchestrator.camera = mock.MagicMock()
    orchestrator.tracker = mock.MagicMock()
    orchestrator.inference_worker.tracker = orchestrator.tracker
    orchestrator.frame_recorder = None
    orchestrator._discard_warmed_yolo_tracker = mock.MagicMock()
    return orchestrator


def test_orchestrator_does_not_release_native_handles_after_worker_timeout():
    orchestrator = _make_orchestrator()
    orchestrator.inference_worker.stop.return_value = False
    orchestrator.voice_command.stop.return_value = True
    orchestrator.voice_dictation.stop.return_value = True

    assert not orchestrator.close(timeout_sec=0.01)

    orchestrator.camera.release.assert_not_called()
    orchestrator.tracker.close.assert_not_called()
    assert orchestrator._shutdown_incomplete


def test_orchestrator_releases_native_handles_after_clean_worker_stop():
    orchestrator = _make_orchestrator()
    orchestrator.inference_worker.stop.return_value = True
    orchestrator.voice_command.stop.return_value = True
    orchestrator.voice_dictation.stop.return_value = True

    assert orchestrator.close(timeout_sec=0.01)

    orchestrator.camera.release.assert_called_once()
    orchestrator.tracker.close.assert_called_once()
    assert not orchestrator._shutdown_incomplete


def test_orchestrator_reports_owned_background_task_timeout():
    orchestrator = _make_orchestrator()
    orchestrator.inference_worker.stop.return_value = True
    orchestrator.voice_command.stop.return_value = True
    orchestrator.voice_dictation.stop.return_value = True
    entered = threading.Event()
    release = threading.Event()

    def blocked_task():
        entered.set()
        release.wait()

    thread = orchestrator._start_background_thread(blocked_task, "BlockedBuild")
    assert entered.wait(timeout=1.0)
    started = time.monotonic()
    assert not orchestrator.close(timeout_sec=0.03)
    assert time.monotonic() - started < 0.5
    assert orchestrator._shutdown_incomplete

    release.set()
    thread.join(timeout=1.0)
    assert not thread.is_alive()
