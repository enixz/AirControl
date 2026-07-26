import os
import sys
import threading
import time
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from services import voice_dictation as voice_dictation_module
from services.voice_command import VoiceCommandService
from services.voice_dictation import VoiceDictationService


class _FakeConfig(dict):
    def get(self, key, default=None):
        return super().get(key, default)


class TestVoiceShutdown(unittest.TestCase):
    def test_dictation_stop_timeout_preserves_live_recognizer(self):
        entered = threading.Event()
        release = threading.Event()

        class FakeStream:
            def accept_waveform(self, _sample_rate, _samples):
                pass

            class Result:
                text = "完成"

            result = Result()

        class FakeRecognizer:
            def create_stream(self):
                entered.set()
                release.wait()
                return FakeStream()

            def decode_stream(self, _stream):
                pass

        service = VoiceDictationService(_FakeConfig(dictation_model_dir="unused"))
        recognizer = FakeRecognizer()
        service._recognizer = recognizer
        with mock.patch.object(voice_dictation_module, "sherpa_onnx", object()):
            worker = threading.Thread(
                target=lambda: service.dictate(
                    np.array([0.1], dtype=np.float32),
                    16000,
                ),
                daemon=True,
            )
            worker.start()
            self.assertTrue(entered.wait(timeout=1.0))

            started = time.perf_counter()
            self.assertFalse(service.stop(timeout_sec=0.05))
            self.assertLess(time.perf_counter() - started, 0.5)
            self.assertIs(service._recognizer, recognizer)

            release.set()
            worker.join(timeout=1.0)
            self.assertTrue(service.stop(timeout_sec=1.0))
            self.assertIsNone(service._recognizer)

    def test_dictation_stop_waits_for_active_native_decode(self):
        entered = threading.Event()
        release = threading.Event()

        class FakeStream:
            def accept_waveform(self, _sample_rate, _samples):
                pass

            class Result:
                text = "完成"

            result = Result()

        class FakeRecognizer:
            def create_stream(self):
                entered.set()
                release.wait(timeout=2.0)
                return FakeStream()

            def decode_stream(self, _stream):
                pass

        service = VoiceDictationService(_FakeConfig(dictation_model_dir="unused"))
        service._recognizer = FakeRecognizer()
        result = []
        with mock.patch.object(voice_dictation_module, "sherpa_onnx", object()):
            worker = threading.Thread(
                target=lambda: result.append(
                    service.dictate(np.array([0.1], dtype=np.float32), 16000)
                )
            )
            worker.start()
            self.assertTrue(entered.wait(timeout=1.0))
            timer = threading.Timer(0.1, release.set)
            timer.start()
            started = time.perf_counter()
            service.stop()
            elapsed = time.perf_counter() - started
            worker.join(timeout=1.0)
            timer.join(timeout=1.0)

        self.assertGreaterEqual(elapsed, 0.08)
        self.assertFalse(worker.is_alive())
        self.assertEqual(result, ["完成"])
        self.assertIsNone(service._recognizer)

    def test_voice_command_stop_joins_tracked_asr_jobs(self):
        service = VoiceCommandService(_FakeConfig())
        entered = threading.Event()
        release = threading.Event()

        def blocking_asr():
            entered.set()
            release.wait(timeout=2.0)

        self.assertTrue(
            service._start_asr_thread(
                blocking_asr,
                name="TestBlockingAsrWorker",
            )
        )
        self.assertTrue(entered.wait(timeout=1.0))
        timer = threading.Timer(0.1, release.set)
        timer.start()
        started = time.perf_counter()
        service.stop()
        elapsed = time.perf_counter() - started
        timer.join(timeout=1.0)

        self.assertGreaterEqual(elapsed, 0.08)
        with service._asr_threads_lock:
            self.assertEqual(service._asr_threads, set())
            self.assertFalse(service._asr_accepting)

    def test_voice_command_stop_timeout_is_bounded(self):
        service = VoiceCommandService(_FakeConfig())
        entered = threading.Event()
        release = threading.Event()

        def blocking_asr():
            entered.set()
            release.wait()

        self.assertTrue(
            service._start_asr_thread(
                blocking_asr,
                name="TestUnboundedAsrWorker",
            )
        )
        self.assertTrue(entered.wait(timeout=1.0))

        started = time.perf_counter()
        self.assertFalse(service.stop(timeout_sec=0.05))
        self.assertLess(time.perf_counter() - started, 0.5)

        release.set()
        self.assertTrue(service.stop(timeout_sec=1.0))

    def test_voice_audio_cleanup_timeout_reuses_single_cleanup_thread(self):
        service = VoiceCommandService(_FakeConfig())
        entered = threading.Event()
        release = threading.Event()

        class BlockingAudioStream:
            def stop(self):
                entered.set()
                release.wait()

            def close(self):
                pass

        stream = BlockingAudioStream()
        service._audio_stream = stream

        self.assertFalse(service.stop(timeout_sec=0.05))
        self.assertTrue(entered.is_set())
        first_cleanup = service._audio_cleanup_thread
        self.assertTrue(first_cleanup.is_alive())

        release.set()
        self.assertTrue(service.stop(timeout_sec=1.0))
        self.assertIs(service._audio_cleanup_thread, first_cleanup)
        self.assertIsNone(service._audio_stream)


if __name__ == "__main__":
    unittest.main()
