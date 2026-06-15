import os
import sys
import tempfile
import threading
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from services.voice_command import VoiceCommandService


class FakeConfig:
    def get(self, key, default=None):
        return default


class TestVoiceCommandReload(unittest.TestCase):
    def test_mode_keywords_are_written_outside_repository(self):
        service = VoiceCommandService(FakeConfig())
        with tempfile.TemporaryDirectory() as temp_dir:
            service._keywords_cache_dir = temp_dir
            path = service._generate_mode_keywords("draw")
            self.assertTrue(path.startswith(temp_dir))
            self.assertTrue(os.path.isfile(path))
            with open(path, encoding="utf-8") as stream:
                text = stream.read()
            self.assertIn("@清屏", text)
            self.assertNotIn("@下一页", text)

    def test_reload_requests_are_coalesced_to_latest_mode(self):
        service = VoiceCommandService(FakeConfig())
        service._running = True
        service._kws = object()
        service._kws_stream = object()
        built_modes = []
        first_started = threading.Event()
        allow_first = threading.Event()

        def build(mode):
            built_modes.append(mode)
            if len(built_modes) == 1:
                first_started.set()
                allow_first.wait(2)
            marker = object()
            return marker, marker

        with mock.patch.object(service, "_build_kws_for_mode", side_effect=build):
            service.on_mode_changed("presentation")
            self.assertTrue(first_started.wait(1))
            service.on_mode_changed("mouse")
            service.on_mode_changed("draw")
            allow_first.set()
            service._reload_thread.join(2)

        self.assertEqual(built_modes, ["presentation", "draw"])
        self.assertEqual(service._current_mode, "draw")


if __name__ == "__main__":
    unittest.main()
