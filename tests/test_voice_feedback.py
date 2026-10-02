"""voice_feedback 本地 SAPI 播报测试：全部 mock，不发声。"""
import os
import sys
import threading
import time
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'app'))

from services import voice_feedback as vf  # noqa: E402


def _make_service(enabled=True):
    svc = vf.VoiceFeedbackService(enabled=enabled)
    svc._voice = mock.MagicMock()
    return svc


class TestVoiceFeedback(unittest.TestCase):
    def test_disabled_when_pywin32_missing(self):
        """pywin32 缺失时构造安全，is_available=False。"""
        with mock.patch.object(vf, "win32com", None):
            svc = vf.VoiceFeedbackService()
            self.assertFalse(svc.is_available())
            self.assertFalse(svc.say("你好"))

    def test_config_disables(self):
        """voice_feedback_enabled=false 时禁用。"""
        cfg = mock.MagicMock()
        cfg.get.side_effect = lambda k, d=None: {
            "voice_feedback_enabled": False,
        }.get(k, d)
        svc = vf.VoiceFeedbackService(cfg)
        self.assertFalse(svc.is_available())

    def test_empty_text_ignored(self):
        svc = _make_service()
        self.assertFalse(svc.say(""))
        self.assertFalse(svc.say("   "))
        self.assertFalse(svc.is_speaking())

    def test_say_and_worker_speak(self):
        """say 投递后 worker 线程调用 SAPI Speak。"""
        svc = _make_service()
        spoken = []
        svc._voice = None  # 让 worker 自己 Dispatch

        fake_voice = mock.MagicMock()
        fake_voice.Speak.side_effect = lambda text, flags: spoken.append(text)

        with mock.patch.object(vf.win32com.client, "Dispatch",
                               return_value=fake_voice), \
             mock.patch.object(vf.pythoncom, "CoInitialize"), \
             mock.patch.object(vf.pythoncom, "CoUninitialize"):
            self.assertTrue(svc.say("已切到板书模式"))
            deadline = time.monotonic() + 3
            while not spoken and time.monotonic() < deadline:
                time.sleep(0.02)
        self.assertEqual(spoken, ["已切到板书模式"])
        svc.stop()

    def test_is_speaking_lifecycle(self):
        """播报中 is_speaking=True，播完回落 False。"""
        svc = _make_service()
        svc._voice = None

        started = threading.Event()

        def slow_speak(text, flags):
            started.set()
            time.sleep(0.3)

        fake_voice = mock.MagicMock()
        fake_voice.Speak.side_effect = slow_speak

        with mock.patch.object(vf.win32com.client, "Dispatch",
                               return_value=fake_voice), \
             mock.patch.object(vf.pythoncom, "CoInitialize"), \
             mock.patch.object(vf.pythoncom, "CoUninitialize"):
            svc.say("较长的一句播报")
            self.assertTrue(started.wait(2))
            self.assertTrue(svc.is_speaking())
            deadline = time.monotonic() + 3
            while svc.is_speaking() and time.monotonic() < deadline:
                time.sleep(0.02)
        self.assertFalse(svc.is_speaking())
        svc.stop()

    def test_stop_interrupts(self):
        """stop 清空队列并终止 worker。"""
        svc = _make_service()
        svc._voice = None
        fake_voice = mock.MagicMock()
        with mock.patch.object(vf.win32com.client, "Dispatch",
                               return_value=fake_voice), \
             mock.patch.object(vf.pythoncom, "CoInitialize"), \
             mock.patch.object(vf.pythoncom, "CoUninitialize"):
            svc.say("第一句")
            self.assertTrue(svc.stop(timeout_sec=3))
        self.assertFalse(svc.is_speaking())


if __name__ == "__main__":
    unittest.main(verbosity=2)
