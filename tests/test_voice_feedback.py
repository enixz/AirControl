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

    def test_concurrent_say_starts_single_worker(self):
        """多线程并发 say()（UI 线程 + 云端健康线程）：_ensure_worker
        加锁后只能有一个 worker，双 worker 会共享/覆盖 COM 对象。"""
        svc = _make_service()
        spoken = []
        fake_voice = mock.MagicMock()
        fake_voice.Speak.side_effect = lambda text, flags: spoken.append(text)
        with mock.patch.object(vf.win32com.client, "Dispatch",
                               return_value=fake_voice), \
             mock.patch.object(vf.pythoncom, "CoInitialize"), \
             mock.patch.object(vf.pythoncom, "CoUninitialize"):
            threads = [threading.Thread(target=svc.say, args=(f"第{i}句",))
                       for i in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            deadline = time.monotonic() + 3
            while len(spoken) < 8 and time.monotonic() < deadline:
                time.sleep(0.02)
            workers = [t for t in threading.enumerate()
                       if t.name == "VoiceFeedbackWorker"]
            self.assertEqual(len(workers), 1)
            self.assertEqual(len(spoken), 8)
        svc.stop()

    def test_say_after_stop_revives_worker(self):
        """stop() 后再次 say() 应拉起新 worker 继续播报（事件对象按
        worker 隔离，旧事件的 set 不影响新 worker）。"""
        svc = _make_service()
        spoken = []
        fake_voice = mock.MagicMock()
        fake_voice.Speak.side_effect = lambda text, flags: spoken.append(text)
        with mock.patch.object(vf.win32com.client, "Dispatch",
                               return_value=fake_voice), \
             mock.patch.object(vf.pythoncom, "CoInitialize"), \
             mock.patch.object(vf.pythoncom, "CoUninitialize"):
            svc.say("第一句")
            deadline = time.monotonic() + 3
            while not spoken and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertTrue(svc.stop(timeout_sec=3))
            self.assertTrue(svc.say("第二句"))
            deadline = time.monotonic() + 3
            while len(spoken) < 2 and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertEqual(spoken, ["第一句", "第二句"])
        svc.stop()

    def test_volume_zero_respected(self):
        """音量 0 是合法配置（静音），不应被 `or` 兜底改回默认值。"""
        cfg = mock.MagicMock()
        cfg.get.side_effect = lambda k, d=None: {
            "voice_feedback_volume": 0,
        }.get(k, d)
        svc = vf.VoiceFeedbackService(cfg)
        self.assertEqual(svc._volume, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
