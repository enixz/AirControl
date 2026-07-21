"""真值事件采集器单元测试（注入假按键状态源，不依赖 win32）。"""
import json
import os
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from services.truth_event_logger import (
    MARKER_VK,
    TruthEventLogger,
    normalize_markers,
)


class _FakeKeys:
    """可编程按键状态源：测试线程翻转字典模拟按下/松开。"""

    def __init__(self):
        self.state = {}

    def get(self, vk):
        return self.state.get(vk, False)


def _read_rows(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class TestTruthEventLogger(unittest.TestCase):
    def test_header_and_down_up_transitions(self):
        fake = _FakeKeys()
        vk = MARKER_VK["space"]
        with tempfile.TemporaryDirectory() as d:
            logger = TruthEventLogger(
                d, markers=("space",), get_state=fake.get, poll_interval=0.001,
            )
            time.sleep(0.03)
            fake.state[vk] = True
            time.sleep(0.03)
            fake.state[vk] = False
            time.sleep(0.03)
            logger.close()
            rows = _read_rows(logger.path)

        self.assertEqual(rows[0]["type"], "header")
        self.assertIn("space", rows[0]["markers"])
        events = [r for r in rows if r.get("event")]
        self.assertEqual([r["event"] for r in events], ["down", "up"])
        self.assertEqual(events[0]["key"], "space")
        self.assertLessEqual(events[0]["t"], events[1]["t"])
        self.assertEqual(rows[-1]["type"], "footer")

    def test_close_synthesizes_up_for_held_key(self):
        fake = _FakeKeys()
        vk = MARKER_VK["space"]
        with tempfile.TemporaryDirectory() as d:
            logger = TruthEventLogger(
                d, markers=("space",), get_state=fake.get, poll_interval=0.001,
            )
            time.sleep(0.02)
            fake.state[vk] = True
            time.sleep(0.02)
            logger.close()  # 仍按住 → 应补 synthetic up
            rows = _read_rows(logger.path)

        events = [r for r in rows if r.get("event")]
        self.assertEqual([r["event"] for r in events], ["down", "up"])
        self.assertEqual(events[-1].get("note"), "synthetic_on_close")

    def test_initial_held_key_recorded(self):
        fake = _FakeKeys()
        vk = MARKER_VK["space"]
        fake.state[vk] = True  # 构造前已按住
        with tempfile.TemporaryDirectory() as d:
            logger = TruthEventLogger(
                d, markers=("space",), get_state=fake.get, poll_interval=0.001,
            )
            time.sleep(0.02)
            fake.state[vk] = False
            time.sleep(0.02)
            logger.close()
            rows = _read_rows(logger.path)

        events = [r for r in rows if r.get("event")]
        self.assertEqual([r["event"] for r in events], ["down", "up"])

    def test_unknown_marker_skipped(self):
        fake = _FakeKeys()
        with tempfile.TemporaryDirectory() as d:
            logger = TruthEventLogger(
                d, markers=("nope",), get_state=fake.get, poll_interval=0.001,
            )
            logger.close()
            rows = _read_rows(logger.path)
        self.assertEqual(rows[0]["markers"], {})


class TestMarkerChannels(unittest.TestCase):
    def test_remote_channels_exist(self):
        """远距标记通道（无线鼠标/翻页笔）必须有键码映射。"""
        for name in ("rbutton", "mbutton", "pageup", "pagedown"):
            self.assertIn(name, MARKER_VK)

    def test_normalize_markers(self):
        self.assertEqual(normalize_markers("space, pagedown"), ("space", "pagedown"))
        self.assertEqual(normalize_markers("space"), ("space",))
        self.assertEqual(normalize_markers(("enter",)), ("enter",))

    def test_comma_separated_multi_markers(self):
        fake = _FakeKeys()
        with tempfile.TemporaryDirectory() as d:
            logger = TruthEventLogger(
                d, markers="pagedown,pageup", get_state=fake.get, poll_interval=0.001,
            )
            logger.close()
            rows = _read_rows(logger.path)
        self.assertEqual(set(rows[0]["markers"]), {"pagedown", "pageup"})

    def test_multi_marker_events_pair_independently(self):
        """两个标记键同时按住时，down/up 各自独立配对。"""
        fake = _FakeKeys()
        pd_vk = MARKER_VK["pagedown"]
        pu_vk = MARKER_VK["pageup"]
        with tempfile.TemporaryDirectory() as d:
            logger = TruthEventLogger(
                d, markers=("pagedown", "pageup"), get_state=fake.get,
                poll_interval=0.001,
            )
            time.sleep(0.02)
            fake.state[pd_vk] = True
            time.sleep(0.02)
            fake.state[pu_vk] = True   # pagedown 仍按住
            time.sleep(0.02)
            fake.state[pd_vk] = False
            time.sleep(0.02)
            fake.state[pu_vk] = False
            time.sleep(0.02)
            logger.close()
            rows = _read_rows(logger.path)

        events = [(r["key"], r["event"]) for r in rows if r.get("event")]
        self.assertEqual(events, [
            ("pagedown", "down"), ("pageup", "down"),
            ("pagedown", "up"), ("pageup", "up"),
        ])


if __name__ == "__main__":
    unittest.main()
