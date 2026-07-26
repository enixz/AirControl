import json
import os
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from services.frame_recorder import FrameRecorder


class TestFrameRecorder(unittest.TestCase):
    def test_close_timeout_never_evicts_a_queued_frame_for_sentinel(self):
        class BlockingWriter:
            def __init__(self):
                self.entered = threading.Event()
                self.release_event = threading.Event()

            def write(self, _frame):
                self.entered.set()
                self.release_event.wait()

            def release(self):
                pass

        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = FrameRecorder(
                out_root=temp_dir,
                max_frames=9,
                max_seconds=10,
            )
            recorder._writer = BlockingWriter()
            frame = np.zeros((16, 16, 3), dtype=np.uint8)
            recorder.write(frame)
            self.assertTrue(recorder._writer.entered.wait(timeout=1.0))
            for _ in range(8):
                recorder.write(frame)
            self.assertEqual(recorder._queue.qsize(), 8)

            self.assertFalse(recorder.close(timeout_sec=0.03))
            self.assertEqual(recorder._queue.qsize(), 8)
            self.assertFalse(recorder._sentinel_queued)

            recorder._writer.release_event.set()
            self.assertTrue(recorder.close(timeout_sec=1.0))
            self.assertEqual(recorder._count, 9)

    def test_close_timeout_preserves_writer_owned_resources(self):
        class BlockingWriter:
            def __init__(self):
                self.entered = threading.Event()
                self.release_event = threading.Event()
                self.released = False

            def write(self, _frame):
                self.entered.set()
                self.release_event.wait()

            def release(self):
                self.released = True

        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = FrameRecorder(
                out_root=temp_dir,
                max_frames=1,
                max_seconds=10,
            )
            writer = BlockingWriter()
            recorder._writer = writer
            recorder.write(np.zeros((16, 16, 3), dtype=np.uint8))
            self.assertTrue(writer.entered.wait(timeout=1.0))

            started = time.perf_counter()
            self.assertFalse(recorder.close(timeout_sec=0.05))
            self.assertLess(time.perf_counter() - started, 0.5)
            self.assertFalse(writer.released)
            self.assertFalse(recorder._meta.closed)

            writer.release_event.set()
            self.assertTrue(recorder.close(timeout_sec=1.0))
            self.assertTrue(writer.released)
            self.assertTrue(recorder._meta.closed)

    def test_close_waits_for_slow_writer_before_closing_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = FrameRecorder(
                out_root=temp_dir,
                max_frames=1,
                max_seconds=10,
            )
            recorder._use_png = True
            entered = threading.Event()
            release = threading.Event()

            def slow_write(*_args, **_kwargs):
                entered.set()
                release.wait(timeout=2.0)
                return True

            with mock.patch(
                "services.frame_recorder.cv2.imwrite",
                side_effect=slow_write,
            ):
                recorder.write(np.zeros((16, 16, 3), dtype=np.uint8))
                self.assertTrue(entered.wait(timeout=1.0))
                timer = threading.Timer(0.1, release.set)
                timer.start()
                started = time.perf_counter()
                recorder.close()
                elapsed = time.perf_counter() - started
                timer.join(timeout=1.0)

            self.assertGreaterEqual(elapsed, 0.08)
            self.assertFalse(recorder._thread.is_alive())
            self.assertTrue(recorder._meta.closed)

    def test_async_recorder_flushes_metadata_on_close(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = FrameRecorder(
                out_root=temp_dir,
                max_frames=3,
                max_seconds=10,
            )
            frame = np.zeros((16, 16, 3), dtype=np.uint8)
            for _ in range(3):
                recorder.write(frame)
            recorder.close()

            meta_path = os.path.join(recorder.dir, "meta.jsonl")
            with open(meta_path, encoding="utf-8") as stream:
                rows = [json.loads(line) for line in stream if line.strip()]
            self.assertEqual(len(rows), 3)
            self.assertEqual([row["i"] for row in rows], [0, 1, 2])

    def test_write_with_meta_merges_fields_into_jsonl(self):
        """write(frame, meta=...) 应把 meta 字段合并到 meta.jsonl 的对应行。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = FrameRecorder(
                out_root=temp_dir, max_frames=10, max_seconds=10
            )
            frame = np.zeros((16, 16, 3), dtype=np.uint8)
            recorder.write(frame, meta={
                "hands": 2,
                "primary_wrist": [100.0, 200.0],
                "zoom_on": True,
                "wrists": [[100.0, 200.0], [400.0, 50.0]],
            })
            recorder.write(frame, meta={"hands": 1, "primary_wrist": [105.0, 198.0]})
            recorder.write(frame)  # 无 meta
            recorder.close()

            meta_path = os.path.join(recorder.dir, "meta.jsonl")
            with open(meta_path, encoding="utf-8") as stream:
                rows = [json.loads(line) for line in stream if line.strip()]

            self.assertEqual(len(rows), 3)
            # 基础字段始终存在
            for r in rows:
                self.assertIn("i", r)
                self.assertIn("t", r)
                self.assertIn("w", r)
                self.assertIn("h", r)
            # 第一帧：完整 meta
            self.assertEqual(rows[0]["hands"], 2)
            self.assertEqual(rows[0]["primary_wrist"], [100.0, 200.0])
            self.assertTrue(rows[0]["zoom_on"])
            self.assertEqual(rows[0]["wrists"], [[100.0, 200.0], [400.0, 50.0]])
            # 第二帧：部分 meta
            self.assertEqual(rows[1]["hands"], 1)
            self.assertEqual(rows[1]["primary_wrist"], [105.0, 198.0])
            self.assertNotIn("zoom_on", rows[1])
            # 第三帧：无 meta，只有基础字段
            self.assertNotIn("hands", rows[2])
            self.assertNotIn("primary_wrist", rows[2])

    def test_write_meta_aligns_with_frame_index(self):
        """连续多帧带不同 meta，验证每行 meta 与 frame index 严格对齐。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = FrameRecorder(
                out_root=temp_dir, max_frames=5, max_seconds=10
            )
            frame = np.zeros((16, 16, 3), dtype=np.uint8)
            for i in range(5):
                recorder.write(frame, meta={"hands": i, "primary_wrist": [float(i), 0.0]})
            recorder.close()

            meta_path = os.path.join(recorder.dir, "meta.jsonl")
            with open(meta_path, encoding="utf-8") as stream:
                rows = [json.loads(line) for line in stream if line.strip()]

            self.assertEqual(len(rows), 5)
            for i, r in enumerate(rows):
                self.assertEqual(r["i"], i)
                self.assertEqual(r["hands"], i)
                self.assertEqual(r["primary_wrist"], [float(i), 0.0])


if __name__ == "__main__":
    unittest.main()
