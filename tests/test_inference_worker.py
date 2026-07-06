import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from services.inference_worker import InferenceWorker


class FakeTracker:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class TestInferenceWorker(unittest.TestCase):
    def test_latest_frame_replaces_stale_frames(self):
        worker = InferenceWorker(camera=None, tracker=FakeTracker())
        worker.running = True
        old = np.zeros((2, 2, 3), dtype=np.uint8)
        new = np.ones((2, 2, 3), dtype=np.uint8)
        with worker._capture_condition:
            worker._latest_sequence = 1
            worker._latest_frame = (1, old, 10.0, 1.0)
            worker._latest_sequence = 2
            worker._latest_frame = (2, new, 11.0, 2.0)

        frame, capture_ms, captured_at = worker._wait_for_latest_frame()
        self.assertIs(frame, new)
        self.assertEqual(capture_ms, 11.0)
        self.assertEqual(captured_at, 2.0)
        self.assertEqual(worker._processed_sequence, 2)

    def test_tracker_swap_is_async_and_closes_old_on_apply(self):
        """P0-4: update_tracker 是异步的——挂起 pending，不立即替换/关闭。

        旧 tracker 在 worker 下个推理循环真正 swap 时才关闭（见 _process_frame）。
        这避免了主线程在推理进行中抢锁阻塞。
        """
        old = FakeTracker()
        new = FakeTracker()
        worker = InferenceWorker(camera=None, tracker=old)
        worker.update_tracker(new)
        # 异步语义：挂起 pending，active 仍是 old，old 未关闭
        self.assertFalse(old.closed)
        self.assertIs(worker.tracker, old)
        self.assertIs(worker._pending_tracker, new)
        # 模拟 _process_frame 里的 swap 逻辑
        with worker.lock:
            if worker._pending_tracker is not None:
                old_active = worker.tracker
                worker.tracker = worker._pending_tracker
                worker._pending_tracker = None
                if old_active is not worker.tracker:
                    close = getattr(old_active, "close", None)
                    if callable(close):
                        close()
        self.assertTrue(old.closed)
        self.assertIs(worker.tracker, new)
        self.assertIsNone(worker._pending_tracker)

    def test_stop_flushes_pending_tracker(self):
        """P0-4: stop() 必须关闭从未 swap-in 的 pending tracker，防止句柄泄漏。

        active tracker 不由 stop() 关闭（orchestrator 持有并复用，例如切摄像头）。
        """
        old = FakeTracker()
        new = FakeTracker()
        worker = InferenceWorker(camera=None, tracker=old)
        worker.update_tracker(new)
        # pending 已挂起，尚未 swap
        self.assertFalse(new.closed)
        worker.stop()
        # pending tracker 被 flush 关闭
        self.assertTrue(new.closed)
        self.assertIsNone(worker._pending_tracker)
        # active tracker 不被 stop() 关闭（由 orchestrator 负责）
        self.assertFalse(old.closed)
        self.assertIs(worker.tracker, old)

    def test_percentile_is_stable_for_empty_and_populated_samples(self):
        self.assertEqual(InferenceWorker._percentile([], 0.95), 0.0)
        self.assertEqual(InferenceWorker._percentile([1, 2, 3, 4], 0.5), 3)
        self.assertEqual(InferenceWorker._percentile([1, 2, 3, 4], 0.95), 4)

    def test_only_one_result_can_be_pending_for_the_ui(self):
        worker = InferenceWorker(camera=None, tracker=FakeTracker())
        self.assertTrue(worker._claim_result_slot())
        self.assertFalse(worker._claim_result_slot())
        worker.mark_result_consumed()
        self.assertTrue(worker._claim_result_slot())


if __name__ == "__main__":
    unittest.main()
