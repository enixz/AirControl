import os
import sys
import threading
import time
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from services.inference_worker import InferenceWorker


class FakeTracker:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True

    def find_hands(self, frame, draw=True):
        return frame, [], []


class TestInferenceWorker(unittest.TestCase):
    def test_stop_timeout_is_bounded_and_does_not_release_live_capture(self):
        entered = threading.Event()
        release = threading.Event()

        class BlockingCamera:
            cap = object()

            def read_frame(self):
                entered.set()
                release.wait()
                return False, None

        worker = InferenceWorker(camera=BlockingCamera(), tracker=FakeTracker())
        worker.start()
        self.assertTrue(entered.wait(timeout=1.0))

        started = time.perf_counter()
        self.assertFalse(worker.stop(timeout_ms=50))
        elapsed = time.perf_counter() - started

        self.assertLess(elapsed, 0.5)
        self.assertTrue(worker.isRunning())
        self.assertTrue(worker._capture_thread.is_alive())

        release.set()
        self.assertTrue(worker.stop(timeout_ms=1000))
        self.assertFalse(worker.isRunning())

    def test_update_tracker_does_not_wait_for_native_inference_lock(self):
        entered = threading.Event()
        release = threading.Event()

        class BlockingTracker(FakeTracker):
            def find_hands(self, frame, draw=True):
                entered.set()
                release.wait(timeout=2.0)
                return frame, [], []

        worker = InferenceWorker(camera=None, tracker=BlockingTracker())
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        inference = threading.Thread(
            target=lambda: worker._process_frame(
                frame,
                0.0,
                time.perf_counter(),
            )
        )
        inference.start()
        self.assertTrue(entered.wait(timeout=1.0))

        started = time.perf_counter()
        self.assertTrue(worker.update_tracker(FakeTracker()))
        elapsed = time.perf_counter() - started
        release.set()
        inference.join(timeout=1.0)

        self.assertLess(elapsed, 0.05)
        self.assertFalse(inference.is_alive())

    def test_stop_waits_until_blocking_camera_read_finishes(self):
        """stop 返回时采集线程必须已退出，调用方才能安全 release 摄像头。"""

        entered = threading.Event()
        release = threading.Event()

        class BlockingCamera:
            cap = object()

            def read_frame(self):
                entered.set()
                release.wait(timeout=2.0)
                return False, None

        worker = InferenceWorker(camera=BlockingCamera(), tracker=FakeTracker())
        worker.start()
        self.assertTrue(entered.wait(timeout=1.0))

        timer = threading.Timer(0.1, release.set)
        timer.start()
        started = time.perf_counter()
        worker.stop()
        elapsed = time.perf_counter() - started
        timer.join(timeout=1.0)

        self.assertGreaterEqual(elapsed, 0.08)
        self.assertFalse(worker.isRunning())
        self.assertIsNotNone(worker._capture_thread)
        self.assertFalse(worker._capture_thread.is_alive())

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
        worker._process_frame(
            np.zeros((2, 2, 3), dtype=np.uint8),
            0.0,
            time.perf_counter(),
        )
        self.assertTrue(old.closed)
        self.assertIs(worker.tracker, new)
        self.assertIsNone(worker._pending_tracker)

    def test_tracker_migration_seed_and_commit_are_worker_owned_and_ordered(self):
        events = []

        class MigratingTracker(FakeTracker):
            def migrate_state_from(self, old_tracker):
                self.migrated_from = old_tracker
                events.append("migrate")

            def seed_crop_zoom_from_hint(self):
                events.append("seed")
                return True

            def find_hands(self, frame, draw=True):
                events.append("find")
                return frame, [], []

        old = FakeTracker()
        new = MigratingTracker()
        worker = InferenceWorker(camera=None, tracker=old)
        commits = []
        worker.tracker_swapped.connect(
            lambda source, tracker, context, details: commits.append(
                (source, tracker, context, details)
            )
        )
        context = {"signature": ("mediapipe",), "request_id": 7}
        worker.update_tracker(
            new,
            context=context,
            seed_crop_zoom=True,
        )

        worker._process_frame(
            np.zeros((2, 2, 3), dtype=np.uint8),
            0.0,
            time.perf_counter(),
        )

        self.assertEqual(events, ["migrate", "seed", "find"])
        self.assertIs(new.migrated_from, old)
        self.assertTrue(old.closed)
        self.assertEqual(len(commits), 1)
        source, active, committed_context, details = commits[0]
        self.assertIs(source, worker)
        self.assertIs(active, new)
        self.assertEqual(committed_context, context)
        self.assertEqual(
            details,
            {"seed_requested": True, "seeded": True},
        )

    def test_rapid_tracker_updates_do_not_retire_latest_instance(self):
        active = FakeTracker()
        first = FakeTracker()
        second = FakeTracker()
        worker = InferenceWorker(camera=None, tracker=active)

        worker.update_tracker(first)
        worker.update_tracker(second)
        worker.update_tracker(first)

        self.assertIs(worker._pending_tracker, first)
        self.assertNotIn(first, worker._retired_pending_trackers)
        self.assertIn(second, worker._retired_pending_trackers)
        worker.stop()
        self.assertTrue(first.closed)
        self.assertTrue(second.closed)

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
