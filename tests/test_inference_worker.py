import os
import sys
import threading
import time
import unittest
from unittest.mock import patch

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
    def test_capture_exception_reports_error_and_recovers_next_frame(self):
        frame = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)

        class RecoveringCamera:
            cap = object()
            calls = 0
            releases = 0

            def read_frame(self):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("camera read failed")
                worker._capture_stop.set()
                return True, frame

            def release(self):
                self.releases += 1

        camera = RecoveringCamera()
        worker = InferenceWorker(camera=camera, tracker=FakeTracker())
        errors = []
        worker.error_occurred.connect(errors.append)
        with patch.object(worker._capture_stop, "wait", return_value=False):
            worker._capture_loop()

        self.assertEqual(errors, ["camera read failed"])
        self.assertEqual(camera.releases, 1)
        self.assertEqual(camera.calls, 2)
        self.assertEqual(worker._latest_sequence, 1)
        np.testing.assert_array_equal(worker._latest_frame[1], frame[:, ::-1])

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

    def test_shadow_installed_mid_frame_does_not_crash_worker(self):
        """影子 TOCTOU：帧首无影子（clean_frame=None）后、find_hands 前装上
        影子（set_shadow_tracker 在推理锁外）。worker 必须用帧首快照——本帧
        跳过影子，而不是拿 None 帧喂影子崩溃。"""
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        shadow_calls = []

        class Shadow(FakeTracker):
            def find_hands(self, f, draw=True):
                shadow_calls.append(f)
                return f, [], []

        worker = InferenceWorker(camera=None, tracker=FakeTracker())
        received = []
        worker.shadow_hand_ready.connect(
            lambda source_frame, hands, gestures, source: received.append(hands)
        )
        # 模拟竞态：在帧首快照之后、推理中安装影子
        original_find = worker.tracker.find_hands

        def find_then_install(frame_arg, draw=True):
            worker.set_shadow_tracker(Shadow())  # 恰落在两步之间
            return original_find(frame_arg, draw=draw)

        worker.tracker.find_hands = find_then_install
        with patch.object(worker, "_update_fps"):
            worker._process_frame(frame, 0.0, time.perf_counter())
        # 快照语义：本帧影子不应被喂（无 None 帧调用），下一帧正常
        self.assertEqual(shadow_calls, [])
        self.assertEqual(received, [])
        # 下一帧：影子已装，正常跑
        worker.tracker.find_hands = original_find
        worker.tracker.find_hands = original_find
        with patch.object(worker, "_update_fps"):
            worker._process_frame(frame.copy(), 0.0, time.perf_counter())
        self.assertEqual(len(shadow_calls), 1)
        self.assertIsNotNone(shadow_calls[0])

    def test_frame_ready_carries_captured_at_for_frame_age(self):
        """frame_ready 随帧透传 captured_at（perf_counter 时钟），主线程用
        它算端到端帧龄（验收口径，见 docs/shadow_zoom_acceptance.md）。"""
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        worker = InferenceWorker(camera=None, tracker=FakeTracker())
        received = []

        def on_frame(source_frame, hands, gestures, source, captured_at):
            received.append((source, captured_at))

        worker.frame_ready.connect(on_frame)
        captured_at = time.perf_counter()
        with patch.object(worker, "_update_fps"):
            worker._process_frame(frame, 0.0, captured_at)
        self.assertEqual(len(received), 1)
        self.assertIs(received[0][0], worker)
        self.assertEqual(received[0][1], captured_at)

        # record_frame_age / frame_age_percentile：样本入队、无样本返回 None
        self.assertIsNone(worker.frame_age_percentile(0.95))
        worker.record_frame_age(120.0)
        worker.record_frame_age(140.0)
        self.assertEqual(worker.frame_age_percentile(0.5), 120.0)

    def test_backpressure_drop_counts_and_perf_metrics_include_shadow(self):
        """背压槽占满时 _dropped_results 计数；performance metrics 含影子
        耗时与丢帧计数（影子未跑时样本为空 deque，分位数为 0）。"""
        worker = InferenceWorker(camera=None, tracker=FakeTracker())
        self.assertTrue(worker._claim_result_slot())  # 占满唯一槽位
        worker._record_performance(1.0, 1.0, 1.0, 1.0)
        self.assertEqual(worker._dropped_results, 0)
        # 第二帧被丢
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        with patch.object(worker, "_update_fps"):
            worker._process_frame(frame, 0.0, time.perf_counter())
        self.assertEqual(worker._dropped_results, 1)
        # 影子耗时样本：跑了影子后 shadow_ms 非空
        self.assertFalse(worker._shadow_ms)

    def test_skip_migration_keeps_pending_state(self):
        """skip_migration 提交后，换装段不执行 migrate_state_from（影子晋升
        保留自身接续状态），seed 也跳过。"""
        events = []

        class Migrating(FakeTracker):
            def migrate_state_from(self, old):
                events.append("migrate")

            def seed_crop_zoom_from_hint(self):
                events.append("seed")
                return True

        old = FakeTracker()
        promoted = Migrating()
        worker = InferenceWorker(camera=None, tracker=old)
        worker.update_tracker(promoted, context={}, seed_crop_zoom=False,
                             skip_migration=True)
        worker._process_frame(np.zeros((2, 2, 3), dtype=np.uint8), 0.0,
                               time.perf_counter())
        self.assertEqual(events, [])
        self.assertIs(worker.tracker, promoted)
        self.assertTrue(old.closed)


if __name__ == "__main__":
    unittest.main()
