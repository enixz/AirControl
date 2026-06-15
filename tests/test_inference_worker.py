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

    def test_tracker_swap_closes_old_tracker(self):
        old = FakeTracker()
        new = FakeTracker()
        worker = InferenceWorker(camera=None, tracker=old)
        worker.update_tracker(new)
        self.assertTrue(old.closed)
        self.assertIs(worker.tracker, new)

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
