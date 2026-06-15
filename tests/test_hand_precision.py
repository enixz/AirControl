import os
import sys
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from services.base_hand_tracker import BaseHandTracker, OneEuroFilter, OneEuroSmoother
from services.hand_tracker import HandTracker
from services.mouse_controller import blended_landmark_point


class DummyTracker(BaseHandTracker):
    def __init__(self):
        super().__init__(max_num_hands=1)
        self.crop_calls = 0
        self.full_calls = 0

    @property
    def engine_name(self):
        return "dummy"

    def _detect_crop_zoom(self, frame, hint_center, hint_size):
        self.crop_calls += 1
        return [], [], []

    def _detect(self, frame):
        self.full_calls += 1
        landmarks = [[i, 100.25 + i, 120.75 + i] for i in range(21)]
        gesture = {
            "label": "POINTING_UP",
            "handedness": "Right",
            "handedness_score": 1.0,
            "bbox_area": 2500.0,
        }
        return [landmarks], [gesture], []


class TestHandPrecision(unittest.TestCase):
    def test_one_euro_uses_raw_samples_for_derivative(self):
        filt = OneEuroFilter(0.0, 0.0, min_cutoff=0.5, beta=1.0)
        filt(0.1, 10.0)
        filt(0.2, 10.0)
        self.assertAlmostEqual(filt.x_raw_prev, 10.0)
        self.assertLess(abs(filt.dx_prev), 50.0)

    def test_smoother_preserves_fractional_coordinates(self):
        smoother = OneEuroSmoother()
        landmarks = [[i, i + 0.25, i + 0.75] for i in range(21)]
        result = smoother.update(landmarks)
        self.assertIsInstance(result[0][1], float)
        self.assertEqual(result[0][1], 0.25)
        self.assertEqual(result[0][2], 0.75)

    def test_mediapipe_extraction_keeps_subpixel_coordinates(self):
        tracker = object.__new__(HandTracker)
        tracker._is_pure_landmarker = True
        detection = mock.Mock()
        detection.hand_landmarks = [
            [mock.Mock(x=0.1005 + i * 0.001, y=0.2005 + i * 0.001) for i in range(21)]
        ]
        detection.handedness = []
        hands, _, _ = tracker._extract_results(detection, 1000, 500, min_area_ratio=0.0)
        self.assertAlmostEqual(hands[0][0][1], 100.5)
        self.assertAlmostEqual(hands[0][0][2], 100.25)

    def test_blended_pointer_rejects_isolated_tip_jump(self):
        landmarks = [[i, 0.0, 0.0] for i in range(21)]
        for index in (9, 10, 11, 12):
            landmarks[index][1] = 100.0
        baseline, _ = blended_landmark_point(
            landmarks, ((12, 0.5), (11, 0.25), (10, 0.15), (9, 0.1))
        )
        landmarks[12][1] += 20.0
        shifted, _ = blended_landmark_point(
            landmarks, ((12, 0.5), (11, 0.25), (10, 0.15), (9, 0.1))
        )
        self.assertAlmostEqual(shifted - baseline, 10.0)

    def test_zoom_miss_retries_full_frame_on_third_frame(self):
        tracker = DummyTracker()
        tracker._crop_zoom_mode = True
        tracker._last_hint_center = (160.0, 120.0)
        tracker._last_hint_size = 60.0
        tracker._current_crop_center = (160.0, 120.0)
        tracker._current_crop_size = 150.0
        frame = np.zeros((240, 320, 3), dtype=np.uint8)

        tracker.find_hands(frame, draw=False)
        tracker.find_hands(frame, draw=False)
        _, hands, _ = tracker.find_hands(frame, draw=False)

        self.assertEqual(tracker.crop_calls, 3)
        self.assertEqual(tracker.full_calls, 1)
        self.assertEqual(len(hands), 1)
        self.assertIsInstance(hands[0][0][1], float)


if __name__ == "__main__":
    unittest.main()
