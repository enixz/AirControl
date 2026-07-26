import os
import sys
import time
import unittest

import numpy as np
from PyQt6.QtWidgets import QApplication

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from main_ui import _make_preview_pixmap


class TestPreviewPerformance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_preview_preserves_aspect_ratio(self):
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        pixmap = _make_preview_pixmap(frame, 480, 360)
        self.assertEqual((pixmap.width(), pixmap.height()), (480, 270))

    def test_1080p_preview_conversion_stays_within_ui_budget(self):
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        for _ in range(10):
            _make_preview_pixmap(frame, 480, 360)
        samples = []
        for _ in range(100):
            started = time.perf_counter()
            _make_preview_pixmap(frame, 480, 360)
            samples.append((time.perf_counter() - started) * 1000.0)
        samples.sort()
        median_ms = samples[len(samples) // 2]
        self.assertLess(
            median_ms,
            10.0,
            f"1080p preview median {median_ms:.2f}ms exceeds 10ms budget",
        )


if __name__ == "__main__":
    unittest.main()
