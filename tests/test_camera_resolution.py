import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from services.camera import _RESOLUTION_CANDIDATES


class TestCameraResolutionCandidates(unittest.TestCase):
    def test_automatic_probe_is_capped_at_1080p(self):
        self.assertLessEqual(max(height for _, height in _RESOLUTION_CANDIDATES), 1080)


if __name__ == "__main__":
    unittest.main()
