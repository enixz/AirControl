"""T-Long-Range: 远距离专项优化测试

验证：
  1. 运动先验：wrist 速度计算和位置预测
  2. 多尺度检测：0.5x 缩小检测 + 坐标映射
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'app'))

import numpy as np
from services.base_hand_tracker import BaseHandTracker


class _DummyTracker(BaseHandTracker):
    """可实例化的测试用 tracker。"""
    engine_name = "dummy"

    def _detect(self, frame):
        return [], [], []

    def _detect_crop_zoom(self, frame, cx, cy, size):
        return [], [], []


class TestMultiscaleDetection(unittest.TestCase):
    """测试多尺度检测"""

    def setUp(self):
        self.tracker = _DummyTracker(config={})

    def test_multiscale_returns_none_when_no_detection(self):
        """检测不到手时返回 None"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # _detect 返回空列表
        result = self.tracker._detect_multiscale(frame, 640, 480)
        self.assertIsNone(result)

    def test_multiscale_scales_coordinates(self):
        """多尺度检测后坐标被映射回原图"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # mock _detect 在缩小图上检到手
        def mock_detect(small_frame):
            # 缩小图上的关键点坐标
            landmarks = [[i, 50.0 + i, 50.0] for i in range(21)]
            gestures = [{"handedness": "Right", "bbox_area": 500.0, "score": 0.9}]
            return [landmarks], gestures, []

        self.tracker._detect = mock_detect
        result = self.tracker._detect_multiscale(frame, 640, 480)

        self.assertIsNotNone(result)
        hands_landmarks, hands_gestures, _ = result
        self.assertEqual(len(hands_landmarks), 1)

        # 坐标应被 ×2 映射回原图
        lm = hands_landmarks[0]
        self.assertAlmostEqual(lm[0][1], 100.0, delta=1.0)  # 50 * 2
        self.assertAlmostEqual(lm[0][2], 100.0, delta=1.0)  # 50 * 2

        # bbox_area 应被 ×4
        self.assertAlmostEqual(hands_gestures[0]["bbox_area"], 2000.0, delta=1.0)

    def test_multiscale_handles_exception(self):
        """异常时返回 None"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        def mock_detect(small_frame):
            raise RuntimeError("test error")

        self.tracker._detect = mock_detect
        result = self.tracker._detect_multiscale(frame, 640, 480)
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
