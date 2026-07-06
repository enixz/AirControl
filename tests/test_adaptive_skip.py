"""T-Adaptive-Skip: 自适应推理频率测试

验证：
  1. 静态手势时跳帧（motion_ema 低）
  2. 动态手势时不跳帧（motion_ema 高）
  3. 跳帧时用 smoother 预测补帧
  4. 无 smoother 初始化时不跳帧
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


class TestAdaptiveSkip(unittest.TestCase):
    """测试自适应推理频率"""

    def setUp(self):
        self.tracker = _DummyTracker(config={})

    def test_skip_disabled_by_config(self):
        """config 禁用时永不跳帧"""
        tracker = _DummyTracker(config={"adaptive_skip_enabled": False})
        self.assertFalse(tracker._skip_enabled)
        self.assertFalse(tracker._should_skip_frame())

    def test_no_skip_without_initialized_smoother(self):
        """无初始化 smoother 时不跳帧"""
        self.tracker._skip_enabled = True
        self.tracker._motion_ema["Right"] = 0.0  # 静态
        # smoother 未初始化
        self.assertFalse(self.tracker._should_skip_frame())

    def test_skip_when_static(self):
        """静态手势时跳帧"""
        self.tracker._skip_enabled = True
        self.tracker._motion_ema["Right"] = 0.01  # 远低于阈值 0.15
        # 初始化 smoother
        self.tracker.smoothers["Right"].initialized = True
        self.tracker._active_handedness = {"Right"}

        # 第一次调用：counter=1 < interval=2，应该跳帧
        self.assertTrue(self.tracker._should_skip_frame())
        # 第二次调用：counter=2 >= interval=2，应该推理
        self.assertFalse(self.tracker._should_skip_frame())

    def test_no_skip_when_dynamic(self):
        """动态手势时不跳帧"""
        self.tracker._skip_enabled = True
        self.tracker._motion_ema["Right"] = 0.5  # 远高于阈值 0.15
        self.tracker.smoothers["Right"].initialized = True
        self.tracker._active_handedness = {"Right"}

        # 连续调用多次，都不应该跳帧
        for _ in range(5):
            self.assertFalse(self.tracker._should_skip_frame())

    def test_predict_skip_frame_returns_predicted(self):
        """跳帧时返回预测关键点"""
        self.tracker._skip_enabled = True
        self.tracker._active_handedness = {"Right"}

        # 初始化 smoother 并设置 last_landmarks
        smoother = self.tracker.smoothers["Right"]
        smoother.initialized = True
        smoother.lost_frames = 0
        smoother.last_landmarks = [[i, 100.0 + i, 100.0] for i in range(21)]

        # 设置上一帧手势
        self.tracker.last_gestures = [
            {"handedness": "Right", "label": "FIST", "ml_label": "Closed_Fist", "score": 0.9}
        ]

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = self.tracker._predict_skip_frame(frame, draw=False, w_frame=640, h_frame=480)

        self.assertIsNotNone(result)
        ret_frame, landmarks, gestures = result
        self.assertEqual(len(landmarks), 1)
        self.assertEqual(len(landmarks[0]), 21)
        self.assertTrue(gestures[0].get("predicted"))
        self.assertTrue(gestures[0].get("skipped"))

    def test_predict_skip_frame_returns_none_when_no_data(self):
        """无预测数据时返回 None"""
        self.tracker._skip_enabled = True
        self.tracker._active_handedness = {"Right"}
        # smoother 未初始化，无 last_landmarks
        self.tracker.smoothers["Right"].initialized = False

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = self.tracker._predict_skip_frame(frame, draw=False, w_frame=640, h_frame=480)
        self.assertIsNone(result)

    def test_skip_interval_configurable(self):
        """跳帧间隔可配置"""
        tracker = _DummyTracker(config={"skip_max_interval": 3})
        self.assertEqual(tracker._skip_max_interval, 3)

    def test_motion_threshold_configurable(self):
        """运动阈值可配置"""
        tracker = _DummyTracker(config={"skip_motion_threshold": 0.3})
        self.assertAlmostEqual(tracker._skip_motion_threshold, 0.3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
