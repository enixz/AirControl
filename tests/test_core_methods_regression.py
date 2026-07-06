"""核心方法回归测试 — 阶段 A 安全网。

为 BaseHandTracker 的三个核心方法补表征测试：
  - _perform_crop_zoom: crop-zoom 通用逻辑（裁剪→放大→坐标映射）
  - _update_zoom_mode:  zoom 状态机（far/near 阈值切换）
  - _priority_score:    主手锁定评分（高度/手大小/惯用手）

作为阶段 B（find_hands 拆分）的回归保护网。
"""
import os
import sys
import unittest
from unittest.mock import MagicMock

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'app'))

from services.base_hand_tracker import BaseHandTracker


class _DummyTracker(BaseHandTracker):
    """可实例化的测试用 tracker。"""

    engine_name = "dummy"

    def _detect(self, frame):
        return [], [], []

    def _detect_crop_zoom(self, frame, hint_center, hint_size):
        return [], [], []


def _make_tracker_with_mock_sr(**kwargs):
    """构造 DummyTracker 并用 MagicMock 替换 _sr，避免 ESPCN/ONNX 模型加载。

    默认 resolve 返回 'none' → crop-zoom 走 cv2.resize 插值路径。
    """
    config = kwargs.pop("config", {})
    config.setdefault("long_range_enabled", False)
    tracker = _DummyTracker(config=config)
    tracker._sr = MagicMock()
    tracker._sr.resolve.return_value = "none"
    tracker._sr.init.return_value = None
    tracker._sr.log_tier.return_value = None
    return tracker


# ---------------------------------------------------------------------------
# _perform_crop_zoom
# ---------------------------------------------------------------------------

class TestPerformCropZoom(unittest.TestCase):
    """_perform_crop_zoom: 裁剪→放大→子类检测→坐标映射。"""

    def test_returns_empty_when_crop_size_near_full_frame(self):
        """crop_size >= 0.95*min(w,h) → 返回 ([], [], []) 不调用 run_sub_detect。"""
        tracker = _make_tracker_with_mock_sr()
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        called = [False]

        def run_sub_detect(zoomed):
            called[0] = True
            return [], [], []

        ret = tracker._perform_crop_zoom(
            frame, crop_center=(100, 100), current_crop_size=195,
            run_sub_detect=run_sub_detect
        )
        self.assertEqual(ret, ([], [], []))
        self.assertFalse(called[0], "crop_size 接近全帧时不应调用 run_sub_detect")

    def test_returns_empty_when_frame_too_small(self):
        """帧本身小于 _crop_min_size 时 crop 被 clamp 后仍触发早返回。"""
        tracker = _make_tracker_with_mock_sr()
        # _crop_min_size 默认 32；frame 30x30 → crop_size=max(32,..) clamp 到 30
        # 30 >= 0.95*30=28.5 → 早返回
        frame = np.zeros((30, 30, 3), dtype=np.uint8)
        ret = tracker._perform_crop_zoom(
            frame, crop_center=(15, 15), current_crop_size=32,
            run_sub_detect=lambda z: ([], [], [])
        )
        self.assertEqual(ret, ([], [], []))

    def test_coordinates_mapped_back_to_original_frame(self):
        """to_orig(norm_x, norm_y) 应映射回原帧像素坐标。

        公式：to_orig(nx, ny) = (x0 + nx*target*scale, y0 + ny*target*scale)
        其中 scale = crop_size / target，故 target*scale = crop_size。
        """
        tracker = _make_tracker_with_mock_sr()
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        # crop_center=(100,100), crop_size=100 → x0=y0=50, crop=[50:150,50:150]
        # target=384, scale=100/384
        result, crop_info, to_orig = tracker._perform_crop_zoom(
            frame, crop_center=(100, 100), current_crop_size=100,
            run_sub_detect=lambda z: ([], [], [])
        )
        x0, y0, crop_size, target, scale = crop_info
        self.assertEqual((x0, y0), (50, 50))
        self.assertEqual(crop_size, 100)
        self.assertEqual(target, tracker._crop_target_size)
        self.assertAlmostEqual(scale, 100.0 / tracker._crop_target_size)

        # crop 中心 (0.5, 0.5) 应映射回原帧 (100, 100)
        ox, oy = to_orig(0.5, 0.5)
        self.assertAlmostEqual(ox, 100.0, places=5)
        self.assertAlmostEqual(oy, 100.0, places=5)
        # crop 左上角 (0, 0) → (x0, y0) = (50, 50)
        ox0, oy0 = to_orig(0.0, 0.0)
        self.assertAlmostEqual(ox0, 50.0, places=5)
        self.assertAlmostEqual(oy0, 50.0, places=5)

    def test_run_sub_detect_receives_zoomed_frame_of_target_size(self):
        """run_sub_detect 收到的帧尺寸 = (target, target, 3)。"""
        tracker = _make_tracker_with_mock_sr()
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        received_shape = [None]

        def run_sub_detect(zoomed):
            received_shape[0] = zoomed.shape
            return [], [], []

        tracker._perform_crop_zoom(
            frame, crop_center=(100, 100), current_crop_size=100,
            run_sub_detect=run_sub_detect
        )
        target = tracker._crop_target_size
        self.assertEqual(received_shape[0], (target, target, 3))

    def test_detection_result_passed_through_unchanged(self):
        """run_sub_detect 的返回值应原样作为 detection_result 返回。"""
        tracker = _make_tracker_with_mock_sr()
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        sentinel = (["hands"], ["gestures"], "raw")

        def run_sub_detect(zoomed):
            return sentinel

        result, _, _ = tracker._perform_crop_zoom(
            frame, crop_center=(100, 100), current_crop_size=100,
            run_sub_detect=run_sub_detect
        )
        self.assertIs(result, sentinel)


# ---------------------------------------------------------------------------
# _update_zoom_mode
# ---------------------------------------------------------------------------

class TestUpdateZoomMode(unittest.TestCase):
    """_update_zoom_mode: zoom 状态机 far/near 阈值切换。"""

    def _make_tracker(self):
        tracker = _make_tracker_with_mock_sr()
        # 固定阈值便于测试：far=0.008, near=0.040, streak=3
        tracker._zoom_far_threshold = 0.008
        tracker._zoom_near_threshold = 0.040
        tracker._zoom_switch_streak = 3
        return tracker

    def _feed(self, tracker, bbox_area, frame_w=1000, frame_h=1000):
        """喂一帧 bbox_area，调用 _update_zoom_mode。"""
        hands_landmarks = [[[0, 0.0, 0.0]]]
        hands_gestures = [{"bbox_area": float(bbox_area)}]
        tracker._update_zoom_mode(hands_landmarks, hands_gestures, frame_w, frame_h)

    def test_no_hands_returns_early(self):
        """hands_landmarks=[] → 不修改任何 streak 状态。"""
        tracker = self._make_tracker()
        tracker._far_streak = 5
        tracker._near_streak = 5
        tracker._crop_zoom_mode = True
        tracker._update_zoom_mode([], [], 1000, 1000)
        self.assertEqual(tracker._far_streak, 5)
        self.assertEqual(tracker._near_streak, 5)
        self.assertTrue(tracker._crop_zoom_mode)

    def test_far_streak_triggers_zoom_on(self):
        """bbox_ratio < far_threshold 连续 streak 帧 → _crop_zoom_mode=True。"""
        tracker = self._make_tracker()
        # frame_area=1e6, bbox=1000 → ratio=0.001 < 0.008 → far
        for _ in range(tracker._zoom_switch_streak):
            self._feed(tracker, bbox_area=1000)
        self.assertTrue(tracker._crop_zoom_mode)
        self.assertEqual(tracker._far_streak, 0, "进入 ZOOM 后 _far_streak 应清零")
        self.assertIsNone(tracker._current_crop_center, "进入 ZOOM 应置空 _current_crop_center")
        self.assertIsNone(tracker._current_crop_size, "进入 ZOOM 应置空 _current_crop_size")

    def test_near_streak_triggers_zoom_off(self):
        """先进 ZOOM ON，再喂 near 阈值连续 streak 帧 → _crop_zoom_mode=False。"""
        tracker = self._make_tracker()
        # 先进 ZOOM ON
        for _ in range(tracker._zoom_switch_streak):
            self._feed(tracker, bbox_area=1000)
        self.assertTrue(tracker._crop_zoom_mode)
        # 再喂 near: bbox=50000 → ratio=0.05 > 0.040 → near
        for _ in range(tracker._zoom_switch_streak):
            self._feed(tracker, bbox_area=50000)
        self.assertFalse(tracker._crop_zoom_mode)
        self.assertEqual(tracker._near_streak, 0, "退出 ZOOM 后 _near_streak 应清零")
        tracker._sr.reset_tier.assert_called_once_with()

    def test_middle_ratio_decays_both_streaks(self):
        """far < ratio < near → _far_streak 和 _near_streak 都衰减。"""
        tracker = self._make_tracker()
        # 先累积 streak
        tracker._far_streak = 2
        tracker._near_streak = 2
        # bbox=10000 → ratio=0.01，在 0.008~0.040 之间 → middle
        self._feed(tracker, bbox_area=10000)
        self.assertEqual(tracker._far_streak, 1)
        self.assertEqual(tracker._near_streak, 1)


# ---------------------------------------------------------------------------
# _priority_score
# ---------------------------------------------------------------------------

class TestPriorityScore(unittest.TestCase):
    """_priority_score: 主手锁定评分（阶段 2.11 恢复老版完整公式）。"""

    def test_dominant_hand_gets_bonus(self):
        """惯用手 +1.0 偏好（frame_w/h=0 时仅此项 + handedness_score 生效）。"""
        tracker = _make_tracker_with_mock_sr()
        tracker.dominant_hand = "Right"
        tracker._inference_max_width = 720

        score_right = tracker._priority_score(
            {"handedness": "Right", "bbox_area": 0.0, "handedness_score": 0.0},
            frame_w=0, frame_h=0
        )
        score_left = tracker._priority_score(
            {"handedness": "Left", "bbox_area": 0.0, "handedness_score": 0.0},
            frame_w=0, frame_h=0
        )
        self.assertAlmostEqual(score_right, 1.0)
        self.assertAlmostEqual(score_left, 0.0)

    def test_higher_wrist_gets_higher_score(self):
        """wrist_y 越小（举越高）分越高。"""
        tracker = _make_tracker_with_mock_sr()
        high_hand = [[0, 50.0, 10.0]]  # wrist_y=10 (高)
        low_hand = [[0, 50.0, 90.0]]   # wrist_y=90 (低)
        meta = {"handedness": "Unknown", "bbox_area": 1000.0, "handedness_score": 0.0}
        score_high = tracker._priority_score(meta, 100, 100, landmarks=high_hand)
        score_low = tracker._priority_score(meta, 100, 100, landmarks=low_hand)
        self.assertGreater(score_high, score_low)

    def test_bbox_size_affects_score(self):
        """bbox_area × 8 计入分数，大手得分略高。"""
        tracker = _make_tracker_with_mock_sr()
        landmarks = [[0, 50.0, 50.0]]
        meta_big = {"handedness": "Unknown", "bbox_area": 1000.0, "handedness_score": 0.0}
        meta_small = {"handedness": "Unknown", "bbox_area": 100.0, "handedness_score": 0.0}
        score_big = tracker._priority_score(meta_big, 100, 100, landmarks=landmarks)
        score_small = tracker._priority_score(meta_small, 100, 100, landmarks=landmarks)
        self.assertGreater(score_big, score_small)

    def test_motion_affects_score(self):
        """motion × 6 计入分数，运动大的手得分略高。"""
        tracker = _make_tracker_with_mock_sr()
        meta = {"handedness": "Right", "bbox_area": 500.0, "handedness_score": 0.0}
        landmarks = [[0, 50.0, 50.0]]
        score_no_motion = tracker._priority_score(meta, 100, 100, landmarks=landmarks, motion=0.0)
        score_with_motion = tracker._priority_score(meta, 100, 100, landmarks=landmarks, motion=1.0)
        self.assertGreater(score_with_motion, score_no_motion)


if __name__ == "__main__":
    unittest.main()
