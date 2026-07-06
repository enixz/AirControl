"""T-Face-Guide: 人脸引导捕获表征测试

锁定 FaceGuide._ensure_detector / _guided_region / acquire 的行为，
为 FaceGuide 抽出提供回归保护。

acquire() 返回 (hands, gestures, raw, cx, cy, size) 或 None；
视口状态（_crop_zoom_mode 等）由 tracker 在 find_hands 中写入，不在 acquire 内。
"""
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'app'))

import cv2
from services.base_hand_tracker import BaseHandTracker


class _DummyTracker(BaseHandTracker):
    """可实例化的测试用 tracker。"""
    engine_name = "dummy"

    def _detect(self, frame):
        return [], [], []

    def _detect_crop_zoom(self, frame, hint_center, hint_size):
        return [], [], []


def _make_face_frame(w=640, h=480, face_rect=(100, 100, 80, 80)):
    """构造一帧带伪"人脸"（白色方块）的黑帧。"""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    x, y, fw, fh = face_rect
    frame[y:y+fh, x:x+fw] = 255
    return frame


class TestEnsureDetector(unittest.TestCase):
    """测试 FaceGuide._ensure_detector 的懒加载与缓存。"""

    def setUp(self):
        self.tracker = _DummyTracker(config={})
        self.fg = self.tracker._face_guide

    def test_first_call_loads_cascade(self):
        """首次调用加载级联，_face_detector_init 置 True。"""
        fake_cascade = MagicMock()
        fake_cascade.empty.return_value = False
        fake_path = os.path.abspath(__file__)
        with patch.object(cv2, 'data', MagicMock(haarcascades=os.path.dirname(fake_path))):
            with patch('os.path.exists', return_value=True):
                with patch('cv2.CascadeClassifier', return_value=fake_cascade):
                    result = self.fg._ensure_detector()
        self.assertIs(result, fake_cascade)
        self.assertTrue(self.fg._face_detector_init)

    def test_second_call_uses_cache(self):
        """二次调用走缓存，不重新加载。"""
        fake_cascade = MagicMock()
        fake_cascade.empty.return_value = False
        with patch.object(cv2, 'data', MagicMock(haarcascades='/fake')):
            with patch('os.path.exists', return_value=True):
                with patch('cv2.CascadeClassifier', return_value=fake_cascade) as mock_cc:
                    self.fg._ensure_detector()
                    self.fg._ensure_detector()
        self.assertEqual(mock_cc.call_count, 1)

    def test_no_haarcascades_dir_returns_none(self):
        """cv2.data.haarcascades 不存在时返回 None，不抛异常。"""
        with patch.object(cv2, 'data', MagicMock(haarcascades=None)):
            result = self.fg._ensure_detector()
        self.assertIsNone(result)
        self.assertTrue(self.fg._face_detector_init)

    def test_model_file_missing_returns_none(self):
        """级联文件不存在时返回 None，不抛异常。"""
        with patch.object(cv2, 'data', MagicMock(haarcascades='/fake')):
            with patch('os.path.exists', return_value=False):
                result = self.fg._ensure_detector()
        self.assertIsNone(result)

    def test_empty_classifier_returns_none(self):
        """级联加载但 empty()=True 时返回 None。"""
        fake_cascade = MagicMock()
        fake_cascade.empty.return_value = True
        with patch.object(cv2, 'data', MagicMock(haarcascades='/fake')):
            with patch('os.path.exists', return_value=True):
                with patch('cv2.CascadeClassifier', return_value=fake_cascade):
                    result = self.fg._ensure_detector()
        self.assertIsNone(result)


class TestGuidedRegion(unittest.TestCase):
    """测试 FaceGuide._guided_region 的人脸检测与坐标映射。"""

    def setUp(self):
        self.tracker = _DummyTracker(config={})
        self.fg = self.tracker._face_guide
        self.fake_cascade = MagicMock()

    def _patch_cascade(self):
        return patch.object(self.fg, '_ensure_detector', return_value=self.fake_cascade)

    def test_no_cascade_returns_none(self):
        """级联不可用时返回 None。"""
        with patch.object(self.fg, '_ensure_detector', return_value=None):
            frame = _make_face_frame()
            result = self.fg._guided_region(frame)
        self.assertIsNone(result)

    def test_no_faces_returns_none(self):
        """未检到人脸时返回 None。"""
        self.fake_cascade.detectMultiScale.return_value = []
        with self._patch_cascade():
            frame = _make_face_frame()
            result = self.fg._guided_region(frame)
        self.assertIsNone(result)

    def test_face_detected_returns_region(self):
        """检到人脸时返回 (cx, cy, size)，坐标映射回原帧。

        帧 640x480，_face_detect_short=400 → scale=400/480=0.833。
        mock 返回 scaled 帧中的 (100,100,80,80)，映射回原帧 (120,120,96,96)。
        """
        self.fake_cascade.detectMultiScale.return_value = [(100, 100, 80, 80)]
        with self._patch_cascade():
            frame = _make_face_frame(w=640, h=480)
            result = self.fg._guided_region(frame)
        self.assertIsNotNone(result)
        cx, cy, size = result
        # mapped: fx=120, fy=120, fw=96, fh=96
        # face_cx = 120 + 96/2 = 168
        self.assertAlmostEqual(cx, 168.0, places=1)
        # region_cy = (120+48) + 96*1.0 = 264
        self.assertAlmostEqual(cy, 264.0, places=1)
        # region_size = 96 * 7.0 = 672
        self.assertAlmostEqual(size, 672.0, places=1)

    def test_largest_face_selected(self):
        """多个人脸时选最大的。"""
        self.fake_cascade.detectMultiScale.return_value = [
            (50, 50, 40, 40),    # 小脸
            (200, 100, 100, 100),  # 大脸
        ]
        with self._patch_cascade():
            frame = _make_face_frame(w=640, h=480)
            result = self.fg._guided_region(frame)
        self.assertIsNotNone(result)
        cx, cy, size = result
        # scale=0.833: (200,100,100,100) → (240,120,120,120)
        # face_cx = 240 + 60 = 300
        self.assertAlmostEqual(cx, 300.0, places=1)
        # region_size = 120 * 7.0 = 840
        self.assertAlmostEqual(size, 840.0, places=1)

    def test_downscale_applied_for_large_frame(self):
        """大帧被缩小到 _face_detect_short 短边后检测，坐标映射回原帧。

        帧 800x600，_face_detect_short=400 → scale=400/600=0.667。
        mock 返回 scaled 帧中的 (200,200,160,160)，映射回原帧 (300,300,240,240)。
        """
        self.fg._face_detect_short = 400
        self.fake_cascade.detectMultiScale.return_value = [(200, 200, 160, 160)]
        with self._patch_cascade():
            frame = _make_face_frame(w=800, h=600)
            result = self.fg._guided_region(frame)
        self.assertIsNotNone(result)
        cx, cy, size = result
        # mapped: fx=300, fy=300, fw=240, fh=240
        # face_cx = 300 + 240/2 = 420
        self.assertAlmostEqual(cx, 420.0, places=1)
        # region_size = 240 * 7.0 = 1680
        self.assertAlmostEqual(size, 1680.0, places=1)

    def test_detect_exception_returns_none(self):
        """detectMultiScale 抛异常时返回 None。"""
        self.fake_cascade.detectMultiScale.side_effect = RuntimeError("boom")
        with self._patch_cascade():
            frame = _make_face_frame()
            result = self.fg._guided_region(frame)
        self.assertIsNone(result)


class TestAcquire(unittest.TestCase):
    """测试 FaceGuide.acquire 的节流、返回值、异常处理。"""

    def setUp(self):
        self.tracker = _DummyTracker(config={})
        self.fg = self.tracker._face_guide

    def test_disabled_returns_none(self):
        """_face_acquire_enabled=False 时直接返回 None。"""
        self.fg._face_acquire_enabled = False
        frame = _make_face_frame()
        cb = MagicMock(return_value=([], [], []))
        result = self.fg.acquire(frame, 640, 480, 32, cb)
        self.assertIsNone(result)

    def test_throttle_first_call_returns_none(self):
        """首次调用因节流返回 None（counter < interval）。"""
        self.fg._face_acquire_enabled = True
        self.fg._face_scan_interval = 4
        self.fg._face_scan_counter = 0
        frame = _make_face_frame()
        cb = MagicMock(return_value=([], [], []))
        result = self.fg.acquire(frame, 640, 480, 32, cb)
        self.assertIsNone(result)

    def test_throttle_after_interval_proceeds(self):
        """达到 interval 后不再节流，执行检测。"""
        self.fg._face_acquire_enabled = True
        self.fg._face_scan_interval = 2
        frame = _make_face_frame()
        cb = MagicMock(return_value=([], [], []))
        # 第一次调用：counter=1 < 2 → 节流
        self.fg.acquire(frame, 640, 480, 32, cb)
        # 第二次调用：counter=2 >= 2 → 执行检测
        with patch.object(self.fg, '_guided_region', return_value=None):
            result = self.fg.acquire(frame, 640, 480, 32, cb)
        self.assertIsNone(result)

    def test_hit_returns_six_tuple(self):
        """命中时返回 (hands, gestures, raw, cx, cy, size)。"""
        self.fg._face_acquire_enabled = True
        self.fg._face_scan_interval = 1  # 不节流
        fake_hands = [[0, 100.0, 100.0]]
        fake_gestures = [{"label": "OPEN"}]
        fake_raw = MagicMock()
        cb = MagicMock(return_value=(fake_hands, fake_gestures, fake_raw))
        with patch.object(self.fg, '_guided_region', return_value=(140.0, 220.0, 200.0)):
            frame = _make_face_frame()
            result = self.fg.acquire(frame, 640, 480, 32, cb)
        self.assertIsNotNone(result)
        hands, gestures, raw, cx, cy, size = result
        self.assertEqual(hands, fake_hands)
        self.assertEqual(gestures, fake_gestures)
        self.assertIs(raw, fake_raw)
        self.assertEqual(cx, 140.0)
        self.assertEqual(cy, 220.0)
        self.assertAlmostEqual(size, 200.0, places=1)

    def test_miss_no_result(self):
        """未命中（无区域）时返回 None。"""
        self.fg._face_acquire_enabled = True
        self.fg._face_scan_interval = 1
        cb = MagicMock(return_value=([], [], []))
        with patch.object(self.fg, '_guided_region', return_value=None):
            frame = _make_face_frame()
            result = self.fg.acquire(frame, 640, 480, 32, cb)
        self.assertIsNone(result)

    def test_detect_returns_empty_returns_none(self):
        """crop-zoom 检测返回空手时返回 None。"""
        self.fg._face_acquire_enabled = True
        self.fg._face_scan_interval = 1
        cb = MagicMock(return_value=([], [], []))
        with patch.object(self.fg, '_guided_region', return_value=(140.0, 220.0, 200.0)):
            frame = _make_face_frame()
            result = self.fg.acquire(frame, 640, 480, 32, cb)
        self.assertIsNone(result)

    def test_exception_swallowed(self):
        """_guided_region 抛异常时被吞掉，返回 None。"""
        self.fg._face_acquire_enabled = True
        self.fg._face_scan_interval = 1
        cb = MagicMock(return_value=([], [], []))
        with patch.object(self.fg, '_guided_region', side_effect=RuntimeError("boom")):
            frame = _make_face_frame()
            result = self.fg.acquire(frame, 640, 480, 32, cb)
        self.assertIsNone(result)

    def test_size_clamped_to_min(self):
        """size 小于 crop_min_size 时被钳到下限。"""
        self.fg._face_acquire_enabled = True
        self.fg._face_scan_interval = 1
        fake_hands = [[0, 50.0, 50.0]]
        cb = MagicMock(return_value=(fake_hands, [], MagicMock()))
        # region 返回 size=10，应被钳到 crop_min_size=32
        with patch.object(self.fg, '_guided_region', return_value=(100.0, 100.0, 10.0)):
            frame = _make_face_frame()
            result = self.fg.acquire(frame, 640, 480, 32, cb)
        self.assertIsNotNone(result)
        _, _, _, _, _, size = result
        self.assertAlmostEqual(size, 32.0, places=1)

    def test_size_clamped_to_frame(self):
        """size 大于 min(w,h) 时被钳到帧短边。"""
        self.fg._face_acquire_enabled = True
        self.fg._face_scan_interval = 1
        fake_hands = [[0, 50.0, 50.0]]
        cb = MagicMock(return_value=(fake_hands, [], MagicMock()))
        # region 返回 size=1000，帧 640x480 → 钳到 480
        with patch.object(self.fg, '_guided_region', return_value=(100.0, 100.0, 1000.0)):
            frame = _make_face_frame(w=640, h=480)
            result = self.fg.acquire(frame, 640, 480, 32, cb)
        self.assertIsNotNone(result)
        _, _, _, _, _, size = result
        self.assertAlmostEqual(size, 480.0, places=1)

    def test_callback_receives_frame_and_region(self):
        """detect_crop_zoom_cb 收到原帧 + 裁剪区参数。"""
        self.fg._face_acquire_enabled = True
        self.fg._face_scan_interval = 1
        fake_hands = [[0, 50.0, 50.0]]
        cb = MagicMock(return_value=(fake_hands, [], MagicMock()))
        with patch.object(self.fg, '_guided_region', return_value=(140.0, 220.0, 200.0)):
            frame = _make_face_frame()
            self.fg.acquire(frame, 640, 480, 32, cb)
        cb.assert_called_once_with(frame, (140.0, 220.0), 200.0)


class TestFindHandsViewportWrite(unittest.TestCase):
    """测试 find_hands 在人脸引导命中后正确写视口状态。"""

    def test_acquire_hit_writes_viewport_state(self):
        """find_hands 调 acquire 命中后，_crop_zoom_mode/_zoom_miss_streak/
        _current_crop_center/_current_crop_size 被正确写入。"""
        tracker = _DummyTracker(config={"long_range_enabled": True})
        tracker._face_guide._face_acquire_enabled = True
        tracker._face_guide._face_scan_interval = 1  # 不节流

        fake_hands = [[[i, 100.0, 100.0] for i in range(21)]]
        fake_gestures = [{"label": "OPEN", "handedness": "Right", "bbox_area": 500.0}]
        fake_raw = MagicMock()

        # _detect 返回空（模拟全帧没检到手）→ 触发人脸引导
        with patch.object(tracker, '_detect', return_value=([], [], [])):
            with patch.object(tracker._face_guide, '_guided_region',
                              return_value=(140.0, 220.0, 200.0)):
                with patch.object(tracker, '_detect_crop_zoom',
                                  return_value=(fake_hands, fake_gestures, fake_raw)):
                    frame = _make_face_frame()
                    tracker.find_hands(frame, draw=False)

        self.assertTrue(tracker._crop_zoom_mode)
        self.assertEqual(tracker._zoom_miss_streak, 0)
        self.assertEqual(tracker._current_crop_center, (140.0, 220.0))
        self.assertAlmostEqual(tracker._current_crop_size, 200.0, places=1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
