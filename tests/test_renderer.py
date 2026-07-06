"""T-Renderer: HandTrackerRenderer 表征测试

锁定从 base_hand_tracker.py 抽出的三个纯渲染方法行为：
  - draw_points: landmark 圆点绘制
  - draw_zoom_badge: ZOOM/FULL 状态徽章
  - apply_visual_zoom: 视觉放大视口

所有测试用 np.zeros 合成帧，不依赖真实模型。
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'app'))

import cv2
from services.renderer import HandTrackerRenderer


def _make_black_frame(w=640, h=480):
    return np.zeros((h, w, 3), dtype=np.uint8)


class TestDrawPoints(unittest.TestCase):
    """测试 draw_points 在帧上画 landmark 圆点。"""

    def setUp(self):
        self.renderer = HandTrackerRenderer(crop_min_size=32)

    def test_draws_non_black_on_black_frame(self):
        """在黑帧上画点后，帧不再全黑。"""
        frame = _make_black_frame()
        landmarks = [[i, 100.0 + i * 10, 100.0] for i in range(3)]
        self.renderer.draw_points(frame, landmarks, (255, 0, 255))
        self.assertGreater(frame.sum(), 0)

    def test_multiple_points_drawn(self):
        """画多个点后，每个点位置像素非零。"""
        frame = _make_black_frame()
        landmarks = [[0, 50.0, 50.0], [1, 100.0, 100.0], [2, 200.0, 200.0]]
        self.renderer.draw_points(frame, landmarks, (255, 0, 255))
        for _, x, y in landmarks:
            px = int(round(x))
            py = int(round(y))
            self.assertGreater(frame[py, px].sum(), 0)

    def test_empty_landmarks_no_change(self):
        """空 landmarks 列表不修改帧。"""
        frame = _make_black_frame()
        original = frame.copy()
        self.renderer.draw_points(frame, [], (255, 0, 255))
        np.testing.assert_array_equal(frame, original)

    def test_color_applied(self):
        """指定的颜色被应用到像素上。"""
        frame = _make_black_frame()
        color = (0, 255, 0)  # 绿色
        landmarks = [[0, 100.0, 100.0]]
        self.renderer.draw_points(frame, landmarks, color)
        px = int(round(100.0))
        py = int(round(100.0))
        # cv2.circle 用 FILLED 画实心圆，中心点颜色应接近指定颜色
        self.assertEqual(frame[py, px, 1], 255)  # G 通道


class TestDrawZoomBadge(unittest.TestCase):
    """测试 draw_zoom_badge 画 ZOOM/FULL 状态徽章。"""

    def setUp(self):
        self.renderer = HandTrackerRenderer(crop_min_size=32)

    def test_badge_draws_on_right_side(self):
        """画徽章后，帧右上角区域非原色（黑帧变非黑）。"""
        frame = _make_black_frame(w=640, h=480)
        gestures = [{"bbox_area": 5000.0}]
        self.renderer.draw_zoom_badge(frame, gestures, 640, 480, used_zoom=True)
        # 右上角区域应被徽章覆盖
        top_right = frame[5:30, 600:635]
        self.assertGreater(top_right.sum(), 0)

    def test_empty_gestures_no_exception(self):
        """hands_gestures=[] 时不抛异常（ratio_pct=0）。"""
        frame = _make_black_frame()
        try:
            self.renderer.draw_zoom_badge(frame, [], 640, 480, used_zoom=True)
        except Exception as e:
            self.fail(f"空 gestures 抛异常: {e}")

    def test_zoom_vs_full_different_pixels(self):
        """used_zoom=True (ZOOM) 和 False (FULL) 画出的帧像素不同。"""
        frame_zoom = _make_black_frame()
        frame_full = _make_black_frame()
        gestures = [{"bbox_area": 5000.0}]
        self.renderer.draw_zoom_badge(frame_zoom, gestures, 640, 480, used_zoom=True)
        self.renderer.draw_zoom_badge(frame_full, gestures, 640, 480, used_zoom=False)
        # 两帧不应完全相同（ZOOM 用橙色，FULL 用灰色）
        self.assertFalse(np.array_equal(frame_zoom, frame_full))

    def test_exception_swallowed(self):
        """异常被吞掉，不抛出（如 gestures 含非 dict 元素）。"""
        frame = _make_black_frame()
        # 传入会触发异常的 gestures（None 无 .get）
        try:
            self.renderer.draw_zoom_badge(frame, [None], 640, 480, used_zoom=True)
        except Exception as e:
            self.fail(f"异常未被吞掉: {e}")


class TestApplyVisualZoom(unittest.TestCase):
    """测试 apply_visual_zoom 裁剪放大显示视口。"""

    def setUp(self):
        self.renderer = HandTrackerRenderer(crop_min_size=32)

    def test_large_crop_returns_original(self):
        """crop_size >= min(w,h)*0.95 时返回原帧（不裁剪）。"""
        frame = _make_black_frame(w=640, h=480)
        # 480 * 0.95 = 456，传 460 应返回原帧
        result = self.renderer.apply_visual_zoom(frame, (320.0, 240.0), 460)
        self.assertIs(result, frame)

    def test_normal_crop_returns_same_size(self):
        """正常裁剪后返回帧尺寸 == 原帧尺寸（resize 回原尺寸）。"""
        frame = _make_black_frame(w=640, h=480)
        result = self.renderer.apply_visual_zoom(frame, (320.0, 240.0), 200)
        self.assertEqual(result.shape, frame.shape)

    def test_normal_crop_modifies_frame(self):
        """正常裁剪后帧内容变化（边框矩形绘制）。"""
        frame = _make_black_frame(w=640, h=480)
        # 先在原帧中心画一个白块
        frame[200:280, 300:380] = 255
        result = self.renderer.apply_visual_zoom(frame, (340.0, 240.0), 200)
        # 结果帧不应与原帧相同（裁剪放大 + 边框）
        self.assertFalse(np.array_equal(result, frame))

    def test_center_outside_frame_no_exception(self):
        """裁剪中心在帧外时不抛异常（边界钳制）。"""
        frame = _make_black_frame(w=640, h=480)
        try:
            result = self.renderer.apply_visual_zoom(frame, (-100.0, -100.0), 200)
        except Exception as e:
            self.fail(f"帧外中心抛异常: {e}")
        # 应返回有效帧
        self.assertEqual(result.shape, frame.shape)

    def test_crop_size_below_min_clamped(self):
        """crop_size 小于 crop_min_size 时被钳到下限。"""
        renderer = HandTrackerRenderer(crop_min_size=64)
        frame = _make_black_frame(w=640, h=480)
        # 传 10 应被钳到 64
        result = renderer.apply_visual_zoom(frame, (320.0, 240.0), 10)
        self.assertEqual(result.shape, frame.shape)

    def test_zero_size_returns_frame(self):
        """crop_size=0 时不抛异常（被钳到 crop_min_size）。"""
        frame = _make_black_frame(w=640, h=480)
        try:
            result = self.renderer.apply_visual_zoom(frame, (320.0, 240.0), 0)
        except Exception as e:
            self.fail(f"size=0 抛异常: {e}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
