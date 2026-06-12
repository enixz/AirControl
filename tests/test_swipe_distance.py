"""挥动手势的距离自适应阈值测试。

验证：同一幅度的挥动在"远处（掌窄）"能触发，在"近处（掌宽）"按比例需要更大幅度，
从而修复 ZOOM/远距离下 "avg_speed too low" 导致翻页失效的问题。
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'app'))

from services.gesture_recognizer import GestureRecognizer


def _make_rightward(recognizer, start=200, step=8, n=10, y=240):
    """构造一段远离边缘、方向一致的向右轨迹。"""
    recognizer.frame_w = 640
    recognizer.frame_h = 480
    recognizer.history_x = [start + step * i for i in range(n)]
    recognizer.history_y = [y for _ in range(n)]


class TestSwipeDistanceScaling(unittest.TestCase):
    def test_far_small_hand_triggers_swipe(self):
        """远处掌窄（30px）：小幅向右挥动（dx=72, speed=7.2）应触发 SWIPE_RIGHT。"""
        r = GestureRecognizer(cooldown=1.0, swipe_threshold=60)
        r._last_hand_width = 30.0          # 远距离
        _make_rightward(r)
        self.assertEqual(r._check_swipe(), "SWIPE_RIGHT")

    def test_near_large_hand_rejects_same_small_motion(self):
        """近处掌宽（150px）：同样的小幅挥动应被拒绝（近处真实挥动幅度会大得多）。"""
        r = GestureRecognizer(cooldown=1.0, swipe_threshold=60)
        r._last_hand_width = 150.0         # 近距离
        _make_rightward(r)
        self.assertEqual(r._check_swipe(), "NONE")

    def test_reference_distance_matches_legacy_threshold(self):
        """参考掌宽（=REFERENCE）时 scale=1.0：低速小幅挥动仍按原阈值被判为太慢。"""
        r = GestureRecognizer(cooldown=1.0, swipe_threshold=60)
        r._last_hand_width = r.REFERENCE_HAND_WIDTH
        _make_rightward(r)                 # avg_speed 7.2 < 15 → 太慢
        self.assertEqual(r._check_swipe(), "NONE")

    def test_far_fast_enough_up_swipe(self):
        """远处竖直方向同理：掌窄时小幅向上挥动可触发 SWIPE_UP。"""
        r = GestureRecognizer(cooldown=1.0, swipe_threshold=60)
        r._last_hand_width = 30.0
        r.frame_w = 640
        r.frame_h = 480
        # 向上 = y 递减；远离边缘
        r.history_x = [320 for _ in range(10)]
        r.history_y = [300 - 8 * i for i in range(10)]   # dy=-72
        self.assertEqual(r._check_swipe(), "SWIPE_UP")


if __name__ == "__main__":
    unittest.main(verbosity=2)
