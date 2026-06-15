"""🤟 保持切模式手势的单元测试。

验证基于 MediaPipe ML 标签（I_LOVE_YOU）的滑动时间窗多数投票逻辑：
- 满窗高占比触发、低占比/短保持不触发
- 偶发单帧漏检与短暂丢手不影响触发（这正是旧版四段"握拳-张开"序列
  在远距离失效的原因）
- 触发后须放下手势重新武装，持续保持不会连环切换
- 不依赖 30fps 假设，低帧率（8fps）同样可触发
"""
import contextlib
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'app'))

from mode_manager import ModeManager


class FakeConfig:
    def __init__(self, data=None):
        self.data = dict(data or {})

    def get(self, key, default=None):
        return self.data.get(key, default)

    def set(self, key, value):
        self.data[key] = value

    @contextlib.contextmanager
    def batch_update(self):
        yield


class FakeMode:
    def on_enter(self):
        pass

    def on_exit(self):
        pass


HAND = [[0, 0.5, 0.5]]  # 任意非空 landmarks


class IlyHoldTest(unittest.TestCase):
    def setUp(self):
        self.mm = ModeManager(
            modes={"presentation": FakeMode(), "mouse": FakeMode(), "draw": FakeMode()},
            config=FakeConfig(),
            recognizer=None,
        )
        self.now = [1000.0]
        patcher = mock.patch("mode_manager.time")
        self.addCleanup(patcher.stop)
        patcher.start().time.side_effect = lambda: self.now[0]
        self.mm.switch_to("presentation")
        self.now[0] += 2.0  # 跳过切换后 1.5s 手势保护期

    def feed(self, duration, fps=17, label_for_frame=lambda i: "I_LOVE_YOU", hand=True):
        """以给定帧率连续喂帧，返回期间是否发生过模式切换。"""
        switched = False
        for i in range(int(duration * fps)):
            self.now[0] += 1.0 / fps
            if hand:
                gestures = [{"label": label_for_frame(i)}]
                if self.mm.maybe_switch_by_gesture([HAND], gestures):
                    switched = True
            else:
                if self.mm.maybe_switch_by_gesture([], []):
                    switched = True
        return switched

    def test_full_hold_switches_once(self):
        """17fps 下 100% 标签保持 1.2s：触发一次切换（演示→鼠标）。"""
        self.assertTrue(self.feed(1.2))
        self.assertEqual(self.mm.current_mode_name, "mouse")

    def test_flicker_70_percent_still_switches(self):
        """30% 帧漏检（标签抖动为 OTHER）仍应触发——投票容错的核心收益。"""
        def flicker(i):
            return "I_LOVE_YOU" if i % 10 < 7 else "OTHER"

        self.assertTrue(self.feed(1.5, label_for_frame=flicker))

    def test_half_ratio_rejected(self):
        """占比 50% 低于 0.6 阈值：不触发。"""
        def half(i):
            return "I_LOVE_YOU" if i % 2 == 0 else "OTHER"

        self.assertFalse(self.feed(1.5, label_for_frame=half))
        self.assertEqual(self.mm.current_mode_name, "presentation")

    def test_short_hold_rejected(self):
        """只保持 0.5s（不足时间窗的 90%）：不触发。"""
        self.assertFalse(self.feed(0.5))
        self.assertEqual(self.mm.current_mode_name, "presentation")

    def test_continuous_hold_does_not_chain_switch(self):
        """触发后持续摆 🤟 不应连环切换；放下并重新保持后才能再切。"""
        self.assertTrue(self.feed(1.2))
        self.assertEqual(self.mm.current_mode_name, "mouse")
        # 继续保持 3 秒：未重新武装，不得再切
        self.assertFalse(self.feed(3.0))
        self.assertEqual(self.mm.current_mode_name, "mouse")
        # 放下手势 1.2s（重新武装），再保持 1.2s：第二次切换（鼠标→板书）
        self.assertFalse(self.feed(1.2, label_for_frame=lambda i: "OTHER"))
        self.assertTrue(self.feed(1.2))
        self.assertEqual(self.mm.current_mode_name, "draw")

    def test_missing_frames_after_switch_do_not_rearm(self):
        """保护窗口或丢手不是明确放下，不得让持续手势再次触发。"""
        self.assertTrue(self.feed(1.2))
        self.assertEqual(self.mm.current_mode_name, "mouse")
        self.assertFalse(self.feed(1.2, hand=False))
        self.assertFalse(self.feed(1.2))
        self.assertEqual(self.mm.current_mode_name, "mouse")

    def test_candidate_flag_tracks_real_ily_only(self):
        self.feed(0.1)
        self.assertTrue(self.mm.is_switch_candidate)
        self.feed(0.1, label_for_frame=lambda i: "OTHER")
        self.assertFalse(self.mm.is_switch_candidate)

    def test_brief_hand_loss_tolerated(self):
        """保持中途丢手 0.2s（小于半个时间窗）：不清空采样，仍可触发。"""
        self.assertFalse(self.feed(0.5))
        self.assertFalse(self.feed(0.2, hand=False))
        self.assertTrue(self.feed(0.6))

    def test_long_hand_loss_resets(self):
        """丢手超过半个时间窗：采样清空，恢复后须重新累计。"""
        self.assertFalse(self.feed(0.8))
        self.assertFalse(self.feed(0.7, hand=False))
        # 恢复后仅 0.4s 不足以触发
        self.assertFalse(self.feed(0.4))
        self.assertEqual(self.mm.current_mode_name, "presentation")

    def test_low_fps_still_works(self):
        """8fps（远低于标称 30fps）下保持 1.2s 仍触发——不做帧率假设。"""
        self.assertTrue(self.feed(1.2, fps=8))

    def test_second_hand_ily_counts(self):
        """两只手时任意一只摆 🤟 均计入。"""
        switched = False
        for _ in range(int(1.2 * 17)):
            self.now[0] += 1.0 / 17
            gestures = [{"label": "OTHER"}, {"label": "I_LOVE_YOU"}]
            if self.mm.maybe_switch_by_gesture([HAND, HAND], gestures):
                switched = True
        self.assertTrue(switched)

    def test_old_grab_release_sequence_no_longer_switches(self):
        """旧版"握拳-张开-握拳-张开"序列不再触发切换。"""
        def seq(i):
            return ("FIST", "OPEN")[(i // 8) % 2]

        self.assertFalse(self.feed(2.5, label_for_frame=seq))
        self.assertEqual(self.mm.current_mode_name, "presentation")


if __name__ == "__main__":
    unittest.main()
