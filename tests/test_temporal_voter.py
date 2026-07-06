"""T-Voter: 时序手势投票器单元测试

验证核心逻辑：
  1. 单帧高置信度 → 需多帧累积才确认（防抖）
  2. 滞回：确认后短暂低分不退出
  3. 距离自适应窗口
  4. 不应期：触发后短时间内不重复触发
  5. 手丢失重置
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'app'))

from services.temporal_voter import (
    MAX_WINDOW,
    MIN_WINDOW,
    REFERENCE_HAND_WIDTH,
    RELEASE_TIMEOUT_MS,
    TemporalGestureVoter,
)


def _make_gesture(ml_label, score=0.9):
    """构造一个 MediaPipe 风格的手势 dict。"""
    return {
        "ml_label": ml_label,
        "label": "OTHER",
        "score": score,
        "handedness": "Right",
        "handedness_score": 0.9,
        "bbox_area": 10000,
    }


class TestAdaptiveWindow(unittest.TestCase):
    """距离自适应窗口长度。"""

    def test_close_distance_uses_min_window(self):
        """近距离（掌宽=参考值）→ 最小窗口。"""
        voter = TemporalGestureVoter()
        w = voter._adaptive_window(REFERENCE_HAND_WIDTH)
        self.assertEqual(w, MIN_WINDOW)

    def test_far_distance_uses_max_window(self):
        """远距离（掌宽很小）→ 最大窗口。"""
        voter = TemporalGestureVoter()
        w = voter._adaptive_window(20.0)  # 很远
        self.assertEqual(w, MAX_WINDOW)

    def test_window_monotonic_with_distance(self):
        """手越小（越远）→ 窗口越长（单调）。"""
        voter = TemporalGestureVoter()
        w_near = voter._adaptive_window(90.0)
        w_mid = voter._adaptive_window(50.0)
        w_far = voter._adaptive_window(25.0)
        self.assertGreaterEqual(w_mid, w_near)
        self.assertGreaterEqual(w_far, w_mid)


class TestHysteresis(unittest.TestCase):
    """双阈值滞回决策。"""

    def test_single_frame_does_not_trigger(self):
        """单帧高置信度不立即触发（FSM 需要至少2帧：IDLE→DETECTING→CONFIRMED）。"""
        voter = TemporalGestureVoter()
        # 第1帧：进入 DETECTING，返回 None
        result1 = voter.update([_make_gesture("Closed_Fist", 0.95)], hand_width=90.0)
        self.assertEqual(result1, "None")
        # 第2帧：进入 CONFIRMED，返回手势名
        result2 = voter.update([_make_gesture("Closed_Fist", 0.95)], hand_width=90.0)
        self.assertEqual(result2, "Closed_Fist")

    def test_low_confidence_does_not_trigger(self):
        """低置信度单帧不触发。"""
        voter = TemporalGestureVoter()
        result = voter.update([_make_gesture("Closed_Fist", 0.3)], hand_width=90.0)
        self.assertEqual(result, "None")

    def test_hysteresis_prevents_flicker(self):
        """确认后短暂低分不退出（滞回）。"""
        voter = TemporalGestureVoter()
        # 用高置信度连续帧确认
        for _ in range(5):
            voter.update([_make_gesture("Closed_Fist", 0.9)], hand_width=90.0)
        self.assertEqual(voter._active_gesture, "Closed_Fist")

        # 一帧低分不应退出（exit_threshold=0.30）
        result = voter.update([_make_gesture("Closed_Fist", 0.2)], hand_width=90.0)
        self.assertEqual(result, "Closed_Fist")
        self.assertEqual(voter._active_gesture, "Closed_Fist")

    def test_exit_after_sustained_low_score(self):
        """持续低分后退出（RELEASING 超时 → IDLE）。"""
        voter = TemporalGestureVoter()
        for _ in range(5):
            voter.update([_make_gesture("Open_Palm", 0.9)], hand_width=90.0)
        self.assertEqual(voter._active_gesture, "Open_Palm")

        # 持续低分进入 RELEASING
        for _ in range(10):
            voter.update([_make_gesture("None", 0.0)], hand_width=90.0)
        # RELEASING 状态保持 _active_gesture，需超时才退出
        # 模拟时间前进超过 RELEASE_TIMEOUT_MS
        voter._state_enter_time -= RELEASE_TIMEOUT_MS + 100
        result = voter.update([_make_gesture("None", 0.0)], hand_width=90.0)
        self.assertEqual(result, "None")
        self.assertIsNone(voter._active_gesture)


class TestRefractory(unittest.TestCase):
    """每手势独立不应期。"""

    def test_refractory_blocks_immediate_retrigger(self):
        """触发后不应期内不重复触发。"""
        voter = TemporalGestureVoter()
        # 触发 Thumb_Up
        for _ in range(5):
            voter.update([_make_gesture("Thumb_Up", 0.95)], hand_width=90.0)
        self.assertEqual(voter._active_gesture, "Thumb_Up")

        # 持续低分进入 RELEASING
        for _ in range(10):
            voter.update([_make_gesture("None", 0.0)], hand_width=90.0)
        # 模拟时间前进超过 RELEASE_TIMEOUT_MS，让 RELEASING → IDLE
        voter._state_enter_time -= RELEASE_TIMEOUT_MS + 100
        result = voter.update([_make_gesture("None", 0.0)], hand_width=90.0)
        self.assertEqual(result, "None")
        self.assertIsNone(voter._active_gesture)

        # 立即再次给高置信度 → 不应期内不应触发
        # Thumb_Up refractory=800ms，_last_trigger_time 刚刚设置
        # 需要足够多帧让得分达到 enter_th，但 refractory 应阻止进入 CONFIRMED
        for _ in range(5):
            result = voter.update([_make_gesture("Thumb_Up", 0.95)], hand_width=90.0)
            self.assertEqual(result, "None")
        self.assertIsNone(voter._active_gesture)


class TestReset(unittest.TestCase):
    """手丢失重置。"""

    def test_reset_clears_state(self):
        voter = TemporalGestureVoter()
        for _ in range(5):
            voter.update([_make_gesture("Victory", 0.9)], hand_width=90.0)
        self.assertIsNotNone(voter._active_gesture)
        self.assertGreater(len(voter._window), 0)

        voter.reset()
        self.assertIsNone(voter._active_gesture)
        self.assertEqual(len(voter._window), 0)


class TestEmptyInput(unittest.TestCase):
    """空输入处理。"""

    def test_empty_gestures_returns_none(self):
        voter = TemporalGestureVoter()
        result = voter.update([], hand_width=90.0)
        self.assertEqual(result, "None")

    def test_none_label_ignored(self):
        """ml_label="None" 的手势不计入任何类别得分。"""
        voter = TemporalGestureVoter()
        for _ in range(10):
            voter.update([_make_gesture("None", 0.0)], hand_width=90.0)
        self.assertIsNone(voter._active_gesture)


class TestDistanceAdaptiveBehavior(unittest.TestCase):
    """远距离下行为更保守。"""

    def test_far_distance_needs_more_frames(self):
        """远距离窗口对单帧噪声冲击更鲁棒。

        场景：5帧稳定 Thumb_Up 后插入1帧 Closed_Fist 噪声。
        - 近距离窗口=5：噪声帧 displaces 一帧 Thumb_Up，噪声占比高
        - 远距离窗口=12：噪声帧只是 1/6，被历史 Thumb_Up 稀释
        → 远距离的噪声手势(Closed_Fist)得分应更低
        """
        voter_far = TemporalGestureVoter()
        voter_near = TemporalGestureVoter()

        # 5帧稳定 Thumb_Up
        for _ in range(5):
            g = _make_gesture("Thumb_Up", 0.9)
            voter_far.update([g], hand_width=25.0)
            voter_near.update([g], hand_width=90.0)

        # 1帧 Closed_Fist 噪声
        noise = _make_gesture("Closed_Fist", 0.7)
        voter_far.update([noise], hand_width=25.0)
        voter_near.update([noise], hand_width=90.0)

        far_scores = voter_far._compute_weighted_scores()
        near_scores = voter_near._compute_weighted_scores()

        # 远距离窗口更长，历史 Thumb_Up 帧更多，噪声 Closed_Fist 占比更低
        self.assertLess(
            far_scores["Closed_Fist"], near_scores["Closed_Fist"],
            "远距离窗口应使噪声手势得分更低（被更多历史帧稀释）",
        )


class TestWeightedScores(unittest.TestCase):
    """加权得分计算。"""

    def test_recent_frames_weighted_more(self):
        """新帧权重高于旧帧。"""
        voter = TemporalGestureVoter()
        # 旧帧：Victory 高分
        voter.update([_make_gesture("Victory", 0.9)], hand_width=90.0)
        voter.update([_make_gesture("Victory", 0.9)], hand_width=90.0)
        # 新帧：Closed_Fist 高分
        voter.update([_make_gesture("Closed_Fist", 0.9)], hand_width=90.0)
        voter.update([_make_gesture("Closed_Fist", 0.9)], hand_width=90.0)

        scores = voter._compute_weighted_scores()
        # Closed_Fist（新帧）得分应高于 Victory（旧帧）
        self.assertGreater(scores["Closed_Fist"], scores["Victory"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
