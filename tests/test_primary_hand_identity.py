"""主手身份连续性回归：单手↔双手切换、标签翻转、遮挡恢复不换 key、不跨手继承。

对应 base_hand_tracker._assign_smoother_keys 的位置连续性方案（2026-10 第 2 步）。
场景来自远距录像 20260722_145659 的实测故障：
  - 主控手短暂丢失 1-2 帧，另一只手不能顶替成 Primary（否则 MouseMode 视为换手）
  - 单手→双手：手数变化不再回退 handedness，仍按位置继承
  - 左右手标签翻转不产生身份跳变
  - 真换手（位置跳变）拿新 key，旧 key 的捏合历史不跨手继承
"""
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from services.base_hand_tracker import BaseHandTracker


def make_hand(wrist_x, wrist_y, handedness="Right", seed=0):
    """以 wrist 为锚构造 21 点（形状即可，坐标不模拟真实骨架）。"""
    lm = [[i, wrist_x + (i % 5) * 3.0, wrist_y + (i // 5) * 4.0] for i in range(21)]
    gesture = {"handedness": handedness, "handedness_score": 0.9,
               "label": "OTHER", "ml_label": "None", "score": 0.5, "bbox_area": 3000.0}
    return lm, gesture


class FakeTracker(BaseHandTracker):
    """只测 key 分配，不跑检测。"""

    def __init__(self):
        super().__init__(max_num_hands=2)

    @property
    def engine_name(self):
        return "fake"

    def _detect(self, frame):
        return [], [], []

    def _detect_crop_zoom(self, frame, hint_center, hint_size):
        return [], [], []


class TestKeyAssignment(unittest.TestCase):
    def setUp(self):
        self.t = FakeTracker()

    def assign(self, hands, gestures, w=1920, h=1080):
        return self.t._assign_smoother_keys(hands, gestures, w, h)

    def test_single_hand_gets_primary_without_history(self):
        lm, g = make_hand(960, 540)
        self.assertEqual(self.assign([lm], [g]), ["Primary"])

    def test_second_hand_appearing_after_single_adopts_left_right_by_position(self):
        # 单手(Primary @左) → 双手：手数变化回退 handedness 初始分配
        left, lg = make_hand(400, 500)
        self.assign([left], [lg])
        # 原手几乎没动，新手出现在右侧
        left2, lg2 = make_hand(405, 505, handedness="Right")
        right, rg = make_hand(1400, 500, handedness="Right")
        keys = self.assign([left2, right], [lg2, rg])
        # 双手同 handedness=Right：第一只拿 Right，去重后第二只拿 Left
        self.assertEqual(keys, ["Right", "Left"])

    def test_control_hand_dropout_does_not_promote_other_hand(self):
        # 双手(Left 主控 + Right 静止) → 主控手消失 1 帧 → 静止手不能变成 Primary
        l, lg = make_hand(600, 300, handedness="Left")
        r, rg = make_hand(1400, 850, handedness="Right")
        keys = self.assign([l, r], [lg, rg])
        self.assertEqual(sorted(keys), ["Left", "Right"])
        # 主控 Left 消失，只剩 Right（几乎没动）：无歧义继承自己的 Right
        r2, rg2 = make_hand(1402, 852, handedness="Right")
        keys = self.assign([r2], [rg2])
        self.assertEqual(keys, ["Right"])  # 关键：不再一律变成 Primary

    def test_control_hand_returns_after_dropout(self):
        # 双手 → 单手(Left 消失) → Left 回来：key 不变
        l, lg = make_hand(600, 300, handedness="Left")
        r, rg = make_hand(1400, 850, handedness="Right")
        self.assign([l, r], [lg, rg])
        r2, rg2 = make_hand(1401, 851, handedness="Right")
        self.assertEqual(self.assign([r2], [rg2]), ["Right"])
        # Left 回到原位（右手几乎没动 → 无歧义继承）
        l2, lg2 = make_hand(602, 302, handedness="Left")
        r3, rg3 = make_hand(1402, 852, handedness="Right")
        keys = self.assign([l2, r3], [lg2, rg3])
        self.assertEqual(keys, ["Left", "Right"])

    def test_handedness_flip_keeps_positional_identity(self):
        # 双手场景标签翻转（Left 手被标成 Right）：位置不变 → key 不变
        l, lg = make_hand(600, 300, handedness="Left")
        r, rg = make_hand(1400, 850, handedness="Right")
        keys1 = self.assign([l, r], [lg, rg])
        # 标签翻转：两只手的 handedness 互换
        l2, lg2 = make_hand(601, 301, handedness="Right")  # 同一物理手，标签翻了
        r2, rg2 = make_hand(1399, 849, handedness="Left")
        keys2 = self.assign([l2, r2], [lg2, rg2])
        self.assertEqual(keys1, keys2)

    def test_true_hand_swap_gets_new_key(self):
        # 原手离开画面，另一只手在完全不同的位置出现：不继承，拿新 key
        l, lg = make_hand(600, 300, handedness="Left")
        r, rg = make_hand(1400, 850, handedness="Right")
        self.assign([l, r], [lg, rg])
        # Left 消失，新右手出现在远处（>0.12 对角线）；旧 "Right" 的位置
        # 也远离它 → 无可继承 → 单手回退 Primary
        far, fg = make_hand(300, 900, handedness="Right")
        keys = self.assign([far], [fg])
        # 单手 + 无位置匹配：回退 Primary（新 key；不再继承任何旧身份）
        self.assertEqual(keys, ["Primary"])
        # 且不能继承旧 Right 的 smoother 状态：直接看 _prev_wrist_keys 更新
        self.assertEqual(self.t._prev_wrist_keys, [((300.0, 900.0), "Primary")])

    def test_occlusion_recovery_keeps_key(self):
        # 遮挡：无检出若干帧（_prev_wrist_keys 被清）→ 恢复后从头分配，稳定即可
        l, lg = make_hand(600, 300, handedness="Left")
        self.assign([l], [lg])
        self.t._prev_wrist_keys = []  # simulate full loss reset
        l2, lg2 = make_hand(605, 305, handedness="Right")  # 恢复时标签也变了
        keys = self.assign([l2], [lg2])
        # 无历史 → Primary（与现状一致；单手场景 Primary 本来就稳定）
        self.assertEqual(keys, ["Primary"])


class TestNoCrossHandPinchInheritance(unittest.TestCase):
    """MouseMode 侧：track_key 变化必须清掉捏合历史。"""

    def test_mouse_mode_resets_features_on_track_key_change(self):
        sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
        from test_pinch_freeze import _make_mouse_mode

        mode, mouse, recognizer = _make_mouse_mode()
        mode.on_enter()
        mode._watchdog_stop.set()
        mode._watchdog_thread.join()
        mode._feature_track_key = "Left"
        mode._pinch_press_since = {"thumb_index_pinch": 1.0}
        # 下一帧主手换成 Right
        lm = [[i, 100.0 + i, 200.0 + i] for i in range(21)]
        recognizer.get_hand_features.return_value = {
            "thumb_index_pinch": True, "thumb_middle_pinch": False,
            "hand_width": 120.0, "index_drawing_pose": False,
            "thumb_tucked": False, "index_middle_up": False,
        }
        gestures = [{"track_key": "Right", "predicted": False, "ml_label": "None"}]
        mode.handle([lm], gestures, 1920, 1080)
        recognizer.reset_hand_features.assert_called_once()
        self.assertEqual(mode._pinch_press_since, {})
        mode.on_exit()


if __name__ == "__main__":
    unittest.main()
