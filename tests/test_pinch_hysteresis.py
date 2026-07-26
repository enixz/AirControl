"""实施方案 Phase 3.2: Pinch 双阈值滞回单元测试

验证 ENTER/EXIT 双阈值滞回逻辑：
1. 默认关闭时保持旧版单阈值行为（PINCH_RATIO=0.35）
2. 开启后 ENTER（0.30）比旧阈值更严格，边界附近不误进入
3. 开启后 EXIT（0.40）比旧阈值更宽松，已捏合时不易退出
4. 滞回带 0.30-0.40 内状态稳定不跳变
5. _reset_state() 清除滞回状态

注意：单元测试只测滞回逻辑本身，不测实际手势场景效果（参考项目 memory 教训）。
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'app'))

from services.gesture_recognizer import GestureRecognizer


def make_landmark(idx, x, y, z=0.0):
    return [idx, float(x), float(y), float(z)]


def make_hand(landmark_xy_pairs):
    """从 [(idx, x, y), ...] 列表创建完整 21 点 landmark 数组。"""
    coords = {idx: (x, y) for idx, x, y in landmark_xy_pairs}
    return [make_landmark(i, coords.get(i, (0, 0))[0], coords.get(i, (0, 0))[1]) for i in range(21)]


def make_hand_with_pinch_ratio(ratio, hand_width=90.0):
    """创建正面对相机、指定 pinch 比值的手。

    pinch_ratio = thumb_index_distance / hand_width
    hand_width = distance(landmark 5, landmark 17) = hand_width 参数

    布局：掌根在下方，手指向上。拇指尖和食指尖的距离 = ratio * hand_width。
    """
    cx, cy = 200.0, 200.0
    hw = hand_width
    # 5 和 17 在同一 y，间距 = hw
    half_w = hw / 2.0
    target_dist = ratio * hw  # 期望的拇指-食指距离

    # 食指尖在 (cx, cy - 60)，拇指尖在 (cx + target_dist, cy - 60)
    # 这样距离 = target_dist
    return make_hand([
        (0, cx, cy + 40),                    # wrist
        (5, cx - half_w, cy),                # index MCP (左) → 掌宽左端
        (6, cx - half_w, cy - 20),            # index PIP
        (8, cx, cy - 60),                    # index tip
        (9, cx, cy),                          # middle MCP
        (10, cx, cy - 15),                   # middle PIP
        (12, cx, cy - 20),                   # middle tip
        (14, cx + 15, cy - 15),              # ring PIP
        (16, cx + 15, cy - 20),              # ring tip
        (17, cx + half_w, cy),               # pinky MCP (右) → 掌宽右端
        (18, cx + half_w, cy - 15),          # pinky PIP
        (20, cx + half_w, cy - 20),          # pinky tip
        (4, cx + target_dist, cy - 60),      # thumb tip（与食指尖同 y，距离 = target_dist）
        (2, cx - 30, cy + 5),                # thumb MCP
        (3, cx - 32, cy - 5),                # thumb IP
    ])


class TestPinchHysteresisDefaults(unittest.TestCase):
    """验证默认行为：关闭滞回时与旧版单阈值一致。"""

    def test_hysteresis_disabled_by_default(self):
        """Recognizer 默认关闭开关；ConfigManager 负责将仅退出方案默认设为开启。"""
        gr = GestureRecognizer()
        self.assertFalse(gr.pinch_hysteresis_enabled)
        self.assertFalse(gr.pinch_exit_hysteresis_enabled)

    def test_single_threshold_when_disabled(self):
        """关闭滞回时，PINCH_RATIO=0.35 是唯一阈值。"""
        gr = GestureRecognizer()
        gr.pinch_hysteresis_enabled = False

        # 比值 0.34（< 0.35）→ pinch
        hand = make_hand_with_pinch_ratio(0.34)
        features = gr.get_hand_features(hand)
        self.assertTrue(features["thumb_index_pinch"])

        # 比值 0.36（> 0.35）→ no pinch
        hand = make_hand_with_pinch_ratio(0.36)
        features = gr.get_hand_features(hand)
        self.assertFalse(features["thumb_index_pinch"])


class TestPinchHysteresisEnterExit(unittest.TestCase):
    """验证双阈值滞回的 ENTER/EXIT 行为。"""

    def test_enter_threshold_stricter_than_single(self):
        """开启滞回后，未捏合时用 ENTER=0.30，比单阈值 0.35 更严格。

        比值在 0.30-0.35 之间时：单阈值会 pinch，滞回不会（还没进入）。
        """
        gr = GestureRecognizer()
        gr.pinch_hysteresis_enabled = True
        # 确保从"未捏合"状态开始
        gr._was_thumb_index_pinch = False

        # 比值 0.32：在 0.30-0.35 滞回带内
        # 单阈值（0.35）会判定为 pinch，但滞回 ENTER（0.30）不会
        hand = make_hand_with_pinch_ratio(0.32)
        features = gr.get_hand_features(hand)
        self.assertFalse(features["thumb_index_pinch"],
                         "比值 0.32 在滞回带内，未捏合时 ENTER=0.30 不应触发")

    def test_exit_threshold_lenient_than_single(self):
        """开启滞回后，已捏合时用 EXIT=0.40，比单阈值 0.35 更宽松。

        比值在 0.35-0.40 之间时：单阈值不会 pinch，滞回会（仍保持）。
        """
        gr = GestureRecognizer()
        gr.pinch_hysteresis_enabled = True
        # 模拟"已捏合"状态
        gr._was_thumb_index_pinch = True

        # 比值 0.37：在 0.35-0.40 滞回带内
        # 单阈值（0.35）会判定为 no pinch，但滞回 EXIT（0.40）仍保持 pinch
        hand = make_hand_with_pinch_ratio(0.37)
        features = gr.get_hand_features(hand)
        self.assertTrue(features["thumb_index_pinch"],
                        "比值 0.37 在滞回带内，已捏合时 EXIT=0.40 应保持 pinch")

    def test_enter_below_enter_ratio(self):
        """比值 < ENTER=0.30 时，从未捏合进入捏合。"""
        gr = GestureRecognizer()
        gr.pinch_hysteresis_enabled = True
        gr._was_thumb_index_pinch = False

        hand = make_hand_with_pinch_ratio(0.20)  # 真实捏合距离
        features = gr.get_hand_features(hand)
        self.assertTrue(features["thumb_index_pinch"])

    def test_exit_above_exit_ratio(self):
        """比值 > EXIT=0.40 时，从已捏合退出到未捏合。"""
        gr = GestureRecognizer()
        gr.pinch_hysteresis_enabled = True
        gr._was_thumb_index_pinch = True

        hand = make_hand_with_pinch_ratio(0.45)  # 远超 EXIT
        features = gr.get_hand_features(hand)
        self.assertFalse(features["thumb_index_pinch"])


class TestPinchExitOnlyHysteresis(unittest.TestCase):
    """仅退出方向滞回必须不改变旧版的进入灵敏度。"""

    def test_keeps_legacy_enter_threshold(self):
        gr = GestureRecognizer()
        gr.pinch_exit_hysteresis_enabled = True

        features = gr.get_hand_features(make_hand_with_pinch_ratio(0.34))

        self.assertTrue(features["thumb_index_pinch"])

    def test_keeps_pinch_until_exit_threshold(self):
        gr = GestureRecognizer()
        gr.pinch_exit_hysteresis_enabled = True
        gr._was_thumb_index_pinch = True

        features = gr.get_hand_features(make_hand_with_pinch_ratio(0.37))

        self.assertTrue(features["thumb_index_pinch"])

    def test_full_hysteresis_takes_precedence_when_both_enabled(self):
        gr = GestureRecognizer()
        gr.pinch_hysteresis_enabled = True
        gr.pinch_exit_hysteresis_enabled = True

        features = gr.get_hand_features(make_hand_with_pinch_ratio(0.32))

        self.assertFalse(features["thumb_index_pinch"])


class TestPinchHysteresisStateTransition(unittest.TestCase):
    """验证完整的滞回状态转换序列。"""

    def test_full_hysteresis_cycle(self):
        """完整序列：未捏合 → 进入 → 滞回带保持 → 退出 → 未捏合。"""
        gr = GestureRecognizer()
        gr.pinch_hysteresis_enabled = True
        gr._was_thumb_index_pinch = False

        # 1. 比值 0.50：未捏合（远高于 EXIT）
        hand = make_hand_with_pinch_ratio(0.50)
        features = gr.get_hand_features(hand)
        self.assertFalse(features["thumb_index_pinch"])

        # 2. 比值 0.32：在滞回带内（0.30-0.40），仍未捏合（ENTER=0.30 未达）
        hand = make_hand_with_pinch_ratio(0.32)
        features = gr.get_hand_features(hand)
        self.assertFalse(features["thumb_index_pinch"])

        # 3. 比值 0.20：低于 ENTER=0.30，进入捏合
        hand = make_hand_with_pinch_ratio(0.20)
        features = gr.get_hand_features(hand)
        self.assertTrue(features["thumb_index_pinch"])

        # 4. 比值 0.32：回到滞回带内，但已捏合 → 保持 pinch（EXIT=0.40）
        hand = make_hand_with_pinch_ratio(0.32)
        features = gr.get_hand_features(hand)
        self.assertTrue(features["thumb_index_pinch"],
                        "滞回带内已捏合应保持 pinch 状态")

        # 5. 比值 0.37：仍在滞回带内 → 保持 pinch
        hand = make_hand_with_pinch_ratio(0.37)
        features = gr.get_hand_features(hand)
        self.assertTrue(features["thumb_index_pinch"])

        # 6. 比值 0.45：超过 EXIT=0.40 → 退出 pinch
        hand = make_hand_with_pinch_ratio(0.45)
        features = gr.get_hand_features(hand)
        self.assertFalse(features["thumb_index_pinch"])

        # 7. 比值 0.37：退回滞回带，但已退出 → 不重新进入（ENTER=0.30）
        hand = make_hand_with_pinch_ratio(0.37)
        features = gr.get_hand_features(hand)
        self.assertFalse(features["thumb_index_pinch"],
                         "退出后滞回带内不应重新进入（需低于 ENTER=0.30）")

    def test_reset_state_clears_pinch_hysteresis(self):
        """_reset_state() 清除 pinch 滞回状态。"""
        gr = GestureRecognizer()
        gr.pinch_hysteresis_enabled = True
        gr._was_thumb_index_pinch = True
        gr._was_thumb_middle_pinch = True

        gr._reset_state()

        self.assertFalse(gr._was_thumb_index_pinch)
        self.assertFalse(gr._was_thumb_middle_pinch)


if __name__ == '__main__':
    unittest.main()
