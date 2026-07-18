"""实施方案 Phase 3.3: thumb_extended 旋转不变判定单元测试

验证：
1. _thumb_perp_ratio 计算正确（拇指张开时 perp 大，内收时小）
2. 旋转不变性：同一手势旋转后 perp_ratio 保持稳定
3. 新旧特征并存：thumb_perp_ratio 和 thumb_extended_new 始终输出
4. config 开关：thumb_perp_ratio_enabled=False 时用旧特征，True 时用新特征
5. 默认关闭

注意：单元测试只测特征计算逻辑，不测实际手势场景效果（参考项目 memory 教训）。
"""
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'app'))

from services.gesture_recognizer import GestureRecognizer


def make_landmark(idx, x, y):
    return [idx, float(x), float(y)]


def make_hand(landmark_xy_pairs):
    """从 [(idx, x, y), ...] 列表创建完整 21 点 landmark 数组。"""
    coords = {idx: (x, y) for idx, x, y in landmark_xy_pairs}
    return [make_landmark(i, coords.get(i, (0.0, 0.0))[0], coords.get(i, (0.0, 0.0))[1]) for i in range(21)]


def make_thumb_extended_hand(cx=200.0, cy=200.0, hw=90.0):
    """拇指张开的手（正面对相机，掌心朝镜头）。

    手竖直：wrist 在下方，手指向上，拇指水平向左伸出。
    """
    half_w = hw / 2.0
    return make_hand([
        (0, cx, cy + 50),                # wrist
        (5, cx - half_w, cy),            # index MCP
        (6, cx - half_w, cy - 20),        # index PIP
        (8, cx - half_w + 5, cy - 60),    # index tip
        (9, cx, cy),                      # middle MCP (掌宽右端参考)
        (10, cx, cy - 20),
        (12, cx, cy - 60),
        (17, cx + half_w, cy),           # pinky MCP
        (18, cx + half_w, cy - 20),
        (20, cx + half_w, cy - 60),
        (4, cx - hw * 0.6, cy - 10),     # thumb tip（水平向左伸出，远离掌心中轴）
        (2, cx - half_w - 5, cy + 5),    # thumb MCP
        (3, cx - half_w - 10, cy - 5),   # thumb IP
    ])


def make_thumb_tucked_hand(cx=200.0, cy=200.0, hw=90.0):
    """拇指内收的手（拇指横跨掌心，tip 在中轴附近）。"""
    half_w = hw / 2.0
    return make_hand([
        (0, cx, cy + 50),                # wrist
        (5, cx - half_w, cy),            # index MCP
        (6, cx - half_w, cy - 20),
        (8, cx - half_w + 5, cy - 60),
        (9, cx, cy),                      # middle MCP
        (10, cx, cy - 20),
        (12, cx, cy - 60),
        (17, cx + half_w, cy),           # pinky MCP
        (18, cx + half_w, cy - 20),
        (20, cx + half_w, cy - 60),
        (4, cx + 5, cy - 10),            # thumb tip（横跨掌心，靠近中轴）
        (2, cx - 10, cy + 5),
        (3, cx - 5, cy - 5),
    ])


def rotate_hand(landmarks, angle_deg, cx=200.0, cy=200.0):
    """绕 (cx, cy) 旋转所有 landmark 坐标（2D 旋转）。"""
    angle = math.radians(angle_deg)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    rotated = []
    for lm in landmarks:
        idx = lm[0]
        x, y = lm[1] - cx, lm[2] - cy
        rx = x * cos_a - y * sin_a + cx
        ry = x * sin_a + y * cos_a + cy
        rotated.append([idx, rx, ry])
    return rotated


class TestThumbPerpRatioComputation(unittest.TestCase):
    """验证 _thumb_perp_ratio 计算的正确性。"""

    def test_extended_thumb_has_high_perp(self):
        """拇指张开时 perp_ratio 应高于阈值 0.5。"""
        gr = GestureRecognizer()
        hand = make_thumb_extended_hand()
        ratio = gr._thumb_perp_ratio(hand)
        self.assertGreater(ratio, 0.5,
                           f"拇指张开时 perp_ratio={ratio:.3f} 应 > 0.5")

    def test_tucked_thumb_has_low_perp(self):
        """拇指内收时 perp_ratio 应低于阈值 0.5。"""
        gr = GestureRecognizer()
        hand = make_thumb_tucked_hand()
        ratio = gr._thumb_perp_ratio(hand)
        self.assertLess(ratio, 0.5,
                        f"拇指内收时 perp_ratio={ratio:.3f} 应 < 0.5")

    def test_degenerate_axis_returns_zero(self):
        """掌心中轴退化（wrist == middle MCP）时返回 0。"""
        gr = GestureRecognizer()
        hand = make_hand([
            (0, 200.0, 200.0),
            (9, 200.0, 200.0),  # 同位置 → axis_len = 0
        ])
        ratio = gr._thumb_perp_ratio(hand)
        self.assertEqual(ratio, 0.0)


class TestRotationInvariance(unittest.TestCase):
    """验证 perp_ratio 的旋转不变性（Phase 3.3 核心目标）。"""

    def test_perp_ratio_stable_under_rotation(self):
        """同一拇指张开手势旋转 0/45/90 度后 perp_ratio 应保持稳定（±20%）。"""
        gr = GestureRecognizer()
        base_hand = make_thumb_extended_hand()
        base_ratio = gr._thumb_perp_ratio(base_hand)

        for angle in [45, 90, 135]:
            rotated = rotate_hand(base_hand, angle)
            rotated_ratio = gr._thumb_perp_ratio(rotated)
            # 旋转不变：比值变化应在 20% 以内（landmark 离散化会引入小误差）
            rel_diff = abs(rotated_ratio - base_ratio) / max(base_ratio, 0.01)
            self.assertLess(
                rel_diff, 0.20,
                f"旋转 {angle}° 后 perp_ratio={rotated_ratio:.3f} "
                f"vs 原始 {base_ratio:.3f}，相对差异 {rel_diff:.1%} 超过 20%"
            )


class TestFeatureOutput(unittest.TestCase):
    """验证新旧特征并存输出。"""

    def test_both_features_in_output(self):
        """thumb_perp_ratio 和 thumb_extended_new 始终在 features dict 中。"""
        gr = GestureRecognizer()
        gr.thumb_perp_ratio_enabled = False  # 默认关闭

        hand = make_thumb_extended_hand()
        features = gr.get_hand_features(hand)

        self.assertIn("thumb_perp_ratio", features)
        self.assertIn("thumb_extended_new", features)
        self.assertIn("thumb_extended", features)
        # perp_ratio 应为正数
        self.assertGreater(features["thumb_perp_ratio"], 0.0)

    def test_extended_hand_perp_above_threshold(self):
        """拇指张开手 → thumb_extended_new=True。"""
        gr = GestureRecognizer()
        hand = make_thumb_extended_hand()
        features = gr.get_hand_features(hand)
        self.assertTrue(features["thumb_extended_new"],
                        f"perp_ratio={features['thumb_perp_ratio']:.3f} 应 > 0.5")

    def test_tucked_hand_perp_below_threshold(self):
        """拇指内收手 → thumb_extended_new=False。"""
        gr = GestureRecognizer()
        hand = make_thumb_tucked_hand()
        features = gr.get_hand_features(hand)
        self.assertFalse(features["thumb_extended_new"],
                         f"perp_ratio={features['thumb_perp_ratio']:.3f} 应 < 0.5")


class TestConfigSwitch(unittest.TestCase):
    """验证 config 开关行为。"""

    def test_default_uses_old_feature(self):
        """thumb_perp_ratio_enabled=False → thumb_extended 跟随旧特征（thumb_tip_to_index_mcp 距离）。"""
        gr = GestureRecognizer()
        gr.thumb_perp_ratio_enabled = False

        hand = make_thumb_extended_hand()
        features = gr.get_hand_features(hand)

        # 直接计算旧逻辑值：thumb_tip_to_index_mcp > hand_width * 0.9
        thumb_tip_to_index_mcp = math.hypot(
            hand[4][1] - hand[5][1], hand[4][2] - hand[5][2]
        )
        hand_width = math.hypot(
            hand[5][1] - hand[17][1], hand[5][2] - hand[17][2]
        )
        old_expected = thumb_tip_to_index_mcp > hand_width * gr.THUMB_EXTEND_RATIO
        self.assertEqual(features["thumb_extended"], old_expected,
                         "关闭时 thumb_extended 应跟随旧特征（thumb_tip_to_index_mcp 距离）")

    def test_enabled_uses_new_feature(self):
        """thumb_perp_ratio_enabled=True → thumb_extended 跟随新特征。"""
        gr = GestureRecognizer()
        gr.thumb_perp_ratio_enabled = True

        hand = make_thumb_extended_hand()
        features = gr.get_hand_features(hand)
        # 新特征为 True，thumb_extended 应跟随
        self.assertEqual(features["thumb_extended"], features["thumb_extended_new"])

    def test_disabled_by_default(self):
        """GestureRecognizer 默认 thumb_perp_ratio_enabled=False。"""
        gr = GestureRecognizer()
        self.assertFalse(gr.thumb_perp_ratio_enabled)


if __name__ == '__main__':
    unittest.main()
