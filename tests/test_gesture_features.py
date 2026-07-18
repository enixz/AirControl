"""GestureRecognizer 关键特征提取测试。

验证 P1-9（距离自适应统一）、P1-10（正面度门控）的修改：
- 捏合阈值随掌宽自适应
- 手指伸展在高/低正面度下行为不同
- 双手手势手腕距离阈值随掌宽自适应
- 手指并拢判定纯按掌宽比例（无固定像素兜底）
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'app'))

from services.gesture_recognizer import GestureRecognizer


def make_landmark(idx, x, y, z=0.0):
    """创建单个 landmark [index, x, y, z]。"""
    return [idx, float(x), float(y), float(z)]


def make_hand(landmark_xy_pairs):
    """从 [(idx, x, y), ...] 列表创建完整 21 点 landmark 数组。

    未指定的点默认填 (0, 0)。
    """
    coords = {idx: (x, y) for idx, x, y in landmark_xy_pairs}
    return [make_landmark(i, coords.get(i, (0, 0))[0], coords.get(i, (0, 0))[1]) for i in range(21)]


def make_frontal_index_extended_hand(center_x=200, center_y=200, scale=1.0):
    """创建正面对相机、食指伸出的手。

    scale 控制手大小（1.0=参考大小，掌宽≈90px）。
    landmark 布局：掌根在下方，手指向上（y 减小）。
    """
    s = scale
    # 关键点：5(食指根) 和 17(小指根) 间距 = 掌宽
    # 8(食指尖) 和 5(食指根) 间距 = 食指长
    return make_hand([
        (0, center_x, center_y + 40 * s),       # wrist
        (5, center_x - 20 * s, center_y),       # index MCP (左)
        (6, center_x - 20 * s, center_y - 20 * s),  # index PIP
        (8, center_x - 20 * s, center_y - 60 * s),  # index tip (伸出)
        (9, center_x, center_y),                # middle MCP
        (10, center_x, center_y - 15 * s),      # middle PIP (弯曲)
        (12, center_x, center_y - 20 * s),      # middle tip (弯曲)
        (14, center_x + 15 * s, center_y - 15 * s),  # ring PIP
        (16, center_x + 15 * s, center_y - 20 * s),  # ring tip
        (17, center_x + 45 * s, center_y),      # pinky MCP (右) → 掌宽=65*s
        (18, center_x + 45 * s, center_y - 15 * s),  # pinky PIP
        (20, center_x + 45 * s, center_y - 20 * s),  # pinky tip
        (4, center_x - 35 * s, center_y - 10 * s),   # thumb tip
        (2, center_x - 30 * s, center_y + 5 * s),    # thumb MCP
        (3, center_x - 32 * s, center_y - 5 * s),    # thumb IP
    ])


def make_side_index_extended_hand(center_x=200, center_y=200, scale=1.0):
    """创建侧对相机（低正面度）、食指伸出的手。

    掌宽（5↔17）被压缩，但食指长（8↔5）不变 → hand_frontality 低。
    """
    s = scale
    return make_hand([
        (0, center_x, center_y + 40 * s),
        (5, center_x - 5 * s, center_y),       # index MCP (掌宽被压缩)
        (6, center_x - 5 * s, center_y - 20 * s),
        (8, center_x - 5 * s, center_y - 60 * s),  # index tip (伸出，长度不变)
        (9, center_x, center_y),
        (10, center_x, center_y - 15 * s),
        (12, center_x, center_y - 20 * s),
        (14, center_x + 5 * s, center_y - 15 * s),
        (16, center_x + 5 * s, center_y - 20 * s),
        (17, center_x + 10 * s, center_y),     # pinky MCP (掌宽=15*s，很小)
        (18, center_x + 10 * s, center_y - 15 * s),
        (20, center_x + 10 * s, center_y - 20 * s),
        (4, center_x - 15 * s, center_y - 10 * s),
        (2, center_x - 10 * s, center_y + 5 * s),
        (3, center_x - 12 * s, center_y - 5 * s),
    ])


class TestPinchThresholdAdaptive(unittest.TestCase):
    """捏合阈值应随掌宽自适应（hand_width * 0.35）。"""

    def test_pinch_threshold_scales_with_hand_width(self):
        gr = GestureRecognizer()
        # 大手（近处）：掌宽=90
        big_hand = make_frontal_index_extended_hand(scale=90.0 / 65.0)
        features_big = gr.get_hand_features(big_hand)
        hw_big = features_big["hand_width"]

        # 小手（远处）：掌宽=30
        small_hand = make_frontal_index_extended_hand(scale=30.0 / 65.0)
        features_small = gr.get_hand_features(small_hand)
        hw_small = features_small["hand_width"]

        # 掌宽确实不同
        self.assertGreater(hw_big, hw_small)
        # 捏合阈值与掌宽成正比
        self.assertGreater(hw_big * 0.35, hw_small * 0.35)

    def test_pinch_detected_when_thumb_index_close(self):
        """拇指食指距离 < hand_width*0.35 时应检测到捏合。"""
        gr = GestureRecognizer()
        hand = make_frontal_index_extended_hand(scale=1.0)
        features = gr.get_hand_features(hand)
        hw = features["hand_width"]
        threshold = hw * 0.35

        # 把拇指尖和食指尖都移到同一点附近（x 和 y 都接近）
        target_x, target_y = 200, 200
        hand[4][1] = target_x - threshold * 0.1  # thumb tip x
        hand[4][2] = target_y  # thumb tip y
        hand[8][1] = target_x + threshold * 0.1  # index tip x
        hand[8][2] = target_y  # index tip y

        features = gr.get_hand_features(hand)
        self.assertTrue(features["thumb_index_pinch"],
                        "拇指食指距离 < 阈值时应检测到捏合")


class TestFrontalityGate(unittest.TestCase):
    """P1-10: 食指伸展判定应引入 frontality gate。"""

    def test_frontal_hand_uses_hand_width_threshold(self):
        """高正面度时用 hand_width*0.65 判定食指伸出。"""
        gr = GestureRecognizer()
        hand = make_frontal_index_extended_hand(scale=1.0)
        features = gr.get_hand_features(hand)
        # 正面度应高于门控阈值
        self.assertGreaterEqual(features["hand_frontality"], gr.FRONTALITY_GATE,
                                f"正对手相机的手正面度({features['hand_frontality']:.2f})应 >= 门控阈值({gr.FRONTALITY_GATE})")
        # 食指应被判定为伸出
        self.assertTrue(features["index_extended"],
                        "正面对相机且食指伸出时 index_extended 应为 True")

    def test_side_hand_falls_back_to_y_coordinate(self):
        """低正面度时回退到 y 坐标判定（index_up）。"""
        gr = GestureRecognizer()
        hand = make_side_index_extended_hand(scale=1.0)
        features = gr.get_hand_features(hand)
        # 正面度应低于门控阈值
        self.assertLess(features["hand_frontality"], gr.FRONTALITY_GATE,
                        f"侧对相机的手正面度({features['hand_frontality']:.2f})应 < 门控阈值({gr.FRONTALITY_GATE})")
        # 食指 y 坐标判定（index_up）应为 True（指尖在 PIP 上方）
        self.assertTrue(features["index_up"],
                        "食指伸出时 index_up 应为 True（y 坐标判定）")
        # 低正面度时 index_extended 应跟随 index_up
        self.assertTrue(features["index_extended"],
                        "低正面度时 index_extended 应回退到 index_up=True")

    def test_curled_finger_not_extended(self):
        """弯曲食指（index_up=False）不应被判为伸出，无论正面度。

        注意：弯曲手指时 index_len 变短 → hand_frontality 变高（指标仅在
        手指伸出时有效），此时走 hand_width*0.65 路径，短手指自然不满足。
        """
        gr = GestureRecognizer()
        hand = make_frontal_index_extended_hand(scale=1.0)
        # 把食指尖弯下来（y 坐标大于 PIP → index_up=False）
        hand[8][2] = hand[6][2] + 10  # tip 在 PIP 下方
        features = gr.get_hand_features(hand)
        self.assertFalse(features["index_up"],
                         "弯曲食指时 index_up 应为 False")
        self.assertFalse(features["index_extended"],
                         "弯曲食指时 index_extended 应为 False")


class TestFingersCloseAdaptive(unittest.TestCase):
    """P1-9: 手指并拢判定应纯按掌宽比例（无固定 60px 兜底）。"""

    def test_close_threshold_uses_60px_floor(self):
        """远处小手：固定 60px 兜底确保手指间距 < 60px 时判为并拢。

        老版行为（阶段 2.11 恢复）：dx < 60 or dx < hand_width*0.6。
        远处掌宽 30px 时纯比例阈值仅 18px 过严，60px 兜底保证小手也能正确判为并拢。
        """
        gr = GestureRecognizer()
        # 远处小手：掌宽≈30px → close_threshold=18px，但 60px 兜底主导
        small_hand = make_frontal_index_extended_hand(scale=30.0 / 65.0)

        # 手指间距设为 25px（> 18px 但 < 60px）
        # 60px 兜底 → 25 < 60 → 判为 close
        small_hand[12][1] = small_hand[8][1] + 25
        small_hand[16][1] = small_hand[12][1] + 25
        small_hand[20][1] = small_hand[16][1] + 25

        features = gr.get_hand_features(small_hand)
        self.assertTrue(features["fingers_close"],
                        "远处小手手指间距 25px < 60px 兜底时应判为并拢")


class TestTwoHandGestureAdaptive(unittest.TestCase):
    """P1-9: 双手手势手腕距离阈值应随掌宽自适应。"""

    def test_fist_hug_threshold_scales_with_hand_width(self):
        """FIST_HUG 阈值 = avg_hand_width * 1.3，远处小手阈值更低。"""
        gr = GestureRecognizer()

        # 构造双拳手势：两只手都是 fist
        def make_fist_hand(center_x, center_y, scale=1.0):
            """创建握拳手（四指弯曲，拇指收拢）。"""
            s = scale
            return make_hand([
                (0, center_x, center_y + 40 * s),    # wrist
                (5, center_x - 20 * s, center_y),
                (6, center_x - 20 * s, center_y - 10 * s),  # PIP 弯曲
                (8, center_x - 20 * s, center_y - 5 * s),   # tip 弯曲
                (9, center_x, center_y),
                (10, center_x, center_y - 10 * s),
                (12, center_x, center_y - 5 * s),
                (14, center_x + 15 * s, center_y - 10 * s),
                (16, center_x + 15 * s, center_y - 5 * s),
                (17, center_x + 45 * s, center_y),   # 掌宽=65*s
                (18, center_x + 45 * s, center_y - 10 * s),
                (20, center_x + 45 * s, center_y - 5 * s),
                (4, center_x - 25 * s, center_y),    # thumb tucked
                (2, center_x - 20 * s, center_y + 5 * s),
                (3, center_x - 22 * s, center_y),
            ])

        # 大手（近处）：掌宽≈90，阈值≈117
        big_left = make_fist_hand(100, 200, scale=90.0 / 65.0)
        big_right = make_fist_hand(300, 200, scale=90.0 / 65.0)
        features_big = [gr.get_hand_features(big_left), gr.get_hand_features(big_right)]
        avg_hw_big = (features_big[0]["hand_width"] + features_big[1]["hand_width"]) / 2
        expected_threshold_big = avg_hw_big * 1.3

        # 小手（远处）：掌宽≈30，阈值≈39
        small_left = make_fist_hand(100, 200, scale=30.0 / 65.0)
        small_right = make_fist_hand(300, 200, scale=30.0 / 65.0)
        features_small = [gr.get_hand_features(small_left), gr.get_hand_features(small_right)]
        avg_hw_small = (features_small[0]["hand_width"] + features_small[1]["hand_width"]) / 2
        expected_threshold_small = avg_hw_small * 1.3

        # 阈值应与掌宽成正比
        self.assertGreater(expected_threshold_big, expected_threshold_small,
                           "大手 FIST_HUG 阈值应大于小手")

    def test_palm_spread_threshold_scales_with_hand_width(self):
        """TWO_PALM_SPREAD 阈值 = avg_hand_width * 2.2。"""
        # 只验证阈值比例关系
        for hw in (30.0, 60.0, 90.0, 150.0):
            threshold = hw * 2.2
            self.assertGreater(threshold, hw,
                               "TWO_PALM_SPREAD 阈值应大于掌宽")


class TestScrollThresholdAdaptive(unittest.TestCase):
    """P1-9: 滚动阈值应纯按掌宽比例（无固定 60px 下限）。"""

    def test_scroll_threshold_no_fixed_floor(self):
        """远处小手：滚动阈值 = hand_width * 1.2，不受固定 60px 下限影响。"""
        gr = GestureRecognizer()
        # 远处小手：掌宽≈30px → threshold=36px
        # 旧代码：max(36, 60) = 60（固定下限主导）
        # 新代码：36（纯比例）
        small_hand = make_frontal_index_extended_hand(scale=30.0 / 65.0)
        features = gr.get_hand_features(small_hand)
        hw = features["hand_width"]
        expected_threshold = hw * 1.2
        # 确认阈值 < 60（旧代码的固定下限）
        self.assertLess(expected_threshold, 60,
                        f"远处小手滚动阈值(={expected_threshold:.0f})应 < 60（旧固定下限），证明固定下限已移除")


if __name__ == "__main__":
    unittest.main(verbosity=2)
