"""板书模式侧偏鲁棒性测试：双指悬停 + 拇指可观测性门控。

背景：手臂以手肘为轴横扫时手会偏航，掌宽按 cos 塌缩、拇指被整只手遮挡，
拇指关键点是模型脑补的。本组测试验证：
- ✌️ 双指悬停判定宽松：两指贴紧（不满足 is_scissor 的张开要求）也算
- 侧偏（正面度低）时拇指"分开"被忽略，书写状态冻结，笔画不断
- 正面时拇指交互照旧：并拢落笔、分开抬笔
- VICTORY / POINTING_UP ML 标签分别兜底抬笔/落笔
"""
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'app'))

from modes.draw_mode import DrawMode
from services.gesture_recognizer import GestureRecognizer


class FakeConfig:
    def __init__(self, data=None):
        self.data = dict(data or {})

    def get(self, key, default=None):
        return self.data.get(key, default)


def make_hand(thumb_apart=False, two_finger=False, yaw=1.0):
    """合成 21 点手部关键点（[id, x, y] 像素坐标，y 向下）。

    基准为正面书写姿势：食指竖直伸出，中指/无名指/小指弯曲，拇指并拢。
    yaw<1 模拟偏航：所有 x 坐标绕 x=300 按 cos 因子压缩（掌宽塌缩，
    竖直方向的食指长度不变）。
    """
    pts = {
        0: (300, 400),                      # 腕
        2: (240, 360), 3: (235, 340),       # 拇指 MCP/IP
        4: (180, 330) if thumb_apart else (250, 320),  # 拇指尖
        5: (260, 300), 6: (255, 260), 8: (250, 200),   # 食指 MCP/PIP/TIP（伸直）
        9: (285, 295), 10: (282, 265),                 # 中指 MCP/PIP
        12: (262, 205) if two_finger else (284, 300),  # 中指尖：伸直(贴紧食指)或弯曲
        14: (310, 270), 16: (312, 305),     # 无名指 PIP/TIP（弯曲）
        17: (340, 310), 18: (335, 280), 20: (336, 315),  # 小指 MCP/PIP/TIP（弯曲）
    }
    lm = []
    for i in range(21):
        x, y = pts.get(i, (300, 350))
        x = 300 + (x - 300) * yaw
        lm.append([i, x, y])
    return lm


def make_draw_mode():
    overlay = mock.MagicMock()
    overlay.REFERENCE_HAND_SIZE = 100.0
    overlay.isVisible.return_value = True
    mouse = mock.MagicMock()
    mouse.to_screen.return_value = (500, 500)
    dm = DrawMode(
        FakeConfig(),
        GestureRecognizer(),
        mouse,
        overlay,
        mock.MagicMock(),  # cursor_overlay
        mock.MagicMock(),  # toolbar
        mock.MagicMock(),  # ppt
    )
    dm.on_enter()
    return dm, overlay


def step(dm, landmarks, label="OTHER"):
    return dm.handle([landmarks], [{"label": label, "bbox_area": 0.0}], 640, 480)


class TwoFingerFeatureTest(unittest.TestCase):
    """特征层：双指判定必须宽松（贴紧也算）。"""

    def test_two_fingers_together_counts_as_hover_but_not_scissor(self):
        r = GestureRecognizer()
        f = r.get_hand_features(make_hand(two_finger=True))
        self.assertTrue(f["two_finger_hover"])
        # 两指贴紧不满足剪刀手的张开要求——证明悬停判定确实更松
        self.assertFalse(f["is_scissor"])

    def test_two_finger_survives_yaw(self):
        r = GestureRecognizer()
        f = r.get_hand_features(make_hand(two_finger=True, yaw=0.45))
        self.assertTrue(f["two_finger_hover"])

    def test_frontality_proxy_drops_under_yaw(self):
        r = GestureRecognizer()
        frontal = r.get_hand_features(make_hand())["hand_frontality"]
        yawed = r.get_hand_features(make_hand(yaw=0.45))["hand_frontality"]
        self.assertGreaterEqual(frontal, 0.55)
        self.assertLess(yawed, 0.55)

    def test_thumb_tucked_threshold_restored(self):
        """d/掌宽=0.58 应判为并拢（恢复 v1.1.0 等效阈值 0.62；被收紧的 0.5 会误判分开）。"""
        r = GestureRecognizer()
        lm = make_hand()
        lm[4] = [4, 250.0, 345.7]  # 距食指根 ≈0.58×掌宽
        f = r.get_hand_features(lm)
        self.assertTrue(f["thumb_tucked"])


class DrawStateMachineTest(unittest.TestCase):
    def test_frontal_thumb_tucked_starts_writing(self):
        dm, overlay = make_draw_mode()
        result = step(dm, make_hand())
        self.assertEqual(result.gesture, "DRAW")
        overlay.draw_to.assert_called()

    def test_side_yaw_thumb_apart_does_not_break_stroke(self):
        """书写中侧偏、拇指被脑补成"分开"：状态冻结，笔画不断。"""
        dm, overlay = make_draw_mode()
        for _ in range(3):
            self.assertEqual(step(dm, make_hand()).gesture, "DRAW")
        overlay.force_lift_pen.reset_mock()
        for _ in range(15):
            result = step(dm, make_hand(thumb_apart=True, yaw=0.45))
            self.assertEqual(result.gesture, "DRAW")
        overlay.force_lift_pen.assert_not_called()

    def test_frontal_thumb_apart_lifts_pen(self):
        """正面故意分开拇指：3 帧内抬笔——习惯交互保留。"""
        dm, overlay = make_draw_mode()
        for _ in range(3):
            step(dm, make_hand())
        results = [step(dm, make_hand(thumb_apart=True)) for _ in range(3)]
        self.assertEqual(results[-1].gesture, "DRAW_HOVER")
        overlay.force_lift_pen.assert_called()

    def test_two_finger_lifts_even_sideways(self):
        """侧偏中伸出中指（与食指贴紧）：2 帧内抬笔——治本路径。"""
        dm, overlay = make_draw_mode()
        for _ in range(3):
            step(dm, make_hand())
        overlay.force_lift_pen.reset_mock()
        results = [step(dm, make_hand(two_finger=True, yaw=0.45)) for _ in range(2)]
        self.assertEqual(results[-1].gesture, "DRAW_HOVER")
        overlay.force_lift_pen.assert_called()

    def test_victory_label_lifts_without_geometry(self):
        """几何仍是单指书写姿势但 ML 标签报 VICTORY：标签兜底抬笔。"""
        dm, overlay = make_draw_mode()
        for _ in range(3):
            step(dm, make_hand())
        results = [step(dm, make_hand(), label="VICTORY") for _ in range(2)]
        self.assertEqual(results[-1].gesture, "DRAW_HOVER")

    def test_pointing_up_label_starts_writing_sideways(self):
        """侧偏下拇指不可读时，POINTING_UP 标签可判定落笔意图。"""
        dm, overlay = make_draw_mode()
        result = step(dm, make_hand(thumb_apart=True, yaw=0.45), label="POINTING_UP")
        self.assertEqual(result.gesture, "DRAW")

    def test_sideways_without_label_stays_hover(self):
        """侧偏且无标签支持时不落笔（拇指不可读、意图不明，宁可悬停）。"""
        dm, overlay = make_draw_mode()
        result = step(dm, make_hand(thumb_apart=True, yaw=0.45))
        self.assertEqual(result.gesture, "DRAW_HOVER")


if __name__ == "__main__":
    unittest.main()
