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

import modes.draw_mode as draw_mode_mod
from modes.draw_mode import DrawMode
from services.gesture_recognizer import GestureRecognizer


class FakeConfig:
    def __init__(self, data=None):
        self.data = dict(data or {})

    def get(self, key, default=None):
        return self.data.get(key, default)


def make_hand(thumb_apart=False, two_finger=False, half_curl=False, yaw=1.0):
    """合成 21 点手部关键点（[id, x, y] 像素坐标，y 向下）。

    基准为正面书写姿势：食指竖直伸出，中指/无名指/小指弯曲，拇指并拢。
    two_finger: 中指与食指等长伸出且贴紧（mi≈0.99）。
    half_curl: 中指半弯，2D 投影偏长（mi≈0.82）——实测单指书写时 mi 最大
    到 0.81，正是旧"中指长>掌宽×0.6"判定大量误判断笔的形态。
    yaw<1 模拟偏航：所有 x 坐标绕 x=300 按 cos 因子压缩（掌宽塌缩，
    竖直方向的食指长度不变）。
    """
    if two_finger:
        middle_tip = (264, 198)   # mi≈0.99，与食指尖间距 14px < 张开要求
    elif half_curl:
        middle_tip = (205, 275)   # mi≈0.82，且指尖低于 PIP（不破坏书写姿势）
    else:
        middle_tip = (284, 300)   # 正常弯曲
    pts = {
        0: (300, 400),                      # 腕
        2: (240, 360), 3: (235, 340),       # 拇指 MCP/IP
        4: (180, 330) if thumb_apart else (250, 320),  # 拇指尖
        5: (260, 300), 6: (255, 260), 8: (250, 200),   # 食指 MCP/PIP/TIP（伸直）
        9: (285, 295), 10: (282, 265),                 # 中指 MCP/PIP
        12: middle_tip,
        14: (310, 270), 16: (312, 305),     # 无名指 PIP/TIP（弯曲）
        17: (340, 310), 18: (335, 280), 20: (336, 315),  # 小指 MCP/PIP/TIP（弯曲）
    }
    lm = []
    for i in range(21):
        x, y = pts.get(i, (300, 350))
        x = 300 + (x - 300) * yaw
        lm.append([i, x, y])
    return lm


def make_open_palm():
    """合成张掌（食指+中指+无名指+小指都伸出）——is_open_palm 为真。
    也用于模拟书写中单帧关键点抖动导致 middle_up & ring_up 偶发同真的噪声帧。"""
    lm = make_hand()
    lm[12] = [12, 284.0, 200.0]   # 中指尖抬过 PIP(10.y=265) → middle_up
    lm[16] = [16, 312.0, 200.0]   # 无名指尖抬过 PIP(14.y=270) → ring_up
    lm[20] = [20, 336.0, 200.0]   # 小指尖抬过 PIP(18.y=280) → pinky_up
    return lm


def make_draw_mode(extra_config=None):
    overlay = mock.MagicMock()
    overlay.REFERENCE_HAND_SIZE = 100.0
    overlay.isVisible.return_value = True
    mouse = mock.MagicMock()
    mouse.to_screen.return_value = (500, 500)
    cfg = {"draw_record_trace": False}
    if extra_config:
        cfg.update(extra_config)
    dm = DrawMode(
        FakeConfig(cfg),
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


DT = 0.05  # 模拟 ~20fps 帧间隔，让时间窗投票按真实节奏累积证据


class DrawStateMachineTest(unittest.TestCase):
    """中央投票笔状态机：起落都需时间窗（默认 0.3s）内多数帧的持续证据，
    单帧布尔值只投票、不直接决定。故每个用例先 warm_writing() 累积证据确认
    落笔，再验证维持/抬笔。用受控时钟驱动，使时间窗按真实帧间隔推进。"""

    def setUp(self):
        self._now = [1000.0]
        patcher = mock.patch.object(draw_mode_mod.time, "time", lambda: self._now[0])
        patcher.start()
        self.addCleanup(patcher.stop)
        self.dm, self.overlay = make_draw_mode()

    def step(self, landmarks, label="OTHER", dm=None):
        self._now[0] += DT
        return (dm or self.dm).handle(
            [landmarks], [{"label": label, "bbox_area": 0.0}], 640, 480
        )

    def warm_writing(self, n=6):
        """步进 n 帧正面书写姿势，累积证据确认落笔。"""
        res = None
        for _ in range(n):
            res = self.step(make_hand())
        self.assertEqual(res.gesture, "DRAW")
        return res

    def test_frontal_thumb_tucked_starts_writing(self):
        self.warm_writing()
        self.overlay.draw_to.assert_called()

    def test_side_yaw_thumb_apart_does_not_break_stroke(self):
        """书写中侧偏、拇指被脑补成"分开"：该帧投 write（拇指不可读时冻结），
        笔画不断。"""
        self.warm_writing()
        self.overlay.force_lift_pen.reset_mock()
        for _ in range(15):
            self.assertEqual(
                self.step(make_hand(thumb_apart=True, yaw=0.45)).gesture, "DRAW"
            )
        self.overlay.force_lift_pen.assert_not_called()

    def test_frontal_thumb_apart_does_not_lift_when_disabled(self):
        """draw_thumb_lift=False：拇指分开不再抬笔——消除近距正面书写时
        拇指间歇被读成"分开"的笔画中途误断。单指姿势维持落笔，抬笔交给
        ✌️ / 握拳 / 张掌。（v1.3.6 稳定档默认就是 False，此测试锁定关闭行为）"""
        dm, overlay = make_draw_mode({"draw_thumb_lift": False})
        for _ in range(6):
            self.step(make_hand(), dm=dm)  # warm writing
        overlay.force_lift_pen.reset_mock()
        for _ in range(15):
            self.assertEqual(self.step(make_hand(thumb_apart=True), dm=dm).gesture, "DRAW")
        overlay.force_lift_pen.assert_not_called()

    def test_frontal_thumb_apart_lifts_when_enabled(self):
        """draw_thumb_lift=True 恢复旧习惯：正面分开拇指 → 投 hover，多数后抬笔。"""
        dm, overlay = make_draw_mode({"draw_thumb_lift": True})
        for _ in range(6):
            self.step(make_hand(), dm=dm)
        results = [self.step(make_hand(thumb_apart=True), dm=dm) for _ in range(15)]
        self.assertEqual(results[-1].gesture, "DRAW_HOVER")
        overlay.force_lift_pen.assert_called()

    def test_geom_two_finger_does_not_lift_by_default(self):
        """默认 draw_two_finger_geom=False：几何双指（label 非 VICTORY）不再抬笔——
        消除侧视下中指 2D 投影 mi>0.95 把单指误判成双指的断笔。抬笔改走
        VICTORY 标签 / 握拳 / 张掌。"""
        self.warm_writing()
        self.overlay.force_lift_pen.reset_mock()
        for _ in range(15):
            self.assertEqual(
                self.step(make_hand(two_finger=True, yaw=0.45)).gesture, "DRAW"
            )
        self.overlay.force_lift_pen.assert_not_called()

    def test_two_finger_geom_lifts_when_enabled(self):
        """draw_two_finger_geom=True 恢复几何兜底：侧偏双指 → 投 hover，多数后抬笔。"""
        dm, overlay = make_draw_mode({"draw_two_finger_geom": True})
        for _ in range(6):
            self.step(make_hand(), dm=dm)
        results = [
            self.step(make_hand(two_finger=True, yaw=0.45), dm=dm) for _ in range(15)
        ]
        self.assertEqual(results[-1].gesture, "DRAW_HOVER")
        overlay.force_lift_pen.assert_called()

    def test_victory_label_lifts_without_geometry(self):
        """几何仍是单指书写姿势但 ML 标签报 VICTORY：标签把该帧投成 hover，
        多数后抬笔。"""
        self.warm_writing()
        self.overlay.force_lift_pen.reset_mock()
        results = [self.step(make_hand(), label="VICTORY") for _ in range(15)]
        self.assertEqual(results[-1].gesture, "DRAW_HOVER")
        self.overlay.force_lift_pen.assert_called()

    def test_half_curled_middle_does_not_lift(self):
        """实测回归：书写中中指半弯（mi≈0.82，旧判定误判为双指导致 16/24 次
        断笔）——新判定该帧投 write，笔画必须不断。"""
        self.warm_writing()
        self.overlay.force_lift_pen.reset_mock()
        for _ in range(15):
            self.assertEqual(self.step(make_hand(half_curl=True)).gesture, "DRAW")
        self.overlay.force_lift_pen.assert_not_called()

    def test_pointing_up_label_vetoes_two_finger_geometry(self):
        """即使开启几何兜底（draw_two_finger_geom=True）且几何满足双指，分类器报
        POINTING_UP 也投 write——ML 标签否决几何，笔画不断。"""
        dm, overlay = make_draw_mode({"draw_two_finger_geom": True})
        for _ in range(6):
            self.step(make_hand(), dm=dm)
        overlay.force_lift_pen.reset_mock()
        for _ in range(10):
            self.assertEqual(
                self.step(make_hand(two_finger=True), label="POINTING_UP", dm=dm).gesture,
                "DRAW",
            )
        overlay.force_lift_pen.assert_not_called()

    def test_pointing_up_label_starts_writing_sideways(self):
        """侧偏下拇指不可读时，POINTING_UP 单指姿势累积证据后落笔。"""
        results = [
            self.step(make_hand(thumb_apart=True, yaw=0.45), label="POINTING_UP")
            for _ in range(6)
        ]
        self.assertEqual(results[-1].gesture, "DRAW")

    def test_single_frame_insufficient_evidence_stays_hover(self):
        """证据不足（单帧 < VOTE_MIN）时不落笔：投票窗未达门槛，宁可悬停。
        注意：与旧机器不同，持续的侧偏单指姿势在累积足够证据后会落笔
        （见 test_pointing_up_label_starts_writing_sideways）。"""
        result = self.step(make_hand(thumb_apart=True, yaw=0.45))
        self.assertEqual(result.gesture, "DRAW_HOVER")

    def test_single_open_palm_frame_does_not_break_stroke(self):
        """书写中单帧 is_open_palm 噪声（middle_up & ring_up 偶发同真）不立即抬笔：
        旧实现单帧即 force_lift_pen 绕过去抖，是断笔主因。默认需连续 3 帧确认。"""
        self.warm_writing()
        self.overlay.force_lift_pen.reset_mock()
        palm = make_open_palm()
        # 前 2 帧（< draw_open_palm_lift_frames=3）按噪声处理，保持书写、不抬笔
        self.assertEqual(self.step(palm).gesture, "DRAW")
        self.assertEqual(self.step(palm).gesture, "DRAW")
        self.overlay.force_lift_pen.assert_not_called()
        # 第 3 帧确认抬笔（流向清屏分支）
        self.step(palm)
        self.overlay.force_lift_pen.assert_called()

    def test_open_palm_lift_frames_one_restores_single_frame_lift(self):
        """draw_open_palm_lift_frames=1 恢复旧的单帧立即抬笔行为（A/B 可回退）。"""
        dm, overlay = make_draw_mode({"draw_open_palm_lift_frames": 1})
        for _ in range(6):
            self.step(make_hand(), dm=dm)
        overlay.force_lift_pen.reset_mock()
        self.step(make_open_palm(), dm=dm)
        overlay.force_lift_pen.assert_called()


if __name__ == "__main__":
    unittest.main()
