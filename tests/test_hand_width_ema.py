"""hand_width EMA 归一化表征测试 — 阶段 C。

验证 GestureRecognizer.get_hand_features 中的掌宽慢速 EMA：
  - 首帧用 raw 值初始化（无滞后）
  - 高正面度时更新 EMA（10% 新值 + 90% 历史）
  - 低正面度时不更新 EMA（掌宽塌缩不可信）
  - _reset_state 不清 EMA（保持跨模式连续性）

EMA 消除手势阈值随单帧掌宽抖动（实测 58↔208）跳变。
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'app'))

from services.gesture_recognizer import GestureRecognizer


def _make_hand(raw_hand_width, index_len, cx=200.0, cy=200.0):
    """构造满足指定掌宽和食指长的 21 点 landmarks。

    frontality = raw_hand_width / index_len
    高正面度：raw_hand_width 大 / index_len 小 → frontality >= 0.45
    低正面度：raw_hand_width 小 / index_len 大 → frontality < 0.45
    """
    pts = [[i, 0.0, 0.0] for i in range(21)]
    pts[0] = [0, cx, cy + 40.0]                    # wrist
    pts[2] = [2, cx - 10.0, cy]                    # thumb MCP
    pts[3] = [3, cx - 15.0, cy - 5.0]              # thumb IP
    pts[4] = [4, cx - 20.0, cy - 10.0]             # thumb tip
    pts[5] = [5, cx, cy]                           # index MCP (掌宽起点)
    pts[6] = [6, cx, cy - index_len * 0.5]         # index PIP
    pts[8] = [8, cx, cy - index_len]               # index tip (食指长 = index_len)
    pts[9] = [9, cx + 20.0, cy]                    # middle MCP
    pts[10] = [10, cx + 20.0, cy - 10.0]           # middle PIP
    pts[12] = [12, cx + 20.0, cy - 20.0]           # middle tip
    pts[13] = [13, cx + 40.0, cy]                  # ring MCP
    pts[14] = [14, cx + 40.0, cy - 10.0]           # ring PIP
    pts[16] = [16, cx + 40.0, cy - 20.0]           # ring tip
    pts[17] = [17, cx + raw_hand_width, cy]        # pinky MCP (掌宽 = raw_hand_width)
    pts[18] = [18, cx + 60.0, cy - 10.0]           # pinky PIP
    pts[20] = [20, cx + 60.0, cy - 20.0]           # pinky tip
    return pts


class TestHandWidthEma(unittest.TestCase):
    """hand_width 慢速 EMA 行为表征。"""

    def test_first_frame_uses_raw_value(self):
        """EMA=None 时首帧 hand_width = raw_hand_width（无滞后）。"""
        gr = GestureRecognizer()
        # raw=90, index_len=100 → frontality=0.9 >= 0.45 → 高正面度
        hand = _make_hand(raw_hand_width=90.0, index_len=100.0)
        features = gr.get_hand_features(hand)
        self.assertAlmostEqual(features["hand_width"], 90.0, places=5)
        self.assertIsNotNone(gr._hand_width_ema)

    def test_ema_smooths_high_frontality_spike(self):
        """高正面度序列中掌宽突变（90→208→90），EMA 平滑不直接跟到极值。

        模拟实测的掌宽抖动：同一只手因检测噪声 raw 在 58↔208 间跳变。
        """
        gr = GestureRecognizer()
        hand_90 = _make_hand(raw_hand_width=90.0, index_len=100.0)   # frontality=0.9
        hand_208 = _make_hand(raw_hand_width=208.0, index_len=100.0) # frontality=2.08

        # 帧1-2: 稳定在 90
        gr.get_hand_features(hand_90)
        gr.get_hand_features(hand_90)
        self.assertAlmostEqual(gr._hand_width_ema, 90.0, places=3)

        # 帧3: 突然跳到 208（高正面度，EMA 会更新但不直接跟到 208）
        f3 = gr.get_hand_features(hand_208)
        self.assertLess(f3["hand_width"], 208.0, "EMA 不应直接跳到极值 208")
        self.assertGreater(f3["hand_width"], 90.0, "EMA 应被 208 拉高")
        # EMA = 0.9*90 + 0.1*208 = 81 + 20.8 = 101.8
        self.assertAlmostEqual(f3["hand_width"], 101.8, places=1)

        # 帧4-5: 回落到 90，EMA 慢慢收敛但仍有滞后
        f5 = gr.get_hand_features(hand_90)
        gr.get_hand_features(hand_90)
        self.assertGreater(f5["hand_width"], 90.0, "EMA 滞后应保持 > 90")

    def test_low_frontality_does_not_update_ema(self):
        """低正面度时 EMA 不更新，hand_width 保持上一帧 EMA 值。"""
        gr = GestureRecognizer()
        # 先用高正面度手初始化 EMA=90
        hand_high = _make_hand(raw_hand_width=90.0, index_len=100.0)  # frontality=0.9
        gr.get_hand_features(hand_high)
        self.assertAlmostEqual(gr._hand_width_ema, 90.0, places=3)

        # 喂低正面度手：raw=30, index_len=100 → frontality=0.3 < 0.45
        hand_low = _make_hand(raw_hand_width=30.0, index_len=100.0)
        features = gr.get_hand_features(hand_low)
        # EMA 未更新，仍为 90
        self.assertAlmostEqual(gr._hand_width_ema, 90.0, places=3,
                               msg="低正面度不应更新 EMA")
        # hand_width 返回 EMA 值（90），不是 raw 值（30）
        self.assertAlmostEqual(features["hand_width"], 90.0, places=3,
                               msg="低正面度时 hand_width 应用 EMA 值而非 raw")

    def test_ema_first_frame_low_frontality_uses_raw(self):
        """首帧即低正面度 → EMA 未初始化，hand_width = raw（兜底）。"""
        gr = GestureRecognizer()
        # raw=30, index_len=100 → frontality=0.3 < 0.45
        hand = _make_hand(raw_hand_width=30.0, index_len=100.0)
        features = gr.get_hand_features(hand)
        # EMA 未更新（仍 None）
        self.assertIsNone(gr._hand_width_ema,
                          "低正面度不应初始化 EMA")
        # hand_width 用 raw 兜底
        self.assertAlmostEqual(features["hand_width"], 30.0, places=5,
                               msg="EMA 未初始化时应用 raw 值兜底")

    def test_reset_state_does_not_clear_ema(self):
        """_reset_state 不清 EMA，保持跨模式切换的掌宽连续性。

        用户切换模式（如 mouse → draw）时不应重置 EMA，否则又要重新预热，
        导致模式切换瞬间手势阈值跳变。
        """
        gr = GestureRecognizer()
        # 初始化 EMA=90
        hand = _make_hand(raw_hand_width=90.0, index_len=100.0)
        gr.get_hand_features(hand)
        self.assertAlmostEqual(gr._hand_width_ema, 90.0, places=3)

        # 调用 _reset_state
        gr._reset_state()
        # EMA 仍为 90（保持连续性）
        self.assertAlmostEqual(gr._hand_width_ema, 90.0, places=3,
                               msg="_reset_state 不应清空 EMA")


if __name__ == "__main__":
    unittest.main()
