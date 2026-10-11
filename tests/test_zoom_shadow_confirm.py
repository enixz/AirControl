"""影子 ZOOM 确认状态机（ZoomShadowConfirm）单元测试。

覆盖 docs/small_hand_tracking_progress.md 第 4 步的确认协议：
  - 窗口开启：仅同帧双新鲜 + 影子手腕落在捕获手中心 0.75×捕获手跨度内。
  - 窗口自持：仅影子新鲜 + 自身手腕连续；非新鲜帧挂起等待（与探针一致），
    新鲜帧跳变关窗。
  - 重播种：miss ≥ reseed_miss_frames 且当帧捕获新鲜时给出
    (bbox 中心, hint 尺寸)。
  - 晋升门槛：confirmations 达标且 0 < capture_bbox_ratio < near_bbox_ratio。

Tracker 输出格式：hands = [[[idx, x, y], ...], ...]（wrist 是第 0 点），
gestures = [{"predicted": bool, ...}, ...]。
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'app'))

from services.zoom_shadow_confirm import ZoomShadowConfirm


def make_hand(wrist=(100.0, 100.0), spread=40.0):
    """构造一只手：wrist 在 (wrist)，其余点向外扩散 spread 像素。"""
    hand = [[0, wrist[0], wrist[1]]]
    for i in range(1, 21):
        hand.append([i, wrist[0] + spread, wrist[1] + spread])
    return hand


def fresh_gesture():
    return {"predicted": False, "bbox_area": 100.0}


def predicted_gesture():
    return {"predicted": True, "bbox_area": 100.0}


class TestZoomShadowConfirm(unittest.TestCase):
    def _observe(self, confirm, cap_hand=None, cap_gesture=None,
                  shadow_hand=None, shadow_gesture=None,
                  bbox_ratio=0.001, miss=0, frame=0, zoom_on=True):
        """默认 None 表示"有新鲜手"，便于少参数调用。"""
        cap_hand = cap_hand or make_hand()
        cap_gesture = cap_gesture or fresh_gesture()
        shadow_hand = shadow_hand or make_hand()
        shadow_gesture = shadow_gesture or fresh_gesture()
        return confirm.observe(
            [cap_hand], [cap_gesture], bbox_ratio,
            [shadow_hand], [shadow_gesture], zoom_on, miss, frame,
        )

    # ------------------------------------------------------------------
    # 窗口开启
    # ------------------------------------------------------------------

    def test_window_opens_only_on_same_frame_match(self):
        """双新鲜 + 几何一致开窗；远手/预测帧不开窗。"""
        confirm = ZoomShadowConfirm(confirm_window_frames=5)
        # 远手：影子手腕距捕获手中心 > 0.75×跨度
        promote, _, _ = self._observe(
            confirm, shadow_hand=make_hand(wrist=(500.0, 500.0)))
        self.assertFalse(promote)
        self.assertEqual(confirm.confirm_remaining, 0)
        # 近手（在 0.75×跨度内）开窗
        self._observe(confirm, shadow_hand=make_hand(wrist=(120.0, 100.0)))
        self.assertEqual(confirm.confirm_remaining, 4)

    def test_window_does_not_open_on_predicted_either_side(self):
        """任一侧预测帧不算新鲜，开不了窗。"""
        confirm = ZoomShadowConfirm(confirm_window_frames=5)
        self._observe(confirm, cap_gesture=predicted_gesture(),
                      shadow_hand=make_hand(wrist=(120.0, 100.0)))
        self.assertEqual(confirm.confirm_remaining, 0)
        self._observe(confirm, shadow_gesture=predicted_gesture(),
                      shadow_hand=make_hand(wrist=(120.0, 100.0)))
        self.assertEqual(confirm.confirm_remaining, 0)

    def test_window_opens_on_empty_hands_never(self):
        """空输出不算新鲜。"""
        confirm = ZoomShadowConfirm(confirm_window_frames=5)
        confirm.observe([], [], 0.001, [make_hand(wrist=(120.0, 100.0))],
                        [fresh_gesture()], True, 0, 0)
        self.assertEqual(confirm.confirm_remaining, 0)
        confirm.observe([make_hand()], [fresh_gesture()], 0.001, [], [],
                        True, 0, 1)
        self.assertEqual(confirm.confirm_remaining, 0)

    # ------------------------------------------------------------------
    # 窗口自持
    # ------------------------------------------------------------------

    def test_window_completes_and_promotes(self):
        """开窗 + 4 帧自持完成 → confirmations=1 → 晋升。"""
        confirm = ZoomShadowConfirm(confirm_window_frames=5)
        self._observe(confirm)  # 开窗（剩余 4）
        for frame in range(1, 5):
            promote, reseed, _ = self._observe(confirm, frame=frame)
            if frame < 4:
                self.assertFalse(promote)
        # 第 5 帧（frame=4）窗口完成
        self.assertEqual(confirm.confirmations, 1)
        self.assertEqual(confirm.confirmed_frame, 4)
        # bbox_ratio=0.001 < near_bbox_ratio=0.04 → 可晋升
        promote, _, _ = self._observe(confirm, frame=5)
        self.assertTrue(promote)

    def test_predicted_frame_suspends_window(self):
        """自持期影子预测帧不关窗，挂起等待下一新鲜帧（与探针一致）。"""
        confirm = ZoomShadowConfirm(confirm_window_frames=5)
        self._observe(confirm)  # 开窗，剩余 4
        self._observe(confirm, frame=1)  # 自持 1
        # 预测帧：窗口挂起
        self._observe(confirm, shadow_gesture=predicted_gesture(), frame=2)
        self.assertEqual(confirm.confirm_remaining, 3)
        # 恢复新鲜且连续 → 窗口继续
        self._observe(confirm, frame=3)
        self.assertEqual(confirm.confirm_remaining, 2)

    def test_jump_closes_window(self):
        """自持期影子新鲜但跳变（不连续）→ 关窗，需重新开窗。"""
        confirm = ZoomShadowConfirm(confirm_window_frames=5)
        self._observe(confirm)  # 开窗，剩余 4，last_wrist=(100,100)
        self._observe(confirm, frame=1)  # 自持 1
        # 跳变：手腕跳到 500,500（远超 0.75×影子跨度）
        self._observe(confirm, shadow_hand=make_hand(wrist=(500.0, 500.0)),
                      frame=2)
        self.assertEqual(confirm.confirm_remaining, 0)
        self.assertEqual(confirm.confirmations, 0)
        # 重新开窗需要同帧一致
        self._observe(confirm, frame=3)
        self.assertEqual(confirm.confirm_remaining, 4)

    # ------------------------------------------------------------------
    # 视口播种（冷启动 / 失检重播种 / ZOOM OFF 边沿）
    # ------------------------------------------------------------------

    def test_cold_start_seeds_once(self):
        """冷启动：影子未开 ZOOM 时首个新鲜捕获位置播种，且只播一次。"""
        confirm = ZoomShadowConfirm()
        _, reseed, hint = self._observe(confirm, zoom_on=False)
        self.assertTrue(reseed)
        self.assertIsNotNone(hint)
        self.assertTrue(confirm.seeded)
        # 已播种且仍无 ZOOM：不再重复播（等影子自己把 ZOOM 开起来）
        _, reseed, hint = self._observe(confirm, zoom_on=False, miss=0)
        self.assertFalse(reseed)

    def test_cold_start_seeding_survives_until_shadow_zooms(self):
        """已播种但影子还没开 ZOOM 期间 miss 不触发额外播种（防抖）。"""
        confirm = ZoomShadowConfirm(reseed_miss_frames=2)
        self._observe(confirm, zoom_on=False)  # 冷启动播种
        _, reseed, _ = self._observe(confirm, zoom_on=False, miss=1)
        self.assertFalse(reseed)
        # miss 达阈值仍播种（救塌缩优先于防抖）
        _, reseed, _ = self._observe(confirm, zoom_on=False, miss=2)
        self.assertTrue(reseed)

    def test_zoom_off_edge_reseeds(self):
        """ZOOM ON→OFF 下降沿触发重播种（探针 zoom_reset 另一半条件）。"""
        confirm = ZoomShadowConfirm()
        self._observe(confirm, zoom_on=True)   # last_zoom_on=True
        _, reseed, _ = self._observe(confirm, zoom_on=False)
        self.assertTrue(reseed)
        # OFF 电平持续不再触发（边沿非电平）
        _, reseed, _ = self._observe(confirm, zoom_on=False)
        self.assertFalse(reseed)

    def test_reseed_fires_at_miss_threshold(self):
        """miss ≥ reseed_miss_frames 且捕获新鲜 → 建议重播种。"""
        confirm = ZoomShadowConfirm(reseed_miss_frames=2)
        _, reseed, hint = self._observe(confirm, miss=1)
        self.assertFalse(reseed)
        _, reseed, hint = self._observe(confirm, miss=2)
        self.assertTrue(reseed)
        center, size = hint
        # 默认手：wrist(100,100)、其余点 (140,140) → bbox 中心 (120,120)、
        # hint size = max(w,h) = 40
        self.assertAlmostEqual(center[0], 120.0)
        self.assertAlmostEqual(center[1], 120.0)
        self.assertAlmostEqual(size, 40.0)

    def test_reseed_requires_capture_fresh(self):
        """捕获预测帧时 miss 再大也不重播种（没有可信位置）。"""
        confirm = ZoomShadowConfirm(reseed_miss_frames=2)
        _, reseed, hint = self._observe(
            confirm, cap_gesture=predicted_gesture(), miss=5)
        self.assertFalse(reseed)
        self.assertIsNone(hint)

    # ------------------------------------------------------------------
    # 晋升门槛
    # ------------------------------------------------------------------

    def test_promotion_blocked_when_bbox_ratio_too_large(self):
        """近距大目标（bbox_ratio ≥ near_bbox_ratio）不晋升。"""
        confirm = ZoomShadowConfirm(near_bbox_ratio=0.04)
        # 先完成一次确认
        self._observe(confirm)
        for frame in range(1, 5):
            self._observe(confirm, frame=frame)
        self.assertEqual(confirm.confirmations, 1)
        # bbox_ratio = 0.05 ≥ 0.04 → 不晋升
        promote, _, _ = self._observe(confirm, bbox_ratio=0.05, frame=5)
        self.assertFalse(promote)
        # bbox_ratio = 0（无捕获手）→ 不晋升
        promote, _, _ = self._observe(confirm, bbox_ratio=0.0, frame=6)
        self.assertFalse(promote)
        # 正常远距小手 → 晋升
        promote, _, _ = self._observe(confirm, bbox_ratio=0.001, frame=7)
        self.assertTrue(promote)

    def test_promotion_requires_required_confirmations(self):
        """required_confirmations=2 时一次确认不够。"""
        confirm = ZoomShadowConfirm(
            required_confirmations=2, confirm_window_frames=5)
        self._observe(confirm)  # 开窗
        for frame in range(1, 5):
            promote, _, _ = self._observe(confirm, frame=frame)
        # 窗口完成 → confirmations=1，但需要 2
        self.assertEqual(confirm.confirmations, 1)
        promote, _, _ = self._observe(confirm, frame=5)
        self.assertFalse(promote)
        # 第二轮窗口
        self._observe(confirm, frame=6)  # 开窗
        for frame in range(7, 11):
            promote, _, _ = self._observe(confirm, frame=frame)
        self.assertEqual(confirm.confirmations, 2)
        promote, _, _ = self._observe(confirm, frame=11)
        self.assertTrue(promote)

    def test_reset_clears_state(self):
        confirm = ZoomShadowConfirm()
        self._observe(confirm)
        self._observe(confirm, frame=1)
        confirm.reset()
        self.assertEqual(confirm.confirm_remaining, 0)
        self.assertEqual(confirm.confirmations, 0)
        self.assertIsNone(confirm.last_confirmed_wrist)
        self.assertIsNone(confirm.confirmed_frame)
        # 会话簿记一并重置：新影子段可重新冷启动播种
        self.assertFalse(confirm.seeded)
        self.assertFalse(confirm.last_zoom_on)

    # ------------------------------------------------------------------
    # 参数约束
    # ------------------------------------------------------------------

    def test_constructor_clamps(self):
        confirm = ZoomShadowConfirm(required_confirmations=0,
                                    reseed_miss_frames=0,
                                    confirm_window_frames=1)
        self.assertEqual(confirm.required_confirmations, 1)
        self.assertEqual(confirm.reseed_miss_frames, 1)
        self.assertEqual(confirm.window_frames, 3)


if __name__ == "__main__":
    unittest.main()
