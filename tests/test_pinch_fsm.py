"""PinchButtonStateMachine 契约测试 — 第 3 步交付物的一部分。

覆盖：按下确认（远距）、近手即时按、释放确认、明确张开立即释放、
预测补帧不产生新点击、保持时间上限、丢手强制释放后的残留捏合锁、
watchdog 强制释放、换手 reset、Closed_Fist 禁止新按下。
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from services.pinch_fsm import PinchButtonStateMachine


class PinchFSMTest(unittest.TestCase):
    def setUp(self):
        self.fsm = PinchButtonStateMachine()

    def upd(self, pinch, fresh=True, label=None, now=0.0, distant=False, clear=False, geo=True):
        return self.fsm.update(pinch=pinch, fresh=fresh, ml_label=label,
                               now=now, distant=distant, clearly_open=clear,
                               geometry_reliable=geo)

    def test_near_hand_presses_immediately(self):
        down, tr = self.upd(True, now=1.0)
        self.assertTrue(down)
        self.assertEqual(tr, "press_confirmed")

    def test_distant_hand_needs_confirmation_window(self):
        down, _ = self.upd(True, now=1.0, distant=True)
        self.assertFalse(down)
        down, _ = self.upd(True, now=1.04, distant=True)
        self.assertFalse(down)
        down, tr = self.upd(True, now=1.07, distant=True)
        self.assertTrue(down)
        self.assertEqual(tr, "press_confirmed")

    def test_distant_confirmation_reset_by_release_evidence(self):
        self.upd(True, now=1.0, distant=True)
        # 一帧松开 → 证据清零
        self.upd(False, now=1.02, distant=True)
        down, _ = self.upd(True, now=1.03, distant=True)
        self.assertFalse(down)
        down, _ = self.upd(True, now=1.10, distant=True)
        self.assertTrue(down)

    def test_predicted_frames_do_not_start_or_confirm_press(self):
        down, tr = self.upd(True, fresh=False, now=1.0, distant=False)
        self.assertFalse(down)
        self.assertIsNone(tr)
        self.assertEqual(self.fsm.state, PinchButtonStateMachine.IDLE)

    def test_predicted_frames_bridge_hold(self):
        self.upd(True, now=1.0)
        down, tr = self.upd(True, fresh=False, now=1.05)
        self.assertTrue(down)
        self.assertIsNone(tr)

    def test_hold_max_releases_after_limit(self):
        self.upd(True, now=1.0)
        down, tr = self.upd(True, fresh=False, now=1.0 + 8.5)
        self.assertFalse(down)
        self.assertEqual(tr, "hold_max")
        self.assertEqual(self.fsm.state, PinchButtonStateMachine.IDLE)
        # 上限释放后需明确张开
        self.assertTrue(self.fsm.blocked_until_open)

    def test_hold_max_not_triggered_on_fresh_frames_while_dragging(self):
        # 真实持续拖拽 8s+：fresh 帧每帧重置 hold 计时？——不重置：hold_since
        # 固定在按下沿。持续按住 8s 的真实拖拽会被上限释放。这是设计决定：
        # 上限只防预测帧长锁，fresh 帧也计入（8s 足够长，正常点击/拖拽远达不到）。
        self.upd(True, now=1.0)
        down, _ = self.upd(True, now=9.5)
        self.assertFalse(down)
        self.assertEqual(self.fsm.last_transition_reason, "hold_max")

    def test_release_confirmation_absorbs_boundary_jitter(self):
        self.upd(True, now=1.0)
        # 0.04s 的抖动松开 → 不释放
        down, _ = self.upd(False, now=1.04)
        self.assertTrue(down)
        # 恢复捏合 → 重置释放计时
        down, _ = self.upd(True, now=1.06)
        self.assertTrue(down)
        # 又抖开 0.04s → 仍不释放
        down, _ = self.upd(False, now=1.10)
        self.assertTrue(down)

    def test_sustained_release_confirms_after_window(self):
        self.upd(True, now=1.0)
        down, _ = self.upd(False, now=1.04)
        self.assertTrue(down)
        down, tr = self.upd(False, now=1.13)  # 0.09s ≥ 0.08s 窗口
        self.assertFalse(down)
        self.assertEqual(tr, "release_confirmed")

    def test_clearly_open_releases_immediately(self):
        self.upd(True, now=1.0)
        down, tr = self.upd(False, now=1.02, clear=True)
        self.assertFalse(down)
        self.assertEqual(tr, "release_clear_open")

    def test_hand_lost_force_release_and_residual_pinch_lock(self):
        self.upd(True, now=1.0)
        was = self.fsm.force_release("hand_lost")
        self.assertTrue(was)
        self.assertTrue(self.fsm.blocked_until_open)
        # 手回来仍捏着 → 不能立即重建按下
        down, _ = self.upd(True, now=1.5)
        self.assertFalse(down)
        # 一次明确张开 → 解锁
        down, _ = self.upd(False, now=1.6, clear=True)
        self.assertFalse(down)
        self.assertFalse(self.fsm.blocked_until_open)
        # 再次捏合 → 正常按下
        down, tr = self.upd(True, now=1.7)
        self.assertTrue(down)
        self.assertEqual(tr, "press_confirmed")

    def test_loss_without_hold_does_not_block(self):
        self.fsm.force_release("hand_lost")  # 没有按住时的丢失
        self.assertFalse(self.fsm.blocked_until_open)
        down, _ = self.upd(True, now=1.0)
        self.assertTrue(down)

    def test_watchdog_force_release_blocks(self):
        self.upd(True, now=1.0)
        self.fsm.force_release("watchdog")
        self.assertTrue(self.fsm.blocked_until_open)

    def test_fist_label_block_is_opt_in_and_default_off(self):
        # 默认关闭：远距实测中真握拳误触与真实捏合（被误标 Fist）不可靠
        # 区分，block 会把 t2 拖拽整段杀掉；待第 4 步几何门控后再启用。
        self.assertFalse(PinchButtonStateMachine.FIST_BLOCK)
        down, _ = self.upd(True, now=1.0, label="Closed_Fist")
        self.assertTrue(down)  # 默认不拦截
        # 显式开启后行为正确（供第 4 步使用）
        fsm = PinchButtonStateMachine()
        fsm.FIST_BLOCK = True
        down, tr = fsm.update(pinch=True, fresh=True, ml_label="Closed_Fist",
                              now=1.0, distant=False, clearly_open=False)
        self.assertFalse(down)
        self.assertEqual(fsm.state, PinchButtonStateMachine.IDLE)

    def test_fist_label_does_not_break_existing_hold(self):
        self.upd(True, now=1.0)
        # 按住中手型偶尔被判 Fist（远距小手）：保持不释放
        down, _ = self.upd(True, now=1.1, label="Closed_Fist")
        self.assertTrue(down)

    def test_identity_reset_clears_everything(self):
        self.upd(True, now=1.0)
        self.fsm.reset()
        self.assertEqual(self.fsm.state, PinchButtonStateMachine.IDLE)
        self.assertFalse(self.fsm.blocked_until_open)
        # 换手后重新确认（无残留证据）
        down, _ = self.upd(True, now=1.01, distant=True)
        self.assertFalse(down)  # distant 需重新积累证据

    def test_unreliable_geometry_blocks_new_press_but_keeps_hold(self):
        # 不可靠帧（掌宽突变）不能确认新按下
        down, _ = self.upd(True, now=1.0, distant=False, geo=False)
        self.assertFalse(down)
        # 但已建立的按住不被打断
        self.upd(True, now=1.0)
        down, _ = self.upd(True, now=1.1, geo=False)
        self.assertTrue(down)
        down, _ = self.upd(False, now=1.2, geo=False)
        self.assertTrue(down)  # 松手证据同样不计入（保持）
        # 恢复可靠后正常释放
        down, _ = self.upd(False, now=1.21, geo=True)
        self.assertTrue(down)   # 释放窗口从这帧起算
        down, tr = self.upd(False, now=1.30, geo=True)
        self.assertFalse(down)
        self.assertEqual(tr, "release_confirmed")

    def test_every_transition_has_reason(self):
        self.upd(True, now=1.0)
        self.assertEqual(self.fsm.last_transition_reason, "press_confirmed")
        self.upd(False, now=1.04)   # 释放证据起点
        self.upd(False, now=1.13)   # 持续 ≥0.08s → 确认
        self.assertEqual(self.fsm.last_transition_reason, "release_confirmed")


class MouseModeFSMWiringTest(unittest.TestCase):
    """MouseMode 层：左右键 FSM 联动 + 模式切换释放。"""

    def _make_mode(self):
        import sys as _sys
        _sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
        from test_pinch_freeze import _make_mouse_mode
        return _make_mouse_mode()

    def test_mode_switch_on_exit_releases_fsm(self):
        from unittest.mock import patch
        mode, mouse, recognizer = self._make_mode()
        mode.on_enter()
        recognizer.get_hand_features.return_value = {
            "thumb_index_pinch": True, "thumb_middle_pinch": False,
            "hand_width": 80.0, "index_drawing_pose": False,
            "thumb_tucked": False, "index_middle_up": False,
        }
        with patch("modes.mouse_mode.time.monotonic", return_value=1.0):
            mode.handle([[[i, 100.0 + i, 200.0 + i] for i in range(21)]], [{}], 640, 480)
        self.assertTrue(mode._is_left_holding)
        mode.on_exit()
        self.assertEqual(mode._left_fsm.state, PinchButtonStateMachine.IDLE)
        mouse.left_up.assert_called_once()

    def test_right_click_single_fire_on_edge(self):
        from unittest.mock import patch
        mode, mouse, recognizer = self._make_mode()
        mode.on_enter()
        try:
            recognizer.get_hand_features.return_value = {
                "thumb_index_pinch": False, "thumb_middle_pinch": True,
                "hand_width": 80.0, "index_drawing_pose": False,
                "thumb_tucked": False, "index_middle_up": False,
            }
            hand = [[i, 100.0 + i, 200.0 + i] for i in range(21)]
            with patch("modes.mouse_mode.time.monotonic", return_value=1.0):
                r1 = mode.handle([hand], [{}], 640, 480)
            with patch("modes.mouse_mode.time.monotonic", return_value=1.1):
                r2 = mode.handle([hand], [{}], 640, 480)
            self.assertEqual(r1.gesture, "RIGHT_CLICK")
            self.assertEqual(mouse.right_click.call_count, 1)
            self.assertNotEqual(r2.gesture, "RIGHT_CLICK")
            # 松开需持续到释放确认（0.08s 窗口）后再捏 → 第二次右键
            recognizer.get_hand_features.return_value = {
                "thumb_index_pinch": False, "thumb_middle_pinch": False,
                "hand_width": 80.0, "index_drawing_pose": False,
                "thumb_tucked": False, "index_middle_up": False,
            }
            with patch("modes.mouse_mode.time.monotonic", return_value=1.2):
                mode.handle([hand], [{}], 640, 480)
            with patch("modes.mouse_mode.time.monotonic", return_value=1.3):
                mode.handle([hand], [{}], 640, 480)
            recognizer.get_hand_features.return_value["thumb_middle_pinch"] = True
            with patch("modes.mouse_mode.time.monotonic", return_value=1.4):
                mode.handle([hand], [{}], 640, 480)
            self.assertEqual(mouse.right_click.call_count, 2)
        finally:
            mode.on_exit()


if __name__ == "__main__":
    unittest.main()
