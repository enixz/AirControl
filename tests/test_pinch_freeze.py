"""实施方案 Phase 3.1: Freeze-on-pinch 状态机单元测试

验证捏合上升沿冻结光标的 grace 期逻辑：
1. 默认关闭时光标正常移动
2. 开启后上升沿记录冻结位置
3. grace 期内光标锁定在冻结位置（跳过 move_to_normalized）
4. grace 结束后解冻恢复移动（DRAG）
5. pinch 释放立即解冻
6. 手丢失 / on_exit 清除冻结状态

注意：单元测试只测状态机逻辑本身，不测实际手势场景效果（参考项目 memory 教训）。
"""
import os
import sys
import time
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'app'))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modes.mouse_mode import MouseMode


def _make_landmarks(pinch=False, hand_width=80.0):
    """构造 21 关键点 landmarks [idx, x, y]。

    pinch=True 时拇指尖靠近食指尖（距离 < 掌宽×0.35）；
    pinch=False 时拇指尖远离食指尖。
    """
    pts = []
    wrist_x, wrist_y = 320.0, 240.0
    # 0: wrist
    pts.append([0, wrist_x, wrist_y])
    # 1-4: thumb
    pts.append([1, wrist_x - 15, wrist_y - 10])
    pts.append([2, wrist_x - 25, wrist_y - 20])
    pts.append([3, wrist_x - 30, wrist_y - 25])
    if pinch:
        pts.append([4, 330.0, 200.0])   # 拇指尖靠近食指尖
    else:
        pts.append([4, 280.0, 210.0])   # 拇指尖远离食指尖
    # 5: index MCP
    pts.append([5, wrist_x + 10, wrist_y - 30])
    # 6-8: index
    pts.append([6, wrist_x + 15, wrist_y - 50])
    pts.append([7, wrist_x + 18, wrist_y - 65])
    pts.append([8, 340.0, 190.0])      # 食指尖
    # 9: middle MCP
    pts.append([9, wrist_x + 25, wrist_y - 35])
    # 10-12: middle
    pts.append([10, wrist_x + 30, wrist_y - 55])
    pts.append([11, wrist_x + 33, wrist_y - 70])
    pts.append([12, 360.0, 185.0])
    # 13-16: ring
    pts.append([13, wrist_x + 40, wrist_y - 38])
    pts.append([14, wrist_x + 45, wrist_y - 58])
    pts.append([15, wrist_x + 48, wrist_y - 72])
    pts.append([16, 390.0, 190.0])
    # 17: pinky MCP
    pts.append([17, wrist_x + 55, wrist_y - 35])
    # 18-20: pinky
    pts.append([18, wrist_x + 60, wrist_y - 50])
    pts.append([19, wrist_x + 63, wrist_y - 62])
    pts.append([20, 410.0, 205.0])
    return pts


def _features_for(pinch):
    """返回与 _make_landmarks(pinch) 对应的特征字典子集。"""
    return {
        "thumb_index_pinch": pinch,
        "thumb_middle_pinch": False,
        "hand_width": 80.0,
        "index_drawing_pose": False,
        "thumb_tucked": False,
        "index_middle_up": False,
    }


class _FakeConfig:
    """简单的 dict 包装，支持 .get(key, default) 和真值检测。"""

    def __init__(self, **kwargs):
        self._d = dict(kwargs)

    def get(self, key, default=None):
        return self._d.get(key, default)

    def __bool__(self):
        return True


def _make_mouse_mode(pinch_freeze_enabled=False, grace_sec=0.3):
    """构造一个带 mock 依赖的 MouseMode 实例。"""
    config = _FakeConfig(
        edge_acceleration_enabled=False,
        pinch_freeze_enabled=pinch_freeze_enabled,
        pinch_freeze_grace_sec=grace_sec,
    )
    recognizer = MagicMock()
    recognizer.REFERENCE_HAND_WIDTH = 90.0
    recognizer.get_hand_features.return_value = _features_for(pinch=False)
    recognizer.check_scroll.return_value = 0
    mouse = MagicMock()
    # move_to_normalized 返回 (screen_x, screen_y)
    mouse.move_to_normalized.return_value = (500, 400)
    overlay = MagicMock()
    cursor_overlay = MagicMock()
    toolbar = MagicMock()
    ppt = MagicMock()

    mode = MouseMode(config, recognizer, mouse, overlay, cursor_overlay, toolbar, ppt)
    return mode, mouse, recognizer


class TestPinchFreezeDefaults(unittest.TestCase):
    """验证配置默认值和 schema。"""

    def test_freeze_disabled_by_default(self):
        """pinch_freeze_enabled 默认 False（可回退）。"""
        mode, mouse, recognizer = _make_mouse_mode(pinch_freeze_enabled=False)
        mode.on_enter()
        self.assertFalse(mode.config.get("pinch_freeze_enabled"))
        self.assertIsNone(mode._frozen_pos)
        self.assertIsNone(mode._freeze_start)
        mode.on_exit()


class TestPinchFreezeStateMachine(unittest.TestCase):
    """验证 freeze-on-pinch 状态机转换。"""

    def test_freeze_disabled_cursor_moves_during_pinch(self):
        """关闭 freeze 时，pinch 期间光标照常移动。"""
        mode, mouse, recognizer = _make_mouse_mode(pinch_freeze_enabled=False)
        mode.on_enter()

        # 帧 1：无 pinch，光标移动到 (500, 400)
        recognizer.get_hand_features.return_value = _features_for(pinch=False)
        mode.handle([_make_landmarks(pinch=False)], [], 640, 480)
        self.assertEqual(mouse.move_to_normalized.call_count, 1)

        # 帧 2：pinch 上升沿，left_down 触发，但 freeze 关闭→光标仍移动
        mouse.move_to_normalized.reset_mock()
        mouse.move_to_normalized.return_value = (510, 410)
        recognizer.get_hand_features.return_value = _features_for(pinch=True)
        mode.handle([_make_landmarks(pinch=True)], [], 640, 480)
        # move_to_normalized 仍然被调用（未冻结）
        self.assertEqual(mouse.move_to_normalized.call_count, 1)
        # 冻结状态未设置
        self.assertIsNone(mode._frozen_pos)
        mode.on_exit()

    def test_freeze_records_position_on_rising_edge(self):
        """开启 freeze 后，pinch 上升沿记录冻结位置。"""
        mode, mouse, recognizer = _make_mouse_mode(
            pinch_freeze_enabled=True, grace_sec=0.3
        )
        mode.on_enter()

        # 帧 1：无 pinch
        mouse.move_to_normalized.return_value = (500, 400)
        recognizer.get_hand_features.return_value = _features_for(pinch=False)
        mode.handle([_make_landmarks(pinch=False)], [], 640, 480)
        self.assertIsNone(mode._frozen_pos)

        # 帧 2：pinch 上升沿，move_to_normalized 返回 (510, 410)
        mouse.move_to_normalized.return_value = (510, 410)
        recognizer.get_hand_features.return_value = _features_for(pinch=True)
        mode.handle([_make_landmarks(pinch=True)], [], 640, 480)
        # 冻结位置已记录为上升沿的瞄准点
        self.assertIsNotNone(mode._frozen_pos)
        self.assertEqual(mode._frozen_pos, (510, 410))
        self.assertIsNotNone(mode._freeze_start)
        mode.on_exit()

    def test_freeze_locks_cursor_during_grace(self):
        """grace 期内光标锁定在冻结位置，跳过 move_to_normalized。"""
        mode, mouse, recognizer = _make_mouse_mode(
            pinch_freeze_enabled=True, grace_sec=0.3
        )
        mode.on_enter()

        # 上升沿：记录冻结位置 (510, 410)
        mouse.move_to_normalized.return_value = (510, 410)
        recognizer.get_hand_features.return_value = _features_for(pinch=True)
        mode.handle([_make_landmarks(pinch=True)], [], 640, 480)
        self.assertEqual(mode._frozen_pos, (510, 410))
        rising_edge_calls = mouse.move_to_normalized.call_count

        # 后续帧（grace 期内）：仍 pinch，但 move_to_normalized 不应被再次调用
        mouse.move_to_normalized.reset_mock()
        mouse.move_to_normalized.return_value = (999, 999)  # 若被调用会返回此值
        for _ in range(3):
            mode.handle([_make_landmarks(pinch=True)], [], 640, 480)

        # move_to_normalized 在 grace 期内未被调用（光标冻结）
        self.assertEqual(mouse.move_to_normalized.call_count, 0)
        # 冻结位置未变
        self.assertEqual(mode._frozen_pos, (510, 410))
        mode.on_exit()

    def test_freeze_releases_after_grace(self):
        """grace 结束后解冻，恢复 move_to_normalized 调用（DRAG）。"""
        mode, mouse, recognizer = _make_mouse_mode(
            pinch_freeze_enabled=True, grace_sec=0.05  # 50ms grace，便于测试
        )
        mode.on_enter()

        # 上升沿
        mouse.move_to_normalized.return_value = (510, 410)
        recognizer.get_hand_features.return_value = _features_for(pinch=True)
        mode.handle([_make_landmarks(pinch=True)], [], 640, 480)
        self.assertEqual(mode._frozen_pos, (510, 410))

        # 等待 grace 过期
        time.sleep(0.08)

        # grace 过期后下一帧：仍 pinch，但应解冻并调用 move_to_normalized
        mouse.move_to_normalized.reset_mock()
        mouse.move_to_normalized.return_value = (520, 420)
        mode.handle([_make_landmarks(pinch=True)], [], 640, 480)
        # move_to_normalized 被调用（已解冻）
        self.assertEqual(mouse.move_to_normalized.call_count, 1)
        # 冻结状态已清除
        self.assertIsNone(mode._frozen_pos)
        self.assertIsNone(mode._freeze_start)
        mode.on_exit()

    def test_freeze_clears_on_pinch_release(self):
        """pinch 释放时立即清除冻结状态（grace 是上限，不是固定时长）。"""
        mode, mouse, recognizer = _make_mouse_mode(
            pinch_freeze_enabled=True, grace_sec=10.0  # 长 grace
        )
        mode.on_enter()

        # 上升沿
        mouse.move_to_normalized.return_value = (510, 410)
        recognizer.get_hand_features.return_value = _features_for(pinch=True)
        mode.handle([_make_landmarks(pinch=True)], [], 640, 480)
        self.assertIsNotNone(mode._frozen_pos)

        # pinch 释放（grace 远未结束）
        recognizer.get_hand_features.return_value = _features_for(pinch=False)
        mode.handle([_make_landmarks(pinch=False)], [], 640, 480)
        # 冻结状态立即清除
        self.assertIsNone(mode._frozen_pos)
        self.assertIsNone(mode._freeze_start)
        mode.on_exit()

    def test_freeze_clears_on_hand_lost(self):
        """手丢失时清除冻结状态。"""
        mode, mouse, recognizer = _make_mouse_mode(
            pinch_freeze_enabled=True, grace_sec=10.0
        )
        mode.on_enter()

        # 上升沿
        mouse.move_to_normalized.return_value = (510, 410)
        recognizer.get_hand_features.return_value = _features_for(pinch=True)
        mode.handle([_make_landmarks(pinch=True)], [], 640, 480)
        self.assertIsNotNone(mode._frozen_pos)

        # 手丢失
        mode.handle([], [], 640, 480)
        self.assertIsNone(mode._frozen_pos)
        self.assertIsNone(mode._freeze_start)
        mode.on_exit()

    def test_freeze_clears_on_exit(self):
        """on_exit 清除冻结状态。"""
        mode, mouse, recognizer = _make_mouse_mode(
            pinch_freeze_enabled=True, grace_sec=10.0
        )
        mode.on_enter()

        # 上升沿
        mouse.move_to_normalized.return_value = (510, 410)
        recognizer.get_hand_features.return_value = _features_for(pinch=True)
        mode.handle([_make_landmarks(pinch=True)], [], 640, 480)
        self.assertIsNotNone(mode._frozen_pos)

        mode.on_exit()
        self.assertIsNone(mode._frozen_pos)
        self.assertIsNone(mode._freeze_start)


if __name__ == '__main__':
    unittest.main()
