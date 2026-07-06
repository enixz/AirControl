"""BaseHandTracker.migrate_state_from 测试（P1-12 RCU 风格状态迁移）。

验证配置变更触发 tracker 重建时，关键运行时状态被正确迁移：
- crop-zoom 视口状态
- 活动手标识
- 平滑器（光标位置连续性）
- 运动 EMA
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'app'))

from services.base_hand_tracker import BaseHandTracker, OneEuroSmoother


class DummyHandTracker(BaseHandTracker):
    """用于测试 BaseHandTracker 逻辑的哑实现。"""
    @property
    def engine_name(self) -> str:
        return "dummy"

    def _detect(self, frame):
        return [], [], []

    def _detect_crop_zoom(self, frame, hint_center, hint_size):
        return [], [], []


class TestMigrateStateFrom(unittest.TestCase):
    def setUp(self):
        self.old = DummyHandTracker(config={})
        self.new = DummyHandTracker(config={})

    def test_migrate_crop_zoom_state(self):
        """crop-zoom 视口状态应被迁移。"""
        self.old._crop_zoom_mode = True
        self.old._current_crop_center = (100.0, 200.0)
        self.old._current_crop_size = 300.0
        self.old._last_hint_center = (150.0, 180.0)
        self.old._last_hint_size = 120.0

        self.new.migrate_state_from(self.old)

        self.assertTrue(self.new._crop_zoom_mode)
        self.assertEqual(self.new._current_crop_center, (100.0, 200.0))
        self.assertEqual(self.new._current_crop_size, 300.0)
        self.assertEqual(self.new._last_hint_center, (150.0, 180.0))
        self.assertEqual(self.new._last_hint_size, 120.0)

    def test_migrate_active_handedness(self):
        """活动手标识应被迁移（副本，非共享引用）。"""
        self.old._active_handedness = {"Left", "Right"}

        self.new.migrate_state_from(self.old)

        self.assertEqual(self.new._active_handedness, {"Left", "Right"})
        # 修改新 tracker 的集合不影响旧 tracker
        self.new._active_handedness.add("Extra")
        self.assertNotIn("Extra", self.old._active_handedness)

    def test_migrate_smoothers(self):
        """平滑器应被迁移（保持光标位置连续性）。"""
        # 给旧 tracker 的平滑器设置一些状态
        smoother = OneEuroSmoother()
        smoother.initialized = True
        smoother.last_landmarks = [[i, float(i), float(i)] for i in range(21)]
        self.old.smoothers["Right"] = smoother

        self.new.migrate_state_from(self.old)

        # 新 tracker 应有相同的平滑器
        self.assertIn("Right", self.new.smoothers)
        self.assertIs(self.new.smoothers["Right"], smoother)
        self.assertTrue(self.new.smoothers["Right"].initialized)
        self.assertIsNotNone(self.new.smoothers["Right"].last_landmarks)

    def test_migrate_motion_state(self):
        """运动 EMA 和手腕位置应被迁移。"""
        self.old._motion_ema = {"Right": 0.5, "Left": 0.3}
        self.old._last_wrist_pos = {"Right": (100, 200, 12345)}

        self.new.migrate_state_from(self.old)

        self.assertEqual(self.new._motion_ema, {"Right": 0.5, "Left": 0.3})
        self.assertEqual(self.new._last_wrist_pos, {"Right": (100, 200, 12345)})

    def test_migrate_from_none_is_noop(self):
        """从 None 迁移应安全无操作。"""
        tracker = DummyHandTracker(config={})
        tracker.migrate_state_from(None)  # 不应抛异常
        self.assertFalse(tracker._crop_zoom_mode)

    def test_migrate_does_not_share_dicts(self):
        """迁移后的 dict 应是副本，修改新 tracker 不影响旧 tracker。"""
        self.old._motion_ema = {"Right": 0.5}
        self.old._last_wrist_pos = {"Right": (1, 2, 3)}

        self.new.migrate_state_from(self.old)

        self.new._motion_ema["Right"] = 0.9
        self.new._last_wrist_pos["Right"] = (9, 8, 7)

        self.assertEqual(self.old._motion_ema["Right"], 0.5,
                         "旧 tracker 的 motion_ema 不应被修改")
        self.assertEqual(self.old._last_wrist_pos["Right"], (1, 2, 3),
                         "旧 tracker 的 last_wrist_pos 不应被修改")


if __name__ == "__main__":
    unittest.main(verbosity=2)
