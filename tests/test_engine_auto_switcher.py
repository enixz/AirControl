"""远距引擎自动切换的单元测试。

覆盖：
  - counts_toward_hand_streak 多手约束（YOLO 误检帧不计入"有手"计帧）
  - EngineAutoSwitcher FSM：mediapipe 连续无手 → hagrid_yolo；
    hagrid_yolo 连续稳定单手 → mediapipe；冷却防抖；手动 reset/configure
  - config schema 校验：engine_auto_switch_* 非法值回退默认
  - orchestrator 集成：FSM 判定后复用 _request_tracker_rebuild 路径、
    手动切引擎时状态机重置、环境变量强制引擎时不介入

仓库约定：不走真实 QApplication（mock/源码级）。
"""
import json
import os
import sys
import tempfile
import unittest
from unittest import mock
from unittest.mock import MagicMock, patch

_app_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app")
sys.path.insert(0, _app_dir)

from config_manager import ConfigManager
from orchestrator import AirControlOrchestrator
from services.engine_auto_switcher import (
    ENGINE_HAGRID_YOLO,
    ENGINE_MEDIAPIPE,
    EngineAutoSwitcher,
    counts_toward_hand_streak,
)

_TRACKER_SIG_MP = ("mediapipe", "Heavy", "Auto", 0.5, 0.5, 0.5, 0.5, 0.015)
_TRACKER_SIG_YOLO = ("hagrid_yolo", "Heavy", "Auto", 0.5, 0.5, 0.5, 0.5, 0.015)


class TestCountsTowardHandStreak(unittest.TestCase):
    """多手约束：恰好 1 只手才计入连续有手帧数。"""

    def test_single_hand_counts(self):
        self.assertTrue(counts_toward_hand_streak(1))

    def test_no_hand_does_not_count(self):
        self.assertFalse(counts_toward_hand_streak(0))

    def test_multi_hand_does_not_count(self):
        # YOLO 远距单人场景多手率 58%（conf=0.25 误检），≥2 只手不计入
        self.assertFalse(counts_toward_hand_streak(2))
        self.assertFalse(counts_toward_hand_streak(3))


class TestEngineAutoSwitcherFSM(unittest.TestCase):
    def _make(self, **kwargs):
        defaults = {"enabled": True, "no_hand_frames": 3,
                    "hand_frames": 4, "cooldown_sec": 5.0}
        defaults.update(kwargs)
        return EngineAutoSwitcher(**defaults)

    def test_disabled_never_switches(self):
        sw = EngineAutoSwitcher(enabled=False)
        for i in range(100):
            self.assertIsNone(sw.update(ENGINE_MEDIAPIPE, 0, now=float(i)))

    def test_mediapipe_no_hand_streak_switches_to_yolo(self):
        sw = self._make()
        self.assertIsNone(sw.update(ENGINE_MEDIAPIPE, 0, now=0.0))
        self.assertIsNone(sw.update(ENGINE_MEDIAPIPE, 0, now=0.1))
        self.assertEqual(sw.update(ENGINE_MEDIAPIPE, 0, now=0.2), ENGINE_HAGRID_YOLO)
        self.assertIn("未检到手", sw.last_reason)

    def test_hand_frame_resets_no_hand_streak(self):
        sw = self._make()
        sw.update(ENGINE_MEDIAPIPE, 0, now=0.0)
        sw.update(ENGINE_MEDIAPIPE, 0, now=0.1)
        sw.update(ENGINE_MEDIAPIPE, 1, now=0.2)  # 有手 → 连续无手清零
        self.assertIsNone(sw.update(ENGINE_MEDIAPIPE, 0, now=0.3))
        self.assertIsNone(sw.update(ENGINE_MEDIAPIPE, 0, now=0.4))
        self.assertEqual(sw.update(ENGINE_MEDIAPIPE, 0, now=0.5), ENGINE_HAGRID_YOLO)

    def test_cooldown_blocks_and_pauses_counting(self):
        sw = self._make()
        self.assertEqual(sw.update(ENGINE_MEDIAPIPE, 0, now=0.0), None)
        sw.update(ENGINE_MEDIAPIPE, 0, now=0.1)
        self.assertEqual(sw.update(ENGINE_MEDIAPIPE, 0, now=0.2), ENGINE_HAGRID_YOLO)
        # 冷却 5 秒内：有手帧不计入、不切换
        for i in range(10):
            self.assertIsNone(
                sw.update(ENGINE_HAGRID_YOLO, 1, now=1.0 + i * 0.1)
            )
        # 冷却结束后需重新累计 hand_frames 帧才能切回
        t = 0.2 + 5.0
        for i in range(3):
            self.assertIsNone(sw.update(ENGINE_HAGRID_YOLO, 1, now=t + i * 0.1))
        self.assertEqual(
            sw.update(ENGINE_HAGRID_YOLO, 1, now=t + 0.3), ENGINE_MEDIAPIPE
        )
        self.assertIn("稳定单手", sw.last_reason)

    def test_yolo_no_hand_frame_resets_hand_streak(self):
        sw = self._make()
        for i in range(3):
            sw.update(ENGINE_HAGRID_YOLO, 1, now=float(i))
        sw.update(ENGINE_HAGRID_YOLO, 0, now=3.0)  # 无手 → 连续有手清零
        for i in range(3):
            self.assertIsNone(sw.update(ENGINE_HAGRID_YOLO, 1, now=4.0 + i))
        self.assertEqual(sw.update(ENGINE_HAGRID_YOLO, 1, now=7.0), ENGINE_MEDIAPIPE)

    def test_multi_hand_frames_neither_count_nor_reset(self):
        """疑似误检的多手帧：不计入也不清零（58% 多手率下仍能凑够切回阈值）。"""
        sw = self._make()
        sw.update(ENGINE_HAGRID_YOLO, 1, now=0.0)
        sw.update(ENGINE_HAGRID_YOLO, 1, now=0.1)
        # 插入多手帧：连续计帧暂停但不清零
        for i in range(10):
            self.assertIsNone(sw.update(ENGINE_HAGRID_YOLO, 2, now=1.0 + i * 0.1))
        self.assertIsNone(sw.update(ENGINE_HAGRID_YOLO, 1, now=2.0))
        self.assertEqual(sw.update(ENGINE_HAGRID_YOLO, 1, now=2.1), ENGINE_MEDIAPIPE)

    def test_multi_hand_alone_never_switches_back(self):
        sw = self._make()
        for i in range(50):
            self.assertIsNone(sw.update(ENGINE_HAGRID_YOLO, 2, now=float(i)))

    def test_unknown_engine_is_noop(self):
        sw = self._make()
        for i in range(10):
            self.assertIsNone(sw.update("unknown_engine", 0, now=float(i)))

    def test_reset_clears_streaks_and_cooldown(self):
        sw = self._make()
        sw.update(ENGINE_MEDIAPIPE, 0, now=0.0)
        sw.update(ENGINE_MEDIAPIPE, 0, now=0.1)
        self.assertEqual(sw.update(ENGINE_MEDIAPIPE, 0, now=0.2), ENGINE_HAGRID_YOLO)
        sw.reset()  # 手动切引擎 → 冷却与计帧清空
        self.assertEqual(sw.last_reason, "")
        # 冷却已清空：立即可以重新累计（此处连续无手 3 帧又触发切 yolo）
        for i in range(2):
            self.assertIsNone(sw.update(ENGINE_MEDIAPIPE, 0, now=0.3 + i * 0.1))
        self.assertEqual(sw.update(ENGINE_MEDIAPIPE, 0, now=0.5), ENGINE_HAGRID_YOLO)

    def test_configure_updates_params_and_resets(self):
        sw = self._make()
        sw.update(ENGINE_MEDIAPIPE, 0, now=0.0)
        sw.update(ENGINE_MEDIAPIPE, 0, now=0.1)
        sw.configure(no_hand_frames=10, enabled=True)
        # 计帧已重置 + 新阈值 10：旧累计不带入
        for i in range(9):
            self.assertIsNone(sw.update(ENGINE_MEDIAPIPE, 0, now=1.0 + i))
        self.assertEqual(sw.update(ENGINE_MEDIAPIPE, 0, now=10.0), ENGINE_HAGRID_YOLO)

    def test_default_clock_is_used_when_now_omitted(self):
        sw = self._make()
        # 不传 now 不应抛异常（走 time.monotonic）
        self.assertIsNone(sw.update(ENGINE_MEDIAPIPE, 1))


class TestEngineAutoSwitchConfigSchema(unittest.TestCase):
    def _load(self, payload):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "config.json")
            with open(path, "w", encoding="utf-8") as stream:
                json.dump(payload, stream)
            return ConfigManager(path)

    def test_defaults_present_in_fresh_config(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ConfigManager(os.path.join(temp_dir, "config.json"))
            self.assertIs(config.get("engine_auto_switch"), False)
            self.assertEqual(config.get("engine_auto_switch_no_hand_frames"), 60)
            self.assertEqual(config.get("engine_auto_switch_hand_frames"), 90)
            self.assertEqual(config.get("engine_auto_switch_cooldown_sec"), 5.0)

    def test_invalid_values_fall_back_to_defaults(self):
        config = self._load({
            "engine_auto_switch": "yes",
            "engine_auto_switch_no_hand_frames": 0,
            "engine_auto_switch_hand_frames": 9999,
            "engine_auto_switch_cooldown_sec": -1,
        })
        self.assertIs(config.get("engine_auto_switch"), False)
        self.assertEqual(config.get("engine_auto_switch_no_hand_frames"), 60)
        self.assertEqual(config.get("engine_auto_switch_hand_frames"), 90)
        self.assertEqual(config.get("engine_auto_switch_cooldown_sec"), 5.0)

    def test_valid_values_accepted(self):
        config = self._load({
            "engine_auto_switch": True,
            "engine_auto_switch_no_hand_frames": 45,
            "engine_auto_switch_hand_frames": 120,
            "engine_auto_switch_cooldown_sec": 3.5,
        })
        self.assertIs(config.get("engine_auto_switch"), True)
        self.assertEqual(config.get("engine_auto_switch_no_hand_frames"), 45)
        self.assertEqual(config.get("engine_auto_switch_hand_frames"), 120)
        self.assertEqual(config.get("engine_auto_switch_cooldown_sec"), 3.5)


def _make_orchestrator():
    """构造跳过 init_services 的 orchestrator（仓库 mock 风格，不起 QApplication）。"""
    with patch("orchestrator.AirControlOrchestrator.init_services"), \
         patch("orchestrator.AirControlOrchestrator._init_modes"), \
         patch("orchestrator.AirControlOrchestrator.set_mode"), \
         patch("orchestrator.ConfigManager"), \
         patch("orchestrator.MouseController"):
        return AirControlOrchestrator(MagicMock(), MagicMock(), MagicMock())


def _wire_for_auto_switch(orch, no_hand_frames=3, hand_frames=3):
    orch._tracker_config_signature = _TRACKER_SIG_MP
    orch._engine_switcher.configure(
        enabled=True, no_hand_frames=no_hand_frames,
        hand_frames=hand_frames, cooldown_sec=5.0,
    )
    orch._request_tracker_rebuild = MagicMock()
    orch._tracker_signature = MagicMock(return_value=_TRACKER_SIG_YOLO)


class TestOrchestratorAutoSwitch(unittest.TestCase):
    def test_fsm_decision_triggers_tracker_rebuild_with_override(self):
        orch = _make_orchestrator()
        _wire_for_auto_switch(orch)
        orch._maybe_auto_switch_engine([object()])  # 有手：不计
        orch._maybe_auto_switch_engine([])
        orch._maybe_auto_switch_engine([])
        orch._request_tracker_rebuild.assert_not_called()
        orch._maybe_auto_switch_engine([])  # 第 3 帧连续无手 → 切换
        self.assertEqual(orch._engine_override, ENGINE_HAGRID_YOLO)
        orch._request_tracker_rebuild.assert_called_once_with(_TRACKER_SIG_YOLO)

    def test_pending_rebuild_blocks_duplicate_requests(self):
        orch = _make_orchestrator()
        _wire_for_auto_switch(orch)
        for _ in range(3):
            orch._maybe_auto_switch_engine([])
        orch._request_tracker_rebuild.assert_called_once()
        # 重建未落地（签名仍是 mediapipe）：继续喂帧不重复发起
        for _ in range(10):
            orch._maybe_auto_switch_engine([])
        orch._request_tracker_rebuild.assert_called_once()

    def test_disabled_switcher_is_noop(self):
        orch = _make_orchestrator()
        orch._tracker_config_signature = _TRACKER_SIG_MP
        orch._request_tracker_rebuild = MagicMock()
        for _ in range(100):
            orch._maybe_auto_switch_engine([])
        orch._request_tracker_rebuild.assert_not_called()
        self.assertIsNone(orch._engine_override)

    def test_env_var_forced_engine_blocks_auto_switch(self):
        orch = _make_orchestrator()
        _wire_for_auto_switch(orch)
        with mock.patch.dict(os.environ, {"AIRCONTROL_ENGINE": ENGINE_MEDIAPIPE}):
            for _ in range(10):
                orch._maybe_auto_switch_engine([])
        orch._request_tracker_rebuild.assert_not_called()
        self.assertIsNone(orch._engine_override)

    def test_failed_rebuild_rolls_back_override(self):
        orch = _make_orchestrator()
        _wire_for_auto_switch(orch)
        for _ in range(3):
            orch._maybe_auto_switch_engine([])
        self.assertEqual(orch._engine_override, ENGINE_HAGRID_YOLO)
        # 后台重建失败：override 回退、FSM 重置，之后可以重试
        orch._on_tracker_ready(None, _TRACKER_SIG_YOLO, orch._tracker_request_id, "boom")
        self.assertIsNone(orch._engine_override)

    def _apply_config_with(self, orch, values):
        orch.config = MagicMock()
        orch.config.get.side_effect = lambda key, default=None: values.get(key, default)
        orch.recognizer = MagicMock()
        orch.ppt = MagicMock()
        orch.mouse = MagicMock()
        orch.overlay = MagicMock()
        orch.voice_assistant = MagicMock()
        orch.voice_command = MagicMock()
        orch.mode_manager = MagicMock(current_mode_name="mouse")
        orch._tracker_signature = MagicMock(return_value="same")
        orch._tracker_config_signature = "same"
        orch._current_voice_kws_signature = MagicMock(return_value="same")
        orch._voice_kws_signature = "same"
        orch.apply_config()

    def test_manual_engine_change_respects_user_choice(self):
        """用户手动切引擎：清除 FSM 覆盖并重置状态机，不与手动选择打架。"""
        orch = _make_orchestrator()
        orch._last_config_engine = ENGINE_MEDIAPIPE
        orch._engine_override = ENGINE_HAGRID_YOLO
        orch._engine_switcher.enabled = True
        orch._engine_switcher._no_hand_streak = 7
        self._apply_config_with(orch, {
            "detection_engine": ENGINE_HAGRID_YOLO,
            "interaction_mode": "mouse",
            "cooldown": 1.0,
        })
        self.assertIsNone(orch._engine_override)
        self.assertEqual(orch._last_config_engine, ENGINE_HAGRID_YOLO)
        self.assertEqual(orch._engine_switcher._no_hand_streak, 0)
        # config 未开 engine_auto_switch → 状态机被配置为关闭
        self.assertFalse(orch._engine_switcher.enabled)

    def test_disabling_auto_switch_reverts_to_configured_engine(self):
        """跑着 FSM 覆盖引擎时用户关掉开关：回到 config 配置的引擎。"""
        orch = _make_orchestrator()
        orch._last_config_engine = ENGINE_MEDIAPIPE
        orch._engine_override = ENGINE_HAGRID_YOLO
        orch._engine_switcher.enabled = True
        self._apply_config_with(orch, {
            "detection_engine": ENGINE_MEDIAPIPE,
            "interaction_mode": "mouse",
            "cooldown": 1.0,
        })
        self.assertIsNone(orch._engine_override)
        self.assertFalse(orch._engine_switcher.enabled)


if __name__ == "__main__":
    unittest.main()
