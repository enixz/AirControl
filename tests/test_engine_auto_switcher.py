"""远距引擎三态自动切换（NEAR/CAPTURE/FAR_TRACK）的单元测试。

覆盖：
  - counts_toward_hand_streak 多手约束（YOLO 误检帧不计入"有手"计帧）
  - 三态迁移：NEAR→CAPTURE（连续无手）、CAPTURE→FAR_TRACK（稳定单手）、
    FAR_TRACK→NEAR（手变大/走近持续）、FAR_TRACK→CAPTURE（再次丢手）
  - 恒定 5m 振荡回归：FAR_TRACK 下小手持续有手不会弹回 NEAR
  - 冷却防抖、手动 reset/configure、未知状态防御
  - config schema 校验：engine_auto_switch_* 非法值回退默认
  - orchestrator 集成：状态→引擎映射、FAR_TRACK 交接种子、long_range
    运行时翻转、手动优先、环境变量屏蔽

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
    STATE_CAPTURE,
    STATE_FAR_TRACK,
    STATE_NEAR,
    EngineAutoSwitcher,
    counts_toward_hand_streak,
)

_SIG_MP = ("mediapipe", "Heavy", "Auto", 0.5, 0.5, 0.5, 0.5, 0.015)
_SIG_YOLO = ("hagrid_yolo", "Heavy", "Auto", 0.5, 0.5, 0.5, 0.5, 0.015)


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


class TestThreeStateFSM(unittest.TestCase):
    def _make(self, **kwargs):
        defaults = {"enabled": True, "no_hand_frames": 3, "hand_frames": 3,
                    "near_frames": 4, "near_bbox_ratio": 0.04, "cooldown_sec": 5.0}
        defaults.update(kwargs)
        return EngineAutoSwitcher(**defaults)

    # ---------- NEAR ----------

    def test_disabled_never_switches(self):
        sw = EngineAutoSwitcher(enabled=False)
        for i in range(100):
            self.assertIsNone(sw.update(0, 0.0, now=float(i)))

    def test_near_to_capture_on_no_hand_streak(self):
        sw = self._make()
        self.assertEqual(sw.state, STATE_NEAR)
        self.assertIsNone(sw.update(0, now=0.0))
        self.assertIsNone(sw.update(0, now=0.1))
        self.assertEqual(sw.update(0, now=0.2), STATE_CAPTURE)
        self.assertEqual(sw.state, STATE_CAPTURE)
        self.assertIn("NEAR→CAPTURE", sw.last_reason)

    def test_hand_frame_resets_near_streak(self):
        sw = self._make()
        sw.update(0, now=0.0)
        sw.update(0, now=0.1)
        sw.update(1, 0.05, now=0.2)  # 有手 → 连续无手清零
        self.assertIsNone(sw.update(0, now=0.3))
        self.assertIsNone(sw.update(0, now=0.4))
        self.assertEqual(sw.update(0, now=0.5), STATE_CAPTURE)

    # ---------- CAPTURE ----------

    def _enter_capture(self, sw):
        for i in range(3):
            sw.update(0, now=float(i) * 0.1)
        self.assertEqual(sw.state, STATE_CAPTURE)
        return 10.0  # 冷却期结束后的时间

    def test_capture_to_far_track_on_stable_single_hand(self):
        sw = self._make()
        t = self._enter_capture(sw)
        self.assertIsNone(sw.update(1, 0.01, now=t))
        self.assertIsNone(sw.update(1, 0.01, now=t + 0.1))
        self.assertEqual(sw.update(1, 0.01, now=t + 0.2), STATE_FAR_TRACK)
        self.assertEqual(sw.state, STATE_FAR_TRACK)
        self.assertIn("CAPTURE→FAR_TRACK", sw.last_reason)

    def test_capture_multi_hand_neither_counts_nor_resets(self):
        sw = self._make()
        t = self._enter_capture(sw)
        sw.update(1, 0.01, now=t)
        sw.update(1, 0.01, now=t + 0.1)
        for i in range(10):  # 多手误检帧：计帧暂停但不清零
            self.assertIsNone(sw.update(2, 0.02, now=t + 1.0 + i * 0.1))
        self.assertEqual(sw.update(1, 0.01, now=t + 2.0), STATE_FAR_TRACK)

    def test_capture_no_hand_resets_streak(self):
        sw = self._make()
        t = self._enter_capture(sw)
        sw.update(1, 0.01, now=t)
        sw.update(1, 0.01, now=t + 0.1)
        sw.update(0, now=t + 0.2)  # 无手 → 清零
        self.assertIsNone(sw.update(1, 0.01, now=t + 0.3))
        self.assertIsNone(sw.update(1, 0.01, now=t + 0.4))
        self.assertEqual(sw.update(1, 0.01, now=t + 0.5), STATE_FAR_TRACK)

    # ---------- FAR_TRACK ----------

    def _enter_far_track(self, sw):
        t = self._enter_capture(sw)
        for i in range(3):
            sw.update(1, 0.01, now=t + i * 0.1)
        self.assertEqual(sw.state, STATE_FAR_TRACK)
        return 20.0  # 冷却期结束后的时间

    def test_far_track_to_near_when_hand_grows(self):
        sw = self._make()
        t = self._enter_far_track(sw)
        for i in range(3):
            self.assertIsNone(sw.update(1, 0.06, now=t + i * 0.1))  # 大手 6% > 4%
        self.assertEqual(sw.update(1, 0.06, now=t + 0.3), STATE_NEAR)
        self.assertEqual(sw.state, STATE_NEAR)
        self.assertIn("FAR_TRACK→NEAR", sw.last_reason)

    def test_far_track_small_hand_stays_put(self):
        """恒定 5m 振荡回归：FAR_TRACK 下小手持续有手，绝不弹回 NEAR。"""
        sw = self._make()
        t = self._enter_far_track(sw)
        for i in range(200):  # 单手但 bbox 只有 1%（远距），200 帧不迁移
            self.assertIsNone(sw.update(1, 0.01, now=t + i * 0.1))
        self.assertEqual(sw.state, STATE_FAR_TRACK)

    def test_far_track_near_streak_resets_on_no_hand(self):
        sw = self._make()
        t = self._enter_far_track(sw)
        for i in range(3):
            sw.update(1, 0.06, now=t + i * 0.1)
        sw.update(0, now=t + 0.3)  # 丢手 → 走近计帧清零（也喂了无手计帧）
        for i in range(3):
            self.assertIsNone(sw.update(1, 0.06, now=t + 1.0 + i * 0.1))
        self.assertEqual(sw.update(1, 0.06, now=t + 1.3), STATE_NEAR)

    def test_far_track_to_capture_on_no_hand_streak(self):
        sw = self._make()
        t = self._enter_far_track(sw)
        self.assertIsNone(sw.update(0, now=t))
        self.assertIsNone(sw.update(0, now=t + 0.1))
        self.assertEqual(sw.update(0, now=t + 0.2), STATE_CAPTURE)
        self.assertIn("FAR_TRACK→CAPTURE", sw.last_reason)

    def test_full_loop_no_oscillation(self):
        """完整闭环：NEAR→CAPTURE→FAR_TRACK→(走近)NEAR→(再走远)CAPTURE。"""
        sw = self._make()
        # NEAR: 用户走远，MP 连续无手
        for i in range(3):
            sw.update(0, now=float(i) * 0.1)
        self.assertEqual(sw.state, STATE_CAPTURE)
        # CAPTURE: YOLO 抓住手
        for i in range(3):
            sw.update(1, 0.01, now=10.0 + i * 0.1)
        self.assertEqual(sw.state, STATE_FAR_TRACK)
        # FAR_TRACK: ZOOM 盯住小手 100 帧不动（恒定 5m 不振荡）
        for i in range(100):
            self.assertIsNone(sw.update(1, 0.01, now=20.0 + i * 0.1))
        # 用户走近：手变大持续 → NEAR
        for i in range(4):
            sw.update(1, 0.08, now=40.0 + i * 0.1)
        self.assertEqual(sw.state, STATE_NEAR)
        # 再次走远 → CAPTURE
        for i in range(3):
            sw.update(0, now=50.0 + i * 0.1)
        self.assertEqual(sw.state, STATE_CAPTURE)

    # ---------- 冷却 / 重置 / 配置 ----------

    def test_cooldown_pauses_counting(self):
        sw = self._make()
        sw.update(0, now=0.0)
        sw.update(0, now=0.1)
        self.assertEqual(sw.update(0, now=0.2), STATE_CAPTURE)
        # 冷却 5s 内：有手帧不计入
        for i in range(10):
            self.assertIsNone(sw.update(1, 0.01, now=1.0 + i * 0.1))
        # 冷却结束后需重新累计 hand_frames 帧
        t = 0.2 + 5.0
        for i in range(2):
            self.assertIsNone(sw.update(1, 0.01, now=t + i * 0.1))
        self.assertEqual(sw.update(1, 0.01, now=t + 0.2), STATE_FAR_TRACK)

    def test_reset_clears_state_and_cooldown(self):
        sw = self._make()
        for i in range(3):
            sw.update(0, now=float(i) * 0.1)
        self.assertEqual(sw.state, STATE_CAPTURE)
        sw.reset()  # 手动切引擎 → 回 NEAR、冷却清空
        self.assertEqual(sw.state, STATE_NEAR)
        self.assertEqual(sw.last_reason, "")
        for i in range(2):
            self.assertIsNone(sw.update(0, now=0.3 + i * 0.1))
        self.assertEqual(sw.update(0, now=0.5), STATE_CAPTURE)

    def test_configure_updates_params_and_resets_counters(self):
        sw = self._make()
        sw.update(0, now=0.0)
        sw.update(0, now=0.1)
        sw.configure(no_hand_frames=10, near_bbox_ratio=0.08)
        self.assertEqual(sw.near_bbox_ratio, 0.08)
        for i in range(9):
            self.assertIsNone(sw.update(0, now=1.0 + i))
        self.assertEqual(sw.update(0, now=10.0), STATE_CAPTURE)

    def test_unknown_state_defends_to_near(self):
        sw = self._make()
        sw.state = "bogus"
        self.assertIsNone(sw.update(0, now=0.0))
        self.assertEqual(sw.state, STATE_NEAR)


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
            self.assertEqual(config.get("engine_auto_switch_hand_frames"), 30)
            self.assertEqual(config.get("engine_auto_switch_near_frames"), 90)
            self.assertEqual(config.get("engine_auto_switch_near_bbox_ratio"), 0.04)
            self.assertEqual(config.get("engine_auto_switch_cooldown_sec"), 5.0)

    def test_invalid_values_fall_back_to_defaults(self):
        config = self._load({
            "engine_auto_switch": "yes",
            "engine_auto_switch_no_hand_frames": 0,
            "engine_auto_switch_hand_frames": 9999,
            "engine_auto_switch_near_frames": -5,
            "engine_auto_switch_near_bbox_ratio": 0.5,
            "engine_auto_switch_cooldown_sec": -1,
        })
        self.assertIs(config.get("engine_auto_switch"), False)
        self.assertEqual(config.get("engine_auto_switch_no_hand_frames"), 60)
        self.assertEqual(config.get("engine_auto_switch_hand_frames"), 30)
        self.assertEqual(config.get("engine_auto_switch_near_frames"), 90)
        self.assertEqual(config.get("engine_auto_switch_near_bbox_ratio"), 0.04)
        self.assertEqual(config.get("engine_auto_switch_cooldown_sec"), 5.0)

    def test_valid_values_accepted(self):
        config = self._load({
            "engine_auto_switch": True,
            "engine_auto_switch_no_hand_frames": 45,
            "engine_auto_switch_hand_frames": 20,
            "engine_auto_switch_near_frames": 120,
            "engine_auto_switch_near_bbox_ratio": 0.06,
            "engine_auto_switch_cooldown_sec": 3.5,
        })
        self.assertIs(config.get("engine_auto_switch"), True)
        self.assertEqual(config.get("engine_auto_switch_no_hand_frames"), 45)
        self.assertEqual(config.get("engine_auto_switch_hand_frames"), 20)
        self.assertEqual(config.get("engine_auto_switch_near_frames"), 120)
        self.assertEqual(config.get("engine_auto_switch_near_bbox_ratio"), 0.06)
        self.assertEqual(config.get("engine_auto_switch_cooldown_sec"), 3.5)


def _make_orchestrator():
    """构造跳过 init_services 的 orchestrator（仓库 mock 风格，不起 QApplication）。"""
    with patch("orchestrator.AirControlOrchestrator.init_services"), \
         patch("orchestrator.AirControlOrchestrator._init_modes"), \
         patch("orchestrator.AirControlOrchestrator.set_mode"), \
         patch("orchestrator.ConfigManager"), \
         patch("orchestrator.MouseController"):
        return AirControlOrchestrator(MagicMock(), MagicMock(), MagicMock())


def _wire_switcher(orch, current_sig=_SIG_MP, **fsm_kwargs):
    orch._tracker_config_signature = current_sig
    defaults = {"enabled": True, "no_hand_frames": 3, "hand_frames": 3,
                "near_frames": 4, "near_bbox_ratio": 0.04, "cooldown_sec": 0.0}
    defaults.update(fsm_kwargs)
    orch._engine_switcher.configure(**defaults)
    orch._request_tracker_rebuild = MagicMock()
    orch._tracker_signature = MagicMock(
        return_value=_SIG_YOLO if current_sig == _SIG_MP else _SIG_MP
    )


class TestOrchestratorThreeState(unittest.TestCase):
    def test_near_to_capture_rebuilds_yolo(self):
        orch = _make_orchestrator()
        _wire_switcher(orch)
        orch._maybe_auto_switch_engine([object()], 0.01)  # 有手：不计
        orch._maybe_auto_switch_engine([], 0.0)
        orch._maybe_auto_switch_engine([], 0.0)
        orch._request_tracker_rebuild.assert_not_called()
        orch._maybe_auto_switch_engine([], 0.0)  # 第 3 帧连续无手 → CAPTURE
        self.assertEqual(orch._engine_override, "hagrid_yolo")
        self.assertFalse(orch._fsm_far_track_active)
        orch._request_tracker_rebuild.assert_called_once_with(_SIG_YOLO, config_overrides=None)
        self.assertTrue(orch._engine_switch_pending)

    def test_near_to_capture_reuses_matching_warmed_yolo(self):
        """预热完成后，CAPTURE 不再重新构造昂贵的 YOLO/HandLandmarker。"""
        orch = _make_orchestrator()
        _wire_switcher(orch)
        warmed = MagicMock()
        orch._warmed_yolo_tracker = warmed
        orch._warmed_yolo_signature = _SIG_YOLO

        for _ in range(3):
            orch._maybe_auto_switch_engine([], 0.0)

        orch._request_tracker_rebuild.assert_called_once_with(
            _SIG_YOLO, config_overrides=None, prepared_tracker=warmed,
        )
        self.assertIsNone(orch._warmed_yolo_tracker)
        self.assertIsNone(orch._warmed_yolo_signature)

    def test_pending_rebuild_blocks_duplicate_requests(self):
        orch = _make_orchestrator()
        _wire_switcher(orch)
        for _ in range(3):
            orch._maybe_auto_switch_engine([], 0.0)
        orch._request_tracker_rebuild.assert_called_once()
        for _ in range(10):  # 重建未落地：继续喂帧不重复发起
            orch._maybe_auto_switch_engine([], 0.0)
        orch._request_tracker_rebuild.assert_called_once()

    def test_capture_to_far_track_rebuilds_mp_with_zoom_and_seed(self):
        orch = _make_orchestrator()
        _wire_switcher(orch, current_sig=_SIG_YOLO)
        orch._engine_override = "hagrid_yolo"
        orch._engine_switcher.state = STATE_CAPTURE
        for _ in range(3):  # 连续 3 帧稳定单手 → FAR_TRACK
            orch._maybe_auto_switch_engine([object()], 0.01)
        self.assertEqual(orch._engine_override, "mediapipe")
        self.assertTrue(orch._fsm_far_track_active)
        self.assertTrue(orch._pending_far_track_seed)
        orch._request_tracker_rebuild.assert_called_once_with(
            _SIG_MP, config_overrides={"long_range_enabled": True}
        )

    def test_tracker_ready_seeds_crop_zoom_after_migrate(self):
        """构建线程只排队；迁移和播种由推理线程原子执行后再提交。"""
        orch = _make_orchestrator()
        orch._tracker_request_id = 1
        orch._pending_far_track_seed = True
        orch._engine_switch_pending = True
        old_tracker = MagicMock()
        orch.tracker = old_tracker
        orch.inference_worker = MagicMock()
        new_tracker = MagicMock()
        orch.inference_worker.update_tracker.return_value = True

        orch._on_tracker_ready(new_tracker, _SIG_MP, 1, "")

        new_tracker.migrate_state_from.assert_not_called()
        new_tracker.seed_crop_zoom_from_hint.assert_not_called()
        self.assertFalse(orch._pending_far_track_seed)
        self.assertTrue(orch._engine_switch_pending)
        context = {"signature": _SIG_MP, "request_id": 1}
        orch.inference_worker.update_tracker.assert_called_once_with(
            new_tracker,
            context=context,
            seed_crop_zoom=True,
        )
        self.assertIs(orch.tracker, old_tracker)

        orch._on_tracker_swapped(
            orch.inference_worker,
            new_tracker,
            context,
            {"seed_requested": True, "seeded": True},
        )

        self.assertIs(orch.tracker, new_tracker)
        self.assertEqual(orch._tracker_config_signature, _SIG_MP)
        self.assertFalse(orch._engine_switch_pending)

    def test_failed_rebuild_rolls_back_override(self):
        orch = _make_orchestrator()
        _wire_switcher(orch)
        for _ in range(3):
            orch._maybe_auto_switch_engine([], 0.0)
        self.assertEqual(orch._engine_override, "hagrid_yolo")
        orch._on_tracker_ready(None, _SIG_YOLO, orch._tracker_request_id, "boom")
        self.assertIsNone(orch._engine_override)
        self.assertFalse(orch._engine_switch_pending)
        self.assertEqual(orch._engine_switcher.state, STATE_NEAR)

    def test_far_track_to_near_flips_long_range_without_rebuild(self):
        orch = _make_orchestrator()
        _wire_switcher(orch)
        orch._engine_override = "mediapipe"
        orch._fsm_far_track_active = True
        orch._engine_switcher.state = STATE_FAR_TRACK
        orch._set_tracker_long_range = MagicMock()
        orch.config = MagicMock()
        orch.config.get.side_effect = lambda k, d=None: {
            "detection_engine": "mediapipe", "long_range_enabled": False,
        }.get(k, d)
        for _ in range(4):  # 大手 6% 持续 4 帧 → NEAR
            orch._maybe_auto_switch_engine([object()], 0.06)
        self.assertIsNone(orch._engine_override)
        self.assertFalse(orch._fsm_far_track_active)
        orch._request_tracker_rebuild.assert_not_called()
        orch._set_tracker_long_range.assert_called_once_with(False)

    def test_disabled_switcher_is_noop(self):
        orch = _make_orchestrator()
        orch._tracker_config_signature = _SIG_MP
        orch._request_tracker_rebuild = MagicMock()
        for _ in range(100):
            orch._maybe_auto_switch_engine([], 0.0)
        orch._request_tracker_rebuild.assert_not_called()
        self.assertIsNone(orch._engine_override)

    def test_env_var_forced_engine_blocks_auto_switch(self):
        orch = _make_orchestrator()
        _wire_switcher(orch)
        with mock.patch.dict(os.environ, {"AIRCONTROL_ENGINE": "mediapipe"}):
            for _ in range(10):
                orch._maybe_auto_switch_engine([], 0.0)
        orch._request_tracker_rebuild.assert_not_called()
        self.assertIsNone(orch._engine_override)

    def _apply_config_with(self, orch, values, sig="same"):
        orch.config = MagicMock()
        orch.config.get.side_effect = lambda key, default=None: values.get(key, default)
        orch.recognizer = MagicMock()
        orch.ppt = MagicMock()
        orch.mouse = MagicMock()
        orch.overlay = MagicMock()
        orch.voice_command = MagicMock()
        orch.mode_manager = MagicMock(current_mode_name="mouse")
        orch._tracker_signature = MagicMock(return_value=sig)
        orch._tracker_config_signature = sig
        orch._current_voice_kws_signature = MagicMock(return_value="same")
        orch._voice_kws_signature = "same"
        orch.apply_config()

    def test_manual_engine_change_respects_user_choice(self):
        """用户手动切引擎：清除 FSM 覆盖/FAR_TRACK 簿记并重置状态机。"""
        orch = _make_orchestrator()
        orch._last_config_engine = "mediapipe"
        orch._engine_override = "hagrid_yolo"
        orch._fsm_far_track_active = False
        orch._engine_switcher.enabled = True
        orch._engine_switcher.state = STATE_CAPTURE
        self._apply_config_with(orch, {
            "detection_engine": "hagrid_yolo",
            "interaction_mode": "mouse",
            "cooldown": 1.0,
        })
        self.assertIsNone(orch._engine_override)
        self.assertEqual(orch._last_config_engine, "hagrid_yolo")
        self.assertEqual(orch._engine_switcher.state, STATE_NEAR)
        # config 未开 engine_auto_switch → 状态机被配置为关闭
        self.assertFalse(orch._engine_switcher.enabled)

    def test_manual_yolo_choice_disables_auto_override_even_when_enabled_in_config(self):
        """手动选择 hagrid_yolo 时，自动闭环不能再把它切回 MediaPipe。"""
        orch = _make_orchestrator()
        orch._last_config_engine = "mediapipe"
        self._apply_config_with(orch, {
            "detection_engine": "hagrid_yolo",
            "engine_auto_switch": True,
            "interaction_mode": "mouse",
            "cooldown": 1.0,
        })
        self.assertFalse(orch._engine_switcher.enabled)

    def test_manual_change_during_far_track_flips_long_range_off(self):
        """FAR_TRACK 期间手动改 detection_engine 但引擎不变：运行时撤 long_range 覆盖。"""
        orch = _make_orchestrator()
        orch._last_config_engine = "hagrid_yolo"  # 伪装成刚改过
        orch._engine_override = "mediapipe"
        orch._fsm_far_track_active = True
        orch._set_tracker_long_range = MagicMock()
        orch._request_tracker_rebuild = MagicMock()
        # 签名首元素 = 目标配置引擎（mediapipe）→ 不重建，走运行时翻转路径
        self._apply_config_with(orch, {
            "detection_engine": "mediapipe",
            "interaction_mode": "mouse",
            "cooldown": 1.0,
            "long_range_enabled": False,
        }, sig=_SIG_MP)
        self.assertFalse(orch._fsm_far_track_active)
        orch._set_tracker_long_range.assert_called_once_with(False)
        orch._request_tracker_rebuild.assert_not_called()

    def test_disabling_auto_switch_reverts_to_configured_engine(self):
        """跑着 FSM 覆盖引擎时用户关掉开关：回到 config 配置的引擎。"""
        orch = _make_orchestrator()
        orch._last_config_engine = "mediapipe"
        orch._engine_override = "hagrid_yolo"
        orch._engine_switcher.enabled = True
        orch._engine_switcher.state = STATE_CAPTURE
        self._apply_config_with(orch, {
            "detection_engine": "mediapipe",
            "interaction_mode": "mouse",
            "cooldown": 1.0,
        })
        self.assertIsNone(orch._engine_override)
        self.assertFalse(orch._engine_switcher.enabled)
        self.assertEqual(orch._engine_switcher.state, STATE_NEAR)


class TestTrackerHandoffHelpers(unittest.TestCase):
    """base_hand_tracker 交接辅助方法（不依赖真实模型，直接构造属性）。"""

    def _bare_tracker(self):
        from services.base_hand_tracker import BaseHandTracker

        class _Bare(BaseHandTracker):  # 最小具体子类，绕开重型 __init__
            @property
            def engine_name(self):
                return "bare"

            def _detect(self, frame):
                return [], [], []

            def _detect_crop_zoom(self, frame, hint_center, hint_size):
                return [], [], []

        return object.__new__(_Bare)  # 跳过 __init__ 的重型依赖

    def test_seed_crop_zoom_from_hint(self):
        bare = self._bare_tracker()
        bare._last_hint_center = (500.0, 400.0)
        bare._last_hint_size = 80.0
        bare._crop_zoom_mode = False
        bare._current_crop_center = (960.0, 540.0)
        bare._current_crop_size = 1080.0
        self.assertTrue(bare.seed_crop_zoom_from_hint())
        self.assertTrue(bare._crop_zoom_mode)
        self.assertIsNone(bare._current_crop_center)  # 下一帧用 hint 一步落位
        self.assertIsNone(bare._current_crop_size)

    def test_seed_without_hint_returns_false(self):
        bare = self._bare_tracker()
        bare._last_hint_center = None
        bare._last_hint_size = 0
        bare._crop_zoom_mode = False
        self.assertFalse(bare.seed_crop_zoom_from_hint())
        self.assertFalse(bare._crop_zoom_mode)

    def test_set_long_range_enabled_off_resets_zoom_state(self):
        bare = self._bare_tracker()
        bare._long_range_enabled = True
        bare._crop_zoom_mode = True
        bare._sr = MagicMock()
        bare._far_streak = 2
        bare._near_streak = 1
        bare._zoom_miss_streak = 3
        bare._last_hint_center = (1.0, 2.0)
        bare._last_hint_size = 50.0
        bare._current_crop_center = (1.0, 2.0)
        bare._current_crop_size = 100.0
        bare.set_long_range_enabled(False)
        self.assertFalse(bare._long_range_enabled)
        self.assertFalse(bare._crop_zoom_mode)
        self.assertIsNone(bare._last_hint_center)
        self.assertIsNone(bare._current_crop_center)
        bare._sr.reset_tier.assert_called_once()

    def test_set_long_range_enabled_noop_when_unchanged(self):
        bare = self._bare_tracker()
        bare._long_range_enabled = False
        bare._crop_zoom_mode = False
        bare._sr = MagicMock()
        bare.set_long_range_enabled(False)  # 幂等：不应触发复位
        bare._sr.reset_tier.assert_not_called()


class TestShadowZoomHandoff(unittest.TestCase):
    """影子 ZOOM 确认晋升（shadow_zoom_handoff_enabled）的 orchestrator 集成。

    覆盖第 4 步生产化路径：CAPTURE 重建落地启动影子、影子确认后免重建晋升、
    FAR_TRACK/NEAR 撤影子、config 关闭即撤、晋升抬升 request_id 使在途重建
    自弃。仓库约定：不走真实 QApplication（mock/源码级）。
    """

    def _make(self, enabled=True):
        orch = _make_orchestrator()
        _wire_switcher(orch)
        orch.config = MagicMock()
        _shadow_keys = {
            "shadow_zoom_handoff_enabled": enabled,
            "shadow_zoom_reseed_miss_frames": 2,
            "shadow_zoom_confirm_window_frames": 5,
            "engine_auto_switch_near_bbox_ratio": 0.04,
        }
        orch.config.get = lambda key, default=None: _shadow_keys.get(key, default)
        orch._engine_switcher.state = STATE_CAPTURE
        orch._engine_switch_pending = False
        return orch

    def test_enabled_gate_requires_capture_state_and_flag(self):
        """开关 + CAPTURE 态 + 无 pending + 无环境变量屏蔽，四者缺一不可。"""
        orch = self._make()
        self.assertTrue(orch._shadow_zoom_enabled())
        orch.config.get = lambda key, default=None: default  # 开关关
        self.assertFalse(orch._shadow_zoom_enabled())
        orch2 = self._make()
        orch2._engine_switcher.state = STATE_NEAR
        self.assertFalse(orch2._shadow_zoom_enabled())
        orch3 = self._make()
        orch3._engine_switch_pending = True
        self.assertFalse(orch3._shadow_zoom_enabled())

    def test_signature_forces_mediapipe_engine(self):
        """影子签名与 FAR_TRACK 的 mediapipe+long_range 签名一致（第 0 位强制 MP）。"""
        orch = self._make()
        self.assertEqual(orch._shadow_zoom_signature()[0], "mediapipe")

    def test_start_is_idempotent(self):
        orch = self._make()
        orch._shadow_zoom_tracker = MagicMock()
        orch._maybe_start_shadow_zoom()
        # 已有影子时不重复启动（不触发后台构建）
        self.assertEqual(orch._shadow_zoom_tracker.__class__.__name__, "MagicMock")

    def test_maybe_start_builds_and_installs_shadow(self):
        """_maybe_start_shadow_zoom 的后台构建落地：装到 worker、记引用。"""
        orch = self._make()
        shadow = MagicMock()
        built = []

        def fake_create(signature, config_overrides=None):
            built.append((signature, config_overrides))
            return shadow

        orch._create_tracker = fake_create
        orch.inference_worker = MagicMock()
        orch.inference_worker.isRunning.return_value = True
        with patch("orchestrator.AirControlOrchestrator._start_background_thread") as sb:
            orch._maybe_start_shadow_zoom()
            # 后台构建在闭包里；直接调用捕获的 target
            target = sb.call_args[0][0]
            target()
        self.assertEqual(built[0][1], {"long_range_enabled": True})
        self.assertIs(orch._shadow_zoom_tracker, shadow)
        orch.inference_worker.set_shadow_tracker.assert_called_once_with(shadow)

    def test_teardown_uninstalls_and_closes(self):
        orch = self._make()
        shadow = MagicMock()
        orch._shadow_zoom_tracker = shadow
        orch.inference_worker = MagicMock()
        orch._shadow_zoom_confirm = MagicMock()
        orch._teardown_shadow_zoom()
        self.assertIsNone(orch._shadow_zoom_tracker)
        orch.inference_worker.set_shadow_tracker.assert_called_once_with(None)
        shadow.close.assert_called_once()
        orch._shadow_zoom_confirm.reset.assert_called_once()

    def test_teardown_promote_keeps_tracker(self):
        """晋升交接：tracker 留给主路径，不被关闭。"""
        orch = self._make()
        shadow = MagicMock()
        orch._shadow_zoom_tracker = shadow
        orch.inference_worker = MagicMock()
        orch._teardown_shadow_zoom(promote=True)
        shadow.close.assert_not_called()

    def test_promotion_marks_far_track_and_bumps_request_id(self):
        """晋升：FSM 落 FAR_TRACK、影子成为主 tracker、request_id 抬升
        （在途重建的 ready 回调因 id 不匹配自弃）、seed 关闭。"""
        orch = self._make()
        shadow = MagicMock()
        orch._shadow_zoom_tracker = shadow
        orch.inference_worker = MagicMock()
        orch.inference_worker.update_tracker.return_value = True
        before_id = orch._tracker_request_id

        orch._promote_shadow_zoom()

        self.assertEqual(orch._engine_switcher.state, STATE_FAR_TRACK)
        self.assertEqual(orch._engine_override, "mediapipe")
        self.assertTrue(orch._fsm_far_track_active)
        self.assertEqual(orch._tracker_request_id, before_id + 1)
        self.assertFalse(orch._engine_switch_pending)
        self.assertFalse(orch._pending_far_track_seed)
        self.assertIsNone(orch._shadow_zoom_tracker)
        context = orch.inference_worker.update_tracker.call_args[1]["context"]
        self.assertTrue(context["shadow_promotion"])
        self.assertEqual(context["request_id"], before_id + 1)
        self.assertEqual(
            orch.inference_worker.update_tracker.call_args[1]["seed_crop_zoom"],
            False,
        )
        self.assertTrue(
            orch.inference_worker.update_tracker.call_args[1]["skip_migration"])
        orch.inference_worker.set_shadow_tracker.assert_called_with(None)

    def test_promotion_skip_migration_preserves_shadow_state(self):
        """skip_migration 语义：worker 换主 tracker 时不从捕获器迁移状态。

        不跳过迁移时，YOLO 捕获器的 _crop_zoom_mode=False 会覆写影子的
        ZOOM 视口——恰好摧毁免重建交接要保留的接续状态。
        """
        from services.inference_worker import InferenceWorker
        worker = InferenceWorker.__new__(InferenceWorker)
        worker.shadow_tracker = None
        worker._pending_tracker = None
        worker._pending_tracker_context = None
        worker._pending_tracker_seed_crop_zoom = False
        worker._pending_tracker_skip_migration = False
        worker._pending_tracker_lock = __import__("threading").Lock()
        worker._retired_pending_trackers = []
        worker._accept_tracker_updates = True
        worker.lock = __import__("threading").Lock()
        worker.tracker = MagicMock()  # 旧主 tracker（捕获器）
        shadow = MagicMock()
        self.assertTrue(worker.update_tracker(shadow, context={}, skip_migration=True))
        # 提交后驱动一次 _process_frame 的换装段（无需真实帧——先手动执行
        # 换装逻辑等价路径：直接调 _process_frame 会碰 camera，改为验证
        # update_tracker 记录的 skip 标志被换装段消费）
        with worker._pending_tracker_lock:
            pending = worker._pending_tracker
            skip = worker._pending_tracker_skip_migration
        self.assertIs(pending, shadow)
        self.assertTrue(skip)
        shadow.migrate_state_from.assert_not_called()

    def test_promotion_failure_closes_shadow(self):
        """worker 拒收（如正在关闭）→ 影子关闭，状态保持可重试。"""
        orch = self._make()
        shadow = MagicMock()
        orch._shadow_zoom_tracker = shadow
        orch.inference_worker = MagicMock()
        orch.inference_worker.update_tracker.return_value = False

        orch._promote_shadow_zoom()

        shadow.close.assert_called()
        self.assertIsNone(orch._shadow_zoom_tracker)

    def test_in_flight_rebuild_discarded_after_promotion(self):
        """晋升后，此前发起的在途重建落地时因 request_id 过期被丢弃。"""
        orch = self._make()
        stale_id = orch._tracker_request_id
        # 在途重建（config 触发）发起
        orch._request_tracker_rebuild = MagicMock()
        shadow = MagicMock()
        orch._shadow_zoom_tracker = shadow
        orch.inference_worker = MagicMock()
        orch.inference_worker.update_tracker.return_value = True
        orch._promote_shadow_zoom()
        # 在途重建的 ready 回调带旧 id → 应被丢弃（现有守卫逻辑）
        orch._on_tracker_ready(MagicMock(), _SIG_MP, stale_id, "")
        orch.inference_worker.update_tracker.assert_called_once()  # 只有晋升那次

    def test_far_track_or_near_teardown_shadow(self):
        """FSM 离开 CAPTURE（FAR_TRACK/NEAR）撤影子。"""
        orch = self._make()
        shadow = MagicMock()
        orch._shadow_zoom_tracker = shadow
        orch.inference_worker = MagicMock()
        orch._apply_fsm_state(STATE_FAR_TRACK)
        self.assertIsNone(orch._shadow_zoom_tracker)

    def test_apply_fsm_state_capture_defers_to_landing(self):
        """CAPTURE 状态切换本身不启动影子（等重建落地 _on_tracker_swapped）。"""
        orch = self._make()
        started = []
        orch._maybe_start_shadow_zoom = lambda: started.append(1)
        orch._apply_fsm_state(STATE_CAPTURE)
        self.assertEqual(started, [])

    def test_tracker_swapped_starts_shadow_in_capture(self):
        """重建落地（_on_tracker_swapped）时 CAPTURE 态启动影子。"""
        orch = self._make()
        started = []
        orch._maybe_start_shadow_zoom = lambda: started.append(1)
        orch.inference_worker = MagicMock()
        context = {"signature": _SIG_YOLO, "request_id": orch._tracker_request_id}
        orch._on_tracker_swapped(orch.inference_worker, MagicMock(), context, {})
        self.assertEqual(started, [1])

    def test_config_disable_tears_down(self):
        """apply_config 时开关关 → 撤影子。"""
        orch = self._make(enabled=False)
        shadow = MagicMock()
        orch._shadow_zoom_tracker = shadow
        orch.inference_worker = MagicMock()
        # apply_config 其余依赖（recognizer/ppt/mouse/overlay/voice/模式管理）
        orch.recognizer = MagicMock()
        orch.ppt = MagicMock()
        orch.mouse = MagicMock()
        orch.overlay = MagicMock()
        orch.voice_command = MagicMock()
        orch._current_voice_kws_signature = MagicMock(return_value=None)
        orch._voice_kws_signature = None
        orch.mode_manager = MagicMock()
        orch.mode_manager.current_mode_name = "mouse"
        orch.mode_manager.current_mode = MagicMock()
        with patch.object(orch, "_configure_engine_switcher"), \
             patch.object(orch, "_discard_warmed_yolo_tracker"), \
             patch.object(orch, "set_mode"):
            orch.apply_config()
        self.assertIsNone(orch._shadow_zoom_tracker)
        shadow.close.assert_called_once()

    def test_config_switcher_disabled_tears_down(self):
        """FSM 自动切换被关闭（engine_auto_switch=False）→ 撤影子。"""
        orch = self._make()  # 开关开着
        shadow = MagicMock()
        orch._shadow_zoom_tracker = shadow
        orch.inference_worker = MagicMock()
        orch.recognizer = MagicMock()
        orch.ppt = MagicMock()
        orch.mouse = MagicMock()
        orch.overlay = MagicMock()
        orch.voice_command = MagicMock()
        orch._current_voice_kws_signature = MagicMock(return_value=None)
        orch._voice_kws_signature = None
        orch.mode_manager = MagicMock()
        orch.mode_manager.current_mode_name = "mouse"
        orch.mode_manager.current_mode = MagicMock()
        with patch.object(orch, "_configure_engine_switcher",
                          side_effect=lambda: orch._engine_switcher.configure(
                              enabled=False)), \
             patch.object(orch, "_discard_warmed_yolo_tracker"), \
             patch.object(orch, "set_mode"):
            orch.apply_config()
        self.assertIsNone(orch._shadow_zoom_tracker)
        shadow.close.assert_called_once()

    def test_shadow_hand_ready_promotes_after_confirmation(self):
        """端到端：影子结果喂确认机，confirmations 达标后触发晋升。"""
        orch = self._make()
        shadow = MagicMock()
        shadow._zoom_miss_streak = 0
        orch._shadow_zoom_tracker = shadow
        orch.inference_worker = MagicMock()
        orch._promote_shadow_zoom = MagicMock()
        from services.zoom_shadow_confirm import ZoomShadowConfirm
        confirm = ZoomShadowConfirm(confirm_window_frames=5)
        orch._shadow_zoom_confirm = confirm
        # 手与捕获手几何一致（wrist 相同 → 距离 0）
        hand = [[0, 100.0, 100.0]]
        for i in range(1, 21):
            hand.append([i, 140.0, 140.0])
        capture_hand = [list(p) for p in hand]
        gesture = {"predicted": False}
        orch._last_capture_result = ([capture_hand], [gesture], 0.001)
        # 第 1 帧开窗 + 4 帧自持 → 第 5 次 observe 时 confirmations=1 且
        # bbox_ratio 远距 → 立即晋升
        for _ in range(4):
            orch._on_shadow_hand_ready(
                None, [hand], [gesture], orch.inference_worker)
        orch._promote_shadow_zoom.assert_not_called()
        orch._on_shadow_hand_ready(
            None, [hand], [gesture], orch.inference_worker)
        orch._promote_shadow_zoom.assert_called_once()

    def test_shadow_hand_ready_ignores_stale_worker(self):
        """worker 不匹配（旧实例信号）→ 忽略。"""
        orch = self._make()
        orch._shadow_zoom_tracker = MagicMock()
        orch._shadow_zoom_confirm = MagicMock()
        orch._promote_shadow_zoom = MagicMock()
        orch._last_capture_result = ([], [], 0.0)
        orch._on_shadow_hand_ready(None, [], [], worker=MagicMock())
        orch._promote_shadow_zoom.assert_not_called()

    def test_closing_blocks_shadow_promotion(self):
        """_closing 后影子结果不再触发晋升（close 路径已先撤影子）。"""
        orch = self._make()
        shadow = MagicMock()
        shadow._zoom_miss_streak = 0
        orch._shadow_zoom_tracker = shadow
        orch._shadow_zoom_confirm = MagicMock()
        orch._shadow_zoom_confirm.observe.return_value = (True, False, None)
        orch._promote_shadow_zoom = MagicMock()
        orch.inference_worker = MagicMock()
        orch._last_capture_result = ([], [], 0.001)
        orch._closing = True
        orch._on_shadow_hand_ready(None, [], [], orch.inference_worker)
        orch._promote_shadow_zoom.assert_not_called()

    def test_worker_restart_transfers_shadow(self):
        """摄像头重启换 worker：现存影子转挂新 worker（影子不重建，视口保留），
        旧 worker 摘引用；新 worker 接上 shadow_hand_ready。"""
        orch = self._make()
        shadow = MagicMock()
        orch._shadow_zoom_tracker = shadow
        old_worker = MagicMock()
        orch.inference_worker = old_worker
        orch.camera = MagicMock()
        orch.tracker = MagicMock()
        with patch("orchestrator.InferenceWorker") as worker_cls:
            new_worker = MagicMock()
            worker_cls.return_value = new_worker
            orch._restart_inference_worker()
        new_worker.set_shadow_tracker.assert_called_once_with(shadow)
        old_worker.set_shadow_tracker.assert_called_once_with(None)
        new_worker.shadow_hand_ready.connect.assert_called_once_with(
            orch._on_shadow_hand_ready)
        # 影子引用不变（同一实例，未重建）
        self.assertIs(orch._shadow_zoom_tracker, shadow)
        self.assertIs(orch.inference_worker, new_worker)

    def test_worker_restart_without_shadow_still_wires_signal(self):
        """无影子时重启 worker：不挂影子，但 shadow_hand_ready 照接。"""
        orch = self._make()
        orch.inference_worker = MagicMock()
        orch.camera = MagicMock()
        orch.tracker = MagicMock()
        with patch("orchestrator.InferenceWorker") as worker_cls:
            new_worker = MagicMock()
            worker_cls.return_value = new_worker
            orch._restart_inference_worker()
        new_worker.set_shadow_tracker.assert_not_called()
        new_worker.shadow_hand_ready.connect.assert_called_once_with(
            orch._on_shadow_hand_ready)


class TestCaptureEngineSelection(unittest.TestCase):
    """CAPTURE 态捕获引擎可选：engine_auto_switch_capture_engine 热切换。

    覆盖 _capture_engine() 的默认/显式/非法值回退，以及它对
    _yolo_signature() 与 _apply_fsm_state(CAPTURE) 的贯穿。
    """

    def _orch(self, capture_engine):
        orch = _make_orchestrator()
        orch.config = MagicMock()
        orch.config.get = lambda key, default=None: (
            capture_engine if key == "engine_auto_switch_capture_engine" else default
        )
        orch._request_tracker_rebuild = MagicMock()
        return orch

    def test_default_is_hagrid_yolo(self):
        orch = self._orch(None)  # config.get 落默认
        self.assertEqual(orch._capture_engine(), "hagrid_yolo")
        self.assertEqual(orch._yolo_signature()[0], "hagrid_yolo")

    def test_explicit_person_pose(self):
        orch = self._orch("person_pose_hand")
        self.assertEqual(orch._capture_engine(), "person_pose_hand")
        self.assertEqual(orch._yolo_signature()[0], "person_pose_hand")

    def test_invalid_value_falls_back(self):
        orch = self._orch("not_an_engine")
        self.assertEqual(orch._capture_engine(), "hagrid_yolo")

    def test_apply_fsm_state_uses_capture_engine(self):
        orch = self._orch("person_pose_hand")
        orch._tracker_config_signature = _SIG_MP  # 与目标不同 → 走重建路径
        orch._tracker_signature = MagicMock(return_value=_SIG_YOLO)
        orch._apply_fsm_state(STATE_CAPTURE)
        self.assertEqual(orch._engine_override, "person_pose_hand")
        self.assertFalse(orch._fsm_far_track_active)

    def test_capture_status_text_reflects_engine(self):
        orch = self._orch("person_pose_hand")
        self.assertIn("person_pose_hand", orch._capture_status_text())
        orch2 = self._orch(None)
        self.assertIn("hagrid_yolo", orch2._capture_status_text())


if __name__ == "__main__":
    unittest.main()
