"""M3/M5/M6 接线测试：LLM 智能指令会话、云端健康回调、TTS 播报防自触发。

所有云端/音频依赖均 mock，不发起网络、不启动麦克风。
"""
import os
import sys
import time
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'app'))

from orchestrator import AirControlOrchestrator  # noqa: E402
from services.voice_actions import MODE_KEYWORDS, VOICE_KEYWORD_TO_ACTION  # noqa: E402
from services.voice_command import VoiceCommandService  # noqa: E402


class _Cfg:
    """最小 config 桩：get() 返回预置值，缺省回落 default。"""

    def __init__(self, **kv):
        self._kv = kv

    def get(self, key, default=None):
        value = self._kv.get(key)
        return value if value is not None else default


def _make_voice_command():
    svc = VoiceCommandService(_Cfg(dictation_model_dir="models/sense-voice"))
    svc.dictation_service = MagicMock()
    svc.dictation_service.is_available.return_value = True
    svc.dictation_service.dictate.return_value = "把上一页再放一遍"
    svc._running = True
    return svc


def _silence_frame():
    return bytes(1600 * 2)  # 100ms 全零（int16）


def _speech_frame():
    t = np.linspace(0, 0.1, 1600, endpoint=False)
    wave = (np.sin(2 * np.pi * 440 * t) * 12000).astype(np.int16)
    return wave.tobytes()  # RMS≈0.26，远高于语音阈值 0.012


def _make_orchestrator():
    with patch('orchestrator.AirControlOrchestrator.init_services'), \
         patch('orchestrator.AirControlOrchestrator._init_modes'), \
         patch('orchestrator.AirControlOrchestrator.set_mode'), \
         patch('orchestrator.ConfigManager'), \
         patch('orchestrator.MouseController'):
        return AirControlOrchestrator(MagicMock(), MagicMock(), MagicMock())


class TestLLMCommandSession(unittest.TestCase):
    def setUp(self):
        self.svc = _make_voice_command()
        self.status = []
        self.texts = []
        self.actions = []
        self.svc.action_callback = self.actions.append

    def _start(self):
        return self.svc.start_llm_command(
            on_text=self.texts.append,
            on_status=lambda phase, payload: self.status.append(phase),
        )

    def _sync_asr(self):
        """把 ASR 线程替换为同步执行，便于断言回调。"""
        def fake_start(target, *args, name="X"):
            target(*args)
            return True
        self.svc._start_asr_thread = fake_start

    def test_start_emits_started(self):
        self.assertTrue(self._start())
        self.assertEqual(self.status, ["started"])
        self.assertIsNotNone(self.svc._llm_session)

    def test_duplicate_start_rejected(self):
        self.assertTrue(self._start())
        self.assertFalse(self._start())

    def test_dictation_mode_excludes(self):
        self.svc._dictation_mode = True
        self.assertFalse(self._start())
        self.assertIsNone(self.svc._llm_session)

    def test_unavailable_dictation_rejected(self):
        self.svc.dictation_service.is_available.return_value = False
        self.assertFalse(self._start())
        self.assertEqual(self.status, ["failed"])

    def test_keywords_ignored_in_session(self):
        self._start()
        self.svc._handle_keyword("下一页")
        self.assertEqual(self.actions, [])

    def test_vad_silence_finishes_and_transcribes(self):
        self._start()
        session = self.svc._llm_session
        session["has_speech"] = True
        session["silence_start"] = time.time() - 1.3  # 已静音超阈值
        session["buffer"].extend(_speech_frame() * 10)  # ≥0.5s 音频
        self._sync_asr()
        self.svc._update_llm_vad(_silence_frame())
        self.assertIsNone(self.svc._llm_session)
        self.assertEqual(self.texts, ["把上一页再放一遍"])

    def test_wait_speech_timeout_cancels(self):
        self._start()
        session = self.svc._llm_session
        session["start"] = time.time() - 6.0  # 一直没说话
        self._sync_asr()
        self.svc._update_llm_vad(_silence_frame())
        self.assertIsNone(self.svc._llm_session)
        self.assertEqual(self.texts, [""])  # cancelled → 不转写

    def test_speech_frame_resets_silence(self):
        """能量 VAD 退回分支：语音帧清零静音计时（固定 energy 路径，
        模型存在时 _start 会加载 Silero，440Hz 合成音不触发神经网络判定）。"""
        self._start()
        session = self.svc._llm_session
        session["has_speech"] = True
        session["silence_start"] = time.time() - 1.0
        saved = self.svc._silero_vad
        self.svc._silero_vad = None
        try:
            self.svc._update_llm_vad(_speech_frame())
        finally:
            self.svc._silero_vad = saved
        self.assertIsNone(session["silence_start"])
        self.assertIsNotNone(self.svc._llm_session)

    def test_feedback_drops_audio_flag(self):
        feedback = MagicMock()
        feedback.is_speaking.return_value = True
        self.svc.set_feedback(feedback)
        self.assertTrue(self.svc._should_drop_audio())
        feedback.is_speaking.return_value = False
        self.assertFalse(self.svc._should_drop_audio())

    def test_cancel(self):
        self._start()
        self.svc.cancel_llm_command()
        self.assertIsNone(self.svc._llm_session)


class TestOrchestratorLLMWiring(unittest.TestCase):
    def setUp(self):
        self.orch = _make_orchestrator()
        self.status = []
        self.orch.voice_status_updated.connect(self.status.append)
        self.classroom = MagicMock()
        self.orch.classroom_ai = self.classroom
        self.feedback = MagicMock()
        self.orch.voice_feedback = self.feedback
        self.voice_command = MagicMock()
        self.voice_command.is_running.return_value = True
        self.orch.voice_command = self.voice_command
        # execute_action 的委托目标（信号直连会触发完整链路）
        self.orch.llm_client = MagicMock()
        self.orch.ppt = MagicMock()
        self.orch.mouse = MagicMock()

    # ------------------------------------------------------------------
    # 触发与分发
    # ------------------------------------------------------------------

    def test_execute_action_dispatches_llm_command(self):
        with patch.object(self.orch, '_start_llm_command') as start:
            self.orch.execute_action("llm_command")
            start.assert_called_once_with()

    def test_unconfigured_cloud_speaks_hint(self):
        self.classroom.is_available.return_value = False
        self.orch._start_llm_command()
        self.feedback.say.assert_called_once_with("云端未配置")
        self.assertTrue(any("云端未配置" in s for s in self.status))
        self.voice_command.start_llm_command.assert_not_called()

    def test_degraded_cloud_blocks_llm_command(self):
        """auto_fallback：degraded 时不开录音会话，当场提示改用固定指令。"""
        self.classroom.is_available.return_value = True
        self.orch.cloud_status = "degraded"
        self.orch._start_llm_command()
        self.voice_command.start_llm_command.assert_not_called()
        self.feedback.say.assert_called_once_with("网络不佳，请用固定指令")
        self.assertTrue(any("网络不佳" in s for s in self.status))

    def test_recovered_cloud_allows_llm_command(self):
        """恢复 online 后不再拦截。"""
        self.classroom.is_available.return_value = True
        self.orch.cloud_status = "online"
        self.voice_command.start_llm_command.return_value = True
        self.orch._start_llm_command()
        self.assertTrue(self.voice_command.start_llm_command.called)

    def test_start_passes_callbacks(self):
        self.classroom.is_available.return_value = True
        self.voice_command.start_llm_command.return_value = True
        self.orch._start_llm_command()
        self.assertTrue(self.voice_command.start_llm_command.called)
        kwargs = self.voice_command.start_llm_command.call_args.kwargs
        self.assertIn("on_text", kwargs)
        self.assertIn("on_status", kwargs)

    def test_start_rejected_shows_wait_hint(self):
        self.classroom.is_available.return_value = True
        self.voice_command.start_llm_command.return_value = False
        self.orch._start_llm_command()
        self.assertTrue(any("请说完当前指令" in s for s in self.status))

    # ------------------------------------------------------------------
    # 转写文本 → 意图解析 → 执行/拒绝
    # ------------------------------------------------------------------

    def test_empty_text_polite_refusal(self):
        self.orch._on_llm_command_text("")
        self.feedback.say.assert_called_once_with("没听清，请再说一次")

    def test_text_routes_to_intent_worker(self):
        self.orch.llm_client = MagicMock()
        with patch.object(self.orch, '_start_background_thread') as bg:
            self.orch._on_llm_command_text("把上一页再放一遍")
            self.assertTrue(bg.called)
            target = bg.call_args.args[0]
            args = bg.call_args.kwargs.get("args", ())
            with patch('orchestrator.parse_intent') as pi:
                pi.return_value = MagicMock(accepted=True, action="prev_slide")
                target(*args)
            pi.assert_called_once()

    def test_accepted_intent_executes_action(self):
        executed = []
        with patch.object(self.orch, 'execute_action', side_effect=executed.append):
            self.orch._on_llm_intent_result("prev_slide", "ok")
        self.assertEqual(executed, ["prev_slide"])
        self.assertTrue(any("已执行" in s for s in self.status))

    def test_rejected_intent_speaks_refusal(self):
        self.orch._on_llm_intent_result(None, "no_action")
        self.feedback.say.assert_called_once_with("没听懂，请换种说法")
        self.assertTrue(any("没听懂" in s for s in self.status))

    def test_intent_worker_signals_back(self):
        """_resolve_llm_intent 在后台线程经信号回 UI（直接验证信号 payload）。"""
        received = []
        self.orch._llm_intent_result_signal.connect(
            lambda action, reason: received.append((action, reason)))
        self.orch.mode_manager = MagicMock(current_mode_name="presentation")
        self.orch.llm_client = MagicMock()
        with patch('orchestrator.parse_intent') as pi:
            pi.return_value = MagicMock(accepted=False, reason="no_action")
            self.orch._resolve_llm_intent("今天天气不错")
        self.assertEqual(received, [(None, "no_action")])

    # ------------------------------------------------------------------
    # 云端健康（M6）
    # ------------------------------------------------------------------

    def test_degraded_updates_status_and_speaks(self):
        cloud = []
        self.orch.cloud_status_signal.connect(cloud.append)
        self.orch._on_cloud_degraded()
        self.assertEqual(self.orch.cloud_status, "degraded")
        self.assertEqual(cloud, ["degraded"])
        self.feedback.say.assert_called_once_with("网络异常，已切换离线模式")

    def test_recovered_updates_status(self):
        cloud = []
        self.orch.cloud_status_signal.connect(cloud.append)
        self.orch._on_cloud_recovered()
        self.assertEqual(self.orch.cloud_status, "online")
        self.assertEqual(cloud, ["online"])
        self.feedback.say.assert_called_once_with("云端已恢复")

    # ------------------------------------------------------------------
    # 触发词注册
    # ------------------------------------------------------------------

    def test_wake_word_registered_all_modes(self):
        self.assertEqual(VOICE_KEYWORD_TO_ACTION["小助手"], "llm_command")
        for mode in ("presentation", "mouse", "draw"):
            self.assertIn("小助手", MODE_KEYWORDS[mode])


if __name__ == "__main__":
    unittest.main(verbosity=2)
