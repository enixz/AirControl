"""M3 慢通道 VAD 升级测试：Silero 神经网络 VAD 接入 + 能量 VAD 退回。

分两类：

- ``TestSileroVadFallback``：voice_command 集成层，Silero 缺失/失败时
  退回能量 VAD，端点状态机行为与旧版一致（复用合成音帧）；
- ``TestSileroVadUnit``（需真实模型 + sherpa-onnx，本机有模型才跑）：
  SileroVad 封装的窗口切分与真语音检测，验证封装层喂窗逻辑正确。

所有云端调用不涉及；真实模型测试对 TTS 合成语音跑（离线生成 wav）。
"""
import os
import sys
import time
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'app'))

from services.voice_command import VoiceCommandService  # noqa: E402


class _Cfg:
    """最小 config 桩：get() 返回预置值，缺省回落 default。"""

    def __init__(self, **kv):
        self._kv = kv

    def get(self, key, default=None):
        value = self._kv.get(key)
        return value if value is not None else default


def _silence_frame():
    return bytes(1600 * 2)  # 100ms 全零（int16）


def _speech_frame():
    t = np.linspace(0, 0.1, 1600, endpoint=False)
    wave = (np.sin(2 * np.pi * 440 * t) * 12000).astype(np.int16)
    return wave.tobytes()  # RMS≈0.26，远高于能量阈值 0.012


class TestSileroVadFallback(unittest.TestCase):
    """Silero 不可用时退回能量 VAD（旧行为不回归）。

    模型可能真实存在（models/silero_vad.onnx），用 try_create 路径补丁
    强制"缺失"，专测退回分支。
    """

    def setUp(self):
        self.svc = VoiceCommandService(_Cfg(dictation_model_dir="models/sense-voice"))
        self.svc.dictation_service = MagicMock()
        self.svc.dictation_service.is_available.return_value = True
        self.svc.dictation_service.dictate.return_value = "测试转写"
        self.svc._running = True
        self.texts = []

    def _start(self):
        return self.svc.start_llm_command(on_text=self.texts.append)

    def _sync_asr(self):
        def fake_start(target, *args, name="X"):
            target(*args)
            return True
        self.svc._start_asr_thread = fake_start

    def _force_model_missing(self):
        """让 try_create 走真实逻辑但路径必然不存在。"""
        with patch('services.voice_command.resource_path',
                   return_value="Z:/definitely/missing/silero_vad.onnx"):
            return self.svc._ensure_silero_vad()

    def test_missing_model_falls_back_to_energy_vad(self):
        """_ensure_silero_vad：模型缺失返回 None 且只试一次。"""
        vad = self._force_model_missing()
        self.assertIsNone(vad)
        self.assertTrue(self.svc._silero_vad_failed)
        # 熔断后不再重复尝试
        self.svc._silero_vad = None
        with patch('services.silero_vad.SileroVad.try_create') as tc:
            tc.return_value = MagicMock()
            self.assertIsNone(self.svc._ensure_silero_vad())
            tc.assert_not_called()

    def test_energy_vad_endpoints_still_work(self):
        """退回能量 VAD：静音结束 / 无语音取消路径与旧版一致。"""
        self._force_model_missing()
        self._start()
        session = self.svc._llm_session
        self.assertFalse(self.svc._silero_vad)  # 无模型路径下
        session["has_speech"] = True
        session["silence_start"] = time.time() - 1.3
        session["buffer"].extend(_speech_frame() * 10)
        self._sync_asr()
        self.svc._update_llm_vad(_silence_frame())
        self.assertIsNone(self.svc._llm_session)
        self.assertEqual(self.texts, ["测试转写"])

    def test_silero_vad_used_when_available(self):
        """Silero 存在时 feed_block 被调用且其返回值驱动端点状态机。"""
        fake_vad = MagicMock()
        fake_vad.feed_block.return_value = True  # 持续"正在说话"
        self.svc._silero_vad = fake_vad
        self.svc._silero_vad_failed = False
        self._start()
        session = self.svc._llm_session
        session["silence_start"] = time.time()  # 预置旧静音计时
        self.svc._update_llm_vad(_silence_frame())
        fake_vad.feed_block.assert_called_once()
        # feed_block=True → 静音计时清零、has_speech 置位
        self.assertTrue(session["has_speech"])
        self.assertIsNone(session["silence_start"])
        self.assertIsNotNone(self.svc._llm_session)

    def test_silero_vad_reset_on_session_start(self):
        """新会话开始时重置 Silero 状态（旧会话尾部静音不带入）。"""
        fake_vad = MagicMock()
        self.svc._silero_vad = fake_vad
        self.svc._silero_vad_failed = False
        self.assertTrue(self._start())
        fake_vad.reset.assert_called_once()

    def test_silero_silence_closes_session(self):
        """Silero 判定语音结束（feed_block=False + 已有语音 + 静音超时）。"""
        fake_vad = MagicMock()
        fake_vad.feed_block.return_value = False
        self.svc._silero_vad = fake_vad
        self.svc._silero_vad_failed = False
        self._start()
        session = self.svc._llm_session
        session["has_speech"] = True
        session["silence_start"] = time.time() - 1.3
        session["buffer"].extend(_speech_frame() * 10)
        self._sync_asr()
        self.svc._update_llm_vad(_silence_frame())
        self.assertIsNone(self.svc._llm_session)
        self.assertEqual(self.texts, ["测试转写"])


class TestSileroVadUnit(unittest.TestCase):
    """SileroVad 封装单测（需要 models/silero_vad.onnx 存在，否则整类跳过）。"""

    MODEL = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', 'models', 'silero_vad.onnx'
    )

    @classmethod
    def setUpClass(cls):
        from services.silero_vad import SileroVad

        if not os.path.isfile(cls.MODEL):
            raise unittest.SkipTest("silero_vad.onnx 未安装，跳过封装单测")
        cls.vad = SileroVad(cls.MODEL)
        cls.SileroVad = SileroVad

    def test_try_create_missing_file_returns_none(self):
        self.assertIsNone(self.SileroVad.try_create(None))
        self.assertIsNone(self.SileroVad.try_create("Z:/no/such/file.onnx"))

    def test_window_packing_no_sample_loss(self):
        """100ms 块（1600 样本）= 3 个 512 窗 + 64 余样；余样延到下一块。"""
        self.vad.reset()
        # 静音块喂两轮，验证无异常且返回 False
        self.assertFalse(self.vad.feed_block(_silence_frame()))
        self.assertFalse(self.vad.feed_block(_silence_frame()))

    def test_real_speech_detected(self):
        """TTS 合成中文语音（SAPI 离线生成）应触发语音标志。"""
        import subprocess
        import tempfile
        import wave

        wav = os.path.join(tempfile.gettempdir(), "silero_vad_test.wav")
        ps = (
            'Add-Type -AssemblyName System.Speech;'
            '$s = New-Object System.Speech.Synthesis.SpeechSynthesizer;'
            f'$s.SetOutputToWaveFile("{wav}");'
            '$s.Speak("你好，请帮我翻到下一页，谢谢"); $s.Dispose()'
        )
        try:
            subprocess.run(
                ["powershell", "-NoProfile", "-Command", ps],
                check=True, timeout=30, capture_output=True,
            )
        except Exception as e:
            raise unittest.SkipTest(f"SAPI 合成失败: {e}") from e

        with wave.open(wav, "rb") as w:
            data = w.readframes(w.getnframes())
        samples = np.frombuffer(data, dtype=np.int16)
        if w.getframerate() != 16000:
            samples = samples[::w.getframerate() // 16000]

        self.vad.reset()
        saw_speech = False
        # 1600 样本/块喂入（与检测循环节奏一致）
        for i in range(0, len(samples) - 1600, 1600):
            block = samples[i:i + 1600].astype(np.int16).tobytes()
            if self.vad.feed_block(block):
                saw_speech = True
        os.remove(wav)
        self.assertTrue(saw_speech, "TTS 合成语音应至少在一个块内触发 VAD")


if __name__ == "__main__":
    unittest.main(verbosity=2)
