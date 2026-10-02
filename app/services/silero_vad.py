"""Silero VAD 智能指令语音活动检测（M3 慢通道）

背景：原能量 VAD 用固定 RMS 阈值（~-38dBFS）判定"有人说话"。教室环境
（学生说话、风扇、投影仪底噪）下该阈值两个方向都会翻车：底噪高于阈值
→ 永不判定静音 → 会话总跑满 8s 上限；呼吸声即触发 → 误判说话。
Silero VAD 是神经网络端点检测，训练目标就是"人声 vs 非人声"，对底噪
鲁棒；sherpa-onnx 1.13 自带推理支持，项目已依赖，仅缺模型文件。

模型：silero_vad.onnx（629KB，MIT 许可，sherpa-onnx releases 下载），
不入 git（见 .gitignore）；缺失时 is_available() 返回 False，
调用方退回能量 VAD，行为与旧版一致。

线程模型：本类只在检测循环线程（_detection_loop）使用，无内部锁。
"""

import logging
import os

import numpy as np

logger = logging.getLogger(__name__)


class SileroVad:
    """Silero VAD 封装 — 逐 100ms 音频块喂入，返回该块是否含人声。

    sherpa-onnx VoiceActivityDetector 的语义：

    - ``accept_waveform`` 逐窗喂入（窗口固定 512 样本 @16kHz = 32ms）；
    - ``is_speech_detected()`` 在检测到语音时为 True，语音段闭合
      （静音达到 min_silence_duration）后回 False；
    - 本封装不使用段队列（那用于分段转写），只用布尔标志做端点判定，
      与原能量 VAD 的 has_speech / silence_start 状态机同构。

    Args:
        model_path: silero_vad.onnx 路径；不存在时抛 FileNotFoundError
                    由工厂函数捕获。
        threshold: 语音概率阈值（默认 0.5）
        min_silence_duration: 判定"说完"所需的持续静音秒数
    """

    # 100ms 块 @16kHz = 1600 样本 = 3.125 个 512 窗口；逐窗喂入
    WINDOW = 512

    def __init__(self, model_path, threshold=0.5, min_silence_duration=1.2):
        import sherpa_onnx  # 局部导入：无模型环境不拖累其他测试

        cfg = sherpa_onnx.VadModelConfig()
        cfg.silero_vad.model = model_path
        cfg.silero_vad.threshold = float(threshold)
        cfg.silero_vad.min_silence_duration = float(min_silence_duration)
        cfg.sample_rate = 16000
        self._vad = sherpa_onnx.VoiceActivityDetector(cfg)
        self._sample_rate = 16000
        # 缓冲跨块剩余样本（100ms 块不整除 32ms 窗）
        self._pending = np.empty(0, dtype=np.float32)

    @classmethod
    def try_create(cls, model_path, **kwargs):
        """尝试加载模型；失败（文件缺失/sherpa 异常）返回 None，调用方退回能量 VAD。"""
        if not model_path or not os.path.isfile(model_path):
            logger.info("Silero VAD 模型缺失（%s），慢通道退回能量 VAD", model_path)
            return None
        try:
            return cls(model_path, **kwargs)
        except Exception:
            logger.warning("Silero VAD 初始化失败，慢通道退回能量 VAD", exc_info=True)
            return None

    def reset(self):
        """新会话开始时清空状态。"""
        try:
            self._vad.reset()
        except Exception:
            pass
        self._pending = np.empty(0, dtype=np.float32)

    def feed_block(self, audio_data):
        """喂入一个音频块（bytes，int16 PCM @16kHz），返回本块内是否检测到语音。

        实现细节：sherpa VAD 要求按 512 样本整窗喂入；把跨块剩余样本与
        当前块拼接后切窗，尾部不足一窗的留到下一块。
        """
        block = np.frombuffer(audio_data, dtype=np.int16).astype(
            np.float32
        ) / 32768.0
        samples = np.concatenate([self._pending, block])
        n_windows = len(samples) // self.WINDOW
        if n_windows == 0:
            self._pending = samples
            return False
        for i in range(n_windows):
            window = samples[i * self.WINDOW:(i + 1) * self.WINDOW]
            self._vad.accept_waveform(window.tolist())
        self._pending = samples[n_windows * self.WINDOW:]
        # is_speech_detected 为 True 表示"正在说话"（段未闭合）；
        # 段闭合（静音够久）后回 False —— 与端点判定语义一致
        try:
            return bool(self._vad.is_speech_detected())
        except Exception:
            return False
