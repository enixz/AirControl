"""
语音听写服务 — 基于 SenseVoice-Small 的离线 ASR

用途：板书模式说"开始板书"开始录音，说"结束板书"停止并转文字写到画布。
模型：sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17 (int8)
"""

import logging
import os
import re
import threading

import numpy as np

try:
    import sherpa_onnx
except ImportError:
    sherpa_onnx = None

logger = logging.getLogger(__name__)

# SenseVoice 输出会包含元标签 <|zh|><|NEUTRAL|><|Speech|><|withitn|> 等，需要剥离
_META_TAG_RE = re.compile(r"<\|[^|]*\|>")


class VoiceDictationService:
    """SenseVoice-Small 离线语音听写。

    模型懒加载——首次调用 dictate() 时才载入，避免拖慢启动。
    """

    SAMPLE_RATE = 16000

    def __init__(self, config):
        self.config = config
        self._recognizer = None
        self._load_lock = threading.Lock()
        self._load_failed = False
        # partial ASR 与"结束板书"复核可能并发调用 dictate()，串行化以避免
        # 同一 recognizer 上的潜在线程不安全
        self._dictate_lock = threading.Lock()

        base_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        model_subdir = config.get("dictation_model_dir") or "models/sense-voice"
        self._model_dir = (
            model_subdir
            if os.path.isabs(model_subdir)
            else os.path.join(base_dir, model_subdir)
        )
        self._language = config.get("dictation_language") or "auto"
        self._use_itn = config.get("dictation_use_itn") is not False
        self._num_threads = config.get("dictation_num_threads") or 2

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    def is_available(self):
        """检查模型文件是否存在（不触发加载）"""
        if sherpa_onnx is None:
            return False
        model_file = self._resolve_model_file()
        tokens_file = os.path.join(self._model_dir, "tokens.txt")
        return model_file is not None and os.path.isfile(tokens_file)

    def dictate(self, samples, sample_rate=None):
        """对一段音频做语音识别，返回纯文本（已去除元标签）。

        Args:
            samples: np.float32 数组，范围 [-1, 1]
            sample_rate: 默认 16000

        Returns:
            识别出的文本字符串；失败或空时返回 ""。
        """
        if sherpa_onnx is None:
            logger.warning("sherpa-onnx 未安装，听写不可用")
            return ""

        if not self._ensure_loaded():
            return ""

        sr = sample_rate or self.SAMPLE_RATE
        if not isinstance(samples, np.ndarray):
            samples = np.asarray(samples, dtype=np.float32)
        elif samples.dtype != np.float32:
            samples = samples.astype(np.float32)

        if samples.size == 0:
            return ""

        try:
            with self._dictate_lock:
                stream = self._recognizer.create_stream()
                stream.accept_waveform(sr, samples)
                self._recognizer.decode_stream(stream)
                raw = stream.result.text or ""
        except Exception as e:
            logger.error("SenseVoice 识别失败: %s", e, exc_info=True)
            return ""

        return self._strip_meta(raw).strip()

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _ensure_loaded(self):
        if self._recognizer is not None:
            return True
        if self._load_failed:
            return False
        with self._load_lock:
            if self._recognizer is not None:
                return True
            if self._load_failed:
                return False
            try:
                self._load_model()
                return True
            except Exception as e:
                logger.error("SenseVoice 模型加载失败: %s", e, exc_info=True)
                self._load_failed = True
                return False

    def _resolve_model_file(self):
        """优先 int8，回退到全精度"""
        for name in ("model.int8.onnx", "model.onnx"):
            path = os.path.join(self._model_dir, name)
            if os.path.isfile(path):
                return path
        return None

    def _load_model(self):
        if not os.path.isdir(self._model_dir):
            raise FileNotFoundError(
                f"SenseVoice 模型目录不存在: {self._model_dir}"
            )

        model_file = self._resolve_model_file()
        tokens_file = os.path.join(self._model_dir, "tokens.txt")
        if model_file is None:
            raise FileNotFoundError(
                f"找不到 model.int8.onnx 或 model.onnx 于 {self._model_dir}"
            )
        if not os.path.isfile(tokens_file):
            raise FileNotFoundError(f"找不到 tokens.txt 于 {self._model_dir}")

        logger.info("加载 SenseVoice 模型: %s", model_file)
        self._recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=model_file,
            tokens=tokens_file,
            num_threads=self._num_threads,
            use_itn=self._use_itn,
            language=self._language,
            debug=False,
        )
        logger.info("SenseVoice 已就绪（language=%s, itn=%s）", self._language, self._use_itn)

    @staticmethod
    def _strip_meta(text):
        return _META_TAG_RE.sub("", text)
