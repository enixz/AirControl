"""语音反馈（M5）— 本地 Windows SAPI 播报

MoMA 免费包内不含云端 TTS（见 docs/m0_api_verification.md），播报改用本地
Windows SAPI（SpVoice，经 pywin32 Dispatch），离线零成本、零新增依赖。

设计：

- 队列 + 单 worker 线程：``say()`` 非阻塞投递，SAPI ``Speak`` 阻塞到播完；
- COM 线程模型：worker 线程内 ``pythoncom.CoInitialize`` 后再 Dispatch；
- KWS 防冲突：``is_speaking()`` 供 voice_command 在播报窗口内丢弃麦克风帧，
  避免自触发；
- ``stop()`` 清空待播队列并打断当前播报；
- 配置 ``voice_feedback_enabled``（默认开）、``voice_feedback_volume``
  （0-100，默认 100）；SAPI 不可用（缺 pywin32 / 非 Windows）自动禁用。
"""

import logging
import queue
import threading

logger = logging.getLogger(__name__)

try:
    import pythoncom
    import win32com.client
except ImportError:  # 非 Windows / 缺 pywin32
    pythoncom = None
    win32com = None

_DEFAULT_VOLUME = 100
_DEFAULT_RATE = 0

# SpVoice.Speak 的 SVSFDefault（同步阻塞）与 SVSFPurgeBeforeSpeak（打断清空）
_SVSF_DEFAULT = 0
_SVSF_PURGE = 2


class VoiceFeedbackService:
    """本地 SAPI 语音播报（线程安全）。"""

    def __init__(self, config=None, enabled=None):
        if enabled is not None:
            self._enabled = bool(enabled)
        else:
            self._enabled = bool(config is None or
                                config.get("voice_feedback_enabled") is not False)
        volume = config.get("voice_feedback_volume") if config else None
        self._volume = max(1, min(100, int(volume or _DEFAULT_VOLUME)))
        rate = config.get("voice_feedback_rate") if config else None
        self._rate = max(-10, min(10, int(rate or _DEFAULT_RATE)))

        self._queue = queue.Queue()
        self._speaking = False
        self._speaking_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = None
        self._voice = None  # 仅 worker 线程内访问

        if self._enabled and win32com is None:
            logger.warning("pywin32 不可用，本地语音播报禁用")
            self._enabled = False

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    def is_available(self):
        """播报服务可用（已启用且 SAPI 可用）。"""
        return self._enabled and win32com is not None

    def is_speaking(self):
        """正在播报（含队列中还有待播文本）。"""
        with self._speaking_lock:
            return self._speaking

    def say(self, text):
        """非阻塞播报一句（不可用 / 空文本静默忽略）。"""
        if not self.is_available() or not text or not str(text).strip():
            return False
        self._ensure_worker()
        with self._speaking_lock:
            self._speaking = True
        self._queue.put(str(text).strip())
        return True

    def stop(self, timeout_sec=2.0):
        """停止播报：清空队列并打断当前语音。"""
        self._stop_event.set()
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(max(0.0, float(timeout_sec)))
        with self._speaking_lock:
            self._speaking = False
        # 复位以便后续 say() 重启 worker
        self._thread = None
        self._stop_event = threading.Event()
        return thread is None or not thread.is_alive()

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _ensure_worker(self):
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(
                target=self._worker, name="VoiceFeedbackWorker", daemon=True
            )
            self._thread.start()

    def _worker(self):
        try:
            pythoncom.CoInitialize()
        except Exception:  # noqa: BLE001 - 无 COM 环境时直接退出
            logger.warning("SAPI COM 初始化失败，语音播报不可用")
            with self._speaking_lock:
                self._speaking = False
            return
        try:
            self._voice = win32com.client.Dispatch("SAPI.SpVoice")
            try:
                self._voice.Volume = self._volume
                self._voice.Rate = self._rate
            except Exception:  # noqa: BLE001
                pass
            while not self._stop_event.is_set():
                try:
                    text = self._queue.get(timeout=0.2)
                except queue.Empty:
                    with self._speaking_lock:
                        self._speaking = False
                    continue
                if self._stop_event.is_set():
                    break
                try:
                    self._voice.Speak(text, _SVSF_DEFAULT)
                except Exception:  # noqa: BLE001
                    logger.warning("SAPI 播报失败", exc_info=True)
                with self._speaking_lock:
                    self._speaking = not self._queue.empty()
        finally:
            self._voice = None
            try:
                pythoncom.CoUninitialize()
            except Exception:  # noqa: BLE001
                pass
