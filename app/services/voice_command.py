"""
语音指令服务 — 基于 Sherpa-ONNX KWS 的离线关键词检测

架构：双引擎协同
- 引擎 A：Sherpa-ONNX KWS（离线，常驻）— 处理全部固定指令词
- 引擎 B：SenseVoice-Small ASR（离线，按需）— 板书模式"开始板书/结束板书"触发

与现有手势系统共享 execute_action() 调度。
"""

import logging
import os
import queue
import re
import tempfile
import threading
import time

import numpy as np
from runtime_paths import resource_path

try:
    import sounddevice as sd
except ImportError:
    sd = None

try:
    import sherpa_onnx
except ImportError:
    sherpa_onnx = None

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 关键词 → Action 映射
# ---------------------------------------------------------------------------

VOICE_KEYWORD_TO_ACTION = {
    # 全局：助手 = 本程序窗口；豆包 = 外部 AI 助手
    "最小化助手": "minimize_assistant",
    "显示助手": "restore_assistant",
    "召唤豆包": "launch_voice_assistant",
    # 演示模式
    "开始播放": "start_presentation",
    "结束播放": "end_presentation",
    "下一页": "next_slide",
    "上一页": "prev_slide",
    # 鼠标模式
    "点一下": "left_click",
    "双击": "double_click",
    "右键": "right_click",
    # 板书模式
    "清屏": "clear_canvas",
    "开始板书": "start_dictation",
    "结束板书": "stop_dictation",
    # 模式直跳（已经覆盖"切模式"的需求，无需循环切换指令）
    "板书模式": "switch_to_draw",
    "鼠标模式": "switch_to_mouse",
    "演示模式": "switch_to_presentation",
    # 板书模式 — 图形修正
    "图形修正": "toggle_shape_correction",
}

# 各模式可用的关键词（None 表示全部可用）
MODE_KEYWORDS = {
    "presentation": [
        "开始播放", "结束播放", "下一页", "上一页",
        "最小化助手", "显示助手", "召唤豆包",
        "板书模式", "鼠标模式",
    ],
    "mouse": [
        "点一下", "双击", "右键",
        "最小化助手", "显示助手", "召唤豆包",
        "板书模式", "演示模式",
    ],
    "draw": [
        "清屏", "开始板书", "结束板书", "图形修正",
        "最小化助手", "显示助手", "召唤豆包",
        "演示模式", "鼠标模式",
    ],
}


class VoiceCommandService:
    """语音指令服务 — KWS 常驻检测 + 按需 SenseVoice 离线 ASR"""

    SAMPLE_RATE = 16000
    CHUNK_DURATION_MS = 100  # 每次 KWS 处理 100ms 音频
    CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)  # 1600 samples

    def __init__(self, config, action_callback=None, dictation_service=None):
        """
        Args:
            config: ConfigManager 实例
            action_callback: callable(action_name: str) — 回调到 FloatingWindow.execute_action()
            dictation_service: 可选的 VoiceDictationService 实例，注入后才能启用听写
        """
        self.config = config
        self.action_callback = action_callback
        self.dictation_service = dictation_service

        self._kws = None
        self._kws_stream = None
        self._audio_stream = None
        self._audio_queue = None
        self._running = False
        self._thread = None
        self._current_mode = None
        self._last_keyword_time = 0
        self._cooldown = 1.0  # 关键词触发冷却（秒）

        # 听写模式状态
        self._dictation_mode = False
        self._dictation_buffer = bytearray()
        self._dictation_start_time = 0.0
        self._dictation_callback = None
        self._dictation_status_callback = None
        self._dictation_partial_callback = None
        # session_id 区分不同的听写会话，partial ASR 在独立线程跑，
        # 完成时比对 session_id，若用户已经结束/重新开始则丢弃结果。
        self._dictation_session_id = 0
        self._partial_busy = False
        self._last_partial_time = 0.0

        # 线程安全：保护 _kws 和 _kws_stream 的并发访问
        # _detection_loop（工作线程）和 on_mode_changed（主线程）共享这些对象
        self._kws_lock = threading.Lock()
        self._reload_lock = threading.Lock()
        self._reload_thread = None
        self._pending_reload_mode = None

        # 模型路径 — 基于 AirControl 项目根目录
        # 本文件位于 app/services/voice_command.py，向上两层即为项目根目录
        self._model_dir = resource_path("models", "kws-zh-wenetspeech")
        self._keywords_dir = resource_path("app", "voice_keywords")
        cache_root = os.environ.get("LOCALAPPDATA") or tempfile.gettempdir()
        self._keywords_cache_dir = os.path.join(cache_root, "AirControl", "kws")
        os.makedirs(self._keywords_cache_dir, exist_ok=True)

        # 状态通知（供 UI 绑定）
        self._status_text = ""
        self._status_callback = None

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    def start(self):
        """启动语音指令检测"""
        if self._running:
            return

        if sherpa_onnx is None:
            logger.warning("sherpa-onnx 未安装，语音指令功能不可用")
            return False

        if sd is None:
            logger.warning("sounddevice 未安装，语音指令功能不可用")
            return False

        if not os.path.isdir(self._model_dir):
            logger.warning("KWS 模型目录不存在: %s", self._model_dir)
            return False

        try:
            self._init_kws()
            self._init_audio()
        except Exception as e:
            logger.error("语音指令服务初始化失败: %s", e, exc_info=True)
            return False

        self._running = True
        self._thread = threading.Thread(target=self._detection_loop, daemon=True)
        self._thread.start()
        logger.info("语音指令服务已启动")
        return True

    def stop(self):
        """停止语音指令检测"""
        self._running = False
        with self._reload_lock:
            self._pending_reload_mode = None
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        reload_thread = self._reload_thread
        if reload_thread and reload_thread.is_alive():
            reload_thread.join(timeout=3.0)
        self._cleanup_audio()
        self._kws = None
        self._kws_stream = None
        logger.info("语音指令服务已停止")

    def on_mode_changed(self, mode_name):
        """模式切换时更新关键词集（异步，不阻塞调用方）

        KWS 重建需要加载几百 MB 模型，同步执行会卡住调用线程（通常是主 UI 线程）
        几秒。这里只记录目标模式并派发到后台线程，加载完成后再原子替换 _kws/_kws_stream。

        若加载期间用户又切了模式，旧的加载结果会被丢弃（_current_mode 比对）。
        """
        if self._current_mode == mode_name:
            return
        self._current_mode = mode_name
        if self._kws is None:
            return  # 服务尚未启动，start() 时会按当前模式构建
        self._schedule_reload(mode_name)

    def request_reload(self):
        """Reload the current mode after a KWS configuration change."""
        mode = self._current_mode or self.config.get("interaction_mode") or "mouse"
        self._current_mode = mode
        if self._kws is not None:
            self._schedule_reload(mode)

    def _schedule_reload(self, mode_name):
        with self._reload_lock:
            self._pending_reload_mode = mode_name
            if self._reload_thread and self._reload_thread.is_alive():
                return
            self._reload_thread = threading.Thread(
                target=self._reload_worker,
                name="KwsReloadWorker",
                daemon=True,
            )
            self._reload_thread.start()

    def _reload_worker(self):
        """Build one KWS instance at a time and coalesce rapid mode changes."""
        while self._running:
            with self._reload_lock:
                mode_name = self._pending_reload_mode
                self._pending_reload_mode = None
            if mode_name is None:
                return

            try:
                new_kws, new_stream = self._build_kws_for_mode(mode_name)
            except Exception:
                logger.exception("KWS 异步重建失败 (mode=%s)", mode_name)
                continue

            with self._kws_lock:
                if not self._running:
                    return
                if self._current_mode == mode_name:
                    self._kws = new_kws
                    self._kws_stream = new_stream
                    logger.info("KWS 关键词已异步更新为模式: %s", mode_name)
                else:
                    logger.info(
                        "KWS 重建完成但模式已变更 (%s -> %s)，丢弃本次结果",
                        mode_name, self._current_mode,
                    )

    def set_status_callback(self, callback):
        """设置状态回调（用于 UI 更新）"""
        self._status_callback = callback

    @property
    def is_running(self):
        return self._running

    @property
    def is_dictating(self):
        return self._dictation_mode

    # 听写最长录音保护（秒），防止用户忘记说"结束板书"
    MAX_DICTATION_DURATION = 60.0
    # 实时字幕节奏：每隔多少秒触发一次 partial ASR
    PARTIAL_INTERVAL = 1.0
    # 触发 partial 前累积音频的最小时长（秒），太短识别基本是空，浪费 CPU
    PARTIAL_MIN_AUDIO_SEC = 0.6
    # 听写开始后多少秒内忽略"结束板书"触发；防止刚开口就被 KWS 误触发截断
    MIN_DICTATION_BEFORE_STOP = 1.5
    # KWS 命中"结束板书"后，用 SenseVoice 复核最近多少秒音频
    STOP_VERIFY_AUDIO_SEC = 3.0

    def start_dictation(self, on_text=None, on_status=None, on_partial=None):
        """切换到听写模式：持续录音直到 stop_dictation() 或超时。

        说 "结束板书" 会被 KWS 自动检测并内部调用 stop_dictation()。

        Args:
            on_text: callable(text: str) — 最终识别完成回调（worker 线程触发）
            on_status: callable(phase: str, payload) — 状态变化通知
                       phase ∈ {"started", "tick", "decoding", "done", "failed"}
            on_partial: callable(text: str) — 实时增量识别回调（worker 线程触发）

        Returns:
            True 表示已进入听写模式，False 表示拒绝
        """
        if not self._running:
            logger.warning("语音服务未运行，无法启动听写")
            return False
        if self._dictation_mode:
            logger.info("听写已在进行中，忽略重复触发")
            return False
        if self.dictation_service is None or not self.dictation_service.is_available():
            logger.warning("听写服务不可用（模型未找到？）")
            if on_status:
                try:
                    on_status("failed", "model_missing")
                except Exception:
                    pass
            return False

        self._dictation_buffer = bytearray()
        self._dictation_start_time = time.time()
        self._dictation_callback = on_text
        self._dictation_status_callback = on_status
        self._dictation_partial_callback = on_partial
        self._dictation_session_id += 1
        self._partial_busy = False
        self._last_partial_time = time.time()
        self._dictation_mode = True

        if on_status:
            try:
                on_status("started", None)
            except Exception:
                pass
        logger.info("听写已开始（说'结束板书'停止，最长 %ds）", int(self.MAX_DICTATION_DURATION))
        return True

    def stop_dictation(self):
        """手动停止听写并触发 ASR。由 "结束板书" 关键词或超时自动调用。"""
        if not self._dictation_mode:
            return
        self._finish_dictation()

    # ------------------------------------------------------------------
    # KWS 初始化
    # ------------------------------------------------------------------

    def _init_kws(self):
        """初始化 Sherpa-ONNX KWS 引擎"""
        int8_encoder = os.path.join(
            self._model_dir,
            "encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx",
        )
        decoder = os.path.join(
            self._model_dir,
            "decoder-epoch-12-avg-2-chunk-16-left-64.onnx",
        )
        int8_joiner = os.path.join(
            self._model_dir,
            "joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx",
        )
        tokens = os.path.join(self._model_dir, "tokens.txt")

        # 验证文件存在
        for f in [int8_encoder, decoder, int8_joiner, tokens]:
            if not os.path.isfile(f):
                raise FileNotFoundError(f"KWS 模型文件不存在: {f}")

        # 先生成当前模式的关键词文件
        mode = self._current_mode or self.config.get("interaction_mode") or "mouse"
        keywords_file = self._generate_mode_keywords(mode)
        if not os.path.isfile(keywords_file):
            raise FileNotFoundError(f"关键词文件不存在: {keywords_file}")

        self._kws = sherpa_onnx.KeywordSpotter(
            tokens=tokens,
            encoder=int8_encoder,
            decoder=decoder,
            joiner=int8_joiner,
            keywords_file=keywords_file,
            keywords_threshold=self.config.get("voice_command_threshold") or 0.25,
            num_threads=2,
            provider="cpu",
            max_active_paths=4,
        )
        self._kws_stream = self._kws.create_stream()
        logger.info("KWS 引擎初始化成功，模型: %s", self._model_dir)

    def _init_audio(self):
        """初始化 sounddevice 麦克风采集"""
        self._audio_queue = queue.Queue(maxsize=20)
        self._audio_stream = sd.RawInputStream(
            samplerate=self.SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=self.CHUNK_SAMPLES,
            callback=self._audio_callback,
        )
        self._audio_stream.start()
        logger.info("麦克风音频流已启动 (16kHz mono, sounddevice)")

    def _audio_callback(self, indata, frames, time_info, status):
        """sounddevice 回调 — 将音频数据放入队列"""
        if status:
            logger.warning("音频回调状态: %s", status)
        try:
            self._audio_queue.put_nowait(bytes(indata))
        except queue.Full:
            # 队列满时丢弃最旧数据
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._audio_queue.put_nowait(bytes(indata))
            except queue.Full:
                pass

    # ------------------------------------------------------------------
    # 关键词管理
    # ------------------------------------------------------------------

    def _generate_mode_keywords(self, mode_name):
        """Generate a mode-specific keyword file outside the repository."""
        raw_file = os.path.join(self._keywords_dir, "keywords.txt")
        safe_mode = mode_name if mode_name in MODE_KEYWORDS else "all"
        active_file = os.path.join(
            self._keywords_cache_dir, f"keywords_{safe_mode}.txt"
        )

        # 读取完整关键词文件
        with open(raw_file, "r", encoding="utf-8") as f:
            all_lines = [line.strip() for line in f if line.strip()]

        # 筛选当前模式可用的关键词
        allowed = set(MODE_KEYWORDS.get(mode_name, set()))
        if not allowed:
            # 如果没有指定，使用全部关键词
            filtered_lines = all_lines
        else:
            filtered_lines = []
            for line in all_lines:
                # 提取 @ 后面的显示名称
                if "@" in line:
                    display = line.split("@")[-1].strip()
                else:
                    display = line.strip()
                if display in allowed:
                    filtered_lines.append(line)

        temp_file = active_file + ".tmp"
        with open(temp_file, "w", encoding="utf-8") as f:
            for line in filtered_lines:
                f.write(line + "\n")
        os.replace(temp_file, active_file)

        logger.info("模式 %s 激活 %d 个关键词", mode_name, len(filtered_lines))
        return active_file

    def _build_kws_for_mode(self, mode_name):
        """生成关键词文件并构建一个新的 KeywordSpotter + stream。

        纯构建，不修改 self 上的任何引用 — 调用方负责在锁内做原子替换。
        sherpa-onnx KWS 不支持运行时更换关键词文件，必须重建实例。
        """
        keywords_file = self._generate_mode_keywords(mode_name)

        int8_encoder = os.path.join(
            self._model_dir,
            "encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx",
        )
        decoder = os.path.join(
            self._model_dir,
            "decoder-epoch-12-avg-2-chunk-16-left-64.onnx",
        )
        int8_joiner = os.path.join(
            self._model_dir,
            "joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx",
        )
        tokens = os.path.join(self._model_dir, "tokens.txt")

        new_kws = sherpa_onnx.KeywordSpotter(
            tokens=tokens,
            encoder=int8_encoder,
            decoder=decoder,
            joiner=int8_joiner,
            keywords_file=keywords_file,
            keywords_threshold=self.config.get("voice_command_threshold") or 0.25,
            num_threads=2,
            provider="cpu",
            max_active_paths=4,
        )
        return new_kws, new_kws.create_stream()

    # ------------------------------------------------------------------
    # 检测循环
    # ------------------------------------------------------------------

    def _detection_loop(self):
        """核心检测循环（独立线程）

        听写模式下音频同时送入 KWS 和听写缓冲：
        - KWS 仅响应 "结束板书"（其余关键词被 _handle_keyword 忽略）
        - 听写缓冲不断累积直到 stop_dictation() 或超时

        线程安全设计：
        - KWS 操作（accept_waveform / decode / get_result / reset_stream）全部在
          _kws_lock 保护下执行，防止 on_mode_changed() 重建 KWS 时使用已释放的 C++ 对象
        - action_callback 通过 queue + 主线程 marshal 执行，避免同线程重入
        """
        logger.info("KWS 检测循环已启动")

        while self._running:
            try:
                try:
                    audio_data = self._audio_queue.get(timeout=0.2)
                except queue.Empty:
                    if self._dictation_mode:
                        self._check_dictation_timeout()
                    continue

                # 听写模式：音频同时入缓冲 + KWS（检测"结束板书"）
                if self._dictation_mode:
                    self._dictation_buffer.extend(audio_data)
                    elapsed = time.time() - self._dictation_start_time
                    if self._dictation_status_callback:
                        try:
                            self._dictation_status_callback("tick", elapsed)
                        except Exception:
                            pass
                    self._check_dictation_timeout()
                    self._maybe_trigger_partial_asr()
                    # 不 continue — 继续往下走 KWS，检测"结束板书"

                # int16 → float32 归一化
                samples = np.frombuffer(audio_data, dtype=np.int16).astype(
                    np.float32
                ) / 32768.0

                # KWS 操作全部在锁内完成，防止 on_mode_changed 重建实例时
                # 访问已释放的 C++ 对象导致段错误/线程崩溃
                with self._kws_lock:
                    kws = self._kws
                    kws_stream = self._kws_stream
                    if kws is None or kws_stream is None:
                        continue
                    kws_stream.accept_waveform(self.SAMPLE_RATE, samples)

                    # 尝试解码
                    detected_keyword = None
                    while kws.is_ready(kws_stream):
                        kws.decode_stream(kws_stream)
                        result = kws.get_result(kws_stream)
                        if result:
                            detected_keyword = result
                            kws.reset_stream(kws_stream)
                            break

                # 关键词处理在锁外执行（_handle_keyword 可能触发 action_callback
                # → execute_action → _set_mode → on_mode_changed，需要重新获取锁）
                if detected_keyword:
                    self._handle_keyword(detected_keyword)

            except IOError as e:
                logger.warning("麦克风读取错误: %s", e)
                time.sleep(0.1)
            except Exception as e:
                logger.error("KWS 检测循环异常: %s", e, exc_info=True)
                time.sleep(0.1)

        logger.info("KWS 检测循环已退出")

    def _check_dictation_timeout(self):
        """超时保护：录音超过 MAX_DICTATION_DURATION 自动停止。"""
        if not self._dictation_mode:
            return
        elapsed = time.time() - self._dictation_start_time
        if elapsed >= self.MAX_DICTATION_DURATION:
            logger.warning("听写超时 (%ds)，自动停止", int(self.MAX_DICTATION_DURATION))
            self._finish_dictation()

    def _maybe_trigger_partial_asr(self):
        """听写过程中按节奏触发增量 ASR，把当前累积音频拿去识别，
        结果通过 on_partial 实时回调。在独立 daemon 线程跑，避免阻塞
        KWS 检测循环错过"结束板书"。
        """
        if self._dictation_partial_callback is None:
            return
        if self._partial_busy:
            return
        now = time.time()
        if now - self._last_partial_time < self.PARTIAL_INTERVAL:
            return
        # 音频太短直接跳过，识别基本是空字符串，浪费 CPU
        # int16 = 2 bytes/sample
        min_bytes = int(self.PARTIAL_MIN_AUDIO_SEC * self.SAMPLE_RATE) * 2
        if len(self._dictation_buffer) < min_bytes:
            return

        self._last_partial_time = now
        self._partial_busy = True
        max_window_sec = float(
            self.config.get("dictation_partial_window_sec", 12.0) or 12.0
        )
        max_bytes = int(max_window_sec * self.SAMPLE_RATE) * 2
        snapshot = bytes(self._dictation_buffer[-max_bytes:])
        session_id = self._dictation_session_id
        callback = self._dictation_partial_callback
        threading.Thread(
            target=self._run_partial_asr,
            args=(snapshot, session_id, callback),
            daemon=True,
        ).start()

    def _run_partial_asr(self, audio_bytes, session_id, callback):
        """独立线程：跑一次 ASR 并把结果回调出去（若 session 仍有效）。"""
        text = ""
        try:
            samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(
                np.float32
            ) / 32768.0
            text = self.dictation_service.dictate(samples, self.SAMPLE_RATE)
            text = self._strip_stop_keyword(text)
        except Exception:
            logger.exception("Partial ASR 失败")
        finally:
            self._partial_busy = False

        # 用户已"结束板书"或开始新会话，本次结果作废，避免覆盖最终字幕
        if session_id != self._dictation_session_id:
            return
        if not callback:
            return
        try:
            callback(text)
        except Exception:
            logger.exception("Partial 回调异常")

    def _verify_and_finish_dictation(self, session_id):
        """异步：用 SenseVoice 复核最近一段音频是否真的有"结束板书"。

        KWS 是低开销的关键词检测器（threshold=0.25），误识别率较高。
        SenseVoice 是完整 ASR，准确率高得多，用它做二次确认。
        - 命中：调用 _finish_dictation 正常结束
        - 未命中：当作 KWS 误触发，听写继续，用户可以继续说
        """
        if self.dictation_service is None:
            # 没法验证，按原逻辑直接结束
            if session_id == self._dictation_session_id and self._dictation_mode:
                self._finish_dictation()
            return

        # 用 session_id 防止跨会话错乱
        if session_id != self._dictation_session_id or not self._dictation_mode:
            return

        tail_bytes = int(self.STOP_VERIFY_AUDIO_SEC * self.SAMPLE_RATE) * 2  # int16
        buf = self._dictation_buffer
        audio_tail = bytes(buf[-tail_bytes:]) if len(buf) > tail_bytes else bytes(buf)

        # 太短就不验证（< 0.5s），直接信任 KWS
        if len(audio_tail) < self.SAMPLE_RATE:  # 0.5s * 16k * 2bytes
            if session_id == self._dictation_session_id and self._dictation_mode:
                self._finish_dictation()
            return

        try:
            samples = np.frombuffer(audio_tail, dtype=np.int16).astype(
                np.float32
            ) / 32768.0
            text = self.dictation_service.dictate(samples, self.SAMPLE_RATE)
        except Exception:
            logger.exception("'结束板书' 验证 ASR 失败，按 KWS 结果停止")
            if session_id == self._dictation_session_id and self._dictation_mode:
                self._finish_dictation()
            return

        # 验证期间用户可能已通过其它方式停了听写
        if session_id != self._dictation_session_id or not self._dictation_mode:
            return

        text_norm = (text or "").replace(" ", "")
        if any(v in text_norm for v in ("结束板书", "结束版书", "结束办书",
                                         "结束半数", "结束半书")):
            logger.info("'结束板书' 验证通过 (audio=%r)，停止听写", text)
            self._finish_dictation()
        else:
            logger.info(
                "'结束板书' KWS 误触发（最近 %.1fs 音频识别为 %r），继续听写",
                self.STOP_VERIFY_AUDIO_SEC, text,
            )

    def _finish_dictation(self):
        """结束听写：同步清状态，ASR 推到后台线程异步跑。

        可能从 KWS 工作线程（"结束板书"）/ 主线程（stop_dictation/字幕写满）
        / 超时检查 调用。同步跑 ASR 会卡调用线程几秒（尤其主线程触发时会
        冻 UI），因此 ASR 部分扔进 daemon 线程。
        """
        if not self._dictation_mode:
            return

        on_text = self._dictation_callback
        on_status = self._dictation_status_callback
        audio_bytes = bytes(self._dictation_buffer)

        # 立刻清状态，避免重入；session_id 自增让在飞的 partial ASR 结果作废
        self._dictation_mode = False
        self._dictation_buffer = bytearray()
        self._dictation_callback = None
        self._dictation_status_callback = None
        self._dictation_partial_callback = None
        self._dictation_session_id += 1
        if self._kws is not None and self._kws_stream is not None:
            try:
                with self._kws_lock:
                    if self._kws is not None and self._kws_stream is not None:
                        self._kws.reset_stream(self._kws_stream)
            except Exception:
                pass

        # ASR 异步跑，调用方立即返回不阻塞
        threading.Thread(
            target=self._run_final_asr,
            args=(audio_bytes, on_text, on_status),
            daemon=True,
        ).start()

    def _run_final_asr(self, audio_bytes, on_text, on_status):
        """后台线程：跑最终 ASR + 状态/文本回调。"""
        if on_status:
            try:
                on_status("decoding", None)
            except Exception:
                pass

        text = ""
        try:
            samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(
                np.float32
            ) / 32768.0
            text = self.dictation_service.dictate(samples, self.SAMPLE_RATE)
            text = self._strip_stop_keyword(text)
            logger.info("听写结果: %r", text)
        except Exception as e:
            logger.error("听写 ASR 失败: %s", e, exc_info=True)

        if on_status:
            try:
                on_status("done" if text else "failed", text)
            except Exception:
                pass
        if on_text:
            try:
                on_text(text)
            except Exception:
                logger.exception("听写回调异常")

    @staticmethod
    def _strip_stop_keyword(text):
        """去除 ASR 结果中的"结束板书"指令词（含同音误识别变体）。

        "结束板书"是停止指令词，不该作为内容显示。原本只 strip 结尾，
        但 ASR 常加标点（如"...结束板书。"）破坏 endswith，
        且 partial 结果中可能出现在中间。直接全文 replace 最稳。
        """
        if not text:
            return text
        for variant in ("结束板书", "结束版书", "结束办书"):
            text = text.replace(variant, "")
        # 清理替换后可能多出的连续空白
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip("，。、 \t")

    def _handle_keyword(self, keyword):
        """处理检测到的关键词"""
        now = time.time()

        # 听写模式下：只响应 "结束板书"，忽略其余所有关键词
        if self._dictation_mode:
            if keyword == "结束板书":
                # KWS 阈值 0.25 偏低，正常说话中含"结束/出版/出书/今天结"等
                # 相近音节时会被误触发，导致用户没说完就被截断。两道保险：
                # (1) 听写开始 1.5s 内一律忽略 — 刚开口不会是结束指令
                # (2) 异步用 SenseVoice 复核最近 ~3s 音频，确认有"结束板书"才停
                elapsed = now - self._dictation_start_time
                if elapsed < self.MIN_DICTATION_BEFORE_STOP:
                    logger.info(
                        "'结束板书' 触发时听写仅 %.1fs，忽略（可能是 KWS 误触发）",
                        elapsed,
                    )
                    return
                self._last_keyword_time = now
                threading.Thread(
                    target=self._verify_and_finish_dictation,
                    args=(self._dictation_session_id,),
                    daemon=True,
                ).start()
            else:
                logger.debug("听写中忽略关键词: %s", keyword)
            return

        # 冷却期防抖
        if now - self._last_keyword_time < self._cooldown:
            logger.debug("关键词冷却中，忽略: %s", keyword)
            return

        self._last_keyword_time = now
        logger.info("语音指令: %s", keyword)

        # 更新状态
        self._status_text = f"语音: {keyword}"
        if self._status_callback:
            try:
                self._status_callback(keyword)
            except Exception:
                pass

        # 查找映射的 action
        action = VOICE_KEYWORD_TO_ACTION.get(keyword)
        if action and self.action_callback:
            self.action_callback(action)
        else:
            logger.debug("未映射的关键词: %s", keyword)

    # ------------------------------------------------------------------
    # 资源清理
    # ------------------------------------------------------------------

    def _cleanup_audio(self):
        """清理音频资源"""
        if self._audio_stream is not None:
            try:
                self._audio_stream.stop()
                self._audio_stream.close()
            except Exception:
                pass
            self._audio_stream = None

        # 清空队列
        if self._audio_queue is not None:
            while not self._audio_queue.empty():
                try:
                    self._audio_queue.get_nowait()
                except queue.Empty:
                    break
            self._audio_queue = None
