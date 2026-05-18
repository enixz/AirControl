"""
语音指令服务 — 基于 Sherpa-ONNX KWS 的离线关键词检测

架构：双引擎协同
- 引擎 A：Sherpa-ONNX KWS（离线，常驻）— 处理全部固定指令词
- 引擎 B：腾讯云实时 ASR（在线，按需）— 仅板书模式"打在屏幕上"触发

与现有手势系统共享 execute_action() 调度。
"""

import logging
import os
import queue
import threading
import time

import numpy as np

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
    # 全局
    "最小化助手": "minimize_assistant",
    "唤醒助手": "launch_voice_assistant",
    "切模式": "switch_mode",
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
    "打在屏幕上": "dictate_to_screen",
}

# 各模式可用的关键词（None 表示全部可用）
MODE_KEYWORDS = {
    "presentation": [
        "开始播放", "结束播放", "下一页", "上一页",
        "最小化助手", "唤醒助手", "切模式",
        "板书模式", "鼠标模式",
    ],
    "mouse": [
        "点一下", "双击", "右键",
        "最小化助手", "唤醒助手", "切模式",
        "板书模式", "演示模式",
    ],
    "draw": [
        "清屏", "打在屏幕上",
        "最小化助手", "唤醒助手", "切模式",
        "演示模式", "鼠标模式",
    ],
}


class VoiceCommandService:
    """语音指令服务 — KWS 常驻检测 + 按需在线 ASR"""

    SAMPLE_RATE = 16000
    CHUNK_DURATION_MS = 100  # 每次 KWS 处理 100ms 音频
    CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)  # 1600 samples

    def __init__(self, config, action_callback=None):
        """
        Args:
            config: ConfigManager 实例
            action_callback: callable(action_name: str) — 回调到 FloatingWindow.execute_action()
        """
        self.config = config
        self.action_callback = action_callback

        self._kws = None
        self._kws_stream = None
        self._audio_stream = None
        self._audio_queue = None
        self._running = False
        self._thread = None
        self._current_mode = None
        self._last_keyword_time = 0
        self._cooldown = 1.0  # 关键词触发冷却（秒）

        # 模型路径 — 基于 AirControl 项目根目录
        # 本文件位于 app/services/voice_command.py，向上两层即为项目根目录
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self._model_dir = os.path.join(base_dir, "models", "kws-zh-wenetspeech")
        self._keywords_dir = os.path.join(base_dir, "app", "voice_keywords")

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
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        self._cleanup_audio()
        self._kws = None
        self._kws_stream = None
        logger.info("语音指令服务已停止")

    def on_mode_changed(self, mode_name):
        """模式切换时更新关键词集"""
        self._current_mode = mode_name
        if self._kws is not None:
            self._load_keywords_for_mode(mode_name)
            # 切换关键词后需要重置 stream
            if self._kws_stream is not None:
                self._kws.reset_stream(self._kws_stream)

    def set_status_callback(self, callback):
        """设置状态回调（用于 UI 更新）"""
        self._status_callback = callback

    @property
    def is_running(self):
        return self._running

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
        self._generate_mode_keywords(mode)

        keywords_file = os.path.join(self._keywords_dir, "keywords_active.txt")
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
        """根据当前模式生成激活的关键词文件"""
        raw_file = os.path.join(self._keywords_dir, "keywords.txt")
        active_file = os.path.join(self._keywords_dir, "keywords_active.txt")

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

        with open(active_file, "w", encoding="utf-8") as f:
            for line in filtered_lines:
                f.write(line + "\n")

        logger.info("模式 %s 激活 %d 个关键词", mode_name, len(filtered_lines))

    def _load_keywords_for_mode(self, mode_name):
        """切换模式时重新生成关键词文件并重新加载"""
        self._generate_mode_keywords(mode_name)
        keywords_file = os.path.join(self._keywords_dir, "keywords_active.txt")

        # sherpa-onnx KWS 不支持运行时更换关键词文件，
        # 需要重新创建 KeywordSpotter 实例
        if self._kws is not None:
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
            logger.info("KWS 关键词已更新为模式: %s", mode_name)

    # ------------------------------------------------------------------
    # 检测循环
    # ------------------------------------------------------------------

    def _detection_loop(self):
        """核心检测循环（独立线程）"""
        logger.info("KWS 检测循环已启动")

        while self._running:
            try:
                # 从队列获取音频数据（超时 200ms，避免线程卡死）
                try:
                    audio_data = self._audio_queue.get(timeout=0.2)
                except queue.Empty:
                    continue

                # int16 → float32 归一化
                samples = np.frombuffer(audio_data, dtype=np.int16).astype(
                    np.float32
                ) / 32768.0

                # 送入 KWS
                self._kws_stream.accept_waveform(self.SAMPLE_RATE, samples)

                # 尝试解码
                while self._kws.is_ready(self._kws_stream):
                    self._kws.decode_stream(self._kws_stream)
                    result = self._kws.get_result(self._kws_stream)

                    if result:
                        self._handle_keyword(result)
                        # 检测到关键词后必须重置 stream，防止持续触发
                        self._kws.reset_stream(self._kws_stream)
                        break

            except IOError as e:
                # 麦克风读取错误（可能是设备断开）
                logger.warning("麦克风读取错误: %s", e)
                time.sleep(0.1)
            except Exception as e:
                logger.error("KWS 检测循环异常: %s", e, exc_info=True)
                time.sleep(0.1)

        logger.info("KWS 检测循环已退出")

    def _handle_keyword(self, keyword):
        """处理检测到的关键词"""
        now = time.time()

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
