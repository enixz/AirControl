import logging
import os
import threading
import time
import cv2
import winsound
from PyQt6.QtCore import QObject, pyqtSignal

from config_manager import ConfigManager
from mode_manager import ModeManager
from modes import DrawMode, MouseMode, PresentationMode
from services.camera import CameraService
from services.gesture_recognizer import GestureRecognizer
from services.hand_tracker_factory import create_hand_tracker
from services.inference_worker import InferenceWorker
from services.mouse_controller import MouseController
from services.ppt_controller import PptController
from services.voice_assistant import VoiceAssistantService
from services.voice_command import VoiceCommandService
from services.voice_dictation import VoiceDictationService

logger = logging.getLogger(__name__)


class AirControlOrchestrator(QObject):
    # Public signals to communicate with the UI view
    frame_processed = pyqtSignal(object, list, list, str)  # frame (ndarray), landmarks, gestures, current_gesture
    status_updated = pyqtSignal(str, tuple)  # text, RGB color tuple
    voice_status_updated = pyqtSignal(str)  # status text to be displayed
    fps_updated = pyqtSignal(float)
    mode_changed = pyqtSignal(str)  # mode name (e.g. "presentation")
    dictation_status_signal = pyqtSignal(str, object)   # phase, payload
    dictation_text_signal = pyqtSignal(str, object)     # text, anchor_pos
    dictation_partial_signal = pyqtSignal(str)          # partial transcription
    caption_full = pyqtSignal()
    
    # Signals for window control requested by actions
    minimize_requested = pyqtSignal()
    restore_requested = pyqtSignal()

    # Internal voice & dictation signals (marshalled from backend threads to main thread)
    _voice_action_signal = pyqtSignal(str)
    _dictation_status_signal = pyqtSignal(str, object)   # phase, payload
    _dictation_text_signal = pyqtSignal(str, object)     # text, anchor_pos
    _dictation_partial_signal = pyqtSignal(str)          # partial transcription
    _tracker_ready_signal = pyqtSignal(object, object, int, str)

    def __init__(
        self, overlay, cursor_overlay, toolbar, hwnd=None, parent=None, config=None
    ):
        super().__init__(parent)
        self.overlay = overlay
        self.cursor_overlay = cursor_overlay
        self.toolbar = toolbar
        self.aircontrol_hwnd = hwnd

        self.config = config or ConfigManager()
        if parent is not None and hasattr(parent, 'mouse'):
            self.mouse = parent.mouse
        else:
            self.mouse = MouseController(
                sensitivity=self.config.get("mouse_sensitivity"),
                edge_enabled=self.config.get("edge_acceleration_enabled"),
                edge_strength=self.config.get("edge_acceleration_strength"),
                edge_y_canvas=self.config.get("edge_y_canvas_enabled"),
                edge_y_dz_bottom=self.config.get("edge_y_canvas_deadzone_bottom"),
                edge_y_dz_top=self.config.get("edge_y_canvas_deadzone_top"),
            )

        # Connect internal threads signals to safe handler methods in UI thread
        self._voice_action_signal.connect(self.execute_action)
        self._dictation_status_signal.connect(self._on_dictation_status)
        self._dictation_text_signal.connect(self._on_dictation_text)
        self._dictation_partial_signal.connect(self._on_dictation_partial)
        self._tracker_ready_signal.connect(self._on_tracker_ready)
        self._tracker_request_id = 0
        self._closing = False

        self.status_text = "Ready"
        self.status_color = (0, 255, 0)
        self.status_timer = 0.0

        self._voice_keyword_flash = None
        self._voice_keyword_time = 0.0

        self.init_services()
        self._init_modes()
        
        # Enter default mode (no sound on init)
        self._set_mode(self.config.get("interaction_mode"), sound=False)

    def set_hwnd(self, hwnd):
        self.aircontrol_hwnd = hwnd
        if hasattr(self, 'voice_assistant'):
            self.voice_assistant.aircontrol_hwnd = hwnd

    def init_services(self):
        self.camera = CameraService(
            camera_index=self.config.get("camera_index"),
            width=self.config.get("camera_width"),
            height=self.config.get("camera_height"),
            force_mjpeg=self.config.get("camera_force_mjpeg") is not False,
            min_fps=self.config.get("camera_min_fps") or 20,
        )
        self.camera.start()

        self._tracker_config_signature = self._tracker_signature()
        self.tracker = self._create_tracker()

        self.recognizer = GestureRecognizer(
            cooldown=self.config.get("cooldown"),
            swipe_threshold=self.config.get("swipe_threshold"),
        )
        self.ppt = PptController(target_app=self.config.get("target_app"), config=self.config)
        
        self.voice_assistant = VoiceAssistantService(
            assistant=self.config.get("voice_assistant")
        )
        if self.aircontrol_hwnd:
            self.voice_assistant.aircontrol_hwnd = self.aircontrol_hwnd

        # Voice dictation service (SenseVoice-Small offline ASR)
        self.voice_dictation = VoiceDictationService(self.config)
        if not self.voice_dictation.is_available():
            logger.info("SenseVoice 模型未就绪，听写功能不可用")

        # Voice command service (KWS offline keyword spotting)
        self.voice_command = VoiceCommandService(
            self.config,
            action_callback=self._voice_action_signal.emit,
            dictation_service=self.voice_dictation,
        )
        self.voice_command.set_status_callback(self._on_voice_keyword_detected)
        if self.config.get("voice_command_enabled") is not False:
            try:
                self.voice_command.start()
            except Exception as e:
                logger.warning("语音指令服务启动失败: %s", e)
        self._voice_kws_signature = self._current_voice_kws_signature()

        # Self startup diagnostic check
        self._run_startup_check()

        # 原始帧录制器（默认关；config record_raw_video=true 时录到 raw_capture/，
        # 供 replay_video.py 离线回放、用同一段真实画面客观对比检测质量）
        self.frame_recorder = self._make_frame_recorder()

        # Start Inference thread worker
        self.inference_worker = InferenceWorker(
            self.camera, self.tracker, max_fps=30,
            debug_overlay=bool(self.config.get("debug_overlay")),
            frame_recorder=self.frame_recorder,
        )
        self.inference_worker.frame_ready.connect(self._on_frame_ready)
        self.inference_worker.error_occurred.connect(self._on_inference_error)
        self.inference_worker.fps_updated.connect(self._on_fps_updated)
        self.inference_worker.performance_updated.connect(
            self._on_performance_updated
        )
        self.inference_worker.start()

    def _tracker_signature(self):
        return (
            os.environ.get("AIRCONTROL_ENGINE")
            or self.config.get("detection_engine", "mediapipe"),
            self.config.get("model_type"),
            self.config.get("dominant_hand", "Right"),
            self.config.get("hand_detection_confidence", 0.6),
            self.config.get("hand_presence_confidence", 0.5),
            self.config.get("hand_tracking_confidence", 0.5),
            self.config.get("hand_smoothing_min_cutoff", 0.5),
            self.config.get("hand_smoothing_beta", 0.015),
        )

    def _create_tracker(self, signature=None):
        signature = signature or self._tracker_signature()
        return create_hand_tracker(
            engine=signature[0],
            max_num_hands=2,
            min_detection_confidence=signature[3],
            min_presence_confidence=signature[4],
            min_tracking_confidence=signature[5],
            preferred_model_type=signature[1],
            dominant_hand=signature[2],
            config=self.config,
        )

    def _current_voice_kws_signature(self):
        return (self.config.get("voice_command_threshold", 0.25),)

    def _run_startup_check(self):
        """Startup diagnostic checks logic."""
        lines = ["", "=" * 60, "AirControl 启动自检", "=" * 60]

        # Camera status
        try:
            cap = self.camera.cap
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            fourcc_v = int(cap.get(cv2.CAP_PROP_FOURCC))
            fourcc_s = "".join(chr((fourcc_v >> 8 * i) & 0xFF) for i in range(4))
            lines.append(f"[OK]   摄像头 #{self.camera.camera_index}: {w}x{h}@{fps:.0f}fps ({fourcc_s})")
        except Exception as e:
            lines.append(f"[BAD]  摄像头不可用: {e}")

        # Hand tracking model
        try:
            model_name = os.path.basename(self.tracker.model_path)
            lines.append(f"[OK]   手部模型: {model_name}  (dominant={self.tracker.dominant_hand})")
        except Exception as e:
            lines.append(f"[BAD]  手部模型加载失败: {e}")

        # PPT Controller
        try:
            from services.ppt_controller import find_executable
            wpp = find_executable("wpp")
            ppt = find_executable("powerpnt")
            if wpp:
                lines.append(f"[OK]   WPS 演示: {wpp}")
            else:
                lines.append("[--]   WPS 演示未安装（演示模式仅支持 PowerPoint）")
            if ppt:
                lines.append(f"[OK]   PowerPoint: {ppt}")
            else:
                lines.append("[--]   PowerPoint 未安装")
            if not wpp and not ppt:
                lines.append("[BAD]  WPS 和 PPT 都没找到，演示模式不可用")
        except Exception as e:
            lines.append(f"[BAD]  演示控制器检查失败: {e}")

        # Voice dictation
        try:
            if self.voice_dictation.is_available():
                lines.append("[OK]   语音听写 SenseVoice 模型就绪")
            else:
                lines.append("[--]   语音听写不可用（缺 models/sense-voice/ 模型文件）")
        except Exception as e:
            lines.append(f"[BAD]  语音听写检查失败: {e}")

        # Voice command
        try:
            if bool(getattr(self.voice_command, "is_running", False)):
                lines.append("[OK]   语音指令 KWS 已启动")
            elif self.config.get("voice_command_enabled") is False:
                lines.append("[--]   语音指令已在 config 中关闭")
            else:
                lines.append("[BAD]  语音指令未启动（检查麦克风/模型）")
        except Exception as e:
            lines.append(f"[BAD]  语音指令检查失败: {e}")

        # Voice assistant
        assistant = self.config.get("voice_assistant") or "未配置"
        lines.append(f"[OK]   语音助手: {assistant}")

        lines.append("=" * 60)
        lines.append("")
        for line in lines:
            logger.info(line)

    def switch_camera(self, new_index):
        """Runtime hot-swapping of cameras."""
        try:
            old_index = self.camera.camera_index
        except Exception:
            old_index = self.config.get("camera_index") or 0
        if new_index == old_index:
            return True

        logger.info("切换摄像头: %d → %d", old_index, new_index)

        # 1. Stop active inference worker
        if hasattr(self, "inference_worker") and self.inference_worker is not None:
            try:
                self.inference_worker.stop()
            except Exception:
                logger.exception("停 InferenceWorker 时异常")

        # 2. Release old camera
        try:
            self.camera.release()
        except Exception:
            logger.exception("释放旧摄像头时异常")

        # 3. Try to start new camera service
        new_cam = CameraService(
            camera_index=new_index,
            width=self.config.get("camera_width"),
            height=self.config.get("camera_height"),
            force_mjpeg=self.config.get("camera_force_mjpeg") is not False,
            min_fps=self.config.get("camera_min_fps") or 20,
        )
        try:
            new_cam.start()
        except Exception:
            logger.exception("新摄像头 %d 启动失败，回滚到旧摄像头 %d", new_index, old_index)
            # Rollback
            try:
                self.camera = CameraService(
                    camera_index=old_index,
                    width=self.config.get("camera_width"),
                    height=self.config.get("camera_height"),
                    force_mjpeg=self.config.get("camera_force_mjpeg") is not False,
                    min_fps=self.config.get("camera_min_fps") or 20,
                )
                self.camera.start()
            except Exception:
                logger.exception("回滚旧摄像头 %d 也失败", old_index)
            self._restart_inference_worker()
            return False

        self.camera = new_cam
        self._restart_inference_worker()
        logger.info("摄像头已切换到 %d (%dx%d)", new_index, new_cam.width or 0, new_cam.height or 0)
        return True

    def _make_frame_recorder(self):
        """按 config 创建原始帧录制器；默认关闭、失败返回 None（绝不影响主流程）。"""
        if not self.config.get("record_raw_video"):
            return None
        try:
            from services.frame_recorder import FrameRecorder
            return FrameRecorder(
                max_frames=int(self.config.get("record_raw_max_frames") or 2000),
                max_seconds=float(self.config.get("record_raw_max_seconds") or 120.0),
            )
        except Exception as e:
            logger.warning("原始帧录制初始化失败: %s", e)
            return None

    def _restart_inference_worker(self):
        new_worker = InferenceWorker(
            self.camera, self.tracker, max_fps=30,
            debug_overlay=bool(self.config.get("debug_overlay")),
            frame_recorder=getattr(self, "frame_recorder", None),
        )
        new_worker.frame_ready.connect(self._on_frame_ready)
        new_worker.error_occurred.connect(self._on_inference_error)
        new_worker.fps_updated.connect(self._on_fps_updated)
        new_worker.performance_updated.connect(self._on_performance_updated)
        new_worker.start()
        self.inference_worker = new_worker

    def apply_config(self):
        self.recognizer.cooldown = self.config.get("cooldown")
        self.ppt.set_target_app(self.config.get("target_app"))
        self.mouse.set_sensitivity(self.config.get("mouse_sensitivity"))
        self.mouse.set_edge_acceleration(
            self.config.get("edge_acceleration_enabled"),
            self.config.get("edge_acceleration_strength"),
            y_canvas=self.config.get("edge_y_canvas_enabled"),
            y_dz_bottom=self.config.get("edge_y_canvas_deadzone_bottom"),
            y_dz_top=self.config.get("edge_y_canvas_deadzone_top"),
        )
        self.overlay.set_pen_width(self.config.get("pen_width"))
        self.overlay.set_pen_auto_scale(
            self.config.get("pen_width_auto_scale") is not False
        )
        self.voice_assistant.set_assistant(self.config.get("voice_assistant"))
        if hasattr(self, "inference_worker"):
            self.inference_worker.set_debug_overlay(
                bool(self.config.get("debug_overlay"))
            )

        signature = self._tracker_signature()
        if signature != self._tracker_config_signature:
            self._request_tracker_rebuild(signature)

        voice_signature = self._current_voice_kws_signature()
        if voice_signature != self._voice_kws_signature:
            self._voice_kws_signature = voice_signature
            self.voice_command.request_reload()
        
        new_mode = self.config.get("interaction_mode")
        if new_mode != self.mode_manager.current_mode_name:
            self._set_mode(new_mode)
        logger.info("配置已更新: 模式 -> %s / 目标软件 -> %s", new_mode, self.ppt.target_app)

    def _request_tracker_rebuild(self, signature):
        self._tracker_request_id += 1
        request_id = self._tracker_request_id

        def build():
            tracker = None
            error = ""
            try:
                tracker = self._create_tracker(signature)
            except Exception as exc:
                logger.exception("后台重建 tracker 失败")
                error = str(exc)
            if self._closing and tracker is not None:
                close = getattr(tracker, "close", None)
                if callable(close):
                    close()
                return
            self._tracker_ready_signal.emit(
                tracker, signature, request_id, error
            )

        threading.Thread(
            target=build,
            name="TrackerBuildWorker",
            daemon=True,
        ).start()

    def _on_tracker_ready(self, tracker, signature, request_id, error):
        if self._closing or request_id != self._tracker_request_id:
            close = getattr(tracker, "close", None)
            if callable(close):
                close()
            return
        if tracker is None:
            self.status_text = f"模型更新失败: {error}"
            self.status_color = (255, 0, 0)
            self.status_updated.emit(self.status_text, self.status_color)
            return
        self.tracker = tracker
        self._tracker_config_signature = signature
        self.inference_worker.update_tracker(tracker)
        logger.info("tracker 已在后台完成更新")

    # ------------------------------------------------------------------
    # Modes System
    # ------------------------------------------------------------------

    def _init_modes(self):
        self.modes = {
            "presentation": PresentationMode(
                self.config, self.recognizer, self.mouse,
                self.overlay, self.cursor_overlay, self.toolbar, self.ppt,
            ),
            "mouse": MouseMode(
                self.config, self.recognizer, self.mouse,
                self.overlay, self.cursor_overlay, self.toolbar, self.ppt,
            ),
            "draw": DrawMode(
                self.config, self.recognizer, self.mouse,
                self.overlay, self.cursor_overlay, self.toolbar, self.ppt,
            ),
        }
        self.mode_manager = ModeManager(self.modes, self.config, self.recognizer)

    def _set_mode(self, mode_name, sound=True):
        self.mode_manager.switch_to(mode_name)
        if hasattr(self, 'voice_command') and self.voice_command.is_running:
            self.voice_command.on_mode_changed(mode_name)
        if sound:
            try:
                winsound.PlaySound("SystemAsterisk", winsound.SND_ALIAS | winsound.SND_ASYNC)
            except RuntimeError:
                winsound.MessageBeep(winsound.MB_ICONASTERISK)
        
        self.status_text = f"已切换到{self._mode_name_zh()}"
        self.status_color = (0, 255, 255)
        self.status_timer = time.time()
        self.status_updated.emit(self.status_text, self.status_color)
        self.mode_changed.emit(mode_name)

    def _mode_name_zh(self):
        mode = self.mode_manager.current_mode_name
        return {
            "presentation": "演示模式",
            "mouse": "鼠标模式",
            "draw": "板书模式",
        }.get(mode, "未知模式")

    # ------------------------------------------------------------------
    # Inference Callbacks & Processing
    # ------------------------------------------------------------------

    def _on_frame_ready(self, frame, hands_landmarks, hands_gestures):
        worker = self.sender()
        try:
            self._process_frame_results(frame, hands_landmarks, hands_gestures)
        except Exception as e:
            logger.error("_on_frame_ready error: %s", e, exc_info=True)
            self.status_text = f"error: {e}"
            self.status_color = (255, 0, 0)
            self.status_timer = time.time()
            self.status_updated.emit(self.status_text, self.status_color)
        finally:
            if worker is None:
                worker = getattr(self, "inference_worker", None)
            mark_consumed = getattr(worker, "mark_result_consumed", None)
            if callable(mark_consumed):
                mark_consumed()

    def _on_inference_error(self, error_msg):
        logger.error("推理错误: %s", error_msg)
        self.status_text = f"推理错误: {error_msg}"
        self.status_color = (255, 0, 0)
        self.status_timer = time.time()
        self.status_updated.emit(self.status_text, self.status_color)

    def _on_fps_updated(self, fps):
        self.fps_updated.emit(fps)
        # 节流（每5秒）记录实际推理帧率到 gesture.log，便于诊断"体感帧率不高"，
        # 并能看出远距离 crop-zoom + ESPCN 超分是否在拖慢帧率。
        now = time.time()
        if now - getattr(self, "_last_fps_log_time", 0.0) >= 5.0:
            self._last_fps_log_time = now
            logging.getLogger("gesture").info(
                "[FPS] 推理帧率 %.1f | 模式=%s", fps,
                self.mode_manager.current_mode_name,
            )

    def _on_performance_updated(self, metrics):
        logging.getLogger("gesture").info(
            "[PERF] capture p50/p95=%.1f/%.1fms | queue=%.1f/%.1fms | "
            "inference=%.1f/%.1fms | total=%.1f/%.1fms",
            metrics.get("capture_p50_ms", 0.0),
            metrics.get("capture_p95_ms", 0.0),
            metrics.get("queue_p50_ms", 0.0),
            metrics.get("queue_p95_ms", 0.0),
            metrics.get("inference_p50_ms", 0.0),
            metrics.get("inference_p95_ms", 0.0),
            metrics.get("total_p50_ms", 0.0),
            metrics.get("total_p95_ms", 0.0),
        )

    def _process_frame_results(self, frame, hands_landmarks, hands_gestures):
        frame_h, frame_w = frame.shape[:2]

        # The global switch recognizer must always observe real frames. Clearing
        # them first made the post-switch protection window look like a release
        # and allowed one continuous hold to switch through multiple modes.
        switched = self.mode_manager.maybe_switch_by_gesture(hands_landmarks, hands_gestures, frame_w)

        gesture = "NONE"
        if switched:
            self._set_mode(self.mode_manager.current_mode_name)
            gesture = "MODE_SWITCH"
        elif self.mode_manager.is_switch_candidate:
            # Do not let a global ILY hold become a mode-specific action. This
            # also prevents an accidental pen-down while switching from draw.
            gesture = "MODE_SWITCH_HOLD"
        else:
            mode_landmarks = hands_landmarks
            mode_gestures = hands_gestures
            if time.time() - self.mode_manager.last_mode_switch_time < 1.0:
                mode_landmarks = []
                mode_gestures = []
            result = self.mode_manager.handle(
                mode_landmarks, mode_gestures, frame_w, frame_h
            )
            gesture = result.gesture
            if result.status_text:
                self.status_text = result.status_text
                self.status_color = result.status_color
                self.status_timer = time.time()
                self.status_updated.emit(self.status_text, self.status_color)
            if result.action:
                self.execute_action(result.action)

        if time.time() - self.status_timer > 1.0:
            self.status_text = "准备就绪" if hands_landmarks else "未检测到手"
            self.status_color = (0, 255, 0) if hands_landmarks else (0, 0, 255)
            self.status_updated.emit(self.status_text, self.status_color)

        if gesture == "COOLDOWN":
            cv2.putText(
                frame, "Cooldown...", (10, 220),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2
            )

        # Update voice keywords state machine transitions
        dictating = (
            hasattr(self, "voice_command")
            and self.voice_command is not None
            and self.voice_command.is_dictating
        )
        if not dictating:
            if self._voice_keyword_flash and time.time() - self._voice_keyword_time < 2.0:
                self.voice_status_updated.emit(f"🎤 {self._voice_keyword_flash}")
            else:
                self._update_voice_label_text()

        # Emit the processed frame to the UI view layer
        self.frame_processed.emit(frame, hands_landmarks, hands_gestures, gesture)

    def _update_voice_label_text(self):
        if hasattr(self, 'voice_command') and self.voice_command.is_running:
            self.voice_status_updated.emit("🎤 语音开")
        else:
            self.voice_status_updated.emit("语音关")

    def _on_voice_keyword_detected(self, keyword):
        self._voice_keyword_flash = keyword
        self._voice_keyword_time = time.time()

    # ------------------------------------------------------------------
    # Voice Dictation Functions
    # ------------------------------------------------------------------

    def _start_voice_dictation(self):
        if not hasattr(self, "voice_command") or not self.voice_command.is_running:
            logger.warning("语音服务未运行，无法听写")
            return

        current_mode = self.mode_manager.current_mode_name
        if current_mode != "draw":
            logger.info("非板书模式（%s）忽略听写请求", current_mode)
            return

        anchor_pos = None
        if self.overlay.cursor_pos is not None:
            anchor_pos = (self.overlay.cursor_pos.x(), self.overlay.cursor_pos.y())

        # Callbacks routed through safe Qt signals
        def on_status(phase, payload):
            self._dictation_status_signal.emit(phase, payload)

        def on_text(text):
            self._dictation_text_signal.emit(text, anchor_pos)

        def on_partial(text):
            self._dictation_partial_signal.emit(text or "")

        ok = self.voice_command.start_dictation(
            on_text=on_text,
            on_status=on_status,
            on_partial=on_partial,
        )
        if not ok:
            try:
                winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
            except Exception:
                pass
            self.voice_status_updated.emit("⚠️ 听写不可用（模型未安装？）")

    def _on_dictation_status(self, phase, payload):
        self.dictation_status_signal.emit(phase, payload)
        if phase == "started":
            self.voice_status_updated.emit('🎙️ 听写中（说"结束板书"停止）')
        elif phase == "tick":
            elapsed = payload if isinstance(payload, (int, float)) else 0.0
            self.voice_status_updated.emit(f"🎙️ 听写中... {elapsed:.0f}s")
        elif phase == "decoding":
            self.voice_status_updated.emit("🧠 识别中...")
        elif phase == "failed":
            reason = payload if isinstance(payload, str) else "no_text"
            self.voice_status_updated.emit(f"⚠️ 听写失败: {reason}")

    def _on_dictation_text(self, text, anchor_pos):
        self.dictation_text_signal.emit(text, anchor_pos)
        self._render_dictation_text(text, anchor_pos)

    def _on_dictation_partial(self, text):
        self.dictation_partial_signal.emit(text)
        if hasattr(self, "overlay") and self.overlay is not None:
            self.overlay.set_dictation_caption(text)

    def _on_caption_full(self):
        if hasattr(self, "voice_command") and self.voice_command.is_dictating:
            logger.info("字幕已写满屏幕，自动停止听写")
            self.voice_command.stop_dictation()
            self.caption_full.emit()

    def _render_dictation_text(self, text, anchor_pos):
        if hasattr(self, "overlay") and self.overlay is not None:
            self.overlay.clear_dictation_caption()
        if not text:
            self.voice_status_updated.emit("⚠️ 没听清，请再试一次")
            return
        x = y = None
        if anchor_pos is not None:
            x, y = anchor_pos
        try:
            self.overlay.draw_text(text, x=x, y=y)
        except Exception as e:
            logger.error("写文字到画布失败: %s", e, exc_info=True)
            self.voice_status_updated.emit("⚠️ 渲染失败")
            return
        self.voice_status_updated.emit(f"✍️ {text[:20]}")

    # ------------------------------------------------------------------
    # Action Dispatcher
    # ------------------------------------------------------------------

    def execute_action(self, action_name):
        if action_name == "next_slide":
            self.ppt.next_slide()
        elif action_name == "prev_slide":
            self.ppt.prev_slide()
        elif action_name == "start_presentation":
            self.ppt.start_presentation()
        elif action_name == "end_presentation":
            self.ppt.end_presentation()
        elif action_name == "switch_app":
            self.ppt.switch_app()
        elif action_name == "launch_voice_assistant":
            threading.Thread(target=self.voice_assistant.activate, daemon=True).start()
        elif action_name == "hang_up_voice_assistant":
            threading.Thread(target=self.voice_assistant.hang_up, daemon=True).start()
            
        # UI/Window Action requests
        elif action_name == "minimize_assistant":
            self.minimize_requested.emit()
        elif action_name == "restore_assistant":
            self.restore_requested.emit()
            
        # Mouse Actions
        elif action_name == "left_click":
            self.mouse.left_click()
        elif action_name == "double_click":
            self.mouse.double_click()
        elif action_name == "right_click":
            self.mouse.right_click()
            
        # Drawing / Overlay Actions
        elif action_name == "clear_canvas":
            self.overlay.clear_canvas()
            try:
                winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS | winsound.SND_ASYNC)
            except RuntimeError:
                winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
        elif action_name == "start_dictation":
            self._start_voice_dictation()
        elif action_name == "stop_dictation":
            if hasattr(self, "voice_command"):
                self.voice_command.stop_dictation()
        elif action_name == "switch_to_draw":
            self._set_mode("draw")
        elif action_name == "switch_to_mouse":
            self._set_mode("mouse")
        elif action_name == "switch_to_presentation":
            self._set_mode("presentation")
        elif action_name == "toggle_shape_correction":
            if self.mode_manager.current_mode_name == "draw":
                enabled = self.overlay.toggle_shape_correction()
                self.toolbar.set_shape_correction(enabled)

    def close(self):
        """Resource release when window is closed."""
        self._closing = True
        self._tracker_request_id += 1
        if self.mode_manager.current_mode:
            self.mode_manager.current_mode.on_exit()

        if hasattr(self, 'inference_worker'):
            try:
                self.inference_worker.frame_ready.disconnect(self._on_frame_ready)
            except (TypeError, RuntimeError):
                pass
            self.inference_worker.stop()

        if hasattr(self, 'voice_command'):
            self.voice_command.stop()

        self.camera.release()
        close_tracker = getattr(self.tracker, "close", None)
        if callable(close_tracker):
            try:
                close_tracker()
            except Exception:
                logger.exception("关闭 tracker 失败")
        if getattr(self, "frame_recorder", None) is not None:
            self.frame_recorder.close()
