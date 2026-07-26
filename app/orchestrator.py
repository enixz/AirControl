import logging
import math
import os
import threading
import time
import winsound

import cv2
import numpy as np
from config_manager import ConfigManager
from mode_manager import ModeManager
from modes import MODE_NAME_ZH, DrawMode, MouseMode, PresentationMode
from PyQt6.QtCore import QObject, pyqtSignal
from services.camera import CameraService, list_available_cameras
from services.engine_auto_switcher import (
    ENGINE_HAGRID_YOLO,
    STATE_CAPTURE,
    STATE_FAR_TRACK,
    EngineAutoSwitcher,
)

# 远距 CAPTURE 态可选捕获引擎（在 ENGINE_HAGRID_YOLO 基础上扩展）
ENGINE_PERSON_POSE = "person_pose_hand"
_CAPTURE_ENGINES = (ENGINE_HAGRID_YOLO, ENGINE_PERSON_POSE)
from services.geometric_classifier import WeightedVoteClassifier
from services.gesture_recognizer import GestureRecognizer
from services.hand_tracker_factory import create_hand_tracker
from services.inference_worker import InferenceWorker
from services.mouse_controller import MouseController
from services.ppt_controller import PptController
from services.temporal_voter import TemporalGestureVoter
from services.voice_assistant import VoiceAssistantService
from services.voice_command import VoiceCommandService
from services.voice_dictation import VoiceDictationService

logger = logging.getLogger(__name__)


def choose_camera_device(requested_index, available):
    """Return (index, backend, used_fallback) from verified probe results."""
    available_by_index = {camera["index"]: camera for camera in available}
    requested = available_by_index.get(requested_index)
    if requested is not None:
        return requested_index, requested.get("backend"), False
    if available:
        fallback = available[0]
        return fallback["index"], fallback.get("backend"), True
    return requested_index, None, False


def choose_startup_resolution(width, height):
    """Use an explicit saved resolution or a fast 720p startup default."""
    try:
        width = int(width) if width is not None else 0
        height = int(height) if height is not None else 0
    except (TypeError, ValueError):
        width = height = 0
    if width > 0 and height > 0:
        return width, height
    return 1280, 720


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
        self, overlay, cursor_overlay, toolbar, hwnd=None, parent=None, config=None,
        mouse=None,
    ):
        super().__init__(parent)
        self.overlay = overlay
        self.cursor_overlay = cursor_overlay
        self.toolbar = toolbar
        self.aircontrol_hwnd = hwnd

        self.config = config or ConfigManager()
        if mouse is not None:
            self.mouse = mouse
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
        self._closing_event = threading.Event()
        self._background_threads = set()
        self._background_threads_lock = threading.Lock()

        self.status_text = "Ready"
        self.status_color = (0, 255, 0)
        self.status_timer = 0.0

        self._voice_keyword_flash = None
        self._voice_keyword_time = 0.0

        # 远距引擎自动切换（config engine_auto_switch，默认关闭）：
        # 三态闭环 NEAR/CAPTURE/FAR_TRACK（见 services/engine_auto_switcher.py）。
        # _engine_override 是 FSM 的运行时引擎覆盖，只存在内存里、不写 config.json；
        # 用户手动改 detection_engine 时清除覆盖并重置状态机（见 apply_config）。
        self._engine_override = None
        self._engine_switcher = EngineAutoSwitcher()
        self._fsm_far_track_active = False    # 当前 tracker 是否带 long_range 运行时覆盖
        self._engine_switch_pending = False   # FSM 发起的 tracker 重建是否在途
        self._pending_far_track_seed = False  # 下次重建落地后要播种 crop-zoom 交接 hint
        # 自动切换只在远距丢手时才需要 YOLO。把它放到后台预热，避免首次
        # NEAR→CAPTURE 时同步创建 ONNX + HandLandmarker 造成数秒空窗。
        # 预热实例未参与推理，取用后立即清空；关闭/禁用时统一释放。
        self._warmed_yolo_tracker = None
        self._warmed_yolo_signature = None
        self._yolo_prewarm_inflight = False
        self._tracker_prewarm_lock = threading.Lock()
        _configured_engine = self.config.get("detection_engine", "mediapipe")
        self._last_config_engine = (
            _configured_engine if isinstance(_configured_engine, str) else "mediapipe"
        )

        self.init_services()
        self._init_modes()

        # Enter default mode (no sound on init)
        self.set_mode(self.config.get("interaction_mode"), sound=False)

    def set_hwnd(self, hwnd):
        self.aircontrol_hwnd = hwnd
        if hasattr(self, 'voice_assistant'):
            self.voice_assistant.aircontrol_hwnd = hwnd

    def _start_background_thread(self, target, name, args=()):
        """Start and track a bounded-lifecycle helper owned by the orchestrator."""
        if self._closing:
            return None

        def run():
            try:
                target(*args)
            finally:
                with self._background_threads_lock:
                    self._background_threads.discard(threading.current_thread())

        thread = threading.Thread(target=run, name=name, daemon=True)
        with self._background_threads_lock:
            if self._closing:
                return None
            self._background_threads.add(thread)
            thread.start()
        return thread

    def _wait_for_background_threads(self, timeout_sec):
        """Wait for owned helpers without exceeding the caller's deadline."""
        deadline = time.monotonic() + max(0.0, float(timeout_sec))
        current = threading.current_thread()
        while True:
            with self._background_threads_lock:
                threads = [
                    thread for thread in self._background_threads
                    if thread is not current and thread.is_alive()
                ]
            if not threads:
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                logger.error(
                    "后台任务未能在关闭期限内退出: %s",
                    ", ".join(sorted(thread.name for thread in threads)),
                )
                return False
            threads[0].join(remaining)

    def _find_available_cameras(self, exclude_index=None):
        """枚举系统可用摄像头，返回完整列表。

        exclude_index: 已经由本进程打开、无法重复探测的当前摄像头索引。
        返回 [] 表示没有任何可用摄像头
        """
        try:
            return list_available_cameras(max_probe=4, exclude_index=exclude_index)
        except Exception:
            logger.exception("枚举摄像头失败")
            return []

    def init_services(self):
        requested_index = self.config.get("camera_index")
        startup_width, startup_height = choose_startup_resolution(
            self.config.get("camera_width"),
            self.config.get("camera_height"),
        )

        # 正常启动先直接打开偏好设备。CameraService 自身会做后端回退，因此无须
        # 为常见的单摄像头场景同步枚举 0..3；只有偏好设备真实启动失败时才做全量
        # 枚举并选择备用设备。
        chosen_index = requested_index
        chosen_backend = None
        self.camera = CameraService(
            camera_index=chosen_index,
            width=startup_width,
            height=startup_height,
            force_mjpeg=self.config.get("camera_force_mjpeg") is not False,
            min_fps=self.config.get("camera_min_fps") or 20,
            preferred_backend=chosen_backend,
        )
        self._camera_available = True
        try:
            self.camera.start()
        except Exception as e:
            logger.warning("偏好摄像头 %s 启动失败: %s", requested_index, e)
            self.camera.release()
            available = self._find_available_cameras()
            chosen_index, chosen_backend, used_fallback = choose_camera_device(
                requested_index,
                available,
            )
            if available:
                logger.warning(
                    "%s摄像头 %d（后端: %s）",
                    "自动回退到" if used_fallback else "按已验证后端重试",
                    chosen_index,
                    chosen_backend if chosen_backend is not None else "默认",
                )
                self.camera = CameraService(
                    camera_index=chosen_index,
                    width=startup_width,
                    height=startup_height,
                    force_mjpeg=self.config.get("camera_force_mjpeg") is not False,
                    min_fps=self.config.get("camera_min_fps") or 20,
                    preferred_backend=chosen_backend,
                )
                try:
                    self.camera.start()
                except Exception as fallback_error:
                    logger.exception("备用摄像头启动失败: %s", fallback_error)
                    self.camera.release()
                    e = fallback_error
                    self._camera_available = False
            else:
                self._camera_available = False
            if not self._camera_available:
                self.status_text = f"摄像头不可用: {e}"
                self.status_color = (255, 0, 0)
                self.status_timer = time.time()
                self.status_updated.emit(self.status_text, self.status_color)

        self._tracker_config_signature = self._tracker_signature()
        self.tracker = self._create_tracker()
        self._configure_engine_switcher()

        self.recognizer = GestureRecognizer(
            cooldown=self.config.get("cooldown"),
            swipe_threshold=self.config.get("swipe_threshold"),
        )
        # Phase 3.2: pinch 双阈值滞回开关（默认关闭，保持旧版单阈值行为）
        self.recognizer.pinch_hysteresis_enabled = bool(
            self.config.get("pinch_hysteresis_enabled", False)
        )
        self.recognizer.pinch_exit_hysteresis_enabled = bool(
            self.config.get("pinch_exit_hysteresis_enabled", False)
        )
        # Phase 3.3: thumb_extended 旋转不变判定开关（默认关闭，新旧并存对照）
        self.recognizer.thumb_perp_ratio_enabled = bool(
            self.config.get("thumb_perp_ratio_enabled", False)
        )
        # 时序投票器：在 MediaPipe 单帧输出和规则识别器之间插入滞回层，
        # 稳定远距离下的手势标签抖动。CPU纯算法，<0.1ms/帧。
        # 阶段1：temporal_voter_enabled=False 时旁路（见 _stabilize_gestures），
        # 阶段2.11：默认关闭——其 FSM 给标签加延迟，回到老版基线；需要时手动开启做 A/B。
        self._temporal_voter_enabled = bool(self.config.get("temporal_voter_enabled", False))
        self.gesture_voter = TemporalGestureVoter(config=self.config)
        # 几何分类器：提取30+维连续特征，融合ML标签输出置信度。
        # 作为MediaPipe标签的交叉验证，远距离下几何特征比像素更稳定。
        self.geo_classifier = WeightedVoteClassifier(
            ml_weight=self.config.get("geo_ml_weight", 0.4),
            geo_weight=self.config.get("geo_geo_weight", 0.6),
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
        self.inference_worker.tracker_swapped.connect(self._on_tracker_swapped)
        self.inference_worker.start()
        self._schedule_yolo_prewarm()

    def _tracker_signature(self):
        return (
            # 引擎优先级：环境变量（自检/回放强制）> FSM 运行时覆盖 > config 配置
            os.environ.get("AIRCONTROL_ENGINE")
            or getattr(self, "_engine_override", None)
            or self.config.get("detection_engine", "mediapipe"),
            self.config.get("model_type"),
            self.config.get("dominant_hand", "Right"),
            self.config.get("hand_detection_confidence", 0.6),
            self.config.get("hand_presence_confidence", 0.5),
            self.config.get("hand_tracking_confidence", 0.5),
            self.config.get("hand_smoothing_min_cutoff", 0.5),
            self.config.get("hand_smoothing_beta", 0.015),
        )

    def _create_tracker(self, signature=None, config_overrides=None):
        signature = signature or self._tracker_signature()
        config = self.config
        if config_overrides:
            # FAR_TRACK 等运行时覆盖：把 ConfigManager 底层 dict 拷一份改键，
            # 只影响本次创建的 tracker，不写回、不落盘 config.json。
            merged = dict(getattr(self.config, "config", {}) or {})
            if merged:
                merged.update(config_overrides)
                config = merged
        max_num_hands = 2
        if signature[0] in (ENGINE_HAGRID_YOLO, ENGINE_PERSON_POSE):
            # YOLO/姿态 远距捕获只服务一个主控手；多手在该场景通常来自背景误检，
            # 既拖慢逐 crop Landmarker，又会阻塞 CAPTURE 的稳定单手判据。
            max_num_hands = int(config.get("yolo_max_hands", 1))
        return create_hand_tracker(
            engine=signature[0],
            max_num_hands=max_num_hands,
            min_detection_confidence=signature[3],
            min_presence_confidence=signature[4],
            min_tracking_confidence=signature[5],
            preferred_model_type=signature[1],
            dominant_hand=signature[2],
            config=config,
        )

    def _configure_engine_switcher(self):
        """按 config 刷新远距引擎自动切换状态机的参数（参数变更即重置计帧）。"""
        # detection_engine 是用户的显式引擎选择。自动闭环只以 mediapipe
        # 作为近距基线；用户手动选 hagrid_yolo 时不再反向覆盖成 MediaPipe。
        enabled = (
            self.config.get("engine_auto_switch", False) is True
            and self.config.get("detection_engine", "mediapipe") == "mediapipe"
        )
        self._engine_switcher.configure(
            enabled=enabled,
            no_hand_frames=self.config.get("engine_auto_switch_no_hand_frames", 60),
            hand_frames=self.config.get("engine_auto_switch_hand_frames", 30),
            near_frames=self.config.get("engine_auto_switch_near_frames", 90),
            near_bbox_ratio=self.config.get("engine_auto_switch_near_bbox_ratio", 0.04),
            cooldown_sec=self.config.get("engine_auto_switch_cooldown_sec", 5.0),
        )

    def _capture_engine(self):
        """CAPTURE 态使用的捕获引擎（config 热切换，默认 hagrid_yolo）。

        可配 engine_auto_switch_capture_engine=person_pose_hand 换成
        框人→姿态→手腕的远距/侧位捕获引擎。非法值回退 hagrid_yolo。
        """
        eng = str(self.config.get(
            "engine_auto_switch_capture_engine", ENGINE_HAGRID_YOLO)).strip().lower()
        return eng if eng in _CAPTURE_ENGINES else ENGINE_HAGRID_YOLO

    def _yolo_signature(self):
        """返回与当前配置匹配、但固定使用 CAPTURE 捕获引擎的 tracker 签名。"""
        signature = self._tracker_signature()
        return (self._capture_engine(), *signature[1:])

    @staticmethod
    def _close_tracker_safely(tracker, context):
        if tracker is None:
            return
        close = getattr(tracker, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                logger.exception("关闭 %s tracker 失败", context)

    def _take_warmed_yolo_tracker(self, signature):
        """取走签名匹配的预热 YOLO；不匹配的旧实例立即释放。"""
        stale = None
        with self._tracker_prewarm_lock:
            tracker = self._warmed_yolo_tracker
            warmed_signature = self._warmed_yolo_signature
            self._warmed_yolo_tracker = None
            self._warmed_yolo_signature = None
            if tracker is not None and warmed_signature == signature:
                logger.info("引擎自动切换: 复用已预热的 hagrid_yolo tracker")
                return tracker
            stale = tracker
        self._close_tracker_safely(stale, "过期预热")
        return None

    def _discard_warmed_yolo_tracker(self):
        """释放空闲的预热 YOLO。正在构建的线程会在完成时自行看到 _closing。"""
        with self._tracker_prewarm_lock:
            tracker = self._warmed_yolo_tracker
            self._warmed_yolo_tracker = None
            self._warmed_yolo_signature = None
        self._close_tracker_safely(tracker, "预热")

    def _schedule_yolo_prewarm(self, delay_sec=0.0):
        """后台预热下一个 CAPTURE 态所需的 YOLO tracker，绝不阻塞 UI/推理。"""
        if self._closing or not self._engine_switcher.enabled:
            return
        signature = self._yolo_signature()
        with self._tracker_prewarm_lock:
            if (
                self._warmed_yolo_signature == signature
                or self._yolo_prewarm_inflight
            ):
                return
            self._yolo_prewarm_inflight = True

        def build():
            tracker = None
            ready = False
            try:
                if delay_sec > 0 and self._closing_event.wait(delay_sec):
                    return
                if not self._closing and self._engine_switcher.enabled:
                    tracker = self._create_tracker(signature)
            except Exception:
                logger.exception("后台预热 hagrid_yolo tracker 失败")
            finally:
                stale = None
                with self._tracker_prewarm_lock:
                    self._yolo_prewarm_inflight = False
                    if (
                        tracker is not None
                        and not self._closing
                        and self._engine_switcher.enabled
                    ):
                        stale = self._warmed_yolo_tracker
                        self._warmed_yolo_tracker = tracker
                        self._warmed_yolo_signature = signature
                        tracker = None
                        ready = True
                self._close_tracker_safely(stale, "旧预热")
                self._close_tracker_safely(tracker, "未使用预热")
                if ready:
                    logger.info("引擎自动切换: hagrid_yolo 预热就绪")

        self._start_background_thread(build, "YoloTrackerPrewarm")

    def _maybe_auto_switch_engine(self, hands_landmarks, hand_bbox_ratio=0.0):
        """远距三态自动切换：逐帧喂检出结果给 FSM，按目标态执行迁移。

        仅主流程生效；AIRCONTROL_ENGINE 环境变量强制引擎时（自检/回放类场景）
        不介入。引擎重建复用既有 _request_tracker_rebuild 后台路径，平滑器与
        crop-zoom 状态由 migrate_state_from 迁移；覆盖只落内存，不写 config.json。
        """
        switcher = self._engine_switcher
        if not switcher.enabled or os.environ.get("AIRCONTROL_ENGINE"):
            return
        if not getattr(self, "_tracker_config_signature", None):
            return
        # FSM 发起的 tracker 重建仍在途：等其落地，避免重复发起
        if self._engine_switch_pending:
            return
        target = switcher.update(len(hands_landmarks or []), hand_bbox_ratio)
        if target is None:
            return
        logger.info("引擎自动切换: 目标态 %s（%s）", target, switcher.last_reason)
        self._apply_fsm_state(target)

    # 三态 → 用户可见状态提示（中文）。CAPTURE 文案随捕获引擎动态生成（见 _capture_status_text）。
    _FSM_STATE_STATUS = {
        STATE_FAR_TRACK: "远距跟踪中：ZOOM 已接管",
    }

    def _capture_status_text(self):
        if self._capture_engine() == ENGINE_PERSON_POSE:
            return "远距捕获中：person_pose_hand 框人拿手腕找手"
        return "远距捕获中：hagrid_yolo 全帧找手"

    def _apply_fsm_state(self, target):
        """执行 FSM 目标态：CAPTURE/FAR_TRACK 走引擎重建，NEAR 撤覆盖回配置档。"""
        if target == STATE_CAPTURE:
            self._engine_override = self._capture_engine()
            self._fsm_far_track_active = False
            self._request_fsm_rebuild(seed=False)
        elif target == STATE_FAR_TRACK:
            self._engine_override = "mediapipe"
            self._fsm_far_track_active = True
            # CAPTURE→FAR_TRACK 总是 yolo→mediapipe 的引擎变化，必走重建；
            # 落地时把 YOLO 最后的手部 hint 播种给新 MP tracker 的 crop-zoom。
            self._request_fsm_rebuild(seed=True)
        else:  # STATE_NEAR：撤掉 FSM 覆盖，回到用户配置档
            self._engine_override = None
            was_far = self._fsm_far_track_active
            self._fsm_far_track_active = False
            config_engine = self.config.get("detection_engine", "mediapipe")
            if self._tracker_config_signature[0] != config_engine:
                # 引擎要变（如 yolo → 配置引擎）：走重建
                self._request_fsm_rebuild(seed=False)
            elif was_far:
                # 引擎同为 mediapipe：运行时撤 long_range 覆盖即可，无需重建
                self._set_tracker_long_range(
                    bool(self.config.get("long_range_enabled", False))
                )
            self.status_text = "回到近距模式：mediapipe"
            self.status_color = (0, 255, 255)
            self.status_timer = time.time()
            self.status_updated.emit(self.status_text, self.status_color)

    def _request_fsm_rebuild(self, seed):
        """以 FSM 目标态发起 tracker 后台重建（FAR_TRACK 附带 long_range 覆盖）。"""
        signature = self._tracker_signature()
        overrides = {"long_range_enabled": True} if self._fsm_far_track_active else None
        if signature == self._tracker_config_signature and overrides is None and not seed:
            # 目标引擎已在运行（如用户手动 yolo + FSM 判 CAPTURE）：只更新簿记
            logger.info("引擎自动切换: 目标引擎 %s 已在运行，无需重建", signature[0])
            return
        self._pending_far_track_seed = bool(seed)
        self._engine_switch_pending = True
        prepared = None
        if signature[0] in _CAPTURE_ENGINES:
            prepared = self._take_warmed_yolo_tracker(signature)
        if prepared is not None:
            self._request_tracker_rebuild(
                signature, config_overrides=overrides, prepared_tracker=prepared,
            )
        else:
            self._request_tracker_rebuild(signature, config_overrides=overrides)
        if self._engine_switcher.state == STATE_CAPTURE:
            self.status_text = self._capture_status_text()
        else:
            self.status_text = self._FSM_STATE_STATUS.get(
                self._engine_switcher.state, "检测引擎自动切换中"
            )
        self.status_color = (0, 255, 255)
        self.status_timer = time.time()
        self.status_updated.emit(self.status_text, self.status_color)

    def _set_tracker_long_range(self, enabled):
        """运行时翻转当前 tracker 的 long_range 开关（在 worker 锁内，免重建）。"""
        tracker = getattr(self, "tracker", None)
        setter = getattr(tracker, "set_long_range_enabled", None)
        if not callable(setter):
            return
        worker = getattr(self, "inference_worker", None)
        lock = getattr(worker, "lock", None)
        if lock is not None:
            with lock:
                setter(enabled)
        else:
            setter(enabled)

    def _current_voice_kws_signature(self):
        return (self.config.get("voice_command_threshold", 0.25),)

    def _run_startup_check(self):
        """Startup diagnostic checks logic."""
        lines = ["", "=" * 60, "AirControl 启动自检", "=" * 60]

        # Camera status
        if not getattr(self, "_camera_available", True):
            lines.append("[BAD]  摄像头不可用（启动失败，请在设置中切换摄像头索引）")
        else:
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

    def switch_camera(self, new_index, camera_info=None):
        """Runtime hot-swapping of cameras."""
        try:
            old_index = self.camera.camera_index
        except Exception:
            old_index = self.config.get("camera_index") or 0
        if new_index == old_index:
            return True

        logger.info("切换摄像头: %d → %d", old_index, new_index)
        old_backend = getattr(self.camera, "_backend", None)
        old_width = getattr(self.camera, "width", None)
        old_height = getattr(self.camera, "height", None)

        # 1. Stop active inference worker
        if hasattr(self, "inference_worker") and self.inference_worker is not None:
            try:
                old_worker = self.inference_worker
                if old_worker.stop() is False:
                    logger.error(
                        "摄像头切换取消：旧采集线程未能安全退出，禁止释放其设备句柄"
                    )
                    try:
                        old_worker.finished.connect(
                            lambda worker=old_worker: (
                                self._recover_inference_after_stop_timeout(worker)
                            )
                        )
                    except (TypeError, RuntimeError):
                        pass
                    if not old_worker.isRunning():
                        self._recover_inference_after_stop_timeout(old_worker)
                    return False
                # A swap signal can still be queued in the UI event loop while
                # stop() has already completed. Use the worker's authoritative
                # active tracker when constructing the replacement worker.
                self.tracker = old_worker.tracker
            except Exception:
                logger.exception("停 InferenceWorker 时异常")
                return False

        # 2. Release old camera
        try:
            self.camera.release()
        except Exception:
            logger.exception("释放旧摄像头时异常")

        # 3. 设置窗口已在后台枚举并预探测候选设备，优先复用结果；
        # 仅程序化调用未提供 camera_info 时才同步补探测。
        camera_info = camera_info if isinstance(camera_info, dict) else {}
        if camera_info.get("index") != new_index:
            available = self._find_available_cameras(exclude_index=old_index)
            camera_info = next(
                (camera for camera in available if camera["index"] == new_index),
                {},
            )
        new_backend = camera_info.get("backend")
        new_width, new_height = choose_startup_resolution(
            camera_info.get("width") or self.config.get("camera_width"),
            camera_info.get("height") or self.config.get("camera_height"),
        )

        # 4. Try to start new camera service
        new_cam = CameraService(
            camera_index=new_index,
            width=new_width,
            height=new_height,
            force_mjpeg=self.config.get("camera_force_mjpeg") is not False,
            min_fps=self.config.get("camera_min_fps") or 20,
            preferred_backend=new_backend,
        )
        try:
            new_cam.start()
        except Exception:
            logger.exception("新摄像头 %d 启动失败，回滚到旧摄像头 %d", new_index, old_index)
            try:
                new_cam.release()
            except Exception:
                logger.exception("释放启动失败的新摄像头时异常")
            # Rollback
            try:
                self.camera = CameraService(
                    camera_index=old_index,
                    width=old_width,
                    height=old_height,
                    force_mjpeg=self.config.get("camera_force_mjpeg") is not False,
                    min_fps=self.config.get("camera_min_fps") or 20,
                    preferred_backend=old_backend,
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

    def _recover_inference_after_stop_timeout(self, worker):
        """Restart capture on the unchanged camera after a timed-out switch."""
        if self._closing or self.inference_worker is not worker:
            return
        if worker.isRunning():
            return
        logger.warning("旧摄像头采集线程已延迟退出，恢复原摄像头推理")
        self._restart_inference_worker()

    def _make_frame_recorder(self):
        """按 config 创建原始帧录制器；默认关闭、失败返回 None（绝不影响主流程）。"""
        if not self.config.get("record_raw_video"):
            return None
        try:
            from services.frame_recorder import FrameRecorder
            return FrameRecorder(
                max_frames=int(self.config.get("record_raw_max_frames") or 2000),
                max_seconds=float(self.config.get("record_raw_max_seconds") or 120.0),
                codec=str(self.config.get("record_raw_codec") or "mp4v"),
                record_truth=bool(self.config.get("record_truth_events", True)),
                truth_marker=str(self.config.get("record_truth_marker") or "space"),
            )
        except Exception as e:
            logger.warning("原始帧录制初始化失败: %s", e)
            return None

    def toggle_recording(self):
        """热切换原始帧录制：未录则开始，录中则停止并落盘。

        供开发测试专用——按 F8（全局热键，无需窗口焦点）或在悬浮窗聚焦时按
        F5，即可录一段真实画面到 raw_capture/<时间戳>/，然后用 replay_video.py
        离线回放做 A/B 对比，免去每次真人摆姿势实测。

        Returns:
            (now_recording: bool, path: str|None)  # True=刚开始录制, False=刚停止
        """
        current = getattr(self, "frame_recorder", None)
        if current is not None:
            # 正在录 → 停止
            # 先从 worker 脱离，阻止后续帧再取得旧 recorder；已经取得快照的
            # 帧由 FrameRecorder 的状态锁与 close sentinel 安全收尾。
            self.frame_recorder = None
            if self.inference_worker is not None:
                self.inference_worker.set_frame_recorder(None)
            try:
                current.close()
                path = getattr(current, "dir", None)
            except Exception:
                logger.exception("停止录制失败")
                path = None
            logger.info("F5 录制已停止 -> %s", path)
            return False, path

        # 未录 → 开始
        try:
            from services.frame_recorder import FrameRecorder
            recorder = FrameRecorder(
                max_frames=int(self.config.get("record_raw_max_frames") or 2000),
                max_seconds=float(self.config.get("record_raw_max_seconds") or 120.0),
                codec=str(self.config.get("record_raw_codec") or "mp4v"),
                record_truth=bool(self.config.get("record_truth_events", True)),
                truth_marker=str(self.config.get("record_truth_marker") or "space"),
            )
        except Exception:
            logger.exception("启动录制失败")
            return False, None
        self.frame_recorder = recorder
        if self.inference_worker is not None:
            self.inference_worker.set_frame_recorder(recorder)
        logger.info("F5 录制已开始 -> %s", recorder.dir)
        return True, recorder.dir

    def is_recording(self):
        """当前是否在录制原始帧（供 UI 显示状态）。"""
        return getattr(self, "frame_recorder", None) is not None

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
        new_worker.tracker_swapped.connect(self._on_tracker_swapped)
        new_worker.start()
        self.inference_worker = new_worker

    def apply_config(self):
        self.recognizer.cooldown = self.config.get("cooldown")
        self.recognizer.pinch_hysteresis_enabled = bool(
            self.config.get("pinch_hysteresis_enabled", False)
        )
        self.recognizer.pinch_exit_hysteresis_enabled = bool(
            self.config.get("pinch_exit_hysteresis_enabled", False)
        )
        self.recognizer.thumb_perp_ratio_enabled = bool(
            self.config.get("thumb_perp_ratio_enabled", False)
        )
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

        # --- 远距引擎自动切换的调和（必须先于 tracker 签名比较）---
        # 用户手动改 detection_engine：尊重手动选择，清掉 FSM 运行时覆盖并重置计帧。
        config_engine = self.config.get("detection_engine", "mediapipe")
        if config_engine != self._last_config_engine:
            logger.info(
                "检测引擎被手动切换: %s → %s，自动切换状态机已重置",
                self._last_config_engine, config_engine,
            )
            self._last_config_engine = config_engine
            self._engine_override = None
            self._engine_switch_pending = False
            self._pending_far_track_seed = False
            if self._fsm_far_track_active:
                self._fsm_far_track_active = False
                # 引擎不变（同为 mediapipe）时不会触发重建，需运行时撤掉
                # long_range 覆盖；引擎不同则随重建自然回到 config 档位。
                if self._tracker_config_signature[0] == config_engine:
                    self._set_tracker_long_range(
                        bool(self.config.get("long_range_enabled", False))
                    )
            self._engine_switcher.reset()
        self._configure_engine_switcher()
        if self._engine_switcher.enabled:
            self._schedule_yolo_prewarm()
        else:
            self._discard_warmed_yolo_tracker()
        # 用户关闭了自动切换但当前跑在 FSM 覆盖的引擎上：回到 config 配置的引擎。
        if not self._engine_switcher.enabled and self._engine_override is not None:
            logger.info("引擎自动切换已关闭，回到配置引擎 %s", config_engine)
            self._engine_override = None
            self._engine_switch_pending = False
            self._pending_far_track_seed = False
            if self._fsm_far_track_active:
                self._fsm_far_track_active = False
                if self._tracker_config_signature[0] == config_engine:
                    self._set_tracker_long_range(
                        bool(self.config.get("long_range_enabled", False))
                    )
            self._engine_switcher.reset()

        signature = self._tracker_signature()
        if signature != self._tracker_config_signature:
            # FAR_TRACK 期间的常规签名变化（如平滑参数）：重建需保留 long_range
            # 覆盖，否则 FSM 还在 FAR_TRACK 态但 ZOOM 链路被悄悄撤掉。
            overrides = (
                {"long_range_enabled": True} if self._fsm_far_track_active else None
            )
            self._request_tracker_rebuild(signature, config_overrides=overrides)

        voice_signature = self._current_voice_kws_signature()
        if voice_signature != self._voice_kws_signature:
            self._voice_kws_signature = voice_signature
            self.voice_command.request_reload()

        new_mode = self.config.get("interaction_mode")
        if new_mode != self.mode_manager.current_mode_name:
            self.set_mode(new_mode)
        logger.info("配置已更新: 模式 -> %s / 目标软件 -> %s", new_mode, self.ppt.target_app)

    def _request_tracker_rebuild(self, signature, config_overrides=None, prepared_tracker=None):
        self._tracker_request_id += 1
        request_id = self._tracker_request_id

        if prepared_tracker is not None:
            self._tracker_ready_signal.emit(prepared_tracker, signature, request_id, "")
            return

        def build():
            tracker = None
            error = ""
            try:
                tracker = self._create_tracker(signature, config_overrides=config_overrides)
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

        self._start_background_thread(build, "TrackerBuildWorker")

    def _on_tracker_ready(self, tracker, signature, request_id, error):
        if self._closing or request_id != self._tracker_request_id:
            close = getattr(tracker, "close", None)
            if callable(close):
                close()
            return
        if tracker is None:
            # 自动切换发起的重建失败：回退运行时覆盖并重置 FSM，否则
            # _engine_override != 当前签名 会让自动切换永久停摆。
            if self._engine_override is not None and signature[0] == self._engine_override:
                logger.warning(
                    "自动切换到 %s 失败（%s），回退引擎覆盖",
                    self._engine_override, error,
                )
                self._engine_override = None
                self._fsm_far_track_active = False
                self._pending_far_track_seed = False
                self._engine_switch_pending = False
                self._engine_switcher.reset()
            self.status_text = f"模型更新失败: {error}"
            self.status_color = (255, 0, 0)
            self.status_updated.emit(self.status_text, self.status_color)
            return
        seed_crop_zoom = bool(self._pending_far_track_seed)
        self._pending_far_track_seed = False
        context = {"signature": signature, "request_id": request_id}
        accepted = self.inference_worker.update_tracker(
            tracker,
            context=context,
            seed_crop_zoom=seed_crop_zoom,
        )
        if accepted is False:
            self._engine_switch_pending = False
            self.status_text = "模型更新失败: 推理线程已停止"
            self.status_color = (255, 0, 0)
            self.status_updated.emit(self.status_text, self.status_color)
            return
        logger.info("tracker 已构建完成，等待推理线程原子切换")

    def _on_tracker_swapped(self, worker, tracker, context, details):
        """Commit UI-visible tracker state in worker frame-signal order."""
        if self._closing or worker is not self.inference_worker:
            return
        context = context if isinstance(context, dict) else {}
        details = details if isinstance(details, dict) else {}
        signature = context.get("signature")
        request_id = context.get("request_id")
        self.tracker = tracker
        if signature is not None:
            self._tracker_config_signature = signature
        if request_id == self._tracker_request_id:
            self._engine_switch_pending = False
        if details.get("seed_requested"):
            logger.info(
                "CAPTURE→FAR_TRACK 交接: crop-zoom 种子%s",
                (
                    "已播种（YOLO hint → MP ZOOM）"
                    if details.get("seeded")
                    else "无 hint，靠人脸引导捕获"
                ),
            )
        # Yolo 已交接给 MP 后，再异步准备下一次 CAPTURE 所需实例。
        if (
            signature is not None
            and signature[0] == "mediapipe"
            and request_id == self._tracker_request_id
            and self._engine_switcher.enabled
        ):
            self._schedule_yolo_prewarm(delay_sec=0.5)
        logger.info("tracker 已由推理线程原子切换完成")

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

    @property
    def current_mode_name(self) -> str:
        """当前模式名称（代理 mode_manager，避免 UI 穿透内部对象）。"""
        return self.mode_manager.current_mode_name

    def set_mode(self, mode_name, sound=True):
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
        return MODE_NAME_ZH.get(mode, "未知模式")

    # ------------------------------------------------------------------
    # Inference Callbacks & Processing
    # ------------------------------------------------------------------

    def _on_frame_ready(self, frame, hands_landmarks, hands_gestures, worker=None):
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

    def _stabilize_gestures(self, hands_landmarks, hands_gestures):
        """用时序投票器稳定 MediaPipe 单帧手势标签。

        MediaPipe 远距离下关键点抖动导致 ml_label 逐帧跳变（如 Closed_Fist ↔
        None ↔ Open_Palm）。投票器累积多帧证据 + 滞回门控，输出稳定标签，
        覆盖回 hands_gestures 的 ml_label，使下游规则识别器收到干净输入。

        无手时重置投票器状态。
        """
        # 阶段1：旁路开关。关闭时直接透传 MediaPipe 原始标签（同时跳过几何分类器
        # 交叉验证），交给下游 draw_mode/各模式自有的去抖逻辑处理，避免多级时序门控叠加。
        if not self._temporal_voter_enabled:
            return hands_gestures

        if not hands_landmarks or not hands_gestures:
            self.gesture_voter.reset()
            return hands_gestures

        # 长度不一致属于异常输入，防御性重置并返回原值
        if len(hands_landmarks) != len(hands_gestures):
            logger.warning("_stabilize_gestures: landmarks/gestures 长度不一致 (%d vs %d)",
                           len(hands_landmarks), len(hands_gestures))
            self.gesture_voter.reset()
            return hands_gestures

        # 从第一只手的 landmarks 计算掌宽（5↔17），用于距离自适应窗口
        lm = hands_landmarks[0]
        if len(lm) < 21:
            logger.warning("_stabilize_gestures: 第一只手关键点数不足 21，跳过稳定化")
            return hands_gestures

        hand_width = max(
            20.0,
            math.hypot(lm[5][1] - lm[17][1], lm[5][2] - lm[17][2]),
        )

        # 从 tracker 的 smoother 获取关键点抖动 EMA，用于抖动自适应窗口
        jitter = 0.0
        tracker = getattr(self, 'tracker', None)
        if tracker is not None and hands_gestures:
            # 阶段 2.10：smoother 按 handedness 分配，取第一只手的 handedness 查 smoother
            key = hands_gestures[0].get("handedness", "Unknown")
            if key not in tracker.smoothers:
                key = "Unknown"
            smoother = tracker.smoothers.get(key)
            if smoother is not None and hasattr(smoother, 'get_jitter_stats'):
                jitter_stats = smoother.get_jitter_stats()
                # 用 21 个关键点的平均抖动作为整体抖动指标
                jitter = float(np.mean(jitter_stats)) if len(jitter_stats) > 0 else 0.0

        # 投票器输出稳定标签
        stable_label = self.gesture_voter.update(hands_gestures, hand_width, jitter)

        # 几何分类器交叉验证：当 MediaPipe 标签置信度低时，用几何特征仲裁
        # 远距离下 MediaPipe 像素特征退化，几何比例特征更稳定
        if hands_gestures:
            g = hands_gestures[0]
            ml_label = g.get("ml_label", "None")
            ml_score = g.get("score", 0.0)
            # ML 置信度低于 0.7 时启用几何仲裁
            if ml_score < 0.7 and stable_label != "None":
                geo_result = self.geo_classifier.classify(lm, ml_label, ml_score)
                geo_label = geo_result["label"]
                geo_conf = geo_result["confidence"]
                # 几何分类器置信度高且与 ML 标签不一致时，采用几何结果
                from services.geometric_classifier import ML_LABEL_TO_INTERNAL
                stable_internal = ML_LABEL_TO_INTERNAL.get(stable_label, "OTHER")
                if geo_conf > 0.5 and geo_label != stable_internal and geo_label != "OTHER":
                    # 几何分类器给出了不同的高置信度结果，覆盖稳定标签
                    # 反向映射：内部标签 → MediaPipe 标签名
                    internal_to_ml = {v: k for k, v in ML_LABEL_TO_INTERNAL.items()}
                    geo_ml_label = internal_to_ml.get(geo_label, stable_label)
                    logger.debug(
                        "[GeoClassifier] ML=%s(%.2f) → Geo=%s(%.2f), overriding",
                        stable_label, ml_score, geo_ml_label, geo_conf,
                    )
                    stable_label = geo_ml_label

        # 用稳定标签覆盖 hands_gestures 中的 ml_label
        # 只覆盖第一只手（主控手），其余手保持 MediaPipe 原始输出
        if stable_label != "None" and hands_gestures:
            g = hands_gestures[0]
            # 同步更新内部 label（ML_GESTURE_TO_INTERNAL 映射）
            from services.hand_tracker import ML_GESTURE_TO_INTERNAL
            g["ml_label"] = stable_label
            g["label"] = ML_GESTURE_TO_INTERNAL.get(stable_label, "OTHER")

        return hands_gestures

    def _process_frame_results(self, frame, hands_landmarks, hands_gestures):
        frame_h, frame_w = frame.shape[:2]

        # 远距三态自动切换：逐帧喂检出结果（回放/benchmark 不经过这里，不受影响）。
        # hand_bbox_ratio 供 FAR_TRACK 判"手变大/走近"；幽灵手预测补帧 bbox_area=0，
        # 不会虚增该比值；≤5 帧的幽灵手在 60/90 帧阈值下影响可忽略。
        hand_bbox_ratio = 0.0
        if hands_gestures:
            hand_bbox_ratio = max(
                float(g.get("bbox_area", 0.0)) for g in hands_gestures
            ) / float(max(frame_w * frame_h, 1))
        self._maybe_auto_switch_engine(hands_landmarks, hand_bbox_ratio)

        # 全局模式切换手势必须在时序稳定化之前判断。TemporalGestureVoter 的
        # FSM 会把 ILoveYou 延迟/过滤为 None/OTHER，导致 ModeManager 看不到
        # I_LOVE_YOU 而无法切换。模式内手势的抖动稳定放在后面处理。
        switched = self.mode_manager.maybe_switch_by_gesture(hands_landmarks, hands_gestures, frame_w)

        # 时序投票：稳定 MediaPipe 单帧手势标签的抖动（仅用于模式内手势）
        hands_gestures = self._stabilize_gestures(hands_landmarks, hands_gestures)

        gesture = "NONE"
        if switched:
            self.set_mode(self.mode_manager.current_mode_name)
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

    # 模式切换动作名 → 目标模式名（与 voice_command.py 的动作映射对称）
    _SWITCH_MODE_ACTIONS = {
        "switch_to_draw": "draw",
        "switch_to_mouse": "mouse",
        "switch_to_presentation": "presentation",
    }

    def execute_action(self, action_name):
        # —— 简单委托类动作：字典分发，避免长 if-elif 链 ——
        # 用 lambda 延迟属性访问，避免动作不需要的控制器被强制引用。
        ppt_actions = {
            "next_slide": lambda: self.ppt.next_slide(),
            "prev_slide": lambda: self.ppt.prev_slide(),
            "start_presentation": lambda: self.ppt.start_presentation(),
            "end_presentation": lambda: self.ppt.end_presentation(),
            "switch_app": lambda: self.ppt.switch_app(),
        }
        handler = ppt_actions.get(action_name)
        if handler is not None:
            handler()
            return

        voice_actions = {
            "launch_voice_assistant": lambda: self.voice_assistant.activate(),
            "hang_up_voice_assistant": lambda: self.voice_assistant.hang_up(),
        }
        handler = voice_actions.get(action_name)
        if handler is not None:
            self._start_background_thread(handler, "VoiceAssistantAction")
            return

        window_actions = {
            "minimize_assistant": lambda: self.minimize_requested.emit(),
            "restore_assistant": lambda: self.restore_requested.emit(),
        }
        handler = window_actions.get(action_name)
        if handler is not None:
            handler()
            return

        mouse_actions = {
            "left_click": lambda: self.mouse.left_click(),
            "double_click": lambda: self.mouse.double_click(),
            "right_click": lambda: self.mouse.right_click(),
        }
        handler = mouse_actions.get(action_name)
        if handler is not None:
            handler()
            return

        # 模式切换
        target_mode = self._SWITCH_MODE_ACTIONS.get(action_name)
        if target_mode is not None:
            self.set_mode(target_mode)
            return

        # —— 含额外逻辑的复杂动作 ——
        if action_name == "clear_canvas":
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
        elif action_name == "toggle_shape_correction":
            if self.mode_manager.current_mode_name == "draw":
                enabled = self.overlay.toggle_shape_correction()
                self.toolbar.set_shape_correction(enabled)

    def close(self, timeout_sec=3.0):
        """Resource release when window is closed."""
        deadline = time.monotonic() + max(0.0, float(timeout_sec))

        def remaining():
            return max(0.0, deadline - time.monotonic())

        self._closing = True
        self._closing_event.set()
        self._tracker_request_id += 1
        shutdown_incomplete = False
        if self.mode_manager.current_mode:
            try:
                mode_stopped = self.mode_manager.current_mode.on_exit() is not False
                shutdown_incomplete = shutdown_incomplete or not mode_stopped
            except Exception:
                shutdown_incomplete = True
                logger.exception("停止当前交互模式失败")

        inference_stopped = True
        if hasattr(self, 'inference_worker'):
            try:
                self.inference_worker.frame_ready.disconnect(self._on_frame_ready)
            except (TypeError, RuntimeError):
                pass
            try:
                inference_stopped = self.inference_worker.stop(
                    timeout_ms=max(0, int(remaining() * 1000))
                ) is not False
            except Exception:
                inference_stopped = False
                logger.exception("停止推理服务失败")
            shutdown_incomplete = shutdown_incomplete or not inference_stopped

        voice_stopped = True
        if hasattr(self, 'voice_command'):
            try:
                voice_stopped = self.voice_command.stop(
                    timeout_sec=remaining()
                ) is not False
            except Exception:
                voice_stopped = False
                logger.exception("停止语音指令服务失败")
            shutdown_incomplete = shutdown_incomplete or not voice_stopped

        # 停止语音听写服务（释放 SenseVoice ONNX session）
        if voice_stopped and hasattr(self, 'voice_dictation'):
            stop_dictation = getattr(self.voice_dictation, "stop", None)
            if callable(stop_dictation):
                try:
                    dictation_stopped = stop_dictation(
                        timeout_sec=remaining()
                    ) is not False
                    shutdown_incomplete = (
                        shutdown_incomplete or not dictation_stopped
                    )
                except Exception:
                    shutdown_incomplete = True
                    logger.exception("停止语音听写服务失败")
        elif not voice_stopped and hasattr(self, 'voice_dictation'):
            # ASR worker may still own the recognizer. Never release it here.
            shutdown_incomplete = True

        # 停止语音助手服务（释放可能持有的窗口句柄引用）
        if hasattr(self, 'voice_assistant'):
            stop_assistant = getattr(self.voice_assistant, "stop", None)
            if callable(stop_assistant):
                try:
                    stop_assistant()
                except Exception:
                    logger.exception("停止语音助手服务失败")

        background_stopped = self._wait_for_background_threads(remaining())
        shutdown_incomplete = shutdown_incomplete or not background_stopped

        if inference_stopped:
            if hasattr(self, "inference_worker"):
                self.tracker = getattr(
                    self.inference_worker,
                    "tracker",
                    self.tracker,
                )
            self.camera.release()
            close_tracker = getattr(self.tracker, "close", None)
            if callable(close_tracker):
                try:
                    close_tracker()
                except Exception:
                    logger.exception("关闭 tracker 失败")
        else:
            logger.error("推理线程仍持有摄像头/tracker，跳过并发释放")
        self._discard_warmed_yolo_tracker()
        if getattr(self, "frame_recorder", None) is not None:
            try:
                recorder_stopped = self.frame_recorder.close(
                    timeout_sec=remaining()
                ) is not False
                shutdown_incomplete = (
                    shutdown_incomplete or not recorder_stopped
                )
            except Exception:
                shutdown_incomplete = True
                logger.exception("停止录像服务失败")
        self._shutdown_incomplete = shutdown_incomplete
        return not shutdown_incomplete
