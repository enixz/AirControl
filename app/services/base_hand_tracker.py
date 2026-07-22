"""手部追踪器抽象基类 — 引擎无关的公共逻辑。

所有检测引擎继承此基类，只需实现 _detect() 和 _detect_crop_zoom()
两个抽象方法。find_hands() 的编排逻辑（crop-zoom 调度、运动追踪、
主控手排序、卡尔曼平滑、幽灵手预测）全部由基类提供。

设计原则：
  - 引擎切换时下游代码零修改（find_hands 接口不变）
  - KalmanSmoother 和运动追踪与检测引擎完全解耦
  - crop-zoom 状态机由基类管理，子类只负责"在裁剪区域上跑推理"
"""

import logging
import time
from abc import ABC, abstractmethod

import cv2

from .face_guide import FaceGuide
from .renderer import HandTrackerRenderer

# 平滑器与几何约束已拆到独立模块；此处 re-export 保持
# `from services.base_hand_tracker import KalmanSmoother/OneEuroSmoother/...`
# 等历史 import 路径可用（hand_tracker.py 与多个测试依赖此路径）。
from .smoothers import (  # noqa: F401
    _BONE_CONNECTIONS,
    GeometricConstraintFilter,
    KalmanSmoother,
    OneEuroFilter,
    OneEuroSmoother,
    _pack_landmarks,
)
from .sr_engine import SREngine

_zoom_logger = logging.getLogger("gesture")


# ---------------------------------------------------------------------------
# BaseHandTracker — 抽象基类
# ---------------------------------------------------------------------------

class BaseHandTracker(ABC):
    """手部追踪器抽象基类。

    子类必须实现：
      - _detect(frame)                → (hands_landmarks, hands_gestures, raw_data)
      - _detect_crop_zoom(frame, cx, cy, size) → 同上，或 ([], [], []) 触发回退
      - engine_name (property)        → str，引擎标识（用于日志与调试）

    find_hands() 的完整编排逻辑由基类提供，下游代码无需感知引擎差异。
    """

    # "Primary" 用于单手场景的位置连续性 smoother key（不依赖 handedness，
    # 避免 MediaPipe handedness 帧间翻转导致幽灵手双影）。
    # "Left"/"Right" 用于双手场景的位置匹配分配。
    HAND_KEYS = ("Primary", "Left", "Right", "Unknown")

    def __init__(self, max_num_hands=2, dominant_hand="Right", config=None):
        self.max_num_hands = max_num_hands
        self.dominant_hand = dominant_hand
        self._config = config

        # === 投机式增强层总开关（阶段1：默认关闭，详见 docs/修复记录_阶段0-1.md）===
        # 这些层是后期"全面强化"叠加的，互相打架反而产生闪烁/断笔/不跟手并降低识别率。
        # 默认关闭 = 回到接近原版的直管线；需要远距板书等场景时再单独打开做 A/B。
        self._long_range_enabled = bool(
            config.get("long_range_enabled", True)) if config else True
        self._geometric_constraint_enabled = bool(
            config.get("geometric_constraint_enabled", False)) if config else False
        # 幽灵手预测补帧：默认开启（与 D:\airControl 老版一致）。
        # 丢手时用 smoother 预测下一帧位置，避免检测抖动导致的瞬间丢帧。
        self._hand_prediction_enabled = bool(
            config.get("hand_prediction_enabled", True)) if config else True

        # 推理降采样宽度：MediaPipe 返回的是归一化坐标，与输入分辨率无关，故可把高分辨率
        # 帧先缩到这个宽度再喂给模型，用原始 w/h 反算像素即可（坐标系不变、无需补偿）。
        # 实测：1080p 整帧推理 ~42ms，降到 ~640-720px 后 ~15ms——直接决定"快速移动跟不跟手"。
        # 0 或负 = 不降采样（喂原帧）。子类 _detect 读取本属性。
        self._inference_max_width = int(
            config.get("inference_max_width", 720)) if config else 720

        # 平滑器（OneEuroSmoother 自适应低通滤波器）
        # 阶段 2.10（2026-07-05）：回到 D:\airControl 的 handedness-keyed 设计——
        # 每只手用 MediaPipe 的 handedness 标签（Left/Right/Unknown）作为 smoother key，
        # 每只手独立跟踪、永不跨手插值。此前按排序索引分配 Primary/Secondary 槽位，
        # 主手切换时 Primary smoother 跨手插值 → 识别点"飞在两手中间"拉扯。
        # min_cutoff/beta 回到 0.5/0.015（D:\airControl 实测不拉扯的参数）。
        _sm_min_cutoff = float(config.get("hand_smoothing_min_cutoff", 0.5)) if config else 0.5
        _sm_beta = float(config.get("hand_smoothing_beta", 0.015)) if config else 0.015
        self.smoothers = {
            k: OneEuroSmoother(min_cutoff=_sm_min_cutoff, beta=_sm_beta)
            for k in self.HAND_KEYS
        }
        # 几何约束后处理：在 smoother 输出后应用，抑制"手指乱飞"
        self._geo_filters = {
            k: GeometricConstraintFilter()
            for k in self.HAND_KEYS
        }
        self.last_gestures = []
        self._active_handedness = set()

        # 位置连续性 smoother key 分配状态（替代 handedness 做 key）
        # _prev_wrist_keys: [(wrist_pos, key), ...] 上一帧的 wrist 位置和分配的 key
        # 用于双手场景的位置匹配：当前帧手腕与上一帧最近的匹配为同一 smoother
        self._prev_wrist_keys = []

        # 运动追踪
        self._last_wrist_pos = {}
        self._motion_ema = {k: 0.0 for k in self.HAND_KEYS}
        self._motion_alpha = 0.4
        self._motion_window_ms = 500

        # === Crop-zoom 远距离增强 ===
        self._crop_zoom_mode = False
        # SR 引擎调度（ESPCN/Real-ESRGAN/GPU 自适应）已拆为独立 SREngine；
        # 档位日志去重与 auto 滞回状态封装在 self._sr 内，find_hands 在
        # ZOOM OFF/丢手时调 self._sr.reset_tier() 清空。
        self._sr = SREngine(logger=_zoom_logger)
        self._crop_padding_ratio = 2.5
        self._crop_target_size = 384
        # 裁剪框机械下限（仅防止把极少像素的退化裁剪喂进 resize/超分）。
        # 不再设人为的"画质下限"——裁剪框跟着手一路缩小、超分一路放大，
        # 直到 MediaPipe 真的认不出（由 miss-streak 退回全帧）为止。
        self._crop_min_size = 32
        # 远/近 ZOOM 触发阈值（手 bbox 占全帧面积比），可在 config.json 覆盖、便于现场调。
        # far 调低 → 仅当手更小/更远时才放大：板书在 ~50cm 书写距离不再误触发 ZOOM；
        # 也避免书写↔握拳↔张掌换姿势导致 bbox 大幅摆动而反复"拉风箱"——这正是
        # 板书近距离不稳的元凶（每次 ZOOM 切换都会重映射坐标，光标/笔尖跳变）。
        # 嫌还早就继续调小（0.006 / 0.004…）；想更早放大就调大。
        cfg = self._config
        self._zoom_far_threshold = float(
            cfg.get("zoom_far_threshold", 0.008) if cfg else 0.008)
        self._zoom_near_threshold = float(
            cfg.get("zoom_near_threshold", 0.040) if cfg else 0.040)
        self._zoom_switch_streak = 3         # 降低至 3 帧，加快远距离 ZOOM 触发响应
        self._near_streak = 0
        self._far_streak = 0
        self._zoom_miss_streak = 0
        # 连续多少帧 crop-zoom 检不到手才放弃 ZOOM。调高 → 远距离（如 3 米）检测偶尔
        # 漏帧时不会动不动就断 ZOOM；代价是近距离真把手收走后多停留几帧。可在 config 调。
        self._zoom_miss_threshold = int(
            self._config.get("zoom_miss_frames", 10) if self._config else 10)
        self._last_hint_center = None
        self._last_hint_size = 0

        # 渐进式连续缩放视口变量
        self._zoom_alpha = 0.15
        self._current_crop_center = None
        self._current_crop_size = None

        # === 人脸引导的远距离手部捕获 ===
        # 远处人脸比小手好检测得多：丢手时用人脸位置+大小预测手的搜索区，
        # 再对该区域 crop-zoom 放大检测，解决"手在画面角落、居中扫描抓不到"。
        # 人脸引导三方法已拆为独立 FaceGuide；视口状态（_crop_zoom_mode 等）
        # 由 tracker 在 acquire() 返回后自己写入。
        self._face_guide = FaceGuide(self._config, logger=_zoom_logger)
        self._renderer = HandTrackerRenderer(crop_min_size=self._crop_min_size)

        # === 自适应推理频率（跳帧优化）===
        # 静态手势时跳帧推理，中间帧用 smoother 预测补帧，提升 FPS。
        # 动态手势时全速推理，保证响应。
        self._skip_enabled = bool(self._config.get("adaptive_skip_enabled", False)) if self._config else False
        self._skip_motion_threshold = float(self._config.get("skip_motion_threshold", 0.15)) if self._config else 0.15
        self._skip_max_interval = int(self._config.get("skip_max_interval", 2)) if self._config else 2
        self._skip_counter = 0
        self._skip_current_interval = 1  # 当前跳帧间隔（1=不跳，2=每2帧推理1次）

    # ------------------------------------------------------------------
    # 抽象接口 — 子类必须实现
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def engine_name(self) -> str:
        """引擎标识，如 'mediapipe'。"""
        ...

    @abstractmethod
    def _detect(self, frame):
        """全帧检测。

        Args:
            frame: BGR 原帧（已镜像翻转）

        Returns:
            (hands_landmarks, hands_gestures, raw_data)
            - hands_landmarks: list of [[idx, x_px, y_px], ...] × 21 点
            - hands_gestures:  list of dict（必须含 label/score/handedness/handedness_score/bbox_area）
            - raw_data:        引擎原始输出（供子类内部使用，基类不解析）
        """
        ...

    @abstractmethod
    def _detect_crop_zoom(self, frame, hint_center, hint_size):
        """以 hint_center 为中心裁剪放大后检测。

        坐标必须映射回原帧像素坐标系。裁剪框过小或无放大收益时返回 ([], [], [])，
        基类会自动回退到 _detect()。

        Args:
            frame:       BGR 原帧
            hint_center: (cx, cy) 上一帧手部中心（原帧像素）
            hint_size:   上一帧手 bbox 最长边（像素）

        Returns:
            同 _detect()
        """
        ...

    # ------------------------------------------------------------------
    # 公共方法
    # ------------------------------------------------------------------

    def set_dominant_hand(self, dominant_hand):
        self.dominant_hand = dominant_hand
        logging.info("%s dominant_hand 切换为 %s", self.engine_name, dominant_hand)

    def migrate_state_from(self, old_tracker):
        """从旧 tracker 迁移关键追踪状态（RCU 风格：创建新实例后复制状态）。

        配置变更触发 tracker 重建时，新 tracker 默认从零开始，导致：
        - 平滑器重置 → 光标跳变
        - crop-zoom 退出 → 远距离突然丢失放大
        - 活动手标识丢失 → 幽灵手预测失效

        本方法将旧 tracker 的运行时状态迁移到新实例，保证用户体验连续性。
        """
        if old_tracker is None:
            return
        # crop-zoom 视口状态：保持远距离放大连续
        self._crop_zoom_mode = old_tracker._crop_zoom_mode
        self._current_crop_center = old_tracker._current_crop_center
        self._current_crop_size = old_tracker._current_crop_size
        self._last_hint_center = old_tracker._last_hint_center
        self._last_hint_size = old_tracker._last_hint_size
        # 活动手标识：保持幽灵手预测连续
        self._active_handedness = set(old_tracker._active_handedness)
        # 位置连续性 smoother key 匹配状态：保持双手场景匹配连续
        self._prev_wrist_keys = list(getattr(old_tracker, '_prev_wrist_keys', []))
        # 平滑器：保持光标位置连续（OneEuroSmoother 是纯 Python 对象，
        # close() 不影响它们，直接复制引用即可）
        self.smoothers = dict(old_tracker.smoothers)
        # 运动追踪：保持运动 EMA 连续
        self._motion_ema = dict(old_tracker._motion_ema)
        self._last_wrist_pos = dict(old_tracker._last_wrist_pos)
        logging.info("%s tracker 状态已从旧实例迁移", self.engine_name)

    def seed_crop_zoom_from_hint(self):
        """用迁移来的 _last_hint_* 播种 crop-zoom 视口（引擎交接用）。

        三态自动切换 CAPTURE→FAR_TRACK 时，YOLO 侧最后的手部 hint 已随
        migrate_state_from 复制过来；但 _compute_crop_viewport 只在
        _crop_zoom_mode=True 时才采用 hint（base_hand_tracker.py:342），
        而 ZOOM ON 正常路径要求"先检到手"（:707）——新 MP tracker 在远距
        裸检抓不到手，永远等不到这个前提。本方法直接翻转 _crop_zoom_mode
        并清空 _current_crop_*，下一帧 _compute_crop_viewport 用 hint 一步
        落位（_current_crop_center is None 分支），crop-zoom 立即锁住交接点。

        Returns:
            True 表示已播种；False 表示无可用 hint（调用方退回人脸引导捕获）
        """
        if self._last_hint_center is None or self._last_hint_size <= 0:
            return False
        self._crop_zoom_mode = True
        self._current_crop_center = None  # 下一帧用 hint 一步落位，避免从满幅爬回
        self._current_crop_size = None
        logging.info(
            "%s crop-zoom 已由交接 hint 播种: center=%s size=%.0f",
            self.engine_name, self._last_hint_center, self._last_hint_size,
        )
        return True

    def set_long_range_enabled(self, enabled):
        """运行时开关远距增强链路（crop-zoom/超分/人脸引导/多尺度）。

        三态自动切换 FAR_TRACK→NEAR 时用于撤掉 long_range 运行时覆盖，
        不重建 tracker。关闭时同时复位 crop-zoom 状态，避免视口残留。
        线程安全由调用方保证（orchestrator 在 inference_worker.lock 内调用）。
        """
        enabled = bool(enabled)
        if enabled == self._long_range_enabled:
            return
        self._long_range_enabled = enabled
        if not enabled:
            self._crop_zoom_mode = False
            self._sr.reset_tier()
            self._far_streak = 0
            self._near_streak = 0
            self._zoom_miss_streak = 0
            self._last_hint_center = None
            self._last_hint_size = 0
            self._current_crop_center = None
            self._current_crop_size = None
        logging.info("%s long_range_enabled 运行时切换为 %s", self.engine_name, enabled)

    def _should_skip_frame(self):
        """判断当前帧是否应该跳过推理（自适应推理频率）。

        静态手势（motion_ema 低）时跳帧，提升 FPS。
        动态手势（motion_ema 高）时全速推理，保证响应。

        Returns:
            True 表示跳过本帧推理，用 smoother 预测补帧
        """
        if not self._skip_enabled:
            return False

        # 没有初始化的 smoother → 不能跳帧（没有预测数据）
        if not any(sm.initialized for sm in self.smoothers.values()):
            return False

        # 计算主控手的运动 EMA
        max_motion = 0.0
        for key in self.HAND_KEYS:
            motion = self._motion_ema.get(key, 0.0)
            max_motion = max(max_motion, motion)

        # 动态调整跳帧间隔
        if max_motion < self._skip_motion_threshold:
            # 静态：跳帧
            self._skip_current_interval = self._skip_max_interval
        else:
            # 动态：不跳帧
            self._skip_current_interval = 1
            self._skip_counter = 0
            return False

        # 计数器逻辑：每 interval 帧推理一次
        self._skip_counter += 1
        if self._skip_counter >= self._skip_current_interval:
            self._skip_counter = 0
            return False  # 本帧要推理
        return True  # 本帧跳过

    def _predict_skip_frame(self, frame, draw, w_frame, h_frame):
        """跳帧时用 smoother 预测补帧，不调用 MediaPipe 推理。

        画面上仍然绘制预测的关键点，保持视觉连续性。
        """
        predicted_all = []
        gesture_all = []

        for key in sorted(self._active_handedness):
            smoother = self.smoothers.get(key)
            if smoother is None:
                continue
            predicted = smoother.predict()
            if predicted is not None:
                predicted_all.append(predicted)
                # 复用上一帧的手势标签，标记为 predicted
                g = None
                for gg in self.last_gestures:
                    if gg.get("handedness", "Unknown") == key:
                        g = dict(gg)
                        break
                if g is None:
                    g = {"handedness": key, "label": "NONE", "ml_label": "None", "score": 0.0}
                g["predicted"] = True
                g["skipped"] = True
                gesture_all.append(g)
                if draw:
                    # 统一用紫色：预测补帧与真实检测视觉一致，避免黄紫交替闪烁
                    self._renderer.draw_points(frame, predicted, (255, 0, 255))

        if predicted_all:
            if draw:
                self._renderer.draw_zoom_badge(frame, gesture_all, w_frame, h_frame, used_zoom=self._crop_zoom_mode)
            return frame, predicted_all, gesture_all

        # 没有预测数据 → 回退到正常推理
        return None  # 调用方需要检查 None 并回退

    def _handle_skip_frame(self, frame, draw, w_frame, h_frame):
        """自适应跳帧：静态手势时用 smoother 预测补帧。

        Returns:
            预测结果 (frame, landmarks, gestures) 或 None（继续正常推理）
        """
        if not self._should_skip_frame():
            return None
        return self._predict_skip_frame(frame, draw, w_frame, h_frame)

    def _compute_crop_viewport(self, w_frame, h_frame):
        """计算当前帧的 crop 视口目标并应用 EMA 平滑。

        基于 _last_hint_* 计算 target，再 EMA 平滑到 _current_crop_*。
        _current_crop_center 为 None 时（ZOOM 刚进入）直接用 target 初始化，
        实现一步落位、避免从整屏慢慢缩进（拉风箱元凶）。
        """
        if self._crop_zoom_mode and self._last_hint_center is not None and self._last_hint_size > 0:
            target_center = self._last_hint_center
            target_size = int(self._last_hint_size * self._crop_padding_ratio)
            # 不设人为下限：手越远裁剪框越小、放大倍率越高（手在检测帧中的占比恒为 1/padding）
            target_size = max(target_size, self._crop_min_size)
            target_size = min(target_size, min(w_frame, h_frame))
        else:
            target_center = (w_frame / 2.0, h_frame / 2.0)
            target_size = min(w_frame, h_frame)

        if self._current_crop_center is None:
            self._current_crop_center = target_center
            self._current_crop_size = float(target_size)
        else:
            alpha = self._zoom_alpha
            self._current_crop_center = (
                (1.0 - alpha) * self._current_crop_center[0] + alpha * target_center[0],
                (1.0 - alpha) * self._current_crop_center[1] + alpha * target_center[1],
            )
            self._current_crop_size = (
                (1.0 - alpha) * self._current_crop_size + alpha * float(target_size)
            )

    def find_hands(self, frame, draw=True):
        """统一入口 — 检测 + crop-zoom 调度 + 平滑 + 排序 + 预测。"""
        h_frame, w_frame, _ = frame.shape

        # 自适应推理频率：静态手势时跳帧，用 smoother 预测补帧
        skip_result = self._handle_skip_frame(frame, draw, w_frame, h_frame)
        if skip_result is not None:
            return skip_result

        # 1. 确定当前帧的缩放视口目标并 EMA 平滑
        self._compute_crop_viewport(w_frame, h_frame)

        # 2. 检测分发（crop-zoom / 全帧 / 人脸引导 / 多尺度）+ zoom 模式更新
        hands_landmarks, hands_gestures, use_crop_zoom = self._run_detection(
            frame, w_frame, h_frame
        )

        # 3. 有手检出：排序 + 平滑 + 幽灵手；否则尝试预测补帧 / 复位
        if hands_landmarks:
            return self._handle_hands_present(
                frame, hands_landmarks, hands_gestures,
                use_crop_zoom, draw, w_frame, h_frame
            )
        return self._handle_all_lost(frame, use_crop_zoom, draw, w_frame, h_frame)

    def _run_detection(self, frame, w_frame, h_frame):
        """检测分发：crop-zoom / 全帧 / 人脸引导 / 多尺度 + zoom 模式更新。

        Returns:
            (hands_landmarks, hands_gestures, use_crop_zoom)
        """
        # 决定是否使用 Crop-Zoom。若当前裁剪框接近全图大小（例如 >= 92%），直接使用 _detect 节省性能
        # 阶段1：long_range_enabled=False 时彻底关闭 crop-zoom（避免"拉风箱"坐标重映射跳变）。
        use_crop_zoom = (
            self._long_range_enabled
            and self._current_crop_size < min(w_frame, h_frame) * 0.92
        )

        if use_crop_zoom:
            hands_landmarks, hands_gestures, raw = self._detect_crop_zoom(
                frame, self._current_crop_center, self._current_crop_size,
            )
            if not hands_landmarks:
                self._zoom_miss_streak += 1
                # A hand can leave the predicted crop after a quick movement.
                # Retry the full frame periodically instead of waiting for the
                # whole zoom miss window to expire before reacquiring it.
                if self._zoom_miss_streak % 3 == 0:
                    full_landmarks, full_gestures, full_raw = self._detect(frame)
                    if full_landmarks:
                        hands_landmarks = full_landmarks
                        hands_gestures = full_gestures
                        raw = full_raw
                        self._zoom_miss_streak = 0
                if self._zoom_miss_streak >= self._zoom_miss_threshold:
                    self._crop_zoom_mode = False
                    self._sr.reset_tier()
                    self._zoom_miss_streak = 0
                    self._far_streak = 0
                    # 强行复位裁剪窗口到全屏，防止跟丢后延迟拉回
                    self._current_crop_center = (w_frame / 2.0, h_frame / 2.0)
                    self._current_crop_size = float(min(w_frame, h_frame))
                    _zoom_logger.info(
                        "=> ZOOM OFF (%s: crop-zoom 连续 %d 帧未检出，退回全帧)",
                        self.engine_name, self._zoom_miss_threshold,
                    )
            else:
                self._zoom_miss_streak = 0
        else:
            hands_landmarks, hands_gestures, raw = self._detect(frame)
            # 阶段1：远距增强（人脸引导 + 多尺度回退）仅在 long_range_enabled 时启用。
            if self._long_range_enabled:
                # 全帧没检到手 → 人脸引导捕获（解决远距离冷启动：手太小直接检不出）
                if not hands_landmarks:
                    acq = self._face_guide.acquire(
                        frame, w_frame, h_frame, self._crop_min_size, self._detect_crop_zoom
                    )
                    if acq is not None:
                        hands_landmarks, hands_gestures, raw, cx, cy, size = acq
                        self._crop_zoom_mode = True
                        self._zoom_miss_streak = 0
                        self._current_crop_center = (cx, cy)
                        self._current_crop_size = float(size)
                # 仍未检到手 → 多尺度检测（缩小到0.5x，远距离小手相对变大）
                if not hands_landmarks:
                    ms_result = self._detect_multiscale(frame, w_frame, h_frame)
                    if ms_result is not None:
                        hands_landmarks, hands_gestures, raw = ms_result

        # === 更新 zoom 模式 ===
        # 阶段1：关闭远距增强时不再驱动 zoom 状态机（_crop_zoom_mode 恒为 False）。
        if self._long_range_enabled:
            self._update_zoom_mode(hands_landmarks, hands_gestures, w_frame, h_frame)

        return hands_landmarks, hands_gestures, use_crop_zoom

    def _assign_smoother_keys(self, hands_landmarks, hands_gestures):
        """用位置连续性分配 smoother key，替代 handedness。

        核心问题：MediaPipe handedness 帧间翻转（Left↔Right）时，旧 handedness
        的 smoother 进入 missing_keys 触发幽灵手 predict()，产生双影和位置跳变。

        解决方案：
        - 单手场景：统一用 "Primary"，不依赖 handedness → 翻转不影响 smoother
        - 双手场景：用 wrist 位置匹配上一帧 → 手不会瞬间从画面左侧跳到右侧
        - 无历史或匹配失败：回退到 handedness

        Returns:
            list of str: 每只手对应的 smoother key
        """
        n = len(hands_landmarks)
        if n == 0:
            return []

        # 单手：统一用 Primary（不依赖 handedness，翻转不影响 smoother）
        if n == 1:
            key = "Primary"
            wrist = (float(hands_landmarks[0][0][1]), float(hands_landmarks[0][0][2]))
            self._prev_wrist_keys = [(wrist, key)]
            return [key]

        # 双手：用 wrist 位置匹配上一帧
        wrists = [
            (float(lm[0][1]), float(lm[0][2]))
            for lm in hands_landmarks
        ]

        if not self._prev_wrist_keys or len(self._prev_wrist_keys) != n:
            # 无历史或手数变化：用 handedness 做初始分配
            keys = []
            for g in hands_gestures:
                h = g.get("handedness", "Unknown")
                if h not in ("Left", "Right"):
                    h = "Primary" if not keys else "Unknown"
                keys.append(h)
            # 去重：如果两只手分到同一个 key，第二只用另一个
            if keys[0] == keys[1] and keys[0] in ("Left", "Right"):
                keys[1] = "Right" if keys[0] == "Left" else "Left"
            self._prev_wrist_keys = list(zip(wrists, keys, strict=True))
            return keys

        # 位置匹配：当前帧手腕与上一帧最近的匹配为同一 smoother
        prev_wrists = [wk[0] for wk in self._prev_wrist_keys]
        prev_keys = [wk[1] for wk in self._prev_wrist_keys]

        keys = [None] * n
        used_prev = set()
        # 按距离排序，先匹配最近的（减少误匹配）
        pairs = []
        for i, w in enumerate(wrists):
            for j, pw in enumerate(prev_wrists):
                dist = (w[0] - pw[0]) ** 2 + (w[1] - pw[1]) ** 2
                pairs.append((dist, i, j))
        pairs.sort()

        assigned_current = set()
        for _dist, i, j in pairs:
            if i in assigned_current or j in used_prev:
                continue
            keys[i] = prev_keys[j]
            assigned_current.add(i)
            used_prev.add(j)

        # 未匹配的手用 handedness 兜底
        for i in range(n):
            if keys[i] is None:
                h = hands_gestures[i].get("handedness", "Unknown")
                if h not in self.smoothers:
                    h = "Unknown"
                keys[i] = h

        self._prev_wrist_keys = list(zip(wrists, keys, strict=True))
        return keys

    def _handle_hands_present(self, frame, hands_landmarks, hands_gestures,
                              use_crop_zoom, draw, w_frame, h_frame):
        """有手检出：运动→打分→排序→平滑→幽灵手→绘制。

        smoother key 分配策略（P1 修复）：
        - 单手场景：统一用 "Primary"（位置连续性），不依赖 handedness
        - 双手场景：用 wrist 位置匹配上一帧，避免 handedness 翻转导致切换
        - 排序只影响 zoom 视口 hint 和返回顺序，不影响 smoother 归属
        """
        h, w, _ = frame.shape

        motion_map = self._update_motion(hands_landmarks, hands_gestures, w, h)
        scores = [
            self._priority_score(
                hands_gestures[i], w, h,
                landmarks=hands_landmarks[i],
                motion=motion_map.get(
                    hands_gestures[i].get("handedness", "Unknown"), 0.0
                ),
            )
            for i in range(len(hands_landmarks))
        ]

        # 按分数降序排列：分数最高的手排在 index 0（zoom 视口锁定 + 下游主光标）
        order = sorted(
            range(len(hands_landmarks)),
            key=lambda i: scores[i],
            reverse=True,
        )
        hands_landmarks = [hands_landmarks[i] for i in order]
        hands_gestures = [hands_gestures[i] for i in order]

        # zoom 视口只锁优先级最高的那只手（_priority_score 选出的"举得最高"者）。
        hint = self._compute_hint_from_landmarks(hands_landmarks[:1])
        if hint[0] is not None:
            self._last_hint_center, self._last_hint_size = hint

        # 用位置连续性分配 smoother key（替代 handedness）
        smoother_keys = self._assign_smoother_keys(hands_landmarks, hands_gestures)

        smoothed_all = []
        gesture_all = []
        seen_keys = set()
        for idx, (landmarks, gesture) in enumerate(zip(hands_landmarks, hands_gestures, strict=True)):
            key = smoother_keys[idx] if idx < len(smoother_keys) else "Unknown"
            if key not in self.smoothers:
                key = "Unknown"
            smoothed = self.smoothers[key].update(landmarks)
            # 几何约束后处理：抑制"手指乱飞"。阶段1默认关闭。
            if self._geometric_constraint_enabled:
                geo_filter = self._geo_filters.get(key)
                if geo_filter is not None:
                    smoothed = geo_filter.apply(smoothed)
            seen_keys.add(key)
            smoothed_all.append(smoothed)
            gesture_all.append(gesture)
            if draw:
                self._renderer.draw_points(frame, smoothed, (255, 0, 255))

        # 幽灵手预测补帧（gated by _hand_prediction_enabled，默认开启）
        # 单手场景下 active 只有 "Primary"，handedness 翻转不再产生幽灵手
        if self._hand_prediction_enabled:
            missing_keys = self._active_handedness - seen_keys
            for key in missing_keys:
                smoother = self.smoothers.get(key)
                if smoother is None or smoother.lost_frames >= 5:
                    continue
                ghost = smoother.predict()
                if ghost is None:
                    continue
                smoothed_all.append(ghost)
                gesture_all.append({
                    "ml_label": "None",
                    "label": "OTHER",
                    "score": 0.0,
                    "handedness": key,
                    "handedness_score": 0.0,
                    "bbox_area": 0.0,
                    "predicted": True,
                })
                seen_keys.add(key)
                if draw:
                    self._renderer.draw_points(frame, ghost, (255, 0, 255))

        self._active_handedness = seen_keys
        self.last_gestures = gesture_all

        if use_crop_zoom and self._current_crop_center:
            frame = self._renderer.apply_visual_zoom(frame, self._current_crop_center, self._current_crop_size)
        if draw:
            self._renderer.draw_zoom_badge(frame, gesture_all, w_frame, h_frame, used_zoom=use_crop_zoom)

        return frame, smoothed_all, gesture_all

    def _handle_all_lost(self, frame, use_crop_zoom, draw, w_frame, h_frame):
        """全部丢失：尝试预测补帧，否则复位状态并返回空。

        阶段1默认关闭预测：丢手即如实报告无手，避免冻结的"幽灵光标/笔尖"产生闪烁。
        上层（draw_mode 的跟丢缓冲、mouse_mode）自有短暂丢手的优雅降级。
        """
        has_prediction = False
        predicted_all = []
        if self._hand_prediction_enabled:
            for key in sorted(self._active_handedness):
                smoother = self.smoothers.get(key)
                if smoother is None:
                    continue
                predicted = smoother.predict()
                if predicted is not None:
                    predicted_all.append(predicted)
                    has_prediction = True

        if has_prediction:
            if draw and predicted_all:
                # 统一用紫色：预测补帧与真实检测视觉一致，避免黄紫交替闪烁
                self._renderer.draw_points(frame, predicted_all[0], (255, 0, 255))
            if use_crop_zoom and self._current_crop_center:
                frame = self._renderer.apply_visual_zoom(frame, self._current_crop_center, self._current_crop_size)
            if draw:
                self._renderer.draw_zoom_badge(frame, self.last_gestures, w_frame, h_frame, used_zoom=use_crop_zoom)
            return frame, predicted_all, list(self.last_gestures)

        if any(sm.initialized for sm in self.smoothers.values()):
            logging.info("Tracking fully lost. Resetting smoothers.")
            for sm in self.smoothers.values():
                sm.reset()
            for gf in self._geo_filters.values():
                gf.reset()
            self.last_gestures = []
            self._active_handedness.clear()
            self._prev_wrist_keys = []
            self._crop_zoom_mode = False
            self._sr.reset_tier()
            self._last_hint_center = None
            self._last_hint_size = 0
            self._far_streak = 0
            self._near_streak = 0
            self._zoom_miss_streak = 0
            self._current_crop_center = None
            self._current_crop_size = None

        if use_crop_zoom and self._current_crop_center:
            frame = self._renderer.apply_visual_zoom(frame, self._current_crop_center, self._current_crop_size)
        if draw:
            self._renderer.draw_zoom_badge(frame, [], w_frame, h_frame, used_zoom=use_crop_zoom)

        return frame, [], []

    # ------------------------------------------------------------------
    # 内部方法 — crop-zoom 状态机
    # ------------------------------------------------------------------

    def _compute_hint_from_landmarks(self, all_hands_landmarks):
        """计算 crop-zoom 视口的中心和大小的目标值。"""
        if not all_hands_landmarks:
            return None, 0
        xs, ys = [], []
        for landmarks in all_hands_landmarks:
            for lm in landmarks:
                xs.append(lm[1])
                ys.append(lm[2])
        if not xs:
            return None, 0
        cx = (min(xs) + max(xs)) / 2.0
        cy = (min(ys) + max(ys)) / 2.0
        size = max(max(xs) - min(xs), max(ys) - min(ys))
        return (cx, cy), size

    def _update_zoom_mode(self, hands_landmarks, hands_gestures, frame_w, frame_h):
        if not hands_landmarks or not hands_gestures:
            return
        max_bbox = max(g.get("bbox_area", 0.0) for g in hands_gestures)
        frame_area = max(frame_w * frame_h, 1)
        ratio = max_bbox / frame_area

        if ratio < self._zoom_far_threshold:
            self._far_streak += 1
            self._near_streak = 0
            if not self._crop_zoom_mode and self._far_streak >= self._zoom_switch_streak:
                self._crop_zoom_mode = True
                self._far_streak = 0
                # 进入 ZOOM 时让视口一步落到手上，不再从整屏 EMA 慢慢缩进。
                # 否则每次 ZOOM ON 都先显示整屏、再花约 0.5s 爬回手部——zoom 反复
                # 进入时这段来回缩放正是"拉风箱"的元凶。置空后下一帧 find_hands 顶部
                # 会用当前 hint 直接初始化视口（_current_crop_center is None 分支）。
                self._current_crop_center = None
                self._current_crop_size = None
                _zoom_logger.info(
                    "=> ZOOM ON (%s bbox=%.2f%% < %.2f%% far_threshold)",
                    self.engine_name, ratio * 100, self._zoom_far_threshold * 100,
                )
        elif ratio > self._zoom_near_threshold:
            self._near_streak += 1
            self._far_streak = 0
            if self._crop_zoom_mode and self._near_streak >= self._zoom_switch_streak:
                self._crop_zoom_mode = False
                self._sr.reset_tier()
                self._near_streak = 0
                _zoom_logger.info(
                    "=> ZOOM OFF (%s bbox=%.2f%% > %.2f%% near_threshold)",
                    self.engine_name, ratio * 100, self._zoom_near_threshold * 100,
                )
        else:
            self._far_streak = max(0, self._far_streak - 1)
            self._near_streak = max(0, self._near_streak - 1)

    # ------------------------------------------------------------------
    # 内部方法 — 运动追踪
    # ------------------------------------------------------------------

    def _update_motion(self, landmarks_list, gestures_list, frame_w, frame_h):
        diag = (frame_w * frame_w + frame_h * frame_h) ** 0.5
        if diag <= 0:
            return {}
        now_ms = time.time() * 1000.0
        seen_keys = set()
        motion_map = {}

        for landmarks, gesture in zip(landmarks_list, gestures_list, strict=True):
            key = gesture.get("handedness", "Unknown")
            if key not in self._motion_ema:
                key = "Unknown"
            seen_keys.add(key)

            wrist = landmarks[0]
            cx, cy = float(wrist[1]), float(wrist[2])
            prev = self._last_wrist_pos.get(key)

            if prev is not None and (now_ms - prev[2]) < self._motion_window_ms:
                dx = cx - prev[0]
                dy = cy - prev[1]
                dist = (dx * dx + dy * dy) ** 0.5
                sample = min(dist / (diag * 0.05), 1.0)
            else:
                sample = 0.0

            self._motion_ema[key] = (
                (1.0 - self._motion_alpha) * self._motion_ema[key]
                + self._motion_alpha * sample
            )
            self._last_wrist_pos[key] = (cx, cy, now_ms)
            motion_map[key] = self._motion_ema[key]

        for key in self._motion_ema:
            if key not in seen_keys:
                self._motion_ema[key] *= (1.0 - self._motion_alpha)

        return motion_map

    def _priority_score(self, gesture_meta, frame_w, frame_h, landmarks=None, motion=0.0):
        """多手并存时选"锁定/光标/放大"手的优先级分。

        score 仅用于排序（决定 zoom 视口锁哪只手、smoothed_all[0] 是谁），
        不影响 smoother 归属——smoother 按 handedness 独立跟踪，不会跨手插值。
        """
        score = 0.0
        handedness = gesture_meta.get("handedness", "Unknown")

        # 惯用手偏好：轻微 +1.0，让惯用手在等高时优先成为主控手
        if self.dominant_hand in ("Left", "Right"):
            if handedness == self.dominant_hand:
                score += 1.0

        # 手大小分：bbox 占比 × 8（近处的手略优先）
        if frame_w > 0 and frame_h > 0:
            bbox_ratio = gesture_meta.get("bbox_area", 0.0) / (frame_w * frame_h)
            score += bbox_ratio * 8.0

        # handedness_score：MediaPipe 的左右手置信度
        score += float(gesture_meta.get("handedness_score", 0.0))

        # 主判据：wrist 举得越高（y 越小）分越高。权重 25 显著高于其他项，
        # 确保举起来的手稳定地成为锁定/放大对象。
        if landmarks and frame_h > 0:
            wrist_y_norm = max(0.0, min(1.0, float(landmarks[0][2]) / float(frame_h)))
            score += (1.0 - wrist_y_norm) * 25.0

        # 运动分：运动大的手略优先（避免锁住静止手）
        score += float(motion) * 6.0
        return score

    # ------------------------------------------------------------------
    # 多尺度检测
    # ------------------------------------------------------------------

    def _detect_multiscale(self, frame, w, h):
        """多尺度检测：缩小到0.5x再检测，远距离小手相对变大。

        原图检不到手时调用。缩小后检到的手坐标会映射回原图。
        只尝试0.5x一个尺度，避免性能损失。

        Returns:
            (hands_landmarks, hands_gestures, raw) 或 None
        """
        try:
            # 缩小到0.5x
            small = cv2.resize(frame, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
            result = self._detect(small)
            if not result or not result[0]:
                return None

            hands_landmarks, hands_gestures, raw = result
            # 坐标映射回原图（×2）
            scaled_landmarks = []
            for landmarks in hands_landmarks:
                scaled = [[lm[0], lm[1] * 2.0, lm[2] * 2.0] for lm in landmarks]
                scaled_landmarks.append(scaled)

            # 更新 bbox_area（×4，因为面积放大4倍）
            for g in hands_gestures:
                g["bbox_area"] = g.get("bbox_area", 0.0) * 4.0

            _zoom_logger.debug(
                "[MULTISCALE] 0.5x 检出 %d 只手（原图未检出）", len(scaled_landmarks),
            )
            return scaled_landmarks, hands_gestures, raw
        except Exception as e:
            _zoom_logger.debug("[MULTISCALE] 多尺度检测异常: %s", e)
            return None

    def _perform_crop_zoom(self, frame, crop_center, current_crop_size, run_sub_detect):
        """通用 crop-zoom 逻辑：裁剪 → 超分/插值放大 → 子类检测 → 坐标映射。

        Args:
            frame, crop_center, current_crop_size: 同 _detect_crop_zoom
            run_sub_detect: callable(cropped_bgr) → detection_result
                子类提供在裁剪区域上跑推理的回调

        Returns:
            (hands_landmarks, hands_gestures, raw_data) 或 ([], [], [])
        """
        h, w, _ = frame.shape

        crop_size = int(round(current_crop_size))
        crop_size = max(crop_size, self._crop_min_size)
        crop_size = min(crop_size, min(w, h))

        if crop_size >= int(min(w, h) * 0.95):
            return [], [], []

        cx, cy = crop_center
        x0 = int(round(cx - crop_size / 2))
        y0 = int(round(cy - crop_size / 2))
        x0 = max(0, min(x0, w - crop_size))
        y0 = max(0, min(y0, h - crop_size))

        crop = frame[y0:y0 + crop_size, x0:x0 + crop_size]
        # 仅退化裁剪（小于机械下限）才放弃 crop-zoom；其余一律尽力放大
        if crop.size == 0 or crop.shape[0] < self._crop_min_size or crop.shape[1] < self._crop_min_size:
            return [], [], []

        # 读取超分辨率配置
        sr_engine = "auto"
        if self._config is not None:
            sr_engine = self._config.get("zoom_sr_engine", "auto")

        # 初始化超分引擎状态（具体模型按需加载）
        self._sr.init()

        target = self._crop_target_size

        # 解析 auto / 显式选择 → 具体引擎
        actual_engine = self._sr.resolve(sr_engine, crop_size, target)

        # 执行放大
        zoomed = None
        if actual_engine == "espcn":
            zoomed = self._sr.espcn(crop, target)
        elif actual_engine in ("realesrgan_cpu", "realesrgan_gpu"):
            zoomed = self._sr.realesrgan(
                crop, target, prefer_gpu=(actual_engine == "realesrgan_gpu")
            )

        # 兜底：超分关闭或执行失败 → 普通插值（放大用 LINEAR，缩小用 AREA）
        if zoomed is None:
            interp = cv2.INTER_LINEAR if crop_size < target else cv2.INTER_AREA
            zoomed = cv2.resize(crop, (target, target), interpolation=interp)
            # 实际生效档位：显式 none 用插值；否则说明所选引擎执行失败、回退插值
            effective_tier = (
                "none(interp)" if actual_engine == "none"
                else f"none(interp,{actual_engine}_failed)"
            )
        else:
            effective_tier = actual_engine
        self._sr.log_tier(effective_tier, crop_size, target)

        # 子类在 zoomed 上跑推理
        detection_result = run_sub_detect(zoomed)

        # 坐标映射回原帧
        scale = crop_size / target

        def to_orig(norm_x, norm_y):
            return (x0 + norm_x * target * scale, y0 + norm_y * target * scale)

        return detection_result, (x0, y0, crop_size, target, scale), to_orig
