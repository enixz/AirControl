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
import math
import os
import time
from abc import ABC, abstractmethod

import cv2
import numpy as np

_zoom_logger = logging.getLogger("gesture")


# ---------------------------------------------------------------------------
# KalmanSmoother — 卡尔曼滤波 + EMA 双重平滑器
# ---------------------------------------------------------------------------

class KalmanSmoother:
    """为 21 个关键点各自维护 [x,y,vx,vy] 状态的卡尔曼 + EMA 平滑器。"""

    def __init__(
        self,
        num_keypoints=21,
        process_noise=0.5,
        measurement_noise=3.0,
        ema_alpha=0.4,
        max_lost_frames=8,
    ):
        self.num_kp = num_keypoints
        self.ema_alpha = ema_alpha
        self.max_lost_frames = max_lost_frames

        self.filters = []
        for _ in range(num_keypoints):
            kf = cv2.KalmanFilter(4, 2)
            kf.measurementMatrix = np.eye(2, 4, dtype=np.float32)
            kf.transitionMatrix = np.array(
                [[1, 0, 1, 0],
                 [0, 1, 0, 1],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]], dtype=np.float32,
            )
            kf.processNoiseCov = np.array(
                [[0.25, 0, 0.5, 0],
                 [0, 0.25, 0, 0.5],
                 [0.5, 0, 1, 0],
                 [0, 0.5, 0, 1]], dtype=np.float32,
            ) * process_noise
            kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise
            kf.errorCovPost = np.eye(4, dtype=np.float32) * 10.0
            self.filters.append(kf)

        self.ema = None
        self.lost_frames = 0
        self.initialized = False

    def update(self, landmarks):
        raw = np.array([[lm[1], lm[2]] for lm in landmarks], dtype=np.float32)

        if not self.initialized:
            self.ema = raw.copy()
            for i, kf in enumerate(self.filters):
                kf.statePost = np.array(
                    [[raw[i, 0]], [raw[i, 1]], [0], [0]], dtype=np.float32,
                )
            self.initialized = True
        else:
            self.ema = self.ema_alpha * raw + (1 - self.ema_alpha) * self.ema
            for i, kf in enumerate(self.filters):
                kf.predict()
                kf.correct(self.ema[i].reshape(2, 1))

        self.lost_frames = 0
        return [
            [lm[0], int(round(self.ema[i, 0])), int(round(self.ema[i, 1]))]
            for i, lm in enumerate(landmarks)
        ]

    def predict(self):
        if not self.initialized or self.lost_frames >= self.max_lost_frames:
            return None
        self.lost_frames += 1
        pred = np.zeros((self.num_kp, 2), dtype=np.float32)
        for i, kf in enumerate(self.filters):
            p = kf.predict()
            pred[i] = [p[0, 0], p[1, 0]]
        return [
            [i, int(round(pred[i, 0])), int(round(pred[i, 1]))]
            for i in range(self.num_kp)
        ]

    def reset(self):
        self.ema = None
        self.lost_frames = 0
        self.initialized = False


# ---------------------------------------------------------------------------
# OneEuroFilter & OneEuroSmoother — 自适应低通平滑滤波器
# ---------------------------------------------------------------------------

class OneEuroFilter:
    """一欧元自适应低通滤波器。

    能够根据信号的变化速度自动调整截止频率：
      - 慢速时降低截止频率以消除抖动；
      - 快速时提高截止频率以消除延迟。
    """
    def __init__(self, t0, x0, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = float(x0)
        self.dx_prev = 0.0
        self.t_prev = float(t0)

    def __call__(self, t, x):
        t = float(t)
        x = float(x)
        dt = t - self.t_prev
        if dt <= 0.0:
            return self.x_prev

        # 1. 计算一阶导数（速度）并应用低通滤波
        dx = (x - self.x_prev) / dt
        r_d = 2.0 * math.pi * self.d_cutoff * dt
        a_d = r_d / (r_d + 1.0)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev

        # 2. 根据运动速度自适应计算截止频率
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)

        # 3. 对位置进行低通滤波
        r_x = 2.0 * math.pi * cutoff * dt
        a_x = r_x / (r_x + 1.0)
        x_hat = a_x * x + (1.0 - a_x) * self.x_prev

        # 4. 保存状态
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat


class OneEuroSmoother:
    """为手部 21 个关键点各自维护 X 和 Y 轴一欧元滤波器的平滑器。

    接口设计与原 KalmanSmoother 完全一致，实现无缝替换。
    """
    def __init__(
        self,
        num_keypoints=21,
        min_cutoff=1.5,     # 手部微抖动截止频率（静态时）
        beta=0.01,          # 速度响应系数（运动时防延迟）
        d_cutoff=1.0,       # 速度低通滤波截止频率
        max_lost_frames=8,
    ):
        self.num_kp = num_keypoints
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.max_lost_frames = max_lost_frames

        self.filters_x = []
        self.filters_y = []
        self.initialized = False
        self.lost_frames = 0
        self.last_landmarks = None

    def update(self, landmarks):
        t = time.time()
        raw = np.array([[lm[1], lm[2]] for lm in landmarks], dtype=np.float32)

        if not self.initialized:
            self.filters_x = [
                OneEuroFilter(t, raw[i, 0], min_cutoff=self.min_cutoff, beta=self.beta, d_cutoff=self.d_cutoff)
                for i in range(self.num_kp)
            ]
            self.filters_y = [
                OneEuroFilter(t, raw[i, 1], min_cutoff=self.min_cutoff, beta=self.beta, d_cutoff=self.d_cutoff)
                for i in range(self.num_kp)
            ]
            self.initialized = True
            smoothed = raw.copy()
        else:
            smoothed = np.zeros((self.num_kp, 2), dtype=np.float32)
            for i in range(self.num_kp):
                smoothed[i, 0] = self.filters_x[i](t, raw[i, 0])
                smoothed[i, 1] = self.filters_y[i](t, raw[i, 1])

        self.lost_frames = 0
        self.last_landmarks = [
            [lm[0], int(round(smoothed[i, 0])), int(round(smoothed[i, 1]))]
            for i, lm in enumerate(landmarks)
        ]
        return self.last_landmarks

    def predict(self):
        if not self.initialized or self.lost_frames >= self.max_lost_frames:
            return None
        self.lost_frames += 1
        return self.last_landmarks

    def reset(self):
        self.initialized = False
        self.lost_frames = 0
        self.last_landmarks = None
        self.filters_x.clear()
        self.filters_y.clear()


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

    HAND_KEYS = ("Left", "Right", "Unknown")

    def __init__(self, max_num_hands=2, dominant_hand="Right", config=None):
        self.max_num_hands = max_num_hands
        self.dominant_hand = dominant_hand
        self._config = config

        # 平滑器（使用 OneEuroSmoother 替换 KalmanSmoother 以实现自适应降噪与零延迟跟手）
        # min_cutoff 决定"静止时"的抖动抑制：越小越稳（手指不动不抖），代价是起步略滞后；
        # beta 决定"运动时"的跟手：越大移动越跟手、但慢速时易漏抖。两者均可在 config 调。
        # 默认 min_cutoff 由原来的 1.5 降到 0.5，主治"手指不动也抖"。
        _sm_min_cutoff = float(config.get("hand_smoothing_min_cutoff", 0.5)) if config else 0.5
        _sm_beta = float(config.get("hand_smoothing_beta", 0.015)) if config else 0.015
        self.smoothers = {
            k: OneEuroSmoother(min_cutoff=_sm_min_cutoff, beta=_sm_beta)
            for k in self.HAND_KEYS
        }
        self.last_gestures = []
        self._active_handedness = set()

        # 运动追踪
        self._last_wrist_pos = {}
        self._motion_ema = {k: 0.0 for k in self.HAND_KEYS}
        self._motion_alpha = 0.4
        self._motion_window_ms = 500

        # === Crop-zoom 远距离增强 ===
        self._crop_zoom_mode = False
        # crop-zoom 实际生效的放大档位日志去重：仅在档位变化时打印；ZOOM OFF 复位，
        # 使每段 ZOOM 重新记录一次，便于与 ZOOM ON/OFF 日志对照。
        self._last_sr_tier = None
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
        self._face_acquire_enabled = True
        self._face_scan_interval = 4          # 每 N 帧（且仅在丢手时）尝试一次人脸扫描
        self._face_scan_counter = 0
        self._face_hand_region_scale = 7.0    # 搜索区边长 = 人脸高 × 该系数
        self._face_hand_down_bias = 1.0       # 搜索区中心相对人脸中心下移 = 人脸高 × 该系数
        # 人脸检测时把帧缩到该短边再跑 Haar。原来 240 太小——3 米外人脸只剩 ~13px，
        # 低于 minSize 检不到 → 丢手后找不回。提高到 400 让 ~3-4 米的人脸仍可检出。
        # 越大越能识别更远的脸（恢复能力更强），但人脸扫描更慢。可在 config 调。
        self._face_detect_short = int(
            self._config.get("face_detect_short", 400) if self._config else 400)
        self._face_detector_init = False
        self._face_cascade = None

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

    def find_hands(self, frame, draw=True):
        """统一入口 — 检测 + crop-zoom 调度 + 平滑 + 排序 + 预测。"""
        h_frame, w_frame, _ = frame.shape

        # 1. 确定当前帧的缩放视口目标 (Target)
        if self._crop_zoom_mode and self._last_hint_center is not None and self._last_hint_size > 0:
            target_center = self._last_hint_center
            target_size = int(self._last_hint_size * self._crop_padding_ratio)
            # 不设人为下限：手越远裁剪框越小、放大倍率越高（手在检测帧中的占比恒为 1/padding）
            target_size = max(target_size, self._crop_min_size)
            target_size = min(target_size, min(w_frame, h_frame))
        else:
            target_center = (w_frame / 2.0, h_frame / 2.0)
            target_size = min(w_frame, h_frame)

        # 2. 对缩放视口应用 EMA 渐进式平滑过渡
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

        # 3. 决定是否使用 Crop-Zoom。若当前裁剪框接近全图大小（例如 >= 92%），直接使用 _detect 节省性能
        use_crop_zoom = self._current_crop_size < min(w_frame, h_frame) * 0.92

        # === 检测 ===
        if use_crop_zoom:
            hands_landmarks, hands_gestures, raw = self._detect_crop_zoom(
                frame, self._current_crop_center, self._current_crop_size,
            )
            if not hands_landmarks:
                self._zoom_miss_streak += 1
                if self._zoom_miss_streak >= self._zoom_miss_threshold:
                    self._crop_zoom_mode = False
                    self._last_sr_tier = None
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
            # 全帧没检到手 → 人脸引导捕获（解决远距离冷启动：手太小直接检不出）
            if not hands_landmarks:
                acq = self._try_face_guided_acquire(frame, w_frame, h_frame)
                if acq is not None:
                    hands_landmarks, hands_gestures, raw = acq

        # === 更新 zoom 模式 ===
        self._update_zoom_mode(hands_landmarks, hands_gestures, w_frame, h_frame)

        # === 有手检出：排序 + 平滑 + 幽灵手 ===
        if hands_landmarks:
            h, w, _ = frame.shape

            motion_map = self._update_motion(hands_landmarks, hands_gestures, w, h)
            order = sorted(
                range(len(hands_landmarks)),
                key=lambda i: self._priority_score(
                    hands_gestures[i], w, h,
                    landmarks=hands_landmarks[i],
                    motion=motion_map.get(
                        hands_gestures[i].get("handedness", "Unknown"), 0.0
                    ),
                ),
                reverse=True,
            )
            hands_landmarks = [hands_landmarks[i] for i in order]
            hands_gestures = [hands_gestures[i] for i in order]

            # zoom 视口只锁优先级最高的那只手（_priority_score 选出的"举得最高"者）。
            # 此前 hint 取所有手的并集包围盒：两只手并存时裁剪框在"单手↔双手跨度"间
            # 跳变，既锁不住上边的手、又使视口一缩一放（拉风箱）。排序后取 [:1] 即锁定手。
            hint = self._compute_hint_from_landmarks(hands_landmarks[:1])
            if hint[0] is not None:
                self._last_hint_center, self._last_hint_size = hint

            smoothed_all = []
            gesture_all = []
            seen_handedness = set()
            for landmarks, gesture in zip(hands_landmarks, hands_gestures):
                key = gesture.get("handedness", "Unknown")
                if key not in self.smoothers:
                    key = "Unknown"
                smoothed = self.smoothers[key].update(landmarks)
                seen_handedness.add(key)
                smoothed_all.append(smoothed)
                gesture_all.append(gesture)
                if draw:
                    self._draw_points_only(frame, smoothed, (255, 0, 255))

            # 幽灵手预测补帧
            missing_keys = self._active_handedness - seen_handedness
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
                seen_handedness.add(key)
                if draw:
                    self._draw_points_only(frame, ghost, (0, 255, 255))

            self._active_handedness = seen_handedness
            self.last_gestures = gesture_all

            if use_crop_zoom and self._current_crop_center:
                frame = self._apply_visual_zoom(frame, self._current_crop_center, self._current_crop_size)
            if draw:
                self._draw_zoom_badge(frame, gesture_all, w_frame, h_frame, used_zoom=use_crop_zoom)

            return frame, smoothed_all, gesture_all

        # === 全部丢失：预测 ===
        has_prediction = False
        predicted_all = []
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
                self._draw_points_only(frame, predicted_all[0], (0, 255, 255))
                cv2.putText(
                    frame, "Smoother Predict", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
                )
            if use_crop_zoom and self._current_crop_center:
                frame = self._apply_visual_zoom(frame, self._current_crop_center, self._current_crop_size)
            if draw:
                self._draw_zoom_badge(frame, self.last_gestures, w_frame, h_frame, used_zoom=use_crop_zoom)
            return frame, predicted_all, list(self.last_gestures)

        if any(sm.initialized for sm in self.smoothers.values()):
            logging.info("Tracking fully lost. Resetting smoothers.")
            for sm in self.smoothers.values():
                sm.reset()
            self.last_gestures = []
            self._active_handedness.clear()
            self._crop_zoom_mode = False
            self._last_sr_tier = None
            self._last_hint_center = None
            self._last_hint_size = 0
            self._far_streak = 0
            self._near_streak = 0
            self._zoom_miss_streak = 0
            self._current_crop_center = None
            self._current_crop_size = None

        if use_crop_zoom and self._current_crop_center:
            frame = self._apply_visual_zoom(frame, self._current_crop_center, self._current_crop_size)
        if draw:
            self._draw_zoom_badge(frame, [], w_frame, h_frame, used_zoom=use_crop_zoom)

        return frame, [], []

    # ------------------------------------------------------------------
    # 内部方法 — crop-zoom 状态机
    # ------------------------------------------------------------------

    def _compute_hint_from_landmarks(self, all_hands_landmarks):
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
                self._last_sr_tier = None
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

        for landmarks, gesture in zip(landmarks_list, gestures_list):
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
        score = 0.0
        handedness = gesture_meta.get("handedness", "Unknown")

        if self.dominant_hand in ("Left", "Right"):
            if handedness == self.dominant_hand:
                score += 1.0

        if frame_w > 0 and frame_h > 0:
            bbox_ratio = gesture_meta.get("bbox_area", 0.0) / (frame_w * frame_h)
            score += bbox_ratio * 8.0

        score += float(gesture_meta.get("handedness_score", 0.0))

        # 双手并存时以"谁抬得高"为主判据：wrist 的 y 越小（举得越高）分越高。
        # 权重(25)显著高于运动/手型/手大小，确保举起来的手稳定地成为锁定/放大对象。
        if landmarks and frame_h > 0:
            wrist_y_norm = max(0.0, min(1.0, float(landmarks[0][2]) / float(frame_h)))
            score += (1.0 - wrist_y_norm) * 25.0

        score += float(motion) * 6.0
        return score

    # ------------------------------------------------------------------
    # 绘图辅助
    # ------------------------------------------------------------------

    def _draw_points_only(self, frame, landmarks, color):
        for point in landmarks:
            cv2.circle(frame, (point[1], point[2]), 4, color, cv2.FILLED)

    def _draw_zoom_badge(self, frame, hands_gestures, frame_w, frame_h, used_zoom):
        try:
            if hands_gestures:
                max_bbox = max(g.get("bbox_area", 0.0) for g in hands_gestures)
                ratio_pct = (max_bbox / max(frame_w * frame_h, 1)) * 100
            else:
                ratio_pct = 0.0

            label = "ZOOM" if used_zoom else "FULL"
            color = (0, 200, 255) if used_zoom else (200, 200, 200)
            text = f"{label} {ratio_pct:.2f}%"

            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            pad = 6
            x1 = frame_w - tw - pad * 2 - 5
            y1 = 5
            x2 = frame_w - 5
            y2 = th + pad * 2 + 5

            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
            cv2.putText(
                frame, text, (x1 + pad, y2 - pad),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1,
            )
        except Exception:
            pass

    def _apply_visual_zoom(self, frame, crop_center, crop_size):
        h, w, _ = frame.shape
        cx, cy = crop_center

        base_size = int(round(crop_size))
        base_size = max(base_size, self._crop_min_size)
        base_size = min(base_size, min(w, h))

        if base_size >= int(min(w, h) * 0.95):
            return frame

        crop_h = base_size
        crop_w = int(crop_h * (w / h))
        if crop_w > w:
            crop_w = w
            crop_h = int(crop_w * (h / w))

        x0 = int(round(cx - crop_w / 2))
        y0 = int(round(cy - crop_h / 2))
        x0 = max(0, min(x0, w - crop_w))
        y0 = max(0, min(y0, h - crop_h))

        crop_img = frame[y0:y0+crop_h, x0:x0+crop_w]
        if crop_img.size > 0 and crop_h > 0 and crop_w > 0:
            zoomed = cv2.resize(crop_img, (w, h), interpolation=cv2.INTER_LINEAR)
            cv2.rectangle(zoomed, (0, 0), (w-1, h-1), (0, 200, 255), 6)
            return zoomed
        return frame

    def _init_sr_engines(self):
        """初始化超分辨率引擎状态（不在此处真正加载模型）。

        仅准备模型路径与占位变量；具体引擎在首次被选中时由
        _ensure_espcn() / _ensure_realesrgan() **按需加载**，避免一次性把
        ESPCN + Real-ESRGAN(GPU) + Real-ESRGAN(CPU) 三份模型全部常驻内存，
        同时支持运行时切换引擎时再加载所需的那一个。
        """
        if getattr(self, "_sr_initialized", False):
            return

        self._sr_initialized = True
        self._espcn_engine = None
        self._realesrgan_cpu_session = None
        self._realesrgan_gpu_session = None
        self._realesrgan_input_name = None
        self._realesrgan_gpu_available = None  # None=未探测, True/False=已探测

        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self._espcn_path = os.path.join(project_root, "ESPCN_x2.pb")
        self._realesrgan_path = os.path.join(project_root, "Real-ESRGAN_x2plus.onnx")

    def _ensure_espcn(self):
        """按需加载 ESPCN（OpenCV dnn_superres，CPU）。返回引擎或 None。"""
        if self._espcn_engine is not None:
            return self._espcn_engine
        if not os.path.exists(self._espcn_path):
            return None
        try:
            sr = cv2.dnn_superres.DnnSuperResImpl_create()
            sr.readModel(self._espcn_path)
            sr.setModel("espcn", 2)
            self._espcn_engine = sr
            _zoom_logger.info("[SR] ESPCN model loaded successfully.")
        except Exception as e:
            _zoom_logger.error("[SR] Failed to load ESPCN: %s", e)
            self._espcn_engine = None
        return self._espcn_engine

    def _ensure_realesrgan(self, prefer_gpu):
        """按需加载 Real-ESRGAN ONNX session。

        返回 (session, input_name)，不可用时返回 (None, None)。仅加载当前需要的
        provider；input_name 取自**真正加载成功**的 session（修复此前只在 CPU
        分支赋值导致 GPU-only 时被静默跳过的问题）。
        """
        if not os.path.exists(self._realesrgan_path):
            return None, None

        import onnxruntime as ort

        if prefer_gpu:
            if self._realesrgan_gpu_session is not None:
                return self._realesrgan_gpu_session, self._realesrgan_input_name
            if self._realesrgan_gpu_available is None:
                self._realesrgan_gpu_available = (
                    "DmlExecutionProvider" in ort.get_available_providers()
                )
            if self._realesrgan_gpu_available:
                try:
                    sess = ort.InferenceSession(
                        self._realesrgan_path,
                        providers=["DmlExecutionProvider", "CPUExecutionProvider"],
                    )
                    self._realesrgan_gpu_session = sess
                    self._realesrgan_input_name = sess.get_inputs()[0].name
                    _zoom_logger.info("[SR] Real-ESRGAN loaded on GPU (DirectML).")
                    return sess, self._realesrgan_input_name
                except Exception as e:
                    _zoom_logger.warning(
                        "[SR] Failed to init Real-ESRGAN on GPU, fallback to CPU: %s", e
                    )
                    self._realesrgan_gpu_available = False
            # GPU 不可用 → 回退 CPU

        if self._realesrgan_cpu_session is not None:
            return self._realesrgan_cpu_session, self._realesrgan_input_name
        try:
            sess = ort.InferenceSession(
                self._realesrgan_path, providers=["CPUExecutionProvider"]
            )
            self._realesrgan_cpu_session = sess
            self._realesrgan_input_name = sess.get_inputs()[0].name
            _zoom_logger.info("[SR] Real-ESRGAN loaded on CPU.")
            return sess, self._realesrgan_input_name
        except Exception as e:
            _zoom_logger.error("[SR] Failed to load Real-ESRGAN session: %s", e)
            return None, None

    def _resolve_sr_engine(self, sr_engine, crop_size, target):
        """将配置（含 auto）解析为具体引擎名。

        进入本函数即说明正处于 crop-zoom（手部在画面中太小、需要放大），因此 auto
        一律选用轻量的 ESPCN——"需要 ZOOM 就必须超分"。即便裁剪框 >= 目标尺寸
        （ZOOM 过程中手部短暂靠近的过渡场景）也不再回退 none，保证整段 ZOOM 档位
        稳定。none（纯插值）只在两种情况出现：用户显式把 zoom_sr_engine 设为
        "none"，或所选引擎执行失败的兜底；不触发 ZOOM 时根本不进入这里，手部无需
        放大也就不超分。Real-ESRGAN 开销大、可能掉帧，仍需用户显式选择。

        例外（近距离关闭 SR）：当 crop_size >= target 时，裁剪框本就 ≥ 目标分辨率，
        这是"下采样"场景——超分加不了任何细节，还白白吃 CPU（实测每段 ZOOM 的
        crop 常达 1000~1300px 远大于 384px 目标）。此时 auto 直接用普通插值，
        等效"手大/近距离不超分"，省下的算力直接体现在帧率上。真正远（crop<target，
        需要上采样放大小手）才用 ESPCN。
        """
        if sr_engine == "auto":
            if crop_size >= target:
                return "none"
            return "espcn"
        return sr_engine

    def _log_sr_tier(self, tier, crop_size, target):
        """记录本次 crop-zoom 实际生效的放大档位（ESPCN / Real-ESRGAN / none 插值）。

        仅在档位变化时打印，避免逐帧刷屏；每次 ZOOM OFF 会复位 _last_sr_tier，
        因此每段 ZOOM 会重新记录一次。
        """
        if tier == self._last_sr_tier:
            return
        self._last_sr_tier = tier
        _zoom_logger.info(
            "[SR] zoom upscaler -> %s (crop=%dpx, target=%dpx)", tier, crop_size, target
        )

    def _sr_espcn(self, crop, target):
        """ESPCN 超分：限幅输入到约 target/2，2x 放大后重采样到 target。

        ESPCN 开销随输入像素数增长（240² 在 CPU 上约 30ms）。由于其放大倍率固定为
        2，把输入限制到约 target/2 既能让输出≈target、又能把单帧开销压到固定上限内
        （≈target/2 输入约 18ms）；当 crop 本就更小则直接喂入不再下采样。
        失败/不可用返回 None。
        """
        engine = self._ensure_espcn()
        if engine is None:
            return None
        try:
            cap = max(64, target // 2)
            src = crop
            longest = max(crop.shape[0], crop.shape[1])
            if longest > cap:
                s = cap / float(longest)
                src = cv2.resize(
                    crop,
                    (max(1, int(round(crop.shape[1] * s))), max(1, int(round(crop.shape[0] * s)))),
                    interpolation=cv2.INTER_AREA,
                )
            out = engine.upsample(src)  # 2x
            if out.shape[0] == target and out.shape[1] == target:
                return out
            interp = cv2.INTER_AREA if out.shape[0] > target else cv2.INTER_LINEAR
            return cv2.resize(out, (target, target), interpolation=interp)
        except Exception as e:
            _zoom_logger.error("[SR] ESPCN upsampling failed: %s", e)
            return None

    def _sr_realesrgan(self, crop, target, prefer_gpu):
        """Real-ESRGAN 超分（固定 64x64 输入的导出）。

        该 ONNX 导出的空间输入被写死为 64x64，若像旧实现那样把整张 crop 全局
        下采样到 64 再 2x，会先丢掉已有分辨率、效果常不如双线性。这里改用
        **分块批量推理**绕开 64 的天花板：把 crop 缩放到 grid*64 的方形后切成
        grid*grid 个 64x64 tile，一次 batch 推理（模型 batch 维为动态），再把
        各 128x128 输出拼接成方形并重采样到 target。模型实际"看到"的有效分辨率
        提升到 grid*64。失败/不可用返回 None。
        """
        session, input_name = self._ensure_realesrgan(prefer_gpu)
        if session is None or input_name is None:
            return None
        try:
            TILE = 64
            # 选择网格数，使输出 grid*128 尽量贴近 target（模型放大倍率为 2）
            grid = max(1, int(round(target / (TILE * 2))))
            side_in = grid * TILE

            interp_in = cv2.INTER_AREA if crop.shape[0] > side_in else cv2.INTER_CUBIC
            crop_sq = cv2.resize(crop, (side_in, side_in), interpolation=interp_in)

            tiles = []
            for gy in range(grid):
                for gx in range(grid):
                    tiles.append(crop_sq[gy * TILE:(gy + 1) * TILE, gx * TILE:(gx + 1) * TILE])
            batch = np.stack(tiles, axis=0).astype(np.float32) / 255.0  # (N,64,64,3)
            batch = np.transpose(batch, (0, 3, 1, 2))                   # (N,3,64,64)

            out = session.run(None, {input_name: batch})[0]            # (N,3,h,w)
            out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
            out = np.transpose(out, (0, 2, 3, 1))                      # (N,h,w,3)

            th, tw = out.shape[1], out.shape[2]
            canvas = np.empty((grid * th, grid * tw, 3), dtype=np.uint8)
            k = 0
            for gy in range(grid):
                for gx in range(grid):
                    canvas[gy * th:(gy + 1) * th, gx * tw:(gx + 1) * tw] = out[k]
                    k += 1

            if canvas.shape[0] == target and canvas.shape[1] == target:
                return canvas
            interp_out = cv2.INTER_AREA if canvas.shape[0] > target else cv2.INTER_LINEAR
            return cv2.resize(canvas, (target, target), interpolation=interp_out)
        except Exception as e:
            _zoom_logger.error("[SR] Real-ESRGAN upsampling failed: %s", e)
            return None

    # ------------------------------------------------------------------
    # 人脸引导的远距离手部捕获
    # ------------------------------------------------------------------

    def _ensure_face_detector(self):
        """按需加载 OpenCV Haar 人脸级联（随 opencv 自带，无需额外下载）。"""
        if self._face_detector_init:
            return self._face_cascade
        self._face_detector_init = True
        try:
            cascade_dir = getattr(getattr(cv2, "data", None), "haarcascades", None)
            if cascade_dir:
                path = os.path.join(cascade_dir, "haarcascade_frontalface_default.xml")
                if os.path.exists(path):
                    c = cv2.CascadeClassifier(path)
                    if not c.empty():
                        self._face_cascade = c
                        _zoom_logger.info("[FACE] Haar 人脸级联已加载（远距捕获）。")
        except Exception as e:
            _zoom_logger.warning("[FACE] 人脸级联不可用: %s", e)
        return self._face_cascade

    def _face_guided_region(self, frame):
        """检测最大人脸，据此预测手部搜索区。返回 (cx, cy, size) 或 None。"""
        cascade = self._ensure_face_detector()
        if cascade is None:
            return None
        h, w = frame.shape[:2]
        # 缩到 _face_detect_short 短边做人脸检测以提速（越大越能检出更远/更小的脸）。
        short = min(w, h)
        target_short = self._face_detect_short
        scale = target_short / short if short > target_short else 1.0
        small = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale < 1.0 else frame
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        try:
            # scaleFactor 1.1（更细金字塔，识别更多尺寸）、minNeighbors 3、minSize 12
            # 都比原来更宽松，专为"远距离小脸"放行，提升 3 米外的恢复成功率。
            faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(12, 12))
        except Exception:
            return None
        if len(faces) == 0:
            return None
        fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        # 映射回原帧坐标
        fx, fy, fw, fh = fx / scale, fy / scale, fw / scale, fh / scale
        face_cx = fx + fw / 2.0
        face_cy = fy + fh / 2.0
        region_size = fh * self._face_hand_region_scale
        region_cx = face_cx
        region_cy = face_cy + fh * self._face_hand_down_bias
        return (region_cx, region_cy, region_size)

    def _try_face_guided_acquire(self, frame, w, h):
        """丢手时的人脸引导捕获：节流跑人脸 → 预测搜索区 → crop-zoom 检测。

        命中则返回 (hands_landmarks, hands_gestures, raw) 并直接进入 ZOOM 跟踪；
        否则返回 None。任何异常都被吞掉，绝不影响主检测流程。
        """
        if not self._face_acquire_enabled:
            return None
        self._face_scan_counter += 1
        if self._face_scan_counter < self._face_scan_interval:
            return None
        self._face_scan_counter = 0
        try:
            region = self._face_guided_region(frame)
            if region is None:
                return None
            cx, cy, size = region
            size = max(self._crop_min_size, min(size, min(w, h)))
            res = self._detect_crop_zoom(frame, (cx, cy), size)
            if res and res[0]:
                # 命中：直接进入 ZOOM，让下一帧用 hint 跟踪，捕获更跟手
                self._crop_zoom_mode = True
                self._zoom_miss_streak = 0
                self._current_crop_center = (cx, cy)
                self._current_crop_size = float(size)
                _zoom_logger.info("=> ACQUIRE (人脸引导 crop-zoom 捕获到手)")
                return res
        except Exception as e:
            _zoom_logger.debug("[FACE] 人脸引导捕获异常: %s", e)
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
        self._init_sr_engines()

        target = self._crop_target_size

        # 解析 auto / 显式选择 → 具体引擎
        actual_engine = self._resolve_sr_engine(sr_engine, crop_size, target)

        # 执行放大
        zoomed = None
        if actual_engine == "espcn":
            zoomed = self._sr_espcn(crop, target)
        elif actual_engine in ("realesrgan_cpu", "realesrgan_gpu"):
            zoomed = self._sr_realesrgan(
                crop, target, prefer_gpu=(actual_engine == "realesrgan_gpu")
            )

        # 兜底：超分关闭或执行失败 → 普通插值（放大用 LINEAR，缩小用 AREA）
        if zoomed is None:
            interp = cv2.INTER_LINEAR if crop_size < target else cv2.INTER_AREA
            zoomed = cv2.resize(crop, (target, target), interpolation=interp)
            # 实际生效档位：显式 none 用插值；否则说明所选引擎执行失败、回退插值
            effective_tier = (
                "none(interp)" if actual_engine == "none"
                else "none(interp,%s_failed)" % actual_engine
            )
        else:
            effective_tier = actual_engine
        self._log_sr_tier(effective_tier, crop_size, target)

        # 子类在 zoomed 上跑推理
        detection_result = run_sub_detect(zoomed)

        # 坐标映射回原帧
        scale = crop_size / target

        def to_orig(norm_x, norm_y):
            return (x0 + norm_x * target * scale, y0 + norm_y * target * scale)

        return detection_result, (x0, y0, crop_size, target, scale), to_orig
