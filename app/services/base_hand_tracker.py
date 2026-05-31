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
# BaseHandTracker — 抽象基类
# ---------------------------------------------------------------------------

class BaseHandTracker(ABC):
    """手部追踪器抽象基类。

    子类必须实现：
      - _detect(frame)                → (hands_landmarks, hands_gestures, raw_data)
      - _detect_crop_zoom(frame, cx, cy, size) → 同上，或 ([], [], []) 触发回退
      - engine_name (property)        → str，引擎标识（用于日志和调试）

    find_hands() 的完整编排逻辑由基类提供，下游代码无需感知引擎差异。
    """

    HAND_KEYS = ("Left", "Right", "Unknown")

    def __init__(self, max_num_hands=2, dominant_hand="Right"):
        self.max_num_hands = max_num_hands
        self.dominant_hand = dominant_hand

        # 平滑器（按 handedness 标签索引，避免帧序变化破坏卡尔曼状态）
        self.smoothers = {k: KalmanSmoother() for k in self.HAND_KEYS}
        self.last_gestures = []
        self._active_handedness = set()

        # 运动追踪
        self._last_wrist_pos = {}
        self._motion_ema = {k: 0.0 for k in self.HAND_KEYS}
        self._motion_alpha = 0.4
        self._motion_window_ms = 500

        # === Crop-zoom 远距离增强 ===
        self._crop_zoom_mode = False
        self._crop_padding_ratio = 2.5
        self._crop_target_size = 384
        self._zoom_near_threshold = 0.018
        self._zoom_far_threshold = 0.010
        self._zoom_switch_streak = 3
        self._near_streak = 0
        self._far_streak = 0
        self._zoom_miss_streak = 0
        self._zoom_miss_threshold = 5
        self._last_hint_center = None
        self._last_hint_size = 0

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
        use_crop_zoom = (
            self._crop_zoom_mode
            and self._last_hint_center is not None
            and self._last_hint_size > 0
        )

        # === 检测 ===
        if use_crop_zoom:
            hands_landmarks, hands_gestures, raw = self._detect_crop_zoom(
                frame, self._last_hint_center, self._last_hint_size,
            )
            if not hands_landmarks:
                self._zoom_miss_streak += 1
                hands_landmarks, hands_gestures, raw = self._detect(frame)
                if self._zoom_miss_streak >= self._zoom_miss_threshold:
                    self._crop_zoom_mode = False
                    self._zoom_miss_streak = 0
                    self._far_streak = 0
                    _zoom_logger.info(
                        "=> ZOOM OFF (%s: crop-zoom 连续 %d 帧未检出，退回全帧)",
                        self.engine_name, self._zoom_miss_threshold,
                    )
            else:
                self._zoom_miss_streak = 0
        else:
            hands_landmarks, hands_gestures, raw = self._detect(frame)

        # === 更新 zoom 模式 + hint ===
        h_frame, w_frame, _ = frame.shape
        self._update_zoom_mode(hands_landmarks, hands_gestures, w_frame, h_frame)

        hint = self._compute_hint_from_landmarks(hands_landmarks)
        if hint[0] is not None:
            self._last_hint_center, self._last_hint_size = hint

        if draw:
            self._draw_zoom_badge(frame, hands_gestures, w_frame, h_frame, used_zoom=use_crop_zoom)

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
            return frame, smoothed_all, gesture_all

        # === 全部丢失：卡尔曼预测 ===
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
                    frame, "Kalman Predict", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
                )
            return frame, predicted_all, list(self.last_gestures)

        if any(sm.initialized for sm in self.smoothers.values()):
            logging.info("Tracking fully lost. Resetting Kalman smoothers.")
            for sm in self.smoothers.values():
                sm.reset()
            self.last_gestures = []
            self._active_handedness.clear()
            self._crop_zoom_mode = False
            self._last_hint_center = None
            self._last_hint_size = 0
            self._far_streak = 0
            self._near_streak = 0
            self._zoom_miss_streak = 0

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
                _zoom_logger.info(
                    "=> ZOOM ON (%s bbox=%.2f%% < %.2f%% far_threshold)",
                    self.engine_name, ratio * 100, self._zoom_far_threshold * 100,
                )
        elif ratio > self._zoom_near_threshold:
            self._near_streak += 1
            self._far_streak = 0
            if self._crop_zoom_mode and self._near_streak >= self._zoom_switch_streak:
                self._crop_zoom_mode = False
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

        if landmarks and frame_h > 0:
            wrist_y_norm = float(landmarks[0][2]) / float(frame_h)
            if wrist_y_norm < 0.5:
                score += (0.5 - wrist_y_norm) * 8.0
            if wrist_y_norm > 0.6:
                score -= (wrist_y_norm - 0.6) * 20.0

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

    def _perform_crop_zoom(self, frame, hint_center, hint_size, run_sub_detect):
        """通用 crop-zoom 逻辑：裁剪 → 放大 → 子类检测 → 坐标映射。

        Args:
            frame, hint_center, hint_size: 同 _detect_crop_zoom
            run_sub_detect: callable(cropped_bgr) → detection_result
                子类提供在裁剪区域上跑推理的回调

        Returns:
            (hands_landmarks, hands_gestures, raw_data) 或 ([], [], [])
        """
        h, w, _ = frame.shape

        crop_size = int(hint_size * self._crop_padding_ratio)
        crop_size = max(crop_size, 240)
        crop_size = min(crop_size, min(w, h))

        if crop_size >= int(min(w, h) * 0.85):
            return [], [], []

        cx, cy = hint_center
        x0 = int(round(cx - crop_size / 2))
        y0 = int(round(cy - crop_size / 2))
        x0 = max(0, min(x0, w - crop_size))
        y0 = max(0, min(y0, h - crop_size))

        crop = frame[y0:y0 + crop_size, x0:x0 + crop_size]
        if crop.size == 0 or crop.shape[0] < 60 or crop.shape[1] < 60:
            return [], [], []

        target = self._crop_target_size
        zoomed = cv2.resize(crop, (target, target), interpolation=cv2.INTER_LINEAR)

        # 子类在 zoomed 上跑推理
        detection_result = run_sub_detect(zoomed)

        # 坐标映射回原帧
        scale = crop_size / target

        def to_orig(norm_x, norm_y):
            return (x0 + norm_x * target * scale, y0 + norm_y * target * scale)

        return detection_result, (x0, y0, crop_size, target, scale), to_orig
