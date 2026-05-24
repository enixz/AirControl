import logging
import os
import sys
import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

ML_GESTURE_TO_INTERNAL = {
    "Closed_Fist": "FIST",
    "Open_Palm": "OPEN",
    "Pointing_Up": "POINTING_UP",
    "Thumb_Up": "THUMB_UP",
    "Thumb_Down": "THUMB_DOWN",
    "Victory": "VICTORY",
    "ILoveYou": "I_LOVE_YOU",
    "None": "OTHER",
}

class KalmanSmoother:
    """卡尔曼滤波 + EMA 双重平滑器，为 21 个关键点各自维护 [x,y,vx,vy] 状态"""

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


class HandTracker:
    # 按 handedness 标签索引平滑器，避免帧序变化破坏卡尔曼状态
    HAND_KEYS = ("Left", "Right", "Unknown")

    def __init__(
        self,
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        preferred_model_type="Heavy",
        dominant_hand="Right",
    ):
        # 注：之前版本这三个阈值是硬编码 0.75/0.65/0.6，构造器参数静默被忽略——bug 已修。
        # 现在默认下调到 0.6/0.5/0.5，对 1m+ 远距离小手识别率显著提升；
        # 配合主控手优先级，副手误识也被排到 index 1+，不影响主控手稳定性。
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.model_path = self._resolve_model_path(project_root, preferred_model_type)
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.GestureRecognizerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE if static_image_mode else vision.RunningMode.VIDEO,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.detector = vision.GestureRecognizer.create_from_options(options)

        self.mp_hands = mp.solutions.hands if hasattr(mp, 'solutions') else None
        self.mp_draw = mp.solutions.drawing_utils if hasattr(mp, 'solutions') else None

        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        # 主控手策略："Right" / "Left" / "Auto"（Auto 完全靠运动+高度）
        self.dominant_hand = dominant_hand
        # 平滑器改为按 handedness 标签索引，排序时不会错位
        self.smoothers = {k: KalmanSmoother() for k in self.HAND_KEYS}
        self.last_gestures = []
        self._active_handedness = set()

        # 运动追踪：按 handedness 维护手腕位置 + EMA 运动量
        # 用于自动判别"哪只手在操作"——垂在腰间的手运动量≈0，自动被压低优先级
        self._last_wrist_pos = {}  # key -> (x, y, time_ms)
        self._motion_ema = {k: 0.0 for k in self.HAND_KEYS}
        self._motion_alpha = 0.4  # 新样本权重（约 3 帧平滑窗口）
        self._motion_window_ms = 500  # 超过这个时长视为"重新出现"，不算运动

        logging.info("HandTracker model loaded: %s (dominant=%s)", os.path.basename(self.model_path), self.dominant_hand)

    def set_dominant_hand(self, dominant_hand):
        """运行时切换惯用手设置。"""
        self.dominant_hand = dominant_hand
        logging.info("HandTracker dominant_hand 切换为 %s", dominant_hand)

    def _resolve_model_path(self, project_root, preferred_model_type):
        if getattr(sys, 'frozen', False):
            base_dir = sys._MEIPASS
        else:
            base_dir = project_root

        gesture_candidates = [
            os.path.join(base_dir, "gesture_recognizer.task"),
        ]
        heavy_candidates = [
            os.path.join(base_dir, "hand_landmarker_heavy.task"),
            os.path.join(base_dir, "hand_landmarker_full.task"),
        ]
        lite_candidates = [
            os.path.join(base_dir, "hand_landmarker.task"),
        ]

        if str(preferred_model_type).lower() == "lite":
            candidates = gesture_candidates + lite_candidates + heavy_candidates
        else:
            candidates = gesture_candidates + heavy_candidates + lite_candidates
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        raise FileNotFoundError(f"未找到手部模型文件，搜索路径: {candidates}")



    def _draw_points_only(self, frame, landmarks, color):
        for point in landmarks:
            cv2.circle(frame, (point[1], point[2]), 4, color, cv2.FILLED)

    def _draw_landmarks(self, frame, hand_landmarks_list, landmarks):
        if self.mp_draw and self.mp_hands and hand_landmarks_list is not None:
            from mediapipe.framework.formats import landmark_pb2

            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
                for landmark in hand_landmarks_list
            ])
            self.mp_draw.draw_landmarks(
                frame,
                hand_landmarks_proto,
                self.mp_hands.HAND_CONNECTIONS
            )
        else:
            self._draw_points_only(frame, landmarks, (255, 0, 255))

    def _detect(self, frame):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

        if self.static_image_mode:
            detection_result = self.detector.recognize(mp_image)
        else:
            timestamp_ms = int(time.time() * 1000)
            detection_result = self.detector.recognize_for_video(mp_image, timestamp_ms)

        hands_landmarks = []
        hands_gestures = []
        raw_hand_lists = []
        if detection_result and detection_result.hand_landmarks:
            h, w, _ = frame.shape
            min_area = w * h * 0.005
            for i, hand_landmarks_list in enumerate(detection_result.hand_landmarks):
                xs = [lm.x for lm in hand_landmarks_list]
                ys = [lm.y for lm in hand_landmarks_list]
                box_w = (max(xs) - min(xs)) * w
                box_h = (max(ys) - min(ys)) * h
                bbox_area = box_w * box_h
                if bbox_area < min_area:
                    continue

                landmarks = []
                for idx, landmark in enumerate(hand_landmarks_list):
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    landmarks.append([idx, cx, cy])
                hands_landmarks.append(landmarks)
                raw_hand_lists.append(hand_landmarks_list)

                # handedness: "Left" / "Right" / "Unknown"，附带置信度
                handedness_name = "Unknown"
                handedness_score = 0.0
                if detection_result.handedness and i < len(detection_result.handedness):
                    cat_list = detection_result.handedness[i]
                    if cat_list:
                        cat = cat_list[0]
                        handedness_name = cat.category_name
                        handedness_score = float(cat.score)

                if detection_result.gestures and i < len(detection_result.gestures):
                    gesture = detection_result.gestures[i][0]
                    internal_label = ML_GESTURE_TO_INTERNAL.get(gesture.category_name, "OTHER")
                    hands_gestures.append({
                        "ml_label": gesture.category_name,
                        "label": internal_label,
                        "score": gesture.score,
                        "handedness": handedness_name,
                        "handedness_score": handedness_score,
                        "bbox_area": bbox_area,
                    })
                else:
                    hands_gestures.append({
                        "ml_label": "None",
                        "label": "OTHER",
                        "score": 0.0,
                        "handedness": handedness_name,
                        "handedness_score": handedness_score,
                        "bbox_area": bbox_area,
                    })

        return hands_landmarks, hands_gestures, raw_hand_lists

    def _update_motion(self, landmarks_list, gestures_list, frame_w, frame_h):
        """更新每只手的运动量 EMA，返回 {handedness: motion 0~1} 映射。

        运动量定义：手腕（landmark 0）帧间位移 / 画面对角线 × 系数；
        垂在腰间不动的手 motion≈0，被排序时压到 index 1+ 。
        """
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

            wrist = landmarks[0]  # landmark 0 = 手腕
            cx, cy = float(wrist[1]), float(wrist[2])
            prev = self._last_wrist_pos.get(key)

            if prev is not None and (now_ms - prev[2]) < self._motion_window_ms:
                dx = cx - prev[0]
                dy = cy - prev[1]
                dist = (dx * dx + dy * dy) ** 0.5
                # 对角线 5% 的位移 = 饱和（约等于挥动的中等幅度）
                sample = min(dist / (diag * 0.05), 1.0)
            else:
                sample = 0.0  # 重新出现，先记位置不算运动

            self._motion_ema[key] = (
                (1.0 - self._motion_alpha) * self._motion_ema[key]
                + self._motion_alpha * sample
            )
            self._last_wrist_pos[key] = (cx, cy, now_ms)
            motion_map[key] = self._motion_ema[key]

        # 本帧未出现的手：运动 EMA 衰减（如果有手丢失但留有残值）
        for key in self._motion_ema:
            if key not in seen_keys:
                self._motion_ema[key] *= (1.0 - self._motion_alpha)

        return motion_map

    def _priority_score(self, gesture_meta, frame_w, frame_h, landmarks=None, motion=0.0):
        """主控手优先级评分，分数越高越优先放在 index 0。

        权重原则——按用户要求 "用近的用高的为主，运动为辅"：
          • 近 (bbox)     最大可贡献 ~+8（演讲者通常占画面 10%+）
          • 高 (height)   最大可贡献 ~+12（抬起 +4 / 不在腰部 -8 的双向力）
          • 运动 (motion) 最大 +6（活跃手的额外提示，不再"一票否决"）
          • 惯用手匹配     +1（只在前三者打平时起作用，dominant=Auto 时为 0）

        典型对比：
          • 左手抬起操作 + 右手垂腰：左 ~+8  vs  右 ~-7  →  左手胜
          • 演讲者(近) + 听众(远)：    演讲 ~+5 vs 听众 ~+1  →  演讲者胜
          • 两手都抬同高度，只一只动：差 ~+5 →  动的胜
        """
        score = 0.0
        handedness = gesture_meta.get("handedness", "Unknown")

        # 1. 惯用手匹配——仅当配置成 Left/Right 时作平局打破
        if self.dominant_hand in ("Left", "Right"):
            if handedness == self.dominant_hand:
                score += 1.0

        # 2. 近 (bbox 占画面)——主信号之一
        if frame_w > 0 and frame_h > 0:
            bbox_ratio = gesture_meta.get("bbox_area", 0.0) / (frame_w * frame_h)
            score += bbox_ratio * 8.0

        # 3. handedness 置信度
        score += float(gesture_meta.get("handedness_score", 0.0))

        # 4/5. 高——主信号之一：抬起加分，腰部以下重罚
        if landmarks and frame_h > 0:
            wrist_y_norm = float(landmarks[0][2]) / float(frame_h)
            if wrist_y_norm < 0.5:
                score += (0.5 - wrist_y_norm) * 8.0  # 顶部最多 +4
            if wrist_y_norm > 0.6:
                score -= (wrist_y_norm - 0.6) * 20.0  # 底部最多 -8

        # 6. 运动量——辅助信号
        score += float(motion) * 6.0

        return score

    def find_hands(self, frame, draw=True):
        hands_landmarks, hands_gestures, raw_hand_lists = self._detect(frame)

        if hands_landmarks:
            h, w, _ = frame.shape

            # 1. 先更新每只手的运动 EMA，再按主控手优先级重排
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
            raw_hand_lists = [raw_hand_lists[i] for i in order]
            hands_gestures = [hands_gestures[i] for i in order]

            # 2. 平滑器按 handedness 标签查找，不再随 index 漂移
            smoothed_all = []
            gesture_all = []
            seen_handedness = set()
            for landmarks, raw_hand, gesture in zip(hands_landmarks, raw_hand_lists, hands_gestures):
                key = gesture.get("handedness", "Unknown")
                if key not in self.smoothers:
                    key = "Unknown"
                smoothed = self.smoothers[key].update(landmarks)
                seen_handedness.add(key)
                smoothed_all.append(smoothed)
                gesture_all.append(gesture)
                if draw:
                    self._draw_points_only(frame, smoothed, (255, 0, 255))

            # 3. 双手抓握恢复：若历史上活跃的另一只手本帧没出现（NMS 合并 / 短暂遮挡）
            #    用卡尔曼预测补一个"幽灵手"。最多 5 帧，超时不再补。
            #    幽灵手打上 predicted=True，下游可选择性使用。
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
                    self._draw_points_only(frame, ghost, (0, 255, 255))  # 黄色 = 幽灵

            self._active_handedness = seen_handedness
            self.last_gestures = gesture_all
            return frame, smoothed_all, gesture_all

        # 全部丢失：只对最近活跃过的 handedness 做卡尔曼预测
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
                    frame,
                    "Kalman Predict",
                    (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )
            return frame, predicted_all, list(self.last_gestures)

        if any(sm.initialized for sm in self.smoothers.values()):
            logging.info("Tracking fully lost. Resetting Kalman smoothers.")
            for sm in self.smoothers.values():
                sm.reset()
            self.last_gestures = []
            self._active_handedness.clear()

        return frame, [], []
