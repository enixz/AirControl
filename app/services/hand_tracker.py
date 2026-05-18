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
    def __init__(
        self,
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
        preferred_model_type="Heavy",
    ):
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.model_path = self._resolve_model_path(project_root, preferred_model_type)
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.GestureRecognizerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE if static_image_mode else vision.RunningMode.VIDEO,
            num_hands=max_num_hands,
            min_hand_detection_confidence=0.75,
            min_hand_presence_confidence=0.65,
            min_tracking_confidence=0.6,
        )
        self.detector = vision.GestureRecognizer.create_from_options(options)

        self.mp_hands = mp.solutions.hands if hasattr(mp, 'solutions') else None
        self.mp_draw = mp.solutions.drawing_utils if hasattr(mp, 'solutions') else None

        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.smoothers = [KalmanSmoother() for _ in range(max_num_hands)]
        self.last_gestures = []
        self._active_hand_indices = set()

        logging.info("HandTracker model loaded: %s", os.path.basename(self.model_path))

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
                if box_w * box_h < min_area:
                    continue

                landmarks = []
                for idx, landmark in enumerate(hand_landmarks_list):
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    landmarks.append([idx, cx, cy])
                hands_landmarks.append(landmarks)
                raw_hand_lists.append(hand_landmarks_list)

                if detection_result.gestures and i < len(detection_result.gestures):
                    gesture = detection_result.gestures[i][0]
                    internal_label = ML_GESTURE_TO_INTERNAL.get(gesture.category_name, "OTHER")
                    hands_gestures.append({
                        "ml_label": gesture.category_name,
                        "label": internal_label,
                        "score": gesture.score,
                    })
                else:
                    hands_gestures.append({"ml_label": "None", "label": "OTHER", "score": 0.0})

        return hands_landmarks, hands_gestures, raw_hand_lists

    def find_hands(self, frame, draw=True):
        hands_landmarks, hands_gestures, raw_hand_lists = self._detect(frame)

        if hands_landmarks:
            smoothed_all = []
            gesture_all = []
            self._active_hand_indices.clear()
            for i, (landmarks, raw_hand) in enumerate(zip(hands_landmarks, raw_hand_lists)):
                gesture = hands_gestures[i] if i < len(hands_gestures) else {"ml_label": "None", "label": "OTHER", "score": 0.0}
                if i < len(self.smoothers):
                    smoothed = self.smoothers[i].update(landmarks)
                    self._active_hand_indices.add(i)
                else:
                    smoothed = landmarks
                smoothed_all.append(smoothed)
                gesture_all.append(gesture)
                if draw:
                    self._draw_points_only(frame, smoothed, (255, 0, 255))

            self.last_gestures = gesture_all
            return frame, smoothed_all, gesture_all

        has_prediction = False
        predicted_all = []
        # 只对最近活跃过的手进行卡尔曼预测，避免未初始化的平滑器产生幻影手
        for i in sorted(self._active_hand_indices):
            if i >= len(self.smoothers):
                continue
            predicted = self.smoothers[i].predict()
            if predicted is not None:
                predicted_all.append(predicted)
                has_prediction = True

        if has_prediction:
            if draw:
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

        if any(sm.initialized for sm in self.smoothers):
            logging.info("Tracking fully lost. Resetting Kalman smoothers.")
            for sm in self.smoothers:
                sm.reset()
            self.last_gestures = []
            self._active_hand_indices.clear()

        return frame, [], []
