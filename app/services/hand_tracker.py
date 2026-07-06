"""MediaPipe 手部追踪器 — 基于 GestureRecognizer Task API。

继承 BaseHandTracker，实现 MediaPipe 特有的检测逻辑。
公共逻辑（KalmanSmoother、crop-zoom 状态机、运动追踪、主控手排序）
全部由 base_hand_tracker.py 提供。
"""

import logging
import os
import sys
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 触发 gesture_recognizer 模块加载
from . import gesture_recognizer as _gr  # noqa: F401
from .base_hand_tracker import (  # noqa: F401 — KalmanSmoother 供外部 import 兼容
    BaseHandTracker,
    KalmanSmoother,
)

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


class HandTracker(BaseHandTracker):
    """MediaPipe GestureRecognizer 手部追踪器。

    实现 BaseHandTracker 的 _detect() 和 _detect_crop_zoom() 抽象方法。
    其余逻辑（find_hands、平滑、排序、crop-zoom 状态机）由基类提供。
    """

    def __init__(
        self,
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        preferred_model_type="Heavy",
        dominant_hand="Right",
        config=None,
    ):
        super().__init__(max_num_hands=max_num_hands, dominant_hand=dominant_hand, config=config)

        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.model_path = self._resolve_model_path(project_root, preferred_model_type)

        # 根据文件名判断是 HandLandmarker 还是 GestureRecognizer
        is_landmarker = "landmarker" in os.path.basename(self.model_path).lower()

        base_options = python.BaseOptions(model_asset_path=self.model_path)
        if is_landmarker:
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE if static_image_mode else vision.RunningMode.VIDEO,
                num_hands=max_num_hands,
                min_hand_detection_confidence=min_detection_confidence,
                min_hand_presence_confidence=min_presence_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            self.detector = vision.HandLandmarker.create_from_options(options)
            self._is_pure_landmarker = True
        else:
            options = vision.GestureRecognizerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE if static_image_mode else vision.RunningMode.VIDEO,
                num_hands=max_num_hands,
                min_hand_detection_confidence=min_detection_confidence,
                min_hand_presence_confidence=min_presence_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            self.detector = vision.GestureRecognizer.create_from_options(options)
            self._is_pure_landmarker = False

        self.mp_hands = mp.solutions.hands if hasattr(mp, 'solutions') else None
        self.mp_draw = mp.solutions.drawing_utils if hasattr(mp, 'solutions') else None

        self.static_image_mode = static_image_mode
        self._last_mp_timestamp = 0

        logging.info("HandTracker model loaded: %s (dominant=%s)",
                     os.path.basename(self.model_path), self.dominant_hand)

    # ------------------------------------------------------------------
    # BaseHandTracker 抽象接口实现
    # ------------------------------------------------------------------

    @property
    def engine_name(self) -> str:
        return "mediapipe"

    def _detect(self, frame):
        """全帧 MediaPipe 检测。

        高分辨率帧（如 1080p）先降采样到 _inference_max_width 再喂给 MediaPipe：
        模型返回归一化坐标，故仍用原始 w/h 反算像素，坐标系不变、无需补偿。
        实测把 1080p 整帧推理 ~42ms 降回 ~15ms，直接改善"快速移动跟手"。
        """
        h, w, _ = frame.shape
        infer_frame = self._downscale_for_inference(frame, w)
        detection_result = self._run_recognizer(infer_frame)
        return self._extract_results(detection_result, w, h)

    def _downscale_for_inference(self, frame, w):
        """按 _inference_max_width 等比缩小帧（仅当原宽更大）。INTER_AREA 适合缩小。"""
        max_w = getattr(self, "_inference_max_width", 720)
        if max_w and max_w > 0 and w > max_w:
            scale = max_w / float(w)
            new_h = max(1, int(round(frame.shape[0] * scale)))
            return cv2.resize(frame, (int(max_w), new_h), interpolation=cv2.INTER_AREA)
        return frame

    def _detect_crop_zoom(self, frame, crop_center, crop_size):
        """裁剪放大 → MediaPipe → 坐标映射回原帧。"""
        h, w, _ = frame.shape

        # 调用基类的通用 _perform_crop_zoom
        res = self._perform_crop_zoom(
            frame, crop_center, crop_size,
            run_sub_detect=self._run_recognizer
        )
        if not res or not callable(res[2]):
            return [], [], []

        detection_result, _, to_orig = res
        return self._extract_results(
            detection_result, w, h,
            min_area_ratio=0.0,
            coord_transform=to_orig,
        )

    # ------------------------------------------------------------------
    # MediaPipe 特有方法
    # ------------------------------------------------------------------

    def _resolve_model_path(self, project_root, preferred_model_type):
        """解析手部模型路径。

        项目实际只有两个模型：
        - models/hand_landmarker.task (Lite) —— 纯 HandLandmarker，只输出 21 关键点。
        - gesture_recognizer.task (Heavy) —— GestureRecognizer 模型，输出 21 关键点 +
          内置手势标签；这是项目里真正的 "Heavy" 模型。

        MediaPipe 官方并没有发布 hand_landmarker_heavy.task（只有 hand_landmarker
        和 hand_landmarker_lite）。因此 config/UI 中的 Heavy 就是 gesture_recognizer.task，
        加载它并非 fallback，而是预期行为。
        """
        if getattr(sys, 'frozen', False):
            base_dir = sys._MEIPASS
        else:
            base_dir = project_root

        # Lite：纯 Landmarker 模型
        lite_candidates = [
            os.path.join(base_dir, "models", "hand_landmarker.task"),
            os.path.join(base_dir, "hand_landmarker.task"),
        ]
        # Heavy / Full：项目里实际对应 gesture_recognizer.task
        heavy_candidates = [
            os.path.join(base_dir, "gesture_recognizer.task"),
            os.path.join(base_dir, "models", "gesture_recognizer.task"),
        ]
        pref = str(preferred_model_type).lower()
        if pref == "lite":
            candidates = lite_candidates + heavy_candidates
        else:  # heavy / full / 未知都偏好 heavy
            candidates = heavy_candidates + lite_candidates
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        raise FileNotFoundError(
            "未找到手部模型文件。请确保 gesture_recognizer.task 或 models/hand_landmarker.task 存在。"
            f"搜索路径: {candidates}"
        )

    def _next_mp_timestamp(self):
        ts = int(time.time() * 1000)
        if ts <= self._last_mp_timestamp:
            ts = self._last_mp_timestamp + 1
        self._last_mp_timestamp = ts
        return ts

    def _run_recognizer(self, image_bgr):
        """跑一次 MediaPipe 推理，返回原始 detection_result。"""
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        if self._is_pure_landmarker:
            if self.static_image_mode:
                return self.detector.detect(mp_image)
            return self.detector.detect_for_video(mp_image, self._next_mp_timestamp())
        else:
            if self.static_image_mode:
                return self.detector.recognize(mp_image)
            return self.detector.recognize_for_video(mp_image, self._next_mp_timestamp())

    def _extract_results(
        self,
        detection_result,
        frame_w,
        frame_h,
        min_area_ratio=0.005,
        coord_transform=None,
    ):
        """从 MediaPipe detection_result 提取标准格式。"""
        hands_landmarks = []
        hands_gestures = []
        raw_hand_lists = []
        if not (detection_result and detection_result.hand_landmarks):
            return hands_landmarks, hands_gestures, raw_hand_lists

        min_area = frame_w * frame_h * min_area_ratio

        for i, hand_landmarks_list in enumerate(detection_result.hand_landmarks):
            if coord_transform is not None:
                pixel_coords = [
                    coord_transform(lm.x, lm.y) for lm in hand_landmarks_list
                ]
            else:
                pixel_coords = [
                    (lm.x * frame_w, lm.y * frame_h) for lm in hand_landmarks_list
                ]
            xs_px = [p[0] for p in pixel_coords]
            ys_px = [p[1] for p in pixel_coords]
            box_w = max(xs_px) - min(xs_px)
            box_h = max(ys_px) - min(ys_px)
            bbox_area = box_w * box_h
            if bbox_area < min_area:
                continue

            landmarks = []
            for idx, (cx, cy) in enumerate(pixel_coords):
                # Keep MediaPipe's sub-pixel coordinates through the smoothing
                # and screen-mapping pipeline. Rounding here creates visible
                # multi-pixel cursor steps on high-resolution displays.
                # z 是 MediaPipe 的深度归一化值（相对手腕、按图像宽归一），
                # 透传给下游几何约束做遮挡判定（指尖 z 突跳=遮挡/翻面）。
                cz = float(getattr(hand_landmarks_list[idx], "z", 0.0))
                landmarks.append([idx, float(cx), float(cy), cz])
            hands_landmarks.append(landmarks)
            raw_hand_lists.append(hand_landmarks_list)

            handedness_name = "Unknown"
            handedness_score = 0.0
            if detection_result.handedness and i < len(detection_result.handedness):
                cat_list = detection_result.handedness[i]
                if cat_list:
                    cat = cat_list[0]
                    handedness_name = cat.category_name
                    handedness_score = float(cat.score)

            if not self._is_pure_landmarker and hasattr(detection_result, "gestures") and detection_result.gestures and i < len(detection_result.gestures):
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

    def _draw_landmarks(self, frame, hand_landmarks_list, landmarks):
        """MediaPipe 原生骨架绘制。"""
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

    def close(self):
        """Release the native MediaPipe task runner and SR engines deterministically."""
        detector = getattr(self, "detector", None)
        self.detector = None
        close = getattr(detector, "close", None)
        if callable(close):
            close()
        # 释放超分辨率 ONNX session，防止反复创建 tracker 时泄漏显存
        self._sr.release()
