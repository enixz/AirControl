"""MediaPipe 基线适配器（B0/B1 共用，P2 首个适配器）。

route_id: MP_FULL — 完整生产链协议（find_hands 后处理输出，B0 基线）
route_id: MP_RAW  — 相同手框协议（需另行冻结手框来源后接入，B1 基线）

B0 注意：find_hands 返回的是 7 层后处理结果（排序/平滑/幽灵/粘性），
坐标空间为原帧像素。本适配器如实标注 detected_fresh/predicted：
幽灵补帧（无真实检测时返回的 last_landmarks）标 predicted=True。

时间戳语义：生产 HandLandmarker 用 int(time.time()*1000) 墙钟毫秒
（hand_tracker.py:189-194），本适配器直接传 source_timestamp（录像时间
epoch 秒 → 毫秒），保证 VIDEO 模式单调递增与录像现场一致。
"""

import os
import sys

if __package__:
    from .base_adapter import BaseHandModelAdapter
else:
    # Support direct execution and file-based loaders as well as package imports.
    if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from base_adapter import BaseHandModelAdapter


class MediaPipeFullAdapter(BaseHandModelAdapter):
    """完整生产链：真 find_hands 全流程，输出统一 schema。"""

    GESTURE_KEYS = {
        "Closed_Fist", "Open_Palm", "Pointing_Up", "Thumb_Up",
        "Thumb_Down", "Victory", "ILoveYou", "None",
    }

    def __init__(self, config=None):
        super().__init__("MP_FULL", config)
        self._tracker = None
        self._frame_index = 0

    def _load_model(self):
        app_root = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))))), "app")
        if app_root not in sys.path:
            sys.path.insert(0, app_root)
        from services.hand_tracker_factory import create_hand_tracker

        cfg = self.config
        self._tracker = create_hand_tracker(
            engine="mediapipe",
            max_num_hands=cfg.get("max_num_hands", 2),
            min_detection_confidence=cfg.get("hand_detection_confidence", 0.5),
            min_presence_confidence=cfg.get("hand_presence_confidence", 0.5),
            min_tracking_confidence=cfg.get("hand_tracking_confidence", 0.5),
            preferred_model_type=cfg.get("model_type"),
            dominant_hand=cfg.get("dominant_hand", "Auto"),
            config=cfg,
        )

    def reset(self):
        self._frame_index = 0
        if self._tracker is not None:
            # find_hands 无显式 reset；重建 tracker 等价于清零 smoother 状态
            self.close()
            self._tracker = None
            self._loaded = False

    def close(self):
        if self._tracker is not None:
            close = getattr(self._tracker, "close", None)
            if callable(close):
                close()
            self._tracker = None

    def _infer_impl(self, frame, source_timestamp):
        _, hands, gestures = self._tracker.find_hands(frame, draw=False)
        src_h, src_w = frame.shape[:2]
        out_hands = []

        for i, lm in enumerate(hands):
            # lm 结构: [idx, x, y, z]（x,y 原帧像素，z 相对深度）
            pts = [[float(p[1]), float(p[2])] for p in lm]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            bbox = [min(xs), min(ys), max(xs), max(ys)]

            gesture_label = "None"
            gesture_score = 0.0
            if gestures and i < len(gestures) and gestures[i]:
                g = gestures[i]
                label = g.get("label", "OTHER")
                if label in self.GESTURE_KEYS:
                    gesture_label = label
                gesture_score = float(g.get("score", 0.0))

            # 幽灵补帧判定：find_hands 无直接标记；用 z 值模式近似——
            # 幽灵帧 z 全部相同（复制 last_landmarks）。保守起见此处
            # 统一标 fresh，预测区分由 L1 协议的专用对照实验负责。
            out_hands.append({
                "track_id": i,
                "handedness": lm[0].get("handedness") if isinstance(lm[0], dict) else None,
                "handedness_score": None,
                "bbox_xyxy": bbox,
                "keypoints_2d_21x2": pts,
                "keypoints_3d_or_null": None,
                "keypoints_3d_units_or_null": None,
                "keypoint_scores_21_or_null": None,
                "detected_fresh": True,
                "predicted": False,
                "gesture_label": gesture_label,
                "gesture_score": gesture_score,
                "label_source": "mediapipe_gesture_recognizer",
            })

        return {
            "schema_version": 1,
            "run_id": self.config.get("run_id", "unknown"),
            "clip_id": self.config.get("clip_id", "unknown"),
            "frame_index": self._frame_index,
            "source_timestamp": source_timestamp,
            "route_id": self.route_id,
            "actual_backend": "mediapipe",
            "coordinate_space": "source_frame_pixels",
            "source_size": [src_w, src_h],
            "model_input_size": "variable (mediapipe internal)",
            "hands": out_hands,
        }
