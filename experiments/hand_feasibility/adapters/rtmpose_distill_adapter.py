"""R1 RTMPose 适配器（相同手框协议：MediaPipe 检测框手 → RTMPose 提点）。

route_id: R1B_DISTILL — tasmulaev/rtmpose-m-distill ONNX（SimCC 解码）
route_id: R1A 需要 mmpose/rtmlib 环境（PyTorch .pth），另立适配器。

相同手框协议（方案 §6.2 L1）：用 MediaPipe detector（与生产同一检测入口）
出框，裁剪后喂 RTMPose 256x256，SimCC 解码回原帧坐标。检测耗时与提点耗时
分开计时。无手势标签（label_source 如实标 none）。

SimCC 解码（mmdeploy 口径）：simcc_x/simcc_y shape (1,21,512)，
split_ratio = 512/256 = 2.0；x_pixel = argmax(simcc_x, axis=2) / 2.0 * (crop_w/256)，
置信度 = min(max(simcc_x), max(simcc_y)) 归一化后取两轴较小者。
"""
import os
import sys

import cv2
import numpy as np

_adapters_dir = os.path.dirname(os.path.abspath(__file__))
if _adapters_dir not in sys.path:
    sys.path.insert(0, _adapters_dir)
from base_adapter import BaseHandModelAdapter

APP_DIR = os.path.normpath(os.path.join(_adapters_dir, "..", "..", "..", "app"))


class RTMPoseDistillAdapter(BaseHandModelAdapter):
    """R1b：MediaPipe 框手 + RTMPose-m Distill ONNX 提点。"""

    def __init__(self, config=None):
        super().__init__("R1B_DISTILL", config)
        self._sess = None
        self._detector = None
        self._frame_index = 0
        self._input_size = 256
        self._simcc_len = 512

    def _load_model(self):
        import onnxruntime as ort
        model_path = self.config.get("rtmpose_onnx_path") or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(_adapters_dir))),
            "experiments", "hand_feasibility", "models", "rtmpose-m-distill-256x256.onnx")
        self._sess = ort.InferenceSession(
            model_path, providers=["DmlExecutionProvider", "CPUExecutionProvider"])
        self._in_name = self._sess.get_inputs()[0].name
        self._x_name = self._sess.get_outputs()[0].name
        self._y_name = self._sess.get_outputs()[1].name
        # MediaPipe HandLandmarker（tasks API，仅用其框）
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        task_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(_adapters_dir))),
            "models", "hand_landmarker.task")
        base = mp_python.BaseOptions(model_asset_path=task_path)
        opts = vision.HandLandmarkerOptions(
            base_options=base,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=self.config.get("hand_detection_confidence", 0.5))
        self._mp_detector = vision.HandLandmarker.create_from_options(opts)

    def reset(self):
        self._frame_index = 0
        if self._mp_detector is not None:
            self._mp_detector.close()
            self._mp_detector = None
            self._loaded = False

    def close(self):
        if self._mp_detector is not None:
            self._mp_detector.close()
            self._mp_detector = None
        self._sess = None

    def _detect_boxes(self, frame):
        """MediaPipe HandLandmarker（IMAGE 模式）出框，返回 [(x0,y0,x1,y1,conf)]。"""
        import mediapipe as mp
        h, w = frame.shape[:2]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        res = self._mp_detector.detect(mp_image)
        boxes = []
        if res.hand_landmarks:
            for lms in res.hand_landmarks:
                xs = [lm.x * w for lm in lms]
                ys = [lm.y * h for lm in lms]
                boxes.append((min(xs), min(ys), max(xs), max(ys), 1.0))
        return boxes

    def _infer_impl(self, frame, source_timestamp):
        src_h, src_w = frame.shape[:2]
        boxes = self._detect_boxes(frame)
        out_hands = []
        simcc_ms = 0.0

        for i, (x0, y0, x1, y1, conf) in enumerate(boxes):
            # 外扩 1.4x（与 hagrid_yolo_hand_tracker 同策略）
            bw, bh = x1 - x0, y1 - y0
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            side = max(bw, bh) * 1.4
            ex0 = max(0, cx - side / 2)
            ey0 = max(0, cy - side / 2)
            ex1 = min(src_w, cx + side / 2)
            ey1 = min(src_h, cy + side / 2)
            crop = frame[int(ey0):int(ey1), int(ex0):int(ex1)]
            if crop.size == 0:
                continue
            ch, cw = crop.shape[:2]

            blob = cv2.resize(crop, (self._input_size, self._input_size))
            blob = cv2.cvtColor(blob, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            blob = blob.transpose(2, 0, 1)[None]  # 1,3,256,256

            import time as _time
            t0 = _time.perf_counter()
            sx, sy = self._sess.run(
                [self._x_name, self._y_name],
                {self._in_name: blob.astype(np.float32)})
            simcc_ms += (_time.perf_counter() - t0) * 1000.0

            # SimCC 解码：argmax over 512 bins, /2.0 → 256 坐标系
            kx = sx.argmax(axis=2)[0].astype(np.float32) / 2.0  # (21,)
            ky = sy.argmax(axis=2)[0].astype(np.float32) / 2.0
            scores = np.minimum(sx.max(axis=2)[0], sy.max(axis=2)[0])  # (21,)

            # 256 → 原帧像素（crop 逆变换）
            kx_px = kx / self._input_size * cw + ex0
            ky_px = ky / self._input_size * ch + ey0
            pts = [[float(x), float(y)] for x, y in zip(kx_px, ky_px)]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            bbox = [min(xs), min(ys), max(xs), max(ys)]

            out_hands.append({
                "track_id": i,
                "handedness": None,
                "handedness_score": None,
                "bbox_xyxy": bbox,
                "keypoints_2d_21x2": pts,
                "keypoints_3d_or_null": None,
                "keypoints_3d_units_or_null": None,
                "keypoint_scores_21_or_null": [float(s) for s in scores],
                "detected_fresh": True,
                "predicted": False,
                "gesture_label": "None",
                "gesture_score": 0.0,
                "label_source": "none",
            })

        return {
            "schema_version": 1,
            "run_id": self.config.get("run_id", "unknown"),
            "clip_id": self.config.get("clip_id", "unknown"),
            "frame_index": self._frame_index,
            "source_timestamp": source_timestamp,
            "route_id": self.route_id,
            "actual_backend": "onnxruntime-directml",
            "coordinate_space": "source_frame_pixels",
            "source_size": [src_w, src_h],
            "model_input_size": "256x256",
            "hands": out_hands,
            "simcc_inference_ms": simcc_ms,
        }
