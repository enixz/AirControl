"""R3 YOLO26 手部专用适配器（完整原生链协议：一次输出框+21点）。

route_id: R3_YOLO26 — poptoz/yolo26_hand_pose_fp16.onnx
输入 (1,3,640,640) NCHW RGB [0,1]；输出 (1,300,69)：
[x1,y1,x2,y2,conf,class_id,kp1_x,kp1_y,kp1_vis,...]（关键点坐标相对输入 640）。
端到端无 NMS。检测与提点耗时合并（原生单模型），如实计入。
无手势标签（label_source 如实标 none）。关键点 vis 含义待验证（方案 §9.3）。
"""
import os
import sys

import cv2
import numpy as np

if __package__:
    from .base_adapter import BaseHandModelAdapter
else:
    # Support direct execution and file-based loaders as well as package imports.
    if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from base_adapter import BaseHandModelAdapter

_adapters_dir = os.path.dirname(os.path.abspath(__file__))


class Yolo26HandAdapter(BaseHandModelAdapter):
    """R3：YOLO26 端到端框+21点（letterbox 640，输出逆变换回原帧）。"""

    CONF_THRESHOLD = 0.5

    def __init__(self, config=None):
        super().__init__("R3_YOLO26", config)
        self._sess = None
        self._frame_index = 0
        self._input_size = 640

    def _load_model(self):
        import onnxruntime as ort
        model_path = self.config.get("yolo26_onnx_path") or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(_adapters_dir))),
            "experiments", "hand_feasibility", "models", "yolo26_hand_pose_fp16.onnx")
        self._sess = ort.InferenceSession(
            model_path, providers=["DmlExecutionProvider", "CPUExecutionProvider"])
        self._in_name = self._sess.get_inputs()[0].name

    def close(self):
        self._sess = None

    def _letterbox(self, frame):
        """640 letterbox，返回 (blob, scale, pad_x, pad_y)。"""
        h, w = frame.shape[:2]
        scale = min(self._input_size / w, self._input_size / h)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(frame, (nw, nh))
        pad_x = (self._input_size - nw) // 2
        pad_y = (self._input_size - nh) // 2
        canvas = np.full((self._input_size, self._input_size, 3), 114, dtype=np.uint8)
        canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
        blob = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[None]
        return blob, scale, pad_x, pad_y

    def _infer_impl(self, frame, source_timestamp):
        src_h, src_w = frame.shape[:2]
        blob, scale, pad_x, pad_y = self._letterbox(frame)

        import time as _time
        t0 = _time.perf_counter()
        outputs = self._sess.run(None, {self._in_name: blob})
        model_ms = (_time.perf_counter() - t0) * 1000.0

        pred = outputs[0][0]  # (300, 69)
        out_hands = []
        for row in pred:
            conf = row[4]
            if conf < self.CONF_THRESHOLD:
                continue
            # The output bbox is derived from the transformed keypoints below.

            kps = []
            kpt_scores = []
            k = 6
            for _ in range(21):
                kx = (row[k] - pad_x) / scale
                ky = (row[k + 1] - pad_y) / scale
                vis = float(row[k + 2])
                kps.append([float(kx), float(ky)])
                kpt_scores.append(vis)
                k += 3

            xs = [p[0] for p in kps]
            ys = [p[1] for p in kps]
            bbox = [min(xs), min(ys), max(xs), max(ys)]

            out_hands.append({
                "track_id": len(out_hands),
                "handedness": None,
                "handedness_score": None,
                "bbox_xyxy": bbox,
                "keypoints_2d_21x2": kps,
                "keypoints_3d_or_null": None,
                "keypoints_3d_units_or_null": None,
                "keypoint_scores_21_or_null": kpt_scores,
                "detected_fresh": True,
                "predicted": False,
                "gesture_label": "None",
                "gesture_score": 0.0,
                "label_source": "none",
                "detector_conf": float(conf),
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
            "model_input_size": "640x640",
            "hands": out_hands,
            "model_inference_ms": model_ms,
        }
