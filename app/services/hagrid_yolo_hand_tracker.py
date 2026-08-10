"""HaGRID YOLO + MediaPipe HandLandmarker 混合手部追踪器。

用 HaGRID 预训练的 YOLO 手部检测器（默认 hand_yolov8n.onnx）替代
MediaPipe 的 BlazePalm 检测头，检测到手部 bbox 后裁剪放大，再用
MediaPipe HandLandmarker 提取 21 关键点。设计目的是 A/B 验证 YOLO
检测器是否在远距离场景比 BlazePalm 有更高召回率。

⚠️ 这是一个实验性引擎，用于离线 A/B 对比验证，不代表 YOLO 一定比
MediaPipe 更强。请配合 analyze_primary_stability.py 使用。

依赖：
  - onnxruntime（已在 requirements.txt 中）
  - models/hand_yolov8n.onnx（默认检测器，不随安装包分发，需手动下载）
  - models/hand_landmarker.task（项目已有，随安装包分发）

模型下载（可选）：hand_yolov8n.onnx 因 AGPL-3.0 许可证不打包进发布版。
如需使用 hagrid_yolo 引擎或 engine_auto_switch 远距自动切换：
  1. 下载 YOLOv8n 权重：
     https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt
  2. 导出为 ONNX（需安装 ultralytics）：
     pip install ultralytics
     yolo export model=yolov8n.pt format=onnx opset=13 simplify imgsz=640
  3. 重命名为 hand_yolov8n.onnx 放到 models/ 目录下。

  或下载 HaGRID v2 预训练权重：
     https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid_v2/models/YOLOv10n_hands.pt
  并以相同方式导出。_resolve_yolo_model 按候选文件名顺序查找。
"""

import logging
import os

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from .base_hand_tracker import BaseHandTracker

_logger = logging.getLogger("gesture")

# YOLO 输入尺寸
_YOLO_INPUT_SIZE = 640
# YOLO 置信度阈值（可由 config 的 yolo_confidence 覆盖）
_DEFAULT_YOLO_CONF = 0.25
# YOLO NMS IoU 阈值
_NMS_IOU = 0.45
# crop padding 比例：YOLO bbox 外扩多少倍后送给 HandLandmarker
_CROP_PADDING = 1.4
# HandLandmarker 最小裁剪尺寸（像素）
_MIN_CROP_SIZE = 48


class HagridYoloHandTracker(BaseHandTracker):
    """HaGRID YOLO 手部检测 + MediaPipe HandLandmarker 关键点混合引擎。

    检测流程：
      1. YOLO 全帧检测手部 bbox
      2. 对每个 bbox 裁剪 + padding + 放大到 256x256
      3. MediaPipe HandLandmarker 在裁剪区域提取 21 关键点 + handedness
      4. 坐标映射回原帧

    继承 BaseHandTracker 的全部公共逻辑（crop-zoom 状态机、平滑、排序等）。
    """

    def __init__(
        self,
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        dominant_hand="Right",
        config=None,
    ):
        super().__init__(max_num_hands=max_num_hands, dominant_hand=dominant_hand, config=config)

        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        # --- 加载 YOLO ONNX 模型 ---
        self._yolo_model_path = self._resolve_yolo_model(project_root)
        self._yolo_session = self._create_yolo_session()
        self._yolo_input_name = self._yolo_session.get_inputs()[0].name

        # YOLO 置信度阈值（可由 config 覆盖）
        self._yolo_conf = float(
            config.get("yolo_confidence", _DEFAULT_YOLO_CONF) if config else _DEFAULT_YOLO_CONF
        )

        # --- 加载 MediaPipe HandLandmarker ---
        self._landmarker_model_path = self._resolve_landmarker_model(project_root)
        base_options = python.BaseOptions(model_asset_path=self._landmarker_model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)

        self._last_mp_timestamp = 0

        _logger.info(
            "HagridYoloHandTracker 初始化: YOLO=%s, Landmarker=%s, conf=%.2f",
            os.path.basename(self._yolo_model_path),
            os.path.basename(self._landmarker_model_path),
            self._yolo_conf,
        )

    # ------------------------------------------------------------------
    # BaseHandTracker 抽象接口实现
    # ------------------------------------------------------------------

    @property
    def engine_name(self) -> str:
        return "hagrid_yolo"

    def _detect(self, frame):
        """全帧检测：YOLO 检测 bbox → HandLandmarker 提取关键点。

        Returns:
            (hands_landmarks, hands_gestures, raw_data)
        """
        h, w, _ = frame.shape
        bboxes = self._yolo_detect(frame)

        if not bboxes:
            return [], [], []

        return self._extract_landmarks_from_bboxes(frame, bboxes, w, h)

    def _detect_crop_zoom(self, frame, crop_center, crop_size):
        """裁剪放大 → YOLO + HandLandmarker → 坐标映射回原帧。"""
        h, w, _ = frame.shape

        res = self._perform_crop_zoom(
            frame, crop_center, crop_size,
            run_sub_detect=self._detect_on_crop,
        )
        if not res or not callable(res[2]):
            return [], [], []

        detection_result, crop_info, to_orig = res
        hands_landmarks, hands_gestures, raw = detection_result

        # crop_info = (x0, y0, crop_size, target, scale)
        # _extract_landmarks_from_bboxes 返回的是 zoomed crop 上的像素坐标，
        # 需要映射回原帧：orig_px = x0 + zoomed_px * scale
        x0, y0, _cs, _target, scale = crop_info

        mapped_landmarks = []
        for landmarks in hands_landmarks:
            mapped = []
            for lm in landmarks:
                ox = x0 + lm[1] * scale
                oy = y0 + lm[2] * scale
                mapped.append([lm[0], float(ox), float(oy), lm[3] if len(lm) > 3 else 0.0])
            mapped_landmarks.append(mapped)

        # 更新 bbox_area 为原帧坐标系的值
        for i, g in enumerate(hands_gestures):
            if i < len(mapped_landmarks):
                lms = mapped_landmarks[i]
                xs = [lm[1] for lm in lms]
                ys = [lm[2] for lm in lms]
                g["bbox_area"] = (max(xs) - min(xs)) * (max(ys) - min(ys))

        return mapped_landmarks, hands_gestures, raw

    # ------------------------------------------------------------------
    # YOLO 检测方法
    # ------------------------------------------------------------------

    def _detect_on_crop(self, cropped_bgr):
        """在裁剪放大后的区域上跑 YOLO + HandLandmarker。

        这是 _perform_crop_zoom 的 run_sub_detect 回调。
        返回的坐标是 crop 区域内的像素坐标（尚未映射回原帧）。
        """
        bboxes = self._yolo_detect(cropped_bgr)
        if not bboxes:
            return [], [], []
        h, w, _ = cropped_bgr.shape
        return self._extract_landmarks_from_bboxes(cropped_bgr, bboxes, w, h)

    def _resolve_yolo_model(self, project_root):
        """解析 YOLO ONNX 模型路径。"""
        import sys
        if getattr(sys, 'frozen', False):
            base_dir = sys._MEIPASS
        else:
            base_dir = project_root

        candidates = [
            os.path.join(base_dir, "models", "yolov10n_hands.onnx"),
            os.path.join(base_dir, "models", "yolov10_hands.onnx"),
            os.path.join(base_dir, "models", "hand_yolov8n.onnx"),
            os.path.join(base_dir, "models", "hand_yolov8s.onnx"),
            os.path.join(base_dir, "yolov10n_hands.onnx"),
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        raise FileNotFoundError(
            "未找到 YOLO 手部检测 ONNX 模型。请将 yolov10n_hands.onnx 放到 models/ 目录下。\n"
            "下载 .pt: https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid_v2/models/YOLOv10n_hands.pt\n"
            "导出 ONNX: yolo export model=YOLOv10n_hands.pt format=onnx opset=13 simplify\n"
            f"搜索路径: {candidates}"
        )

    def _resolve_landmarker_model(self, project_root):
        """解析 MediaPipe HandLandmarker 模型路径。"""
        import sys
        if getattr(sys, 'frozen', False):
            base_dir = sys._MEIPASS
        else:
            base_dir = project_root

        candidates = [
            os.path.join(base_dir, "models", "hand_landmarker.task"),
            os.path.join(base_dir, "hand_landmarker.task"),
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        raise FileNotFoundError(
            "未找到 MediaPipe HandLandmarker 模型 (hand_landmarker.task)。\n"
            f"搜索路径: {candidates}"
        )

    def _create_yolo_session(self):
        """创建 ONNX Runtime session，优先使用 GPU（DirectML/CUDA）。"""
        import onnxruntime as ort

        available = ort.get_available_providers()
        # 优先 GPU provider
        providers = []
        if "DmlExecutionProvider" in available:
            providers.append("DmlExecutionProvider")
        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        session = ort.InferenceSession(self._yolo_model_path, providers=providers)
        actual = session.get_providers()
        _logger.info("[YOLO] ONNX session providers: %s", actual)
        return session

    def _yolo_detect(self, frame_bgr):
        """跑一次 YOLOv10 推理，返回手部 bbox 列表。

        Args:
            frame_bgr: BGR 帧（任意尺寸）

        Returns:
            list of (x0, y0, x1, y1, score) — 像素坐标的 bbox
        """
        h, w, _ = frame_bgr.shape

        # 预处理：resize 到 640x640，保持比例，letterbox 填充
        input_img, ratio, (pad_w, pad_h) = self._letterbox(frame_bgr, _YOLO_INPUT_SIZE)

        # BGR → RGB，归一化，HWC → CHW → NCHW
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        input_img = input_img.astype(np.float32) / 255.0
        input_img = np.transpose(input_img, (2, 0, 1))
        input_img = np.expand_dims(input_img, axis=0)

        # 推理
        outputs = self._yolo_session.run(None, {self._yolo_input_name: input_img})

        # 后处理：解析输出
        bboxes = self._parse_yolo_output(outputs, ratio, pad_w, pad_h, w, h)
        return bboxes

    def _letterbox(self, img, target_size):
        """Letterbox 预处理：等比缩放 + 填充到 target_size。

        Returns:
            (letterboxed_img, ratio, (pad_w, pad_h))
        """
        h, w, _ = img.shape
        ratio = min(target_size / w, target_size / h)
        new_w = int(round(w * ratio))
        new_h = int(round(h * ratio))

        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # 创建灰色画布，居中放置
        canvas = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
        pad_w = (target_size - new_w) // 2
        pad_h = (target_size - new_h) // 2
        canvas[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

        return canvas, ratio, (pad_w, pad_h)

    def _parse_yolo_output(self, outputs, ratio, pad_w, pad_h, orig_w, orig_h):
        """解析 YOLOv10 ONNX 输出，返回 bbox 列表。

        YOLOv10 end-to-end ONNX 输出格式通常为 [1, N, 6]（xywh + score + class）。
        如果不是 end-to-end 格式，则输出为 [1, 84, 8400]，需要手动 NMS。
        本方法兼容两种格式。
        """
        output = outputs[0]

        # --- 格式1: end-to-end [1, N, 6] ---
        if output.ndim == 3 and output.shape[2] == 6:
            detections = output[0]  # [N, 6]
            # 过滤置信度
            mask = detections[:, 4] >= self._yolo_conf
            detections = detections[mask]

            bboxes = []
            for det in detections:
                cx, cy, bw, bh = det[0], det[1], det[2], det[3]
                score = float(det[4])

                # 从 letterbox 坐标 → 原帧坐标
                cx = (cx - pad_w) / ratio
                cy = (cy - pad_h) / ratio
                bw = bw / ratio
                bh = bh / ratio

                x0 = max(0, int(cx - bw / 2))
                y0 = max(0, int(cy - bh / 2))
                x1 = min(orig_w, int(cx + bw / 2))
                y1 = min(orig_h, int(cy + bh / 2))

                if x1 - x0 < 5 or y1 - y0 < 5:
                    continue

                bboxes.append((x0, y0, x1, y1, score))

            # 按置信度排序，取前 max_num_hands 个
            bboxes.sort(key=lambda b: b[4], reverse=True)
            return bboxes[:self.max_num_hands]

        # --- 格式2: [1, C, A] 或 [1, A, C]（需要手动 NMS） ---
        elif output.ndim == 3:
            # 尝试 [1, A, C] 格式
            if output.shape[1] > output.shape[2]:
                detections = output[0]  # [A, C]
            else:
                detections = output[0].T  # [A, C]

            # 手部检测只有 1 个类，假设 C = [cx, cy, w, h, class_score]
            if detections.shape[1] >= 5:
                scores = detections[:, 4]
                mask = scores >= self._yolo_conf
                filtered = detections[mask]

                if len(filtered) == 0:
                    return []

                # 提取 xywh
                boxes_xywh = filtered[:, :4]
                scores_filtered = filtered[:, 4]

                # OpenCV NMSBoxes 需要左上角 x/y + w/h；YOLO 输出是中心 xywh。
                # 直接把中心坐标传进去会使不同尺寸候选的 IoU 失真，保留重复框。
                nms_boxes = np.column_stack((
                    boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2,
                    boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2,
                    boxes_xywh[:, 2],
                    boxes_xywh[:, 3],
                ))

                # NMS
                indices = cv2.dnn.NMSBoxes(
                    nms_boxes.tolist(), scores_filtered.tolist(),
                    self._yolo_conf, _NMS_IOU,
                )
                if len(indices) == 0:
                    return []

                bboxes = []
                for idx in indices.flatten():
                    cx, cy, bw, bh = boxes_xywh[idx]
                    score = float(scores_filtered[idx])

                    # letterbox → 原帧坐标
                    cx = (cx - pad_w) / ratio
                    cy = (cy - pad_h) / ratio
                    bw = bw / ratio
                    bh = bh / ratio

                    x0 = max(0, int(cx - bw / 2))
                    y0 = max(0, int(cy - bh / 2))
                    x1 = min(orig_w, int(cx + bw / 2))
                    y1 = min(orig_h, int(cy + bh / 2))

                    if x1 - x0 < 5 or y1 - y0 < 5:
                        continue

                    bboxes.append((x0, y0, x1, y1, score))

                bboxes.sort(key=lambda b: b[4], reverse=True)
                return bboxes[:self.max_num_hands]

        _logger.warning("[YOLO] 无法解析输出格式: shape=%s", output.shape)
        return []

    # ------------------------------------------------------------------
    # HandLandmarker 关键点提取
    # ------------------------------------------------------------------

    def _extract_landmarks_from_bboxes(self, frame, bboxes, frame_w, frame_h):
        """对每个 YOLO bbox 裁剪后用 HandLandmarker 提取 21 关键点。

        Args:
            frame: BGR 原帧
            bboxes: list of (x0, y0, x1, y1, score)
            frame_w, frame_h: 原帧宽高

        Returns:
            (hands_landmarks, hands_gestures, raw_data)
        """
        hands_landmarks = []
        hands_gestures = []
        raw_data = []

        for x0, y0, x1, y1, yolo_score in bboxes:
            # bbox 外扩 padding，让 HandLandmarker 看到完整手掌
            bw = x1 - x0
            bh = y1 - y0
            pad_w = int(bw * (_CROP_PADDING - 1.0) / 2)
            pad_h = int(bh * (_CROP_PADDING - 1.0) / 2)

            cx0 = max(0, x0 - pad_w)
            cy0 = max(0, y0 - pad_h)
            cx1 = min(frame_w, x1 + pad_w)
            cy1 = min(frame_h, y1 + pad_h)

            crop_w = cx1 - cx0
            crop_h = cy1 - cy0

            if crop_w < _MIN_CROP_SIZE or crop_h < _MIN_CROP_SIZE:
                # 太小，放大到 256x256
                crop = frame[cy0:cy1, cx0:cx1]
                crop = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_LINEAR)
            else:
                crop = frame[cy0:cy1, cx0:cx1]

            if crop.size == 0:
                continue

            # HandLandmarker 推理（IMAGE 模式，逐 crop 独立检测）
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_rgb)
            try:
                result = self._landmarker.detect(mp_image)
            except Exception as e:
                _logger.debug("[YOLO-Hybrid] HandLandmarker 推理异常: %s", e)
                continue

            if not result or not result.hand_landmarks:
                continue

            # 取第一个检测到的手（每个 crop 只应该有一只手）
            hand_lms = result.hand_landmarks[0]
            landmarks = []
            for idx, lm in enumerate(hand_lms):
                # HandLandmarker 返回归一化坐标 [0,1]，映射回原帧像素坐标
                # 无论 crop 是否被 resize，归一化坐标 × crop 原始尺寸 + 偏移 = 原帧坐标
                px = cx0 + lm.x * crop_w
                py = cy0 + lm.y * crop_h
                cz = float(getattr(lm, "z", 0.0))
                landmarks.append([idx, float(px), float(py), cz])

            # 计算 bbox_area（原帧坐标系）
            xs = [lm[1] for lm in landmarks]
            ys = [lm[2] for lm in landmarks]
            bbox_area = (max(xs) - min(xs)) * (max(ys) - min(ys))

            # handedness
            handedness_name = "Unknown"
            handedness_score = 0.0
            if result.handedness and len(result.handedness) > 0:
                cat_list = result.handedness[0]
                if cat_list:
                    cat = cat_list[0]
                    handedness_name = cat.category_name
                    handedness_score = float(cat.score)

            hands_landmarks.append(landmarks)
            hands_gestures.append({
                "ml_label": "None",
                "label": "OTHER",
                "score": float(yolo_score),
                "handedness": handedness_name,
                "handedness_score": handedness_score,
                "bbox_area": bbox_area,
                "yolo_score": float(yolo_score),
            })
            raw_data.append(hand_lms)

        return hands_landmarks, hands_gestures, raw_data

    # ------------------------------------------------------------------
    # 资源释放
    # ------------------------------------------------------------------

    def close(self):
        """释放 YOLO ONNX session 和 MediaPipe Landmarker。"""
        landmarker = getattr(self, "_landmarker", None)
        self._landmarker = None
        if landmarker is not None:
            close_fn = getattr(landmarker, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception as e:
                    _logger.warning("HandLandmarker close 异常: %s", e)

        self._yolo_session = None
        self._sr.release()
        _logger.info("HagridYoloHandTracker 已释放资源")
