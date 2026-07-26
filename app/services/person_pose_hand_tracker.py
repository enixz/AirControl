"""先框人 → 姿态拿手腕 → 手腕锚点框手 → （小手才超分）→ 关键点 的远距/侧位引擎。

动机（远距 3–5m + 手侧对镜头实测瓶颈）：
  MediaPipe BlazePalm 与 HaGRID YOLO 都是在"正面手掌"数据上训的，手一旦
  yaw 侧转/背对，召回断崖下跌；远距小手进一步恶化（裸检 42.7%）。
  借鉴 URGR（arXiv:2311.15361）"框人→超分→识别"的级联思想，但把"框人
  之后"换成**人体姿态估计拿手腕锚点**——不输出姿势类别，而是定位到手、
  仍产出 21 关键点，供现有鼠标/板书逻辑使用。

检测流程（每帧）：
  1. yolov8-pose 全帧检测：同时给出人体 bbox + 17 个 COCO 关键点
     （含左右手腕 9/10、左右手肘 7/8）。
  2. 用肘→腕距离估计手的大小，以手腕为中心框出手（比 YOLO 满图找小手稳）。
     肘腕距不可得时退化为"人体 bbox 高 × 比例"框。
  3. 复用 HagridYoloHandTracker._extract_landmarks_from_bboxes 在每个手框
     上跑 MediaPipe HandLandmarker，产出 21 关键点（坐标已映射回原帧）。
     手框过小时先经 SREngine 超分放大再提点（复用 zoom_sr_engine 配置，
     A4000 走 DirectML）。
  4. 姿态完全找不到手腕时，回退到 HaGRID YOLO 全帧直接框手（父类 _detect），
     保证不比亚军基线差。

与三态闭环的关系：本引擎**不改变** engine_auto_switcher 状态机，只是给
CAPTURE/远距态提供一个"远距+侧位也框得到手"的替代检测器。近距仍用
mediapipe（NEAR 态）。默认关闭，detection_engine=person_pose_hand 显式启用。

依赖：
  - models/yolov8n-pose.onnx（人体姿态，**不入 git**，见 MODEL_PROVENANCE.md）
  - models/hand_yolov8n.onnx（回退路径 + 父类所需）
  - models/hand_landmarker.task（关键点）

⚠️ 实验性引擎，配合 benchmark_ab.py 做 A/B 验证，不代表一定优于基线。
"""

import logging
import os

import cv2
import numpy as np

from .hagrid_yolo_hand_tracker import HagridYoloHandTracker

_logger = logging.getLogger("gesture")

# yolov8-pose 输入尺寸
_POSE_INPUT_SIZE = 640
# pose 人体检测置信度（可由 config 的 person_pose_confidence 覆盖）
_DEFAULT_POSE_CONF = 0.25
# COCO 关键点索引（yolov8-pose 输出顺序）
_KP_LEFT_WRIST = 9
_KP_RIGHT_WRIST = 10
_KP_LEFT_ELBOW = 7
_KP_RIGHT_ELBOW = 8
# 关键点自身置信度低于此值则视为不可见
_KP_MIN_CONF = 0.30
# 手框尺寸 = 肘腕距 × 此系数（手掌长约等于前臂长的 ~0.5，外扩包住整手+留边）
_FOREARM_TO_HAND_SCALE = 0.9
# 肘腕距不可得时，手框 = 人体 bbox 高 × 此比例（兜底）
_BODY_TO_HAND_SCALE = 0.22
# 手框最小/最大像素（clamp）
_MIN_HAND_BOX = 24
_MAX_HAND_BOX = 320
# 手框短边低于此像素才触发超分（更远更小的手才需要放大）
_DEFAULT_SR_TRIGGER = 96


class PersonPoseHandTracker(HagridYoloHandTracker):
    """框人→姿态→手腕锚点框手的混合引擎。

    继承 HagridYoloHandTracker：复用其 YOLO 手部回退、HandLandmarker 提点、
    SREngine 超分与 BaseHandTracker 的全部公共逻辑（平滑、排序、zoom 状态机）。
    本类只重写"如何得到手框"这一环，以及"小手提点前是否超分"。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        config = self._config or {}
        self._pose_conf = float(config.get("person_pose_confidence", _DEFAULT_POSE_CONF))
        self._sr_trigger = int(config.get("person_pose_sr_trigger", _DEFAULT_SR_TRIGGER))
        self._sr_enabled = bool(config.get("person_pose_sr_enabled", True))
        self._pose_session = None
        self._pose_input_name = None
        try:
            pose_path = self._resolve_pose_model()
            self._pose_session = self._create_pose_session(pose_path)
            self._pose_input_name = self._pose_session.get_inputs()[0].name
            _logger.info(
                "PersonPoseHandTracker 初始化: pose=%s, conf=%.2f, sr_trigger=%dpx, sr=%s",
                os.path.basename(pose_path), self._pose_conf, self._sr_trigger,
                self._sr_enabled,
            )
        except FileNotFoundError as e:
            # 姿态模型缺失不致命：_detect 会整路回退到 HaGRID YOLO 直接框手。
            _logger.warning("未找到 yolov8-pose 模型，本引擎退化为 HaGRID YOLO：%s", e)

    @property
    def engine_name(self) -> str:
        return "person_pose_hand"

    # ------------------------------------------------------------------
    # 主检测入口
    # ------------------------------------------------------------------

    def _detect(self, frame):
        """框人→手腕→框手→提点；找不到手腕则回退 YOLO 直接框手。"""
        hand_bboxes = []
        if self._pose_session is not None:
            try:
                hand_bboxes = self._hand_bboxes_from_pose(frame)
            except Exception as e:
                _logger.debug("[person-pose] 姿态路径异常，回退 YOLO：%s", e)
                hand_bboxes = []

        if hand_bboxes:
            h, w, _ = frame.shape
            return self._extract_landmarks_from_bboxes(frame, hand_bboxes, w, h)

        # 回退：HaGRID YOLO 全帧直接框手（父类逻辑），保证不比亚军基线差。
        return super()._detect(frame)

    # ------------------------------------------------------------------
    # 姿态 → 手框
    # ------------------------------------------------------------------

    def _hand_bboxes_from_pose(self, frame):
        """跑 yolov8-pose，从人体关键点推导手框列表 [(x0,y0,x1,y1,score)]。"""
        persons = self._pose_detect(frame)
        if not persons:
            return []
        h, w, _ = frame.shape
        hand_boxes = []
        for body_box, kps in persons:
            body_h = max(1.0, body_box[3] - body_box[1])
            for wrist_i, elbow_i in (
                (_KP_LEFT_WRIST, _KP_LEFT_ELBOW),
                (_KP_RIGHT_WRIST, _KP_RIGHT_ELBOW),
            ):
                wx, wy, wc = kps[wrist_i]
                if wc < _KP_MIN_CONF:
                    continue
                # 手框尺寸：优先用肘腕距（前臂长 ∝ 手掌长）
                ex, ey, ec = kps[elbow_i]
                if ec >= _KP_MIN_CONF:
                    forearm = float(np.hypot(wx - ex, wy - ey))
                    box = forearm * _FOREARM_TO_HAND_SCALE
                else:
                    box = body_h * _BODY_TO_HAND_SCALE
                box = float(np.clip(box, _MIN_HAND_BOX, _MAX_HAND_BOX))
                half = box / 2.0
                x0 = int(max(0, wx - half))
                y0 = int(max(0, wy - half))
                x1 = int(min(w, wx + half))
                y1 = int(min(h, wy + half))
                if x1 - x0 < _MIN_HAND_BOX or y1 - y0 < _MIN_HAND_BOX:
                    continue
                hand_boxes.append((x0, y0, x1, y1, float(wc)))
        # 只保留最强的 max_num_hands 个手腕框
        hand_boxes.sort(key=lambda b: b[4], reverse=True)
        return hand_boxes[: self.max_num_hands]

    # ------------------------------------------------------------------
    # 关键点提取（重写：小手先超分再提点）
    # ------------------------------------------------------------------

    def _extract_landmarks_from_bboxes(self, frame, bboxes, frame_w, frame_h):
        """对每个手框提点；手框短边 < sr_trigger 时先超分放大。

        与父类的区别仅在"提点前对小手做 SR"。超分通过临时替换输入帧实现：
        把该手框 crop 放大成一个独立小图，再复用父类在其上提点，最后把坐标
        换算回原帧。SR 失败/不可用则退化为父类的普通放大。
        """
        if not self._sr_enabled:
            return super()._extract_landmarks_from_bboxes(frame, bboxes, frame_w, frame_h)

        hands_landmarks, hands_gestures, raw_data = [], [], []
        for bbox in bboxes:
            x0, y0, x1, y1 = bbox[0], bbox[1], bbox[2], bbox[3]
            short = min(x1 - x0, y1 - y0)
            if short >= self._sr_trigger:
                # 手够大，走父类普通路径
                sub = super()._extract_landmarks_from_bboxes(frame, [bbox], frame_w, frame_h)
                self._append_detection(sub, hands_landmarks, hands_gestures, raw_data)
                continue

            # 小手：crop → 超分/放大到 target → 在大图上提点 → 坐标映回原帧
            crop = frame[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            target = int(self._crop_target_size)
            zoomed = self._sr_upscale(crop, target)
            zh, zw = zoomed.shape[:2]
            sub = super()._extract_landmarks_from_bboxes(
                zoomed, [(0, 0, zw, zh, bbox[4])], zw, zh)
            if not sub or not sub[0]:
                continue
            # 把 zoomed 图上的像素坐标按比例映回原帧
            sx = (x1 - x0) / float(zw)
            sy = (y1 - y0) / float(zh)
            mapped = []
            for landmarks in sub[0]:
                mapped.append([
                    [lm[0], x0 + lm[1] * sx, y0 + lm[2] * sy,
                     lm[3] if len(lm) > 3 else 0.0]
                    for lm in landmarks
                ])
            for i, lms in enumerate(mapped):
                xs = [lm[1] for lm in lms]; ys = [lm[2] for lm in lms]
                if i < len(sub[1]):
                    sub[1][i]["bbox_area"] = (max(xs) - min(xs)) * (max(ys) - min(ys))
            self._append_detection((mapped, sub[1], sub[2]),
                                   hands_landmarks, hands_gestures, raw_data)
        return hands_landmarks, hands_gestures, raw_data

    @staticmethod
    def _append_detection(sub, hands_landmarks, hands_gestures, raw_data):
        if not sub:
            return
        lms, ges, raw = sub
        hands_landmarks.extend(lms)
        hands_gestures.extend(ges)
        raw_data.extend(raw)

    def _sr_upscale(self, crop, target):
        """用 SREngine 把小手 crop 放大到 target；失败退回普通插值。"""
        self._sr.init()
        engine = "auto"
        if self._config is not None:
            engine = self._config.get("zoom_sr_engine", "auto")
        actual = self._sr.resolve(engine, max(crop.shape[0], crop.shape[1]), target)
        zoomed = None
        if actual == "espcn":
            zoomed = self._sr.espcn(crop, target)
        elif actual in ("realesrgan_cpu", "realesrgan_gpu"):
            zoomed = self._sr.realesrgan(crop, target, prefer_gpu=(actual == "realesrgan_gpu"))
        if zoomed is None:
            zoomed = cv2.resize(crop, (target, target), interpolation=cv2.INTER_LINEAR)
        self._sr.log_tier(actual, max(crop.shape[0], crop.shape[1]), target)
        return zoomed

    # ------------------------------------------------------------------
    # yolov8-pose 推理
    # ------------------------------------------------------------------

    def _resolve_pose_model(self):
        """解析 yolov8-pose ONNX 路径。"""
        import sys
        if getattr(sys, "frozen", False):
            base_dir = sys._MEIPASS
        else:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))))
        candidates = [
            os.path.join(base_dir, "models", "yolov8n-pose.onnx"),
            os.path.join(base_dir, "models", "yolov8s-pose.onnx"),
            os.path.join(base_dir, "yolov8n-pose.onnx"),
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        raise FileNotFoundError(f"未找到 yolov8-pose ONNX 模型。搜索路径: {candidates}")

    def _create_pose_session(self, pose_path):
        """创建 pose ONNX session（GPU 优先，DML/CUDA，回退 CPU）。"""
        import onnxruntime as ort
        available = ort.get_available_providers()
        providers = []
        if "DmlExecutionProvider" in available:
            providers.append("DmlExecutionProvider")
        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
        session = ort.InferenceSession(pose_path, providers=providers)
        _logger.info("[pose] ONNX session providers: %s", session.get_providers())
        return session

    def _pose_detect(self, frame_bgr):
        """跑 yolov8-pose，返回 [(body_box, kps)]。

        body_box = (x0, y0, x1, y1, conf) 像素坐标；kps = np.array(17,3) 的
        (x, y, conf) 像素坐标。yolov8-pose ONNX 输出 shape 为 [1, 56, N]，
        每列 = [cx, cy, w, h, conf, k0x, k0y, k0c, ... k16x, k16y, k16c]。
        """
        h, w, _ = frame_bgr.shape
        input_img, ratio, (pad_w, pad_h) = self._letterbox(frame_bgr, _POSE_INPUT_SIZE)
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        input_img = np.transpose(input_img, (2, 0, 1))[np.newaxis, ...]
        outputs = self._pose_session.run(None, {self._pose_input_name: input_img})
        return self._parse_pose_output(outputs, ratio, pad_w, pad_h, w, h)

    def _parse_pose_output(self, outputs, ratio, pad_w, pad_h, orig_w, orig_h):
        """解析 yolov8-pose 输出 [1,56,N] → [(body_box, kps)]，已做 NMS。"""
        out = outputs[0]
        if out.ndim == 3:
            out = out[0]          # [56, N]
        if out.shape[0] != 56 and out.shape[1] == 56:
            out = out.T           # 兼容 [N,56]
        if out.shape[0] != 56:
            _logger.warning("[pose] 无法解析输出 shape=%s", out.shape)
            return []
        preds = out.T             # [N, 56]
        boxes_xywh = preds[:, 0:4]
        confs = preds[:, 4]
        kps = preds[:, 5:].reshape(-1, 17, 3)

        mask = confs >= self._pose_conf
        if not np.any(mask):
            return []
        boxes_xywh, confs, kps = boxes_xywh[mask], confs[mask], kps[mask]

        nms_boxes = np.column_stack((
            boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2,
            boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2,
            boxes_xywh[:, 2], boxes_xywh[:, 3],
        ))
        idxs = cv2.dnn.NMSBoxes(nms_boxes.tolist(), confs.tolist(),
                                self._pose_conf, 0.45)
        if len(idxs) == 0:
            return []

        persons = []
        for i in idxs.flatten():
            cx, cy, bw, bh = boxes_xywh[i]
            # letterbox → 原帧
            cx = (cx - pad_w) / ratio
            cy = (cy - pad_h) / ratio
            bw = bw / ratio
            bh = bh / ratio
            x0 = max(0, int(cx - bw / 2)); y0 = max(0, int(cy - bh / 2))
            x1 = min(orig_w, int(cx + bw / 2)); y1 = min(orig_h, int(cy + bh / 2))
            body_box = (x0, y0, x1, y1, float(confs[i]))
            pk = kps[i].copy()
            pk[:, 0] = (pk[:, 0] - pad_w) / ratio
            pk[:, 1] = (pk[:, 1] - pad_h) / ratio
            persons.append((body_box, pk))
        return persons

    # ------------------------------------------------------------------
    # 资源释放
    # ------------------------------------------------------------------

    def close(self):
        self._pose_session = None
        super().close()
