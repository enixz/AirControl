"""人脸引导的远距离手部捕获 — 从 base_hand_tracker.py 拆出的独立职责。

丢手时用人脸位置+大小预测手的搜索区，再对该区域 crop-zoom 放大检测，
解决"手在画面角落、居中扫描抓不到"。

原 BaseHandTracker 的 _ensure_face_detector / _face_guided_region /
_try_face_guided_acquire 整体迁入。不持有视口状态——acquire() 返回
(hands, gestures, raw, cx, cy, size)，由 tracker 自己写 _crop_zoom_mode 等。
"""

import logging
import os

import cv2


class FaceGuide:
    """人脸引导的远距离手部捕获：丢手时用人脸位置预测手部搜索区，crop-zoom 检测。"""

    def __init__(self, config, logger=None):
        self._logger = logger or logging.getLogger("gesture")
        self._face_acquire_enabled = True
        self._face_scan_interval = 4          # 每 N 帧（且仅在丢手时）尝试一次人脸扫描
        self._face_scan_counter = 0
        # 搜索区边长 = 人脸高 × 该系数
        self._face_hand_region_scale = 7.0
        # 搜索区中心相对人脸中心下移 = 人脸高 × 该系数
        self._face_hand_down_bias = 1.0
        # 人脸检测时把帧缩到该短边再跑 Haar。原来 240 太小——3 米外人脸只剩 ~13px，
        # 低于 minSize 检不到 → 丢手后找不回。提高到 400 让 ~3-4 米的人脸仍可检出。
        # 越大越能识别更远的脸（恢复能力更强），但人脸扫描更慢。可在 config 调。
        self._face_detect_short = int(
            config.get("face_detect_short", 400)) if config else 400
        self._face_detector_init = False
        self._face_cascade = None

    def acquire(self, frame, w, h, crop_min_size, detect_crop_zoom_cb):
        """节流跑人脸 → 预测搜索区 → 调 detect_crop_zoom_cb 检测。

        Args:
            frame: BGR 原帧
            w, h: 帧宽高
            crop_min_size: 裁剪框机械下限（tracker._crop_min_size）
            detect_crop_zoom_cb: callable(frame, center, size) → (hands, gestures, raw)

        Returns:
            (hands_landmarks, hands_gestures, raw, cx, cy, size) 或 None
            命中时返回检测结果 + 裁剪区坐标；由 tracker 自己写视口状态。
        """
        if not self._face_acquire_enabled:
            return None
        self._face_scan_counter += 1
        if self._face_scan_counter < self._face_scan_interval:
            return None
        self._face_scan_counter = 0
        try:
            region = self._guided_region(frame)
            if region is None:
                return None
            cx, cy, size = region
            size = max(crop_min_size, min(size, min(w, h)))
            res = detect_crop_zoom_cb(frame, (cx, cy), size)
            if res and res[0]:
                self._logger.info("=> ACQUIRE (人脸引导 crop-zoom 捕获到手)")
                return (res[0], res[1], res[2], cx, cy, size)
        except Exception as e:
            self._logger.debug("[FACE] 人脸引导捕获异常: %s", e)
        return None

    def _ensure_detector(self):
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
                        self._logger.info("[FACE] Haar 人脸级联已加载（远距捕获）。")
        except Exception as e:
            self._logger.warning("[FACE] 人脸级联不可用: %s", e)
        return self._face_cascade

    def _guided_region(self, frame):
        """检测最大人脸，据此预测手部搜索区。返回 (cx, cy, size) 或 None。"""
        cascade = self._ensure_detector()
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
