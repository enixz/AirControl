"""手部追踪器渲染层 — 从 base_hand_tracker.py 拆出的纯渲染职责。

集中管理 landmark 圆点绘制、ZOOM/FULL 状态徽章、视觉放大视口。
所有方法纯渲染：仅读写帧像素，不突变 tracker 状态。
"""

import cv2


class HandTrackerRenderer:
    """手部追踪器渲染器：landmark 圆点 + ZOOM 徽章 + 视觉放大。"""

    def __init__(self, crop_min_size=32):
        self._crop_min_size = crop_min_size

    def draw_points(self, frame, landmarks, color):
        """在帧上画 landmark 圆点。

        Args:
            frame: BGR 帧（原地修改）
            landmarks: [[idx, x_px, y_px], ...] × 21 点
            color: (B, G, R) 元组
        """
        for point in landmarks:
            center = (int(round(point[1])), int(round(point[2])))
            cv2.circle(frame, center, 4, color, cv2.FILLED)

    def draw_zoom_badge(self, frame, hands_gestures, frame_w, frame_h, used_zoom):
        """画 ZOOM/FULL 状态徽章（含 bbox 占比）。

        Args:
            frame: BGR 帧（原地修改）
            hands_gestures: gesture dict 列表（读 bbox_area）
            frame_w, frame_h: 帧宽高
            used_zoom: True=ZOOM 徽章，False=FULL 徽章
        """
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

    def apply_visual_zoom(self, frame, crop_center, crop_size):
        """裁剪放大显示视口（视觉 zoom，不影响检测）。

        Args:
            frame: BGR 原帧
            crop_center: (cx, cy) 裁剪中心（原帧像素）
            crop_size: 裁剪边长（像素）

        Returns:
            放大后的帧（尺寸 == 原帧）；若裁剪框接近全图则返回原帧。
        """
        h, w, _ = frame.shape
        cx, cy = crop_center

        base_size = int(round(crop_size))
        base_size = max(base_size, self._crop_min_size)
        base_size = min(base_size, min(w, h))

        if base_size >= int(min(w, h) * 0.95):
            return frame

        crop_h = base_size
        crop_w = int(crop_h * (w / h))
        if crop_w > w:
            crop_w = w
            crop_h = int(crop_w * (h / w))

        x0 = int(round(cx - crop_w / 2))
        y0 = int(round(cy - crop_h / 2))
        x0 = max(0, min(x0, w - crop_w))
        y0 = max(0, min(y0, h - crop_h))

        crop_img = frame[y0:y0+crop_h, x0:x0+crop_w]
        if crop_img.size > 0 and crop_h > 0 and crop_w > 0:
            zoomed = cv2.resize(crop_img, (w, h), interpolation=cv2.INTER_LINEAR)
            cv2.rectangle(zoomed, (0, 0), (w-1, h-1), (0, 200, 255), 6)
            return zoomed
        return frame
