import logging
import math
import time
import winsound

from .base import ModeBase, ModeResult
from services.mouse_controller import ActiveRegionMapper, interp_tiers

logger = logging.getLogger("gesture")


class DrawMode(ModeBase):
    """板书模式：拇指并拢落笔、分开抬笔（仅手正对相机时拇指可信）；
    ✌️ 食指+中指伸出（贴紧也算）随时抬笔——侧面剪影可辨，不依赖拇指。
    手侧偏时拇指被遮挡，冻结笔状态防误抬。张掌清屏，握拳就绪。"""

    # 移动灵敏度随"手在画面中的大小"换算的标定区间（相对参考掌宽的比值）。
    # 我们无法感知真实距离，只能用掌宽作为手在画面里大小的度量：
    #   掌宽 ≤ 参考×RESP_RATIO_FAR（手小/离得远）→ 最敏感（系数 1.0）；
    #   掌宽 ≥ 参考×RESP_RATIO_NEAR（手大/离得近）→ 最迟钝（系数 0.0）。
    RESP_RATIO_FAR = 0.55
    RESP_RATIO_NEAR = 1.5
    # 距离分段「活动区 span_floor」（piecewise-linear）：ratio=掌宽/参考掌宽（越大=越近）。
    # span_floor 越小→小动作越被放大（远距离写满全屏）；越大（→1.0）→趋近直接绝对映射、
    # gain≈1、画圆轻松（找回早期卡尔曼版的近距离手感），且不牺牲触达（近距离手本就充满画面）。
    # 嫌近距离还太快就把近端调到 1.0~1.2（>1 会略损触达）；想远距离更省力就把远端调小。
    SPAN_FLOOR_TIERS = [
        (0.45, 0.22),  # 很远：小动作放大、写满全屏
        (0.70, 0.45),  # 远
        (1.00, 0.90),  # 中（舒适书写距离）：接近直接映射
        (1.30, 1.00),  # 近及以内：直接绝对映射、gain≈1、轻松画圆
        (1.90, 1.00),  # 很近
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._was_writing = False
        self._writing_lost_frames = 0
        self.fist_hold_frames = 0
        self.open_palm_frames = 0
        self._last_special_action_time = 0
        self._was_fist = False
        self._fist_tap_times: list[float] = []
        self._double_fist_cooldown = 0.0
        # 自适应活动区映射：远距离也能写满全屏（替代固定的绝对映射）
        self._region_mapper = ActiveRegionMapper(margin=0.05)
        # 双指悬停 / 拇指分开的连续帧计数（去抖）
        self._two_finger_frames = 0
        self._thumb_apart_frames = 0
        # 标定遥测节流
        self._last_telemetry = 0.0
        # 正面度低于此值视为拇指不可观测（侧对相机），冻结笔状态
        self._frontality_gate = (
            float(self.config.get("draw_frontality_gate", 0.55)) if self.config else 0.55
        )

    def on_enter(self):
        self.overlay.show_fullscreen()
        self.toolbar.show()
        self._position_toolbar()
        self.cursor_overlay.hide()
        self.cursor_overlay.setGeometry(-100, -100, 0, 0)
        self._was_writing = False
        self._writing_lost_frames = 0
        self.fist_hold_frames = 0
        self.open_palm_frames = 0
        self._was_fist = False
        self._fist_tap_times.clear()
        self._double_fist_cooldown = 0.0
        self._two_finger_frames = 0
        self._thumb_apart_frames = 0
        self._region_mapper.reset()

    def on_exit(self):
        self.overlay.hide()
        self.overlay.setGeometry(-100, -100, 0, 0)
        self.overlay.force_lift_pen()
        self.overlay.hide_cursor()
        self.toolbar.hide()
        self._was_writing = False
        self._writing_lost_frames = 0
        self.fist_hold_frames = 0
        self.open_palm_frames = 0
        self._was_fist = False
        self._fist_tap_times.clear()
        self._double_fist_cooldown = 0.0
        self._two_finger_frames = 0
        self._thumb_apart_frames = 0

    def _position_toolbar(self):
        sw = self.overlay.width()
        self.toolbar.move(sw - self.toolbar.width() - 12, 12)
        self.toolbar.raise_()

    def _throttle_special_action(self, interval=0.8):
        if time.time() - self._last_special_action_time < interval:
            return False
        self._last_special_action_time = time.time()
        return True

    def _log_pen(self, event, cause, features, label):
        """笔起落事件：记录触发原因和当时的关键比值，用于实机标定。"""
        logger.info(
            "[DRAW] pen_%s cause=%s label=%s frontality=%.2f thumb_ratio=%.2f middle_ratio=%.2f",
            event, cause, label, features["hand_frontality"],
            features["thumb_ratio"], features["middle_ratio"],
        )

    def _telemetry(self, features, label, state):
        """书写/悬停期间每 0.5s 记录一次比值流，便于在 gesture.log 标定
        draw_frontality_gate 与双指/拇指阈值。"""
        now = time.time()
        if now - self._last_telemetry < 0.5:
            return
        self._last_telemetry = now
        logger.info(
            "[DRAW] state=%s label=%s frontality=%.2f thumb_ratio=%.2f middle_ratio=%.2f tucked=%s",
            state, label, features["hand_frontality"], features["thumb_ratio"],
            features["middle_ratio"], features["thumb_tucked"],
        )

    def handle(self, hands_landmarks, hands_gestures, frame_w, frame_h) -> ModeResult:
        self._sync_frame_size(frame_w, frame_h)
        # 除零保护：异常帧时 frame_w/h 可能为 0
        if frame_w <= 0 or frame_h <= 0:
            return ModeResult(gesture="NONE")
        if not self.overlay.isVisible():
            self.overlay.show_fullscreen()

        if not hands_landmarks:
            self.fist_hold_frames = 0
            self.open_palm_frames = 0
            
            # 手部跟丢缓冲：如果之前正在书写，允许短暂跟丢而不中断笔画
            if self._was_writing:
                self._writing_lost_frames += 1
                if self._writing_lost_frames < 8:
                    # 保持之前的书写状态，使用最后已知的屏幕坐标（由 overlay 维持）
                    return ModeResult(
                        gesture="DRAW",
                        status_text="正在书写 (跟丢缓冲)",
                        status_color=(0, 255, 255),
                    )
            
            self._was_writing = False
            self._writing_lost_frames = 0
            self.overlay.hide_cursor()
            self.overlay.tick_idle()
            return ModeResult(
                gesture="NONE",
                status_text="未检测到手",
                status_color=(0, 0, 255),
            )

        landmarks = hands_landmarks[0]
        features = self.recognizer.get_hand_features(landmarks)

        # 灵敏度随"手在画面中的大小"（掌宽）换算：手小（离得远）→ 高灵敏，
        # 手大（离得近）→ 低灵敏。responsiveness∈[0,1]：1=远/敏感，0=近/迟钝。
        # 不依赖无法感知的真实距离——只用掌宽这一画面度量。
        ref_width = self.recognizer.REFERENCE_HAND_WIDTH
        ratio = features["hand_width"] / ref_width
        resp_span = self.RESP_RATIO_NEAR - self.RESP_RATIO_FAR
        responsiveness = max(0.0, min(1.0, (self.RESP_RATIO_NEAR - ratio) / resp_span))
        # (1) 平滑：手大（近）→ 强平滑、稳，少发抖。
        self.overlay.set_motion_responsiveness(responsiveness)
        # (2) 位移增益：按距离分段设活动区 span_floor（见 SPAN_FLOOR_TIERS）。近→span_floor
        #     趋近 1.0 = 直接绝对映射 gain≈1（画圆轻松、找回卡尔曼手感、且不丢触达）；
        #     远→span_floor 小 = 小动作放大、写满全屏。分段表可独立微调各距离段手感。
        span_floor = interp_tiers(ratio, self.SPAN_FLOOR_TIERS)

        # 自适应活动区：把手最近扫过的范围映射到全屏（远距离也能写满），
        # 不再叠加边缘加速（apply_accel=False），避免双重映射失真。
        # 书写中冻结活动区与增益（update=False）：笔画期间传递函数恒定，避免中途
        # 重标定引起的漂移/抖动；只在悬停/就绪时校准。
        is_drawing_now = features["index_drawing_pose"] and features["thumb_tucked"]
        freeze = is_drawing_now or self._was_writing
        nx, ny = self._region_mapper.map(
            landmarks[5][1] / frame_w, landmarks[5][2] / frame_h,
            update=not freeze, span_floor=span_floor,
        )
        x_screen, y_screen = self.mouse.to_screen(nx, ny, apply_accel=False)
        self.overlay.update_cursor(x_screen, y_screen)

        # 笔粗距离自适应（默认关闭，见 config pen_width_auto_scale）：
        # bbox 大（近）→ 笔粗，bbox 小（远）→ 笔细。sqrt(bbox) 比 bbox 随距离更线性。
        if hands_gestures:
            bbox_area = hands_gestures[0].get("bbox_area", 0.0)
            if bbox_area > 0:
                hand_size = math.sqrt(bbox_area)
                scale = hand_size / self.overlay.REFERENCE_HAND_SIZE
                self.overlay.set_pen_scale(scale)

        is_explicit_stop = features["is_fist"] or features["is_open_palm"]
        label = hands_gestures[0].get("label", "OTHER") if hands_gestures else "OTHER"

        # ✌️ 双指 = 抬笔悬停：食指+中指伸出即可（贴紧也算，不要求张开），
        # VICTORY 标签兜底。伸出手指的剪影在手侧对相机时依然可辨——
        # 拇指被遮挡时的可靠抬笔手段。连续 2 帧去抖。
        if features["two_finger_hover"] or label == "VICTORY":
            self._two_finger_frames += 1
        else:
            self._two_finger_frames = 0
        if self._two_finger_frames >= 2:
            if self._was_writing:
                self._log_pen("lift", "two_finger", features, label)
                self.overlay.force_lift_pen()
            self._was_writing = False
            self._writing_lost_frames = 0
            self._thumb_apart_frames = 0
            self.fist_hold_frames = 0
            self.open_palm_frames = 0
            self._telemetry(features, label, "hover_two_finger")
            return ModeResult(
                gesture="DRAW_HOVER",
                status_text="悬停（双指）",
                status_color=(180, 180, 180),
            )

        # 拇指可观测性门控：手侧对相机（正面度低）时拇指常被整只手遮挡，
        # 关键点是模型脑补的，不允许它改变笔的起落
        thumb_readable = features["hand_frontality"] >= self._frontality_gate

        # 书写中：用宽松条件维持（食指仍伸直即可——距离判定抗偏航）；
        # 严格的 index_drawing_pose 只把守进入书写的门
        if self._was_writing and features["index_extended"] and not is_explicit_stop:
            self.fist_hold_frames = 0
            self.open_palm_frames = 0
            # 抬笔要求拇指可读且连续 3 帧分开：侧偏时冻结书写状态，
            # 故意抬笔发生在笔画末端、手偏正面的时刻，不受影响
            if thumb_readable and not features["thumb_tucked"]:
                self._thumb_apart_frames += 1
            else:
                self._thumb_apart_frames = 0
            if self._thumb_apart_frames >= 3:
                self._thumb_apart_frames = 0
                self._was_writing = False
                self._writing_lost_frames = 0
                self._log_pen("lift", "thumb_apart", features, label)
                self.overlay.force_lift_pen()
                return ModeResult(
                    gesture="DRAW_HOVER",
                    status_text="悬停（拇指分开）",
                    status_color=(180, 180, 180),
                )
            self._writing_lost_frames = 0
            self.overlay.draw_to(x_screen, y_screen)
            self._telemetry(features, label, "writing")
            return ModeResult(
                gesture="DRAW",
                status_text="正在书写",
                status_color=(0, 255, 255),
            )

        # 未在书写、摆出单指书写姿势：决定是否落笔
        if not self._was_writing and features["index_drawing_pose"]:
            self.fist_hold_frames = 0
            self.open_palm_frames = 0
            self._thumb_apart_frames = 0
            # 正面：拇指并拢落笔（习惯不变）；侧面拇指不可读时，
            # 由 ML 标签 Pointing_Up（单指姿势）判定书写意图
            if (thumb_readable and features["thumb_tucked"]) or (
                not thumb_readable and label == "POINTING_UP"
            ):
                self._was_writing = True
                self._writing_lost_frames = 0
                self._log_pen(
                    "down",
                    "thumb_tucked" if thumb_readable else "ml_pointing_up",
                    features, label,
                )
                self.overlay.draw_to(x_screen, y_screen)
                return ModeResult(
                    gesture="DRAW",
                    status_text="正在书写",
                    status_color=(0, 255, 255),
                )
            self._telemetry(features, label, "hover")
            return ModeResult(
                gesture="DRAW_HOVER",
                status_text="悬停（拇指分开）",
                status_color=(180, 180, 180),
            )

        # 书写丢失缓冲（姿势短暂偏离时保持书写）
        if self._was_writing and not is_explicit_stop:
            self._writing_lost_frames += 1
            if self._writing_lost_frames < 10:
                self.overlay.draw_to(x_screen, y_screen)
                return ModeResult(
                    gesture="DRAW",
                    status_text="正在书写",
                    status_color=(0, 255, 255),
                )
            self._log_pen("lift", "pose_lost", features, label)
            self._was_writing = False
            self._writing_lost_frames = 0

        if self._was_writing and is_explicit_stop:
            self._log_pen("lift", "explicit_stop", features, label)
        self._was_writing = False
        self._writing_lost_frames = 0

        is_fist = features["is_fist"]
        now = time.time()

        if self._was_fist and not is_fist:
            if now > self._double_fist_cooldown:
                self._fist_tap_times.append(now)
                self._fist_tap_times = [t for t in self._fist_tap_times if now - t < 1.0]
                if len(self._fist_tap_times) >= 2:
                    self._fist_tap_times.clear()
                    self._double_fist_cooldown = now + 1.0
                    enabled = self.overlay.toggle_shape_correction()
                    self.toolbar.set_shape_correction(enabled)
                    label = "已开启" if enabled else "已关闭"
                    try:
                        winsound.PlaySound(
                            "SystemAsterisk",
                            winsound.SND_ALIAS | winsound.SND_ASYNC,
                        )
                    except RuntimeError:
                        pass
                    self._was_fist = is_fist
                    return ModeResult(
                        gesture="TOGGLE_SHAPE_CORRECTION",
                        status_text=f"图形修正{label}",
                        status_color=(0, 200, 100) if enabled else (180, 180, 180),
                    )
        self._was_fist = is_fist

        # 握拳 -> 就绪
        if is_fist:
            self.fist_hold_frames = 0
            self.open_palm_frames = 0
            self.overlay.tick_idle()
            return ModeResult(
                gesture="DRAW_READY",
                status_text="板书模式就绪",
                status_color=(0, 255, 0),
            )

        # 张掌 -> 清屏（长按）
        if features["is_open_palm"]:
            self.open_palm_frames += 1
            self.overlay.tick_idle()
            if self.open_palm_frames >= 30 and self._throttle_special_action():
                self.overlay.clear_canvas()
                self.fist_hold_frames = 0
                self.open_palm_frames = 0
                try:
                    winsound.PlaySound(
                        "SystemExclamation",
                        winsound.SND_ALIAS | winsound.SND_ASYNC,
                    )
                except RuntimeError:
                    winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
                return ModeResult(
                    gesture="CLEAR_CANVAS",
                    status_text="已清空画布",
                    status_color=(0, 165, 255),
                )
            progress = self.open_palm_frames / 30 * 100
            return ModeResult(
                gesture="DRAW_READY",
                status_text=f"清屏中... {progress:.0f}%",
                status_color=(255, 165, 0),
            )
        else:
            self.fist_hold_frames = 0
            self.open_palm_frames = 0
            self.overlay.tick_idle()

        return ModeResult(
            gesture="DRAW_READY",
            status_text="板书模式就绪",
            status_color=(0, 255, 0),
        )
