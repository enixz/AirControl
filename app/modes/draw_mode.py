import math
import time
import winsound

from .base import ModeBase, ModeResult
from services.mouse_controller import ActiveRegionMapper, interp_tiers


class DrawMode(ModeBase):
    """板书模式：拇指并拢开始书写，拇指分开停止书写。张掌清屏，握拳就绪。"""

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

    def _position_toolbar(self):
        sw = self.overlay.width()
        self.toolbar.move(sw - self.toolbar.width() - 12, 12)
        self.toolbar.raise_()

    def _throttle_special_action(self, interval=0.8):
        if time.time() - self._last_special_action_time < interval:
            return False
        self._last_special_action_time = time.time()
        return True

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

        # 书写手势：食指伸直 + 其余三指弯曲（拇指状态决定启停）
        if features["index_drawing_pose"]:
            self.fist_hold_frames = 0
            self.open_palm_frames = 0

            # 拇指并拢（收拢/靠近手掌）→ 落笔书写
            # 拇指分开（展开/远离手掌）→ 抬笔停止
            if features["thumb_tucked"]:
                self._was_writing = True
                self._writing_lost_frames = 0
                self.overlay.draw_to(x_screen, y_screen)
                return ModeResult(
                    gesture="DRAW",
                    status_text="正在书写",
                    status_color=(0, 255, 255),
                )
            else:
                # 拇指分开缓动：如果之前在书写，允许 5 帧的姿势抖动缓冲
                if self._was_writing:
                    self._writing_lost_frames += 1
                    if self._writing_lost_frames < 5:
                        self.overlay.draw_to(x_screen, y_screen)
                        return ModeResult(
                            gesture="DRAW",
                            status_text="正在书写",
                            status_color=(0, 255, 255),
                        )
                    else:
                        self.overlay.force_lift_pen()
                        self._was_writing = False
                        self._writing_lost_frames = 0
                
                self.overlay.update_cursor(x_screen, y_screen)
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
            self._was_writing = False
            self._writing_lost_frames = 0

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
