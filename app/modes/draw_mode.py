import math
import time
import winsound

from .base import ModeBase, ModeResult


class DrawMode(ModeBase):
    """板书模式：拇指并拢开始书写，拇指分开停止书写。张掌清屏，握拳就绪。"""

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
        x_screen, y_screen = self.mouse.to_screen(
            landmarks[5][1] / frame_w, landmarks[5][2] / frame_h
        )
        self.overlay.update_cursor(x_screen, y_screen)

        # 笔粗距离自适应：bbox 大（近）→ 笔粗，bbox 小（远）→ 笔细
        # sqrt(bbox) 比 bbox 本身随距离变化更线性
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
                # 拇指分开时强制抬笔
                if self._was_writing:
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
