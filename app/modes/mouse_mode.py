import logging
import math
import threading
import time

from services.mouse_controller import (
    ActiveRegionMapper,
    blended_landmark_point,
    interp_tiers,
)

from .base import ModeBase, ModeResult

logger = logging.getLogger("gesture")


class MouseMode(ModeBase):
    """鼠标模式：中指控制光标，拇指-食指捏合左键，拇指-中指捏合右键，剪刀手滚动。"""

    # 鼠标距离分段「活动区 span_floor」：远→小(放大够到全屏)，近→大(趋近直接映射、不过敏)。
    # span_floor 方案不牺牲触达——近距离手充满画面，直接绝对映射即可覆盖全屏四角。
    SPAN_FLOOR_TIERS = [
        (0.45, 0.22),  # 很远：放大够到全屏
        (0.90, 0.50),  # 中偏远
        (1.30, 0.95),  # 近
        (1.90, 1.10),  # 很近：抵消 margin 拉伸，接近真正 1:1
    ]
    # 中指末端骨节加权。实录中单独使用 TIP(12) 的静态相对抖动约为
    # 掌根控制点的 2 倍；保留一半 TIP 权重可兼顾指向感和稳定性。
    POINTER_WEIGHTS = ((12, 0.50), (11, 0.25), (10, 0.15), (9, 0.10))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 自适应活动区映射（与板书一致）：离得越远，手扫过的一小块也能映射到全屏，
        # 不必大幅度挥手才能让光标跨屏；近距离 ≈ 直接映射，保留触达。
        # margin 取小（0.04）：减少活动区相对旧"直接映射"带来的整体灵敏度抬升。
        self._region_mapper = ActiveRegionMapper(margin=0.04)
        # hand_width EMA 平滑：避免帧间剧烈波动导致 span_floor 在分段间跳变。
        # alpha=0.15 → 时间常数约 7 帧，平衡平滑性和响应速度。
        self._hand_width_ema = None
        self._hand_width_alpha = 0.15
        # span_floor 滞回：避免 ratio 在分段边界附近反复跳变。
        # _last_span_floor 记录上一帧使用的 span_floor，新值与之差异超过阈值才更新。
        self._last_span_floor = None
        self._span_floor_hysteresis = 0.08  # 滞回带宽
        # Freeze-on-pinch 状态（实施方案 Phase 3.1）：grace 期内光标锁定在
        # 捏合上升沿的瞄准点，消除捏合瞬间手腕连带微动导致的点击漂移。
        # None = 未冻结；(screen_x, screen_y) = 冻结位置
        self._frozen_pos = None
        self._freeze_start = None  # time.time() 冻结起始时刻

    # 守护线程：handle() 静默超过该秒数且 _is_left_holding=True，认为主线程被
    # DefWindowProc 模态循环堵住（典型场景：手势 LEFTDOWN 落在某对话框标题栏触发
    # Win32 窗口拖拽），强制 LEFTUP 打破死锁。
    _STUCK_HOLD_TIMEOUT_SEC = 0.7
    _WATCHDOG_TICK_SEC = 0.2

    def on_enter(self):
        self.overlay.hide()
        self.overlay.setGeometry(-100, -100, 0, 0)
        self.overlay.force_lift_pen()
        self.overlay.hide_cursor()
        self.toolbar.hide()
        self.mouse.reset()  # 重置鼠标状态，防止从其他模式切回时 last_pos 残留
        self._region_mapper.reset()  # 重置活动区，避免沿用上次会话的标定
        # 重置 hand_width 平滑和 span_floor 滞回状态
        self._hand_width_ema = None
        self._last_span_floor = None
        # 重置 freeze-on-pinch 状态
        self._frozen_pos = None
        self._freeze_start = None
        self.cursor_overlay.show_fullscreen()
        # 新增：状态机初始化
        self._is_left_holding = False
        self._is_right_pinching = False
        # watchdog 触发后阻塞 LEFTDOWN，直到用户真正松开捏合再允许重新按下，
        # 避免主线程恢复后第一帧又把同样的死锁场景重建
        self._left_hold_blocked_until_release = False
        self._last_handle_time = time.time()
        self.cursor_overlay.set_left_hold(False)
        if self.recognizer:
            self.recognizer._reset_state()

        # 启动守护线程
        self._watchdog_stop = threading.Event()
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop,
            name="MouseModeWatchdog",
            daemon=True,
        )
        self._watchdog_thread.start()

    def on_exit(self):
        # 先停 watchdog（防止 watchdog 在 on_exit 期间还在改状态）
        if hasattr(self, "_watchdog_stop"):
            self._watchdog_stop.set()
        if hasattr(self, "_watchdog_thread"):
            self._watchdog_thread.join(timeout=1.0)

        # 必须释放左键
        if getattr(self, '_is_left_holding', False):
            self.mouse.left_up()
            self._is_left_holding = False
            self.cursor_overlay.set_left_hold(False)
        self._is_right_pinching = False
        # 清除 freeze-on-pinch 状态
        self._frozen_pos = None
        self._freeze_start = None
        self.mouse.reset()
        self.cursor_overlay.hide_cursor()
        self.cursor_overlay.hide()

    def _watchdog_loop(self):
        """后台线程：检测主线程是否被某个 Win32 模态循环堵住。

        主线程在 DefWindowProc 模态循环（窗口拖拽/SC_CLOSE 按钮跟踪等）期间，
        QueuedConnection 投递的 _on_frame_ready 不会被处理，handle() 调用就停。
        如果此时 _is_left_holding=True，用户即使松开捏合，状态机也无法 LEFTUP，
        而 DefWindowProc 又在等 LEFTUP——互锁，只有真实物理鼠标 UP 能解。

        watchdog 不走 Qt 事件循环，直接 user32.mouse_event LEFTUP 打断。
        """
        while not self._watchdog_stop.wait(self._WATCHDOG_TICK_SEC):
            if not self._is_left_holding:
                continue
            silent = time.time() - self._last_handle_time
            if silent < self._STUCK_HOLD_TIMEOUT_SEC:
                continue
            logger.warning(
                "鼠标模式 watchdog：handle() 静默 %.2fs 且左键 hold 中，"
                "强制 LEFTUP 打破模态循环死锁",
                silent,
            )
            try:
                self.mouse.left_up()
            except Exception:
                logger.exception("watchdog 强制 LEFTUP 异常")
            # 注意：cursor_overlay 是 Qt 控件，主线程被堵时它的 update 也不会刷新，
            # 暂不在此处调用；等主线程恢复后 handle() 自己会同步视觉状态
            self._is_left_holding = False
            self._left_hold_blocked_until_release = True
            # 复位时间戳：避免主线程恢复后第一帧又被 watchdog 再次判定为 stuck
            self._last_handle_time = time.time()

    def handle(self, hands_landmarks, hands_gestures, frame_w, frame_h) -> ModeResult:
        self._last_handle_time = time.time()
        self._sync_frame_size(frame_w, frame_h)
        # 除零保护：异常帧（摄像头初始化/重连）时 frame_w/h 可能为 0

        # Rate-limited pinch telemetry logging (once every 15 frames) to gesture.log
        if hands_landmarks:
            self._frame_count = getattr(self, "_frame_count", 0) + 1
            if self._frame_count % 15 == 0:
                landmarks = hands_landmarks[0]
                features = self.recognizer.get_hand_features(landmarks)
                thumb_index = math.hypot(landmarks[4][1] - landmarks[8][1], landmarks[4][2] - landmarks[8][2])
                thumb_middle = math.hypot(landmarks[4][1] - landmarks[12][1], landmarks[4][2] - landmarks[12][2])
                hand_width = features.get("hand_width", 40.0)
                pinch_threshold = hand_width * 0.35
                # Phase 3.2: 记录 pinch 比值（距离/掌宽），用于标定 ENTER/EXIT 阈值
                idx_ratio = thumb_index / hand_width if hand_width > 0 else 0.0
                mid_ratio = thumb_middle / hand_width if hand_width > 0 else 0.0
                hyst_on = getattr(self.recognizer, 'pinch_hysteresis_enabled', False)
                logger.info(
                    "[MouseMode] pinch telemetry: thumb_idx=%.1f, thumb_mid=%.1f, hand_w=%.1f, thresh=%.1f, "
                    "idx_ratio=%.3f, mid_ratio=%.3f, hyst=%s, "
                    "left_pinch=%s, mid_pinch=%s, hold=%s, blocked=%s",
                    thumb_index, thumb_middle, hand_width, pinch_threshold,
                    idx_ratio, mid_ratio, hyst_on,
                    features.get("thumb_index_pinch"), features.get("thumb_middle_pinch"),
                    self._is_left_holding, self._left_hold_blocked_until_release
                )
        if frame_w <= 0 or frame_h <= 0:
            return ModeResult(gesture="NONE")
        # 1. 手丢失：释放左键 + 重置
        if not hands_landmarks:
            if getattr(self, '_is_left_holding', False):
                self.mouse.left_up()
                self._is_left_holding = False
                self.cursor_overlay.set_left_hold(False)
            # 手丢失也清掉 watchdog 锁定，下次进入捏合视为新的操作
            self._left_hold_blocked_until_release = False
            self._is_right_pinching = False
            # 清除 freeze-on-pinch 状态
            self._frozen_pos = None
            self._freeze_start = None
            self.mouse.reset()
            self.cursor_overlay.hide_cursor()
            # 重置 hand_width 平滑和 span_floor 滞回状态
            self._hand_width_ema = None
            self._last_span_floor = None
            return ModeResult(
                gesture="NONE",
                status_text="未检测到手",
                status_color=(0, 0, 255),
            )

        landmarks = hands_landmarks[0]
        features = self.recognizer.get_hand_features(landmarks)
        # 先把中指尖归一化坐标经活动区映射拉伸到全屏（远距离也能轻松跨屏），
        # 再交给 move_to_normalized 做灵敏度平滑+落点。v1.3.6 稳定档默认不再叠加
        # 边缘加速，避免“活动区映射 + 边缘加速”双重放大；用户开启边缘加速时才启用。
        # span_floor 按距离分段：近→趋近直接映射（不过敏），远→放大够到全屏。
        # 对 hand_width 做 EMA 平滑 + span_floor 滞回，避免帧间抖动导致灵敏度跳变。
        raw_hand_width = features["hand_width"]
        if self._hand_width_ema is None:
            self._hand_width_ema = raw_hand_width
        else:
            self._hand_width_ema = (
                self._hand_width_ema * (1 - self._hand_width_alpha)
                + raw_hand_width * self._hand_width_alpha
            )
        ratio = self._hand_width_ema / self.recognizer.REFERENCE_HAND_WIDTH
        new_span_floor = interp_tiers(ratio, self.SPAN_FLOOR_TIERS)
        # 滞回：新 span_floor 与上一帧差异超过带宽才更新，避免边界附近反复跳变
        if self._last_span_floor is None or abs(new_span_floor - self._last_span_floor) >= self._span_floor_hysteresis:
            span_floor = new_span_floor
            self._last_span_floor = new_span_floor
        else:
            span_floor = self._last_span_floor
        pointer_x, pointer_y = blended_landmark_point(
            landmarks, self.POINTER_WEIGHTS
        )
        x_norm = pointer_x / frame_w
        y_norm = pointer_y / frame_h
        nx, ny = self._region_mapper.map(x_norm, y_norm, span_floor=span_floor)
        apply_accel = bool(self.config.get("edge_acceleration_enabled", False)) if self.config else False

        # === Freeze-on-pinch（实施方案 Phase 3.1）===
        # 借鉴 Air-Cursor freeze-on-fist，本地化为 pinch（AC-trae 点击手势是捏合）。
        # grace 期内光标锁在捏合上升沿的瞄准点，消除捏合瞬间手腕连带微动+landmark
        # 抖动导致的点击漂移。grace 结束后解冻，恢复正常移动（DRAG）。
        # 参考 draw_mode.py:355-360 的 freeze 先例（书写中冻结活动区映射）。
        pinch_freeze_enabled = bool(self.config.get("pinch_freeze_enabled", False)) if self.config else False
        if (pinch_freeze_enabled and self._is_left_holding
                and self._frozen_pos is not None):
            grace_sec = float(self.config.get("pinch_freeze_grace_sec", 0.3)) if self.config else 0.3
            if time.time() - self._freeze_start < grace_sec:
                # grace 期内：光标锁定在冻结位置，跳过 move_to_normalized
                screen_x, screen_y = self._frozen_pos
            else:
                # grace 结束：解冻，恢复 DRAG 移动
                self._frozen_pos = None
                self._freeze_start = None
                screen_x, screen_y = self.mouse.move_to_normalized(nx, ny, apply_accel=apply_accel)
        else:
            screen_x, screen_y = self.mouse.move_to_normalized(nx, ny, apply_accel=apply_accel)
        self.cursor_overlay.update_cursor(screen_x, screen_y)

        # 2. 滚轮检测（持续滚动：保持剪刀手则持续输出）
        scroll_dir = self.recognizer.check_scroll(landmarks, features, False)
        if scroll_dir != 0:
            amount = 120 * scroll_dir
            if self.mouse.scroll_wheel(amount):
                self.cursor_overlay.trigger_scroll(scroll_dir)
                # 滚动时不提前 return，继续处理移动和点击，实现边滚边操作

        # 3. 左键按住状态机（使用 recognizer 的 pinch 特征，与老版一致）
        if features["thumb_index_pinch"]:
            # watchdog 强制释放过一次后，必须等用户先松手才允许再 LEFTDOWN，
            # 否则主线程刚解锁就在同一位置再次进入模态循环重新死锁
            if self._left_hold_blocked_until_release:
                return ModeResult(
                    gesture="MOUSE_MOVE",
                    status_text="松手后再按",
                    status_color=(200, 160, 0),
                )
            if not self._is_left_holding:
                self.mouse.left_down()
                self._is_left_holding = True
                self.cursor_overlay.set_left_hold(True)
                # Freeze-on-pinch: 在上升沿记录瞄准点为冻结位置
                # （screen_x/y 已由上方 move_to_normalized 计算得出）
                if pinch_freeze_enabled:
                    self._frozen_pos = (screen_x, screen_y)
                    self._freeze_start = time.time()
            # 按住期间继续移动光标，每帧返回 HOLD 状态
            return ModeResult(
                gesture="LEFT_HOLD",
                status_text="Left Button Hold",
                status_color=(255, 0, 0),
            )
        else:
            if self._is_left_holding:
                self.mouse.left_up()
                self._is_left_holding = False
                self.cursor_overlay.set_left_hold(False)
            # 用户松开捏合 → 解除 watchdog 锁定，下次捏合可正常 LEFTDOWN
            self._left_hold_blocked_until_release = False
            # 松开捏合也清除 freeze 状态（grace 是上限，pinch 释放立即解冻）
            self._frozen_pos = None
            self._freeze_start = None

        # 4. 右键单击（单次触发：使用 recognizer 的 pinch 特征）
        if features["thumb_middle_pinch"]:
            if not self._is_right_pinching:
                self._is_right_pinching = True
                self.mouse.right_click()
                self.cursor_overlay.trigger_right_click(screen_x, screen_y)
                return ModeResult(
                    gesture="RIGHT_CLICK",
                    status_text="Mouse Right Click",
                    status_color=(80, 220, 120),
                )
        else:
            self._is_right_pinching = False

        return ModeResult(
            gesture="MOUSE_MOVE",
            status_text="Mouse Move",
            status_color=(0, 255, 0),
        )
