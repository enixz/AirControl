import logging
import threading
import time

from .base import ModeBase, ModeResult

logger = logging.getLogger(__name__)


class MouseMode(ModeBase):
    """鼠标模式：中指控制光标，拇指-食指捏合左键，拇指-中指捏合右键，剪刀手滚动。"""

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
        self.cursor_overlay.show_fullscreen()
        # 新增：状态机初始化
        self._is_left_holding = False
        self._is_right_pinching = False
        # watchdog 触发后阻塞 LEFTDOWN，直到用户真正松开捏合再允许重新按下，
        # 避免主线程恢复后第一帧又把同样的死锁场景重建
        self._left_hold_blocked_until_release = False
        self._last_handle_time = time.time()
        self.cursor_overlay.set_left_hold(False)

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
            self.mouse.reset()
            self.cursor_overlay.hide_cursor()
            return ModeResult(
                gesture="NONE",
                status_text="未检测到手",
                status_color=(0, 0, 255),
            )

        landmarks = hands_landmarks[0]
        features = self.recognizer.get_hand_features(landmarks)
        x_norm = landmarks[12][1] / frame_w
        y_norm = landmarks[12][2] / frame_h
        screen_x, screen_y = self.mouse.move_to_normalized(x_norm, y_norm)
        self.cursor_overlay.update_cursor(screen_x, screen_y)

        # 2. 滚轮检测（持续滚动：保持剪刀手则持续输出）
        scroll_dir = self.recognizer.check_scroll(landmarks, features, False)
        if scroll_dir != 0:
            amount = 120 * scroll_dir
            if self.mouse.scroll_wheel(amount):
                self.cursor_overlay.trigger_scroll(scroll_dir)
                # 滚动时不提前 return，继续处理移动和点击，实现边滚边操作

        # 3. 左键按住状态机（本期核心变更）
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

        # 4. 右键单击（单次触发：捏合期间只触发一次，松开才重置）
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
