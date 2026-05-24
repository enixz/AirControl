import math
import time
import ctypes
import ctypes.wintypes
import logging

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QBrush, QColor, QPainter, QPen
from PyQt6.QtWidgets import QApplication, QWidget

GWL_EXSTYLE = -20
WS_EX_LAYERED = 0x00080000
WS_EX_TRANSPARENT = 0x00000020
WS_EX_TOOLWINDOW = 0x00000080  # 不在任务栏和 Alt-Tab 中显示

# SetWindowPos 用：把光标层钉在 topmost 组的最上面，避免被后开的对话框遮住
_HWND_TOPMOST = -1
_SWP_NOMOVE = 0x0002
_SWP_NOSIZE = 0x0001
_SWP_NOACTIVATE = 0x0010
_SWP_NOSENDCHANGING = 0x0400


class MouseCursorOverlay(QWidget):
    """鼠标模式下的视觉反馈层。

    设计要点：
    - 全屏覆盖，但**完全不拦截**鼠标/键盘事件（纯 Win32 WS_EX_TRANSPARENT）。
    - 不依赖 Qt.WindowTransparentForInput（该枚举在不同 PyQt6 版本行为不一致）。
    - 系统光标由 MouseController 通过 SetCursorPos 直接控制，本层仅做视觉装饰。
    """

    CURSOR_RADIUS = 28
    DOT_RADIUS = 5
    CLICK_DURATION = 0.4
    SCROLL_DURATION = 0.35
    HOLD_PULSE_FREQ = 10   # Hz

    def __init__(self, parent=None):
        super().__init__(parent)
        # 只用最基础的顶层窗口标志；点击穿透完全由 _make_click_through() 通过 Win32 API 控制
        # 设置 parent 可防止无父窗口的 Tool 窗口偶尔出现在任务栏
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        # WA_TransparentForMouseEvents 让 Qt 本身也不处理鼠标事件，与 WS_EX_TRANSPARENT 双重保险
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)

        self._pos = None
        self._click = None
        self._scroll = None
        self._left_hold = False
        self._system_cursor_hidden = False
        self._dpr = QApplication.primaryScreen().devicePixelRatio()
        self._click_through_applied = False

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        # _tick 内部计数器，每 N 次主 tick 重新把自己拉到 topmost 组最上层
        self._topmost_tick = 0
        # 定时器在 show_fullscreen 时启动，hide 时停止，避免隐藏时无意义调用
        self.hide()

    def _hide_system_cursor(self):
        """隐藏 Windows 系统鼠标指针（幂等：多次调用安全）"""
        if self._system_cursor_hidden:
            return
        try:
            while ctypes.windll.user32.ShowCursor(False) >= 0:
                pass
            self._system_cursor_hidden = True
        except Exception as e:
            logging.warning("Hide cursor failed: %s", e)

    def _show_system_cursor(self):
        """恢复 Windows 系统鼠标指针（幂等）"""
        if not self._system_cursor_hidden:
            return
        try:
            while ctypes.windll.user32.ShowCursor(True) < 0:
                pass
            self._system_cursor_hidden = False
        except Exception as e:
            logging.warning("Show cursor failed: %s", e)

    def _make_click_through(self):
        """通过 Win32 API 将窗口设为完全透明输入（鼠标/键盘穿透到底层桌面），
        并加入 WS_EX_TOOLWINDOW 确保不在任务栏和 Alt-Tab 中显示。"""
        if self._click_through_applied:
            return
        try:
            hwnd = int(self.winId())
            style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            style |= WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOOLWINDOW
            ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style)
            SWP_NOMOVE = 0x0002
            SWP_NOSIZE = 0x0001
            SWP_NOZORDER = 0x0004
            SWP_FRAMECHANGED = 0x0020
            ctypes.windll.user32.SetWindowPos(
                hwnd, 0, 0, 0, 0, 0,
                SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_FRAMECHANGED,
            )
            self._click_through_applied = True
        except Exception as e:
            logging.error("_make_click_through failed: %s", e)

    def show_fullscreen(self):
        """显示全屏覆盖层。**必须在 show() 之后再设置 WS_EX_TRANSPARENT**，
        否则 Qt 的 show() 会覆盖掉我们手动设置的扩展样式。
        每次显示前重置 _click_through_applied，防止隐藏后再显示时 Win32 样式丢失。"""
        self._click_through_applied = False  # 强制重新应用 Win32 样式
        screen = QApplication.primaryScreen().geometry()
        self.setGeometry(screen)
        self.show()
        self.raise_()
        # 关键：show() 之后再设置点击穿透
        self._make_click_through()
        # 强制重置标志，确保 _hide_system_cursor 不会因标志不一致而跳过隐藏。
        # 外部 Win32 操作（如语音助手激活）可能改变了 ShowCursor 引用计数器，
        # 导致 _system_cursor_hidden 标志与实际状态不同步。
        self._system_cursor_hidden = False
        self._hide_system_cursor()
        self._timer.start(16)  # 约60 FPS，用于更新点击/滚动动画

    def update_cursor(self, x, y):
        """更新自定义光标位置。"""
        self._pos = (int(x / self._dpr), int(y / self._dpr))
        if self.isVisible():
            self._hide_system_cursor()
            # 鼠标动了就立刻保证可见，不依赖定时器的 50ms 节奏
            self._reapply_topmost()
            self.update()

    def hide_cursor(self):
        """停止绘制自定义光标并恢复系统光标。"""
        self._pos = None
        self._show_system_cursor()
        self.update()

    def hide(self):
        """隐藏窗口并确保系统光标恢复。"""
        self._timer.stop()
        self._show_system_cursor()
        super().hide()

    # ---- 视觉反馈 API ----

    def trigger_left_click(self, x, y):
        self._click = ("left", time.time(), int(x / self._dpr), int(y / self._dpr))
        self.update()

    def trigger_right_click(self, x, y):
        self._click = ("right", time.time(), int(x / self._dpr), int(y / self._dpr))
        self.update()

    def trigger_scroll(self, direction):
        self._scroll = (1 if direction > 0 else -1, time.time())
        self.update()

    def set_left_hold(self, holding: bool):
        """设置左键按住视觉反馈状态。"""
        self._left_hold = holding
        self.update()

    def _tick(self):
        now = time.time()
        dirty = False
        if self._click and now - self._click[1] >= self.CLICK_DURATION:
            self._click = None
            dirty = True
        elif self._click:
            dirty = True
        if self._scroll and now - self._scroll[1] >= self.SCROLL_DURATION:
            self._scroll = None
            dirty = True
        elif self._scroll:
            dirty = True
        if dirty:
            self.update()

        # 每 ~50ms 重新拉到 topmost 组最上层。否则后开的 topmost 窗口
        # （如语音指令面板）会盖在光标上面，用户看不到自己鼠标移到了哪里。
        # WS_EX_TRANSPARENT 已保证点击穿透，把光标钉在最上面不影响交互。
        # SetWindowPos NOACTIVATE/NOMOVE/NOSIZE 路径很轻量，60Hz 调用也没明显开销。
        self._topmost_tick += 1
        if self._topmost_tick >= 3:  # 3 × 16ms ≈ 48ms ≈ 20Hz
            self._topmost_tick = 0
            self._reapply_topmost()

    def _reapply_topmost(self):
        """通过 Win32 SetWindowPos 把自己钉回 topmost 组最上层。
        NOACTIVATE 保证不抢焦点，NOMOVE/NOSIZE 不动几何。
        """
        try:
            ctypes.windll.user32.SetWindowPos(
                int(self.winId()), _HWND_TOPMOST, 0, 0, 0, 0,
                _SWP_NOMOVE | _SWP_NOSIZE | _SWP_NOACTIVATE | _SWP_NOSENDCHANGING,
            )
        except Exception:
            # 静默失败：失效一帧不要紧，下次 tick 再试
            pass

    def paintEvent(self, event):
        if self._pos is None:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        x, y = self._pos
        now = time.time()
        R = self.CURSOR_RADIUS

        if self._click:
            ctype, t0, cx, cy = self._click
            t = min((now - t0) / self.CLICK_DURATION, 1.0)
            alpha = int(200 * (1 - t))
            r = int(R * (1 + t * 2.8))
            if ctype == "left":
                c = QColor(60, 140, 255, alpha)
                ring_border = QColor(100, 180, 255, max(30, alpha))
            else:
                c = QColor(80, 220, 120, alpha)
                ring_border = QColor(120, 255, 160, max(30, alpha))
            p.setPen(QPen(ring_border, max(1, int(3 * (1 - t)))))
            p.setBrush(QBrush(c))
            p.drawEllipse(cx - r, cy - r, r * 2, r * 2)

        # 环颜色：按住时为红色脉冲，否则浅灰色
        if self._left_hold:
            pulse = abs(math.sin(now * self.HOLD_PULSE_FREQ))
            ring_color = QColor(255, int(40 * pulse), int(40 * pulse), 200)
            pen_width = 3.5
        else:
            ring_color = QColor(200, 200, 200, 160)
            pen_width = 2.5

        p.setPen(QPen(ring_color, pen_width))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(x - R, y - R, R * 2, R * 2)

        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(QColor(200, 200, 200, 230)))
        p.drawEllipse(x - self.DOT_RADIUS, y - self.DOT_RADIUS,
                      self.DOT_RADIUS * 2, self.DOT_RADIUS * 2)

        p.setPen(QPen(QColor(200, 200, 200, 60), 1))
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            bx = x + dx * (R + 5)
            by = y + dy * (R + 5)
            p.drawLine(bx - 4 * abs(dy), by - 4 * abs(dx),
                       bx + 4 * abs(dy), by + 4 * abs(dx))

        if self._click:
            ctype, t0, cx, cy = self._click
            t = min((now - t0) / self.CLICK_DURATION, 1.0)
            if t < 1.0:
                label = "L" if ctype == "left" else "R"
                label_alpha = int(255 * (1 - t))
                label_color = QColor(60, 140, 255, label_alpha) if ctype == "left" else QColor(80, 220, 120, label_alpha)
                font = p.font()
                font.setPointSize(max(12, int(18 * (1 - t * 0.3))))
                font.setBold(True)
                p.setFont(font)
                p.setPen(QPen(label_color))
                ly = cy - int(R * (1 + t * 1.5)) - 8
                p.drawText(cx - 8, ly, label)

        if self._scroll:
            direction, t0 = self._scroll
            t = min((now - t0) / self.SCROLL_DURATION, 1.0)
            alpha = int(220 * (1 - t))
            offset = int(25 * t * direction)
            arrow_c = QColor(255, 200, 50, alpha)
            p.setPen(QPen(arrow_c, 3, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            if direction > 0:
                base_y = y - R - 20 + offset
            else:
                base_y = y + R + 20 + offset
            p.drawLine(x - 7, base_y - 5 * direction, x, base_y - 12 * direction)
            p.drawLine(x + 7, base_y - 5 * direction, x, base_y - 12 * direction)
            p.drawLine(x, base_y - 12 * direction, x, base_y + 12 * direction)

        p.end()
