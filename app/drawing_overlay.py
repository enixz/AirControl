import ctypes
import math
from collections import deque

from PyQt6.QtCore import QPoint, QPointF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QApplication, QWidget

from services.shape_recognizer import recognize_and_correct

GWL_EXSTYLE = -20
WS_EX_LAYERED = 0x00080000
WS_EX_TRANSPARENT = 0x00000020
WS_EX_TOOLWINDOW = 0x00000080  # 不在任务栏和 Alt-Tab 中显示


class DrawingOverlay(QWidget):
    """板书模式全屏绘制层。

    设计要点：
    - 全屏覆盖，但**不拦截**鼠标事件（纯 Win32 WS_EX_TRANSPARENT）。
    - 不依赖 Qt.WindowTransparentForInput（行为不稳定）。
    - paintEvent 中始终安全结束 QPainter，避免绘制状态异常。
    """

    undo_changed = pyqtSignal(bool)

    MAX_GAP_FRAMES = 8
    # MAX_BRIDGE_DISTANCE 在 __init__ 中根据 DPI 动态计算
    MAX_BRIDGE_DISTANCE = 120
    SMOOTH_ALPHA_SLOW = 0.30
    SMOOTH_ALPHA_FAST = 0.85
    SPEED_LOW_THRESHOLD = 8.0
    SPEED_HIGH_THRESHOLD = 60.0

    def __init__(self, parent=None, pen_width=15):
        super().__init__(parent)
        self.pen_width = pen_width
        self.pen_color = QColor(255, 0, 0, 220)
        # 根据 DPI 缩放桥接距离，避免 4K 屏断笔
        dpr = QApplication.primaryScreen().devicePixelRatio() if QApplication.primaryScreen() else 1.0
        self.max_bridge_distance = int(self.MAX_BRIDGE_DISTANCE * max(dpr, 1.0))
        self.last_point = None
        self.canvas = None
        self.smooth_point = None
        self.raw_point = None
        self.idle_count = 0
        self.truly_lifted = False
        self.cursor_pos = None
        self._recent_points: deque = deque(maxlen=2)
        self._undo_stack: list = []
        self._max_undo = 30
        self._click_through_applied = False
        self._shape_correction_enabled = False
        self._current_stroke: list = []

        # 只用最基础的顶层窗口标志；点击穿透完全由 _make_click_through() 控制
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.hide()

    def _make_click_through(self):
        """通过 Win32 API 设置点击穿透，并加入 WS_EX_TOOLWINDOW 防止出现在任务栏。"""
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
            import logging
            logging.error("DrawingOverlay _make_click_through failed: %s", e)

    def show_fullscreen(self):
        """显示全屏绘制层。不用 Qt 的 showFullScreen()，避免它重置窗口标志。
        每次显示前重置 _click_through_applied，防止隐藏后再显示时 Win32 样式丢失。"""
        self._click_through_applied = False  # 强制重新应用 Win32 样式
        screen = QApplication.primaryScreen().geometry()
        self.setGeometry(screen)
        self.show()
        self.raise_()
        self._make_click_through()
        self._ensure_canvas()

    def _ensure_canvas(self):
        if self.canvas is None or self.canvas.size() != self.size():
            old = self.canvas
            self.canvas = QPixmap(self.size())
            self.canvas.fill(Qt.GlobalColor.transparent)
            if old is not None:
                painter = QPainter(self.canvas)
                painter.drawPixmap(0, 0, old)
                painter.end()

    def set_pen_width(self, pen_width):
        self.pen_width = pen_width

    def set_pen_color(self, color: QColor):
        self.pen_color = QColor(color.red(), color.green(), color.blue(), 220)

    def _push_undo_snapshot(self):
        if self.canvas is None:
            return
        snapshot = self.canvas.copy()
        self._undo_stack.append(snapshot)
        if len(self._undo_stack) > self._max_undo:
            self._undo_stack.pop(0)
        self.undo_changed.emit(True)

    def undo(self):
        if not self._undo_stack:
            return
        snapshot = self._undo_stack.pop()
        self._ensure_canvas()
        self.canvas.fill(Qt.GlobalColor.transparent)
        painter = QPainter(self.canvas)
        painter.drawPixmap(0, 0, snapshot)
        painter.end()
        self.last_point = None
        self.smooth_point = None
        self.raw_point = None
        self.update()
        self.undo_changed.emit(bool(self._undo_stack))

    def _speed_alpha(self, raw):
        if self.raw_point is None:
            self.raw_point = raw
            return self.SMOOTH_ALPHA_SLOW
        dx = raw.x() - self.raw_point.x()
        dy = raw.y() - self.raw_point.y()
        speed = math.sqrt(dx * dx + dy * dy)
        self.raw_point = raw
        t = (speed - self.SPEED_LOW_THRESHOLD) / (self.SPEED_HIGH_THRESHOLD - self.SPEED_LOW_THRESHOLD)
        t = max(0.0, min(1.0, t))
        return self.SMOOTH_ALPHA_SLOW + (self.SMOOTH_ALPHA_FAST - self.SMOOTH_ALPHA_SLOW) * t

    def _smooth(self, raw):
        if self.smooth_point is None:
            self.smooth_point = raw
        else:
            a = self._speed_alpha(raw)
            self.smooth_point = QPoint(
                int(self.smooth_point.x() * (1 - a) + raw.x() * a),
                int(self.smooth_point.y() * (1 - a) + raw.y() * a),
            )
        return self.smooth_point

    def _bridge_too_far(self, p1, p2):
        dx = p1.x() - p2.x()
        dy = p1.y() - p2.y()
        return dx * dx + dy * dy > self.max_bridge_distance * self.max_bridge_distance

    def update_cursor(self, x, y):
        new_pos = QPoint(int(x), int(y))
        if self.cursor_pos is None or (new_pos - self.cursor_pos).manhattanLength() > 1:
            self.cursor_pos = new_pos
            self.update()

    def hide_cursor(self):
        self.cursor_pos = None
        self.update()

    def draw_to(self, x, y):
        self._ensure_canvas()
        self.idle_count = 0
        self.truly_lifted = False

        raw = QPoint(int(x), int(y))
        point = self._smooth(raw)

        if self.last_point is None:
            self._push_undo_snapshot()
            self.last_point = point
            self._recent_points.clear()
            self._recent_points.append(point)
            self._current_stroke = [(point.x(), point.y())]
            self.update_cursor(x, y)
            return

        if self._bridge_too_far(self.last_point, point):
            self._push_undo_snapshot()
            self.last_point = point
            self._recent_points.clear()
            self._recent_points.append(point)
            self._current_stroke = [(point.x(), point.y())]
            self.update_cursor(x, y)
            return

        painter = QPainter(self.canvas)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(
            QPen(
                self.pen_color,
                self.pen_width,
                Qt.PenStyle.SolidLine,
                Qt.PenCapStyle.RoundCap,
            )
        )
        painter.drawLine(self.last_point, point)
        painter.end()
        self.last_point = point
        self._recent_points.append(point)
        self._current_stroke.append((point.x(), point.y()))
        self.update_cursor(x, y)
        self.update()

    def tick_idle(self):
        if self.last_point is not None:
            self.idle_count += 1
            if self.idle_count >= self.MAX_GAP_FRAMES:
                self.lift_pen()

    def lift_pen(self):
        self._try_correct_shape()
        self.last_point = None
        self.smooth_point = None
        self.raw_point = None
        self.idle_count = 0

    def force_lift_pen(self):
        self._try_correct_shape()
        self.last_point = None
        self.smooth_point = None
        self.raw_point = None
        self.idle_count = 0
        self.truly_lifted = True

    def set_shape_correction_enabled(self, enabled: bool):
        self._shape_correction_enabled = enabled

    def toggle_shape_correction(self):
        self._shape_correction_enabled = not self._shape_correction_enabled
        return self._shape_correction_enabled

    def is_shape_correction_enabled(self):
        return self._shape_correction_enabled

    def _try_correct_shape(self):
        if not self._shape_correction_enabled:
            self._current_stroke = []
            return
        stroke = self._current_stroke
        self._current_stroke = []
        if len(stroke) < 4:
            return

        shape, corrected = recognize_and_correct(stroke)
        if shape is None or corrected == stroke:
            return

        if not self._undo_stack:
            return

        snapshot = self._undo_stack.pop()
        self._ensure_canvas()
        self.canvas.fill(Qt.GlobalColor.transparent)
        painter = QPainter(self.canvas)
        painter.drawPixmap(0, 0, snapshot)
        painter.end()

        self._draw_shape(shape, corrected)
        self.update()

    def _draw_shape(self, shape, points):
        if len(points) < 2:
            return
        painter = QPainter(self.canvas)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(
            self.pen_color,
            self.pen_width,
            Qt.PenStyle.SolidLine,
            Qt.PenCapStyle.RoundCap,
        )
        painter.setPen(pen)

        if shape == "LINE":
            p1 = QPointF(points[0][0], points[0][1])
            p2 = QPointF(points[1][0], points[1][1])
            painter.drawLine(p1, p2)
        elif shape == "TRIANGLE":
            for i in range(len(points) - 1):
                p1 = QPointF(points[i][0], points[i][1])
                p2 = QPointF(points[i + 1][0], points[i + 1][1])
                painter.drawLine(p1, p2)
        elif shape == "ELLIPSE":
            pts = [QPointF(p[0], p[1]) for p in points]
            for i in range(len(pts) - 1):
                painter.drawLine(pts[i], pts[i + 1])
        elif shape == "RECTANGLE":
            for i in range(len(points) - 1):
                p1 = QPointF(points[i][0], points[i][1])
                p2 = QPointF(points[i + 1][0], points[i + 1][1])
                painter.drawLine(p1, p2)
        elif shape == "SMOOTH":
            pts = [QPointF(p[0], p[1]) for p in points]
            for i in range(len(pts) - 1):
                painter.drawLine(pts[i], pts[i + 1])

        painter.end()

    def clear_canvas(self):
        if self.canvas is not None:
            self._push_undo_snapshot()
        self._ensure_canvas()
        self.canvas.fill(Qt.GlobalColor.transparent)
        self.last_point = None
        self.smooth_point = None
        self.raw_point = None
        self.idle_count = 0
        self.truly_lifted = True
        self._current_stroke = []
        self.update()

    def paintEvent(self, event):
        """绘制画布和光标指示器。

        关键修复：即使 canvas 为 None，只要 cursor_pos 存在也绘制光标，
        并且始终显式 end() QPainter。
        """
        painter = QPainter(self)
        if self.canvas is not None:
            painter.drawPixmap(0, 0, self.canvas)
        if self.cursor_pos is not None:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            r = max(self.pen_width + 4, 10)
            cursor_color = QColor(self.pen_color.red(), self.pen_color.green(), self.pen_color.blue(), 160)
            painter.setPen(QPen(cursor_color, 2, Qt.PenStyle.SolidLine))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(self.cursor_pos, r, r)
            inner_color = QColor(self.pen_color.red(), self.pen_color.green(), self.pen_color.blue(), 90)
            painter.setBrush(inner_color)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(self.cursor_pos, r // 2, r // 2)
        painter.end()
