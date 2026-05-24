import ctypes
import math
from collections import deque

from PyQt6.QtCore import QPoint, QPointF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPen, QPixmap
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
    # 实时字幕写满屏幕时触发，由 main_ui 接收并自动停止听写
    caption_full = pyqtSignal()

    MAX_GAP_FRAMES = 8
    # MAX_BRIDGE_DISTANCE 在 __init__ 中根据 DPI 动态计算
    MAX_BRIDGE_DISTANCE = 120
    SMOOTH_ALPHA_SLOW = 0.30
    SMOOTH_ALPHA_FAST = 0.85
    SPEED_LOW_THRESHOLD = 8.0
    SPEED_HIGH_THRESHOLD = 60.0

    # 笔粗距离自适应参数
    MIN_PEN_WIDTH = 4         # 极远时下限
    MAX_PEN_WIDTH = 80        # 极近时上限
    REFERENCE_HAND_SIZE = 150.0  # sqrt(bbox_area) 在"舒适书写距离"下的参考值

    def __init__(self, parent=None, pen_width=15):
        super().__init__(parent)
        # 笔粗：_base 由用户/工具栏设置，_scale 由 DrawMode 每帧根据手 bbox 调整
        # pen_width 属性返回实际生效值（base × scale，钳位）
        self._base_pen_width = pen_width
        self._pen_scale = 1.0
        self._pen_scale_enabled = True
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
        self._shape_correction_enabled = True
        self._current_stroke: list = []
        # 听写实时字幕（partial ASR）— 不写到 canvas，由 paintEvent 动态绘制
        # 这样下一次 partial 来时直接替换，避免叠加；"结束板书"时清掉并由
        # draw_text 把最终文本固化到 canvas。
        self._dictation_caption: str = ""
        # 字幕写满去重标志：溢出时只 emit 一次 caption_full，
        # 避免每帧 repaint 都 emit；text 缩短/清空时重置。
        self._caption_full_emitted = False

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
        size = self.size()
        if size.width() <= 0 or size.height() <= 0:
            return  # 窗口尚未显示时 size 可能为 (0,0)，跳过避免 null pixmap
        if self.canvas is None or self.canvas.size() != size:
            old = self.canvas
            self.canvas = QPixmap(size)
            self.canvas.fill(Qt.GlobalColor.transparent)
            if old is not None:
                painter = QPainter(self.canvas)
                painter.drawPixmap(0, 0, old)
                painter.end()

    @property
    def pen_width(self):
        """实际生效的笔粗 = base × scale，钳在 [MIN, MAX]。"""
        if self._pen_scale_enabled:
            effective = self._base_pen_width * self._pen_scale
        else:
            effective = self._base_pen_width
        return int(max(self.MIN_PEN_WIDTH, min(effective, self.MAX_PEN_WIDTH)))

    @pen_width.setter
    def pen_width(self, value):
        """直接赋值视为设置 base（工具栏来的）。"""
        self._base_pen_width = value

    def set_pen_width(self, pen_width):
        """用户工具栏设置的"基准"笔粗，距离系数会在它的基础上调整。"""
        self._base_pen_width = pen_width

    def set_pen_scale(self, scale):
        """DrawMode 每帧调用，传入 sqrt(bbox)/REFERENCE 的比值。EMA 平滑避免抖动。"""
        self._pen_scale = 0.7 * self._pen_scale + 0.3 * float(scale)

    def set_pen_auto_scale(self, enabled):
        """启停笔粗距离自适应。关闭后回归 base 值不变。"""
        self._pen_scale_enabled = bool(enabled)

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

    def set_dictation_caption(self, text):
        """设置实时听写字幕（partial ASR 结果）。"""
        new_text = text or ""
        if new_text == self._dictation_caption:
            return
        # text 变短了说明被截断/重置，允许下次溢出再次 emit
        if len(new_text) < len(self._dictation_caption):
            self._caption_full_emitted = False
        self._dictation_caption = new_text
        self.update()

    def clear_dictation_caption(self):
        """清空实时字幕。听写结束、把最终文本固化到画布前调用。"""
        self._caption_full_emitted = False
        if not self._dictation_caption:
            return
        self._dictation_caption = ""
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

        # 安全策略：先绘制到临时 pixmap，成功后再替换画布
        snapshot = self._undo_stack.pop()
        self._ensure_canvas()
        if self.canvas is None or self.canvas.isNull():
            return
        temp = QPixmap(self.canvas.size())
        temp.fill(Qt.GlobalColor.transparent)
        painter = QPainter(temp)
        painter.drawPixmap(0, 0, snapshot)
        painter.end()

        # 在临时 pixmap 上绘制修正后的形状
        old_canvas = self.canvas
        self.canvas = temp
        self._draw_shape(shape, corrected)

        # 绘制成功，保存 undo 用的旧画布
        self._undo_stack.append(old_canvas)
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

    # 听写字幕统一样式参数 — 实时和最终用同一套
    # 字幕铺满屏幕：从左上 outer_pad 开始书写、自动换行直到右下 outer_pad
    # 超出可容纳行数时触发 caption_full 信号，由 main_ui 自动停止听写
    CAPTION_FONT_RATIO = 0.05      # 字号 = 屏高 × 该比例
    CAPTION_OUTER_PAD_RATIO = 0.04 # 屏幕到字幕框的外边距比例
    CAPTION_INNER_PAD_X = 28       # 字幕框内左右内边距
    CAPTION_INNER_PAD_Y = 18       # 字幕框内上下内边距
    CAPTION_BG_ALPHA = 180         # 半透明黑底 alpha
    CAPTION_RADIUS = 16
    CAPTION_FONT_MIN = 28
    CAPTION_FONT_MAX = 56

    def _caption_layout(self, text):
        """计算字幕布局：黑底随实际文字宽×高伸缩，避免短文本时遮挡整个屏幕。

        换行/溢出判据仍以"屏幕可容纳区域"为准（max_line_w / max_lines），
        但实际绘制的盒子按当前行内容收缩。锚定左上角，从屏幕外边距处起绘。
        返回 dict（None 表示尺寸异常或文本为空）。
        """
        canvas_w = self.width() if self.canvas is None else max(
            self.width(), self.canvas.width())
        canvas_h = self.height() if self.canvas is None else max(
            self.height(), self.canvas.height())
        if canvas_w <= 0 or canvas_h <= 0:
            return None

        font_pt = max(self.CAPTION_FONT_MIN,
                      min(int(canvas_h * self.CAPTION_FONT_RATIO),
                          self.CAPTION_FONT_MAX))
        font = QFont()
        font.setPixelSize(font_pt)
        font.setBold(True)
        metrics = QFontMetrics(font)
        line_height = metrics.height()

        outer_pad = max(32, int(min(canvas_w, canvas_h)
                                * self.CAPTION_OUTER_PAD_RATIO))
        ipx = self.CAPTION_INNER_PAD_X
        ipy = self.CAPTION_INNER_PAD_Y

        max_box_w = canvas_w - 2 * outer_pad
        max_box_h = canvas_h - 2 * outer_pad
        if max_box_w <= 2 * ipx or max_box_h <= 2 * ipy:
            return None

        max_line_w = max_box_w - 2 * ipx
        max_lines = max(1, (max_box_h - 2 * ipy) // line_height)

        lines = self._wrap_text(text, metrics, max_line_w)
        overflow = len(lines) > max_lines
        if overflow:
            lines = lines[:max_lines]

        # 过滤掉纯空白行用来量尺寸，但渲染时保留以维持换行
        visible_lines = [ln for ln in lines if ln]
        if not visible_lines:
            return None

        text_w = max(metrics.horizontalAdvance(ln) for ln in visible_lines)
        box_w = min(max_box_w, text_w + 2 * ipx)
        box_h = min(max_box_h, len(lines) * line_height + 2 * ipy)
        box_x = outer_pad
        box_y = outer_pad

        return {
            "font": font,
            "metrics": metrics,
            "line_height": line_height,
            "box_x": box_x, "box_y": box_y, "box_w": box_w, "box_h": box_h,
            "text_x": box_x + ipx,
            "text_baseline": box_y + ipy + metrics.ascent(),
            "lines": lines,
            "overflow": overflow,
        }

    def _draw_caption_to(self, painter, layout):
        """把字幕画到给定 painter 上。实时字幕用 widget painter，
        最终字幕用 canvas painter — 画法完全一致。"""
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        painter.setBrush(QColor(0, 0, 0, self.CAPTION_BG_ALPHA))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(layout["box_x"], layout["box_y"],
                                layout["box_w"], layout["box_h"],
                                self.CAPTION_RADIUS, self.CAPTION_RADIUS)
        painter.setFont(layout["font"])
        painter.setPen(QColor(255, 255, 255, 245))
        text_x = layout["text_x"]
        baseline = layout["text_baseline"]
        line_height = layout["line_height"]
        for i, line in enumerate(layout["lines"]):
            painter.drawText(text_x, baseline + i * line_height, line)

    def draw_text(self, text, x=None, y=None):
        """听写结束后把最终字幕写到画布上（持久保留，可撤销/清屏）。

        样式与实时字幕完全一致：屏幕中央、半透明黑底圆角、白字无描边，
        只是字号更大一些便于阅读。x/y 参数保留以兼容旧调用，但忽略不用 —
        统一居中显示，避免和实时字幕位置跳变。
        """
        if not text:
            return
        self._ensure_canvas()
        if self.canvas is None:
            return

        layout = self._caption_layout(text)
        if layout is None:
            return

        self._push_undo_snapshot()
        painter = QPainter(self.canvas)
        try:
            self._draw_caption_to(painter, layout)
        finally:
            painter.end()

        # 强制抬笔，避免下次手势书写连到文字附近
        self.last_point = None
        self.smooth_point = None
        self.raw_point = None
        self._current_stroke = []
        self.update()

    @staticmethod
    def _wrap_text(text, metrics, max_w):
        """简单贪心换行：累加字符宽度，超出 max_w 就换行。"""
        if max_w <= 0:
            return [text]
        lines = []
        # 用户输入的 \n 单独保留
        for paragraph in text.split("\n"):
            if not paragraph:
                lines.append("")
                continue
            cur = ""
            cur_w = 0
            for ch in paragraph:
                cw = metrics.horizontalAdvance(ch)
                if cur_w + cw > max_w and cur:
                    lines.append(cur)
                    cur = ch
                    cur_w = cw
                else:
                    cur += ch
                    cur_w += cw
            if cur:
                lines.append(cur)
        return lines

    def paintEvent(self, event):
        """绘制画布、光标指示器和实时字幕。

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
        if self._dictation_caption:
            self._paint_dictation_caption(painter)
        painter.end()

    def _paint_dictation_caption(self, painter):
        """从屏幕左上铺到右下绘制实时字幕：半透明黑底圆角 + 白字无描边。

        每帧重画，不写到 canvas，partial 更新时直接替换。样式与
        draw_text 完全一致，结束时无视觉跳变。
        溢出（写到右下角写不下）时 emit caption_full，main_ui 自动停止听写。
        """
        layout = self._caption_layout(self._dictation_caption)
        if layout is None:
            return
        self._draw_caption_to(painter, layout)

        # 写满去重 emit：溢出时只发一次，text 缩短后会重置标志
        if layout["overflow"]:
            if not self._caption_full_emitted:
                self._caption_full_emitted = True
                self.caption_full.emit()
        else:
            self._caption_full_emitted = False
