from PyQt6.QtCore import QPoint, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QCursor
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

TOOLBAR_STYLE = """
QWidget#DrawToolbar {
    background-color: rgba(30, 30, 30, 200);
    border-radius: 10px;
}
"""

HANDLE_STYLE = """
QLabel#DragHandle {
    background-color: rgba(255, 255, 255, 25);
    border: 1px solid rgba(255, 255, 255, 40);
    border-radius: 4px;
}
QLabel#DragHandle:hover {
    background-color: rgba(255, 255, 255, 50);
    border: 1px solid rgba(255, 255, 255, 80);
}
"""


class ColorButton(QPushButton):
    def __init__(self, color: QColor, parent=None):
        super().__init__(parent)
        self._color = color
        self.setCheckable(True)
        self.setStyleSheet(
            f"QPushButton {{ background-color: {color.name()}; "
            f"border: none; border-radius: 12px; "
            f"min-width: 24px; max-width: 24px; "
            f"min-height: 24px; max-height: 24px; }}"
            f"QPushButton:hover {{ border: 2px solid rgba(255,255,255,180); background-color: {color.name()}; }}"
            f"QPushButton:checked {{ border: 2px solid white; background-color: {color.name()}; }}"
        )

    @property
    def draw_color(self) -> QColor:
        return self._color


class DrawToolbar(QWidget):
    color_changed = pyqtSignal(QColor)
    pen_width_changed = pyqtSignal(int)
    undo_requested = pyqtSignal()
    clear_requested = pyqtSignal()
    shape_correction_toggled = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("DrawToolbar")
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self._drag_pos = None
        self._build_ui()

    def _build_ui(self):
        self.setStyleSheet(TOOLBAR_STYLE)
        main = QVBoxLayout(self)
        main.setContentsMargins(8, 6, 8, 6)
        main.setSpacing(4)

        self._handle = QLabel("\u2261 拖拽移动")
        self._handle.setObjectName("DragHandle")
        self._handle.setFixedHeight(22)
        self._handle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._handle.setStyleSheet(HANDLE_STYLE)
        self._handle.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
        main.addWidget(self._handle)

        row1 = QHBoxLayout()
        row1.setSpacing(6)
        lbl = QLabel("颜色")
        lbl.setStyleSheet("color: rgba(255,255,255,180); font-size: 10px; border: none; background: transparent;")
        row1.addWidget(lbl)

        self._color_buttons: list[ColorButton] = []
        for c in [
            QColor(255, 0, 0),
            QColor(0, 120, 255),
            QColor(0, 180, 0),
            QColor(255, 160, 0),
            QColor(255, 255, 255),
            QColor(0, 0, 0),
        ]:
            btn = ColorButton(c)
            btn.clicked.connect(self._on_color_clicked)
            self._color_buttons.append(btn)
            row1.addWidget(btn)
        self._color_buttons[0].setChecked(True)
        main.addLayout(row1)

        row2 = QHBoxLayout()
        row2.setSpacing(6)
        lbl2 = QLabel("粗细")
        lbl2.setStyleSheet("color: rgba(255,255,255,180); font-size: 10px; border: none; background: transparent;")
        row2.addWidget(lbl2)

        self._width_slider = QSlider(Qt.Orientation.Horizontal)
        self._width_slider.setRange(2, 20)
        self._width_slider.setValue(15)
        self._width_slider.setFixedWidth(120)
        self._width_slider.valueChanged.connect(lambda v: self.pen_width_changed.emit(v))
        row2.addWidget(self._width_slider)

        self._width_label = QLabel("15")
        self._width_label.setFixedWidth(18)
        self._width_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._width_label.setStyleSheet("color: white; font-size: 11px; border: none; background: transparent;")
        self._width_slider.valueChanged.connect(lambda v: self._width_label.setText(str(v)))
        row2.addWidget(self._width_label)
        main.addLayout(row2)

        row3 = QHBoxLayout()
        row3.setSpacing(8)

        self._btn_undo = QPushButton("\u21B6")
        self._btn_undo.setStyleSheet(
            "QPushButton { background-color: rgba(255,255,255,40); color: white; "
            "min-width:28px; max-width:28px; min-height:24px; max-height:24px; "
            "border:none; border-radius:4px; font-size:16px; }"
            "QPushButton:hover { background-color: rgba(255,255,255,80); }"
            "QPushButton:disabled { color: rgba(255,255,255,60); }"
        )
        self._btn_undo.setEnabled(False)
        self._btn_undo.clicked.connect(self.undo_requested.emit)
        row3.addWidget(self._btn_undo)

        self._btn_clear = QPushButton("\u2716")
        self._btn_clear.setStyleSheet(
            "QPushButton { background-color: rgba(255,80,80,60); color: white; "
            "min-width:28px; max-width:28px; min-height:24px; max-height:24px; "
            "border:none; border-radius:4px; font-size:14px; }"
            "QPushButton:hover { background-color: rgba(255,80,80,120); }"
        )
        self._btn_clear.clicked.connect(self.clear_requested.emit)
        row3.addWidget(self._btn_clear)

        row3.addStretch()
        main.addLayout(row3)

        row4 = QHBoxLayout()
        row4.setSpacing(8)

        self._btn_shape = QPushButton("\u2B21")
        self._btn_shape.setToolTip("图形修正：自动修正直线/圆/矩形")
        self._btn_shape.setCheckable(True)
        self._btn_shape.setChecked(False)
        self._btn_shape.setStyleSheet(
            "QPushButton { background-color: rgba(255,255,255,40); color: white; "
            "min-width:28px; max-width:28px; min-height:24px; max-height:24px; "
            "border:none; border-radius:4px; font-size:14px; }"
            "QPushButton:hover { background-color: rgba(255,255,255,80); }"
            "QPushButton:checked { background-color: rgba(0,200,100,100); }"
        )
        self._btn_shape.toggled.connect(self.shape_correction_toggled.emit)
        row4.addWidget(self._btn_shape)

        self._shape_label = QLabel("图形修正")
        self._shape_label.setStyleSheet("color: rgba(255,255,255,140); font-size: 10px; border: none; background: transparent;")
        row4.addWidget(self._shape_label)

        row4.addStretch()
        main.addLayout(row4)

        self.setFixedWidth(200)

    def _on_color_clicked(self):
        btn = self.sender()
        if not isinstance(btn, ColorButton):
            return
        for b in self._color_buttons:
            b.setChecked(b is btn)
        self.color_changed.emit(btn.draw_color)

    def set_undo_enabled(self, enabled: bool):
        self._btn_undo.setEnabled(enabled)

    def set_pen_width(self, width: int):
        self._width_slider.blockSignals(True)
        self._width_slider.setValue(width)
        self._width_label.setText(str(width))
        self._width_slider.blockSignals(False)

    def set_shape_correction(self, enabled: bool):
        self._btn_shape.blockSignals(True)
        self._btn_shape.setChecked(enabled)
        self._btn_shape.blockSignals(False)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            child = self.childAt(event.pos())
            if child is self._handle:
                self._drag_pos = event.globalPosition().toPoint() - self.pos()
                self._handle.setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))
                event.accept()
                return
            if isinstance(child, (QPushButton, QSlider)):
                super().mousePressEvent(event)
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_pos is not None:
            self.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._drag_pos is not None:
            self._handle.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
        self._drag_pos = None
        super().mouseReleaseEvent(event)
