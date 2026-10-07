"""
Unit tests for DrawingOverlay caption handwriting font.

板书听写字幕的手写体（板书感）：_caption_layout 构造 QFont 时必须设置
CAPTION_FONT_FAMILIES 回退栈（华文行楷 → 楷体 → 华文楷体），保证实时
字幕和最终落屏文字（draw_text）同字体渲染。

Import strategy mirrors test_mouse_cursor_overlay.py: mock PyQt6/ctypes at
module level, exec the real module, and drive methods on a bare instance.
"""
import sys
import types
import unittest
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from unittest.mock import MagicMock, patch

# Mock ctypes for Windows API before importing
mock_ctypes = types.ModuleType('ctypes')
mock_windll = types.ModuleType('ctypes.windll')
mock_user32 = MagicMock()
mock_windll.user32 = mock_user32
mock_ctypes.windll = mock_windll

# Minimal QWidget stand-in so DrawingOverlay can be instantiated without Qt
class FakeQWidget:
    def setWindowFlags(self, *a): pass
    def setAttribute(self, *a): pass
    def hide(self): pass
    def show(self): pass
    def raise_(self): pass
    def setGeometry(self, *a): pass
    def winId(self): return 12345
    def isVisible(self): return False
    def update(self): pass
    def size(self):
        from types import SimpleNamespace

        return SimpleNamespace(width=lambda: 0, height=lambda: 0)


mock_qt_widgets = types.ModuleType('PyQt6.QtWidgets')
mock_qt_app = MagicMock()
mock_screen = MagicMock()
mock_screen.devicePixelRatio.return_value = 1.0
mock_qt_app.primaryScreen.return_value = mock_screen
mock_qt_widgets.QApplication = mock_qt_app
mock_qt_widgets.QWidget = FakeQWidget

# Capture calls made on QFont instances so tests can assert without real Qt
_font_calls = []
class FakeQFont:
    def __init__(self):
        self.families = None
        self.pixel_size = None
        self.bold = None
    def setFamilies(self, fams):
        self.families = list(fams)
        _font_calls.append(self)
    def setPixelSize(self, sz):
        self.pixel_size = sz
    def setBold(self, bold):
        self.bold = bold

mock_qt_gui = types.ModuleType('PyQt6.QtGui')
mock_qt_gui.QColor = MagicMock()
mock_qt_gui.QFont = FakeQFont

class FakeQFontMetrics:
    """Numeric metrics so layout math works without real Qt."""
    def __init__(self, font):
        self._font = font
    def height(self):
        return 40
    def ascent(self):
        return 30
    def horizontalAdvance(self, text):
        return 20 * len(str(text))

mock_qt_gui.QFontMetrics = FakeQFontMetrics
mock_qt_gui.QPainter = MagicMock()
mock_qt_gui.QPen = MagicMock()
mock_qt_gui.QPixmap = MagicMock()

mock_qt_core = MagicMock()
# Qt namespace values used at class definition time must be real enums-ish
class _FakeQtNamespace:
    class WindowType:
        FramelessWindowHint = 1
        WindowStaysOnTopHint = 2
        Tool = 4
    class WidgetAttribute:
        WA_TranslucentBackground = 1
        WA_TransparentForMouseEvents = 2
        WA_ShowWithoutActivating = 3
    class GlobalColor:
        transparent = 0
    class PenStyle:
        SolidLine = 1
        NoPen = 0
    class PenCapStyle:
        RoundCap = 1
    class BrushStyle:
        NoBrush = 0
mock_qt_core.Qt = _FakeQtNamespace

# shape_recognizer is a pure-math stdlib module; let the real import succeed by
# putting app/ on sys.path inside the patch context.
_app_dir = str(Path(__file__).resolve().parents[1] / "app")

_module_mocks = {
    "ctypes": mock_ctypes,
    "PyQt6": types.ModuleType("PyQt6"),
    "PyQt6.QtCore": mock_qt_core,
    "PyQt6.QtGui": mock_qt_gui,
    "PyQt6.QtWidgets": mock_qt_widgets,
}

_module_path = Path(__file__).resolve().parents[1] / "app" / "drawing_overlay.py"
_spec = spec_from_file_location("drawing_overlay_under_test", _module_path)
_overlay_module = module_from_spec(_spec)
with patch.dict(sys.modules, _module_mocks):
    sys.path.insert(0, _app_dir)
    try:
        _spec.loader.exec_module(_overlay_module)
    finally:
        sys.path.remove(_app_dir)

DrawingOverlay = _overlay_module.DrawingOverlay


def _make_overlay():
    """Instantiate DrawingOverlay without a real QApplication (attrs only)."""
    overlay = DrawingOverlay.__new__(DrawingOverlay)
    overlay._base_pen_width = 15
    overlay._pen_scale = 1.0
    overlay._pen_scale_enabled = True
    overlay.pen_color = MagicMock()
    overlay.max_bridge_distance = 120
    overlay.last_point = None
    overlay.canvas = None
    overlay.smooth_point = None
    overlay.raw_point = None
    overlay.idle_count = 0
    overlay.truly_lifted = False
    overlay.cursor_pos = None
    overlay._motion_resp = 0.5
    overlay._recent_points = None
    overlay._undo_stack = []
    overlay._max_undo = 30
    overlay._click_through_applied = False
    overlay._system_cursor_hidden = False
    overlay._shape_correction_enabled = True
    overlay._current_stroke = []
    overlay._dictation_caption = ""
    overlay._caption_full_emitted = False
    return overlay


class TestCaptionHandwritingFont(unittest.TestCase):
    def _layout_font(self, overlay, text="智慧教室板书"):
        """Run _caption_layout with fake geometry; return the FakeQFont it built."""
        overlay.canvas = None  # canvas None → uses self.width()/height()
        overlay.width = lambda: 1920
        overlay.height = lambda: 1080
        _font_calls.clear()
        layout = overlay._caption_layout(text)
        self.assertIsNotNone(layout, "_caption_layout 应返回非 None")
        return layout["font"]

    def test_caption_font_uses_handwriting_families(self):
        """字幕字体必须设置手写体回退栈（行楷→楷体→华文楷体）。"""
        font = self._layout_font(_make_overlay())
        self.assertEqual(
            font.families, list(DrawingOverlay.CAPTION_FONT_FAMILIES)
        )

    def test_handwriting_stack_order(self):
        """字体栈首选方正硬笔楷书（项目自带），楷体兜底 — 顺序不可乱。"""
        fams = list(DrawingOverlay.CAPTION_FONT_FAMILIES)
        self.assertEqual(fams[0], "FZYingBiKaiShu-S15S")
        self.assertIn("KaiTi", fams)

    def test_font_families_is_a_class_constant(self):
        """回退栈是类常量，实时字幕与 draw_text 落屏共用同一来源。"""
        self.assertIsInstance(DrawingOverlay.CAPTION_FONT_FAMILIES, (list, tuple))
        self.assertTrue(all(isinstance(f, str) for f in DrawingOverlay.CAPTION_FONT_FAMILIES))

    def test_bundled_font_file_exists(self):
        """项目自带字体文件必须存在（板书手写体随包分发的前提）。"""
        font_path = (
            Path(__file__).resolve().parents[1]
            / "app" / "assets" / "fonts" / "FZYingBiKaiShu-S15S.ttf"
        )
        self.assertTrue(font_path.is_file(), f"缺少自带字体: {font_path}")


if __name__ == "__main__":
    unittest.main()
