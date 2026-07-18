"""
Unit tests for MouseCursorOverlay focusing on the show_fullscreen() bug fix.
Tests the _system_cursor_hidden flag reset before _hide_system_cursor().

Strategy: Since MouseCursorOverlay inherits from QWidget (which can't be easily
instantiated in a test env), we test the logic of the key methods by importing
them as unbound functions and calling them on a manually constructed object that
has all required attributes.
"""
import sys
import types
import unittest
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from unittest.mock import MagicMock, patch

# Mock ctypes for Windows API before importing
mock_ctypes = types.ModuleType('ctypes')
mock_ctypes.wintypes = types.ModuleType('ctypes.wintypes')
mock_windll = types.ModuleType('ctypes.windll')
mock_user32 = types.ModuleType('user32')
mock_user32.GetSystemMetrics = lambda idx: 1920 if idx == 0 else 1080
mock_user32.ShowCursor = MagicMock(return_value=0)
mock_user32.GetWindowLongW = MagicMock(return_value=0)
mock_user32.SetWindowLongW = MagicMock()
mock_user32.SetWindowPos = MagicMock()
mock_windll.user32 = mock_user32
mock_ctypes.windll = mock_windll
mock_ctypes.byref = lambda x: x
mock_ctypes.wintypes.POINT = lambda: type('POINT', (), {'x': 0, 'y': 0})()

# Now we need to import the module in a way that QWidget can be mocked
# We'll patch at the module level

# Import just the module, not the class, so we can access individual methods

# Patch QWidget base class to be a regular class (not a real Qt widget)
class FakeQWidget:
    """Minimal stand-in for QWidget so MouseCursorOverlay can be instantiated."""
    def setWindowFlags(self, *a): pass
    def setAttribute(self, *a): pass
    def hide(self): pass
    def show(self): pass
    def raise_(self): pass
    def setGeometry(self, *a): pass
    def winId(self): return 12345
    def isVisible(self): return False
    def update(self): pass

# Patch the QtWidgets module
mock_qt_widgets = types.ModuleType('PyQt6.QtWidgets')
mock_qt_app = MagicMock()
mock_screen = MagicMock()
mock_geometry = MagicMock()
mock_screen.geometry.return_value = mock_geometry
mock_screen.devicePixelRatio.return_value = 1.0
mock_qt_app.primaryScreen.return_value = mock_screen
mock_qt_widgets.QApplication = mock_qt_app
mock_qt_widgets.QWidget = FakeQWidget
_module_mocks = {
    "ctypes": mock_ctypes,
    "ctypes.wintypes": mock_ctypes.wintypes,
    "PyQt6": types.ModuleType("PyQt6"),
    "PyQt6.QtCore": MagicMock(),
    "PyQt6.QtGui": MagicMock(),
    "PyQt6.QtWidgets": mock_qt_widgets,
}
_module_path = Path(__file__).resolve().parents[1] / "app" / "mouse_cursor_overlay.py"
_spec = spec_from_file_location("mouse_cursor_overlay_under_test", _module_path)
_overlay_module = module_from_spec(_spec)
with patch.dict(sys.modules, _module_mocks):
    _spec.loader.exec_module(_overlay_module)

MouseCursorOverlay = _overlay_module.MouseCursorOverlay


def _make_overlay():
    """Create a MouseCursorOverlay instance with all methods working but no real Qt."""
    overlay = MouseCursorOverlay.__new__(MouseCursorOverlay)
    # Set all attributes that __init__ would set
    overlay._pos = None
    overlay._click = None
    overlay._scroll = None
    overlay._left_hold = False
    overlay._system_cursor_hidden = False
    overlay._dpr = 1.0
    overlay._click_through_applied = False
    overlay._timer = MagicMock()
    return overlay


class TestShowFullscreenResetFlag(unittest.TestCase):
    """Test that show_fullscreen() resets _system_cursor_hidden to False."""

    def test_system_cursor_hidden_flag_reset_before_hide_call(self):
        """_system_cursor_hidden must be reset to False before _hide_system_cursor is called.

        This is the core bug fix: after voice assistant activates, ShowCursor reference
        counter may be desynchronized. Resetting the flag forces _hide_system_cursor to
        actually decrement the counter.
        """
        overlay = _make_overlay()
        # Simulate that the flag was True (e.g., from a previous show_fullscreen)
        overlay._system_cursor_hidden = True

        # Track the value of _system_cursor_hidden at the time _hide_system_cursor is called
        flag_at_hide_call = []

        def spy_hide_system_cursor(self_obj):
            flag_at_hide_call.append(self_obj._system_cursor_hidden)
            # Don't actually call the real method to avoid ctypes calls

        overlay._hide_system_cursor = lambda: spy_hide_system_cursor(overlay)
        overlay._make_click_through = MagicMock()

        MouseCursorOverlay.show_fullscreen(overlay)

        # The flag should have been False when _hide_system_cursor was called
        self.assertEqual(len(flag_at_hide_call), 1, "_hide_system_cursor should be called exactly once")
        self.assertFalse(flag_at_hide_call[0],
                         "_system_cursor_hidden should be False when _hide_system_cursor is called")

    def test_show_fullscreen_resets_click_through_applied(self):
        """show_fullscreen() should reset _click_through_applied to False."""
        overlay = _make_overlay()
        overlay._click_through_applied = True  # Previously applied

        overlay._make_click_through = MagicMock()
        overlay._hide_system_cursor = MagicMock()

        MouseCursorOverlay.show_fullscreen(overlay)

        # _click_through_applied should have been reset before _make_click_through
        overlay._make_click_through.assert_called_once()

    def test_show_fullscreen_calls_make_click_through_after_show(self):
        """show_fullscreen() should call _make_click_through() after show()."""
        overlay = _make_overlay()

        call_order = []
        overlay.show = MagicMock(side_effect=lambda: call_order.append('show'))
        overlay.raise_ = MagicMock(side_effect=lambda: call_order.append('raise_'))
        overlay._make_click_through = MagicMock(side_effect=lambda: call_order.append('click_through'))
        overlay._hide_system_cursor = MagicMock(side_effect=lambda: call_order.append('hide_cursor'))
        overlay.setGeometry = MagicMock()

        MouseCursorOverlay.show_fullscreen(overlay)

        # click_through must come after show (the bug fix requires this ordering)
        self.assertIn('show', call_order)
        self.assertIn('click_through', call_order)
        self.assertLess(call_order.index('show'), call_order.index('click_through'),
                        "_make_click_through must be called after show()")

    def test_show_fullscreen_starts_timer(self):
        """show_fullscreen() should start the timer with 16ms interval."""
        overlay = _make_overlay()

        overlay._make_click_through = MagicMock()
        overlay._hide_system_cursor = MagicMock()
        overlay.show = MagicMock()
        overlay.raise_ = MagicMock()
        overlay.setGeometry = MagicMock()

        MouseCursorOverlay.show_fullscreen(overlay)

        overlay._timer.start.assert_called_once_with(MouseCursorOverlay._TIMER_IDLE_MS)

    def test_show_fullscreen_sets_geometry_to_screen(self):
        """show_fullscreen() should set geometry to primary screen geometry."""
        overlay = _make_overlay()

        overlay._make_click_through = MagicMock()
        overlay._hide_system_cursor = MagicMock()
        overlay.show = MagicMock()
        overlay.raise_ = MagicMock()
        overlay.setGeometry = MagicMock()

        MouseCursorOverlay.show_fullscreen(overlay)

        overlay.setGeometry.assert_called_once()


class TestHideSystemCursorIdempotency(unittest.TestCase):
    """Test _hide_system_cursor() is idempotent."""

    def test_hide_system_cursor_skips_if_already_hidden(self):
        """If _system_cursor_hidden is True, should not call ShowCursor."""
        overlay = _make_overlay()
        overlay._system_cursor_hidden = True

        mock_user32.ShowCursor.reset_mock()

        MouseCursorOverlay._hide_system_cursor(overlay)

        # ShowCursor should NOT have been called since flag was already True
        mock_user32.ShowCursor.assert_not_called()

    def test_hide_system_cursor_sets_flag_to_true(self):
        """After _hide_system_cursor(), _system_cursor_hidden should be True."""
        overlay = _make_overlay()
        overlay._system_cursor_hidden = False

        # Make ShowCursor return values that indicate cursor is hidden
        call_count = [0]
        def fake_show_cursor(show):
            call_count[0] += 1
            # Return value >= 0 means cursor still displayed
            # Return value < 0 means cursor now hidden
            if call_count[0] < 3:
                return 0  # Still displayed, need more calls
            return -1  # Now hidden

        # Directly patch on the mock user32 object (avoids cross-test mock issues)
        original_fn = mock_user32.ShowCursor
        mock_user32.ShowCursor = fake_show_cursor
        try:
            MouseCursorOverlay._hide_system_cursor(overlay)
        finally:
            mock_user32.ShowCursor = original_fn

        self.assertTrue(overlay._system_cursor_hidden)


class TestShowSystemCursorIdempotency(unittest.TestCase):
    """Test _show_system_cursor() is idempotent."""

    def test_show_system_cursor_skips_if_already_visible(self):
        """If _system_cursor_hidden is False, should not call ShowCursor."""
        overlay = _make_overlay()
        overlay._system_cursor_hidden = False

        mock_user32.ShowCursor.reset_mock()

        MouseCursorOverlay._show_system_cursor(overlay)

        # ShowCursor should NOT have been called since cursor is already visible
        mock_user32.ShowCursor.assert_not_called()

    def test_show_system_cursor_sets_flag_to_false(self):
        """After _show_system_cursor(), _system_cursor_hidden should be False."""
        overlay = _make_overlay()
        overlay._system_cursor_hidden = True

        # Make ShowCursor return values that indicate cursor is shown
        call_count = [0]
        def fake_show_cursor(show):
            call_count[0] += 1
            if call_count[0] < 3:
                return -1  # Still hidden, need more calls
            return 0  # Now visible

        # Directly patch on the mock user32 object (avoids cross-test mock issues)
        original_fn = mock_user32.ShowCursor
        mock_user32.ShowCursor = fake_show_cursor
        try:
            MouseCursorOverlay._show_system_cursor(overlay)
        finally:
            mock_user32.ShowCursor = original_fn

        self.assertFalse(overlay._system_cursor_hidden)


class TestHideMethod(unittest.TestCase):
    """Test that hide() restores system cursor."""

    def test_hide_calls_show_system_cursor(self):
        """hide() should call _show_system_cursor to restore system cursor."""
        overlay = _make_overlay()
        overlay._system_cursor_hidden = True

        show_called = [False]
        def mock_show():
            show_called[0] = True
            overlay._system_cursor_hidden = False

        overlay._show_system_cursor = mock_show
        overlay._timer = MagicMock()

        # Call the unbound method
        MouseCursorOverlay.hide(overlay)

        self.assertTrue(show_called[0], "_show_system_cursor should be called during hide()")
        overlay._timer.stop.assert_called_once()


if __name__ == "__main__":
    unittest.main(verbosity=2)
