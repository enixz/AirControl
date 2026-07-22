"""回归测试：悬浮窗/绘图工具栏按钮必须 NoFocus（2026-07-22 实录"崩溃"）。

根因：空格是近距录像的真值标记键，Qt 中 QPushButton 获得键盘焦点后，
空格松开瞬间会触发 clicked——悬浮窗 X 按钮被激活 = closeEvent =
程序整体退出（用户视角"录像中突然崩溃"，实误触发退出）。
修复：相关按钮/滑条一律 setFocusPolicy(Qt.FocusPolicy.NoFocus)。

注：本仓库 UI 测试约定不走真实 QApplication（mock/源码级），
真实实例化已在 2026-07-22 手动验证（offscreen 下 10 个控件全部 NoFocus）。
"""
import unittest
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
APP_DIR = BASE_DIR / "app"


class TestMainUiButtonFocusPolicy(unittest.TestCase):
    """main_ui.py 悬浮窗顶行三按钮（⚙/─/X）必须设 NoFocus（源码级回归）。"""

    def test_top_row_buttons_no_focus(self):
        src = (APP_DIR / "main_ui.py").read_text(encoding="utf-8")
        pattern = (
            r"for _btn in \(self\.btn_settings, self\.btn_minimize, self\.btn_close\):"
            r"\s*\n\s*_btn\.setFocusPolicy\(Qt\.FocusPolicy\.NoFocus\)"
        )
        self.assertRegex(
            src, pattern,
            "悬浮窗 btn_settings/btn_minimize/btn_close 缺少 NoFocus 设置："
            "空格标记键会在松开瞬间激活焦点按钮（X = 直接退出程序）",
        )


class TestDrawToolbarFocusPolicy(unittest.TestCase):
    """draw_toolbar.py 全部按钮/滑条必须设 NoFocus（源码级回归）。"""

    def test_all_buttons_and_slider_no_focus(self):
        src = (APP_DIR / "draw_toolbar.py").read_text(encoding="utf-8")
        pattern = (
            r"for _w in self\.findChildren\(QPushButton\) \+ self\.findChildren\(QSlider\):"
            r"\s*\n\s*_w\.setFocusPolicy\(Qt\.FocusPolicy\.NoFocus\)"
        )
        self.assertRegex(
            src, pattern,
            "DrawToolbar 按钮/滑条缺少 NoFocus 设置："
            "录像时敲空格会在松开瞬间误触发清空/撤销/改笔粗",
        )


if __name__ == "__main__":
    unittest.main()

