"""
T4: 集成回归测试 — py_compile 验证语法，检查未修改文件。
T5: UI 逻辑验证 — 代码级检查 main_ui.py 中的边缘加速 UI 逻辑。
"""
import ast
import hashlib
import os
import py_compile
import sys
import unittest


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
APP_DIR = os.path.join(BASE_DIR, 'app')

# Expected SHA256 hashes for files that should NOT be modified
_UNCHANGED_FILES = {
    'app/modes/mouse_mode.py': None,
    'app/modes/draw_mode.py': None,
    'app/modes/presentation.py': None,
    'app/services/hand_tracker.py': None,
    'app/services/gesture_recognizer.py': None,
    'app/drawing_overlay.py': None,
}


def _file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        h.update(f.read())
    return h.hexdigest()


class T4IntegrationRegression(unittest.TestCase):
    """集成回归测试。"""

    def test_py_compile_all_modified(self):
        """使用 py_compile 验证所有 3 个改动文件无语法错误"""
        modified = [
            'app/config_manager.py',
            'app/services/mouse_controller.py',
            'app/main_ui.py',
        ]
        for rel in modified:
            full = os.path.join(BASE_DIR, rel)
            with self.subTest(file=rel):
                try:
                    py_compile.compile(full, doraise=True)
                except py_compile.PyCompileError as e:
                    self.fail(f"Syntax error in {rel}: {e}")

    def test_unmodified_files_not_changed(self):
        """检查指定文件没有被意外修改（通过存在性判断，无 baseline 时仅验证存在）"""
        for rel in _UNCHANGED_FILES:
            full = os.path.join(BASE_DIR, rel)
            with self.subTest(file=rel):
                self.assertTrue(os.path.exists(full), f"{rel} does not exist")


class T5UILogicVerification(unittest.TestCase):
    """UI 逻辑代码级验证。"""

    @classmethod
    def setUpClass(cls):
        cls.main_ui_path = os.path.join(APP_DIR, 'main_ui.py')
        with open(cls.main_ui_path, 'r', encoding='utf-8') as f:
            cls.source = f.read()
        cls.tree = ast.parse(cls.source)

    def test_edge_strength_spin_initial_enabled_state(self):
        """edge_strength_spin 初始状态是否与复选框状态一致"""
        # Search for setEnabled(self.edge_check.isChecked()) or equivalent
        found = False
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Attribute) and func.attr == 'setEnabled':
                    # Check if argument refers to edge_check.isChecked()
                    if node.args:
                        arg = node.args[0]
                        if isinstance(arg, ast.Call):
                            inner = arg.func
                            if isinstance(inner, ast.Attribute) and inner.attr == 'isChecked':
                                if isinstance(inner.value, ast.Attribute) and 'edge_check' in ast.dump(inner.value):
                                    found = True
        self.assertTrue(found, "edge_strength_spin.setEnabled should be tied to edge_check.isChecked()")

    def test_on_edge_toggles_set_enabled(self):
        """_on_edge_toggled 是否正确联动 setEnabled"""
        found = False
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef) and node.name == '_on_edge_toggled':
                body_dump = ast.dump(node, include_attributes=False)
                self.assertIn('setEnabled', body_dump,
                    "_on_edge_toggled should call setEnabled on edge_strength_spin")
                found = True
        self.assertTrue(found, "_on_edge_toggled method not found")

    def test_save_settings_writes_both_edge_keys(self):
        """save_settings() 是否将两个新配置项都写入了 batch_update"""
        found_save = False
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef) and node.name == 'save_settings':
                found_save = True
                body_dump = ast.dump(node, include_attributes=False)
                self.assertIn('edge_acceleration_enabled', body_dump,
                    "save_settings must write edge_acceleration_enabled")
                self.assertIn('edge_acceleration_strength', body_dump,
                    "save_settings must write edge_acceleration_strength")
        self.assertTrue(found_save, "save_settings method not found")

    def test_apply_config_calls_set_edge_acceleration(self):
        """apply_config() 是否正确调用了 mouse.set_edge_acceleration()"""
        found_apply = False
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef) and node.name == 'apply_config':
                found_apply = True
                body_dump = ast.dump(node, include_attributes=False)
                self.assertIn('set_edge_acceleration', body_dump,
                    "apply_config must call mouse.set_edge_acceleration()")
        self.assertTrue(found_apply, "apply_config method not found")

    def test_floating_window_constructor_passes_edge_params(self):
        """FloatingWindow 构造时是否将 edge 参数传给 MouseController"""
        found = False
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ClassDef) and node.name == 'FloatingWindow':
                body_dump = ast.dump(node, include_attributes=False)
                self.assertIn('edge_enabled', body_dump,
                    "FloatingWindow should pass edge_enabled to MouseController")
                self.assertIn('edge_strength', body_dump,
                    "FloatingWindow should pass edge_strength to MouseController")
                found = True
        self.assertTrue(found, "FloatingWindow class not found")


if __name__ == "__main__":
    unittest.main(verbosity=2)
