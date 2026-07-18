"""
T3: ConfigManager 配置持久化测试
验证缺失键自动使用默认值，以及保存后 config.json 包含正确的 edge 配置键。
"""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'app'))

from config_manager import ConfigManager


class TestConfigEdge(unittest.TestCase):
    """验证 ConfigManager 对 edge 配置的处理。"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.tmpdir, 'config.json')

    def tearDown(self):
        if os.path.exists(self.config_path):
            os.remove(self.config_path)
        os.rmdir(self.tmpdir)

    def test_missing_keys_use_defaults(self):
        """若 config.json 缺少新键，则自动使用默认值"""
        old_config = {
            "target_app": "WPS",
            "mouse_sensitivity": 40,
        }
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(old_config, f)

        cm = ConfigManager(config_file=self.config_path)
        self.assertIn("edge_acceleration_enabled", cm.config)
        self.assertIn("edge_acceleration_strength", cm.config)
        self.assertEqual(cm.get("edge_acceleration_enabled"), False)
        self.assertEqual(cm.get("edge_acceleration_strength"), 35)
        self.assertEqual(cm.get("pinch_freeze_enabled"), False)
        self.assertEqual(cm.get("pinch_hysteresis_enabled"), False)

    def test_save_preserves_edge_keys(self):
        """保存后 config.json 中包含正确的 edge 配置键"""
        # Start with empty config to trigger defaults
        cm = ConfigManager(config_file=self.config_path)
        self.assertEqual(cm.get("edge_acceleration_enabled"), False)
        self.assertEqual(cm.get("edge_acceleration_strength"), 35)

        # Modify values
        cm.set("edge_acceleration_enabled", False)
        cm.set("edge_acceleration_strength", 75)

        # Read back from file
        with open(self.config_path, encoding='utf-8') as f:
            saved = json.load(f)

        self.assertIn("edge_acceleration_enabled", saved)
        self.assertIn("edge_acceleration_strength", saved)
        self.assertEqual(saved["edge_acceleration_enabled"], False)
        self.assertEqual(saved["edge_acceleration_strength"], 75)

    def test_batch_update_preserves_edge_keys(self):
        """batch_update 保存后 config.json 包含正确的 edge 配置键"""
        cm = ConfigManager(config_file=self.config_path)
        with cm.batch_update():
            cm.set("edge_acceleration_enabled", True)
            cm.set("edge_acceleration_strength", 60)
            cm.set("mouse_sensitivity", 55)

        with open(self.config_path, encoding='utf-8') as f:
            saved = json.load(f)

        self.assertEqual(saved.get("edge_acceleration_enabled"), True)
        self.assertEqual(saved.get("edge_acceleration_strength"), 60)
        self.assertEqual(saved.get("mouse_sensitivity"), 55)

    def test_apply_stability_profile_sets_related_defaults(self):
        """体验档位会同步相关稳定性开关。"""
        cm = ConfigManager(config_file=self.config_path)
        cm.apply_stability_profile("long_range")
        self.assertEqual(cm.get("stability_profile"), "long_range")
        self.assertEqual(cm.get("edge_acceleration_enabled"), True)
        self.assertEqual(cm.get("edge_acceleration_strength"), 60)
        self.assertEqual(cm.get("long_range_enabled"), True)
        self.assertEqual(cm.get("draw_thumb_lift"), False)
        self.assertEqual(cm.get("pinch_freeze_enabled"), False)
        self.assertEqual(cm.get("pinch_hysteresis_enabled"), False)

        cm.apply_stability_profile("stable")
        self.assertEqual(cm.get("stability_profile"), "stable")
        self.assertEqual(cm.get("edge_acceleration_enabled"), False)
        self.assertEqual(cm.get("edge_acceleration_strength"), 35)
        self.assertEqual(cm.get("long_range_enabled"), False)
        self.assertEqual(cm.get("pinch_freeze_enabled"), False)
        self.assertEqual(cm.get("pinch_hysteresis_enabled"), False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
