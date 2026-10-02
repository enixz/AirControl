"""cloud 提供商预设解析测试：moma 默认回落、custom 必填、env key 读取。"""
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'app'))

from services.cloud import providers


class TestCloudProviders(unittest.TestCase):
    def test_moma_preset_fallback(self):
        """字段为 None 时回落到 moma 预设默认值。"""
        with mock.patch.dict(os.environ, {"MOMA_API_KEY": "k-test"}):
            resolved = providers.resolve_provider({"enabled": True, "provider": "moma"})
        self.assertEqual(resolved["base_url"], "https://zhenze-huhehaote.cmecloud.cn/v1")
        self.assertEqual(resolved["api_key_env"], "MOMA_API_KEY")
        self.assertEqual(resolved["llm_fast_model"], "deepseek-v4.1-flash")
        self.assertEqual(resolved["llm_smart_model"], "DeepSeek-V3.2")
        self.assertEqual(resolved["llm_vision_model"], "Qwen2.5-VL-72B-Instruct")
        self.assertEqual(resolved["api_key"], "k-test")

    def test_explicit_fields_override_preset(self):
        """显式配置的字段优先于预设。"""
        cfg = {
            "enabled": True,
            "provider": "moma",
            "base_url": "https://example.com/v1",
            "llm_fast_model": "my-fast",
        }
        with mock.patch.dict(os.environ, {"MOMA_API_KEY": "k-test"}):
            resolved = providers.resolve_provider(cfg)
        self.assertEqual(resolved["base_url"], "https://example.com/v1")
        self.assertEqual(resolved["llm_fast_model"], "my-fast")
        # 未显式配置的字段仍回落预设
        self.assertEqual(resolved["llm_smart_model"], "DeepSeek-V3.2")

    def test_api_key_only_from_env(self):
        """API Key 只从 api_key_env 指定的环境变量读取。"""
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MOMA_API_KEY", None)
            resolved = providers.resolve_provider({"enabled": True, "provider": "moma"})
            self.assertIsNone(resolved["api_key"])
            self.assertFalse(providers.is_configured({"enabled": True, "provider": "moma"}))

    def test_custom_requires_explicit_fields(self):
        """custom 预设无默认值：缺 base_url/api_key_env 视为未配置。"""
        cfg = {"enabled": True, "provider": "custom"}
        resolved = providers.resolve_provider(cfg)
        self.assertIsNone(resolved["base_url"])
        self.assertIsNone(resolved["api_key_env"])
        self.assertIsNone(resolved["llm_fast_model"])
        self.assertFalse(providers.is_configured(cfg))

    def test_custom_fully_configured(self):
        """custom 字段齐全且有 env key 时可用。"""
        cfg = {
            "enabled": True,
            "provider": "custom",
            "base_url": "https://my.api/v1",
            "api_key_env": "MY_API_KEY",
            "llm_fast_model": "f",
            "llm_smart_model": "s",
            "llm_vision_model": "v",
        }
        with mock.patch.dict(os.environ, {"MY_API_KEY": "k-custom"}):
            resolved = providers.resolve_provider(cfg)
            self.assertEqual(resolved["api_key"], "k-custom")
            self.assertTrue(providers.is_configured(cfg))

    def test_unknown_provider_treated_as_custom(self):
        """未知 provider 按 custom 处理（不设默认）。"""
        resolved = providers.resolve_provider({"enabled": True, "provider": "unknown"})
        self.assertIsNone(resolved["base_url"])

    def test_disabled_is_not_configured(self):
        """enabled=False 时即使有 key 也不算已配置。"""
        with mock.patch.dict(os.environ, {"MOMA_API_KEY": "k-test"}):
            self.assertFalse(
                providers.is_configured({"enabled": False, "provider": "moma"})
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
