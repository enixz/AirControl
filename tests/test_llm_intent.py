"""llm_intent 意图解析测试：mock LLMClient，严禁真实网络。"""
import json
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'app'))

from services.cloud import llm_intent  # noqa: E402
from services.voice_actions import (  # noqa: E402
    EXTRA_VOICE_ACTIONS,
    VOICE_KEYWORD_TO_ACTION,
)


def _client(configured=True):
    client = mock.MagicMock()
    client.is_configured.return_value = configured
    return client


def _reply(payload):
    return (json.dumps(payload, ensure_ascii=False), {"total_tokens": 10})


class TestAllowedActions(unittest.TestCase):
    def test_covers_all_mapped_actions(self):
        """白名单覆盖全部关键词映射与补充动作。"""
        allowed = llm_intent.allowed_actions()
        for action in VOICE_KEYWORD_TO_ACTION.values():
            self.assertIn(action, allowed)
        for action in EXTRA_VOICE_ACTIONS:
            self.assertIn(action, allowed)

    def test_prompt_lists_all_actions_without_mode(self):
        """无 mode 时 prompt 列出全局白名单全部动作。"""
        prompt = llm_intent.build_system_prompt()
        for action in llm_intent.allowed_actions():
            self.assertIn(action, prompt)

    def test_prompt_narrows_to_mode(self):
        """mode=presentation 时不列 draw 专属动作，但保留全局补充动作。"""
        prompt = llm_intent.build_system_prompt(mode="presentation")
        self.assertNotIn("clear_canvas", prompt)
        self.assertIn("next_slide", prompt)
        self.assertIn("switch_app", prompt)
        self.assertIn("presentation", prompt)

    def test_allowed_actions_respects_mode(self):
        """白名单按模式收窄；EXTRA 全局动作保留；未知 mode 不收窄。"""
        draw = llm_intent.allowed_actions("draw")
        self.assertNotIn("next_slide", draw)
        self.assertIn("clear_canvas", draw)
        self.assertIn("hang_up_voice_assistant", draw)
        self.assertEqual(
            llm_intent.allowed_actions("unknown-mode"),
            llm_intent.allowed_actions(),
        )


class TestParseIntent(unittest.TestCase):
    def test_maps_fixed_command(self):
        client = _client()
        client.call_fast.return_value = _reply(
            {"action": "next_slide", "args": {}, "confidence": 0.92}
        )
        result = llm_intent.parse_intent("下一页", client)
        self.assertTrue(result.accepted)
        self.assertEqual(result.action, "next_slide")
        self.assertEqual(result.reason, "ok")

    def test_parses_fenced_json(self):
        """容忍 ```json 围栏。"""
        client = _client()
        client.call_fast.return_value = (
            '```json\n{"action": "prev_slide", "confidence": 0.8}\n```',
            {},
        )
        result = llm_intent.parse_intent("帮我把上一页再放一遍", client)
        self.assertTrue(result.accepted)
        self.assertEqual(result.action, "prev_slide")

    def test_rejects_low_confidence(self):
        client = _client()
        client.call_fast.return_value = _reply(
            {"action": "next_slide", "confidence": 0.4}
        )
        result = llm_intent.parse_intent("那啥", client)
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "low_confidence")
        self.assertEqual(result.candidate, "next_slide")

    def test_rejects_unknown_action(self):
        """白名单外动作（如方案提到的 zoom_in_region 尚未实现）被拒绝。"""
        client = _client()
        client.call_fast.return_value = _reply(
            {"action": "zoom_in_region", "confidence": 0.99}
        )
        result = llm_intent.parse_intent("放大这块", client)
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "action_not_allowed")
        self.assertEqual(result.candidate, "zoom_in_region")

    def test_mode_rejects_out_of_mode_action(self):
        """presentation 模式下 draw 专属动作（清屏）被模式白名单拒绝。"""
        client = _client()
        client.call_fast.return_value = _reply(
            {"action": "clear_canvas", "confidence": 0.95}
        )
        result = llm_intent.parse_intent("擦掉", client, mode="presentation")
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "action_not_allowed")
        self.assertEqual(result.candidate, "clear_canvas")

    def test_mode_allows_global_extra_action(self):
        """EXTRA 全局动作（挂断）不受模式限制。"""
        client = _client()
        client.call_fast.return_value = _reply(
            {"action": "hang_up_voice_assistant", "confidence": 0.9}
        )
        result = llm_intent.parse_intent("挂断", client, mode="draw")
        self.assertTrue(result.accepted)
        self.assertEqual(result.action, "hang_up_voice_assistant")

    def test_null_action(self):
        client = _client()
        client.call_fast.return_value = _reply({"action": None, "confidence": 0})
        result = llm_intent.parse_intent("今天天气不错", client)
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "no_action")

    def test_invalid_json(self):
        client = _client()
        client.call_fast.return_value = ("抱歉，我没听懂", {})
        result = llm_intent.parse_intent("嗯", client)
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "invalid_json")

    def test_empty_input(self):
        result = llm_intent.parse_intent("   ", _client())
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "empty_input")

    def test_unconfigured_client(self):
        """云端未配置时直接拒绝，不发起调用。"""
        client = _client(configured=False)
        result = llm_intent.parse_intent("下一页", client)
        self.assertEqual(result.reason, "cloud_not_configured")
        client.call_fast.assert_not_called()

    def test_llm_error_is_swallowed(self):
        client = _client()
        client.call_fast.side_effect = RuntimeError("boom")
        result = llm_intent.parse_intent("下一页", client)
        self.assertFalse(result.accepted)
        self.assertTrue(result.reason.startswith("llm_error:"))

    def test_args_passed_through(self):
        client = _client()
        client.call_fast.return_value = _reply(
            {"action": "switch_to_draw", "args": {"source": "voice"},
             "confidence": 0.9}
        )
        result = llm_intent.parse_intent("切到板书模式", client)
        self.assertEqual(result.action, "switch_to_draw")
        self.assertEqual(result.args, {"source": "voice"})

    def test_non_dict_args_defaults_empty(self):
        client = _client()
        client.call_fast.return_value = _reply(
            {"action": "clear_canvas", "args": "x", "confidence": 0.9}
        )
        result = llm_intent.parse_intent("清屏", client)
        self.assertEqual(result.args, {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
