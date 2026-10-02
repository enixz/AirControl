"""classroom_ai 课堂智能层测试：mock LLMClient，严禁真实网络。"""
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'app'))

from services.classroom_ai import (  # noqa: E402
    ClassroomAI,
    save_markdown,
)


def _client(configured=True):
    client = mock.MagicMock()
    client.is_configured.return_value = configured
    return client


class TestTranscriptBuffer(unittest.TestCase):
    def test_append_and_get(self):
        ai = ClassroomAI(_client())
        ai.append_transcript("第一行")
        ai.append_transcript("第二行")
        self.assertEqual(ai.get_transcript(), "第一行\n第二行")

    def test_blank_ignored(self):
        ai = ClassroomAI(_client())
        ai.append_transcript("   ")
        ai.append_transcript(None)
        self.assertEqual(ai.get_transcript(), "")

    def test_clear(self):
        ai = ClassroomAI(_client())
        ai.append_transcript("x")
        ai.clear_transcript()
        self.assertEqual(ai.get_transcript(), "")


class TestSummaryQuiz(unittest.TestCase):
    def test_summary_ok(self):
        client = _client()
        client.call_smart.return_value = ("## 核心知识点\n- 勾股定理", {"total_tokens": 100})
        ai = ClassroomAI(client)
        ai.append_transcript("勾股定理 a²+b²=c²")
        result = ai.generate_summary()
        self.assertTrue(result.ok)
        self.assertIn("核心知识点", result.text)
        # 转录被送入 user 消息
        messages = client.call_smart.call_args.args[0]
        self.assertIn("勾股定理", messages[-1]["content"])

    def test_quiz_count_in_prompt(self):
        client = _client()
        client.call_smart.return_value = ("题目", {})
        ai = ClassroomAI(client)
        ai.append_transcript("内容")
        ai.generate_quiz(count=5)
        messages = client.call_smart.call_args.args[0]
        self.assertIn("5 道", messages[0]["content"])

    def test_quiz_count_clamped(self):
        client = _client()
        client.call_smart.return_value = ("题目", {})
        ai = ClassroomAI(client)
        ai.append_transcript("内容")
        ai.generate_quiz(count=999)
        messages = client.call_smart.call_args.args[0]
        self.assertIn("10 道", messages[0]["content"])

    def test_outline_in_system_prompt(self):
        client = _client()
        client.call_smart.return_value = ("小结", {})
        ai = ClassroomAI(client, outline="第三章 三角函数")
        ai.append_transcript("sin/cos")
        ai.generate_summary()
        messages = client.call_smart.call_args.args[0]
        self.assertIn("第三章 三角函数", messages[0]["content"])

    def test_set_outline_after_construction(self):
        client = _client()
        client.call_smart.return_value = ("小结", {})
        ai = ClassroomAI(client)
        ai.set_outline("新大纲")
        ai.append_transcript("x")
        ai.generate_summary()
        self.assertIn("新大纲", client.call_smart.call_args.args[0][0]["content"])

    def test_empty_transcript(self):
        ai = ClassroomAI(_client())
        result = ai.generate_summary()
        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "empty_transcript")

    def test_not_configured(self):
        client = _client(configured=False)
        ai = ClassroomAI(client)
        ai.append_transcript("内容")
        result = ai.generate_summary()
        self.assertEqual(result.reason, "cloud_not_configured")
        client.call_smart.assert_not_called()

    def test_llm_error_swallowed(self):
        client = _client()
        client.call_smart.side_effect = RuntimeError("boom")
        ai = ClassroomAI(client)
        ai.append_transcript("内容")
        result = ai.generate_summary()
        self.assertFalse(result.ok)
        self.assertTrue(result.reason.startswith("llm_error:"))

    def test_empty_output(self):
        client = _client()
        client.call_smart.return_value = ("   ", {})
        ai = ClassroomAI(client)
        ai.append_transcript("内容")
        result = ai.generate_summary()
        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "empty_output")

    def test_long_transcript_truncated(self):
        client = _client()
        client.call_smart.return_value = ("小结", {})
        ai = ClassroomAI(client)
        ai.append_transcript("A" * 20000)
        ai.generate_summary()
        sent = client.call_smart.call_args.args[0][-1]["content"]
        self.assertLess(len(sent), 20000)


class TestVision(unittest.TestCase):
    def test_vision_uses_image(self):
        client = _client()
        client.call_vision.return_value = ("结构化笔记", {})
        ai = ClassroomAI(client)
        result = ai.generate_notes_from_image("data:image/png;base64,AAAA")
        self.assertTrue(result.ok)
        messages = client.call_vision.call_args.args[0]
        payload = messages[-1]["content"]
        self.assertEqual(payload[1]["type"], "image_url")
        self.assertTrue(payload[1]["image_url"]["url"].startswith("data:image/png"))

    def test_empty_image(self):
        ai = ClassroomAI(_client())
        result = ai.generate_notes_from_image("")
        self.assertEqual(result.reason, "empty_image")

    def test_vision_not_configured(self):
        client = _client(configured=False)
        ai = ClassroomAI(client)
        result = ai.generate_notes_from_image("data:image/png;base64,AAAA")
        self.assertEqual(result.reason, "cloud_not_configured")
        client.call_vision.assert_not_called()


class TestSaveMarkdown(unittest.TestCase):
    def test_writes_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "summary.md")
            self.assertTrue(save_markdown("# 小结\n内容", path))
            with open(path, encoding="utf-8") as f:
                self.assertEqual(f.read(), "# 小结\n内容")

    def test_bad_path_returns_false(self):
        self.assertFalse(save_markdown("x", os.path.join("Z:", "nope", "a.md")))


if __name__ == "__main__":
    unittest.main(verbosity=2)
