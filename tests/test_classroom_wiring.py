"""M4 UI 接线测试：orchestrator 的课堂智能触发 / 生成 / 写屏 / 转录累积。

所有云端调用均 mock，严禁真实网络。
"""
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'app'))

from orchestrator import AirControlOrchestrator  # noqa: E402


def _make_orchestrator():
    with patch('orchestrator.AirControlOrchestrator.init_services'), \
         patch('orchestrator.AirControlOrchestrator._init_modes'), \
         patch('orchestrator.AirControlOrchestrator.set_mode'), \
         patch('orchestrator.ConfigManager'), \
         patch('orchestrator.MouseController'):
        return AirControlOrchestrator(MagicMock(), MagicMock(), MagicMock())


class TestClassroomWiring(unittest.TestCase):
    def setUp(self):
        self.orch = _make_orchestrator()
        self.status = []
        self.orch.voice_status_updated.connect(self.status.append)
        self.classroom = MagicMock()
        self.orch.classroom_ai = self.classroom
        self.written = []
        self.orch._classroom_text_signal.connect(self.written.append)

    def _run_generation_sync(self, kind):
        """拦截后台线程启动，同步执行生成逻辑（便于断言）。"""
        with patch.object(self.orch, '_start_background_thread') as bg:
            self.orch._start_classroom_generation(kind)
            if not bg.called:
                return None
            target = bg.call_args.args[0]
            args = bg.call_args.kwargs.get("args", ())
            target(*args)
            return bg

    # ------------------------------------------------------------------
    # 触发前置检查
    # ------------------------------------------------------------------

    def test_cloud_unavailable_blocks_generation(self):
        """云端未配置：提示且不启动生成线程。"""
        self.classroom.is_available.return_value = False
        bg = self._run_generation_sync("summary")
        self.assertIsNone(bg)
        self.assertTrue(any("云端未配置" in s for s in self.status))

    def test_empty_transcript_blocks_generation(self):
        """无板书内容：提示且不启动生成线程。"""
        self.classroom.is_available.return_value = True
        self.classroom.get_transcript.return_value = ""
        bg = self._run_generation_sync("summary")
        self.assertIsNone(bg)
        self.assertTrue(any("还没有板书内容" in s for s in self.status))

    def test_busy_guard_rejects_reentry(self):
        """生成进行中再次触发：提示"正在生成中"，不重复启动。"""
        self.classroom.is_available.return_value = True
        self.classroom.get_transcript.return_value = "板书"
        self.orch._classroom_busy = True
        with patch.object(self.orch, '_start_background_thread') as bg:
            self.orch._start_classroom_generation("quiz")
            bg.assert_not_called()
        self.assertTrue(any("正在生成中" in s for s in self.status))

    def test_degraded_cloud_blocks_generation(self):
        """auto_fallback：degraded 时拦截生成，不启动线程、不进 busy。"""
        self.classroom.is_available.return_value = True
        self.classroom.get_transcript.return_value = "板书"
        self.orch.cloud_status = "degraded"
        feedback = MagicMock()
        self.orch.voice_feedback = feedback
        with patch.object(self.orch, '_start_background_thread') as bg:
            self.orch._start_classroom_generation("summary")
            bg.assert_not_called()
        self.assertFalse(self.orch._classroom_busy)
        feedback.say.assert_called_once_with("网络不佳，请稍后再试")
        self.assertTrue(any("网络不佳" in s for s in self.status))

    # ------------------------------------------------------------------
    # 生成主流程
    # ------------------------------------------------------------------

    @patch('orchestrator.writable_data_dir')
    @patch('orchestrator.save_markdown', return_value=True)
    def test_summary_flow(self, mock_save, mock_data_dir):
        """小结生成：导出 md、写屏信号、busy 复位。"""
        mock_data_dir.return_value = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'tmp_notes'
        )
        self.classroom.is_available.return_value = True
        self.classroom.get_transcript.return_value = "勾股定理板书"
        result = MagicMock()
        result.ok = True
        result.text = "# 课堂小结\n- 勾股定理"
        result.reason = "ok"
        self.classroom.generate_summary.return_value = result

        self._run_generation_sync("summary")

        self.classroom.generate_summary.assert_called_once()
        self.assertEqual(self.written, ["# 课堂小结\n- 勾股定理"])
        self.assertTrue(mock_save.called)
        self.assertIn("class_summary_", mock_save.call_args.args[1])
        self.assertFalse(self.orch._classroom_busy)
        self.assertTrue(any("课堂小结" in s for s in self.status))

    @patch('orchestrator.writable_data_dir')
    @patch('orchestrator.save_markdown', return_value=True)
    def test_quiz_flow_uses_generate_quiz(self, mock_save, mock_data_dir):
        """出题走 generate_quiz(count=3)。"""
        mock_data_dir.return_value = "tmp_notes"
        self.classroom.is_available.return_value = True
        self.classroom.get_transcript.return_value = "板书"
        result = MagicMock(ok=True, text="三道题", reason="ok")
        self.classroom.generate_quiz.return_value = result

        self._run_generation_sync("quiz")

        self.classroom.generate_quiz.assert_called_once_with(count=3)
        self.assertIn("class_quiz_", mock_save.call_args.args[1])
        self.assertEqual(self.written, ["三道题"])

    @patch('orchestrator.writable_data_dir')
    @patch('orchestrator.save_markdown', return_value=True)
    def test_generation_failure_reports_reason(self, mock_save, mock_data_dir):
        """生成失败：提示原因，不写屏，busy 复位。"""
        mock_data_dir.return_value = "tmp_notes"
        self.classroom.is_available.return_value = True
        self.classroom.get_transcript.return_value = "板书"
        result = MagicMock(ok=False, reason="empty_output")
        self.classroom.generate_summary.return_value = result

        self._run_generation_sync("summary")

        self.assertEqual(self.written, [])
        self.assertTrue(any("生成失败" in s for s in self.status))
        self.assertFalse(self.orch._classroom_busy)

    @patch('orchestrator.writable_data_dir')
    @patch('orchestrator.save_markdown', return_value=False)
    def test_export_failure_still_writes_screen(self, mock_save, mock_data_dir):
        """导出失败不影响写屏展示。"""
        mock_data_dir.return_value = "tmp_notes"
        self.classroom.is_available.return_value = True
        self.classroom.get_transcript.return_value = "板书"
        result = MagicMock(ok=True, text="小结", reason="ok")
        self.classroom.generate_summary.return_value = result

        self._run_generation_sync("summary")

        self.assertEqual(self.written, ["小结"])
        self.assertFalse(mock_save.return_value or False)
        self.assertFalse(self.orch._classroom_busy)

    # ------------------------------------------------------------------
    # execute_action 分发与转录累积
    # ------------------------------------------------------------------

    def test_execute_action_dispatch(self):
        """class_summary / class_quiz 正确分发到课堂生成。"""
        with patch.object(self.orch, '_start_classroom_generation') as gen:
            self.orch.execute_action("class_summary")
            gen.assert_called_once_with("summary")
            self.orch.execute_action("class_quiz")
            gen.assert_called_with("quiz")

    def test_render_dictation_appends_transcript(self):
        """听写文本写画布成功后累积进课堂 buffer。"""
        self.orch.overlay = MagicMock()
        self.orch._render_dictation_text("勾股定理", None)
        self.classroom.append_transcript.assert_called_once_with("勾股定理")

    def test_render_failure_does_not_append(self):
        """写画布失败时不累积（避免污染 buffer）。"""
        self.orch.overlay = MagicMock()
        self.orch.overlay.draw_text.side_effect = RuntimeError("boom")
        self.orch._render_dictation_text("坏文本", None)
        self.classroom.append_transcript.assert_not_called()

    def test_on_classroom_text_draws_to_canvas(self):
        """生成内容经信号写画布：先清实时字幕再写屏。"""
        self.orch.overlay = MagicMock()
        self.orch._on_classroom_text("小结文本")
        self.orch.overlay.clear_dictation_caption.assert_called_once()
        self.orch.overlay.draw_text.assert_called_once_with("小结文本")
        self.assertTrue(any("已写屏" in s for s in self.status))

    # ------------------------------------------------------------------
    # 触发词注册
    # ------------------------------------------------------------------

    def test_kws_trigger_words_registered(self):
        """新触发词已进映射表与 draw 模式关键词表。"""
        from services.voice_actions import MODE_KEYWORDS, VOICE_KEYWORD_TO_ACTION
        self.assertEqual(VOICE_KEYWORD_TO_ACTION["下课总结"], "class_summary")
        self.assertEqual(VOICE_KEYWORD_TO_ACTION["出三道题"], "class_quiz")
        self.assertIn("下课总结", MODE_KEYWORDS["draw"])
        self.assertIn("出三道题", MODE_KEYWORDS["draw"])

    def test_keywords_file_has_new_lines(self):
        """keywords.txt 已含两个新触发词（拼音行，KWS 启动时自动生效）。"""
        path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), '..', 'app',
            'voice_keywords', 'keywords.txt'
        )
        with open(path, encoding="utf-8") as f:
            content = f.read()
        self.assertIn("@下课总结", content)
        self.assertIn("@出三道题", content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
