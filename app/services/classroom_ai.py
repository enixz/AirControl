"""课堂智能层（M4）— 板书转录 / 画布截图 → 课堂小结、随堂出题、板书成文

背景：云端 ASR 不可用（见 docs/m0_api_verification.md），板书转录由本地
SenseVoice 提供；本模块消费转录文本 buffer（累计）或画布截图，调用 zhenze
LLM 生成教学内容：

- 课堂小结 / 随堂出题 → ``call_smart``（默认 DeepSeek-V3.2）
- 板书成文（截图 → 结构化笔记）→ ``call_vision``（默认 Qwen2.5-VL-72B-Instruct）

纯逻辑、无音频 / UI 依赖，可单测。生成结果由调用方负责展示或导出。
"""

import logging
import threading

logger = logging.getLogger(__name__)

DEFAULT_QUIZ_COUNT = 3
MAX_TRANSCRIPT_CHARS = 12000

_SUMMARY_SYSTEM = (
    "你是课堂记录助手。根据本节课的板书转录文本，生成一份 Markdown 课堂小结，"
    "包含三个部分：## 核心知识点、## 要点、## 遗留问题。语言精炼，忠于原文，"
    "不要编造板书里没有的内容。"
)

_QUIZ_SYSTEM_TEMPLATE = (
    "你是出题助手。根据板书内容出 {count} 道随堂练习题，难度贴合板书知识点。"
    "用 Markdown 输出，每题给出题干与答案（答案用 "
    "<details><summary>查看答案</summary>...</details> 折叠）。只依据板书内容出题。"
)

_VISION_SYSTEM = (
    "你是板书整理助手。识别图片中的板书内容，输出结构化 Markdown 笔记："
    "先归纳主题，再分点整理公式 / 图示 / 文字，最后列出疑问点。"
    "只依据图片可见内容，不要臆测。"
)


class GenerationResult:
    """课堂智能生成结果。``text`` 为空表示失败。"""

    def __init__(self, text=None, usage=None, reason="ok"):
        self.text = text
        self.usage = usage or {}
        self.reason = reason

    @property
    def ok(self):
        return bool(self.text)

    def __repr__(self):
        preview = (self.text or "")[:40].replace("\n", " ")
        return f"GenerationResult(ok={self.ok}, reason={self.reason!r}, text={preview!r})"


class ClassroomAI:
    """课堂智能生成器。

    Args:
        llm_client: ``LLMClient`` 实例；未配置时所有生成直接拒绝。
        outline: 可选课程大纲文本（txt/md 全文），拼进 system prompt 贴合教学进度。
    """

    def __init__(self, llm_client, outline=None):
        self._llm = llm_client
        self._outline = (outline or "").strip()
        self._transcript = []
        self._lock = threading.Lock()

    def is_available(self):
        """云端 LLM 已配置。"""
        return self._llm is not None and self._llm.is_configured()

    # ------------------------------------------------------------------
    # 板书转录 buffer
    # ------------------------------------------------------------------

    def append_transcript(self, text):
        """追加一段板书转录（空文本忽略）。"""
        if not text or not str(text).strip():
            return
        with self._lock:
            self._transcript.append(str(text).strip())

    def get_transcript(self):
        with self._lock:
            return "\n".join(self._transcript)

    def clear_transcript(self):
        with self._lock:
            self._transcript = []

    def set_outline(self, outline):
        self._outline = (outline or "").strip()

    # ------------------------------------------------------------------
    # 生成
    # ------------------------------------------------------------------

    def generate_summary(self):
        """课堂小结：转录 → Markdown（知识点 / 要点 / 遗留问题）。"""
        return self._generate_text(_SUMMARY_SYSTEM, "请生成课堂小结。")

    def generate_quiz(self, count=DEFAULT_QUIZ_COUNT):
        """随堂出题：转录 → N 道题（含折叠答案）。"""
        try:
            count = max(1, min(10, int(count)))
        except (TypeError, ValueError):
            count = DEFAULT_QUIZ_COUNT
        return self._generate_text(_QUIZ_SYSTEM_TEMPLATE.format(count=count), "请出题。")

    def generate_notes_from_image(self, image_data_url):
        """板书成文：画布截图（data URL）→ 结构化 Markdown 笔记。"""
        if not self.is_available():
            return GenerationResult(reason="cloud_not_configured")
        if not image_data_url:
            return GenerationResult(reason="empty_image")
        messages = [
            {"role": "system", "content": self._with_outline(_VISION_SYSTEM)},
            {"role": "user", "content": [
                {"type": "text", "text": "请整理这张板书图片。"},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ]},
        ]
        return self._call(self._llm.call_vision, messages)

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _with_outline(self, system):
        if self._outline:
            return f"{system}\n\n课程大纲（供参考，贴合教学进度）：\n{self._outline}"
        return system

    def _generate_text(self, system, instruction):
        if not self.is_available():
            return GenerationResult(reason="cloud_not_configured")
        transcript = self.get_transcript()
        if not transcript:
            return GenerationResult(reason="empty_transcript")
        if len(transcript) > MAX_TRANSCRIPT_CHARS:
            transcript = transcript[-MAX_TRANSCRIPT_CHARS:]
        messages = [
            {"role": "system", "content": self._with_outline(system)},
            {"role": "user", "content": f"板书转录如下：\n{transcript}\n\n{instruction}"},
        ]
        return self._call(self._llm.call_smart, messages)

    def _call(self, fn, messages):
        try:
            content, usage = fn(messages)
        except Exception as exc:  # noqa: BLE001
            logger.warning("课堂智能生成失败: %s", exc)
            return GenerationResult(reason=f"llm_error:{type(exc).__name__}")
        if not content or not str(content).strip():
            return GenerationResult(usage=usage, reason="empty_output")
        return GenerationResult(text=str(content).strip(), usage=usage)


def save_markdown(text, path):
    """把生成文本写为 Markdown 文件，返回是否成功。"""
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text or "")
        return True
    except OSError as e:
        logger.warning("导出 Markdown 失败: %s", e)
        return False
