"""LLM 意图解析（M3 慢通道）— 自然语言文本 → 结构化 action

背景：MoMA 免费包内不含云端 ASR（见 docs/m0_api_verification.md），慢通道的
「语音 → 文本」由本地 SenseVoice 完成；本模块只负责「文本 → action」的语义
解析，调用 zhenze LLM 的快通道（默认 deepseek-v4.1-flash）。

约束：

- action 白名单由 voice_actions 推导（固定指令词映射 + 手势专属补充），
  已知当前模式时按 MODE_KEYWORDS 收窄到该模式可用动作，杜绝越权 / 越模式；
- 要求模型输出 JSON：``{"action": <name|null>, "args": {...}, "confidence": <0..1>}``；
- 解析后二次校验：JSON 非法 / action 不在白名单 / confidence 低于阈值 → 拒绝；
- 纯逻辑、无音频依赖，可单测。
"""

import json
import logging
import re

from services.voice_actions import (
    EXTRA_VOICE_ACTIONS,
    MODE_KEYWORDS,
    VOICE_KEYWORD_TO_ACTION,
)

logger = logging.getLogger(__name__)

DEFAULT_CONFIDENCE_THRESHOLD = 0.7


def allowed_actions(mode=None):
    """可被 LLM 解析的 action 白名单（冻结集合）。

    mode 给定且出现在 MODE_KEYWORDS 时，收窄到该模式可用关键词映射的动作；
    EXTRA_VOICE_ACTIONS（挂断、切换应用等全局动作）不受模式限制。
    未知 / 未提供 mode 返回全局白名单。
    """
    actions = set(VOICE_KEYWORD_TO_ACTION.values())
    if mode in MODE_KEYWORDS:
        actions = {
            VOICE_KEYWORD_TO_ACTION[keyword]
            for keyword in MODE_KEYWORDS[mode]
            if keyword in VOICE_KEYWORD_TO_ACTION
        }
    return frozenset(actions | set(EXTRA_VOICE_ACTIONS))


def action_examples():
    """action → 典型说法列表，用于生成 system prompt。"""
    examples = {}
    for keyword, action in VOICE_KEYWORD_TO_ACTION.items():
        examples.setdefault(action, []).append(keyword)
    for action, phrases in EXTRA_VOICE_ACTIONS.items():
        examples.setdefault(action, []).extend(phrases)
    return examples


def build_system_prompt(mode=None):
    """构造意图解析的 system prompt（内置 action schema，按模式收窄）。"""
    examples = action_examples()
    allowed = allowed_actions(mode)
    lines = [
        "你是课堂隔空交互助手的语音指令解析器。",
        "把用户的一句话映射到下列动作之一，只输出 JSON，不要任何解释。",
        "",
        "可用动作（action — 典型说法）：",
    ]
    for action in sorted(examples):
        if action in allowed:
            lines.append("- {} — {}".format(action, "、".join(examples[action])))
    lines += [
        "",
        '输出格式：{"action": "<动作名或 null>", "args": {}, "confidence": <0~1的小数>}',
        "无法确定时 action 置 null、confidence 置 0。",
    ]
    if mode:
        lines.append("当前模式：{}，已只列出该模式下可用的动作。".format(mode))
    return "\n".join(lines)


class IntentResult:
    """意图解析结果。``action`` 为 None 表示拒绝执行。"""

    def __init__(self, action=None, args=None, confidence=0.0, raw=None,
                 reason=None, candidate=None):
        self.action = action
        self.args = args or {}
        self.confidence = float(confidence or 0.0)
        self.raw = raw
        self.reason = reason
        self.candidate = candidate

    @property
    def accepted(self):
        return self.action is not None

    def __repr__(self):
        return (
            "IntentResult(action={!r}, args={!r}, confidence={:.2f}, "
            "reason={!r}, candidate={!r})".format(
                self.action, self.args, self.confidence, self.reason, self.candidate
            )
        )


_FENCE_RE = re.compile(r"^```[a-zA-Z0-9]*\s*(.*?)\s*```$", re.DOTALL)
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text):
    """从模型输出中提取 JSON 对象（容忍 ``` 围栏与前后杂讯）。"""
    if not text:
        return None
    candidate = text.strip()
    fence = _FENCE_RE.match(candidate)
    if fence:
        candidate = fence.group(1).strip()
    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else None
    except (ValueError, TypeError):
        pass
    match = _JSON_RE.search(candidate)
    if match:
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else None
        except (ValueError, TypeError):
            return None
    return None


def parse_intent(text, llm_client, mode=None, threshold=DEFAULT_CONFIDENCE_THRESHOLD):
    """把一句自然语言解析为 :class:`IntentResult`。

    Args:
        text: 本地 SenseVoice 转写文本。
        llm_client: ``LLMClient`` 实例；未配置时直接拒绝，不发起网络请求。
        mode: 当前交互模式（presentation / mouse / draw），用于提示模型。
        threshold: confidence 阈值，低于该值则拒绝。
    """
    if not text or not str(text).strip():
        return IntentResult(reason="empty_input")
    if llm_client is None or not llm_client.is_configured():
        return IntentResult(reason="cloud_not_configured")

    messages = [
        {"role": "system", "content": build_system_prompt(mode)},
        {"role": "user", "content": str(text).strip()},
    ]
    try:
        # 推理型模型先吐 reasoning，max_tokens 给足以保证 content 非空
        content, _usage = llm_client.call_fast(
            messages, max_tokens=1024, temperature=0
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("意图解析 LLM 调用失败: %s", exc)
        return IntentResult(reason="llm_error:{}".format(type(exc).__name__))

    parsed = _extract_json(content)
    if parsed is None:
        return IntentResult(raw=content, reason="invalid_json")

    action = parsed.get("action")
    if action in (None, "", "null"):
        return IntentResult(raw=content, reason="no_action")

    if action not in allowed_actions(mode):
        return IntentResult(raw=content, reason="action_not_allowed",
                            candidate=action)

    try:
        confidence = float(parsed.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0

    if confidence < threshold:
        return IntentResult(raw=content, reason="low_confidence",
                            confidence=confidence, candidate=action)

    args = parsed.get("args")
    if not isinstance(args, dict):
        args = {}
    return IntentResult(action=action, args=args, confidence=confidence,
                        raw=content, reason="ok")
