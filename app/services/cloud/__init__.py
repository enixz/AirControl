"""云端接入层 — OpenAI 兼容 LLM API（默认移动云 MoMA，提供商可配置）

ASR/TTS 接口位已删除（MoMA 免费包实测不含云端语音，见
docs/m0_api_verification.md；历史版本可从 git 恢复）。听写/播报
保留本地方案（SenseVoice + Windows SAPI）。
"""

from services.cloud import providers
from services.cloud.cloud_health import CloudHealthMonitor
from services.cloud.llm_client import CloudNotConfiguredError, LLMClient
from services.cloud.llm_intent import IntentResult, parse_intent
from services.cloud.moma_client import (
    CloudAPIError,
    CloudAuthError,
    CloudRateLimitError,
    MoMAClient,
)
from services.cloud.token_meter import TokenMeter

__all__ = [
    "CloudAPIError",
    "CloudAuthError",
    "CloudHealthMonitor",
    "CloudNotConfiguredError",
    "CloudRateLimitError",
    "IntentResult",
    "LLMClient",
    "MoMAClient",
    "TokenMeter",
    "parse_intent",
    "providers",
]
