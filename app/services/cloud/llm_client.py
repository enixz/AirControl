"""LLM 路由客户端 — 按用途（fast/smart/vision）路由到配置的模型

从 ConfigManager 读取 cloud 节构造；模型名由 providers.resolve_provider()
按预设/用户配置解析。未启用或未配置（缺 Key）时调用抛 CloudNotConfiguredError。
"""

import logging

from services.cloud import providers
from services.cloud.moma_client import MoMAClient
from services.cloud.token_meter import TokenMeter

logger = logging.getLogger(__name__)


class CloudNotConfiguredError(RuntimeError):
    """云端未启用或配置不完整（缺 base_url/API Key）。"""


class LLMClient:
    """云端 LLM 统一入口。

    Args:
        config: ConfigManager 实例（读取 "cloud" 节）
        token_meter: 可选 TokenMeter；默认新建（写 logs/token_usage.jsonl）
    """

    def __init__(self, config, token_meter=None):
        self._resolved = providers.resolve_provider(config.get("cloud") or {})
        self._token_meter = token_meter if token_meter is not None else TokenMeter()
        self._client = None
        if self._resolved["enabled"] and self._resolved["base_url"] \
                and self._resolved["api_key"]:
            self._client = MoMAClient(
                self._resolved["base_url"],
                self._resolved["api_key"],
                token_meter=self._token_meter,
            )
        elif self._resolved["enabled"]:
            logger.warning("云端已启用但配置不完整（缺 base_url 或 API Key）")

    def is_configured(self):
        """云端已启用且凭据齐备。"""
        return self._client is not None

    @property
    def token_meter(self):
        return self._token_meter

    def close(self):
        if self._client is not None:
            self._client.close()
            self._client = None

    def call_fast(self, messages, **kwargs):
        """快通道（指令理解，默认 deepseek-v4.1-flash）。

        推理型模型，先吐 reasoning，max_tokens 需给足（≥512），默认 512。
        """
        kwargs.setdefault("max_tokens", 512)
        return self._call("llm_fast_model", messages, **kwargs)

    def call_smart(self, messages, **kwargs):
        """强通道（内容生成，默认 DeepSeek-V3.2）。"""
        kwargs.setdefault("max_tokens", 2048)
        return self._call("llm_smart_model", messages, **kwargs)

    def call_vision(self, messages, **kwargs):
        """视觉通道（板书成文，默认 Qwen2.5-VL-72B-Instruct）。"""
        kwargs.setdefault("max_tokens", 2048)
        return self._call("llm_vision_model", messages, **kwargs)

    def _call(self, model_field, messages, **kwargs):
        if self._client is None:
            raise CloudNotConfiguredError(
                "云端未启用或未配置（cloud.enabled=false 或缺少 API Key）"
            )
        model = self._resolved.get(model_field)
        if not model:
            raise CloudNotConfiguredError(f"未配置模型字段: {model_field}")
        return self._client.chat_completions(model, messages, **kwargs)
