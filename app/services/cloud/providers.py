"""云端提供商预设与解析（OpenAI 兼容 API）

预设表 PROVIDERS 定义各提供商的默认端点与模型；config.json 的 cloud 节
字段为 null 时回落到预设默认值。API Key 只从环境变量读取（变量名由
api_key_env 指定），绝不写入 config.json 明文。

provider="custom" 时 base_url/api_key_env/三个模型字段全部由用户配置提供，
缺失则视为"未配置"（is_configured() 返回 False）。
"""

import os

PROVIDERS = {
    # 移动云 MoMA（湛卢比赛端点），模型选型见 docs/moma_usable_models.md
    "moma": {
        "base_url": "https://zhenze-huhehaote.cmecloud.cn/v1",
        "api_key_env": "MOMA_API_KEY",
        "llm_fast_model": "deepseek-v4.1-flash",
        "llm_smart_model": "DeepSeek-V3.2",
        "llm_vision_model": "Qwen2.5-VL-72B-Instruct",
    },
    # 自定义 OpenAI 兼容端点：所有字段必填，不设默认
    "custom": {
        "base_url": None,
        "api_key_env": None,
        "llm_fast_model": None,
        "llm_smart_model": None,
        "llm_vision_model": None,
    },
}

_RESOLVED_FIELDS = (
    "base_url",
    "api_key_env",
    "llm_fast_model",
    "llm_smart_model",
    "llm_vision_model",
)


def resolve_provider(cloud_cfg):
    """把 config.json 的 cloud 节解析成完整的运行时配置。

    Args:
        cloud_cfg: dict，ConfigManager 的 "cloud" 节内容

    Returns:
        dict，含 enabled/provider/auto_fallback/health_check_interval_sec
        及解析后的 base_url/api_key_env/三个模型字段、api_key（来自环境变量，
        未设置时为 None）。未知 provider 按 custom 处理（全部要求显式配置）。
    """
    cloud_cfg = cloud_cfg or {}
    preset = PROVIDERS.get(cloud_cfg.get("provider")) or PROVIDERS["custom"]

    resolved = {
        "enabled": bool(cloud_cfg.get("enabled", False)),
        "provider": cloud_cfg.get("provider") or "custom",
        "auto_fallback": bool(cloud_cfg.get("auto_fallback", True)),
        "health_check_interval_sec": cloud_cfg.get("health_check_interval_sec") or 60,
    }
    for field in _RESOLVED_FIELDS:
        resolved[field] = cloud_cfg.get(field) or preset.get(field)
    resolved["api_key"] = _read_api_key(resolved["api_key_env"])
    return resolved


def is_configured(cloud_cfg):
    """是否已可用：启用 且 base_url 与 API Key 齐备。"""
    resolved = resolve_provider(cloud_cfg)
    return bool(
        resolved["enabled"] and resolved["base_url"] and resolved["api_key"]
    )


def _read_api_key(api_key_env):
    if not api_key_env:
        return None
    return os.environ.get(api_key_env) or None
