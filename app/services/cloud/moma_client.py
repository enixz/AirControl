"""低层云端 HTTP 客户端 — 提供商无关的 OpenAI 兼容 chat/completions 封装

模块名保留 moma_client（对应 MoMA 预设），实现本身只依赖 OpenAI 兼容协议，
base_url/api_key 由 providers.resolve_provider() 解析后注入。

重试策略：429 / 5xx / 网络错误指数退避最多 3 次；其余 4xx 直接抛。
429 属分钟级 token 限流，退避优先遵循 Retry-After 头，无该头时按分钟级步长放大。
每次调用（含失败）都会记录到 TokenMeter。
"""

import logging
import time

import httpx

logger = logging.getLogger(__name__)

_CONNECT_TIMEOUT_SEC = 5.0
_READ_TIMEOUT_SEC = 30.0
_MAX_RETRIES = 3
_BACKOFF_BASE_SEC = 1.0
# 429 属分钟级 token 限流（免费档 3000 token/分钟）：秒级退避在限流窗口内
# 几乎必然再次 429，无 Retry-After 头时改用分钟级退避步长
_RATE_LIMIT_BACKOFF_SEC = 15.0
_MAX_RETRY_AFTER_SEC = 60.0


def _parse_retry_after(value):
    """解析 Retry-After 头（仅支持秒数格式），非法值返回 None。"""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class CloudAPIError(Exception):
    """云端 API 调用失败（含 HTTP status_code，网络错误时为 None）。"""

    def __init__(self, message, status_code=None):
        super().__init__(message)
        self.status_code = status_code


class CloudAuthError(CloudAPIError):
    """鉴权失败（401/403），不重试。"""


class CloudRateLimitError(CloudAPIError):
    """限流（429）且重试耗尽。"""


class MoMAClient:
    """OpenAI 兼容 chat/completions 客户端。

    Args:
        base_url: OpenAI 兼容端点根（如 https://xxx/v1）
        api_key: Bearer Key（由调用方从环境变量读入）
        token_meter: 可选 TokenMeter，记录每次调用用量
        max_retries: 可重试错误的最大尝试次数
    """

    def __init__(self, base_url, api_key, token_meter=None, max_retries=_MAX_RETRIES):
        if not base_url:
            raise ValueError("base_url 不能为空")
        if not api_key:
            raise ValueError("api_key 不能为空（从 api_key_env 指定的环境变量读取）")
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._token_meter = token_meter
        self._max_retries = max(1, int(max_retries))
        self._client = httpx.Client(
            timeout=httpx.Timeout(_READ_TIMEOUT_SEC, connect=_CONNECT_TIMEOUT_SEC),
            headers={"Authorization": f"Bearer {api_key}"},
        )

    def close(self):
        self._client.close()

    def chat_completions(self, model, messages, max_tokens=1024, temperature=0.7):
        """调用 chat/completions，返回 (content, usage_dict)。

        Raises:
            CloudAuthError: 401/403
            CloudRateLimitError: 429 且重试耗尽
            CloudAPIError: 其余 HTTP 错误、网络错误、响应格式异常
        """
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        url = f"{self._base_url}/chat/completions"

        last_error = None
        for attempt in range(1, self._max_retries + 1):
            started = time.monotonic()
            try:
                response = self._client.post(url, json=payload)
            except httpx.HTTPError as e:
                latency_ms = (time.monotonic() - started) * 1000
                self._record(model, None, latency_ms, success=False,
                             error=f"network:{e.__class__.__name__}")
                last_error = CloudAPIError(f"网络错误: {e}")
                if attempt < self._max_retries:
                    self._sleep_backoff(attempt)
                    continue
                raise last_error from e

            latency_ms = (time.monotonic() - started) * 1000
            status = response.status_code

            if status == 200:
                try:
                    data = response.json()
                    choice = data["choices"][0]
                    content = choice["message"].get("content") or ""
                    usage = data.get("usage") or {}
                except (ValueError, KeyError, IndexError, TypeError) as e:
                    self._record(model, None, latency_ms, success=False,
                                 error=f"bad_response:{e.__class__.__name__}")
                    raise CloudAPIError(f"响应格式异常: {e}", status_code=200) from e
                self._record(model, usage, latency_ms, success=True)
                return content, usage

            error_text = response.text[:200]
            if status in (401, 403):
                self._record(model, None, latency_ms, success=False,
                             error=f"http_{status}")
                raise CloudAuthError(
                    f"鉴权失败 (HTTP {status}): {error_text}", status_code=status
                )

            retryable = status == 429 or status >= 500
            self._record(model, None, latency_ms, success=False,
                         error=f"http_{status}")
            if retryable and attempt < self._max_retries:
                self._sleep_backoff(attempt, response=response)
                continue
            if status == 429:
                raise CloudRateLimitError(
                    f"限流 (HTTP 429) 重试 {self._max_retries} 次后仍失败",
                    status_code=status,
                )
            raise CloudAPIError(
                f"云端调用失败 (HTTP {status}): {error_text}", status_code=status
            )

        # 循环必然在 return 或 raise 结束，此处仅防御
        raise last_error or CloudAPIError("云端调用失败")

    def _sleep_backoff(self, attempt, response=None):
        """退避等待：429 优先遵循 Retry-After（分钟级限流），其余秒级指数退避。"""
        delay = _BACKOFF_BASE_SEC * (2 ** (attempt - 1))
        if getattr(response, "status_code", None) == 429:
            retry_after = _parse_retry_after(
                response.headers.get("Retry-After")
            )
            if retry_after is not None:
                delay = max(delay, min(retry_after, _MAX_RETRY_AFTER_SEC))
            else:
                delay = max(
                    delay, _RATE_LIMIT_BACKOFF_SEC * (2 ** (attempt - 1))
                )
        logger.warning("云端调用失败，%.1fs 后第 %d 次重试", delay, attempt + 1)
        time.sleep(delay)

    def _record(self, model, usage, latency_ms, success, error=None):
        if self._token_meter is None:
            return
        try:
            self._token_meter.record(
                model,
                (usage or {}).get("prompt_tokens", 0),
                (usage or {}).get("completion_tokens", 0),
                latency_ms,
                success,
                error=error,
            )
        except Exception:
            logger.warning("token 计量记录异常", exc_info=True)
