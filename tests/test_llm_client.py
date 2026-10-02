"""LLMClient 路由与重试测试：mock httpx，严禁真实网络。"""
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'app'))

import httpx  # noqa: E402
from services.cloud.llm_client import CloudNotConfiguredError, LLMClient  # noqa: E402
from services.cloud.moma_client import (  # noqa: E402
    CloudAPIError,
    CloudAuthError,
    CloudRateLimitError,
)


def _make_config(cloud):
    config = mock.MagicMock()
    config.get.side_effect = lambda key, default=None: cloud if key == "cloud" else default
    return config


def _ok_response(content="OK", prompt_tokens=3, completion_tokens=1):
    response = mock.MagicMock()
    response.status_code = 200
    response.json.return_value = {
        "choices": [{"message": {"content": content}}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
    return response


def _error_response(status, text="error"):
    response = mock.MagicMock()
    response.status_code = status
    response.text = text
    return response


class _LLMTestBase(unittest.TestCase):
    cloud_cfg = {
        "enabled": True,
        "provider": "moma",
        "api_key_env": None,
        "base_url": None,
        "llm_fast_model": None,
        "llm_smart_model": None,
        "llm_vision_model": None,
        "auto_fallback": True,
        "health_check_interval_sec": 60,
    }

    def setUp(self):
        env_patcher = mock.patch.dict(os.environ, {"MOMA_API_KEY": "k-test"})
        env_patcher.start()
        self.addCleanup(env_patcher.stop)
        client_patcher = mock.patch("services.cloud.moma_client.httpx.Client")
        self.mock_client_cls = client_patcher.start()
        self.addCleanup(client_patcher.stop)
        self.mock_http = self.mock_client_cls.return_value
        sleep_patcher = mock.patch("services.cloud.moma_client.time.sleep")
        self.mock_sleep = sleep_patcher.start()
        self.addCleanup(sleep_patcher.stop)
        self.token_meter = mock.MagicMock()
        self.client = LLMClient(_make_config(dict(self.cloud_cfg)),
                                token_meter=self.token_meter)

    def _last_payload(self):
        return self.mock_http.post.call_args.kwargs["json"]


class TestLLMRouting(_LLMTestBase):
    def test_routes_to_preset_models(self):
        """fast/smart/vision 分别路由到预设模型名。"""
        self.mock_http.post.return_value = _ok_response()
        messages = [{"role": "user", "content": "hi"}]

        self.client.call_fast(messages)
        self.assertEqual(self._last_payload()["model"], "deepseek-v4.1-flash")
        self.assertGreaterEqual(self._last_payload()["max_tokens"], 512)

        self.client.call_smart(messages)
        self.assertEqual(self._last_payload()["model"], "DeepSeek-V3.2")

        self.client.call_vision(messages)
        self.assertEqual(self._last_payload()["model"], "Qwen2.5-VL-72B-Instruct")

    def test_returns_content_and_usage(self):
        self.mock_http.post.return_value = _ok_response(content="你好", prompt_tokens=5,
                                                        completion_tokens=7)
        content, usage = self.client.call_smart([{"role": "user", "content": "hi"}])
        self.assertEqual(content, "你好")
        self.assertEqual(usage["total_tokens"], 12)

    def test_token_recorded_on_success(self):
        self.mock_http.post.return_value = _ok_response(prompt_tokens=5, completion_tokens=7)
        self.client.call_fast([{"role": "user", "content": "hi"}])
        self.token_meter.record.assert_called_once()
        args = self.token_meter.record.call_args.args
        self.assertEqual(args[0], "deepseek-v4.1-flash")
        self.assertEqual(args[1], 5)
        self.assertEqual(args[2], 7)
        self.assertTrue(args[4])


class TestLLMRetry(_LLMTestBase):
    def test_429_retried_then_success(self):
        """429 可重试：前两次 429、第三次 200，最终成功。"""
        self.mock_http.post.side_effect = [
            _error_response(429),
            _error_response(429),
            _ok_response(),
        ]
        content, _ = self.client.call_fast([{"role": "user", "content": "hi"}])
        self.assertEqual(content, "OK")
        self.assertEqual(self.mock_http.post.call_count, 3)

    def test_429_respects_retry_after_header(self):
        """429 时优先遵循 Retry-After 头。"""
        resp = _error_response(429)
        resp.headers = {"Retry-After": "30"}
        self.mock_http.post.side_effect = [resp, _ok_response()]
        content, _ = self.client.call_fast([{"role": "user", "content": "hi"}])
        self.assertEqual(content, "OK")
        self.assertEqual(self.mock_sleep.call_args.args[0], 30)

    def test_429_retry_after_capped(self):
        """Retry-After 超大时封顶 60s，避免服务端异常值卡死。"""
        resp = _error_response(429)
        resp.headers = {"Retry-After": "600"}
        self.mock_http.post.side_effect = [resp, _ok_response()]
        self.client.call_fast([{"role": "user", "content": "hi"}])
        self.assertEqual(self.mock_sleep.call_args.args[0], 60)

    def test_429_without_retry_after_uses_minute_level_backoff(self):
        """429 无 Retry-After 时按分钟级步长退避（15s 起而非 1s）。"""
        resp = _error_response(429)
        resp.headers = {}
        self.mock_http.post.side_effect = [resp, _ok_response()]
        self.client.call_fast([{"role": "user", "content": "hi"}])
        self.assertEqual(self.mock_sleep.call_args.args[0], 15)

    def test_429_exhausted_raises_rate_limit(self):
        """429 重试耗尽后抛 CloudRateLimitError。"""
        self.mock_http.post.side_effect = [_error_response(429)] * 3
        with self.assertRaises(CloudRateLimitError):
            self.client.call_fast([{"role": "user", "content": "hi"}])
        self.assertEqual(self.mock_http.post.call_count, 3)

    def test_5xx_retried(self):
        self.mock_http.post.side_effect = [_error_response(503), _ok_response()]
        self.client.call_fast([{"role": "user", "content": "hi"}])
        self.assertEqual(self.mock_http.post.call_count, 2)

    def test_4xx_not_retried(self):
        """400 等非 429 的 4xx 直接抛，不重试。"""
        self.mock_http.post.return_value = _error_response(400, "bad request")
        with self.assertRaises(CloudAPIError) as ctx:
            self.client.call_fast([{"role": "user", "content": "hi"}])
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(self.mock_http.post.call_count, 1)

    def test_401_raises_auth_error(self):
        self.mock_http.post.return_value = _error_response(401, "unauthorized")
        with self.assertRaises(CloudAuthError):
            self.client.call_fast([{"role": "user", "content": "hi"}])
        self.assertEqual(self.mock_http.post.call_count, 1)

    def test_network_error_retried(self):
        self.mock_http.post.side_effect = [
            httpx.ConnectError("boom"),
            _ok_response(),
        ]
        content, _ = self.client.call_fast([{"role": "user", "content": "hi"}])
        self.assertEqual(content, "OK")
        self.assertEqual(self.mock_http.post.call_count, 2)

    def test_token_recorded_on_failure(self):
        """失败调用也记录到 token_meter（success=False）。"""
        self.mock_http.post.return_value = _error_response(400)
        with self.assertRaises(CloudAPIError):
            self.client.call_fast([{"role": "user", "content": "hi"}])
        self.token_meter.record.assert_called_once()
        self.assertFalse(self.token_meter.record.call_args.args[4])


class TestLLMNotConfigured(unittest.TestCase):
    def test_disabled_raises(self):
        client = LLMClient(_make_config({"enabled": False, "provider": "moma"}))
        self.assertFalse(client.is_configured())
        with self.assertRaises(CloudNotConfiguredError):
            client.call_fast([{"role": "user", "content": "hi"}])

    def test_missing_key_raises(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MOMA_API_KEY", None)
            client = LLMClient(_make_config({"enabled": True, "provider": "moma"}))
        self.assertFalse(client.is_configured())
        with self.assertRaises(CloudNotConfiguredError):
            client.call_smart([{"role": "user", "content": "hi"}])


if __name__ == "__main__":
    unittest.main(verbosity=2)
