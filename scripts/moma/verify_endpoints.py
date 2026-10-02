"""MoMA / 振泽 端点探测（M0 API 验证，可复跑）

用法：
    set MOMA_API_KEY=...        # Windows
    python scripts/moma/verify_endpoints.py

结论记录见 docs/m0_api_verification.md。WS 探测依赖 websockets，缺失时自动跳过。
"""

import os
import time

import httpx

KEY = os.environ.get("MOMA_API_KEY")
ZHENZE = "https://zhenze-huhehaote.cmecloud.cn/v1"
MOMA = "https://moma.cmecloud.cn"
H = {"Authorization": "Bearer {}".format(KEY), "Content-Type": "application/json"}


def probe(label, method, url, headers=None, json_body=None, timeout=30):
    t = time.time()
    try:
        r = httpx.request(method, url, headers=headers or H, json=json_body, timeout=timeout)
        dt = round((time.time() - t) * 1000)
        body = r.text.replace("\n", " ")[:220]
        print("[{}] {} {}ms  {}".format(label, r.status_code, dt, body))
    except Exception as e:  # noqa: BLE001
        print("[{}] ERROR: {}: {}".format(label, type(e).__name__, e))


def main():
    if not KEY:
        raise SystemExit("请先设置环境变量 MOMA_API_KEY")

    print("=== LLM（zhenze chat/completions）===")
    probe("fast", "POST", ZHENZE + "/chat/completions", json_body={
        "model": "deepseek-v4.1-flash",
        "messages": [{"role": "user", "content": "复验"}],
        "max_tokens": 512,
    })
    probe("smart", "POST", ZHENZE + "/chat/completions", json_body={
        "model": "DeepSeek-V3.2",
        "messages": [{"role": "user", "content": "复验"}],
        "max_tokens": 256,
    })
    probe("models", "GET", ZHENZE + "/models", json_body=None)

    print("=== ASR / TTS（moma，预期 404）===")
    probe("asr", "POST", MOMA + "/api/v1/services/audio/asr/transcription",
          headers={**H, "X-DashScope-Async": "enable"},
          json_body={"model": "fun-asr", "input": {"file_urls": []}})
    probe("tts", "POST", MOMA + "/api/v1/services/aigc/multimodal-generation/generation",
          json_body={"model": "qwen3-tts-flash", "input": {"text": "复验"}})

    print("=== moma 模型广场（预期 402 subscription_required）===")
    probe("moma chat", "POST", MOMA + "/v1/chat/completions", json_body={
        "model": "DeepSeek-V3.2",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 8,
    })

    print("=== WS 语音端点（真实握手，预期 404）===")
    try:
        import asyncio

        import websockets
    except ImportError:
        print("[ws] 跳过：未安装 websockets")
        return

    async def try_ws(url):
        try:
            try:
                conn = websockets.connect(url, additional_headers=H, open_timeout=10)
            except TypeError:
                conn = websockets.connect(url, extra_headers=H, open_timeout=10)
            async with conn:
                print("[ws] CONNECTED", url)
        except Exception as e:  # noqa: BLE001
            print("[ws] FAIL", url, type(e).__name__, str(e)[:120])

    for url in (
        "wss://moma.cmecloud.cn/api-ws/v1/inference",
        "wss://zhenze-huhehaote.cmecloud.cn/api-ws/v1/inference",
    ):
        asyncio.run(try_ws(url))


if __name__ == "__main__":
    main()
