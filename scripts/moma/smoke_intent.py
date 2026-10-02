"""llm_intent 真实冒烟（M3）— 用真实 MoMA Key 验证自然语言 → action 映射

用法：
    set MOMA_API_KEY=...
    python scripts/moma/smoke_intent.py
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "app"))

from services.cloud import llm_intent  # noqa: E402
from services.cloud.llm_client import LLMClient  # noqa: E402


class _Cfg:
    def __init__(self, cloud):
        self._cloud = cloud

    def get(self, key, default=None):
        return self._cloud if key == "cloud" else default


CASES = [
    ("下一页", "presentation", "固定指令"),
    ("帮我把上一页再放一遍", "presentation", "自然语言泛化"),
    ("切到板书模式", "mouse", "模式切换"),
    ("帮我把屏幕上的字都擦掉", "draw", "泛化→清屏"),
    ("声音调大一点", None, "应拒绝（无此能力）"),
    ("今天天气不错", None, "应拒绝（闲聊）"),
]


def main():
    if not os.environ.get("MOMA_API_KEY"):
        raise SystemExit("请先设置环境变量 MOMA_API_KEY")
    client = LLMClient(_Cfg({"enabled": True, "provider": "moma"}))
    print("configured:", client.is_configured(),
          "| actions:", len(llm_intent.allowed_actions()))
    for text, mode, note in CASES:
        result = llm_intent.parse_intent(text, client, mode=mode)
        print("[{}] {!r} mode={} -> {}".format(note, text, mode, result))
    client.close()


if __name__ == "__main__":
    main()
