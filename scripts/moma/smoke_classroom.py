"""classroom_ai 真实冒烟（M4）— 用真实 MoMA Key 生成课堂小结与随堂题

用法：
    set MOMA_API_KEY=...
    python scripts/moma/smoke_classroom.py
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "app"))

from services.classroom_ai import ClassroomAI  # noqa: E402
from services.cloud.llm_client import LLMClient  # noqa: E402


class _Cfg:
    def __init__(self, cloud):
        self._cloud = cloud

    def get(self, key, default=None):
        return self._cloud if key == "cloud" else default


def main():
    if not os.environ.get("MOMA_API_KEY"):
        raise SystemExit("请先设置环境变量 MOMA_API_KEY")
    client = LLMClient(_Cfg({"enabled": True, "provider": "moma"}))
    ai = ClassroomAI(client, outline="第三章 勾股定理：直角三角形三边关系、逆定理、应用")
    for line in (
        "直角三角形两直角边平方和等于斜边平方",
        "a²+b²=c²，其中 c 是斜边",
        "逆定理：若三角形三边满足 a²+b²=c²，则为直角三角形",
        "常见勾股数：3,4,5 / 5,12,13 / 8,15,17",
    ):
        ai.append_transcript(line)

    summary = ai.generate_summary()
    print("=== 课堂小结 ===  reason={} usage={}".format(summary.reason, summary.usage))
    print(summary.text)

    quiz = ai.generate_quiz(count=3)
    print("\n=== 随堂出题 ===  reason={} usage={}".format(quiz.reason, quiz.usage))
    print(quiz.text)
    client.close()


if __name__ == "__main__":
    main()
