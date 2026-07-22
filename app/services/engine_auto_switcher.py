"""远距引擎自动切换状态机 — mediapipe ↔ hagrid_yolo 运行时切换。

背景（3–5 米远距实测结论，勿推翻）：
  - mediapipe：近距（1.5–2m）检出 ~99%、延迟低（P95 35ms），
    但 3–5 米远距检出率只有 42.7%
  - hagrid_yolo：远距检出 92.8%，但延迟高 2.5 倍（P95 81ms）、
    抖动约 4 倍、单人场景多手率 58%（conf=0.25 下有误检）

策略：不设全局默认引擎，运行时按"是否检测到手"自动切换：
  - mediapipe 连续 N 帧无手   → 切 hagrid_yolo（远处有人但 MP 检不到）
  - hagrid_yolo 连续 M 帧有手 → 切回 mediapipe（人走近了，回到低延迟引擎）
  - 切换后冷却 C 秒，期间不计帧（新引擎预热 + 防抖）
  - YOLO 误检约束：≥2 只手的帧不计入"有手"计帧（见 counts_toward_hand_streak）

切回判据用"YOLO 持续有手"而不是"YOLO 无手"：无手时无法区分
"人走了"还是"距离合适但 MP 检不出"，只有持续有手才说明人近在眼前。

本模块是纯算法、无 Qt 依赖，由 orchestrator 在主流程逐帧喂检测结果；
replay_video.py / benchmark 脚本不经过这里，行为不受影响。
"""

import logging
import time

logger = logging.getLogger(__name__)

ENGINE_MEDIAPIPE = "mediapipe"
ENGINE_HAGRID_YOLO = "hagrid_yolo"


def counts_toward_hand_streak(hand_count):
    """"有手"计帧的多手约束：恰好 1 只手才计入连续有手帧数。

    YOLO 远距单人场景多手率 58%（conf=0.25 误检），≥2 只手的帧视为疑似
    误检：既不计入、也不清零——计入会让误检帧触发切回；清零会让误检帧
    淹没真实单手帧（58% 占比下永远凑不够切回阈值）。
    """
    return hand_count == 1


class EngineAutoSwitcher:
    """引擎自动切换 FSM。

    状态由"当前引擎"隐式给出（调用方每帧传入），本类只维护计帧与冷却：
      - mediapipe 态：累计连续无手帧数，达 no_hand_frames → 建议 hagrid_yolo
      - hagrid_yolo 态：累计连续单手帧数，达 hand_frames → 建议 mediapipe
      - 任意切换后冷却 cooldown_sec 秒，期间不计帧、不切换

    update() 返回建议切换到的引擎名，或 None 表示不切换。
    """

    def __init__(self, enabled=False, no_hand_frames=60, hand_frames=90,
                 cooldown_sec=5.0, clock=None):
        self.enabled = bool(enabled)
        self.no_hand_frames = max(1, int(no_hand_frames))
        self.hand_frames = max(1, int(hand_frames))
        self.cooldown_sec = max(0.0, float(cooldown_sec))
        self._clock = clock or time.monotonic
        self._no_hand_streak = 0
        self._hand_streak = 0
        self._last_switch_time = None
        self.last_reason = ""

    def configure(self, *, enabled=None, no_hand_frames=None, hand_frames=None,
                  cooldown_sec=None):
        """更新参数（config 热加载）。参数变更即重置计帧与冷却，避免旧证据
        在新阈值下产生突兀切换。"""
        if enabled is not None:
            self.enabled = bool(enabled)
        if no_hand_frames is not None:
            self.no_hand_frames = max(1, int(no_hand_frames))
        if hand_frames is not None:
            self.hand_frames = max(1, int(hand_frames))
        if cooldown_sec is not None:
            self.cooldown_sec = max(0.0, float(cooldown_sec))
        self.reset()

    def reset(self):
        """清空计帧与冷却。用户手动切换引擎、参数变更时调用。"""
        self._no_hand_streak = 0
        self._hand_streak = 0
        self._last_switch_time = None
        self.last_reason = ""

    def update(self, current_engine, hand_count, now=None):
        """喂一帧的检出结果，返回应切换到的引擎名或 None。

        Args:
            current_engine: 当前实际运行的引擎名
            hand_count:     本帧检出的手数量（0/1/2...）
            now:            可选时间戳（测试注入），默认 self._clock()
        """
        if not self.enabled:
            return None
        now = self._clock() if now is None else now
        hand_count = max(0, int(hand_count))

        # 冷却期：不计帧（新引擎预热 + 防抖），直接返回
        if (self._last_switch_time is not None
                and now - self._last_switch_time < self.cooldown_sec):
            self._no_hand_streak = 0
            self._hand_streak = 0
            return None

        if current_engine == ENGINE_MEDIAPIPE:
            self._hand_streak = 0
            if hand_count == 0:
                self._no_hand_streak += 1
            else:
                self._no_hand_streak = 0
            if self._no_hand_streak >= self.no_hand_frames:
                self._mark_switch(now, f"mediapipe 连续 {self._no_hand_streak} 帧未检到手")
                return ENGINE_HAGRID_YOLO
        elif current_engine == ENGINE_HAGRID_YOLO:
            self._no_hand_streak = 0
            if hand_count == 0:
                self._hand_streak = 0
            elif counts_toward_hand_streak(hand_count):
                self._hand_streak += 1
            # ≥2 只手：疑似误检帧，不计入也不清零
            if self._hand_streak >= self.hand_frames:
                self._mark_switch(now, f"hagrid_yolo 连续 {self._hand_streak} 帧稳定单手检出")
                return ENGINE_MEDIAPIPE
        else:
            # 未知引擎（未来扩展或配置错误）：不介入并清空状态
            self.reset()
        return None

    def _mark_switch(self, now, reason):
        self._last_switch_time = now
        self._no_hand_streak = 0
        self._hand_streak = 0
        self.last_reason = reason
        logger.info("引擎自动切换触发: %s", reason)
