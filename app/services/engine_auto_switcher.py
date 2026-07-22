"""远距引擎三态自动切换状态机 — NEAR / CAPTURE / FAR_TRACK 闭环。

背景（3–5 米远距实测结论，勿推翻）：
  - mediapipe 裸检：近距 ~99%、P95 35ms，远距检出率只有 42.7%
  - mediapipe + ZOOM（crop-zoom/超分/人脸引导，long_range_enabled）：
    远距检出 88.1%、多手率 1.4%、抖动 P95 20.6px——锁手后跟踪最强，
    但结构性无法初始捕获（ZOOM ON 以"已检到手"为前提，
    见 base_hand_tracker.py:707 `_update_zoom_mode` 无手直接 return）
  - hagrid_yolo：远距检出 92.8% 且不依赖先验（适合捕获），
    但延迟高（P95 73ms）、抖动大、单人场景多手率 58%（误检）

三态闭环（"丢手时 YOLO 负责抓，抓住后 ZOOM 负责盯"）：
  - NEAR（近距态）：mediapipe + 用户配置档（stable，long_range 关）。
    连续 N 帧无手 → CAPTURE。
  - CAPTURE（捕获态）：hagrid_yolo 全帧找手（不开 zoom，92.8% 已足够）。
    连续 H 帧稳定单手 → FAR_TRACK（而不是切回普通 MP——恒定 5m 场景下
    切回裸 MP 会立刻再丢手，造成来回振荡）。
  - FAR_TRACK（远距跟踪态）：mediapipe + long_range_enabled 运行时覆盖
    （不写 config），ZOOM 接住 YOLO 抓到的手持续跟踪。
      · 手变大/走近（单手且 bbox 占比 ≥ 阈值）持续 M 帧 → NEAR（撤覆盖）
      · 再次连续 N 帧无手 → CAPTURE
  - 任意迁移后冷却 C 秒，期间不计帧（新引擎/新链路预热 + 防抖）。
  - 多手约束：≥2 只手的帧视为疑似误检，不计入也不清零
    （见 counts_toward_hand_streak）。

本模块是纯算法、无 Qt 依赖，由 orchestrator 在主流程逐帧喂检测结果；
replay_video.py / benchmark 脚本不经过这里，行为不受影响。
"""

import logging
import time

logger = logging.getLogger(__name__)

ENGINE_MEDIAPIPE = "mediapipe"
ENGINE_HAGRID_YOLO = "hagrid_yolo"

STATE_NEAR = "near"
STATE_CAPTURE = "capture"
STATE_FAR_TRACK = "far_track"


def counts_toward_hand_streak(hand_count):
    """"有手"计帧的多手约束：恰好 1 只手才计入连续有手帧数。

    YOLO 远距单人场景多手率 58%（conf=0.25 误检），≥2 只手的帧视为疑似
    误检：既不计入、也不清零——计入会让误检帧触发迁移；清零会让误检帧
    淹没真实单手帧（58% 占比下永远凑不够迁移阈值）。
    """
    return hand_count == 1


class EngineAutoSwitcher:
    """远距引擎三态自动切换 FSM。

    状态由本类显式持有（self.state），调用方逐帧喂检出结果：
      - update() 返回应迁移到的目标状态名，或 None 表示留在当前态
      - 迁移判据的帧数/阈值/冷却全部由 configure() 注入（config 热加载）
      - reset() 回到 NEAR 并清空计帧与冷却（手动切引擎/参数变更时调用）
    """

    def __init__(self, enabled=False, no_hand_frames=60, hand_frames=30,
                 near_frames=90, near_bbox_ratio=0.04, cooldown_sec=5.0,
                 clock=None):
        self.enabled = bool(enabled)
        self.no_hand_frames = max(1, int(no_hand_frames))
        self.hand_frames = max(1, int(hand_frames))
        self.near_frames = max(1, int(near_frames))
        self.near_bbox_ratio = max(0.0, float(near_bbox_ratio))
        self.cooldown_sec = max(0.0, float(cooldown_sec))
        self._clock = clock or time.monotonic
        self.state = STATE_NEAR
        self._no_hand_streak = 0
        self._hand_streak = 0
        self._near_streak = 0
        self._last_switch_time = None
        self.last_reason = ""

    def configure(self, *, enabled=None, no_hand_frames=None, hand_frames=None,
                  near_frames=None, near_bbox_ratio=None, cooldown_sec=None):
        """更新参数（config 热加载）。参数变更即重置计帧与冷却，避免旧证据
        在新阈值下产生突兀迁移。状态保持当前态不变。"""
        if enabled is not None:
            self.enabled = bool(enabled)
        if no_hand_frames is not None:
            self.no_hand_frames = max(1, int(no_hand_frames))
        if hand_frames is not None:
            self.hand_frames = max(1, int(hand_frames))
        if near_frames is not None:
            self.near_frames = max(1, int(near_frames))
        if near_bbox_ratio is not None:
            self.near_bbox_ratio = max(0.0, float(near_bbox_ratio))
        if cooldown_sec is not None:
            self.cooldown_sec = max(0.0, float(cooldown_sec))
        self._reset_counters()

    def reset(self, state=STATE_NEAR):
        """清空计帧与冷却并回到指定状态（默认 NEAR）。手动切引擎时调用。"""
        self.state = state
        self._reset_counters()

    def _reset_counters(self):
        self._no_hand_streak = 0
        self._hand_streak = 0
        self._near_streak = 0
        self._last_switch_time = None
        self.last_reason = ""

    def update(self, hand_count, hand_bbox_ratio=0.0, now=None):
        """喂一帧的检出结果，返回应迁移到的目标状态名或 None。

        Args:
            hand_count:      本帧检出的手数量（0/1/2...）
            hand_bbox_ratio: 最大手 bbox 面积占全帧比（无手传 0.0），
                             仅 FAR_TRACK 态用于"手变大/走近"判定
            now:             可选时间戳（测试注入），默认 self._clock()
        """
        if not self.enabled:
            return None
        now = self._clock() if now is None else now
        hand_count = max(0, int(hand_count))
        hand_bbox_ratio = max(0.0, float(hand_bbox_ratio or 0.0))

        # 冷却期：不计帧（新引擎/新链路预热 + 防抖），直接返回
        if (self._last_switch_time is not None
                and now - self._last_switch_time < self.cooldown_sec):
            self._reset_streaks()
            return None

        if self.state == STATE_NEAR:
            return self._update_near(hand_count, now)
        if self.state == STATE_CAPTURE:
            return self._update_capture(hand_count, now)
        if self.state == STATE_FAR_TRACK:
            return self._update_far_track(hand_count, hand_bbox_ratio, now)
        # 未知状态（防御）：回 NEAR
        logger.warning("引擎自动切换 FSM 处于未知状态 %r，重置回 NEAR", self.state)
        self.reset()
        return None

    # ------------------------------------------------------------------
    # 各态迁移逻辑
    # ------------------------------------------------------------------

    def _update_near(self, hand_count, now):
        """NEAR：mediapipe 裸检。连续 N 帧无手 → CAPTURE。"""
        if hand_count == 0:
            self._no_hand_streak += 1
        else:
            self._no_hand_streak = 0
        if self._no_hand_streak >= self.no_hand_frames:
            return self._mark_switch(
                STATE_CAPTURE, now,
                f"NEAR→CAPTURE: mediapipe 连续 {self._no_hand_streak} 帧未检到手",
            )
        return None

    def _update_capture(self, hand_count, now):
        """CAPTURE：hagrid_yolo 全帧捕获。连续 H 帧稳定单手 → FAR_TRACK。"""
        if hand_count == 0:
            self._hand_streak = 0
        elif counts_toward_hand_streak(hand_count):
            self._hand_streak += 1
        # ≥2 只手：疑似误检帧，不计入也不清零
        if self._hand_streak >= self.hand_frames:
            return self._mark_switch(
                STATE_FAR_TRACK, now,
                f"CAPTURE→FAR_TRACK: hagrid_yolo 连续 {self._hand_streak} 帧稳定单手，"
                "交接给 ZOOM 跟踪",
            )
        return None

    def _update_far_track(self, hand_count, hand_bbox_ratio, now):
        """FAR_TRACK：mediapipe + ZOOM 跟踪。
        手变大/走近持续 M 帧 → NEAR；再次连续 N 帧无手 → CAPTURE。"""
        # 丢手判据：连续 N 帧无手 → 回 CAPTURE 让 YOLO 重新抓
        if hand_count == 0:
            self._no_hand_streak += 1
        else:
            self._no_hand_streak = 0
        if self._no_hand_streak >= self.no_hand_frames:
            return self._mark_switch(
                STATE_CAPTURE, now,
                f"FAR_TRACK→CAPTURE: 连续 {self._no_hand_streak} 帧未检到手，"
                "交给 hagrid_yolo 重新捕获",
            )
        # 走近判据：单手且 bbox 占比 ≥ 阈值，持续 M 帧 → 回 NEAR（宁慢勿错）
        if (counts_toward_hand_streak(hand_count)
                and hand_bbox_ratio >= self.near_bbox_ratio):
            self._near_streak += 1
        elif hand_count == 0:
            self._near_streak = 0
        # ≥2 只手或手不够大：不计入也不清零
        if self._near_streak >= self.near_frames:
            return self._mark_switch(
                STATE_NEAR, now,
                f"FAR_TRACK→NEAR: 手 bbox 占比 {hand_bbox_ratio:.1%} ≥ "
                f"{self.near_bbox_ratio:.1%} 持续 {self._near_streak} 帧，用户走近",
            )
        return None

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _reset_streaks(self):
        self._no_hand_streak = 0
        self._hand_streak = 0
        self._near_streak = 0

    def _mark_switch(self, target_state, now, reason):
        self.state = target_state
        self._last_switch_time = now
        self._reset_streaks()
        self.last_reason = reason
        logger.info("引擎自动切换迁移: %s", reason)
        return target_state
