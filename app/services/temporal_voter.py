"""时序手势投票器 + 状态机 — CPU纯算法，统一取代散落的逐手势时序逻辑。

核心思路：
  1. 滑动窗口内每帧手势置信度按指数衰减加权累积
  2. 四状态有限状态机（IDLE→DETECTING→CONFIRMED→RELEASING）
  3. 双阈值滞回（enter/exit），防止边界抖动反复触发
  4. 距离+抖动自适应：手越小/越抖→窗口越长→累积越保守
  5. 每手势独立不应期，取代全局1s cooldown的粗暴 blackout

为什么能优于 MediaPipe：
  - MediaPipe 给出单帧标签，远距离下关键点抖动导致标签逐帧跳变
  - 本投票器累积多帧证据 + 状态机门控，输出稳定标签
  - 状态机的 DETECTING/RELEASING 状态能过滤短暂抖动，比纯滞回更智能
  - 不需要训练、不需要GPU、不需要ONNX，纯Python实现，<0.1ms/帧
"""

import logging
import time
from collections import deque
from enum import Enum

logger = logging.getLogger('gesture')

# MediaPipe GestureRecognizer 输出的7类手势 + None
GESTURE_CLASSES = (
    "Closed_Fist", "Open_Palm", "Pointing_Up",
    "Thumb_Up", "Thumb_Down", "Victory", "ILoveYou",
)

# 每类手势的独立配置：(enter_threshold, exit_threshold, refractory_ms)
GESTURE_PROFILES = {
    "Closed_Fist":  (0.55, 0.30, 400),
    "Open_Palm":    (0.50, 0.25, 300),
    "Pointing_Up":  (0.60, 0.35, 500),
    "Thumb_Up":     (0.65, 0.40, 800),
    "Thumb_Down":   (0.65, 0.40, 800),
    "Victory":      (0.60, 0.35, 600),
    "ILoveYou":     (0.65, 0.40, 800),
}

_DEFAULT_PROFILE = (0.60, 0.35, 500)

# 距离自适应参数
REFERENCE_HAND_WIDTH = 90.0
MIN_WINDOW = 5
MAX_WINDOW = 12

# 状态机参数
DETECT_TIMEOUT_MS = 600.0
RELEASE_TIMEOUT_MS = 400.0
PRE_THRESHOLD_RATIO = 0.7


class GestureState(Enum):
    """手势状态机的四种状态。"""
    IDLE = "IDLE"
    DETECTING = "DETECTING"
    CONFIRMED = "CONFIRMED"
    RELEASING = "RELEASING"


class TemporalGestureVoter:
    """时序手势投票器 + 状态机。"""

    def __init__(self, config=None):
        self._config = config
        self._window = deque()
        self._state = GestureState.IDLE
        self._active_gesture = None
        self._state_enter_time = 0.0
        self._active_enter_time = 0.0
        self._last_trigger_time = {}
        self._decay_alpha = 0.85
        logger.info("=== TemporalGestureVoter Started (FSM + CPU-only) ===")

    def update(self, mp_gestures, hand_width=REFERENCE_HAND_WIDTH, jitter=0.0):
        """每帧调用，返回稳定后的手势标签。"""
        now_ms = time.time() * 1000.0
        window_len = self._adaptive_window(hand_width, jitter)
        current_label, current_score = self._extract_top_gesture(mp_gestures)
        self._window.append((now_ms, current_label, current_score, hand_width))
        while len(self._window) > window_len:
            self._window.popleft()
        scores = self._compute_weighted_scores()
        stable_label = self._fsm_decision(scores, now_ms)
        return stable_label

    def reset(self):
        """重置所有状态（手丢失时调用）。"""
        self._window.clear()
        self._state = GestureState.IDLE
        self._active_gesture = None
        self._active_enter_time = 0.0
        self._state_enter_time = 0.0

    def get_debug_info(self):
        scores = self._compute_weighted_scores()
        return {
            "state": self._state.value,
            "window_len": len(self._window),
            "active_gesture": self._active_gesture,
            "scores": {k: round(v, 3) for k, v in scores.items()},
        }

    @property
    def state(self):
        return self._state

    def _adaptive_window(self, hand_width, jitter=0.0):
        ratio = REFERENCE_HAND_WIDTH / max(hand_width, 20.0)
        dist_window = int(MIN_WINDOW + (MAX_WINDOW - MIN_WINDOW) * min(ratio - 1.0, 3.0) / 3.0)
        jitter_ratio = min(jitter / 10.0, 1.0)
        jitter_window = int(MIN_WINDOW + (MAX_WINDOW - MIN_WINDOW) * jitter_ratio)
        window = max(dist_window, jitter_window)
        return max(MIN_WINDOW, min(MAX_WINDOW, window))

    def _extract_top_gesture(self, mp_gestures):
        if not mp_gestures:
            return "None", 0.0
        best_label = "None"
        best_score = 0.0
        for g in mp_gestures:
            ml_label = g.get("ml_label", "None")
            score = g.get("score", 0.0)
            if ml_label != "None" and score > best_score:
                best_label = ml_label
                best_score = score
        return best_label, best_score

    def _compute_weighted_scores(self):
        if not self._window:
            return {cls: 0.0 for cls in GESTURE_CLASSES}
        n = len(self._window)
        scores = {cls: 0.0 for cls in GESTURE_CLASSES}
        total_weight = 0.0
        for i, (_ts, label, score, _hw) in enumerate(self._window):
            age = n - 1 - i
            weight = self._decay_alpha ** age
            total_weight += weight
            if label in scores:
                scores[label] += weight * score
        if total_weight > 0:
            for cls in scores:
                scores[cls] /= total_weight
        return scores

    def _fsm_decision(self, scores, now_ms):
        """四状态有限状态机决策。"""
        best_cls = max(scores, key=scores.get)
        best_score = scores[best_cls]

        if self._state == GestureState.IDLE:
            return self._handle_idle(best_cls, best_score, now_ms)
        elif self._state == GestureState.DETECTING:
            return self._handle_detecting(scores, now_ms)
        elif self._state == GestureState.CONFIRMED:
            return self._handle_confirmed(scores, now_ms)
        elif self._state == GestureState.RELEASING:
            return self._handle_releasing(scores, now_ms)
        return "None"

    def _handle_idle(self, best_cls, best_score, now_ms):
        """IDLE：检查是否有候选手势。"""
        enter_th, _, _ = GESTURE_PROFILES.get(best_cls, _DEFAULT_PROFILE)
        pre_th = enter_th * PRE_THRESHOLD_RATIO
        if best_score >= pre_th:
            self._state = GestureState.DETECTING
            self._active_gesture = best_cls
            self._state_enter_time = now_ms
        return "None"

    def _handle_detecting(self, scores, now_ms):
        """DETECTING：候选手势等待累积确认。"""
        candidate = self._active_gesture
        candidate_score = scores.get(candidate, 0.0)
        enter_th, _, refractory = GESTURE_PROFILES.get(candidate, _DEFAULT_PROFILE)

        if now_ms - self._state_enter_time > DETECT_TIMEOUT_MS:
            self._state = GestureState.IDLE
            self._active_gesture = None
            return "None"

        if candidate_score >= enter_th:
            last_trigger = self._last_trigger_time.get(candidate, 0.0)
            if now_ms - last_trigger < refractory:
                self._state = GestureState.IDLE
                self._active_gesture = None
                return "None"
            self._state = GestureState.CONFIRMED
            self._active_enter_time = now_ms
            self._state_enter_time = now_ms
            self._last_trigger_time[candidate] = now_ms
            logger.info("[FSM] DETECTING→CONFIRMED %s (score=%.3f)", candidate, candidate_score)
            return candidate

        pre_th = enter_th * PRE_THRESHOLD_RATIO
        if candidate_score < pre_th:
            self._state = GestureState.IDLE
            self._active_gesture = None
        return "None"

    def _handle_confirmed(self, scores, now_ms):
        """CONFIRMED：手势已确认，检查是否开始释放。"""
        active = self._active_gesture
        active_score = scores.get(active, 0.0)
        _, exit_th, _ = GESTURE_PROFILES.get(active, _DEFAULT_PROFILE)

        if active_score < exit_th:
            self._state = GestureState.RELEASING
            self._state_enter_time = now_ms
        return active

    def _handle_releasing(self, scores, now_ms):
        """RELEASING：手势释放中，检查恢复或确认退出。

        只靠超时退出（RELEASE_TIMEOUT_MS），不设额外阈值。
        这样 RELEASING 会持续一段时间，给手势恢复的机会。
        """
        active = self._active_gesture
        active_score = scores.get(active, 0.0)
        _, exit_th, _ = GESTURE_PROFILES.get(active, _DEFAULT_PROFILE)

        # 得分恢复 → 回 CONFIRMED
        if active_score >= exit_th:
            self._state = GestureState.CONFIRMED
            self._state_enter_time = now_ms
            return active

        # 超时 → 确认退出
        if now_ms - self._state_enter_time > RELEASE_TIMEOUT_MS:
            logger.info("[FSM] RELEASING→IDLE %s (timeout)", active)
            self._state = GestureState.IDLE
            self._active_gesture = None
            return "None"

        # RELEASING 期间仍输出当前手势
        return active
