import math
import time

from modes.base import ModeResult


class ModeManager:
    """管理模式生命周期、切换动画和双手握拳切换手势。

    将原来 FloatingWindow 中散落的模式切换逻辑集中到此，
    FloatingWindow 只负责调用 handle() 和读取结果。
    """

    def __init__(self, modes: dict, config, recognizer):
        self.modes = modes
        self.config = config
        self.recognizer = recognizer
        self.current_mode_name = None
        self.current_mode = None
        self.last_mode_switch_time = 0
        self.mode_switch_hold_frames = 0
        # 滑窗：最近 N 帧里只要有 ≥M 帧满足条件就算通过，扛 MediaPipe 偶发丢手
        self._two_fist_history: list[bool] = []
        self._two_fist_window = 10   # 最近 10 帧窗口
        self._two_fist_required = 6  # 其中 6 帧满足条件即触发

    def switch_to(self, mode_name: str):
        if mode_name not in self.modes:
            return
        if self.current_mode_name == mode_name:
            return  # 避免重复切换导致 on_exit/on_enter 被调用两次
        if self.current_mode:
            self.current_mode.on_exit()
        self.current_mode = self.modes[mode_name]
        self.current_mode_name = mode_name
        self.current_mode.on_enter()
        # 统一记录切换时间，确保语音/手势切换后1秒手势保护窗口均生效
        self.last_mode_switch_time = time.time()

    def cycle_mode(self):
        modes = ["presentation", "mouse", "draw"]
        current = self.current_mode_name
        try:
            index = modes.index(current)
        except ValueError:
            index = 0
        next_mode = modes[(index + 1) % len(modes)]
        # 使用 batch_update 避免立即写入磁盘，延迟到上下文退出时统一保存
        with self.config.batch_update():
            self.config.set("interaction_mode", next_mode)
        self.switch_to(next_mode)

    def maybe_switch_by_two_fists(self, hands_landmarks, frame_w=None) -> bool:
        """双拳握拳并分开 → 切模式。

        改良点：
          1. 阈值按画面宽度自适应（远距小手也能触发，不再卡死 120 px）
          2. 滑窗容错：最近 10 帧里有 6 帧满足即可（顶住 MediaPipe 偶发丢手）
          3. 移除"必须连续 8 帧"，改成"过去 0.33 秒里 6 次有效"
        """
        ok = self._is_two_fists_apart(hands_landmarks, frame_w)
        self._two_fist_history.append(ok)
        if len(self._two_fist_history) > self._two_fist_window:
            self._two_fist_history.pop(0)

        hits = sum(self._two_fist_history)
        if hits < self._two_fist_required:
            return False

        if time.time() - self.last_mode_switch_time < 1.5:
            return False

        self._two_fist_history.clear()
        self.cycle_mode()
        return True

    def _is_two_fists_apart(self, hands_landmarks, frame_w=None) -> bool:
        """单帧判定：是否两手都握拳且分开。"""
        if len(hands_landmarks) < 2:
            return False
        first = self.recognizer.get_hand_features(hands_landmarks[0])
        second = self.recognizer.get_hand_features(hands_landmarks[1])
        if not (first["is_fist"] and second["is_fist"]):
            return False
        wrist_dist = math.hypot(
            hands_landmarks[0][0][1] - hands_landmarks[1][0][1],
            hands_landmarks[0][0][2] - hands_landmarks[1][0][2],
        )
        # 自适应：720p 取 ~10% 画宽 ≈ 128 px（与旧 120 接近）；
        # 480p 自动降到 ~64 px，远距离也能触发
        min_dist = max(60.0, (frame_w or 1280) * 0.08)
        return wrist_dist >= min_dist

    def handle(self, hands_landmarks, hands_gestures, frame_w, frame_h) -> ModeResult:
        if self.current_mode is None:
            return ModeResult(gesture="NONE")
        return self.current_mode.handle(hands_landmarks, hands_gestures, frame_w, frame_h)
