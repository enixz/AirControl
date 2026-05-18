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
        self.last_mode_switch_time = time.time()

    def maybe_switch_by_two_fists(self, hands_landmarks) -> bool:
        """检测双手握拳并保持（拳头分开），触发模式切换。"""
        if len(hands_landmarks) < 2:
            self.mode_switch_hold_frames = 0
            return False

        first = self.recognizer.get_hand_features(hands_landmarks[0])
        second = self.recognizer.get_hand_features(hands_landmarks[1])

        if not (first["is_fist"] and second["is_fist"]):
            self.mode_switch_hold_frames = 0
            return False

        wrist_dist = math.hypot(
            hands_landmarks[0][0][1] - hands_landmarks[1][0][1],
            hands_landmarks[0][0][2] - hands_landmarks[1][0][2],
        )
        if wrist_dist < 120:
            self.mode_switch_hold_frames = 0
            return False

        self.mode_switch_hold_frames += 1
        if self.mode_switch_hold_frames < 8:
            return False

        self.mode_switch_hold_frames = 0
        if time.time() - self.last_mode_switch_time < 1.5:
            return False

        self.cycle_mode()
        return True

    def handle(self, hands_landmarks, hands_gestures, frame_w, frame_h) -> ModeResult:
        if self.current_mode is None:
            return ModeResult(gesture="NONE")
        return self.current_mode.handle(hands_landmarks, hands_gestures, frame_w, frame_h)
