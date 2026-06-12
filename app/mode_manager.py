import logging
import math
import time

from modes.base import ModeResult

logger = logging.getLogger("gesture")


class ModeManager:
    """管理模式生命周期、切换动画和单手双次抓取切换手势。

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

        # 抓取状态机：0(初始) -> 1(握拳) -> 2(张开) -> 3(再次握拳) -> 触发(再次张开)
        self._grasp_state = 0
        self._grasp_state_time = 0

    def switch_to(self, mode_name: str):
        if mode_name not in self.modes:
            return
        if self.current_mode_name == mode_name:
            return  # 避免重复切换导致 on_exit/on_enter 被调用两次
        if self.current_mode:
            self.current_mode.on_exit()
        prev = self.current_mode_name
        self.current_mode = self.modes[mode_name]
        self.current_mode_name = mode_name
        self.current_mode.on_enter()
        # 统一记录切换时间，确保语音/手势切换后1秒手势保护窗口均生效
        self.last_mode_switch_time = time.time()
        # 记录到 gesture.log，便于测试时核对三种模式是否正常切换
        logger.info("=> 模式切换: %s -> %s", prev or "(无)", mode_name)

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

    def maybe_switch_by_gesture(self, hands_landmarks, frame_w=None) -> bool:
        """单手抓取与释放连续两次（握拳->张开->握拳->张开）切模式。"""
        if not hands_landmarks:
            return False

        now = time.time()
        # 2.5秒内未完成整个序列则重置状态
        if now - self._grasp_state_time > 2.5:
            self._grasp_state = 0

        # 只看第一只手的特征
        features = self.recognizer.get_hand_features(hands_landmarks[0])
        is_fist = features["is_fist"]
        is_open = features["is_open_palm"]

        if self._grasp_state == 0:
            if is_fist:
                self._grasp_state = 1
                self._grasp_state_time = now
        elif self._grasp_state == 1:
            if is_open:
                self._grasp_state = 2
                self._grasp_state_time = now
        elif self._grasp_state == 2:
            if is_fist:
                self._grasp_state = 3
                self._grasp_state_time = now
        elif self._grasp_state == 3:
            if is_open:
                self._grasp_state = 0
                if now - self.last_mode_switch_time < 1.5:
                    return False
                self.cycle_mode()
                return True

        return False

    def handle(self, hands_landmarks, hands_gestures, frame_w, frame_h) -> ModeResult:
        if self.current_mode is None:
            return ModeResult(gesture="NONE")
        return self.current_mode.handle(hands_landmarks, hands_gestures, frame_w, frame_h)
