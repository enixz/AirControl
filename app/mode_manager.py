import logging
import time
from collections import deque

from modes import MODE_NAMES
from modes.base import ModeResult

logger = logging.getLogger("gesture")


class ModeManager:
    """管理模式生命周期、切换动画和 🤟 保持切换手势。

    将原来 FloatingWindow 中散落的模式切换逻辑集中到此，
    FloatingWindow 只负责调用 handle() 和读取结果。
    """

    ILY_MIN_SAMPLES = 6  # 窗口内最少采样帧数，防止手丢失后用零星陈旧帧凑满时间窗

    def __init__(self, modes: dict, config, recognizer):
        self.modes = modes
        self.config = config
        self.recognizer = recognizer
        self.current_mode_name = None
        self.current_mode = None
        self.last_mode_switch_time = 0

        # 🤟 保持切模式：滑动时间窗内的 (时间戳, 该帧是否为 I_LOVE_YOU) 采样
        self._ily_samples = deque()
        self._ily_armed = True
        self._ily_release_since = None
        self._ily_candidate = False
        self._ily_last_log = 0.0
        self._ily_hold_sec = float(config.get("mode_switch_hold_sec", 1.0)) if config else 1.0
        self._ily_vote_ratio = float(config.get("mode_switch_vote_ratio", 0.6)) if config else 0.6
        self._ily_release_sec = float(
            config.get("mode_switch_release_sec", 0.25)
        ) if config else 0.25

    @property
    def is_switch_candidate(self):
        """Return True while a real ILY gesture currently owns the frame."""
        return self._ily_candidate

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
        current = self.current_mode_name
        try:
            index = MODE_NAMES.index(current)
        except ValueError:
            index = 0
        next_mode = MODE_NAMES[(index + 1) % len(MODE_NAMES)]
        # 使用 batch_update 避免立即写入磁盘，延迟到上下文退出时统一保存
        with self.config.batch_update():
            self.config.set("interaction_mode", next_mode)
        self.switch_to(next_mode)

    def maybe_switch_by_gesture(self, hands_landmarks, hands_gestures=None, frame_w=None) -> bool:
        """单手 🤟（MediaPipe ILoveYou 标签）保持约 1 秒切换模式。

        远距离下逐指几何判定最先失效，ML 标签是本系统最可靠的信号（同
        gesture_recognizer 中 THUMB_DOWN 的教训），故直接对标签做滑动
        时间窗多数投票：偶发单帧漏检只摊薄占比，不像旧版"握拳-张开"
        四段时序那样任一段漏检即整体作废。

        触发后必须先看到 🤟 从窗口中完全消失才重新武装，
        防止持续摆姿势导致连环切换。
        """
        now = time.time()
        has_ily = bool(hands_gestures) and any(
            g.get("label") == "I_LOVE_YOU" for g in hands_gestures if g
        )
        has_real_hand = bool(hands_landmarks)
        self._ily_candidate = has_real_hand and has_ily

        # Missing or deliberately suppressed frames are not evidence that the
        # gesture was released. Require a positively observed non-ILY pose for
        # a short period before another switch can be armed.
        if not self._ily_armed:
            if has_real_hand and not has_ily:
                if self._ily_release_since is None:
                    self._ily_release_since = now
                elif now - self._ily_release_since >= self._ily_release_sec:
                    self._ily_armed = True
                    self._ily_release_since = None
                    self._ily_samples.clear()
            else:
                self._ily_release_since = None
            return False

        if has_real_hand:
            self._ily_samples.append((now, has_ily))
        elif self._ily_samples and now - self._ily_samples[-1][0] > self._ily_hold_sec * 0.5:
            # 手消失超过半个时间窗：清空重来，避免陈旧样本拼出虚假保持
            self._ily_samples.clear()

        # 滑出时间窗的旧样本
        while self._ily_samples and now - self._ily_samples[0][0] > self._ily_hold_sec:
            self._ily_samples.popleft()

        ily_count = sum(1 for _, flag in self._ily_samples if flag)
        total = len(self._ily_samples)

        if ily_count == 0:
            return False

        ratio = ily_count / total
        # 证据跨度 = 最早到最新样本的实际时间差（不用墙钟 now：
        # 手丢失时 now 仍在涨，会用陈旧样本凑满时间窗）
        span = self._ily_samples[-1][0] - self._ily_samples[0][0]

        # 取数标定用：保持期间每 0.5s 记录一次进度，便于在 gesture.log
        # 中观察实际距离下的标签占比，再调 mode_switch_vote_ratio
        if now - self._ily_last_log >= 0.5:
            self._ily_last_log = now
            logger.info(
                "[ILY] 切模式手势进度: ratio=%.2f (%d/%d) span=%.2fs armed=%s",
                ratio, ily_count, total, span, self._ily_armed,
            )

        # 最新样本须新鲜：手刚丢失的瞬间不允许触发
        if now - self._ily_samples[-1][0] > 0.3:
            return False
        # 0.8 系数：滑动窗内证据跨度天然比窗口短一个帧间隔，低帧率下尤甚
        if span < self._ily_hold_sec * 0.8 or total < self.ILY_MIN_SAMPLES:
            return False
        if ratio < self._ily_vote_ratio:
            return False
        if now - self.last_mode_switch_time < 1.5:
            return False

        logger.info(
            "=> 模式切换手势: 🤟 保持 %.2fs, ratio=%.2f (%d/%d)",
            span, ratio, ily_count, total,
        )
        self._ily_samples.clear()
        self._ily_armed = False
        self._ily_release_since = None
        self.cycle_mode()
        return True

    def handle(self, hands_landmarks, hands_gestures, frame_w, frame_h) -> ModeResult:
        if self.current_mode is None:
            return ModeResult(gesture="NONE")
        return self.current_mode.handle(hands_landmarks, hands_gestures, frame_w, frame_h)
