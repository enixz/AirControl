from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModeResult:
    """模式处理结果，由 FloatingWindow 消费以更新 UI 和执行动作。"""
    gesture: str = "NONE"
    status_text: str = ""
    status_color: tuple = field(default_factory=lambda: (0, 255, 0))
    action: Optional[str] = None


class ModeBase(ABC):
    """交互模式策略基类。

    每种模式（演示/鼠标/板书）继承此类，实现 on_enter/on_exit/handle。
    FloatingWindow 只负责调度，不再包含任何模式具体逻辑。
    """

    def __init__(
        self,
        config,
        recognizer,
        mouse,
        overlay,
        cursor_overlay,
        toolbar,
        ppt,
    ):
        self.config = config
        self.recognizer = recognizer
        self.mouse = mouse
        self.overlay = overlay
        self.cursor_overlay = cursor_overlay
        self.toolbar = toolbar
        self.ppt = ppt

    @abstractmethod
    def on_enter(self):
        """进入此模式时调用（一次）。负责设置 UI 状态（显示/隐藏 overlay 等）。"""
        pass

    @abstractmethod
    def on_exit(self):
        """退出此模式时调用（一次）。负责清理状态。"""
        pass

    @abstractmethod
    def handle(self, hands_landmarks, hands_gestures, frame_w, frame_h) -> ModeResult:
        """每帧调用，处理手势逻辑并返回结果。"""
        pass

    def _sync_frame_size(self, frame_w, frame_h):
        """将当前帧尺寸同步到 recognizer，用于边缘检测等自适应逻辑。"""
        if self.recognizer:
            self.recognizer.frame_w = max(frame_w, 1)
            self.recognizer.frame_h = max(frame_h, 1)
