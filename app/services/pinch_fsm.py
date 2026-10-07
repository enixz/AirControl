"""捏合按钮状态机 — MouseMode 左/右键交互的显式状态与转换。

从 mouse_mode.handle() 的内联 if/else 整理而来（2026-10 第 3 步）。
诊断证据（artifacts/click_diagnosis/20260722_145659/ANOMALY_REPORT.md）：
  - 拖拽中断主因：35-45px 小手上 idx_ratio 在 0.35-0.55 阈值带内振荡，
    ml_label 每 5-8 帧 Closed_Fist↔Open_Palm 翻转（t2 三连碎、t6 三连击）
  - 误触主因：真握拳几何上与捏合不可区分（det[7]，Closed_Fist 0.9s）
  - t1 漏检主因：远距按下确认计时被预测补帧逐帧清零

状态与转换（每次转换都有明确原因，全部沿用旧时间参数）：

  IDLE ──fresh+pinch证据持续≥PRESS_CONFIRM_SEC──▶ HOLDING   （新按下）
  IDLE ──预测帧──▶ IDLE（不产生新按下）
  HOLDING ──fresh 非 pinch 持续≥RELEASE_CONFIRM_SEC──▶ IDLE  （释放确认）
  HOLDING ──fresh pinch──▶ HOLDING（重置释放计时；边界抖动被两个
           确认窗口 + recognizer 0.08s 滞回吸收）
  HOLDING ──预测帧──▶ HOLDING（保持，不重置释放计时：预测帧
           不构成松手证据）
  HOLDING ──手丢失──▶ IDLE（强制释放，原因=hand_lost）
  HOLDING ──watchdog 强制释放──▶ IDLE（blocked_until_open：
           需一次明确张开才允许新按下）
  HOLDING ──持续超过 HOLD_MAX_SEC──▶ IDLE（保持上限，防预测
           补帧长锁；现值 8s ≈ 旧 0.12s 预测上限 + smoother 8 帧的
           保守推广，正常拖拽远达不到）
  任意 ──Closed_Fist 标签──▶ 不允许新按下（握拳≠捏合）
  释放后残留捏合（丢手强制释放后手立即回来且仍捏）──▶ 需一次
           OPEN 状态（明确张开）才能重新进入按下确认

不改变 recognizer 层阈值（ENTER 0.35 / EXIT 0.40 / 滞回 0.08s 原样）。
"""

class PinchButtonStateMachine:
    """单个按钮（左键 hold / 右键 click）的捏合状态机。

    MouseMode 对左右键各持一个实例；左键实例驱动 left_down/left_up，
    右键实例驱动单次 right_click。
    """

    # 以下时间常量沿用 mouse_mode 原有参数，不引入新标定
    PRESS_CONFIRM_SEC = 0.05       # 新按下需要连续 fresh pinch 证据（原 distant gate）
    RELEASE_CONFIRM_SEC = 0.08     # 释放需要持续非 pinch（原 recognizer 滞回等值）
    HOLD_MAX_SEC = 8.0             # 保持上限（防预测帧长锁；正常拖拽 << 8s）
    UNRELIABLE_HOLD_MAX_SEC = 0.5  # 几何不可靠仅桥接短暂抖动，不能长期维持按住
    # Closed_Fist 标签期间禁止新按下（握拳≠捏合）。实测远距录像：
    # 真握拳 0.9s 误触与远距真实捏合（Closed_Fist 误标）不可靠区分，
    # 默认关闭；待第 4 步几何可靠性门控落地后再启用（config: pinch_fist_block）
    FIST_BLOCK = False

    IDLE = "idle"
    HOLDING = "holding"

    def __init__(self, log=None):
        self.state = self.IDLE
        self.press_evidence_since = None   # 连续 fresh pinch 起点
        self.release_evidence_since = None  # 连续 fresh 非 pinch 起点
        self.hold_since = None
        self.blocked_until_open = False     # 强制释放后需明确张开
        self.unreliable_since = None
        self.last_transition_reason = "init"
        self._log = log

    def _transition(self, new_state, reason):
        self.state = new_state
        self.last_transition_reason = reason
        if self._log:
            self._log("pinch_fsm: %s (%s)", new_state, reason)

    def force_release(self, reason):
        """外部强制释放（手丢失 / watchdog / on_exit）。"""
        was_holding = self.state == self.HOLDING
        self.press_evidence_since = None
        self.release_evidence_since = None
        self.hold_since = None
        self.unreliable_since = None
        if self.state != self.IDLE:
            self._transition(self.IDLE, reason)
        # 仅当强制释放真的打断了一次按住时，才要求"明确张开"才能再按：
        # 丢手时正在按住 → 手回来若仍捏着，不能立即重建按下（残留捏合）。
        # 没有按住时的丢失（启动/重连/预测帧无历史）不应阻塞后续按下。
        # hold_max 同理：预测帧长锁释放后，残留捏合不能立即重建。
        if was_holding and reason in ("hand_lost", "watchdog", "hold_max", "geometry_unreliable"):
            self.blocked_until_open = True
        return was_holding

    def update(self, *, pinch, fresh, ml_label, now, distant=False,
               clearly_open=False, geometry_reliable=True):
        """喂一帧捏合证据，返回 (should_be_down, transition)。

        Args:
            pinch: recognizer 判定的捏合（含滞回）
            fresh: 本帧是否真实检出（非预测补帧）
            ml_label: 手势 ML 标签（Closed_Fist 用于否决新按下）
            now: time.monotonic()
            distant: 远距小手（hand_width/帧宽 < 0.03）→ 新按下需要
                PRESS_CONFIRM_SEC 连续证据；近手立即可按（保持旧延迟）
            clearly_open: 明确张开（比值 ≥0.55，与 recognizer 立即释放
                规则同界）→ 释放无需等待 RELEASE_CONFIRM_SEC
            geometry_reliable: 几何可靠性（掌宽相对 EMA 偏离 ≤60%）。
                不可靠帧（相机预热/关键点跳变）不构成按下确认证据，
                但也不打断已建立的按住（显示/指针仍可用这些帧）。

        Returns:
            (bool, str|None): 该按钮应处状态；发生进入/退出转换时给出原因。
        """
        if (self.state == self.HOLDING
                and now - self.hold_since >= self.HOLD_MAX_SEC):
            self.force_release("hold_max")
            return False, "hold_max"

        if fresh and not geometry_reliable:
            if self.state == self.HOLDING:
                if self.unreliable_since is None:
                    self.unreliable_since = now
                if now - self.unreliable_since >= self.UNRELIABLE_HOLD_MAX_SEC:
                    self.force_release("geometry_unreliable")
                    return False, "geometry_unreliable"
            return self.state == self.HOLDING, None

        if not fresh:
            # 预测补帧：不构成任何新证据（既不确认按下也不确认释放）
            return self.state == self.HOLDING, None

        self.unreliable_since = None

        is_fist = self.FIST_BLOCK and ml_label == "Closed_Fist"
        confirm_needed = self.PRESS_CONFIRM_SEC if distant else 0.0

        if self.state == self.IDLE:
            # 明确张开 → 解除强制释放锁
            if self.blocked_until_open and not pinch and not is_fist:
                self.blocked_until_open = False
            if self.blocked_until_open or is_fist:
                self.press_evidence_since = None
                return False, None
            if pinch:
                if self.press_evidence_since is None:
                    self.press_evidence_since = now
                if now - self.press_evidence_since >= confirm_needed:
                    self.hold_since = now
                    self.release_evidence_since = None
                    self.press_evidence_since = None
                    self._transition(self.HOLDING, "press_confirmed")
                    return True, "press_confirmed"
            else:
                self.press_evidence_since = None
            return False, None

        # HOLDING
        if pinch:
            # 保持证据：重置释放计时
            self.release_evidence_since = None
            return True, None
        # 非 pinch：明确张开立即释放；边界抖动进入释放确认窗口
        if clearly_open:
            self._transition(self.IDLE, "release_clear_open")
            self.release_evidence_since = None
            self.hold_since = None
            return False, "release_clear_open"
        if self.release_evidence_since is None:
            self.release_evidence_since = now
        if now - self.release_evidence_since >= self.RELEASE_CONFIRM_SEC:
            self._transition(self.IDLE, "release_confirmed")
            self.release_evidence_since = None
            self.hold_since = None
            return False, "release_confirmed"
        return True, None

    def reset(self, preserve_block=False):
        """身份切换（换手）时彻底清零：捏合历史不跨手继承。"""
        self.state = self.IDLE
        self.press_evidence_since = None
        self.release_evidence_since = None
        self.hold_since = None
        if not preserve_block:
            self.blocked_until_open = False
        self.unreliable_since = None
        self.last_transition_reason = "reset_identity"
