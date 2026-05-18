from .base import ModeBase, ModeResult


class MouseMode(ModeBase):
    """鼠标模式：中指控制光标，拇指-食指捏合左键，拇指-中指捏合右键，剪刀手滚动。"""

    def on_enter(self):
        self.overlay.hide()
        self.overlay.setGeometry(-100, -100, 0, 0)
        self.overlay.force_lift_pen()
        self.overlay.hide_cursor()
        self.toolbar.hide()
        self.mouse.reset()  # 重置鼠标状态，防止从其他模式切回时 last_pos 残留
        self.cursor_overlay.show_fullscreen()
        # 新增：状态机初始化
        self._is_left_holding = False
        self._is_right_pinching = False
        self.cursor_overlay.set_left_hold(False)

    def on_exit(self):
        # 必须释放左键
        if getattr(self, '_is_left_holding', False):
            self.mouse.left_up()
            self._is_left_holding = False
            self.cursor_overlay.set_left_hold(False)
        self._is_right_pinching = False
        self.mouse.reset()
        self.cursor_overlay.hide_cursor()
        self.cursor_overlay.hide()

    def handle(self, hands_landmarks, hands_gestures, frame_w, frame_h) -> ModeResult:
        # 1. 手丢失：释放左键 + 重置
        if not hands_landmarks:
            if getattr(self, '_is_left_holding', False):
                self.mouse.left_up()
                self._is_left_holding = False
                self.cursor_overlay.set_left_hold(False)
            self._is_right_pinching = False
            self.mouse.reset()
            self.cursor_overlay.hide_cursor()
            return ModeResult(
                gesture="NONE",
                status_text="未检测到手",
                status_color=(0, 0, 255),
            )

        landmarks = hands_landmarks[0]
        features = self.recognizer.get_hand_features(landmarks)
        x_norm = landmarks[12][1] / frame_w
        y_norm = landmarks[12][2] / frame_h
        screen_x, screen_y = self.mouse.move_to_normalized(x_norm, y_norm)
        self.cursor_overlay.update_cursor(screen_x, screen_y)

        # 2. 滚轮检测（持续滚动：保持剪刀手则持续输出）
        scroll_dir = self.recognizer.check_scroll(landmarks, features, False)
        if scroll_dir != 0:
            amount = 120 * scroll_dir
            if self.mouse.scroll_wheel(amount):
                self.cursor_overlay.trigger_scroll(scroll_dir)
                # 滚动时不提前 return，继续处理移动和点击，实现边滚边操作

        # 3. 左键按住状态机（本期核心变更）
        if features["thumb_index_pinch"]:
            if not self._is_left_holding:
                self.mouse.left_down()
                self._is_left_holding = True
                self.cursor_overlay.set_left_hold(True)
            # 按住期间继续移动光标，每帧返回 HOLD 状态
            return ModeResult(
                gesture="LEFT_HOLD",
                status_text="Left Button Hold",
                status_color=(255, 0, 0),
            )
        else:
            if self._is_left_holding:
                self.mouse.left_up()
                self._is_left_holding = False
                self.cursor_overlay.set_left_hold(False)

        # 4. 右键单击（单次触发：捏合期间只触发一次，松开才重置）
        if features["thumb_middle_pinch"]:
            if not self._is_right_pinching:
                self._is_right_pinching = True
                self.mouse.right_click()
                self.cursor_overlay.trigger_right_click(screen_x, screen_y)
                return ModeResult(
                    gesture="RIGHT_CLICK",
                    status_text="Mouse Right Click",
                    status_color=(80, 220, 120),
                )
        else:
            self._is_right_pinching = False

        return ModeResult(
            gesture="MOUSE_MOVE",
            status_text="Mouse Move",
            status_color=(0, 255, 0),
        )
