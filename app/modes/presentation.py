import math

from .base import ModeBase, ModeResult


class PresentationMode(ModeBase):
    """演示模式：并掌挥动翻页、握拳/点赞执行映射动作。"""

    def on_enter(self):
        self.overlay.hide()
        self.overlay.setGeometry(-100, -100, 0, 0)
        self.overlay.force_lift_pen()
        self.overlay.hide_cursor()
        self.toolbar.hide()
        self.cursor_overlay.hide()
        self.cursor_overlay.setGeometry(-100, -100, 0, 0)

    def on_exit(self):
        pass

    def handle(self, hands_landmarks, hands_gestures, frame_w, frame_h) -> ModeResult:
        self._sync_frame_size(frame_w, frame_h)
        all_features = [self.recognizer.get_hand_features(lm) for lm in hands_landmarks] if hands_landmarks else []

        two_hand = self.recognizer.check_two_hand_gesture(hands_landmarks, all_features)
        if two_hand:
            action = self.config.get_mapping(two_hand)
            return ModeResult(
                gesture=two_hand,
                status_text=f"{two_hand} -> {action}",
                status_color=(0, 255, 255),
                action=action,
            )

        selected_hands, selected_index = self._select_presentation_hand(hands_landmarks, all_features)
        selected_gestures = None
        if hands_gestures and 0 <= selected_index < len(hands_gestures):
            selected_gestures = [hands_gestures[selected_index]]

        selected_features = all_features[selected_index] if all_features and 0 <= selected_index < len(all_features) else None
        gesture = self.recognizer.recognize(selected_hands, selected_gestures, selected_features)

        if gesture not in ["NONE", "COOLDOWN", "OTHER", "OPEN", "CLOSED_PALM"]:
            action = self.config.get_mapping(gesture)
            return ModeResult(
                gesture=gesture,
                status_text=f"{gesture} -> {action}",
                status_color=(0, 255, 255),
                action=action,
            )
        return ModeResult(gesture=gesture)

    def _select_presentation_hand(self, hands_landmarks, all_features=None):
        """选择最大的非拳头手作为主手；如果没有非拳头手，选最大的手。"""
        if not hands_landmarks:
            return [], -1

        candidates = []
        for idx, landmarks in enumerate(hands_landmarks):
            features = all_features[idx] if all_features and idx < len(all_features) else self.recognizer.get_hand_features(landmarks)
            hand_width = math.hypot(landmarks[5][1] - landmarks[17][1], landmarks[5][2] - landmarks[17][2])
            candidates.append((features, hand_width, landmarks, idx))

        non_fists = [item for item in candidates if not item[0]["is_fist"]]
        pool = non_fists if non_fists else candidates
        pool.sort(key=lambda item: item[1], reverse=True)

        return [pool[0][2]], pool[0][3]
