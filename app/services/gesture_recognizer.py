import time
import logging
import os
import sys
import math
from logging.handlers import RotatingFileHandler

if getattr(sys, 'frozen', False):
    base_dir = os.path.dirname(sys.executable)
else:
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

log_file = os.path.join(base_dir, 'gesture.log')
logger = logging.getLogger('gesture')
if not logger.handlers:
    handler = RotatingFileHandler(
        log_file, maxBytes=2 * 1024 * 1024, backupCount=2, encoding='utf-8'
    )
    handler.setFormatter(logging.Formatter(
        '%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    ))
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

class GestureRecognizer:
    # 挥动阈值的参考手宽（landmark 5↔17 的掌宽，单位 px）：
    # 在"舒适距离"下掌宽约等于该值，此时阈值即为配置的 swipe_threshold。
    # 距离变远→掌宽变小→阈值按比例下调，使同一物理挥动在远处也能触发。
    REFERENCE_HAND_WIDTH = 90.0

    def __init__(self, cooldown=1.0, swipe_threshold=60):
        self.cooldown = cooldown
        self.swipe_threshold = swipe_threshold
        self.last_action_time = 0
        self.history_x = []
        self.history_y = []
        self._last_hand_width = self.REFERENCE_HAND_WIDTH
        self.history_length = 10
        self.hand_present_frames = 0
        self.scroll_y_history = []
        # 拇指 tucked 滞后状态：避免边界距离来回切换
        self._was_tucked = False
        # 握拳确认帧计数器
        self._fist_confirm_frames = 0
        # 允许并掌姿态丢帧计数器，容忍最大 3 帧的抖动
        self._pose_broken_frames = 0
        # 当前帧画面尺寸（用于边缘检测自适应）
        self.frame_w = 640
        self.frame_h = 480
        logger.info(f"=== GestureRecognizer Started | CD: {cooldown}s, Threshold: {swipe_threshold} ===")

    def get_hand_features(self, landmarks):
        hand_width = max(20.0, math.hypot(landmarks[5][1] - landmarks[17][1], landmarks[5][2] - landmarks[17][2]))
        index_up = landmarks[8][2] < landmarks[6][2]
        middle_up = landmarks[12][2] < landmarks[10][2]
        ring_up = landmarks[16][2] < landmarks[14][2]
        pinky_up = landmarks[20][2] < landmarks[18][2]
        thumb_index = math.hypot(landmarks[4][1] - landmarks[8][1], landmarks[4][2] - landmarks[8][2])
        thumb_middle = math.hypot(landmarks[4][1] - landmarks[12][1], landmarks[4][2] - landmarks[12][2])
        pinch_threshold = hand_width * 0.35

        fingers_close = self._check_fingers_close(landmarks, hand_width)

        index_len = math.hypot(landmarks[8][1] - landmarks[5][1], landmarks[8][2] - landmarks[5][2])
        index_extended = index_len > hand_width * 0.65

        thumb_up = landmarks[4][2] < landmarks[3][2] and landmarks[4][2] < landmarks[2][2]
        thumb_tip_to_index_mcp = math.hypot(landmarks[4][1] - landmarks[5][1], landmarks[4][2] - landmarks[5][2])
        # 滞后阈值：进入 tucked 需要更近（0.65），退出 tucked 需要更远（0.78）
        # 避免拇指在边界距离时书写/悬停状态来回抖动
        if self._was_tucked:
            thumb_tucked = thumb_tip_to_index_mcp < hand_width * 0.6
        else:
            thumb_tucked = thumb_tip_to_index_mcp < hand_width * 0.5
        self._was_tucked = thumb_tucked
        thumb_extended = thumb_tip_to_index_mcp > hand_width * 0.9

        thumb_folded = thumb_tucked or (not thumb_up and not thumb_extended) or \
                      (not thumb_up and thumb_tip_to_index_mcp < hand_width * 0.7)

        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        thumb_mcp = landmarks[2]
        thumb_tip_to_ip = thumb_tip[2] - thumb_ip[2]
        is_thumbs_up = (
            thumb_up
            and thumb_tip[2] < thumb_mcp[2]
            and thumb_tip_to_ip < -15
        )
        is_thumbs_down = (
            not thumb_up
            and thumb_tip[2] > thumb_mcp[2]
            and thumb_tip[2] > thumb_ip[2]
            and thumb_tip_to_ip > 10
        )

        four_fingers_down = not index_up and not middle_up and not ring_up and not pinky_up

        index_middle_spread = math.hypot(
            landmarks[8][1] - landmarks[12][1], landmarks[8][2] - landmarks[12][2]
        )
        scissor_spread_ok = index_middle_spread > hand_width * 0.28
        is_scissor = (
            index_up and middle_up and not ring_up and not pinky_up
            and scissor_spread_ok
        )

        return {
            "index_up": index_up,
            "middle_up": middle_up,
            "ring_up": ring_up,
            "pinky_up": pinky_up,
            "index_only": index_up and not middle_up and not ring_up and not pinky_up,
            "index_drawing": index_up and not ring_up and not pinky_up,
            "thumb_index_pinch": thumb_index < pinch_threshold,
            "thumb_middle_pinch": thumb_middle < pinch_threshold,
            "is_fist": four_fingers_down and thumb_folded,
            "is_open_palm": index_up and middle_up and ring_up and pinky_up and thumb_up,
            "thumb_tucked": thumb_tucked,
            "thumb_extended": thumb_extended,
            "thumb_writing": index_extended and not middle_up and not ring_up and not pinky_up and not thumb_extended,
            "index_drawing_pose": index_extended and not middle_up and not ring_up and not pinky_up,
            "fingers_close": fingers_close,
            "hand_width": hand_width,
            "index_middle_up": is_scissor,
            "is_thumbs_up": is_thumbs_up,
            "is_thumbs_down": is_thumbs_down,
            "is_scissor": is_scissor,
        }

    def _check_fingers_close(self, landmarks, hand_width):
        dx1 = abs(landmarks[8][1] - landmarks[12][1])
        dx2 = abs(landmarks[12][1] - landmarks[16][1])
        dx3 = abs(landmarks[16][1] - landmarks[20][1])
        is_close1 = dx1 < 60 or dx1 < hand_width * 0.6
        is_close2 = dx2 < 60 or dx2 < hand_width * 0.6
        is_close3 = dx3 < 60 or dx3 < hand_width * 0.6
        return is_close1 and is_close2 and is_close3

    def _reset_state(self):
        self.last_action_time = time.time()
        self.history_x.clear()
        self.history_y.clear()
        self._pose_broken_frames = 0
        # 清理残留状态，防止模式切换后误触发
        self._was_tucked = False
        if hasattr(self, '_scissor_frames'):
            self._scissor_frames = 0
        if hasattr(self, '_scroll_direction'):
            self._scroll_direction = 0
        if hasattr(self, '_scroll_frames'):
            self._scroll_frames = 0
        if hasattr(self, '_two_hand_hold_frames'):
            self._two_hand_hold_frames = 0
        if hasattr(self, '_fist_confirm_frames'):
            self._fist_confirm_frames = 0

    def _check_swipe(self):
        if len(self.history_x) < 4:
            return "NONE"
        dx = self.history_x[-1] - self.history_x[0]
        dy = self.history_y[-1] - self.history_y[0]

        # 边缘区域灵敏度降级：轨迹起点或终点靠近画面边缘时提高阈值
        # 解决 B11：手在边缘做非挥动动作时的误触发
        # 使用相对边缘比例而非硬编码像素值，适配不同摄像头分辨率
        edge_ratio = 0.12  # 画面边缘 12% 区域视为边缘
        edge_x = self.frame_w * edge_ratio
        edge_y = self.frame_h * edge_ratio
        x_vals = (self.history_x[0], self.history_x[-1])
        y_vals = (self.history_y[0], self.history_y[-1])
        in_edge = any(v < edge_x or v > self.frame_w - edge_x for v in x_vals) or \
                  any(v < edge_y or v > self.frame_h - edge_y for v in y_vals)

        # 距离自适应：阈值随掌宽（距离）缩放，使远处的小幅挥动也能触发；
        # clamp 防止极远处噪声放大成误触发 / 极近处阈值过高。
        dist_scale = min(1.5, max(0.3, self._last_hand_width / self.REFERENCE_HAND_WIDTH))
        base_threshold = self.swipe_threshold * dist_scale
        effective_threshold = base_threshold * (1.5 if in_edge else 1.0)
        min_avg_speed = base_threshold / 4.0

        if abs(dx) < effective_threshold and abs(dy) < effective_threshold:
            return "NONE"

        pairwise_dx = [self.history_x[i+1] - self.history_x[i] for i in range(len(self.history_x) - 1)]
        pairwise_dy = [self.history_y[i+1] - self.history_y[i] for i in range(len(self.history_y) - 1)]

        if abs(dx) > abs(dy):
            direction = 1 if dx > 0 else -1
            consistent = sum(1 for d in pairwise_dx if d * direction > 0)
            ratio = consistent / len(pairwise_dx)
            if ratio < 0.6:
                logger.info(f"Swipe rejected: direction consistency {ratio:.2f} < 0.6")
                return "NONE"
            avg_speed = abs(dx) / len(self.history_x)
            if avg_speed < min_avg_speed:
                logger.info(f"Swipe rejected: avg_speed {avg_speed:.1f} too low (min {min_avg_speed:.1f})")
                return "NONE"
            if dx > effective_threshold:
                logger.info(f"=> Trigger: SWIPE_RIGHT (consistency={ratio:.2f}, speed={avg_speed:.1f}, edge={in_edge})")
                self._reset_state()
                return "SWIPE_RIGHT"
            elif dx < -effective_threshold:
                logger.info(f"=> Trigger: SWIPE_LEFT (consistency={ratio:.2f}, speed={avg_speed:.1f}, edge={in_edge})")
                self._reset_state()
                return "SWIPE_LEFT"
        else:
            direction = 1 if dy > 0 else -1
            consistent = sum(1 for d in pairwise_dy if d * direction > 0)
            ratio = consistent / len(pairwise_dy)
            if ratio < 0.6:
                logger.info(f"Swipe rejected: direction consistency {ratio:.2f} < 0.6")
                return "NONE"
            avg_speed = abs(dy) / len(self.history_y)
            if avg_speed < min_avg_speed:
                logger.info(f"Swipe rejected: avg_speed {avg_speed:.1f} too low (min {min_avg_speed:.1f})")
                return "NONE"
            if dy < -effective_threshold:
                logger.info(f"=> Trigger: SWIPE_UP (consistency={ratio:.2f}, speed={avg_speed:.1f}, edge={in_edge})")
                self._reset_state()
                return "SWIPE_UP"
            elif dy > effective_threshold:
                logger.info(f"=> Trigger: SWIPE_DOWN (consistency={ratio:.2f}, speed={avg_speed:.1f}, edge={in_edge})")
                self._reset_state()
                return "SWIPE_DOWN"
        return "NONE"

    def check_scroll(self, landmarks, features, is_scroll_active):
        if not features["index_middle_up"]:
            self.scroll_y_history.clear()
            self._scroll_direction = 0
            self._scroll_frames = 0
            return 0

        wrist_y = landmarks[0][2]
        hand_width = features.get("hand_width", 40.0)
        self.scroll_y_history.append(wrist_y)
        if len(self.scroll_y_history) > 12:
            self.scroll_y_history.pop(0)

        # 已有活跃滚动方向时，保持该方向持续输出
        if getattr(self, '_scroll_direction', 0) != 0:
            self._scroll_frames += 1
            # 每 3 帧输出一次滚动事件，避免过于频繁
            if self._scroll_frames % 3 == 0:
                return self._scroll_direction
            return 0

        if len(self.scroll_y_history) < 6:
            return 0

        dy = self.scroll_y_history[-1] - self.scroll_y_history[0]
        threshold = max(hand_width * 1.2, 60)
        if abs(dy) > threshold:
            direction = 1 if dy > 0 else -1
            consistent = sum(1 for i in range(len(self.scroll_y_history) - 1)
                            if (self.scroll_y_history[i+1] - self.scroll_y_history[i]) * direction > 0)
            ratio = consistent / (len(self.scroll_y_history) - 1)
            if ratio < 0.55:
                return 0
            # 锁定滚动方向，进入持续滚动模式
            self._scroll_direction = direction
            self._scroll_frames = 0
            # 清空历史但保留方向，让手腕继续移动可以累积新的位移
            self.scroll_y_history.clear()
            return direction
        return 0

    def check_two_hand_gesture(self, hands_landmarks, hands_features=None):
        if len(hands_landmarks) < 2:
            self._two_hand_hold_frames = 0
            return None

        if hands_features and len(hands_features) >= 2:
            first, second = hands_features[0], hands_features[1]
        else:
            first = self.get_hand_features(hands_landmarks[0])
            second = self.get_hand_features(hands_landmarks[1])

        both_fists = first["is_fist"] and second["is_fist"]
        both_open = first["is_open_palm"] and second["is_open_palm"]

        if not (both_fists or both_open):
            self._two_hand_hold_frames = 0
            return None

        wrist_dist = math.hypot(
            hands_landmarks[0][0][1] - hands_landmarks[1][0][1],
            hands_landmarks[0][0][2] - hands_landmarks[1][0][2],
        )

        self._two_hand_hold_frames = getattr(self, '_two_hand_hold_frames', 0) + 1
        if self._two_hand_hold_frames < 8:
            return None

        self._two_hand_hold_frames = 0

        if both_fists and wrist_dist < 120:
            logger.info("=> Trigger: FIST_HUG (wrist_dist=%.0f)", wrist_dist)
            self._reset_state()
            return "FIST_HUG"
        elif both_open and wrist_dist > 200:
            logger.info("=> Trigger: TWO_PALM_SPREAD (wrist_dist=%.0f)", wrist_dist)
            self._reset_state()
            return "TWO_PALM_SPREAD"

        return None

    def recognize(self, hands_landmarks, hands_gestures=None, hand_features=None):
        if time.time() - self.last_action_time < self.cooldown:
            return "COOLDOWN"

        if not hands_landmarks:
            if len(self.history_x) > 0 or self.hand_present_frames > 0:
                logger.info("Hand lost from frame. Cleared trajectory.")
            self.history_x.clear()
            self.history_y.clear()
            self.hand_present_frames = 0
            return "NONE"

        self.hand_present_frames += 1

        if self.hand_present_frames < 10:
            return "NONE"

        landmarks = hands_landmarks[0]
        # 记录当前掌宽，用于挥动阈值的距离自适应（远处掌窄→阈值下调）
        self._last_hand_width = max(
            20.0,
            math.hypot(landmarks[5][1] - landmarks[17][1], landmarks[5][2] - landmarks[17][2]),
        )
        ml_label = "OTHER"
        if hands_gestures and len(hands_gestures) > 0:
            ml_label = hands_gestures[0].get("label", "OTHER")

        features = hand_features if hand_features else self.get_hand_features(landmarks)

        if ml_label == "THUMB_UP":
            if features["is_thumbs_down"]:
                logger.info("=> Trigger: THUMB_DOWN")
                self._reset_state()
                return "THUMB_DOWN"
            logger.info("=> Trigger: THUMB_UP (ML: Thumb_Up)")
            self._reset_state()
            return "THUMB_UP"

        # MediaPipe 直接识别出 Thumb_Down：远距离时关键点抖动导致几何判定
        # (is_thumbs_down) 不可靠，必须信任 ML 标签，否则倒赞会被当作"姿势中断"丢弃，
        # 导致远处无法挂断豆包通话。（修复：此前缺少 THUMB_DOWN 的 ML 分支。）
        if ml_label == "THUMB_DOWN":
            logger.info("=> Trigger: THUMB_DOWN (ML: Thumb_Down)")
            self._reset_state()
            return "THUMB_DOWN"

        if features["is_thumbs_down"] and not features["index_up"] and not features["middle_up"] and not features["ring_up"] and not features["pinky_up"]:
            logger.info("=> Trigger: THUMB_DOWN (geometric)")
            self._reset_state()
            return "THUMB_DOWN"

        if ml_label == "FIST":
            logger.info("=> Trigger: FIST (ML)")
            self._reset_state()
            return "FIST"
        if ml_label in ("OTHER", "None") and features["is_fist"]:
            self._fist_confirm_frames += 1
            if self._fist_confirm_frames >= 3:
                logger.info("=> Trigger: FIST (fallback, confirmed %d frames)", self._fist_confirm_frames)
                self._fist_confirm_frames = 0
                self._reset_state()
                return "FIST"
        else:
            self._fist_confirm_frames = 0

        if features["is_scissor"]:
            self._scissor_frames = getattr(self, '_scissor_frames', 0) + 1
            if self._scissor_frames >= 60:
                logger.info("=> Trigger: SCISSOR (hold 2s)")
                self._scissor_frames = 0
                self._reset_state()
                return "SCISSOR"
        else:
            if hasattr(self, '_scissor_frames'):
                self._scissor_frames = 0

        is_closed_palm = ml_label in ("OTHER", "OPEN") and features["fingers_close"]
        if is_closed_palm:
            self._pose_broken_frames = 0
            wrist_x = landmarks[0][1]
            wrist_y = landmarks[0][2]
            self.history_x.append(wrist_x)
            self.history_y.append(wrist_y)

            if len(self.history_x) > self.history_length:
                self.history_x.pop(0)
                self.history_y.pop(0)

            swipe_result = self._check_swipe()
            if swipe_result != "NONE":
                return swipe_result
        else:
            if len(self.history_x) > 0:
                self._pose_broken_frames += 1
                if self._pose_broken_frames > 3:
                    logger.info(f"Pose broken to ml={ml_label} for {self._pose_broken_frames} frames. Cleared trajectory.")
                    self.history_x.clear()
                    self.history_y.clear()
                    self._pose_broken_frames = 0

        return "NONE"
