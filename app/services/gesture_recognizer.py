import logging
import math
import os
import time
from logging.handlers import RotatingFileHandler

from runtime_paths import writable_data_dir

base_dir = writable_data_dir()
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

    # 正面度门控阈值：hand_frontality < 此值时视为手侧对相机，
    # hand_width 塌缩导致基于掌宽的距离阈值不可靠，回退到 y 坐标判定。
    # 正对相机 ≈0.8，侧到 55° ≈0.45。
    FRONTALITY_GATE = 0.45

    # —— 单手手势阈值（均按掌宽比例，距离自适应）——
    PINCH_RATIO = 0.35              # 捏合：拇指尖↔食指尖 < 掌宽×0.35（单阈值，兼容旧版）
    # 双阈值滞回（实施方案 Phase 3.2）：借鉴 Air-Cursor gestures.py 的滞回设计。
    # ENTER 更严格（距离更小才进入捏合），EXIT 更宽松（距离更大才退出），
    # 滞回带 = EXIT - ENTER = 0.10，与 Air-Cursor 归一化值一致。
    # 标定锚点（文档化，供后人调参）：
    #   - 真实捏合 thumb-index ≈ 掌宽×0.15-0.25（实测）→ ENTER_RATIO=0.30 仍高于真实捏合
    #   - 握拳时 thumb-index ≈ 掌宽×0.50+（fist 距离）→ EXIT_RATIO=0.40 远低于 fist
    #   - 因此握拳永远不会被误判为 pinch 释放，与 Air-Cursor 的教训一致
    #   （Air-Cursor 教训：finger-extension guard 不可靠，距离阈值单独做分离即可。
    #    AC-trae 本就没用 finger-extension guard，这点已做对，缺的是滞回+标定文档）
    PINCH_ENTER_RATIO = 0.30        # 进入捏合：距离 < 掌宽×0.30（比单阈值 0.35 更严格）
    PINCH_EXIT_RATIO = 0.40         # 退出捏合：距离 > 掌宽×0.40（比单阈值 0.35 更宽松）
    INDEX_EXTEND_RATIO = 0.60       # 食指伸出（进入）：食指长 > 掌宽×0.60
    INDEX_EXTEND_EXIT_RATIO = 0.50  # 食指伸出（退出）：食指长 > 掌宽×0.50（滞回，防闪烁）
    THUMB_TUCK_ENTER = 0.62         # 拇指内收进入：拇指尖↔食指 MCP < 掌宽×0.62
    THUMB_TUCK_EXIT = 0.75          # 拇指内收退出：拇指尖↔食指 MCP < 掌宽×0.75（滞后）
    THUMB_EXTEND_RATIO = 0.9        # 拇指伸出：拇指尖↔食指 MCP > 掌宽×0.9
    THUMB_FOLD_RATIO = 0.7          # 拇指折叠辅助判定：< 掌宽×0.7
    # 旋转不变的拇指伸出判定（实施方案 Phase 3.3）：拇指 tip 到掌心中轴的垂直距离
    # / 掌宽。借鉴 opencv-example hand_direction.py 的 perp 设计。
    # 标定锚点：拇指张开时 perp ≈ 0.5+×掌宽，内收横跨掌心时 perp ≈ 0.2×掌宽。
    # 阈值 0.5 为初始值，需实测标定（新旧并存期间用 telemetry 对照）。
    THUMB_PERP_RATIO_THRESHOLD = 0.5
    SCISSOR_SPREAD_RATIO = 0.28     # 剪刀手：食指↔中指间距 > 掌宽×0.28
    MIDDLE_INDEX_RATIO = 0.95       # 中指伸出：中指长/食指长 > 0.95
    FINGERS_CLOSE_RATIO = 0.6       # 手指并拢：相邻指尖 dx < 掌宽×0.6

    # —— 拇指上下判定（基于 tip↔ip 的 y 差）——
    THUMBS_UP_TIP_IP_DELTA = -15    # 拇指上：tip 高于 ip 至少 15px
    # THUMB_DOWN 改为掌宽比例（距离自适应），避免固定 10px 在 1080p 下过松
    THUMBS_DOWN_ENTER_RATIO = 0.30  # 拇指下进入：tip 低于 ip 超过 掌宽×0.30
    THUMBS_DOWN_EXIT_RATIO = 0.20   # 拇指下退出：tip 低于 ip 小于 掌宽×0.20（滞回）

    # —— 挥动判定 ——
    EDGE_RATIO = 0.12               # 画面边缘 12% 区域视为边缘
    EDGE_THRESHOLD_BOOST = 1.5      # 边缘区域挥动阈值放大倍数
    SWIPE_DIR_CONSISTENCY = 0.6     # 挥动方向一致性最低比例

    # —— 滚动判定 ——
    SCROLL_THRESHOLD_RATIO = 1.2    # 滚动触发位移 > 掌宽×1.2
    SCROLL_DIR_CONSISTENCY = 0.55   # 滚动方向一致性最低比例

    # —— 双手手势阈值（按平均掌宽比例）——
    FIST_HUG_RATIO = 1.3            # 双拳靠拢：手腕距 < 平均掌宽×1.3
    PALM_SPREAD_RATIO = 2.2         # 双掌张开：手腕距 > 平均掌宽×2.2

    def __init__(self, cooldown=1.0, swipe_threshold=60):
        self.cooldown = cooldown
        self.swipe_threshold = swipe_threshold
        self.last_action_time = 0
        self.history_x = []
        self.history_y = []
        self._last_hand_width = self.REFERENCE_HAND_WIDTH
        # hand_width 慢速 EMA：仅在 frontality >= FRONTALITY_GATE 时更新，
        # 让手势阈值不再随单帧掌宽抖动（实测 58↔208）跳变。
        # None = 未初始化，首帧用 raw 值初始化（避免冷启动滞后）。
        self._hand_width_ema = None
        self._hand_width_ema_alpha = 0.1  # 慢速：10% 新值 + 90% 历史
        self.history_length = 10
        self.hand_present_frames = 0
        self.scroll_y_history = []
        # 拇指 tucked 滞后状态：避免边界距离来回切换
        self._was_tucked = False
        # 食指伸出滞回状态：避免远距离下关键点抖动导致 index_extended 闪烁
        self._was_index_extended = False
        # pinch 双阈值滞回状态（实施方案 Phase 3.2）：避免边界附近逐帧跳变
        self._was_thumb_index_pinch = False
        self._was_thumb_middle_pinch = False
        # pinch 滞回开关：由 orchestrator 从 config 设置，默认关闭保持旧版行为
        self.pinch_hysteresis_enabled = False
        # thumb_extended 旋转不变判定开关（实施方案 Phase 3.3）：默认关闭，
        # 开启后 thumb_extended 使用 perp_ratio 替代旧的 thumb_tip_to_index_mcp 距离。
        # 新旧特征都计算并输出到 features dict，便于 telemetry 并存对照。
        self.thumb_perp_ratio_enabled = False
        # THUMB_DOWN 滞回状态 + 帧数确认：避免几何阈值边缘抖动导致每秒误触发
        self._was_thumbs_down = False
        self._thumbs_down_confirm_frames = 0
        # 握拳确认帧计数器
        self._fist_confirm_frames = 0
        # 允许并掌姿态丢帧计数器，容忍最大 3 帧的抖动
        self._pose_broken_frames = 0
        # 当前帧画面尺寸（用于边缘检测自适应）
        self.frame_w = 640
        self.frame_h = 480
        logger.info(f"=== GestureRecognizer Started | CD: {cooldown}s, Threshold: {swipe_threshold} ===")

    def get_hand_features(self, landmarks):
        raw_hand_width = max(20.0, math.hypot(landmarks[5][1] - landmarks[17][1], landmarks[5][2] - landmarks[17][2]))
        index_len = math.hypot(landmarks[8][1] - landmarks[5][1], landmarks[8][2] - landmarks[5][2])
        # 正面度代理：掌宽（5↔17）随偏航按 cos 塌缩，而竖直伸出的食指
        # 长度几乎不变。正对相机 ≈0.8，侧到 60° ≈0.4。
        # 提前计算以门控 index_extended（仅在食指伸出时有意义，拳头时
        # index_len 短导致比值偏大，但此时 index_up 为 False，gate 不生效）。
        # 正面度用原始掌宽计算（不受 EMA 滞后影响，否则门控判定会被扭曲）
        hand_frontality = raw_hand_width / max(index_len, 1e-6)
        # hand_width 慢速 EMA：仅在高正面度时更新（低正面度时掌宽塌缩，不可信）。
        # 让手势阈值不再随单帧掌宽抖动（实测 58↔208）跳变。
        if hand_frontality >= self.FRONTALITY_GATE:
            if self._hand_width_ema is None:
                self._hand_width_ema = raw_hand_width
            else:
                self._hand_width_ema = (
                    (1.0 - self._hand_width_ema_alpha) * self._hand_width_ema
                    + self._hand_width_ema_alpha * raw_hand_width
                )
        # 首帧或低正面度未初始化时用 raw 值兜底
        hand_width = self._hand_width_ema if self._hand_width_ema is not None else raw_hand_width

        index_up = landmarks[8][2] < landmarks[6][2]
        middle_up = landmarks[12][2] < landmarks[10][2]
        ring_up = landmarks[16][2] < landmarks[14][2]
        pinky_up = landmarks[20][2] < landmarks[18][2]
        thumb_index = math.hypot(landmarks[4][1] - landmarks[8][1], landmarks[4][2] - landmarks[8][2])
        thumb_middle = math.hypot(landmarks[4][1] - landmarks[12][1], landmarks[4][2] - landmarks[12][2])
        # pinch 判定：单阈值（旧版兼容）或双阈值滞回（实施方案 Phase 3.2）
        # 滞回：已捏合时用 EXIT（更宽松的保持），未捏合时用 ENTER（更严格的进入）
        if getattr(self, 'pinch_hysteresis_enabled', False):
            idx_thresh = hand_width * (
                self.PINCH_EXIT_RATIO if self._was_thumb_index_pinch
                else self.PINCH_ENTER_RATIO
            )
            mid_thresh = hand_width * (
                self.PINCH_EXIT_RATIO if self._was_thumb_middle_pinch
                else self.PINCH_ENTER_RATIO
            )
            thumb_index_pinch = thumb_index < idx_thresh
            thumb_middle_pinch = thumb_middle < mid_thresh
        else:
            pinch_threshold = hand_width * self.PINCH_RATIO
            thumb_index_pinch = thumb_index < pinch_threshold
            thumb_middle_pinch = thumb_middle < pinch_threshold
        self._was_thumb_index_pinch = thumb_index_pinch
        self._was_thumb_middle_pinch = thumb_middle_pinch

        fingers_close = self._check_fingers_close(landmarks, hand_width)
        # 正面度门控 + 滞回：高正面度时用掌宽比例判定（精确）；低正面度时
        # hand_width 塌缩导致阈值过低，回退到 y 坐标判定（偏航不变）。
        # 滞回：已伸出时用更低阈值（INDEX_EXTEND_EXIT_RATIO），防远距离抖动闪烁。
        if hand_frontality >= self.FRONTALITY_GATE:
            if self._was_index_extended:
                index_extended = index_len > hand_width * self.INDEX_EXTEND_EXIT_RATIO
            else:
                index_extended = index_len > hand_width * self.INDEX_EXTEND_RATIO
        else:
            # y 坐标判定也加滞回：已伸出时允许 tip 略低于 PIP（3px 容差）
            if self._was_index_extended:
                index_extended = landmarks[8][2] < landmarks[6][2] + 3
            else:
                index_extended = index_up
        self._was_index_extended = index_extended

        thumb_up = landmarks[4][2] < landmarks[3][2] and landmarks[4][2] < landmarks[2][2]
        thumb_tip_to_index_mcp = math.hypot(landmarks[4][1] - landmarks[5][1], landmarks[4][2] - landmarks[5][2])
        # 滞后阈值：进入 tucked 需要更近（0.62），退出 tucked 需要更远（0.75）。
        # 与 v1.1.0 的 0.65/0.78（按 |Δx| 掌宽）等效——掌宽改为欧氏距离时
        # 阈值被过度压缩到 0.5/0.6，导致书写中拇指频繁被误判为分开。
        if self._was_tucked:
            thumb_tucked = thumb_tip_to_index_mcp < hand_width * self.THUMB_TUCK_EXIT
        else:
            thumb_tucked = thumb_tip_to_index_mcp < hand_width * self.THUMB_TUCK_ENTER
        self._was_tucked = thumb_tucked
        # thumb_extended 双路判定（实施方案 Phase 3.3）：新旧并存对照
        # 旧特征：thumb_tip_to_index_mcp 距离（依赖拇指与食指 MCP 的绝对距离）
        # 新特征：_thumb_perp_ratio（旋转不变，拇指 tip 到掌心中轴的垂直距离/掌宽）
        thumb_extended_old = thumb_tip_to_index_mcp > hand_width * self.THUMB_EXTEND_RATIO
        thumb_perp_ratio = self._thumb_perp_ratio(landmarks)
        thumb_extended_new = thumb_perp_ratio > self.THUMB_PERP_RATIO_THRESHOLD
        # config 开关决定 thumb_extended 最终用哪个（默认旧版，可回退）
        if getattr(self, 'thumb_perp_ratio_enabled', False):
            thumb_extended = thumb_extended_new
        else:
            thumb_extended = thumb_extended_old

        thumb_folded = thumb_tucked or (not thumb_up and not thumb_extended) or \
                      (not thumb_up and thumb_tip_to_index_mcp < hand_width * self.THUMB_FOLD_RATIO)

        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        thumb_mcp = landmarks[2]
        thumb_tip_to_ip = thumb_tip[2] - thumb_ip[2]
        is_thumbs_up = (
            thumb_up
            and thumb_tip[2] < thumb_mcp[2]
            and thumb_tip_to_ip < self.THUMBS_UP_TIP_IP_DELTA
        )
        # THUMB_DOWN 滞回判定（掌宽比例自适应）：避免固定 10px 在 1080p 下过松
        thumbs_down_threshold = hand_width * (
            self.THUMBS_DOWN_EXIT_RATIO if self._was_thumbs_down
            else self.THUMBS_DOWN_ENTER_RATIO
        )
        is_thumbs_down = (
            not thumb_up
            and thumb_tip[2] > thumb_mcp[2]
            and thumb_tip[2] > thumb_ip[2]
            and thumb_tip_to_ip > thumbs_down_threshold
        )
        self._was_thumbs_down = is_thumbs_down

        four_fingers_down = not index_up and not middle_up and not ring_up and not pinky_up

        index_middle_spread = math.hypot(
            landmarks[8][1] - landmarks[12][1], landmarks[8][2] - landmarks[12][2]
        )
        scissor_spread_ok = index_middle_spread > hand_width * self.SCISSOR_SPREAD_RATIO
        is_scissor = (
            index_up and middle_up and not ring_up and not pinky_up
            and scissor_spread_ok
        )

        # 中指是否与食指一样伸出：用"中指长/食指长"判定。两指同手同投影，
        # 该比值不随偏航/距离变化。实测标定（2026-06-12 gesture.log）：
        # 单指书写时弯曲中指 mi 中位数 0.36、最大 0.81；真双指 ≥1.15。
        # 0.95 取分离带中点。此前用"中指长>掌宽×0.6"：弯曲中指的 2D 投影
        # 经常超过该值（偏航时掌宽塌缩进一步加剧），16/24 次断笔由此而来。
        middle_len = math.hypot(landmarks[12][1] - landmarks[9][1], landmarks[12][2] - landmarks[9][2])
        middle_index_ratio = middle_len / max(index_len, 1e-6)
        middle_extended = middle_index_ratio > self.MIDDLE_INDEX_RATIO

        # 双指悬停（板书抬笔）：食指+中指伸出即可，**不要求张开**（贴紧也算，
        # 区别于 is_scissor 的 spread 要求）。伸出的手指在轮廓上凸出，
        # 手侧对相机时剪影依然可辨，是偏航下最可靠的抬笔信号。
        two_finger_hover = index_extended and middle_extended and not ring_up and not pinky_up

        return {
            "index_up": index_up,
            "middle_up": middle_up,
            "ring_up": ring_up,
            "pinky_up": pinky_up,
            "index_only": index_up and not middle_up and not ring_up and not pinky_up,
            "index_drawing": index_up and not ring_up and not pinky_up,
            "thumb_index_pinch": thumb_index_pinch,
            "thumb_middle_pinch": thumb_middle_pinch,
            "is_fist": four_fingers_down and thumb_folded,
            "is_open_palm": index_up and middle_up and ring_up,
            "thumb_tucked": thumb_tucked,
            "thumb_extended": thumb_extended,
            # Phase 3.3: 旋转不变新特征（始终输出，供 telemetry 并存对照）
            "thumb_perp_ratio": thumb_perp_ratio,
            "thumb_extended_new": thumb_extended_new,
            "thumb_writing": index_extended and not middle_up and not ring_up and not pinky_up and not thumb_extended,
            "index_drawing_pose": index_extended and not middle_up and not ring_up and not pinky_up,
            "index_extended": index_extended,
            "middle_extended": middle_extended,
            "two_finger_hover": two_finger_hover,
            "hand_frontality": hand_frontality,
            "thumb_ratio": thumb_tip_to_index_mcp / hand_width,
            "middle_ratio": middle_len / hand_width,
            "middle_index_ratio": middle_index_ratio,
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
        # 双判定：固定 60px 兜底 + 掌宽比例。
        # 纯比例在远处掌窄时阈值过严（30px→18px），近处过松（150px→90px）。
        is_close1 = dx1 < 60 or dx1 < hand_width * self.FINGERS_CLOSE_RATIO
        is_close2 = dx2 < 60 or dx2 < hand_width * self.FINGERS_CLOSE_RATIO
        is_close3 = dx3 < 60 or dx3 < hand_width * self.FINGERS_CLOSE_RATIO
        return is_close1 and is_close2 and is_close3

    def _thumb_perp_ratio(self, landmarks):
        """拇指 tip 到掌心中轴的垂直距离（归一化到掌宽）——旋转不变判定。

        借鉴 opencv-example hand_direction.py 的 perp 设计（实施方案 Phase 3.3）：
        - 掌心中轴：wrist(0) → middle MCP(9)
        - 垂直距离：thumb tip(4) 到该中轴的垂线距离
        - 归一化：除以掌宽（landmark 5↔17 距离）

        优势：
        - 旋转不变：不管手怎么转，拇指张开时 tip 总偏离中轴
        - 区分内收vs张开：拇指内收横跨掌心时 tip 在中轴附近，perp 小
        - 归一化到掌宽：距离自适应，不依赖固定像素
        """
        w_x, w_y = landmarks[0][1], landmarks[0][2]
        mcp_x, mcp_y = landmarks[9][1], landmarks[9][2]
        t_x, t_y = landmarks[4][1], landmarks[4][2]
        # 掌心中轴方向向量
        dx = mcp_x - w_x
        dy = mcp_y - w_y
        axis_len = math.hypot(dx, dy)
        if axis_len < 1e-6:
            return 0.0
        # 垂直距离 = |(t-w) × axis_dir| / |axis_dir|
        # （叉积的绝对值 = 平行四边形面积 / 底边 = 高 = 垂直距离）
        perp = abs((t_x - w_x) * dy - (t_y - w_y) * dx) / axis_len
        # 归一化到掌宽
        palm_width = max(20.0, math.hypot(
            landmarks[5][1] - landmarks[17][1],
            landmarks[5][2] - landmarks[17][2]
        ))
        return perp / palm_width

    def _reset_state(self):
        self.last_action_time = time.time()
        self.history_x.clear()
        self.history_y.clear()
        self._pose_broken_frames = 0
        # 清理残留状态，防止模式切换后误触发
        self._was_tucked = False
        # 清理 pinch 滞回状态（实施方案 Phase 3.2）
        self._was_thumb_index_pinch = False
        self._was_thumb_middle_pinch = False
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
        edge_ratio = self.EDGE_RATIO  # 画面边缘比例区域视为边缘
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
        effective_threshold = base_threshold * (self.EDGE_THRESHOLD_BOOST if in_edge else 1.0)
        min_avg_speed = base_threshold / 4.0

        if abs(dx) < effective_threshold and abs(dy) < effective_threshold:
            return "NONE"

        pairwise_dx = [self.history_x[i+1] - self.history_x[i] for i in range(len(self.history_x) - 1)]
        pairwise_dy = [self.history_y[i+1] - self.history_y[i] for i in range(len(self.history_y) - 1)]

        if abs(dx) > abs(dy):
            direction = 1 if dx > 0 else -1
            consistent = sum(1 for d in pairwise_dx if d * direction > 0)
            ratio = consistent / len(pairwise_dx)
            if ratio < self.SWIPE_DIR_CONSISTENCY:
                logger.info(f"Swipe rejected: direction consistency {ratio:.2f} < {self.SWIPE_DIR_CONSISTENCY}")
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
            if ratio < self.SWIPE_DIR_CONSISTENCY:
                logger.info(f"Swipe rejected: direction consistency {ratio:.2f} < {self.SWIPE_DIR_CONSISTENCY}")
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
        # 双判定：掌宽比例 + 固定 60px 下限。
        # 纯比例在远处掌窄时阈值过小（30px→36px），手腕自然抖动即误触发滚动。
        threshold = max(hand_width * self.SCROLL_THRESHOLD_RATIO, 60)
        if abs(dy) > threshold:
            direction = 1 if dy > 0 else -1
            consistent = sum(1 for i in range(len(self.scroll_y_history) - 1)
                            if (self.scroll_y_history[i+1] - self.scroll_y_history[i]) * direction > 0)
            ratio = consistent / (len(self.scroll_y_history) - 1)
            if ratio < self.SCROLL_DIR_CONSISTENCY:
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

        # 距离自适应：双手手腕距离阈值按平均掌宽归一化，
        # 使远处（掌窄）和近处（掌宽）的触发比例一致。
        # 参考掌宽 90px 时：FIST_HUG≈120px(1.3×)，TWO_PALM_SPREAD≈200px(2.2×)
        avg_hand_width = (first.get("hand_width", 90.0) + second.get("hand_width", 90.0)) / 2.0
        fist_hug_threshold = avg_hand_width * self.FIST_HUG_RATIO
        palm_spread_threshold = avg_hand_width * self.PALM_SPREAD_RATIO

        self._two_hand_hold_frames = getattr(self, '_two_hand_hold_frames', 0) + 1
        if self._two_hand_hold_frames < 8:
            return None

        self._two_hand_hold_frames = 0

        if both_fists and wrist_dist < fist_hug_threshold:
            logger.info("=> Trigger: FIST_HUG (wrist_dist=%.0f, threshold=%.0f)", wrist_dist, fist_hug_threshold)
            self._reset_state()
            return "FIST_HUG"
        elif both_open and wrist_dist > palm_spread_threshold:
            logger.info("=> Trigger: TWO_PALM_SPREAD (wrist_dist=%.0f, threshold=%.0f)", wrist_dist, palm_spread_threshold)
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
            self._thumbs_down_confirm_frames += 1
            if self._thumbs_down_confirm_frames >= 3:
                logger.info("=> Trigger: THUMB_DOWN (geometric, confirmed %d frames)", self._thumbs_down_confirm_frames)
                self._thumbs_down_confirm_frames = 0
                self._reset_state()
                return "THUMB_DOWN"
        else:
            self._thumbs_down_confirm_frames = 0

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
