import json
import logging
import math
import queue
import threading
import time
import winsound

from runtime_paths import data_path
from services.mouse_controller import ActiveRegionMapper, interp_tiers

from .base import ModeBase, ModeResult

logger = logging.getLogger("gesture")


class _TraceWriter:
    """Small bounded writer so debug tracing never blocks the Qt thread."""

    def __init__(self, path):
        self._queue = queue.Queue(maxsize=512)
        self._closed = False
        self._thread = threading.Thread(
            target=self._run,
            args=(path,),
            name="DrawTraceWriter",
            daemon=True,
        )
        self._thread.start()

    def write(self, line):
        if self._closed:
            return
        try:
            self._queue.put_nowait(line)
        except queue.Full:
            # Debug tracing is best-effort. Keeping UI latency stable matters
            # more than retaining every sample during disk stalls.
            pass

    def _run(self, path):
        try:
            with open(path, "w", encoding="utf-8", buffering=1) as stream:
                while True:
                    line = self._queue.get()
                    if line is None:
                        return
                    stream.write(line)
        except OSError:
            logger.exception("无法写入板书轨迹: %s", path)

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self._queue.put(None, timeout=0.5)
        except queue.Full:
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(None)
            except (queue.Empty, queue.Full):
                return
        self._thread.join(timeout=1.0)


class DrawMode(ModeBase):
    """板书模式：拇指并拢落笔、分开抬笔（仅手正对相机时拇指可信）；
    ✌️ 食指+中指伸出（贴紧也算）随时抬笔——侧面剪影可辨，不依赖拇指。
    手侧偏时拇指被遮挡，冻结笔状态防误抬。张掌清屏，握拳就绪。"""

    # 移动灵敏度随"手在画面中的大小"换算的标定区间（相对参考掌宽的比值）。
    # 我们无法感知真实距离，只能用掌宽作为手在画面里大小的度量：
    #   掌宽 ≤ 参考×RESP_RATIO_FAR（手小/离得远）→ 最敏感（系数 1.0）；
    #   掌宽 ≥ 参考×RESP_RATIO_NEAR（手大/离得近）→ 最迟钝（系数 0.0）。
    RESP_RATIO_FAR = 0.55
    RESP_RATIO_NEAR = 1.5
    # 距离分段「活动区 span_floor」（piecewise-linear）：ratio=掌宽/参考掌宽（越大=越近）。
    # span_floor 越小→小动作越被放大（远距离写满全屏）；越大（→1.0）→趋近直接绝对映射、
    # gain≈1、画圆轻松（找回早期卡尔曼版的近距离手感），且不牺牲触达（近距离手本就充满画面）。
    # 嫌近距离还太快就把近端调到 1.0~1.2（>1 会略损触达）；想远距离更省力就把远端调小。
    SPAN_FLOOR_TIERS = [
        (0.45, 0.22),  # 很远：小动作放大、写满全屏
        (0.70, 0.45),  # 远
        (1.00, 0.90),  # 中（舒适书写距离）：接近直接映射
        (1.30, 1.08),  # 近：抵消 5% margin 的额外放大
        (1.90, 1.15),  # 很近：略降增益，增强小范围移动精度
    ]

    # 中央投票笔状态机：决定起落前，时间窗内至少要有 VOTE_MIN 帧证据，
    # 防止窗口刚填充时凭 1-2 帧的比例噪声误触（低帧率下尤甚）。
    # v1.3.6 稳定档回到 v1.3 的保守门槛：宁可起笔慢一两帧，也优先避免断笔。
    VOTE_MIN = 3

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._was_writing = False
        self._writing_lost_frames = 0
        self.fist_hold_frames = 0
        self.open_palm_frames = 0
        self._last_special_action_time = 0
        self._was_fist = False
        self._fist_tap_times: list[float] = []
        self._double_fist_cooldown = 0.0
        # 自适应活动区映射：远距离也能写满全屏（替代固定的绝对映射）
        self._region_mapper = ActiveRegionMapper(margin=0.05)
        # 双指悬停 / 拇指分开的连续帧计数（去抖）与滞后状态
        self._two_finger_frames = 0
        self._two_finger_hovering = False
        self._thumb_apart_frames = 0
        # 标定遥测节流
        self._last_telemetry = 0.0
        # 正面度低于此值视为拇指不可观测（侧对相机），冻结笔状态。
        # 0.65 保留为拇指交互的高置信门槛；稳定档默认关闭拇指抬笔。
        self._frontality_gate = (
            float(self.config.get("draw_frontality_gate", 0.65)) if self.config else 0.65
        )
        # 拇指分开抬笔：v1.3.6 稳定档默认关闭，吸收 v1.3 的“单指书写优先”
        # 手感，避免近距正面书写时拇指距离抖动造成 vote_hover 断笔。
        # 需要旧习惯时可在配置中设 draw_thumb_lift=true。
        self._thumb_lift = (
            bool(self.config.get("draw_thumb_lift", False)) if self.config else False
        )
        # ✌️ 双指抬笔的几何兜底：默认关闭。实测（2026-06-13）侧视书写时中指 2D
        # 投影使 mi 频繁 >0.95，单指被误判成双指而中途断笔（一会话 12 次误判
        # vs 1 次真几何双指）。关闭后双指抬笔只认可靠的 VICTORY ML 标签；
        # 设 True 恢复几何兜底（仅在正对相机、几何可信时建议开）。
        self._two_finger_geom = (
            bool(self.config.get("draw_two_finger_geom", False)) if self.config else False
        )
        # 中央投票笔状态机状态：滑动时间窗内的 (时间戳, 帧分类) 与对应屏幕坐标。
        # 与 🤟 切模式同构——单帧只投票、不直接决定笔的起落。窗口与多数票
        # 比例可由 config 调（默认 0.3s/0.6，对应实录回放验证的 ~7 帧/0.71）。
        self._vote: list[tuple[float, str]] = []
        self._recent_points: list[tuple[float, float, float]] = []
        self._vote_window = (
            float(self.config.get("draw_vote_window_sec", 0.30)) if self.config else 0.30
        )
        self._vote_ratio = (
            float(self.config.get("draw_vote_ratio", 0.60)) if self.config else 0.60
        )
        # 书写中"张掌立即抬笔"的连续帧去抖：is_open_palm 单帧噪声（middle_up &
        # ring_up 偶发同真）此前会绕过投票窗立即 force_lift_pen → 断笔。改为需连续
        # N 帧确认；真张掌持续会很快累计到阈值再抬笔，几乎无感。1=恢复旧的单帧行为。
        self._open_palm_lift_min = (
            int(self.config.get("draw_open_palm_lift_frames", 3)) if self.config else 3
        )
        self._open_palm_lift_frames = 0
        # 全帧率关键点轨迹录制（draw_trace.jsonl，项目根目录，会话开始时清空）：
        # 供 simulate_draw.py --replay 离线回放，用真实数据对比判定逻辑变体
        self._trace_writer = None

    def on_enter(self):
        if (
            self._trace_writer is None
            and self.config is not None
            and self.config.get("draw_record_trace", False)
        ):
            self._trace_writer = _TraceWriter(data_path("draw_trace.jsonl"))
        self.overlay.show_fullscreen()
        self.toolbar.show()
        self._position_toolbar()
        self.cursor_overlay.hide()
        self.cursor_overlay.setGeometry(-100, -100, 0, 0)
        self._was_writing = False
        self._writing_lost_frames = 0
        self.fist_hold_frames = 0
        self.open_palm_frames = 0
        self._was_fist = False
        self._fist_tap_times.clear()
        self._double_fist_cooldown = 0.0
        self._two_finger_frames = 0
        self._two_finger_hovering = False
        self._thumb_apart_frames = 0
        self._open_palm_lift_frames = 0
        self._vote.clear()
        self._recent_points.clear()
        self._region_mapper.reset()
        if self.recognizer:
            self.recognizer._reset_state()

    def on_exit(self):
        if self._trace_writer is not None:
            self._trace_writer.close()
            self._trace_writer = None
        self.overlay.hide()
        self.overlay.setGeometry(-100, -100, 0, 0)
        self.overlay.force_lift_pen()
        self.overlay.hide_cursor()
        self.toolbar.hide()
        self._was_writing = False
        self._writing_lost_frames = 0
        self.fist_hold_frames = 0
        self.open_palm_frames = 0
        self._was_fist = False
        self._fist_tap_times.clear()
        self._double_fist_cooldown = 0.0
        self._two_finger_frames = 0
        self._two_finger_hovering = False
        self._thumb_apart_frames = 0
        self._open_palm_lift_frames = 0

    def _position_toolbar(self):
        sw = self.overlay.width()
        self.toolbar.move(sw - self.toolbar.width() - 12, 12)
        self.toolbar.raise_()

    def _throttle_special_action(self, interval=0.8):
        if time.time() - self._last_special_action_time < interval:
            return False
        self._last_special_action_time = time.time()
        return True

    def _log_pen(self, event, cause, features, label):
        """笔起落事件：记录触发原因和当时的关键比值，用于实机标定。"""
        logger.info(
            "[DRAW] pen_%s cause=%s label=%s frontality=%.2f thumb_ratio=%.2f mi=%.2f",
            event, cause, label, features["hand_frontality"],
            features["thumb_ratio"], features["middle_index_ratio"],
        )

    def _telemetry(self, features, label, state):
        """书写/悬停期间每 0.5s 记录一次比值流，便于在 gesture.log 标定
        draw_frontality_gate 与双指/拇指阈值。"""
        now = time.time()
        if now - self._last_telemetry < 0.5:
            return
        self._last_telemetry = now
        logger.info(
            "[DRAW] state=%s label=%s frontality=%.2f thumb_ratio=%.2f mi=%.2f tucked=%s",
            state, label, features["hand_frontality"], features["thumb_ratio"],
            features["middle_index_ratio"], features["thumb_tucked"],
        )

    def _classify_frame(self, features, label):
        """把单帧分类成一票证据：write / hover / stop / other。

        单帧不直接决定笔状态，只投一票，由 handle() 的时间窗多数表决决定
        起落（与 🤟 切模式同构）。判定照搬 analyze_trace.py 中经 draw_trace
        实录回放验证过的 VotedGate.classify（误断 57→2）：
          - 握拳 → stop（就绪）；
          - ✌️ 双指 → hover：VICTORY 标签直接确认；POINTING_UP / FIST 标签
            否决几何；其余靠几何（中指≈食指等长伸出、无名指小指未伸）；
          - 书写姿势（食指伸出 + 至多1个其他手指伸出）→ write（落笔）；
            在 is_open_palm 之前判定，避免3指 open_palm 误覆盖书写姿势；
          - 张掌（3+指伸出）→ stop（清屏）；
          - 其余 → other。
        features 在此处永不为 None（丢手在 handle() 上游已返回）。
        """
        if features["is_fist"]:
            return "stop"
        if label in ("POINTING_UP", "FIST"):
            two_finger = False            # ML 标签否决几何（单指/握拳姿势）
        elif label == "VICTORY":
            two_finger = True             # ML 标签确认 ✌️——侧视剪影仍可辨
        elif self._two_finger_geom:
            # 几何兜底（默认关闭）：侧视下 mi 不可信，详见 __init__ 注释
            two_finger = (
                features["index_extended"]
                and features["middle_index_ratio"] > 0.95
                and not features["ring_up"]
                and not features["pinky_up"]
            )
        else:
            two_finger = False
        if two_finger:
            return "hover"
        # 书写姿势优先于 is_open_palm：食指伸出 + 至多1个其他手指伸出 → write。
        # 避免放宽的3指 is_open_palm 在中指/无名指被误检为 up 时覆盖书写。
        other_fingers_up = (
            features["middle_up"] + features["ring_up"] + features["pinky_up"]
        )
        if features["index_extended"] and other_fingers_up <= 1:
            # 默认不让拇指改变笔状态（噪声大、近距正面误抬）；单指即落笔。
            # draw_thumb_lift=True 时恢复"手正对相机且拇指分开 → 抬笔"。
            if self._thumb_lift:
                readable = features["hand_frontality"] >= self._frontality_gate
                if readable and not features["thumb_tucked"]:
                    return "hover"
            return "write"
        # 张掌（3+指伸出，非书写姿势）→ stop（清屏）
        if features["is_open_palm"]:
            return "stop"
        # 食指伸出但2+其他手指也伸出 → 仍尝试书写（可能是部分张掌的过渡帧）
        if features["index_extended"]:
            return "write"
        return "other"

    def _record_trace(self, landmarks, label):
        """逐帧记录关键点到 draw_trace.jsonl（lm=None 表示该帧未检出手）。"""
        if not self._trace_writer:
            return
        self._trace_writer.write(json.dumps({
            "t": round(time.time(), 3),
            "label": label,
            "lm": [[round(p[1], 1), round(p[2], 1)] for p in landmarks] if landmarks else None,
        }, separators=(",", ":")) + "\n")

    def handle(self, hands_landmarks, hands_gestures, frame_w, frame_h) -> ModeResult:
        self._sync_frame_size(frame_w, frame_h)
        # 除零保护：异常帧时 frame_w/h 可能为 0
        if frame_w <= 0 or frame_h <= 0:
            return ModeResult(gesture="NONE")
        if not self.overlay.isVisible():
            self.overlay.show_fullscreen()

        if not hands_landmarks:
            self._record_trace(None, "NONE")
            self.fist_hold_frames = 0
            self.open_palm_frames = 0
            self._open_palm_lift_frames = 0

            # 手部跟丢缓冲：如果之前正在书写，允许短暂跟丢而不中断笔画
            if self._was_writing:
                self._writing_lost_frames += 1
                if self._writing_lost_frames < 8:
                    # 保持之前的书写状态，使用最后已知的屏幕坐标（由 overlay 维持）
                    return ModeResult(
                        gesture="DRAW",
                        status_text="正在书写 (跟丢缓冲)",
                        status_color=(0, 255, 255),
                    )

            self._was_writing = False
            self._writing_lost_frames = 0
            self.overlay.hide_cursor()
            self.overlay.tick_idle()
            return ModeResult(
                gesture="NONE",
                status_text="未检测到手",
                status_color=(0, 0, 255),
            )

        landmarks = hands_landmarks[0]
        features = self.recognizer.get_hand_features(landmarks)

        # 灵敏度随"手在画面中的大小"（掌宽）换算：手小（离得远）→ 高灵敏，
        # 手大（离得近）→ 低灵敏。responsiveness∈[0,1]：1=远/敏感，0=近/迟钝。
        # 不依赖无法感知的真实距离——只用掌宽这一画面度量。
        ref_width = self.recognizer.REFERENCE_HAND_WIDTH
        ratio = features["hand_width"] / ref_width
        resp_span = self.RESP_RATIO_NEAR - self.RESP_RATIO_FAR
        responsiveness = max(0.0, min(1.0, (self.RESP_RATIO_NEAR - ratio) / resp_span))
        # (1) 平滑：手大（近）→ 强平滑、稳，少发抖。
        self.overlay.set_motion_responsiveness(responsiveness)
        # (2) 位移增益：按距离分段设活动区 span_floor（见 SPAN_FLOOR_TIERS）。近→span_floor
        #     趋近 1.0 = 直接绝对映射 gain≈1（画圆轻松、找回卡尔曼手感、且不丢触达）；
        #     远→span_floor 小 = 小动作放大、写满全屏。分段表可独立微调各距离段手感。
        span_floor = interp_tiers(ratio, self.SPAN_FLOOR_TIERS)

        # 自适应活动区：把手最近扫过的范围映射到全屏（远距离也能写满），
        # 不再叠加边缘加速（apply_accel=False），避免双重映射失真。
        # 书写中冻结活动区与增益（update=False）：笔画期间传递函数恒定，避免中途
        # 重标定引起的漂移/抖动；只在悬停/就绪时校准。
        is_drawing_now = features["index_drawing_pose"] and features["thumb_tucked"]
        freeze = is_drawing_now or self._was_writing
        nx, ny = self._region_mapper.map(
            landmarks[5][1] / frame_w, landmarks[5][2] / frame_h,
            update=not freeze, span_floor=span_floor,
        )
        x_screen, y_screen = self.mouse.to_screen(nx, ny, apply_accel=False)
        self.overlay.update_cursor(x_screen, y_screen)

        # 笔粗距离自适应（默认关闭，见 config pen_width_auto_scale）：
        # bbox 大（近）→ 笔粗，bbox 小（远）→ 笔细。sqrt(bbox) 比 bbox 随距离更线性。
        if hands_gestures:
            bbox_area = hands_gestures[0].get("bbox_area", 0.0)
            if bbox_area > 0:
                hand_size = math.sqrt(bbox_area)
                scale = hand_size / self.overlay.REFERENCE_HAND_SIZE
                self.overlay.set_pen_scale(scale)

        label = hands_gestures[0].get("label", "OTHER") if hands_gestures else "OTHER"
        self._record_trace(landmarks, label)
        now = time.time()

        # === 中央投票笔状态机 ===
        # 实测教训（2026-06-13 draw_trace 回放，185s/3823 帧）：所有单帧布尔
        # 信号在 1-2 帧尺度上都会闪烁——握拳 87 段中 52 段 ≤2 帧、拇指比值
        # 每分钟穿越阈值 53 次。逐路径加去抖是打地鼠：堵上 two_finger
        # （52→7 次）churn 就流向 explicit_stop（23→68 次），总断笔量不降。
        # 唯一有效的形状：笔状态的每次转换都需要时间窗内多数帧的持续证据，
        # 单帧布尔值只产生证据、没有直接决定权（与 🤟 切模式同构）。
        # 实录回放对比：误断 57 次 → 2 次（见 analyze_trace.py / simulate_draw.py）。
        cls = self._classify_frame(features, label)
        self._vote.append((now, cls))
        self._recent_points.append((now, x_screen, y_screen))
        horizon = now - self._vote_window
        while self._vote and self._vote[0][0] < horizon:
            self._vote.pop(0)
        while self._recent_points and self._recent_points[0][0] < horizon:
            self._recent_points.pop(0)
        total = len(self._vote)
        n_write = sum(1 for _, c in self._vote if c == "write")
        n_stop = sum(1 for _, c in self._vote if c == "stop")
        n_up = n_stop + sum(1 for _, c in self._vote if c == "hover")

        if not self._was_writing:
            if total >= self.VOTE_MIN and n_write >= total * self._vote_ratio:
                self._was_writing = True
                self._writing_lost_frames = 0
                self._open_palm_lift_frames = 0
                self._log_pen("down", "vote", features, label)
                # 落笔回填：把确认期间积累的轨迹补画上，消除投票窗带来的起笔断头
                for _, px, py in self._recent_points:
                    self.overlay.draw_to(px, py)
                self._telemetry(features, label, "writing")
                return ModeResult(
                    gesture="DRAW",
                    status_text="正在书写",
                    status_color=(0, 255, 255),
                )
        else:
            # 张掌清屏优先级最高，但需连续 _open_palm_lift_min 帧确认才抬笔。
            # 旧实现单帧 is_open_palm 即 force_lift_pen，绕过投票窗去抖——而
            # is_open_palm = index_up & middle_up & ring_up，关键点抖动时 middle_up
            # 与 ring_up 偶发同真，单帧噪声直接断笔（实测断笔的主因）。改为连续确认：
            # 真张掌持续 → 很快累计到阈值抬笔流向清屏（仅多 ~0.15s，无感）；
            # 一两帧噪声 → 当书写继续，不断笔。
            if features["is_open_palm"]:
                self._open_palm_lift_frames += 1
                if self._open_palm_lift_frames >= self._open_palm_lift_min:
                    self._was_writing = False
                    self._writing_lost_frames = 0
                    self._log_pen("lift", "open_palm_interrupt", features, label)
                    self.overlay.force_lift_pen()
                    # 不返回：流向下方张掌清屏分支
                else:
                    # 未达确认帧数：按噪声处理，保持书写、继续描画，避免断笔。
                    self._writing_lost_frames = 0
                    self.overlay.draw_to(x_screen, y_screen)
                    self._telemetry(features, label, "writing")
                    return ModeResult(
                        gesture="DRAW",
                        status_text="正在书写",
                        status_color=(0, 255, 255),
                    )
            elif total >= self.VOTE_MIN and n_up >= total * self._vote_ratio:
                self._open_palm_lift_frames = 0
                cause = "vote_stop" if n_stop * 2 >= n_up else "vote_hover"
                self._was_writing = False
                self._writing_lost_frames = 0
                self._log_pen("lift", cause, features, label)
                self.overlay.force_lift_pen()
                # 不返回：显式停止（拳/掌）继续流向下方分支（清屏/就绪）
            else:
                self._open_palm_lift_frames = 0
                self._writing_lost_frames = 0
                self.fist_hold_frames = 0
                self.open_palm_frames = 0
                self.overlay.draw_to(x_screen, y_screen)
                self._telemetry(features, label, "writing")
                return ModeResult(
                    gesture="DRAW",
                    status_text="正在书写",
                    status_color=(0, 255, 255),
                )

        # 未在书写：write/hover 候选帧显示悬停光标，其余流向握拳/张掌分支
        if cls in ("write", "hover"):
            self.fist_hold_frames = 0
            self.open_palm_frames = 0
            self._telemetry(features, label, "hover")
            return ModeResult(
                gesture="DRAW_HOVER",
                status_text="悬停",
                status_color=(180, 180, 180),
            )

        is_fist = features["is_fist"]
        now = time.time()

        if self._was_fist and not is_fist:
            if now > self._double_fist_cooldown:
                self._fist_tap_times.append(now)
                self._fist_tap_times = [t for t in self._fist_tap_times if now - t < 1.0]
                if len(self._fist_tap_times) >= 2:
                    self._fist_tap_times.clear()
                    self._double_fist_cooldown = now + 1.0
                    enabled = self.overlay.toggle_shape_correction()
                    self.toolbar.set_shape_correction(enabled)
                    label = "已开启" if enabled else "已关闭"
                    try:
                        winsound.PlaySound(
                            "SystemAsterisk",
                            winsound.SND_ALIAS | winsound.SND_ASYNC,
                        )
                    except RuntimeError:
                        pass
                    self._was_fist = is_fist
                    return ModeResult(
                        gesture="TOGGLE_SHAPE_CORRECTION",
                        status_text=f"图形修正{label}",
                        status_color=(0, 200, 100) if enabled else (180, 180, 180),
                    )
        self._was_fist = is_fist

        # 握拳 -> 就绪
        if is_fist:
            self.fist_hold_frames = 0
            self.open_palm_frames = 0
            self.overlay.tick_idle()
            return ModeResult(
                gesture="DRAW_READY",
                status_text="板书模式就绪",
                status_color=(0, 255, 0),
            )

        # 张掌 -> 清屏（长按）
        # open_palm_frames 阈值 15（20fps 下 0.75s）：30 帧（1.5s）太长用户以为没反应。
        OPEN_PALM_CLEAR_FRAMES = 15
        if features["is_open_palm"]:
            self.open_palm_frames += 1
            self.overlay.tick_idle()
            if self.open_palm_frames >= OPEN_PALM_CLEAR_FRAMES and self._throttle_special_action():
                self.overlay.clear_canvas()
                self.fist_hold_frames = 0
                self.open_palm_frames = 0
                try:
                    winsound.PlaySound(
                        "SystemExclamation",
                        winsound.SND_ALIAS | winsound.SND_ASYNC,
                    )
                except RuntimeError:
                    winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
                return ModeResult(
                    gesture="CLEAR_CANVAS",
                    status_text="已清空画布",
                    status_color=(0, 165, 255),
                )
            progress = min(self.open_palm_frames / OPEN_PALM_CLEAR_FRAMES, 1.0) * 100
            return ModeResult(
                gesture="DRAW_READY",
                status_text=f"清屏中... {progress:.0f}%",
                status_color=(255, 165, 0),
            )
        else:
            self.fist_hold_frames = 0
            self.open_palm_frames = 0
            self.overlay.tick_idle()

        return ModeResult(
            gesture="DRAW_READY",
            status_text="板书模式就绪",
            status_color=(0, 255, 0),
        )
