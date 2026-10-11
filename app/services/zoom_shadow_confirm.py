"""CAPTURE 态影子 ZOOM 确认晋升（shadow-ZOOM confirmation handoff）.

纯算法、无 Qt/模型依赖，由 InferenceWorker 逐帧喂两侧检测结果。

动机（2026-10-11 三步实验结论，docs/small_hand_tracking_progress.md）：
  - CAPTURE→FAR_TRACK 的既有判据是"连续 30 帧稳定单手"。远距小手在冷却后的
    连续新鲜输出常常凑不满，短片反例中直到片尾也进不了 FAR_TRACK。
  - 影子 ZOOM 实验（17 段回放）：捕获位置供应下，真远距小手段影子覆盖
    188/199、双新鲜几何一致 103/103 无一跟错手；miss≥2 时用捕获位置重播种
    可救回窗口塌缩且不引入跟错手。
  - 被否决的提前交接候选（capture_handoff）只缩短等待、直接交班，反例中
    新增 41 帧缺失。本模块与它的本质区别：不猜、先证明——影子 ZOOM 必须
    与捕获器在同一帧对同一只手达成几何一致并连续自持，才允许晋升。

确认协议（与影子探针 shadow_zoom_probe.py 一致，另有第 5 条冷启动播种）：
  1. 窗口开启：双新鲜（捕获主手与影子主手都不是预测）且影子手腕与捕获
     手中心（bbox 中心）距离 ≤ 0.75×捕获手跨度。
  2. 窗口自持：此后连续 CONFIRM_WINDOW_FRAMES-1 帧影子保持新鲜主手且自身
     手腕连续（每帧位移 ≤ 0.75×影子手跨度）；影子出现预测/空输出时窗口
     挂起等待（与探针一致），新鲜帧上跳变则关窗重来。
  3. 影子失检（ZOOM miss ≥ reseed_miss_frames 或 ZOOM OFF 回退）时，若有
     当帧捕获位置，消费方应把该位置重播种给影子视口（一步落位，不走 EMA）。
     影子从未进入 ZOOM（冷启动）也走同一路径——真远距小手在全帧路径检不出，
     ZOOM miss 永远不累计，不播种就会死锁在无输出。
  4. 晋升门槛：confirmations 达到所需次数（默认 1）且捕获手 bbox 占比
     < near_bbox_ratio（近距大目标不晋升，防与 NEAR 判据打架）。
  5. 冷启动播种：影子尚未开过 ZOOM 时，首个新鲜捕获位置即一步落位视口
     （与探针的首次播种同型）。已开过 ZOOM 后此路径关闭，防视口抖动。

本类不持有 tracker 引用、不执行重播种——重播种由消费方（orchestrator）根据
should_reseed 把捕获位置一步落位给影子 tracker 视口，便于离线测试纯逻辑。
"""

import math

CONFIRM_WINDOW_FRAMES = 5
CONFIRM_MATCH_SPAN_RATIO = 0.75


class ZoomShadowConfirm:
    """影子 ZOOM 确认状态机（worker 逐帧喂，orchestrator 消费晋升结果）。"""

    def __init__(self, required_confirmations=1, reseed_miss_frames=2,
                 near_bbox_ratio=0.04, confirm_window_frames=None):
        self.required_confirmations = max(1, int(required_confirmations))
        self.reseed_miss_frames = max(1, int(reseed_miss_frames))
        self.near_bbox_ratio = max(0.0, float(near_bbox_ratio))
        self.window_frames = (
            CONFIRM_WINDOW_FRAMES if confirm_window_frames is None
            else max(3, int(confirm_window_frames)))
        self.seeded = False
        self.last_zoom_on = False
        self.reset()

    def reset(self):
        self.confirm_remaining = 0
        self.confirmations = 0
        self.last_confirmed_wrist = None
        self.confirmed_frame = None
        # 视口播种簿记随会话重置：teardown 后的新影子段需要重新冷启动播种
        self.seeded = False
        self.last_zoom_on = False

    @property
    def promotion_ready(self):
        return self.confirmations >= self.required_confirmations

    def _span(self, points):
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return math.hypot(max(xs) - min(xs), max(ys) - min(ys))

    def _hint(self, points):
        """与 base_hand_tracker._compute_hint_from_landmarks 同型：中心 +
        max(宽,高)。视口播种需要这个语义，不是手腕点/hypot 跨度。"""
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        center = ((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0)
        size = max(max(xs) - min(xs), max(ys) - min(ys))
        return center, size

    def _wrist(self, points):
        # points: [[idx, x, y], ...]（tracker 输出格式），wrist 是第 0 点。
        return (points[0][1], points[0][2])

    def observe(self, capture_hands, capture_gestures, capture_bbox_ratio,
                shadow_hands, shadow_gestures, shadow_zoom_on,
                shadow_miss_streak, frame_index):
        """喂一帧两侧检测结果，返回 (should_promote, should_reseed, reseed_hint)。

        Args:
            capture_hands/gestures: 主 tracker（捕获引擎）本帧输出。
            capture_bbox_ratio: 捕获主手 bbox 面积 / 全帧面积（无手传 0）。
            shadow_hands/gestures: 影子 ZOOM tracker 本帧输出。
            shadow_zoom_on: 影子本帧 ZOOM 是否开启（ZOOM OFF 回退也触发重播种）。
            shadow_miss_streak: 影子连续失检帧数。
            frame_index: 当前帧号（单调递增，记录确认帧用）。

        Returns:
            (bool, bool, tuple|None):
              should_promote — 确认次数达标且为远距小手，可晋升 FAR_TRACK。
              should_reseed — 需要落位影子视口（冷启动首次播种或失检重播种）。
              reseed_hint — should_reseed 时的 (center, size)，与
                _compute_hint_from_landmarks 同型，供视口一步落位。
        """
        capture_fresh = bool(capture_hands) and bool(capture_gestures) \
            and isinstance(capture_gestures[0], dict) \
            and not capture_gestures[0].get("predicted", False)
        shadow_fresh = bool(shadow_hands) and bool(shadow_gestures) \
            and isinstance(shadow_gestures[0], dict) \
            and not shadow_gestures[0].get("predicted", False)

        # 视口播种判定（优先于确认逻辑：影子有输出才有确认可言）。
        # 三条路径共享一个 hint 语义（捕获 bbox 中心 + max(宽,高)）：
        #   ① 冷启动：影子从未播种过且 ZOOM 未开，首个新鲜捕获位置即播种
        #     （探针同型）。真远距小手在全帧路径检不出，不播种则 miss 永不
        #     累计、死锁无输出。
        #   ② 失检：miss ≥ reseed_miss_frames 且当帧捕获位置可用（救窗口塌缩）。
        #   ③ ZOOM OFF 下降沿：影子刚回退全帧（小手必然丢），有捕获位置就
        #     重播种（探针 zoom_reset 条件的另一半；用边沿而非电平，防影子
        #     合法跑全帧的近手段每帧重播种）。
        should_reseed = False
        reseed_hint = None
        capture_center = capture_span = None
        capture_hint_size = 0.0
        zoom_off_edge = self.last_zoom_on and not shadow_zoom_on
        self.last_zoom_on = bool(shadow_zoom_on)
        if capture_fresh:
            capture_center, capture_hint_size = self._hint(
                [(p[1], p[2]) for p in capture_hands[0]])
            capture_span = self._span(
                [(p[1], p[2]) for p in capture_hands[0]])
            needs_viewport = (
                (not self.seeded and not shadow_zoom_on)
                or zoom_off_edge
                or shadow_miss_streak >= self.reseed_miss_frames
            )
            if needs_viewport:
                should_reseed = True
                reseed_hint = (capture_center, capture_hint_size)
                if not self.seeded and not shadow_zoom_on:
                    self.seeded = True

        # 同帧几何一致才开窗（与影子探针一致：影子手腕 vs 捕获手中心，
        # 阈值 0.75×捕获手跨度）。
        same_frame_match = False
        if capture_fresh and shadow_fresh:
            shadow_wrist = self._wrist(shadow_hands[0])
            same_frame_match = math.dist(shadow_wrist, capture_center) \
                <= CONFIRM_MATCH_SPAN_RATIO * max(capture_span, 1.0)

        # 窗口自持：影子新鲜且自身连续。
        if self.confirm_remaining > 0 and shadow_fresh:
            shadow_wrist = self._wrist(shadow_hands[0])
            shadow_span = self._span(
                [(p[1], p[2]) for p in shadow_hands[0]])
            self_continuous = (
                self.last_confirmed_wrist is not None
                and math.dist(shadow_wrist, self.last_confirmed_wrist)
                <= CONFIRM_MATCH_SPAN_RATIO * max(shadow_span, 1.0))
            if self_continuous:
                self.confirm_remaining -= 1
                self.last_confirmed_wrist = shadow_wrist
                if self.confirm_remaining == 0:
                    self.confirmations += 1
                    self.confirmed_frame = frame_index
            else:
                self.confirm_remaining = 0

        if self.confirm_remaining == 0 and same_frame_match:
            self.confirm_remaining = self.window_frames - 1
            self.last_confirmed_wrist = self._wrist(shadow_hands[0])

        should_promote = (
            self.promotion_ready
            and 0 < capture_bbox_ratio < self.near_bbox_ratio)
        return should_promote, should_reseed, reseed_hint
