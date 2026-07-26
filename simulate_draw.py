"""板书断触模拟器：合成贴近实测分布的手部关键点流，离线驱动起落笔判定，
对比三种逻辑变体的断笔率。

所有分布参数标定自 2026-06-12 实机 gesture.log（20fps，979 行遥测）：
- 单指书写时"中指长/食指长"(mi)：中位 0.36 / p75 0.50 / 最大 0.81；真双指 ≥1.15
- 书写时拇指比值：中位 0.26 / p90 0.37；故意抬笔时 0.86~1.10
- 书写姿势的 ML 标签：87.5% POINTING_UP / 12.5% OTHER
- 正面度：正面书写 ≈0.80，横扫侧偏时按 cos 塌缩

变体：
  legacy_f09059d  昨晚实测引发严重断触的版本（中指长>掌宽×0.6 判双指、无标签否决、2帧去抖）
  fixed           直接调用项目当前 DrawMode，不维护易漂移的重复状态机
  bare_mediapipe  裸奔 MediaPipe：笔状态直接绑 POINTING_UP 标签，无任何缓冲/门控

“fixed” 变体直接运行真实 DrawMode，因此实现更新会自动反映到模拟结果。

用法：
  python simulate_draw.py                  # 合成 120s 书写会话对比三变体
  python simulate_draw.py --replay draw_trace.jsonl   # 回放实机录制轨迹
"""
import argparse
import json
import math
import os
import random
import sys
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "app"))

from services.gesture_recognizer import GestureRecognizer  # noqa: E402

FPS = 20.0
DT = 1.0 / FPS


# ---------------------------------------------------------------- 手部合成 --

def build_hand(mi, thumb_ratio, yaw, rng, jitter=2.0):
    """按目标比值合成 21 点关键点（[id,x,y] 像素）。

    几何骨架与 tests/test_draw_side_view.py 相同：食指竖直伸出 100px，
    掌宽 80px；中指尖按 mi、拇指尖按 thumb_ratio 摆放；yaw 压缩 x；
    最后叠加 σ=jitter 的高斯像素噪声模拟关键点抖动。
    """
    index_len = 100.5
    hand_width = 80.62
    # 中指尖：mi<0.85 视为弯曲（偏侧向，指尖不高于 PIP），否则贴紧食指伸出
    m_len = mi * index_len
    if mi < 0.85:
        ang = math.radians(70)   # 偏侧向 → y 不会高过 PIP，书写姿势不被破坏
    else:
        ang = math.radians(12)   # 近竖直 → 与食指贴紧
    middle_tip = (285 - m_len * math.sin(ang), 295 - m_len * math.cos(ang))
    # 拇指尖：从食指根 (260,300) 沿固定方向放到目标距离
    d = thumb_ratio * hand_width
    thumb_tip = (260 - d * 0.633, 300 + d * 0.774)
    pts = {
        0: (300, 400),
        2: (240, 360), 3: (235, 340), 4: thumb_tip,
        5: (260, 300), 6: (255, 260), 8: (250, 200),
        9: (285, 295), 10: (282, 265), 12: middle_tip,
        14: (310, 270), 16: (312, 305),
        17: (340, 310), 18: (335, 280), 20: (336, 315),
    }
    lm = []
    for i in range(21):
        x, y = pts.get(i, (300, 350))
        x = 300 + (x - 300) * yaw
        lm.append([i, x + rng.gauss(0, jitter), y + rng.gauss(0, jitter)])
    return lm


def synth_session(seconds=120.0, seed=7):
    """合成书写会话：2s 笔画（手肘横扫、偏航摆动）+ 0.5s 故意抬笔，循环。

    产出 (landmarks, label, intent) 序列，intent ∈ {"down","up"} 为用户真实意图。
    """
    rng = random.Random(seed)
    frames = []
    t = 0.0
    while t < seconds:
        # --- 2s 笔画：yaw 1.0 -> 0.4 -> 1.0（手肘为轴横扫一个来回）
        stroke_len = 2.0
        n = int(stroke_len * FPS)
        for k in range(n):
            ts = k / FPS
            yaw = 0.7 + 0.3 * math.cos(math.pi * ts / stroke_len * 2)
            mi = min(0.81, max(0.05, rng.gauss(0.40, 0.18)))
            if yaw < 0.6:  # 深侧偏：拇指被遮挡，关键点为脑补值，可能读成"分开"
                thumb = min(1.4, max(0.2, rng.gauss(0.90, 0.30)))
            else:
                thumb = min(0.50, max(0.05, rng.gauss(0.26, 0.08)))
            label = "POINTING_UP" if rng.random() < 0.875 else "OTHER"
            frames.append((build_hand(mi, thumb, yaw, rng), label, "down"))
        t += stroke_len
        # --- 0.5s 故意抬笔：当前稳定档以可靠的 VICTORY 标签悬停/抬笔；
        # 拇指抬笔已因关键点噪声默认关闭，不能再作为当前行为的意图样本。
        gap_len = 0.5
        for _k in range(int(gap_len * FPS)):
            mi = min(0.81, max(0.05, rng.gauss(0.40, 0.18)))
            thumb = min(1.3, max(0.70, rng.gauss(0.95, 0.15)))
            label = "VICTORY"
            frames.append((build_hand(mi, thumb, 0.95, rng), label, "up"))
        t += gap_len
    return frames


# ---------------------------------------------------------------- 判定变体 --

class PenGate:
    """Historical draw-mode gate retained only for legacy comparison."""

    def __init__(self, *, rule, veto, debounce, frontality_gate=0.55):
        self.r = GestureRecognizer()
        self.rule = rule                  # "mi" | "hand_width"
        self.veto = veto                  # POINTING_UP/FIST 否决几何
        self.debounce = debounce
        self.gate = frontality_gate
        self.writing = False
        self.lost = 0
        self.two = 0
        self.hovering = False
        self.thumb_apart = 0
        self.lifts = []                   # (frame_idx, cause)

    def _two_finger(self, f, label):
        if label == "VICTORY":
            return True
        if self.veto and label in ("POINTING_UP", "FIST"):
            return False
        if self.rule == "mi":
            mi_gate = 0.85 if self.hovering else 0.95
            return (
                f["index_extended"] and f["middle_index_ratio"] > mi_gate
                and not f["ring_up"] and not f["pinky_up"]
            )
        # legacy: 中指长 > 掌宽×0.6（middle_ratio = 中指长/掌宽）
        return (
            f["index_extended"] and f["middle_ratio"] > 0.6
            and not f["ring_up"] and not f["pinky_up"]
        )

    def step(self, idx, landmarks, label):
        if landmarks is None:
            if self.writing:
                self.lost += 1
                if self.lost >= 8:
                    self.writing = False
                    self.lost = 0
                    self.lifts.append((idx, "hand_lost"))
            return self.writing
        f = self.r.get_hand_features(landmarks)
        stop = f["is_fist"] or f["is_open_palm"]

        if self._two_finger(f, label):
            self.two += 1
        else:
            self.two = 0
            self.hovering = False
        if self.two >= self.debounce:
            self.hovering = True
            if self.writing:
                self.writing = False
                self.lifts.append((idx, "two_finger"))
            self.thumb_apart = 0
            return False

        readable = f["hand_frontality"] >= self.gate

        if self.writing and f["index_extended"] and not stop:
            if readable and not f["thumb_tucked"]:
                self.thumb_apart += 1
            else:
                self.thumb_apart = 0
            if self.thumb_apart >= 3:
                self.thumb_apart = 0
                self.writing = False
                self.lifts.append((idx, "thumb_apart"))
                return False
            self.lost = 0
            return True

        if not self.writing and f["index_drawing_pose"]:
            self.thumb_apart = 0
            if (readable and f["thumb_tucked"]) or (
                not readable and label == "POINTING_UP"
            ):
                self.writing = True
                self.lost = 0
                return True
            return False

        if self.writing and not stop:
            self.lost += 1
            if self.lost < 10:
                return True
            self.lifts.append((idx, "pose_lost"))
        elif self.writing and stop:
            self.lifts.append((idx, "explicit_stop"))
        self.writing = False
        self.lost = 0
        return False


class CurrentDrawGate:
    """Adapter that drives the real DrawMode on the simulator's frame clock."""

    def __init__(self):
        from modes.draw_mode import DrawMode

        class Config:
            def get(self, key, default=None):
                return {"draw_record_trace": False}.get(key, default)

        self.overlay = mock.MagicMock()
        self.overlay.REFERENCE_HAND_SIZE = 100.0
        self.overlay.isVisible.return_value = True
        mouse = mock.MagicMock()
        mouse.to_screen.return_value = (500, 500)
        self.mode = DrawMode(
            Config(),
            GestureRecognizer(),
            mouse,
            self.overlay,
            mock.MagicMock(),
            mock.MagicMock(),
            mock.MagicMock(),
        )
        self.mode.on_enter()
        self.writing = False
        self.lifts = []

    def step(self, idx, landmarks, label):
        hands = [landmarks] if landmarks is not None else []
        gestures = (
            [{"label": label, "bbox_area": 0.0}]
            if landmarks is not None
            else []
        )
        was_writing = self.writing
        with mock.patch("modes.draw_mode.time.time", return_value=idx / FPS):
            result = self.mode.handle(hands, gestures, 640, 480)
        self.writing = bool(self.mode._was_writing)
        if was_writing and not self.writing:
            self.lifts.append((idx, f"draw_mode_{result.gesture.lower()}"))
        return self.writing


class BareGate:
    """裸奔 MediaPipe：笔状态 = (label == POINTING_UP)，无缓冲无门控。"""

    def __init__(self):
        self.writing = False
        self.lifts = []

    def step(self, idx, landmarks, label):
        was = self.writing
        self.writing = landmarks is not None and label == "POINTING_UP"
        if was and not self.writing:
            self.lifts.append((idx, "label_flicker"))
        return self.writing


def make_variant(name):
    if name == "legacy_f09059d":
        return PenGate(rule="hand_width", veto=False, debounce=2)
    if name == "fixed":
        return CurrentDrawGate()
    if name == "bare_mediapipe":
        return BareGate()
    raise ValueError(name)


# ---------------------------------------------------------------- 评估 --

def evaluate(frames, gate):
    """统计：意图书写期间的误抬笔（断触）、故意抬笔的响应、误落笔帧占比。"""
    false_lifts = 0
    deliberate_lift_ok = 0
    gaps = 0
    false_down_frames = 0
    up_frames = 0
    gap_lifted = False
    prev_intent = None
    for idx, (lm, label, intent) in enumerate(frames):
        pen = gate.step(idx, lm, label)
        if intent == "up":
            up_frames += 1
            if pen:
                false_down_frames += 1
        if prev_intent == "down" and intent == "up":
            gaps += 1
            gap_lifted = False
        if intent == "up" and not pen:
            gap_lifted = True
        if prev_intent == "up" and intent == "down" and gap_lifted:
            deliberate_lift_ok += 1
        prev_intent = intent
    if prev_intent == "up" and gap_lifted:
        deliberate_lift_ok += 1
    for idx, _cause in gate.lifts:
        if frames[idx][2] == "down":
            false_lifts += 1
    minutes = len(frames) / FPS / 60.0
    return {
        "false_lifts_per_min": false_lifts / minutes,
        "deliberate_lift_rate": deliberate_lift_ok / max(gaps, 1),
        "false_down_pct_in_gaps": 100.0 * false_down_frames / max(up_frames, 1),
        "lift_causes": _hist(c for i, c in gate.lifts),
    }


def _hist(it):
    h = {}
    for x in it:
        h[x] = h.get(x, 0) + 1
    return dict(sorted(h.items(), key=lambda kv: -kv[1]))


def load_replay(path):
    frames = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            lm = rec["lm"]
            if lm is not None:
                lm = [[i, p[0], p[1]] for i, p in enumerate(lm)]
            frames.append((lm, rec.get("label", "OTHER"), "down"))
    return frames


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", help="回放 draw_trace.jsonl（实机录制）")
    ap.add_argument("--seconds", type=float, default=120.0)
    args = ap.parse_args(argv)

    if args.seconds <= 0:
        ap.error("--seconds must be greater than zero")

    if args.replay:
        try:
            frames = load_replay(args.replay)
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            print(f"无法读取回放文件: {exc}", file=sys.stderr)
            return 2
        print(f"回放 {args.replay}: {len(frames)} 帧（意图未知，只统计抬笔事件）")
    else:
        frames = synth_session(args.seconds)
        print(f"合成会话: {args.seconds:.0f}s @ {FPS:.0f}fps，"
              f"笔画2s+抬笔0.5s循环，横扫偏航 1.0~0.4，分布取自实测 gesture.log")

    if not frames:
        print("没有可评估的帧", file=sys.stderr)
        return 2

    print()
    header = f"{'变体':<18}{'断触/分钟':>10}{'故意抬笔成功率':>14}{'误落笔帧占比':>12}  抬笔原因"
    print(header)
    print("-" * 78)
    for name in ("legacy_f09059d", "fixed", "bare_mediapipe"):
        gate = make_variant(name)
        m = evaluate(frames, gate)
        print(f"{name:<18}{m['false_lifts_per_min']:>10.1f}"
              f"{100 * m['deliberate_lift_rate']:>13.0f}%"
              f"{m['false_down_pct_in_gaps']:>11.0f}%  {m['lift_causes']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
