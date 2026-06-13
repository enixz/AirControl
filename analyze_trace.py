"""逐帧分析 draw_trace.jsonl：量化每个基础判定信号的真实可靠性，
并把轨迹回放进各判定变体统计"瞬断"。

用法：python analyze_trace.py [draw_trace.jsonl]
"""
import json
import statistics
import sys
from collections import Counter

sys.path.insert(0, "app")
from services.gesture_recognizer import GestureRecognizer  # noqa: E402
from simulate_draw import make_variant  # noqa: E402

path = sys.argv[1] if len(sys.argv) > 1 else "draw_trace.jsonl"
seq = []
for line in open(path, encoding="utf-8"):
    rec = json.loads(line)
    lm = rec["lm"]
    if lm is not None:
        lm = [[i, p[0], p[1]] for i, p in enumerate(lm)]
    seq.append((rec["t"], lm, rec.get("label", "OTHER")))

ts = [t for t, _, _ in seq]
dts = [b - a for a, b in zip(ts, ts[1:]) if 0 < b - a < 1.0]
fps = 1 / statistics.median(dts)
print(f"帧数 {len(seq)}  时长 {ts[-1]-ts[0]:.0f}s  中位帧率 {fps:.1f}fps")

# 手丢失
null_runs, run = [], 0
for _, lm, _ in seq:
    if lm is None:
        run += 1
    elif run:
        null_runs.append(run)
        run = 0
if run:
    null_runs.append(run)
n_null = sum(null_runs)
print(f"未检出手: {n_null} 帧 ({100*n_null/len(seq):.1f}%)  "
      f"丢失段 {len(null_runs)} 次  最长 {max(null_runs, default=0)} 帧  "
      f"段长分布 {dict(sorted(Counter(null_runs).items()))}")

print("标签分布:", dict(Counter(lb for _, _, lb in seq).most_common()))

# 逐帧特征
r = GestureRecognizer()
feats = [r.get_hand_features(lm) if lm else None for _, lm, _ in seq]


def seg_runs(pred):
    out, c = [], 0
    for f in feats:
        if f is not None and pred(f):
            c += 1
        elif c:
            out.append(c)
            c = 0
    if c:
        out.append(c)
    return out


for name, pred in [
    ("is_fist", lambda f: f["is_fist"]),
    ("is_open_palm", lambda f: f["is_open_palm"]),
    ("index_drawing_pose", lambda f: f["index_drawing_pose"]),
    ("index_extended", lambda f: f["index_extended"]),
]:
    rr = seg_runs(pred)
    hist = Counter(rr)
    short = sum(v for k, v in hist.items() if k <= 2)
    print(f"{name:<20} 帧占比 {100*sum(rr)/max(1,len(seq)-n_null):5.1f}%  "
          f"段数 {len(rr):4d}  其中 ≤2 帧的瞬时段 {short:4d}  "
          f"段长分布 {dict(sorted(hist.items())[:8])}")

# 书写姿势下拇指比值的逐帧波动（阈值穿越率）
th_stream = [f["thumb_ratio"] for f in feats if f and f["index_drawing_pose"]]
cross = sum(
    1 for a, b in zip(th_stream, th_stream[1:])
    if (a < 0.62) != (b < 0.62)
)
print(f"\n书写姿势帧 {len(th_stream)}: thumb_ratio 中位 {statistics.median(th_stream):.2f} "
      f"p25 {sorted(th_stream)[len(th_stream)//4]:.2f} "
      f"p75 {sorted(th_stream)[3*len(th_stream)//4]:.2f}  "
      f"穿越0.62阈值 {cross} 次（每分钟 {cross/((ts[-1]-ts[0])/60):.0f} 次）")

mi_stream = [f["middle_index_ratio"] for f in feats if f and f["index_extended"]]
print(f"mi 流: 中位 {statistics.median(mi_stream):.2f}  "
      f"p95 {sorted(mi_stream)[int(0.95*len(mi_stream))]:.2f}  "
      f"max {max(mi_stream):.2f}")
fr_stream = [f["hand_frontality"] for f in feats if f and f["index_extended"]]
print(f"frontality 流: 中位 {statistics.median(fr_stream):.2f}  "
      f"p10 {sorted(fr_stream)[int(0.10*len(fr_stream))]:.2f}  "
      f"<0.55 占比 {100*sum(1 for v in fr_stream if v < 0.55)/len(fr_stream):.0f}%")

class VotedGate:
    """中央投票笔状态机原型：笔状态的每次转换都需要窗口内多数帧的持续
    证据，单帧布尔值不再有直接决定权——与 🤟 切模式同构。

    W=7 帧窗口（~0.3s @22fps），K=5 多数票。落笔同样要投票，生产实现
    需要配合笔画回填（确认落笔时把窗口内轨迹补画上）消除起笔延迟。
    """

    W, K = 7, 5

    def __init__(self, use_thumb=True):
        self.r = GestureRecognizer()
        self.use_thumb = use_thumb
        self.win = []
        self.writing = False
        self.lost = 0
        self.lifts = []

    def classify(self, f, label):
        if f is None:
            return "none"
        if f["is_fist"] or f["is_open_palm"]:
            return "stop"
        two = (label == "VICTORY") or (
            label not in ("POINTING_UP", "FIST")
            and f["index_extended"] and f["middle_index_ratio"] > 0.95
            and not f["ring_up"] and not f["pinky_up"]
        )
        if two:
            return "hover"
        if f["index_extended"]:
            readable = f["hand_frontality"] >= 0.55
            if self.use_thumb and readable and not f["thumb_tucked"]:
                return "hover"
            return "write"
        return "other"

    def step(self, idx, lm, label):
        f = self.r.get_hand_features(lm) if lm else None
        c = self.classify(f, label)
        if c == "none":
            self.lost += 1
            if self.writing and self.lost >= 8:
                self.writing = False
                self.lifts.append((idx, "hand_lost"))
            return self.writing
        self.lost = 0
        self.win.append(c)
        if len(self.win) > self.W:
            self.win.pop(0)
        n_write = self.win.count("write")
        n_up = self.win.count("hover") + self.win.count("stop")
        if not self.writing and n_write >= self.K:
            self.writing = True
        elif self.writing and n_up >= self.K:
            self.writing = False
            cause = ("stop" if self.win.count("stop") >= self.win.count("hover")
                     else "hover")
            self.lifts.append((idx, cause))
        return self.writing


def make_any(name):
    if name == "voted":
        return VotedGate(use_thumb=True)
    if name == "voted_no_thumb":
        return VotedGate(use_thumb=False)
    return make_variant(name)


# 回放各变体：统计笔状态翻转与"瞬抬"（抬笔后 <=0.25s 又落笔 = 几乎必是误断）
print(f"\n{'变体':<18}{'抬笔次数':>8}{'瞬抬(<0.25s)':>12}  抬笔原因")
print("-" * 70)
for name in ("legacy_f09059d", "fixed", "bare_mediapipe", "voted", "voted_no_thumb"):
    g = make_any(name)
    pen = [g.step(i, lm, lb) for i, (t, lm, lb) in enumerate(seq)]
    # 瞬抬：False 段（两侧为 True）时长 <= 0.25s
    flick = 0
    i = 0
    while i < len(pen):
        if not pen[i]:
            j = i
            while j < len(pen) and not pen[j]:
                j += 1
            if i > 0 and j < len(pen) and (ts[min(j, len(ts)-1)] - ts[i]) <= 0.25:
                flick += 1
            i = j
        else:
            i += 1
    causes = Counter(c for _, c in g.lifts)
    print(f"{name:<18}{len(g.lifts):>8}{flick:>12}  {dict(causes.most_common())}")
