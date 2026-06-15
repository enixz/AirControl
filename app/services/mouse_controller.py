import ctypes
import ctypes.wintypes
import logging
import time

logger = logging.getLogger("gesture")

user32 = ctypes.windll.user32

# Win32 虚拟屏幕指标 — 跨越所有显示器
SM_CXVIRTUALSCREEN = 78
SM_CYVIRTUALSCREEN = 79
SM_XVIRTUALSCREEN = 76
SM_YVIRTUALSCREEN = 77

def get_cursor_pos():
    pt = ctypes.wintypes.POINT()
    user32.GetCursorPos(ctypes.byref(pt))
    return pt.x, pt.y

def set_cursor_pos(x, y):
    return user32.SetCursorPos(int(x), int(y))

def mouse_event_click(button='left'):
    MOUSEEVENTF_LEFTDOWN = 0x0002
    MOUSEEVENTF_LEFTUP = 0x0004
    MOUSEEVENTF_RIGHTDOWN = 0x0008
    MOUSEEVENTF_RIGHTUP = 0x0010
    
    if button == 'left':
        user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        time.sleep(0.02)
        user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
    elif button == 'right':
        user32.mouse_event(MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0)
        time.sleep(0.02)
        user32.mouse_event(MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)

def mouse_event_scroll(amount):
    MOUSEEVENTF_WHEEL = 0x0800
    user32.mouse_event(MOUSEEVENTF_WHEEL, 0, 0, amount, 0)

def _edge_map(t, strength):
    """
    非线性边缘加速映射。
    t: [0, 1] 归一化坐标
    strength: [0.0, 1.0] 强度（0=线性，1=最大加速）
    返回: [0, 1] 映射后的坐标
    """
    if strength <= 0:
        return t
    s = strength * 0.5
    val = t + s * (t - 0.5) ** 3 * 2.0
    return max(0.0, min(1.0, val))


def _canvas_map(t, deadzone_top, deadzone_bottom):
    """
    虚拟画布映射：将 [0,1] 的归一化坐标映射到 [0,1]。
    deadzone_top / deadzone_bottom: [0.0, 1.0]
    """
    if deadzone_top <= 0 and deadzone_bottom <= 0:
        return t
    low = deadzone_top
    high = 1.0 - deadzone_bottom
    if high <= low:
        # deadzone_top + deadzone_bottom >= 1.0 时，映射整个区间到边界
        return 0.0 if t <= low else 1.0
    if t <= low:
        return 0.0
    if t >= high:
        return 1.0
    return (t - low) / (high - low)


def blended_landmark_point(landmarks, weighted_indices):
    """Return a weighted 2D control point from a landmark chain.

    Fingertips are the noisiest MediaPipe landmarks. Blending the last few
    joints keeps the cursor aligned with the finger while suppressing isolated
    tip jumps. Weights are normalized so callers can tune them conveniently.
    """
    total = sum(float(weight) for _, weight in weighted_indices)
    if total <= 0:
        raise ValueError("weighted_indices must contain a positive weight")
    x = sum(float(landmarks[index][1]) * weight for index, weight in weighted_indices)
    y = sum(float(landmarks[index][2]) * weight for index, weight in weighted_indices)
    return x / total, y / total


class ActiveRegionMapper:
    """自适应活动区映射器（用于远距离全屏书写/指向）。

    跟踪手部最近扫过的归一化范围（"活动区"），把该区域映射到整屏 [0,1]：
      - 手移动时活动区**快速向外扩张**（立即包含新位置）；
      - 手移动时活动区**缓慢向内收缩**（人靠近/远离导致可达范围变化时自动重校准）；
      - 手静止时**冻结收缩**（避免笔尖在悬停时漂向屏幕中心）。

    效果：无论站多远、手只占画面里偏角的一小块，都能写满全屏；移动几秒后
    自动校准到使用者的实际可达范围。无需手动标定。
    """

    def __init__(self, margin=0.08, min_span=0.22, contract=0.0006, move_eps=0.004):
        self.margin = float(margin)        # 边缘余量：不必触到扫动极值即可抵达屏幕边
        self.min_span = float(min_span)    # 活动区最小跨度，防止微动时灵敏度过高
        self.contract = float(contract)    # 每帧向内收缩速率（归一化/帧）
        self.move_eps = float(move_eps)    # 判定"在移动"的最小位移
        self._lo = [None, None]
        self._hi = [None, None]
        self._last_raw = None
        # 活动区跨度下限（span floor）：决定"小幅手部移动"被放大的程度。
        # 小（如 0.22）→ 小动作也放大、远距离写满全屏；大（→1.0）→ 趋近直接绝对映射，
        # gain≈1、近距离画圆轻松且不丢触达。由 map() 的 span_floor 每帧更新，书写中冻结。
        self._span_floor = float(min_span)

    def reset(self):
        self._lo = [None, None]
        self._hi = [None, None]
        self._last_raw = None
        self._span_floor = float(self.min_span)

    def _map_axis(self, axis, v, moving, update):
        v = min(1.0, max(0.0, float(v)))
        lo, hi = self._lo[axis], self._hi[axis]
        if lo is None:
            lo = hi = v                       # 首帧：以当前点为活动区起点
        elif update:
            if moving:
                lo = min(v, lo + self.contract)   # 快速扩张 + 缓慢收缩
                hi = max(v, hi - self.contract)
            else:
                lo = min(v, lo)                   # 静止：只扩张、不收缩
                hi = max(v, hi)
        # update=False（书写中）：冻结活动区，传递函数恒定，避免笔画中途重标定导致漂移
        self._lo[axis], self._hi[axis] = lo, hi

        center = 0.5 * (lo + hi)
        # span_floor 决定有效跨度下限：越大→同样手部位移映射到越小屏幕位移（gain 越低、
        # 越接近直接绝对映射）；越小→小动作越被放大。span_floor=min_span 即原始行为。
        span = max(hi - lo, self._span_floor)
        eff_lo = center - 0.5 * span
        t = (v - eff_lo) / span                          # 活动区内归一化
        denom = max(1e-6, 1.0 - 2.0 * self.margin)
        t = (t - self.margin) / denom                    # 应用边缘余量并拉伸到全屏
        return min(1.0, max(0.0, t))

    def map(self, x_norm, y_norm, update=True, span_floor=None):
        """映射归一化手部坐标到全屏 [0,1]。

        update=True：持续校准活动区与 span_floor（悬停/就绪时）。
        update=False：冻结活动区与 span_floor，只套用当前传递函数（书写中），
            保证笔画稳定不漂移、不随掌宽抖动而中途改变灵敏度。
        span_floor∈(0,1.5]：有效跨度下限。近距离传 ~1.0 → 趋近直接绝对映射、gain≈1、
            画圆轻松且不丢触达；远距离传 ~0.22 → 小动作放大、写满全屏。None 则维持当前值。
        """
        if update and span_floor is not None:
            self._span_floor = max(0.05, min(1.5, float(span_floor)))
        if self._last_raw is None:
            moving = True
        else:
            d = abs(x_norm - self._last_raw[0]) + abs(y_norm - self._last_raw[1])
            moving = d > self.move_eps
        self._last_raw = (x_norm, y_norm)
        return self._map_axis(0, x_norm, moving, update), self._map_axis(1, y_norm, moving, update)


def interp_tiers(x, points):
    """在分段标定点 points=[(x_anchor, y), ...] 间按 x 线性插值（piecewise-linear）。

    points 需按 x 升序。x 落在范围外取端点值；点之间线性过渡，避免跳变。
    板书/鼠标用它把"掌宽/参考掌宽"比值映射到活动区 span_floor，实现分段距离手感。
    直接增删/改 points 即可重塑各段。
    """
    if not points:
        return 0.0
    if x <= points[0][0]:
        return points[0][1]
    if x >= points[-1][0]:
        return points[-1][1]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x0 <= x <= x1:
            if x1 <= x0:
                return y1
            t = (x - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
    return points[-1][1]


class MouseController:
    def __init__(self, sensitivity=100, edge_enabled=False, edge_strength=30,
                 edge_y_canvas=True, edge_y_dz_bottom=18, edge_y_dz_top=10):
        self.sensitivity = max(10, sensitivity) / 100.0
        self.last_pos = None
        # 屏幕尺寸改为动态获取，见 _get_screen_size()
        # 新增
        self.edge_enabled = edge_enabled
        self.edge_strength = max(0, min(100, edge_strength)) / 100.0
        self.edge_y_canvas = edge_y_canvas
        self.edge_y_dz_bottom = max(0, min(30, edge_y_dz_bottom)) / 100.0
        self.edge_y_dz_top = max(0, min(20, edge_y_dz_top)) / 100.0

    def _get_screen_size(self):
        """动态获取虚拟屏幕尺寸（跨越所有显示器），支持运行时显示器变化。"""
        return user32.GetSystemMetrics(SM_CXVIRTUALSCREEN), user32.GetSystemMetrics(SM_CYVIRTUALSCREEN)

    def set_sensitivity(self, sensitivity):
        self.sensitivity = max(10, sensitivity) / 100.0

    def set_edge_acceleration(self, enabled, strength,
                              y_canvas=None, y_dz_bottom=None, y_dz_top=None):
        self.edge_enabled = enabled
        self.edge_strength = max(0, min(100, strength)) / 100.0
        if y_canvas is not None:
            self.edge_y_canvas = y_canvas
        if y_dz_bottom is not None:
            self.edge_y_dz_bottom = max(0, min(30, y_dz_bottom)) / 100.0
        if y_dz_top is not None:
            self.edge_y_dz_top = max(0, min(20, y_dz_top)) / 100.0

    def to_screen(self, x_norm, y_norm, apply_accel=True):
        # apply_accel=False：跳过边缘加速/画布映射（供已自带活动区映射的板书模式使用）
        if apply_accel:
            if self.edge_enabled and self.edge_strength > 0:
                x_norm = _edge_map(x_norm, self.edge_strength)
            if self.edge_enabled and self.edge_y_canvas:
                y_norm = _canvas_map(y_norm, self.edge_y_dz_top, self.edge_y_dz_bottom)
            elif self.edge_enabled and self.edge_strength > 0:
                y_norm = _edge_map(y_norm, self.edge_strength)
        screen_w, screen_h = self._get_screen_size()
        return int(x_norm * screen_w), int(y_norm * screen_h)

    def move_to_normalized(self, x_norm, y_norm, apply_accel=True):
        # apply_accel=False：坐标已由活动区映射拉伸过（鼠标/板书自带远距离全屏映射），
        # 跳过边缘加速避免双重拉伸失真。
        if apply_accel:
            target_x, target_y = self.to_screen(x_norm, y_norm)
        else:
            target_x, target_y = self.to_screen(x_norm, y_norm, apply_accel=False)
        screen_w, screen_h = self._get_screen_size()

        if self.last_pos is None:
            new_x, new_y = target_x, target_y
        else:
            new_x = self.last_pos[0] + (target_x - self.last_pos[0]) * self.sensitivity
            new_y = self.last_pos[1] + (target_y - self.last_pos[1]) * self.sensitivity

        new_x = max(0, min(screen_w - 1, new_x))
        new_y = max(0, min(screen_h - 1, new_y))

        set_cursor_pos(new_x, new_y)
        self.last_pos = (new_x, new_y)
        return new_x, new_y
    
    def left_down(self):
        """按下左键（不释放）。"""
        MOUSEEVENTF_LEFTDOWN = 0x0002
        user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        return True

    def left_up(self):
        """释放左键。"""
        MOUSEEVENTF_LEFTUP = 0x0004
        user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
        return True

    def left_click(self):
        """单击左键（向后兼容，内部复用 down/up）。"""
        self.left_down()
        time.sleep(0.02)
        self.left_up()
        return True

    def right_click(self):
        mouse_event_click('right')
        return True

    def double_click(self):
        """双击左键。"""
        self.left_click()
        time.sleep(0.05)
        self.left_click()
        return True
    
    def scroll_wheel(self, amount=120):
        mouse_event_scroll(amount)
        return True
    
    def reset(self):
        self.last_pos = None
