import ctypes
import ctypes.wintypes
import time

user32 = ctypes.windll.user32

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
    if t <= low:
        return 0.0
    if t >= high:
        return 1.0
    return (t - low) / (high - low)


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
        """动态获取当前屏幕分辨率，支持运行时显示器变化。"""
        return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)

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

    def to_screen(self, x_norm, y_norm):
        if self.edge_enabled and self.edge_strength > 0:
            x_norm = _edge_map(x_norm, self.edge_strength)
        if self.edge_enabled and self.edge_y_canvas:
            y_norm = _canvas_map(y_norm, self.edge_y_dz_top, self.edge_y_dz_bottom)
        elif self.edge_enabled and self.edge_strength > 0:
            y_norm = _edge_map(y_norm, self.edge_strength)
        screen_w, screen_h = self._get_screen_size()
        return int(x_norm * screen_w), int(y_norm * screen_h)

    def move_to_normalized(self, x_norm, y_norm):
        target_x, target_y = self.to_screen(x_norm, y_norm)
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