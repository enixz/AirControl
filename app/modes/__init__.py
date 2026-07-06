from .base import ModeBase, ModeResult
from .draw_mode import DrawMode
from .mouse_mode import MouseMode
from .presentation import PresentationMode

# 模式元数据单一来源：所有模块统一引用，避免散落硬编码导致不一致。
# 新增模式时只需在此处追加，cycle_mode / 配置校验 / UI 下拉框 / 中文名映射自动同步。
MODE_NAMES = ("presentation", "mouse", "draw")
MODE_NAME_ZH = {
    "presentation": "演示模式",
    "mouse": "鼠标模式",
    "draw": "板书模式",
}

__all__ = [
    "ModeBase", "ModeResult", "PresentationMode", "MouseMode", "DrawMode",
    "MODE_NAMES", "MODE_NAME_ZH",
]
