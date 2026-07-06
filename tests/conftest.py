"""pytest 全局配置：测试间 sys.modules 隔离。

多个测试模块在模块顶层向 sys.modules 注入 mock 模块（ctypes、PyQt6、win32con 等），
如果不清理，后续测试会继承被污染的模块缓存，导致 import 失败或行为异常。

此 autouse fixture 在每个测试前后快照/恢复 sys.modules，确保测试隔离。
"""
import sys

import pytest


@pytest.fixture(autouse=True)
def _isolate_sys_modules():
    """每个测试前后快照/恢复 sys.modules，防止 mock 污染跨测试泄漏。"""
    snapshot = dict(sys.modules)
    yield
    # 删除测试期间新增的模块
    for key in list(sys.modules.keys()):
        if key not in snapshot:
            del sys.modules[key]
    # 恢复被替换的模块
    sys.modules.update(snapshot)
