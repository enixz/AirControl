"""统一日志配置：所有模块的日志写入 gesture.log，避免碎片化。

历史问题：
  · gesture_recognizer.py 模块级配置了 "gesture" logger（写 gesture.log）；
  · voice_assistant.py 用 "voice_assistant" logger（无 handler，打包后丢失）；
  · inference_worker.py 直接调 logging.info()（走 root，无 handler）；
  · orchestrator.py 用 __name__ logger（无 handler）。

本模块在程序入口最早处调用 setup_logging()，配置 root logger 写入
gesture.log，所有子 logger（含 "gesture"/"voice_assistant"/__name__）
自动继承 root 的 handler 与级别，确保打包后日志不丢失。

幂等：重复调用不会重复添加 handler。
"""
import logging
import os
from logging.handlers import RotatingFileHandler

from runtime_paths import writable_data_dir

_INSTALLED = False

_FORMAT = "%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s"
_DATEFMT = "%H:%M:%S"
_MAX_BYTES = 2 * 1024 * 1024
_BACKUP_COUNT = 2


def setup_logging(level=logging.INFO):
    """配置 root logger 写入 gesture.log。幂等。

    在程序入口（main_ui.main / main.main）最早处调用一次。
    crash_handler.install 之前调用，确保崩溃日志的 logger.critical 也能落盘。
    """
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    base_dir = writable_data_dir()
    log_file = os.path.join(base_dir, "gesture.log")

    root = logging.getLogger()
    # 避免重复添加（防御性）
    if any(getattr(h, "_aircontrol", False) for h in root.handlers):
        return

    handler = RotatingFileHandler(
        log_file, maxBytes=_MAX_BYTES, backupCount=_BACKUP_COUNT, encoding="utf-8"
    )
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
    handler._aircontrol = True  # 标记，防止重复添加
    root.addHandler(handler)
    root.setLevel(level)

    # "gesture" logger 历史上由 gesture_recognizer.py 模块级配置；
    # 统一配置后让其继承 root，避免双重 handler 导致日志写两遍。
    gesture_logger = logging.getLogger("gesture")
    gesture_logger.handlers = []  # 清除 gesture_recognizer.py 模块级添加的 handler
    gesture_logger.propagate = True  # 向上传播到 root


def get_logger(name):
    """获取 logger。统一入口，便于后续扩展（如结构化日志）。"""
    return logging.getLogger(name)
