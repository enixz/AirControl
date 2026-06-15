"""崩溃捕获：把崩溃统一落盘到 crash.log，避免只进控制台、关窗即失。

实测教训（2026-06-13）：板书中进程被硬杀，gesture.log（挂在 'gesture'
logger 的 RotatingFileHandler）干净截断、抓不到原因——因为：
  · 推理线程的异常被 try/except 吞掉并继续，不会停日志；
  · 原生段错误（OpenCV/MediaPipe/Qt 的 C++ 层）根本不经过 Python logging；
  · PyQt6 槽/虚函数里的未捕获异常会先调 sys.excepthook 再 abort 进程。

本模块在入口最早处 install()，把以上四条逃逸路径都接住：
  1. faulthandler          → 原生段错误时 dump 所有线程的 Python 栈；
  2. sys.excepthook        → 主线程 + PyQt6 槽函数里的未捕获异常；
  3. threading.excepthook  → 工作线程（InferenceWorker 等）的未捕获异常；
  4. Qt 消息处理器          → Qt 的 Fatal/Critical 消息。
原有行为（打印到 stderr、abort）保留——只是额外抓一份到 crash.log。
"""
import datetime
import faulthandler
import logging
import os
import sys
import threading
import traceback

from runtime_paths import writable_data_dir

logger = logging.getLogger("gesture")

_installed = False
_crash_file = None      # faulthandler 需要的常开文件句柄
_crash_path = None


def _ts():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _record(path, header, text):
    """把一段崩溃信息追加到 path。纯函数、不依赖全局，便于测试。"""
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"\n{'=' * 70}\n{_ts()}  {header}\n{'=' * 70}\n{text}\n")
        return True
    except OSError:
        return False


def _default_base_dir():
    return writable_data_dir()


def _excepthook(exc_type, exc_value, exc_tb):
    # Ctrl+C 不当作崩溃
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_tb)
        return
    text = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    if _crash_path:
        _record(_crash_path, "未捕获异常（主线程 / Qt 槽）", text)
    try:
        logger.critical("未捕获异常已写入 %s:\n%s", _crash_path, text)
    except Exception:
        pass
    # 链到原 hook，保留默认的 stderr 打印 / abort 行为
    sys.__excepthook__(exc_type, exc_value, exc_tb)


def _thread_excepthook(args):
    if issubclass(args.exc_type, SystemExit):
        return
    name = args.thread.name if args.thread is not None else "?"
    text = "".join(traceback.format_exception(
        args.exc_type, args.exc_value, args.exc_traceback))
    if _crash_path:
        _record(_crash_path, f"未捕获异常（线程 {name}）", text)
    try:
        logger.critical("线程 %s 未捕获异常已写入 %s:\n%s", name, _crash_path, text)
    except Exception:
        pass


def _install_qt_handler():
    """Qt 致命/严重消息也抓一份。无 PyQt6 时静默跳过。"""
    try:
        from PyQt6.QtCore import qInstallMessageHandler, QtMsgType
    except Exception:
        return

    fatal = (QtMsgType.QtFatalMsg, QtMsgType.QtCriticalMsg)

    def handler(mode, context, message):
        if mode in fatal and _crash_path:
            loc = f"{getattr(context, 'file', '?')}:{getattr(context, 'line', 0)}"
            _record(_crash_path, f"Qt 消息 {mode}", f"{message}\n  ({loc})")
        try:  # 同时回显到 stderr，保留默认可见性
            (sys.__stderr__ or sys.stderr).write(f"[Qt] {message}\n")
        except Exception:
            pass

    qInstallMessageHandler(handler)


def install(base_dir=None):
    """在程序入口最早处调用一次（幂等）。"""
    global _installed, _crash_file, _crash_path
    if _installed:
        return _crash_path
    base_dir = base_dir or _default_base_dir()
    _crash_path = os.path.join(base_dir, "crash.log")

    # faulthandler 需要一个进程存活期常开的文件，崩溃时直接 dump 进去
    try:
        _crash_file = open(_crash_path, "a", encoding="utf-8", buffering=1)
        _crash_file.write(
            f"\n{'=' * 70}\n{_ts()}  会话启动（faulthandler 已启用）\n{'=' * 70}\n"
        )
        faulthandler.enable(file=_crash_file, all_threads=True)
    except OSError:
        _crash_file = None

    sys.excepthook = _excepthook
    if hasattr(threading, "excepthook"):
        threading.excepthook = _thread_excepthook
    _install_qt_handler()

    _installed = True
    try:
        logger.info("崩溃捕获已启用 → %s", _crash_path)
    except Exception:
        pass
    return _crash_path
