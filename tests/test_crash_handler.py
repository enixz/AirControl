"""crash_handler 崩溃捕获单元测试。

注意：install() 会改全局 sys.excepthook / threading.excepthook / faulthandler，
每个触碰全局的用例都在 finally 里完整复原，避免污染 pytest 自身的异常处理。
"""
import faulthandler
import os
import sys
import threading

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'app'))

import crash_handler  # noqa: E402


def test_record_writes(tmp_path):
    p = str(tmp_path / "crash.log")
    assert crash_handler._record(p, "测试头", "line1\nline2") is True
    txt = open(p, encoding="utf-8").read()
    assert "测试头" in txt and "line1" in txt and "line2" in txt


def test_record_bad_path_returns_false_not_raises():
    bad = os.path.join(os.sep, "nonexistent_dir_xyz_123", "c.log")
    assert crash_handler._record(bad, "h", "t") is False


def test_thread_excepthook_records(tmp_path):
    """工作线程未捕获异常应落盘且带线程名，不抛。"""
    p = str(tmp_path / "crash.log")
    saved = crash_handler._crash_path
    crash_handler._crash_path = p
    try:
        exc = RuntimeError("boom-thread")
        try:
            raise exc
        except RuntimeError:
            tb = sys.exc_info()[2]

        class Args:
            exc_type = RuntimeError
            exc_value = exc
            exc_traceback = tb
            thread = threading.Thread(name="InferenceWorker")

        crash_handler._thread_excepthook(Args)
        txt = open(p, encoding="utf-8").read()
        assert "boom-thread" in txt
        assert "InferenceWorker" in txt
    finally:
        crash_handler._crash_path = saved


def test_install_sets_hooks_and_is_idempotent(tmp_path):
    orig_except = sys.excepthook
    orig_thread = getattr(threading, "excepthook", None)
    g_installed = crash_handler._installed
    g_file = crash_handler._crash_file
    g_path = crash_handler._crash_path
    try:
        crash_handler._installed = False
        crash_handler._crash_file = None
        crash_handler._crash_path = None

        path = crash_handler.install(base_dir=str(tmp_path))
        assert os.path.exists(path)
        assert sys.excepthook is crash_handler._excepthook
        if hasattr(threading, "excepthook"):
            assert threading.excepthook is crash_handler._thread_excepthook
        # 幂等：二次 install 返回同一路径、不重置到别处
        assert crash_handler.install(base_dir=str(tmp_path / "other")) == path
        assert "会话启动" in open(path, encoding="utf-8").read()
    finally:
        # 复原全局，避免污染后续测试与 pytest 自身
        if crash_handler._crash_file:
            try:
                crash_handler._crash_file.close()
            except OSError:
                pass
        faulthandler.enable()  # 恢复到默认（stderr）
        sys.excepthook = orig_except
        if orig_thread is not None:
            threading.excepthook = orig_thread
        crash_handler._installed = g_installed
        crash_handler._crash_file = g_file
        crash_handler._crash_path = g_path
