"""真值输入事件采集器（原始帧录制配套）。

录像期间用独立线程轮询"意图标记键"的物理按键状态，把 down/up 跳变连同
epoch 时间戳写入 truth_events.jsonl（与 meta.jsonl 同目录、同 time.time()
时钟）。离线回放据此把"用户真实意图的点击/拖拽"与检测到的 pinch 事件对
齐，算出检出率/漏检率/误报/延迟——评估报告 P1-1 要求的正是这种带真值
的量化指标，用来决定高级特性默认开还是关。

录制时用法：做手势（捏合）的同时操作标记键——
  - 点按一次 = 意图点击一次；
  - 按住不放 = 意图拖拽中，松开 = 拖拽结束。

近距离录像可直接按键盘空格（默认）。远距离（3–5 米）够不到键盘，用拿在
另一只手里的无线设备做标记通道（config record_truth_marker 切换，支持
逗号分隔多个同时生效）：
  - 无线鼠标【中键】（"mbutton"）：应用从不注入中键，绝对干净；
    有侧键的话 "xbutton1"/"xbutton2" 更顺手（拇指按，同样零注入）；
  - 翻页笔（"pagedown" / "pageup"）：应用不注入任何键盘事件，干净；
  - 无线小键盘 Enter/空格（"enter"/"space"）：干净；
  - 无线鼠标右键（"rbutton"）：应用存在右键手势（拇指-中指捏合触发），
    录制期间不做该手势才可用。

只监听标记键、不监听鼠标左键：应用自身会注入合成左键点击，
GetAsyncKeyState 无法区分物理事件与合成事件，标记通道必须避开左键。

非 Windows 或无 pywin32 时构造即抛错，由 FrameRecorder 捕获降级（录帧主
流程永不受影响）。测试中可注入 get_state 回调摆脱对 win32 的依赖。
"""
import json
import logging
import os
import threading
import time

logger = logging.getLogger(__name__)

# 标记键名 → Windows 虚拟键码（与 config 的 record_truth_marker 取值一一对应；
# 新增取值时同步 config_manager 的 _is_marker_list）
MARKER_VK = {
    "space": 0x20,
    "enter": 0x0D,
    "shift": 0x10,
    "ctrl": 0x11,
    "alt": 0x12,
    "tab": 0x09,
    "x": 0x58,
    "z": 0x5A,
    # 远程标记通道（无线鼠标/翻页笔，供 3–5 米录像使用）
    "rbutton": 0x02,     # 鼠标右键（应用可注入右键：拇指-中指捏合手势，录制时勿做）
    "mbutton": 0x04,     # 鼠标中键/滚轮按下（应用从不注入，绝对干净）
    "xbutton1": 0x05,    # 鼠标侧键 1/后退（应用从不注入，拇指按最顺手）
    "xbutton2": 0x06,    # 鼠标侧键 2/前进
    "pageup": 0x21,      # 翻页笔上页（应用不注入键盘事件）
    "pagedown": 0x22,    # 翻页笔下页
}


def _win32_get_state(vk):
    """默认按键状态源：Windows GetAsyncKeyState（pywin32）。"""
    import win32api
    return bool(win32api.GetAsyncKeyState(vk) & 0x8000)


def normalize_markers(markers):
    """把 "space" 或 "pagedown,pageup" 或 ("space",) 统一成标记键名元组。"""
    if isinstance(markers, str):
        markers = tuple(m.strip() for m in markers.split(","))
    return tuple(m.strip().lower() for m in markers if m and m.strip())


class TruthEventLogger:
    """轮询标记键跳变并落盘的后台线程。"""

    def __init__(self, out_dir, markers=("space",), get_state=None, poll_interval=0.005):
        if get_state is None:
            try:
                import win32api  # noqa: F401
            except ImportError as exc:
                raise RuntimeError("pywin32 不可用，无法采集真值按键事件") from exc
            get_state = _win32_get_state

        self._path = os.path.join(out_dir, "truth_events.jsonl")
        self._markers = {}
        for key in normalize_markers(markers):
            vk = MARKER_VK.get(key)
            if vk is None:
                logger.warning("未知真值标记键 %r，已跳过", key)
                continue
            self._markers[key] = vk
        self._get_state = get_state
        self._poll = float(poll_interval)
        self._stop = threading.Event()
        self._held = set()
        self._closed = False
        self._finalized = False
        self._file = open(self._path, "w", encoding="utf-8", buffering=1)
        self._write({
            "type": "header",
            "markers": {k: "click_drag_intent" for k in self._markers},
            "timebase": "epoch_sec",
            "poll_interval_ms": round(self._poll * 1000, 3),
        })
        self._thread = threading.Thread(
            target=self._loop, name="TruthEventLogger", daemon=True,
        )
        self._thread.start()
        logger.info(
            "真值事件采集 -> %s（标记键: %s）",
            self._path, ",".join(self._markers) or "无",
        )

    @property
    def path(self):
        return self._path

    def _write(self, row):
        try:
            self._file.write(json.dumps(row, separators=(",", ":")) + "\n")
        except Exception:
            logger.exception("真值事件写入失败")

    def _safe_state(self, vk):
        try:
            return bool(self._get_state(vk))
        except Exception:
            return False

    def _loop(self):
        # 先记初始状态：录制开始前已按住的键也能配成完整区间
        states = {}
        for name, vk in self._markers.items():
            down = self._safe_state(vk)
            states[name] = down
            if down:
                self._held.add(name)
                self._write({"t": time.time(), "key": name, "event": "down"})
        while not self._stop.is_set():
            for name, vk in self._markers.items():
                down = self._safe_state(vk)
                if down != states[name]:
                    states[name] = down
                    self._write({
                        "t": time.time(),
                        "key": name,
                        "event": "down" if down else "up",
                    })
                    if down:
                        self._held.add(name)
                    else:
                        self._held.discard(name)
            self._stop.wait(self._poll)

    def close(self, timeout_sec=2.0):
        """停止轮询；仍按住的键补一条 synthetic up，保证区间良构。"""
        if self._finalized:
            return True
        self._closed = True
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=max(0.0, float(timeout_sec)))
        if self._thread.is_alive():
            logger.error("真值事件线程未能在关闭期限内退出；保留事件文件句柄")
            return False
        now = time.time()
        for name in sorted(self._held):
            self._write({
                "t": now, "key": name, "event": "up", "note": "synthetic_on_close",
            })
        self._held.clear()
        self._write({"type": "footer", "t": now})
        try:
            self._file.close()
        except Exception:
            pass
        self._finalized = True
        return True
