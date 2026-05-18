import os
import time
import logging
import winreg

import win32api
import win32con
import win32gui
import win32process
import psutil

logger = logging.getLogger("voice_assistant")

ASSISTANT_PROFILES = {
    "doubao": {
        "display_name": "豆包",
        "process_names": ["Doubao.exe", "doubao.exe"],
        "window_keywords": ["豆包"],
        "window_excludes": ["浏览器", "Chrome", "Edge", "Firefox", "Brave", "Opera"],
        "exe_names": ["Doubao.exe"],
        "registry_keywords": ["doubao", "豆包"],
        "search_roots": [
            os.path.join(os.environ.get("LOCALAPPDATA", ""), "doubao", "Application"),
            os.path.join(os.environ.get("LOCALAPPDATA", ""), "Doubao", "Application"),
            os.path.join(os.environ.get("LOCALAPPDATA", ""), "Doubao"),
            os.path.join(os.environ.get("LOCALAPPDATA", ""), "ByteDance", "Doubao"),
            os.path.join(os.environ.get("LOCALAPPDATA", ""), "ByteDance", "Doubao", "Application"),
            os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Doubao"),
            os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Doubao", "Application"),
            os.path.join(os.environ.get("PROGRAMFILES", ""), "Doubao"),
            os.path.join(os.environ.get("PROGRAMFILES", ""), "Doubao", "Application"),
            os.path.join(os.environ.get("PROGRAMFILES(X86)", ""), "Doubao"),
            os.path.join(os.environ.get("APPDATA", ""), "doubao", "Application"),
            os.path.join(os.environ.get("APPDATA", ""), "Doubao", "Application"),
        ],
        "call_hotkey": (win32con.VK_MENU, 0x44),
        "hangup_hotkey": (win32con.VK_MENU, 0x51),
    },
    "qianwen": {
        "display_name": "通义千问",
        "process_names": ["TongyiQianwen.exe", "Tongyi.exe"],
        "window_keywords": ["通义千问", "Tongyi", "通义"],
        "window_excludes": ["浏览器", "Chrome", "Edge", "Firefox", "Brave", "Opera"],
        "exe_names": ["TongyiQianwen.exe", "Tongyi.exe"],
        "registry_keywords": ["tongyi", "通义千问", "qianwen"],
        "search_roots": [
            os.path.join(os.environ.get("LOCALAPPDATA", ""), "TongyiQianwen"),
            os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "TongyiQianwen"),
            os.path.join(os.environ.get("PROGRAMFILES", ""), "TongyiQianwen"),
            os.path.join(os.environ.get("PROGRAMFILES(X86)", ""), "TongyiQianwen"),
        ],
        "call_hotkey": (win32con.VK_MENU, 0x44),
        "hangup_hotkey": (win32con.VK_MENU, 0x51),
    },
}

_UNINSTALL_KEYS = [
    (winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Uninstall"),
    (winreg.HKEY_LOCAL_MACHINE, r"Software\Microsoft\Windows\CurrentVersion\Uninstall"),
    (winreg.HKEY_LOCAL_MACHINE, r"Software\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall"),
]


def _find_exe_from_registry(profile):
    keywords = profile.get("registry_keywords", [])
    exe_names = {n.lower() for n in profile["exe_names"]}
    for hive, sub_path in _UNINSTALL_KEYS:
        try:
            with winreg.OpenKey(hive, sub_path) as key:
                i = 0
                while True:
                    try:
                        sk_name = winreg.EnumKey(key, i)
                        i += 1
                        with winreg.OpenKey(key, sk_name) as sub_key:
                            try:
                                display_name = winreg.QueryValueEx(sub_key, "DisplayName")[0]
                            except OSError:
                                continue
                            matched = any(kw.lower() in display_name.lower() for kw in keywords)
                            if not matched:
                                continue
                            exe_path = None
                            _SKIP_EXE_PARTS = {"uninstall", "update", "repair", "helper", "crash"}
                            for value_name in ("DisplayIcon", "InstallLocation"):
                                try:
                                    val = winreg.QueryValueEx(sub_key, value_name)[0]
                                except OSError:
                                    continue
                                if not val:
                                    continue
                                val = val.strip().strip('"')
                                if value_name == "DisplayIcon" and "," in val:
                                    val = val.split(",")[0].strip().strip('"')
                                if os.path.isfile(val) and val.lower().endswith(".exe"):
                                    basename = os.path.basename(val).lower()
                                    if not any(skip in basename for skip in _SKIP_EXE_PARTS):
                                        exe_path = val
                                        break
                                    continue
                                if os.path.isdir(val):
                                    for en in profile["exe_names"]:
                                        candidate = os.path.join(val, en)
                                        if os.path.isfile(candidate):
                                            exe_path = candidate
                                            break
                                    if exe_path:
                                        break
                            if exe_path:
                                logger.info("注册表找到: %s -> %s", display_name, exe_path)
                                return exe_path
                    except OSError:
                        break
        except OSError:
            continue
    return None


def _find_exe_from_process(profile):
    exe_names = {n.lower() for n in profile["exe_names"]}
    for proc in psutil.process_iter(["name", "exe"]):
        try:
            if proc.info["name"] and proc.info["name"].lower() in exe_names:
                exe_path = proc.info["exe"]
                if exe_path and os.path.isfile(exe_path):
                    logger.info("从运行进程找到: %s", exe_path)
                    return exe_path
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return None


def _find_exe_from_search_roots(profile):
    for root_dir in profile["search_roots"]:
        if not root_dir or not os.path.isdir(root_dir):
            continue
        for dirpath, _, filenames in os.walk(root_dir):
            for exe_name in profile["exe_names"]:
                if exe_name.lower() in (f.lower() for f in filenames):
                    return os.path.join(dirpath, exe_name)
    return None


def _find_exe(profile):
    result = _find_exe_from_registry(profile)
    if result:
        return result
    result = _find_exe_from_process(profile)
    if result:
        return result
    result = _find_exe_from_search_roots(profile)
    if result:
        return result
    for exe_name in profile["exe_names"]:
        candidate = os.path.join(
            os.environ.get("LOCALAPPDATA", ""), "Microsoft", "WindowsApps", exe_name
        )
        if os.path.isfile(candidate):
            return candidate
    return None


def _is_process_running(profile):
    for proc in psutil.process_iter(["name"]):
        try:
            pname = proc.info["name"]
            if pname and pname.lower() in (n.lower() for n in profile["process_names"]):
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return False


def _enum_windows_by_keywords(profile):
    keywords = profile["window_keywords"]
    excludes = profile.get("window_excludes", [])
    found = []

    def _cb(hwnd, _):
        title = win32gui.GetWindowText(hwnd)
        if not title:
            return
        for exc in excludes:
            if exc.lower() in title.lower():
                return
        for kw in keywords:
            if kw.lower() in title.lower():
                found.append(hwnd)
                return

    win32gui.EnumWindows(_cb, None)
    return found


def _enum_windows_by_process(profile):
    process_names = {n.lower() for n in profile["process_names"]}
    found = []

    def _cb(hwnd, _):
        try:
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            proc = psutil.Process(pid)
            if proc.name().lower() in process_names:
                title = win32gui.GetWindowText(hwnd)
                if title:
                    found.append(hwnd)
        except (psutil.NoSuchProcess, psutil.AccessDenied, Exception):
            pass

    win32gui.EnumWindows(_cb, None)
    return found


def _enum_all_process_windows(profile):
    process_names = {n.lower() for n in profile["process_names"]}
    found = []

    def _cb(hwnd, _):
        try:
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            proc = psutil.Process(pid)
            if proc.name().lower() in process_names:
                title = win32gui.GetWindowText(hwnd)
                if title:
                    found.append((hwnd, title))
        except (psutil.NoSuchProcess, psutil.AccessDenied, Exception):
            pass

    win32gui.EnumWindows(_cb, None)
    return found


def _bring_to_front(hwnd):
    try:
        style = win32gui.GetWindowLong(hwnd, win32con.GWL_STYLE)
        if style & win32con.WS_MINIMIZE:
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        elif not win32gui.IsWindowVisible(hwnd):
            win32gui.ShowWindow(hwnd, win32con.SW_SHOW)
        win32gui.SetForegroundWindow(hwnd)
    except Exception as e:
        logger.warning("置顶窗口失败: %s", e)


def _minimize_window(hwnd):
    try:
        win32gui.ShowWindow(hwnd, win32con.SW_MINIMIZE)
    except Exception as e:
        logger.warning("最小化窗口失败: %s", e)


def _send_hotkey(hotkey):
    if not hotkey or not any(hotkey):
        return
    try:
        for vk in hotkey:
            if vk:
                win32api.keybd_event(vk, 0, 0, 0)
        time.sleep(0.05)
        for vk in reversed(hotkey):
            if vk:
                win32api.keybd_event(vk, 0, win32con.KEYEVENTF_KEYUP, 0)
    except Exception as e:
        logger.warning("发送快捷键失败: %s", e)


def _press_esc():
    try:
        win32api.keybd_event(win32con.VK_ESCAPE, 0, 0, 0)
        time.sleep(0.05)
        win32api.keybd_event(win32con.VK_ESCAPE, 0, win32con.KEYEVENTF_KEYUP, 0)
        time.sleep(0.1)
    except Exception as e:
        logger.warning("发送 Esc 失败: %s", e)


def _find_and_focus(profile):
    hwnds = _enum_windows_by_keywords(profile)
    if hwnds:
        _bring_to_front(hwnds[0])
        return True

    hwnds = _enum_windows_by_process(profile)
    if hwnds:
        _bring_to_front(hwnds[0])
        return True

    if _is_process_running(profile):
        logger.error(
            "%s 进程在运行但未找到窗口，请手动打开主窗口后重试",
            profile["display_name"],
        )
        return False

    return None


class VoiceAssistantService:
    def __init__(self, assistant="doubao"):
        self.assistant = assistant
        self.aircontrol_hwnd = None

    def set_assistant(self, assistant):
        self.assistant = assistant

    def get_profile(self):
        return ASSISTANT_PROFILES.get(self.assistant, ASSISTANT_PROFILES["doubao"])

    def activate(self):
        profile = self.get_profile()
        try:
            result = _find_and_focus(profile)

            if result is True:
                time.sleep(0.3)
                _press_esc()
                _send_hotkey(profile["call_hotkey"])
                logger.info("已唤醒 %s (Esc + Alt+D)", profile["display_name"])
                return True

            if result is False:
                return False

            logger.info("未找到 %s，尝试启动...", profile["display_name"])
            exe_path = _find_exe(profile)
            if not exe_path:
                logger.error("未找到 %s 客户端，请先安装桌面版", profile["display_name"])
                return False

            try:
                os.startfile(exe_path)
                logger.info("已启动: %s", exe_path)
            except Exception as e:
                logger.error("启动 %s 失败: %s", profile["display_name"], e)
                return False

            for _ in range(20):
                time.sleep(0.5)
                hwnds = _enum_windows_by_keywords(profile)
                if not hwnds:
                    hwnds = _enum_windows_by_process(profile)
                if hwnds:
                    _bring_to_front(hwnds[0])
                    time.sleep(0.5)
                    _press_esc()
                    _send_hotkey(profile["call_hotkey"])
                    logger.info("已启动并唤醒 %s (Esc + Alt+D)", profile["display_name"])
                    return True

            logger.warning("%s 启动后未检测到窗口", profile["display_name"])
            return False
        finally:
            self._restore_aircontrol_focus()

    def hang_up(self):
        profile = self.get_profile()
        try:
            if not _is_process_running(profile):
                logger.warning("%s 未运行，无需挂断", profile["display_name"])
                return False

            all_windows = _enum_all_process_windows(profile)
            if not all_windows:
                logger.warning("未找到 %s 的任何窗口", profile["display_name"])
                return False

            sent = False
            for hwnd, title in all_windows:
                try:
                    _bring_to_front(hwnd)
                    time.sleep(0.15)
                    _send_hotkey(profile["hangup_hotkey"])
                    logger.info("已向窗口 [%s] 发送 Alt+Q", title)
                    sent = True
                except Exception as e:
                    logger.warning("向窗口 [%s] 发送 Alt+Q 失败: %s", title, e)

            if sent:
                logger.info("已挂断 %s (Alt+Q)", profile["display_name"])
                time.sleep(0.3)
                for hwnd, title in all_windows:
                    if win32gui.IsWindow(hwnd):
                        _minimize_window(hwnd)
                        logger.info("已最小化窗口 [%s]", title)
            return sent
        finally:
            self._restore_aircontrol_focus()

    def _restore_aircontrol_focus(self):
        """将焦点归还给 AirControl 主窗口。"""
        if self.aircontrol_hwnd:
            try:
                _bring_to_front(self.aircontrol_hwnd)
                logger.info("已恢复 AirControl 窗口焦点 (hwnd=%s)", self.aircontrol_hwnd)
            except Exception as e:
                logger.warning("恢复 AirControl 焦点失败: %s", e)

    def is_running(self):
        profile = self.get_profile()
        return _is_process_running(profile)
