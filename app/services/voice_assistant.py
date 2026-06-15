import glob
import json
import os
import string
import time
import logging
import winreg

import win32api
import win32con
import win32gui
import win32process
import psutil

logger = logging.getLogger("voice_assistant")

# 跨次启动磁盘缓存：避免每次冷启动都重扫开始菜单 + 全盘
_CACHE_FILE = os.path.join(
    os.environ.get("LOCALAPPDATA", os.path.expanduser("~")),
    "AirControl",
    "exe_paths.json",
)


def _load_cache():
    try:
        with open(_CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _save_cache(cache):
    try:
        os.makedirs(os.path.dirname(_CACHE_FILE), exist_ok=True)
        with open(_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except OSError as e:
        logger.debug("缓存写入失败: %s", e)

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
                            # 收集所有候选目录：DisplayIcon / InstallLocation / UninstallString
                            # 任何一个能拿到的"路径"都退化成它所在的目录，再去目录里找目标 exe
                            candidate_dirs = []
                            for value_name in ("DisplayIcon", "InstallLocation", "UninstallString"):
                                try:
                                    val = winreg.QueryValueEx(sub_key, value_name)[0]
                                except OSError:
                                    continue
                                if not val:
                                    continue
                                val = val.strip().strip('"')
                                # DisplayIcon 可能是 "C:\foo\bar.exe,0" 这种形式
                                if "," in val:
                                    val = val.split(",")[0].strip().strip('"')
                                # 直接命中目标 exe
                                if os.path.isfile(val) and val.lower().endswith(".exe"):
                                    basename = os.path.basename(val).lower()
                                    if not any(skip in basename for skip in _SKIP_EXE_PARTS):
                                        exe_path = val
                                        break
                                # val 是文件（.ico / uninstall.exe / 其他）→ 取它的目录
                                if os.path.isfile(val):
                                    candidate_dirs.append(os.path.dirname(val))
                                # val 本身是目录
                                elif os.path.isdir(val):
                                    candidate_dirs.append(val)
                            # 拿到候选目录后，逐一搜目标 exe（仅本目录 + 一层子目录）
                            if not exe_path:
                                for d in candidate_dirs:
                                    if not d or not os.path.isdir(d):
                                        continue
                                    for en in profile["exe_names"]:
                                        # 同级
                                        c = os.path.join(d, en)
                                        if os.path.isfile(c):
                                            exe_path = c
                                            break
                                        # 一层子目录（如 Application\Doubao.exe）
                                        for sub in os.listdir(d):
                                            sub_path = os.path.join(d, sub)
                                            if os.path.isdir(sub_path):
                                                c2 = os.path.join(sub_path, en)
                                                if os.path.isfile(c2):
                                                    exe_path = c2
                                                    break
                                        if exe_path:
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


def _prefer_launcher(path, exe_names):
    """有些 Electron/Chromium 应用的 App Paths 指向内部进程（如 \\app\\xxx.exe），
    但真正的启动器在上一层（\\xxx.exe）。检测到同名 exe 在父目录就改用父目录的版本。
    """
    if not path:
        return path
    parent = os.path.dirname(path)
    grandparent = os.path.dirname(parent)
    parent_name = os.path.basename(parent).lower()
    # 仅当当前文件位于 "app" / "application" / "bin" 这种内部子目录时才提升
    if parent_name in ("app", "application", "bin", "Doubao", "doubao") and grandparent:
        basename = os.path.basename(path)
        sibling = os.path.join(grandparent, basename)
        if os.path.isfile(sibling) and os.path.normcase(sibling) != os.path.normcase(path):
            logger.info("启动器优先: %s -> %s", path, sibling)
            return sibling
    return path


def _find_exe_from_app_paths(profile):
    """从 Windows App Paths 注册表查找（很多正版安装都会注册这里）"""
    base = r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths"
    for exe_name in profile["exe_names"]:
        for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
            try:
                with winreg.OpenKey(hive, f"{base}\\{exe_name}") as k:
                    path, _ = winreg.QueryValueEx(k, None)
                    if path and os.path.isfile(path):
                        path = _prefer_launcher(path, profile["exe_names"])
                        logger.info("App Paths 找到: %s", path)
                        return path
            except OSError:
                continue
    return None


def _find_exe_from_start_menu(profile):
    """解析开始菜单快捷方式 (.lnk)。已安装的应用基本都会创建这里。

    这是最跨机器可靠的方式：不依赖固定路径，不依赖注册表特定字段，
    只要应用在开始菜单里出现过就能找到。
    """
    try:
        import win32com.client  # 延迟导入：未用到时不强依赖
    except ImportError:
        return None

    keywords = [kw.lower() for kw in profile.get("registry_keywords", [])]
    exe_names_lower = {n.lower() for n in profile["exe_names"]}
    _SKIP_PARTS = {"uninstall", "update", "repair", "helper", "crash"}

    start_menu_roots = [
        os.path.join(os.environ.get("PROGRAMDATA", ""), "Microsoft", "Windows", "Start Menu", "Programs"),
        os.path.join(os.environ.get("APPDATA", ""), "Microsoft", "Windows", "Start Menu", "Programs"),
    ]

    try:
        shell = win32com.client.Dispatch("WScript.Shell")
    except Exception as e:
        logger.debug("WScript.Shell 不可用: %s", e)
        return None

    for root in start_menu_roots:
        if not root or not os.path.isdir(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if not fn.lower().endswith(".lnk"):
                    continue
                fn_low = fn.lower()
                # 文件名或所在目录名命中关键字才解析（避免解析整个开始菜单）
                if not any(kw in fn_low or kw in dirpath.lower() for kw in keywords):
                    continue
                lnk_path = os.path.join(dirpath, fn)
                try:
                    sc = shell.CreateShortCut(lnk_path)
                    target = sc.Targetpath
                except Exception:
                    continue
                if not target or not os.path.isfile(target):
                    continue
                basename = os.path.basename(target).lower()
                if basename not in exe_names_lower:
                    continue
                if any(skip in basename for skip in _SKIP_PARTS):
                    continue
                logger.info("开始菜单找到: %s -> %s", fn, target)
                return target
    return None


def _find_exe_from_all_drives(profile):
    """兜底全盘搜（限定常见安装位置，不暴力遍历）。"""
    _SKIP_PARTS = {"uninstall", "update", "repair", "helper", "crash"}
    drives = [d + ":" for d in string.ascii_uppercase if os.path.isdir(d + ":")]
    # 常见安装目录模板（不会触发 C:\Windows 之类的深度遍历）
    tail_patterns = [
        r"Doubao\{exe}",
        r"Doubao\Application\{exe}",
        r"豆包\{exe}",
        r"ByteDance\Doubao\{exe}",
        r"ByteDance\Doubao\Application\{exe}",
        r"Program Files\Doubao\{exe}",
        r"Program Files\Doubao\Application\{exe}",
        r"Program Files (x86)\Doubao\{exe}",
        r"TongyiQianwen\{exe}",
        r"TongyiQianwen\Application\{exe}",
        r"通义千问\{exe}",
        r"Program Files\TongyiQianwen\{exe}",
        # 用户自选盘安装常见模式
        r"Apps\Doubao\{exe}",
        r"Apps\豆包\{exe}",
        r"Software\Doubao\{exe}",
    ]
    for drv in drives:
        for tpl in tail_patterns:
            for exe_name in profile["exe_names"]:
                cand = os.path.join(drv + os.sep, tpl.format(exe=exe_name))
                if "*" in cand:
                    hits = glob.glob(cand)
                    for h in hits:
                        if os.path.isfile(h) and not any(s in h.lower() for s in _SKIP_PARTS):
                            logger.info("全盘搜索找到: %s", h)
                            return h
                elif os.path.isfile(cand):
                    logger.info("全盘搜索找到: %s", cand)
                    return cand
    return None


def _find_exe(profile):
    """六路并联 + 跨次启动缓存，确保换机器也能找到。

    顺序按速度 + 可靠性：
      1. 内存缓存（本进程内重复调用）
      2. 磁盘缓存（上次成功的路径，先验证还存在）
      3. 进程（已在运行 → 路径最权威）
      4. App Paths 注册表
      5. Uninstall 注册表（含 .ico/UninstallString 退化为目录）
      6. 开始菜单 .lnk 解析（最跨机器可靠）
      7. 硬编码 search_roots
      8. WindowsApps 兜底
      9. 全盘常见目录扫描（最后兜底）
    """
    cache_key = profile["display_name"]
    cache = getattr(_find_exe, "_disk_cache", None)
    if cache is None:
        cache = _load_cache()
        _find_exe._disk_cache = cache

    # 2. 磁盘缓存命中，校验文件还在再用
    cached = cache.get(cache_key)
    if cached and os.path.isfile(cached):
        return cached

    # 3-9 依次尝试
    finders = [
        _find_exe_from_process,
        _find_exe_from_app_paths,
        _find_exe_from_registry,
        _find_exe_from_start_menu,
        _find_exe_from_search_roots,
    ]
    for fn in finders:
        try:
            result = fn(profile)
        except Exception as e:
            logger.debug("%s 抛异常: %s", fn.__name__, e)
            continue
        if result:
            cache[cache_key] = result
            _save_cache(cache)
            return result

    # 8. WindowsApps 兜底
    for exe_name in profile["exe_names"]:
        candidate = os.path.join(
            os.environ.get("LOCALAPPDATA", ""), "Microsoft", "WindowsApps", exe_name
        )
        if os.path.isfile(candidate):
            cache[cache_key] = candidate
            _save_cache(cache)
            return candidate

    # 9. 全盘扫常见目录
    result = _find_exe_from_all_drives(profile)
    if result:
        cache[cache_key] = result
        _save_cache(cache)
        return result

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
