import glob
import logging
import os
import shutil
import winreg

import win32api
import win32con
import win32gui

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 可执行文件自动定位
# ---------------------------------------------------------------------------

# 各应用的搜索策略：注册表键 + 常见安装目录通配
_APP_LOCATORS = {
    "wpp": {
        "app_paths_keys": ["wpp.exe", "wps.exe"],  # Windows App Paths
        "vendor_keys": [  # WPS 自己的安装路径键 (InstallRoot 子键)
            r"SOFTWARE\Kingsoft\Office\6.0\common",
            r"SOFTWARE\WOW6432Node\Kingsoft\Office\6.0\common",
        ],
        "vendor_value_name": "InstallRoot",
        "vendor_relative": os.path.join("office6", "wpp.exe"),
        "glob_patterns": [
            r"%LOCALAPPDATA%\Kingsoft\WPS Office\*\office6\wpp.exe",
            r"C:\Program Files\Kingsoft\WPS Office\*\office6\wpp.exe",
            r"C:\Program Files (x86)\Kingsoft\WPS Office\*\office6\wpp.exe",
            r"C:\Program Files\WPS Office\*\office6\wpp.exe",
            r"C:\Users\*\AppData\Local\Kingsoft\WPS Office\*\office6\wpp.exe",
            r"F:\WPS Office\*\office6\wpp.exe",  # 兼容自定义安装盘
            r"D:\WPS Office\*\office6\wpp.exe",
            r"E:\WPS Office\*\office6\wpp.exe",
        ],
        "path_fallback": ["wpp.exe", "wpp"],
        "preferred_filename": "wpp.exe",  # 同目录优先精确文件名
    },
    "powerpnt": {
        "app_paths_keys": ["POWERPNT.EXE"],
        "vendor_keys": [],
        "vendor_value_name": None,
        "vendor_relative": None,
        "glob_patterns": [
            r"C:\Program Files\Microsoft Office\root\Office*\POWERPNT.EXE",
            r"C:\Program Files (x86)\Microsoft Office\root\Office*\POWERPNT.EXE",
            r"C:\Program Files\Microsoft Office\Office*\POWERPNT.EXE",
            r"C:\Program Files (x86)\Microsoft Office\Office*\POWERPNT.EXE",
        ],
        "path_fallback": ["POWERPNT.EXE", "powerpnt"],
    },
}

# 跨实例缓存，避免重复扫描注册表 / 文件系统
_EXECUTABLE_CACHE = {}


def _query_app_paths(key_name):
    """从 Windows App Paths 注册表读取应用完整路径（最权威）"""
    base = r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths"
    for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
        try:
            with winreg.OpenKey(hive, f"{base}\\{key_name}") as k:
                path, _ = winreg.QueryValueEx(k, None)  # 默认值即可执行路径
                if path and os.path.isfile(path):
                    return path
        except OSError:
            continue
    return None


def _query_vendor_root(subkey, value_name):
    """从厂商自己的注册表键读 InstallRoot 之类的安装目录"""
    for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
        try:
            with winreg.OpenKey(hive, subkey) as k:
                root, _ = winreg.QueryValueEx(k, value_name)
                if root:
                    return root
        except OSError:
            continue
    return None


def _prefer_sibling(path, preferred_filename):
    """如果同目录下有更精确的可执行文件，优先返回它。

    例：App Paths 经常把 wpp.exe 指向 wps.exe（WPS 总启动器），
    而我们其实想直接开 wpp.exe（演示模块）。
    """
    if not path or not preferred_filename:
        return path
    sibling = os.path.join(os.path.dirname(path), preferred_filename)
    if os.path.basename(path).lower() != preferred_filename.lower() and os.path.isfile(sibling):
        logger.info("改用同目录精确程序: %s -> %s", path, sibling)
        return sibling
    return path


def find_executable(app_id):
    """跨电脑自动定位可执行文件路径，找不到返回 None。

    搜索顺序：
      1. Windows App Paths 注册表（最权威）
      2. 厂商安装路径注册表 + 相对路径拼接
      3. 常见安装目录通配（取最新版本）
      4. PATH 兜底
    结果会缓存，避免每次启动都扫描。
    """
    if app_id in _EXECUTABLE_CACHE:
        return _EXECUTABLE_CACHE[app_id]

    cfg = _APP_LOCATORS.get(app_id)
    if cfg is None:
        return None

    preferred = cfg.get("preferred_filename")
    found = None
    source = None

    # 1. App Paths
    for key in cfg["app_paths_keys"]:
        hit = _query_app_paths(key)
        if hit:
            found, source = hit, "App Paths"
            break

    # 2. 厂商注册表
    if not found and cfg["vendor_keys"] and cfg["vendor_relative"]:
        for subkey in cfg["vendor_keys"]:
            root = _query_vendor_root(subkey, cfg["vendor_value_name"])
            if root:
                candidate = os.path.join(root, cfg["vendor_relative"])
                if os.path.isfile(candidate):
                    found, source = candidate, "厂商注册表"
                    break

    # 3. 通配搜索（同一目录可能有多个版本号子目录，取最新）
    if not found:
        for pattern in cfg["glob_patterns"]:
            expanded = os.path.expandvars(pattern)
            matches = sorted(glob.glob(expanded), reverse=True)
            for m in matches:
                if os.path.isfile(m):
                    found, source = m, "通配搜索"
                    break
            if found:
                break

    # 4. PATH 兜底
    if not found:
        for name in cfg["path_fallback"]:
            hit = shutil.which(name)
            if hit:
                found, source = hit, "PATH"
                break

    if found:
        found = _prefer_sibling(found, preferred)
        logger.info("通过 %s 找到 %s: %s", source, app_id, found)
        _EXECUTABLE_CACHE[app_id] = found
        return found

    logger.warning("未能在本机定位 %s，请确认软件已安装", app_id)
    _EXECUTABLE_CACHE[app_id] = None
    return None


class PptController:
    def __init__(self, target_app="WPS", config=None):
        self.target_app = target_app
        # 允许 config 显式覆盖搜索结果，应对极端情况（如绿色版/便携版）
        # config.json 示例: {"wps_exe_path": "X:\\portable\\wpp.exe"}
        if config:
            override_wps = config.get("wps_exe_path")
            override_ppt = config.get("powerpoint_exe_path")
            if override_wps and os.path.isfile(override_wps):
                _EXECUTABLE_CACHE["wpp"] = override_wps
                logger.info("使用配置覆盖的 WPS 路径: %s", override_wps)
            if override_ppt and os.path.isfile(override_ppt):
                _EXECUTABLE_CACHE["powerpnt"] = override_ppt
                logger.info("使用配置覆盖的 PowerPoint 路径: %s", override_ppt)

    def set_target_app(self, target_app):
        self.target_app = target_app

    def _ensure_app_active(self):
        """确保目标演示软件（WPS或PowerPoint）处于前台活动状态。"""
        def enum_windows_callback(hwnd, hwnds):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                if self.target_app == "WPS":
                    if "WPS 演示" in title or "WPS Presentation" in title or "- WPS Office" in title:
                        hwnds.append(hwnd)
                else:
                    if "PowerPoint" in title:
                        hwnds.append(hwnd)

        hwnds = []
        win32gui.EnumWindows(enum_windows_callback, hwnds)

        if hwnds:
            try:
                curr_hwnd = win32gui.GetForegroundWindow()
                if curr_hwnd in hwnds:
                    return True  # 已经是当前活动窗口，无需重新切换，避免抖动和延迟
                win32gui.ShowWindow(hwnds[0], win32con.SW_RESTORE)
                win32gui.SetForegroundWindow(hwnds[0])
                logger.info("已自动激活并置顶 %s 窗口", self.target_app)
                return True
            except Exception as e:
                logger.warning("自动激活窗口失败: %s", e)
        return False

    def next_slide(self):
        self._ensure_app_active()
        # 模拟按下 PageDown
        win32api.keybd_event(win32con.VK_NEXT, 0, 0, 0)
        win32api.keybd_event(win32con.VK_NEXT, 0, win32con.KEYEVENTF_KEYUP, 0)
        logger.info("PPT操作: 下一页 (PageDown)")
        return True

    def prev_slide(self):
        self._ensure_app_active()
        # 模拟按下 PageUp
        win32api.keybd_event(win32con.VK_PRIOR, 0, 0, 0)
        win32api.keybd_event(win32con.VK_PRIOR, 0, win32con.KEYEVENTF_KEYUP, 0)
        logger.info("PPT操作: 上一页 (PageUp)")
        return True

    def start_presentation(self):
        self._ensure_app_active()
        # 模拟 F5
        win32api.keybd_event(win32con.VK_F5, 0, 0, 0)
        win32api.keybd_event(win32con.VK_F5, 0, win32con.KEYEVENTF_KEYUP, 0)
        logger.info("PPT操作: 开始播放 (F5)")
        return True

    def end_presentation(self):
        self._ensure_app_active()
        # 模拟 ESC
        win32api.keybd_event(win32con.VK_ESCAPE, 0, 0, 0)
        win32api.keybd_event(win32con.VK_ESCAPE, 0, win32con.KEYEVENTF_KEYUP, 0)
        logger.info("PPT操作: 结束播放 (Esc)")
        return True

    def _launch_by_id(self, app_id, display_name):
        """通用启动逻辑：自动定位 + os.startfile 启动。"""
        exe = find_executable(app_id)
        if not exe:
            logger.error(
                "找不到 %s 可执行文件，请确认软件已安装；"
                "如已安装但仍报错，可在 config.json 加入完整路径覆盖搜索结果。",
                display_name,
            )
            return False
        try:
            logger.info("启动 %s: %s", display_name, exe)
            # os.startfile 等价于双击，不会弹"找不到程序"对话框
            os.startfile(exe)
            return True
        except OSError as e:
            logger.error("启动 %s 失败: %s", display_name, e)
            return False

    def launch_wps(self):
        return self._launch_by_id("wpp", "WPS 演示")

    def launch_ppt(self):
        return self._launch_by_id("powerpnt", "PowerPoint")

    def switch_app(self):
        # 查找包含目标软件的窗口并置顶
        def enum_windows_callback(hwnd, hwnds):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                if self.target_app == "WPS":
                    if "WPS 演示" in title or "WPS Presentation" in title or "- WPS Office" in title:
                        hwnds.append(hwnd)
                else:
                    if "PowerPoint" in title:
                        hwnds.append(hwnd)

        hwnds = []
        win32gui.EnumWindows(enum_windows_callback, hwnds)

        if hwnds:
            try:
                win32gui.ShowWindow(hwnds[0], win32con.SW_RESTORE)
                win32gui.SetForegroundWindow(hwnds[0])
                logger.info("已成功切换到 %s 窗口", self.target_app)
                return True
            except Exception as e:
                logger.error("切换窗口失败: %s", e)
        else:
            logger.warning("未找到运行中的 %s 窗口", self.target_app)
            if self.target_app == "WPS":
                self.launch_wps()
            else:
                self.launch_ppt()
        return False
