import subprocess
import logging
import win32api
import win32con
import win32gui

logger = logging.getLogger(__name__)

class PptController:
    def __init__(self, target_app="WPS"):
        self.target_app = target_app

    def set_target_app(self, target_app):
        self.target_app = target_app

    def next_slide(self):
        # 模拟按下 PageDown
        win32api.keybd_event(win32con.VK_NEXT, 0, 0, 0)
        win32api.keybd_event(win32con.VK_NEXT, 0, win32con.KEYEVENTF_KEYUP, 0)
        logger.info("PPT操作: 下一页 (PageDown)")
        return True

    def prev_slide(self):
        # 模拟按下 PageUp
        win32api.keybd_event(win32con.VK_PRIOR, 0, 0, 0)
        win32api.keybd_event(win32con.VK_PRIOR, 0, win32con.KEYEVENTF_KEYUP, 0)
        logger.info("PPT操作: 上一页 (PageUp)")
        return True
        
    def start_presentation(self):
        # 模拟 F5
        win32api.keybd_event(win32con.VK_F5, 0, 0, 0)
        win32api.keybd_event(win32con.VK_F5, 0, win32con.KEYEVENTF_KEYUP, 0)
        logger.info("PPT操作: 开始播放 (F5)")
        return True
        
    def end_presentation(self):
        # 模拟 ESC
        win32api.keybd_event(win32con.VK_ESCAPE, 0, 0, 0)
        win32api.keybd_event(win32con.VK_ESCAPE, 0, win32con.KEYEVENTF_KEYUP, 0)
        logger.info("PPT操作: 结束播放 (Esc)")
        return True
        
    def launch_wps(self):
        try:
            logger.info("尝试启动 WPS 演示...")
            subprocess.run(["cmd", "/c", "start", "wpp"], shell=False)
            return True
        except Exception as e:
            logger.error("启动 WPS 失败: %s", e)
            return False
            
    def launch_ppt(self):
        try:
            logger.info("尝试启动 PowerPoint...")
            subprocess.run(["cmd", "/c", "start", "powerpnt"], shell=False)
            return True
        except Exception as e:
            logger.error("启动 PowerPoint 失败: %s", e)
            return False

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
