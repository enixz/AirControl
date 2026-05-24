import logging
import os
import sys
import threading
import time
import winsound

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import cv2
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from config_manager import ConfigManager
from drawing_overlay import DrawingOverlay
from draw_toolbar import DrawToolbar
from mode_manager import ModeManager
from modes import DrawMode, MouseMode, PresentationMode
from mouse_cursor_overlay import MouseCursorOverlay
from services.camera import CameraService, list_available_cameras
from services.gesture_recognizer import GestureRecognizer
from services.hand_tracker import HandTracker
from services.inference_worker import InferenceWorker
from services.mouse_controller import MouseController
from services.ppt_controller import PptController
from services.voice_assistant import VoiceAssistantService
from services.voice_command import VoiceCommandService
from services.voice_dictation import VoiceDictationService


class SettingsDialog(QDialog):
    # 后台枚举线程完成时通过此信号回主线程刷下拉框（枚举要试开摄像头，
    # 同步跑会冻住设置对话框 1-5s）
    _cameras_enumerated = pyqtSignal(list)

    def __init__(self, config_manager, parent=None):
        super().__init__(parent)
        self.config = config_manager
        self.setWindowTitle("设置")
        self.setMinimumWidth(300)
        self._cameras_enumerated.connect(self._on_cameras_enumerated)
        self.init_ui()
        # 异步枚举其它可用摄像头（当前摄像头已知，无需探测）
        threading.Thread(
            target=self._enumerate_cameras_worker,
            daemon=True,
        ).start()

    def init_ui(self):
        layout = QFormLayout()

        # 摄像头选择：先用"当前"占位，后台枚举完成后再填充其它选项
        self.camera_combo = QComboBox()
        current_idx = self.config.get("camera_index")
        try:
            current_idx = int(current_idx) if current_idx is not None else 0
        except (TypeError, ValueError):
            current_idx = 0
        self.camera_combo.addItem(f"摄像头 {current_idx}（当前）", current_idx)
        # 加个 "正在检测…" 提示项，枚举完成后会被替换为真实列表
        self.camera_combo.addItem("正在检测其它摄像头…", -1)
        self.camera_combo.model().item(1).setEnabled(False)
        layout.addRow("摄像头:", self.camera_combo)

        self.app_combo = QComboBox()
        self.app_combo.addItems(["PowerPoint", "WPS"])
        self.app_combo.setCurrentText(self.config.get("target_app"))
        layout.addRow("控制目标软件:", self.app_combo)

        self.model_combo = QComboBox()
        self.model_combo.addItems(["Lite", "Heavy"])
        self.model_combo.setCurrentText(self.config.get("model_type"))
        layout.addRow("手势模型精度:", self.model_combo)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["presentation", "mouse", "draw"])
        self.mode_combo.setCurrentText(self.config.get("interaction_mode"))
        layout.addRow("交互模式:", self.mode_combo)

        self.cd_spin = QSpinBox()
        self.cd_spin.setRange(500, 3000)
        self.cd_spin.setSingleStep(100)
        self.cd_spin.setValue(int(self.config.get("cooldown") * 1000))
        self.cd_spin.setSuffix(" ms")
        layout.addRow("手势防抖(冷却):", self.cd_spin)

        self.sensitivity_spin = QSpinBox()
        self.sensitivity_spin.setRange(10, 100)
        self.sensitivity_spin.setValue(int(self.config.get("mouse_sensitivity")))
        self.sensitivity_spin.setSuffix(" %")
        layout.addRow("鼠标灵敏度:", self.sensitivity_spin)

        self.edge_check = QCheckBox("边缘加速")
        self.edge_check.setChecked(bool(self.config.get("edge_acceleration_enabled")))
        self.edge_check.stateChanged.connect(self._on_edge_toggled)
        layout.addRow(self.edge_check)

        self.edge_strength_spin = QSpinBox()
        self.edge_strength_spin.setRange(0, 100)
        self.edge_strength_spin.setValue(int(self.config.get("edge_acceleration_strength")))
        self.edge_strength_spin.setSuffix(" %")
        self.edge_strength_spin.setEnabled(self.edge_check.isChecked())
        layout.addRow("边缘加速强度:", self.edge_strength_spin)

        self.y_canvas_check = QCheckBox("Y 轴虚拟画布（推荐用于任务栏）")
        self.y_canvas_check.setChecked(bool(self.config.get("edge_y_canvas_enabled")))
        layout.addRow(self.y_canvas_check)

        self.y_dz_bottom_spin = QSpinBox()
        self.y_dz_bottom_spin.setRange(0, 30)
        self.y_dz_bottom_spin.setValue(int(self.config.get("edge_y_canvas_deadzone_bottom")))
        self.y_dz_bottom_spin.setSuffix(" %")
        layout.addRow("Y 轴底部死区:", self.y_dz_bottom_spin)

        self.y_dz_top_spin = QSpinBox()
        self.y_dz_top_spin.setRange(0, 20)
        self.y_dz_top_spin.setValue(int(self.config.get("edge_y_canvas_deadzone_top")))
        self.y_dz_top_spin.setSuffix(" %")
        layout.addRow("Y 轴顶部死区:", self.y_dz_top_spin)

        self.pen_spin = QSpinBox()
        self.pen_spin.setRange(1, 20)
        self.pen_spin.setValue(int(self.config.get("pen_width")))
        layout.addRow("画笔粗细:", self.pen_spin)

        self.voice_combo = QComboBox()
        self.voice_combo.addItem("豆包", "doubao")
        self.voice_combo.addItem("通义千问", "qianwen")
        idx = self.voice_combo.findData(self.config.get("voice_assistant"))
        if idx >= 0:
            self.voice_combo.setCurrentIndex(idx)
        layout.addRow("语音助手:", self.voice_combo)

        gesture_actions = [
            "next_slide", "prev_slide", "start_presentation",
            "end_presentation", "switch_app",
            "launch_voice_assistant", "hang_up_voice_assistant", "none",
        ]

        self.right_combo = QComboBox()
        self.right_combo.addItems(gesture_actions)
        self.right_combo.setCurrentText(self.config.get_mapping("SWIPE_RIGHT"))
        layout.addRow("右挥动作映射:", self.right_combo)

        self.left_combo = QComboBox()
        self.left_combo.addItems(gesture_actions)
        self.left_combo.setCurrentText(self.config.get_mapping("SWIPE_LEFT"))
        layout.addRow("左挥动作映射:", self.left_combo)

        self.up_combo = QComboBox()
        self.up_combo.addItems(gesture_actions)
        self.up_combo.setCurrentText(self.config.get_mapping("SWIPE_UP"))
        layout.addRow("上挥动作映射:", self.up_combo)

        self.down_combo = QComboBox()
        self.down_combo.addItems(gesture_actions)
        self.down_combo.setCurrentText(self.config.get_mapping("SWIPE_DOWN"))
        layout.addRow("下挥动作映射:", self.down_combo)

        self.fist_combo = QComboBox()
        self.fist_combo.addItems(gesture_actions)
        self.fist_combo.setCurrentText(self.config.get_mapping("FIST"))
        layout.addRow("握拳动作映射:", self.fist_combo)

        self.thumb_combo = QComboBox()
        self.thumb_combo.addItems(gesture_actions)
        self.thumb_combo.setCurrentText(self.config.get_mapping("THUMB_UP"))
        layout.addRow("点赞动作映射:", self.thumb_combo)

        self.scissor_combo = QComboBox()
        self.scissor_combo.addItems(gesture_actions)
        self.scissor_combo.setCurrentText(self.config.get_mapping("SCISSOR"))
        layout.addRow("剪刀手动作映射:", self.scissor_combo)

        self.thumb_down_combo = QComboBox()
        self.thumb_down_combo.addItems(gesture_actions)
        self.thumb_down_combo.setCurrentText(self.config.get_mapping("THUMB_DOWN"))
        layout.addRow("拇指向下动作映射:", self.thumb_down_combo)

        btn_layout = QHBoxLayout()
        save_btn = QPushButton("保存")
        save_btn.clicked.connect(self.save_settings)
        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(save_btn)
        btn_layout.addWidget(cancel_btn)

        main_layout = QVBoxLayout()
        main_layout.addLayout(layout)
        main_layout.addLayout(btn_layout)
        self.setLayout(main_layout)

    def _on_edge_toggled(self, state):
        self.edge_strength_spin.setEnabled(state == Qt.CheckState.Checked.value)

    def _enumerate_cameras_worker(self):
        """后台线程：跳过当前正在用的摄像头，试开其它索引（Windows 上每个无效索引会阻塞 0.5-2s）"""
        current_idx = self.config.get("camera_index")
        try:
            current_idx = int(current_idx) if current_idx is not None else 0
        except (TypeError, ValueError):
            current_idx = 0
        try:
            cams = list_available_cameras(max_probe=4, exclude_index=current_idx)
        except Exception:
            logging.exception("枚举摄像头失败")
            cams = [{"index": current_idx, "name": f"摄像头 {current_idx}（当前）"}]
        self._cameras_enumerated.emit(cams)

    def _on_cameras_enumerated(self, cameras):
        """主线程 slot：用真实摄像头列表替换"正在检测..."占位项"""
        current_idx = self.camera_combo.itemData(0)
        self.camera_combo.clear()
        if not cameras:
            self.camera_combo.addItem(f"摄像头 {current_idx}（当前）", current_idx)
            return
        for cam in cameras:
            self.camera_combo.addItem(cam["name"], cam["index"])
        # 把"当前"那项设为默认选中
        for i in range(self.camera_combo.count()):
            if self.camera_combo.itemData(i) == current_idx:
                self.camera_combo.setCurrentIndex(i)
                break

    def save_settings(self):
        # 摄像头切换比较重，独立处理：先调父窗口的 switch_camera，失败就不写 config
        new_camera_idx = self.camera_combo.currentData()
        if isinstance(new_camera_idx, int) and new_camera_idx >= 0:
            old_camera_idx = self.config.get("camera_index")
            try:
                old_camera_idx = int(old_camera_idx) if old_camera_idx is not None else 0
            except (TypeError, ValueError):
                old_camera_idx = 0
            if new_camera_idx != old_camera_idx:
                parent = self.parent()
                if parent is not None and hasattr(parent, "switch_camera"):
                    ok = parent.switch_camera(new_camera_idx)
                    if not ok:
                        QMessageBox.warning(
                            self, "摄像头切换失败",
                            f"无法启用摄像头 {new_camera_idx}，已保留原摄像头。\n"
                            "请检查设备是否被其它程序占用，或换 USB 口重试。",
                        )
                        return  # 不关对话框，让用户重新选
                    self.config.set("camera_index", new_camera_idx)

        with self.config.batch_update():
            self.config.set("target_app", self.app_combo.currentText())
            self.config.set("model_type", self.model_combo.currentText())
            self.config.set("interaction_mode", self.mode_combo.currentText())
            self.config.set("cooldown", self.cd_spin.value() / 1000.0)
            self.config.set("mouse_sensitivity", self.sensitivity_spin.value())
            self.config.set("edge_acceleration_enabled", self.edge_check.isChecked())
            self.config.set("edge_acceleration_strength", self.edge_strength_spin.value())
            self.config.set("edge_y_canvas_enabled", self.y_canvas_check.isChecked())
            self.config.set("edge_y_canvas_deadzone_bottom", self.y_dz_bottom_spin.value())
            self.config.set("edge_y_canvas_deadzone_top", self.y_dz_top_spin.value())
            self.config.set("pen_width", self.pen_spin.value())
            self.config.set("voice_assistant", self.voice_combo.currentData())
            self.config.set_mapping("SWIPE_RIGHT", self.right_combo.currentText())
            self.config.set_mapping("SWIPE_LEFT", self.left_combo.currentText())
            self.config.set_mapping("SWIPE_UP", self.up_combo.currentText())
            self.config.set_mapping("SWIPE_DOWN", self.down_combo.currentText())
            self.config.set_mapping("FIST", self.fist_combo.currentText())
            self.config.set_mapping("THUMB_UP", self.thumb_combo.currentText())
            self.config.set_mapping("SCISSOR", self.scissor_combo.currentText())
            self.config.set_mapping("THUMB_DOWN", self.thumb_down_combo.currentText())
        self.accept()


class FloatingWindow(QMainWindow):
    # 语音指令在 KWS 工作线程检测到，通过信号 marshal 到主线程执行
    # execute_action 会调用 mode.on_enter/on_exit 等 Qt 控件操作，必须在主线程
    _voice_action_signal = pyqtSignal(str)
    # 听写状态/结果回调同样在 KWS 工作线程触发，必须 marshal 到主线程才能改 UI。
    # 不能用 QTimer.singleShot(0, lambda) — 没有 context 时 timer 附属于调用线程，
    # 而 KWS 工作线程没有 Qt 事件循环，timer 永远不 fire。
    _dictation_status_signal = pyqtSignal(str, object)   # phase, payload
    _dictation_text_signal = pyqtSignal(str, object)     # text, anchor_pos
    _dictation_partial_signal = pyqtSignal(str)          # 实时增量识别结果

    def __init__(self):
        super().__init__()
        # AutoConnection：跨线程 emit 自动用 QueuedConnection 投递到主线程
        self._voice_action_signal.connect(self.execute_action)
        self._dictation_status_signal.connect(self._on_dictation_status)
        self._dictation_text_signal.connect(self._on_dictation_text)
        self._dictation_partial_signal.connect(self._on_dictation_partial)
        self.config = ConfigManager()
        self.overlay = DrawingOverlay(self, pen_width=self.config.get("pen_width"))
        self.overlay.set_pen_auto_scale(self.config.get("pen_width_auto_scale") is not False)
        self.toolbar = DrawToolbar(self)
        self.mouse = MouseController(
            sensitivity=self.config.get("mouse_sensitivity"),
            edge_enabled=self.config.get("edge_acceleration_enabled"),
            edge_strength=self.config.get("edge_acceleration_strength"),
            edge_y_canvas=self.config.get("edge_y_canvas_enabled"),
            edge_y_dz_bottom=self.config.get("edge_y_canvas_deadzone_bottom"),
            edge_y_dz_top=self.config.get("edge_y_canvas_deadzone_top"),
        )
        self.cursor_overlay = MouseCursorOverlay(self)
        self.init_services()
        self.init_ui()
        self.init_timer()
        self._connect_toolbar()
        self._init_modes()

        self.status_text = "Ready"
        self.status_color = (0, 255, 0)
        self.status_timer = 0

        # 进入默认模式（无音效）
        self._set_mode(self.config.get("interaction_mode"), sound=False)

    def init_services(self):
        self.camera = CameraService(
            camera_index=self.config.get("camera_index"),
            # width/height = null/None 时自动探测最高分辨率，跨设备零配置
            width=self.config.get("camera_width"),
            height=self.config.get("camera_height"),
            force_mjpeg=self.config.get("camera_force_mjpeg") is not False,
            min_fps=self.config.get("camera_min_fps") or 20,
        )
        self.camera.start()
        self.tracker = HandTracker(
            max_num_hands=2,
            min_detection_confidence=self.config.get("hand_detection_confidence") or 0.6,
            min_presence_confidence=self.config.get("hand_presence_confidence") or 0.5,
            min_tracking_confidence=self.config.get("hand_tracking_confidence") or 0.5,
            preferred_model_type=self.config.get("model_type"),
            dominant_hand=self.config.get("dominant_hand") or "Right",
        )
        self.recognizer = GestureRecognizer(
            cooldown=self.config.get("cooldown"),
            swipe_threshold=self.config.get("swipe_threshold"),
        )
        self.ppt = PptController(target_app=self.config.get("target_app"), config=self.config)
        self.voice_assistant = VoiceAssistantService(
            assistant=self.config.get("voice_assistant")
        )
        self.voice_assistant.aircontrol_hwnd = int(self.winId())

        # 语音听写服务（SenseVoice-Small 离线 ASR）— 懒加载
        self.voice_dictation = VoiceDictationService(self.config)
        if not self.voice_dictation.is_available():
            logging.info("SenseVoice 模型未就绪，听写功能不可用（draw 模式'开始板书'将提示缺少模型）")

        # 语音指令服务（KWS 离线关键词检测）
        self.voice_command = VoiceCommandService(
            self.config,
            action_callback=self._voice_action_signal.emit,
            dictation_service=self.voice_dictation,
        )
        self.voice_command.set_status_callback(self._on_voice_keyword_detected)
        if self.config.get("voice_command_enabled") is not False:
            try:
                self.voice_command.start()
            except Exception as e:
                logging.warning("语音指令服务启动失败: %s", e)

        # 启动自检：把所有子系统状态打成一个易读的汇总块
        self._run_startup_check()

        # 启动推理工作线程
        self.inference_worker = InferenceWorker(
            self.camera, self.tracker, max_fps=30,
            debug_overlay=bool(self.config.get("debug_overlay")),
        )
        self.inference_worker.frame_ready.connect(self._on_frame_ready)
        self.inference_worker.error_occurred.connect(self._on_inference_error)
        self.inference_worker.fps_updated.connect(self._on_fps_updated)
        self.inference_worker.start()

    def _run_startup_check(self):
        """启动自检——把所有子系统状态打成一块易读的日志，便于排查"为什么不工作"。"""
        lines = ["", "=" * 60, "AirControl 启动自检", "=" * 60]

        # 摄像头
        try:
            cap = self.camera.cap
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            fourcc_v = int(cap.get(cv2.CAP_PROP_FOURCC))
            fourcc_s = "".join(chr((fourcc_v >> 8 * i) & 0xFF) for i in range(4))
            lines.append(f"[OK]   摄像头 #{self.camera.camera_index}: {w}x{h}@{fps:.0f}fps ({fourcc_s})")
        except Exception as e:
            lines.append(f"[BAD]  摄像头不可用: {e}")

        # 手部模型
        try:
            model_name = os.path.basename(self.tracker.model_path)
            lines.append(
                f"[OK]   手部模型: {model_name}  (dominant={self.tracker.dominant_hand})"
            )
        except Exception as e:
            lines.append(f"[BAD]  手部模型加载失败: {e}")

        # 演示控制器
        try:
            from services.ppt_controller import find_executable
            wpp = find_executable("wpp")
            ppt = find_executable("powerpnt")
            if wpp:
                lines.append(f"[OK]   WPS 演示: {wpp}")
            else:
                lines.append("[--]   WPS 演示未安装（演示模式仅支持 PowerPoint）")
            if ppt:
                lines.append(f"[OK]   PowerPoint: {ppt}")
            else:
                lines.append("[--]   PowerPoint 未安装")
            if not wpp and not ppt:
                lines.append("[BAD]  WPS 和 PPT 都没找到，演示模式不可用")
        except Exception as e:
            lines.append(f"[BAD]  演示控制器检查失败: {e}")

        # 语音听写 (SenseVoice)
        try:
            if self.voice_dictation.is_available():
                lines.append("[OK]   语音听写 SenseVoice 模型就绪")
            else:
                lines.append(
                    "[--]   语音听写不可用（缺 models/sense-voice/ 模型文件，'开始板书'功能将提示）"
                )
        except Exception as e:
            lines.append(f"[BAD]  语音听写检查失败: {e}")

        # 语音指令 (KWS)
        try:
            if getattr(self.voice_command, "is_running", lambda: False)():
                lines.append("[OK]   语音指令 KWS 已启动")
            elif self.config.get("voice_command_enabled") is False:
                lines.append("[--]   语音指令已在 config 中关闭")
            else:
                lines.append("[BAD]  语音指令未启动（检查麦克风/模型）")
        except Exception as e:
            lines.append(f"[BAD]  语音指令检查失败: {e}")

        # 语音助手
        assistant = self.config.get("voice_assistant") or "未配置"
        lines.append(f"[OK]   语音助手: {assistant}")

        lines.append("=" * 60)
        lines.append("")
        for line in lines:
            logging.info(line)

    def init_ui(self):
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        # 浮窗缩放：base 320×240 × scale；config.json 里 floating_window_scale 可调
        # 1.0 = 320×240（紧凑）/ 1.5 = 480×360（默认）/ 2.0 = 640×480（宽敞）
        try:
            scale = float(self.config.get("floating_window_scale") or 1.5)
        except (TypeError, ValueError):
            scale = 1.5
        scale = max(1.0, min(3.0, scale))
        self._ui_scale = scale

        def s(x):  # 像素缩放
            return int(round(x * scale))

        W, H = s(320), s(240)
        self.setFixedSize(W, H)

        self.video_label = QLabel(self)
        self.video_label.resize(W, H)
        self.video_label.setStyleSheet("background-color: black; border-radius: 10px;")

        self.mode_label = QLabel(self)
        # 标签右边缘不超过 btn_minimize 左边缘 - 间距
        btn_min_left = s(244)
        label_left = s(50)
        label_width = min(s(170), btn_min_left - label_left - s(6))
        self.mode_label.setGeometry(label_left, s(10), label_width, s(30))
        self.mode_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.mode_label.setStyleSheet(f"""
            QLabel {{
                color: white;
                background-color: rgba(0, 0, 0, 140);
                border-radius: 8px;
                font-size: {s(15)}px;
                font-weight: bold;
                padding: 1px 6px;
            }}
        """)

        self.hint_label = QLabel(self)
        # 提示条：贴底 + 字号下调
        self.hint_label.setGeometry(s(10), s(212), s(300), s(20))
        self.hint_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.hint_label.setStyleSheet(f"""
            QLabel {{
                color: white;
                background-color: rgba(0, 0, 0, 150);
                border-radius: 6px;
                font-size: {s(9)}px;
                padding: 1px 4px;
            }}
        """)

        # 语音指令状态指示器（可悬停看当前模式指令 / 点击看全部）
        self.voice_label = QLabel(self)
        self.voice_label.setGeometry(s(10), s(186), s(80), s(18))
        self.voice_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.voice_label.setStyleSheet(f"""
            QLabel {{
                color: #00ff88;
                background-color: rgba(0, 80, 40, 160);
                border-radius: 6px;
                font-size: {s(10)}px;
                padding: 1px 4px;
            }}
        """)
        self.voice_label.setCursor(Qt.CursorShape.PointingHandCursor)
        # 点击 → 切换指令面板（已开则关，未开则开）
        self.voice_label.mousePressEvent = lambda ev: self._toggle_voice_cheatsheet()
        self._voice_cheatsheet_dialog = None
        self._update_voice_label()

        self.btn_settings = QPushButton("⚙", self)
        self.btn_settings.setGeometry(s(10), s(10), s(30), s(30))
        self.btn_settings.setStyleSheet(f"""
            QPushButton {{
                background-color: rgba(255, 255, 255, 150);
                border-radius: {s(15)}px;
                font-size: {s(16)}px;
            }}
            QPushButton:hover {{
                background-color: rgba(255, 255, 255, 200);
            }}
        """)
        self.btn_settings.clicked.connect(self.open_settings)

        self.btn_minimize = QPushButton("─", self)
        self.btn_minimize.setGeometry(s(244), s(10), s(30), s(30))
        self.btn_minimize.setStyleSheet(f"""
            QPushButton {{
                background-color: rgba(255, 255, 255, 150);
                border-radius: {s(15)}px;
                font-size: {s(14)}px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: rgba(255, 255, 255, 200);
            }}
        """)
        self.btn_minimize.clicked.connect(self.showMinimized)

        self.btn_close = QPushButton("X", self)
        self.btn_close.setGeometry(s(280), s(10), s(30), s(30))
        self.btn_close.setStyleSheet(f"""
            QPushButton {{
                background-color: rgba(255, 0, 0, 150);
                color: white;
                border-radius: {s(15)}px;
                font-weight: bold;
                font-size: {s(14)}px;
            }}
            QPushButton:hover {{
                background-color: rgba(255, 0, 0, 200);
            }}
        """)
        self.btn_close.clicked.connect(self.close)

        self.drag_pos = None

        # 默认位置：屏幕左下角，留 10px 边距
        screen = QApplication.primaryScreen().geometry()
        self._default_x = screen.left() + 10
        self._default_y = screen.bottom() - self.height() - 10

    def _connect_toolbar(self):
        self.toolbar.color_changed.connect(self.overlay.set_pen_color)
        self.toolbar.pen_width_changed.connect(self._on_pen_width_changed)
        self.toolbar.undo_requested.connect(self.overlay.undo)
        self.toolbar.clear_requested.connect(self._on_toolbar_clear)
        self.toolbar.shape_correction_toggled.connect(self.overlay.set_shape_correction_enabled)
        self.overlay.undo_changed.connect(self.toolbar.set_undo_enabled)
        # 实时字幕写满屏幕时自动停止听写
        self.overlay.caption_full.connect(self._on_caption_full)

    def _on_pen_width_changed(self, width):
        self.overlay.set_pen_width(width)
        self._pending_pen_width = width
        if not hasattr(self, '_pen_width_timer'):
            self._pen_width_timer = QTimer()
            self._pen_width_timer.setSingleShot(True)
            self._pen_width_timer.timeout.connect(self._flush_pen_width)
        self._pen_width_timer.start(500)

    def _flush_pen_width(self):
        if hasattr(self, '_pending_pen_width'):
            self.config.set("pen_width", self._pending_pen_width)
            del self._pending_pen_width

    def _on_toolbar_clear(self):
        self.overlay.clear_canvas()
        try:
            winsound.PlaySound(
                "SystemExclamation", winsound.SND_ALIAS | winsound.SND_ASYNC
            )
        except RuntimeError:
            winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)

    def init_timer(self):
        # QTimer现在用于UI状态更新，而不是帧处理
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_ui_state)
        self.timer.start(100)  # 100ms更新一次UI状态

    def open_settings(self):
        dialog = SettingsDialog(self.config, self)
        if dialog.exec():
            self.apply_config()

    def switch_camera(self, new_index):
        """运行时切换摄像头：停推理→释放旧→启动新→重建推理；失败回滚旧的。

        在 SettingsDialog 保存时被调用。返回 True 表示新摄像头已上线，
        False 表示启动失败、已尝试恢复旧摄像头（旧摄像头若也启不来这就麻烦了
        但实际不太会发生：能停下来说明刚才一直在跑）。
        """
        try:
            old_index = self.camera.camera_index
        except Exception:
            old_index = self.config.get("camera_index") or 0
        if new_index == old_index:
            return True

        logging.info("切换摄像头: %d → %d", old_index, new_index)

        # 1. 停推理线程
        if hasattr(self, "inference_worker") and self.inference_worker is not None:
            try:
                self.inference_worker.stop()
            except Exception:
                logging.exception("停 InferenceWorker 时异常")

        # 2. 释放旧摄像头
        try:
            self.camera.release()
        except Exception:
            logging.exception("释放旧摄像头时异常")

        # 3. 尝试启动新摄像头
        new_cam = CameraService(
            camera_index=new_index,
            width=self.config.get("camera_width"),
            height=self.config.get("camera_height"),
            force_mjpeg=self.config.get("camera_force_mjpeg") is not False,
            min_fps=self.config.get("camera_min_fps") or 20,
        )
        try:
            new_cam.start()
        except Exception:
            logging.exception("新摄像头 %d 启动失败，回滚到旧摄像头 %d",
                              new_index, old_index)
            # 回滚
            try:
                self.camera = CameraService(
                    camera_index=old_index,
                    width=self.config.get("camera_width"),
                    height=self.config.get("camera_height"),
                    force_mjpeg=self.config.get("camera_force_mjpeg") is not False,
                    min_fps=self.config.get("camera_min_fps") or 20,
                )
                self.camera.start()
            except Exception:
                logging.exception("回滚旧摄像头 %d 也失败", old_index)
            self._restart_inference_worker()
            return False

        self.camera = new_cam
        self._restart_inference_worker()
        logging.info("摄像头已切换到 %d (%dx%d)",
                     new_index, new_cam.width or 0, new_cam.height or 0)
        return True

    def _restart_inference_worker(self):
        """用当前 self.camera / self.tracker 起一个新的 InferenceWorker，
        并把信号重连到本窗口的 slot。在 switch_camera 后调用。
        """
        new_worker = InferenceWorker(
            self.camera, self.tracker, max_fps=30,
            debug_overlay=bool(self.config.get("debug_overlay")),
        )
        new_worker.frame_ready.connect(self._on_frame_ready)
        new_worker.error_occurred.connect(self._on_inference_error)
        new_worker.fps_updated.connect(self._on_fps_updated)
        new_worker.start()
        self.inference_worker = new_worker

    def apply_config(self):
        self.recognizer.cooldown = self.config.get("cooldown")
        self.ppt.set_target_app(self.config.get("target_app"))
        self.mouse.set_sensitivity(self.config.get("mouse_sensitivity"))
        self.mouse.set_edge_acceleration(
            self.config.get("edge_acceleration_enabled"),
            self.config.get("edge_acceleration_strength"),
            y_canvas=self.config.get("edge_y_canvas_enabled"),
            y_dz_bottom=self.config.get("edge_y_canvas_deadzone_bottom"),
            y_dz_top=self.config.get("edge_y_canvas_deadzone_top"),
        )
        self.overlay.set_pen_width(self.config.get("pen_width"))
        self.voice_assistant.set_assistant(self.config.get("voice_assistant"))
        
        # 线程安全地更新tracker
        new_tracker = HandTracker(
            max_num_hands=2,
            min_detection_confidence=self.config.get("hand_detection_confidence") or 0.6,
            min_presence_confidence=self.config.get("hand_presence_confidence") or 0.5,
            min_tracking_confidence=self.config.get("hand_tracking_confidence") or 0.5,
            preferred_model_type=self.config.get("model_type"),
            dominant_hand=self.config.get("dominant_hand") or "Right",
        )
        self.tracker = new_tracker
        if hasattr(self, 'inference_worker'):
            self.inference_worker.update_tracker(new_tracker)
        
        new_mode = self.config.get("interaction_mode")
        if new_mode != self.mode_manager.current_mode_name:
            self._set_mode(new_mode)
        print(
            f"配置已更新: 模式 -> {new_mode} / 目标软件 -> {self.ppt.target_app}"
        )

    # ------------------------------------------------------------------
    # 模式系统
    # ------------------------------------------------------------------

    def _init_modes(self):
        self.modes = {
            "presentation": PresentationMode(
                self.config, self.recognizer, self.mouse,
                self.overlay, self.cursor_overlay, self.toolbar, self.ppt,
            ),
            "mouse": MouseMode(
                self.config, self.recognizer, self.mouse,
                self.overlay, self.cursor_overlay, self.toolbar, self.ppt,
            ),
            "draw": DrawMode(
                self.config, self.recognizer, self.mouse,
                self.overlay, self.cursor_overlay, self.toolbar, self.ppt,
            ),
        }
        self.mode_manager = ModeManager(self.modes, self.config, self.recognizer)

    def _set_mode(self, mode_name, sound=True):
        self.mode_manager.switch_to(mode_name)
        # 通知语音指令服务切换关键词集
        if hasattr(self, 'voice_command') and self.voice_command.is_running:
            self.voice_command.on_mode_changed(mode_name)
        if sound:
            try:
                winsound.PlaySound(
                    "SystemAsterisk", winsound.SND_ALIAS | winsound.SND_ASYNC
                )
            except RuntimeError:
                winsound.MessageBeep(winsound.MB_ICONASTERISK)
        self.status_text = f"已切换到{self._mode_name_zh()}"
        self.status_color = (0, 255, 255)
        self.status_timer = time.time()
        # 切模式后刷新 tooltip 显示新模式的指令
        self._refresh_voice_tooltip()

    def _mode_name_zh(self):
        mode = self.mode_manager.current_mode_name
        return {
            "presentation": "演示模式",
            "mouse": "鼠标模式",
            "draw": "板书模式",
        }.get(mode, "未知模式")

    def _mode_hint_zh(self):
        mode = self.mode_manager.current_mode_name
        if mode == "mouse":
            return "双拳切模式 | 中指移动 | 捏拇指左键 | 捏食指右键 | 剪刀手滚动"
        elif mode == "draw":
            return "双拳切模式 | 拇指并拢书写/分开停笔 | 张掌清屏"
        else:
            return "双拳切模式 | 剪刀手唤醒AI | 拇指向下挂断 | 并掌翻页 | 点赞切WPS"

    def _update_voice_label(self):
        """更新语音状态指示器"""
        if hasattr(self, 'voice_command') and self.voice_command.is_running:
            self.voice_label.setText("🎤 语音开")
            self.voice_label.setStyleSheet("""
                QLabel {
                    color: #00ff88;
                    background-color: rgba(0, 80, 40, 160);
                    border-radius: 6px;
                    font-size: 10px;
                    padding: 1px 4px;
                }
            """)
        else:
            self.voice_label.setText("语音关")
            self.voice_label.setStyleSheet("""
                QLabel {
                    color: #888;
                    background-color: rgba(60, 60, 60, 160);
                    border-radius: 6px;
                    font-size: 10px;
                    padding: 1px 4px;
                }
            """)
        # 同步 tooltip
        self._refresh_voice_tooltip()

    # ------------------------------------------------------------------
    # 语音指令提示（tooltip + 完整面板）
    # ------------------------------------------------------------------
    _VOICE_COMMANDS_BY_MODE = {
        "global": [
            ("召唤豆包", "呼出豆包 AI 助手"),
            ("最小化助手", "最小化本程序"),
            ("显示助手", "从最小化恢复本程序"),
        ],
        "presentation": [
            ("开始播放", "F5 开始放映"),
            ("结束播放", "Esc 退出放映"),
            ("下一页 / 上一页", "翻页"),
            ("板书模式 / 鼠标模式", "直接跳转"),
        ],
        "mouse": [
            ("点一下", "左键单击"),
            ("双击", "左键双击"),
            ("右键", "右键单击"),
            ("板书模式 / 演示模式", "直接跳转"),
        ],
        "draw": [
            ("开始板书", "开启语音听写到屏幕"),
            ("结束板书", "停止听写"),
            ("清屏", "清空画布"),
            ("演示模式 / 鼠标模式", "直接跳转"),
        ],
    }

    def _refresh_voice_tooltip(self):
        """悬停浮窗显示当前模式可用语音指令。"""
        if not hasattr(self, "voice_label") or self.voice_label is None:
            return
        # mode_manager 可能在 init_ui 早期还未创建，此时使用默认模式名
        mode = "presentation"
        if hasattr(self, "mode_manager") and self.mode_manager is not None:
            mode = getattr(self.mode_manager, "current_mode_name", None) or "presentation"
        mode_zh = {"presentation": "演示模式", "mouse": "鼠标模式", "draw": "板书模式"}.get(mode, mode)

        lines = [f"【{mode_zh}】"]
        for kw, desc in self._VOICE_COMMANDS_BY_MODE.get(mode, []):
            lines.append(f"  • {kw}    {desc}")
        lines.append("")
        lines.append("【全局】")
        for kw, desc in self._VOICE_COMMANDS_BY_MODE["global"]:
            lines.append(f"  • {kw}    {desc}")
        lines.append("")
        lines.append("点击查看完整面板")
        self.voice_label.setToolTip("\n".join(lines))

    def _toggle_voice_cheatsheet(self):
        """点击🎤标签的处理：开则关、关则开。"""
        dlg = self._voice_cheatsheet_dialog
        if dlg is not None and dlg.isVisible():
            dlg.close()
            return
        self._open_voice_cheatsheet()

    def _open_voice_cheatsheet(self):
        """弹出完整指令面板（所有模式 + 全局）。无系统标题栏，避免手势 LEFTDOWN
        落在标题栏触发 Win32 窗口拖拽模态循环导致主线程被堵。

        没有系统标题栏 → 没有 Windows 内置的拖拽：自己监听 dlg 的鼠标事件
        在面板空白处按下并拖动来移动整个面板，子控件（如 ✕ 按钮）会消费事件不受影响。
        """

        current_mode = getattr(self.mode_manager, "current_mode_name", None)

        dlg = QDialog(self)
        dlg.setWindowTitle("语音指令")
        dlg.setWindowFlags(
            Qt.WindowType.Dialog
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
        )
        # 让样式表的 border-radius 真正绘制（无边框窗口默认不画背景）
        dlg.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        dlg.setStyleSheet("""
            QDialog {
                background-color: #1e1e1e;
                border: 1px solid #444;
                border-radius: 6px;
            }
            QLabel { color: white; }
        """)

        body = QVBoxLayout(dlg)
        body.setContentsMargins(10, 6, 10, 10)
        body.setSpacing(2)

        # 顶部条：标题 + 右上角 ✕ 关闭。无边框对话框唯一的关闭入口
        top_row = QHBoxLayout()
        top_row.setSpacing(4)
        header = QLabel("🎤 语音指令")
        header.setStyleSheet("font-size: 13px; font-weight: bold; color: #00ff88;")
        top_row.addWidget(header)
        top_row.addStretch()
        top_close_btn = QPushButton("✕")
        top_close_btn.setFixedSize(26, 26)
        top_close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        top_close_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(220, 60, 60, 220);
                color: white;
                border: none;
                border-radius: 13px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: rgba(255, 90, 90, 240); }
            QPushButton:pressed { background-color: rgba(180, 40, 40, 240); }
        """)
        top_close_btn.clicked.connect(dlg.close)
        top_row.addWidget(top_close_btn)
        body.addLayout(top_row)

        sections = [
            ("global", "🌐 全局", "#aaaaaa"),
            ("presentation", "📺 演示", "#88ccff"),
            ("mouse", "🖱 鼠标", "#ffcc66"),
            ("draw", "✏ 板书", "#ff99cc"),
        ]

        for key, title, color in sections:
            is_current = (key == current_mode)
            title_text = title + ("  ← 当前" if is_current else "")
            section_title = QLabel(title_text)
            font_weight = "bold" if is_current else "normal"
            section_title.setStyleSheet(
                f"color: {color}; font-size: 11px; font-weight: {font_weight}; "
                f"padding-top: 3px;"
            )
            body.addWidget(section_title)

            for kw, desc in self._VOICE_COMMANDS_BY_MODE.get(key, []):
                row = QLabel(f"「{kw}」 {desc}")
                row.setStyleSheet("font-size: 10px; color: #dddddd; padding-left: 4px;")
                body.addWidget(row)

        tip = QLabel("💡 需开启麦克风+语音识别")
        tip.setStyleSheet("color: #888; font-size: 9px; padding-top: 4px;")
        body.addWidget(tip)

        dlg.setFixedWidth(260)
        dlg.adjustSize()

        # 自实现窗口拖动（FramelessWindowHint 没有系统标题栏，原本拖不动）。
        # 点击落在 QPushButton 等会被子控件 accept；落在 QLabel/空白处会冒泡到
        # QDialog.mousePressEvent，正常进入拖动流程。
        dlg._drag_offset = None

        def _press(ev):
            if ev.button() == Qt.MouseButton.LeftButton:
                dlg._drag_offset = (
                    ev.globalPosition().toPoint() - dlg.frameGeometry().topLeft()
                )
                ev.accept()

        def _move(ev):
            if (
                ev.buttons() == Qt.MouseButton.LeftButton
                and dlg._drag_offset is not None
            ):
                dlg.move(ev.globalPosition().toPoint() - dlg._drag_offset)
                ev.accept()

        def _release(ev):
            dlg._drag_offset = None
            ev.accept()

        dlg.mousePressEvent = _press
        dlg.mouseMoveEvent = _move
        dlg.mouseReleaseEvent = _release

        # 定位到浮窗右侧（右侧没空间放左侧），y 同时 clamp 顶部和底部，
        # 防止 dialog 被屏幕底部裁掉看不到关闭按钮
        screen = QApplication.primaryScreen().geometry()
        x = self.x() + self.width() + 10
        if x + dlg.width() > screen.right():
            x = self.x() - dlg.width() - 10
        if x < screen.left():
            x = screen.left() + 10
        y = self.y()
        y = min(y, screen.bottom() - dlg.height() - 10)
        y = max(y, screen.top() + 10)
        dlg.move(x, y)

        self._voice_cheatsheet_dialog = dlg
        dlg.show()

    def _on_voice_keyword_detected(self, keyword):
        """语音关键词检测回调（从 VoiceCommandService 检测线程调用）

        注意：虽然跨线程写入，但 Python GIL 保证简单属性赋值原子性。
        _voice_keyword_flash 和 _voice_keyword_time 在主线程的
        _process_frame_results 中读取，最坏情况是闪烁一个错误的
        关键词或时间略偏——不影响核心功能。
        如需严格线程安全，应改用 Qt 信号-槽机制。
        """
        self._voice_keyword_flash = keyword
        self._voice_keyword_time = time.time()

    def _on_frame_ready(self, frame, hands_landmarks, hands_gestures):
        """推理完成回调（在主线程中执行）"""
        try:
            self._process_frame_results(frame, hands_landmarks, hands_gestures)
        except Exception as e:
            logging.error("_on_frame_ready error: %s", e, exc_info=True)
            self.status_text = f"error: {e}"
            self.status_color = (255, 0, 0)
            self.status_timer = time.time()

    def _on_inference_error(self, error_msg):
        """推理错误回调"""
        logging.error("推理错误: %s", error_msg)
        self.status_text = f"推理错误: {error_msg}"
        self.status_color = (255, 0, 0)
        self.status_timer = time.time()

    def _on_fps_updated(self, fps):
        """FPS更新回调"""
        self._current_fps = fps

    def _process_frame_results(self, frame, hands_landmarks, hands_gestures):
        """处理推理结果（在主线程中执行）"""
        frame_h, frame_w = frame.shape[:2]

        # 切模式后 1 秒内忽略手势，避免放开双手瞬间被误判
        if time.time() - self.mode_manager.last_mode_switch_time < 1.0:
            hands_landmarks = []
            hands_gestures = []

        switched = self.mode_manager.maybe_switch_by_two_fists(hands_landmarks, frame_w)

        if switched:
            self._set_mode(self.mode_manager.current_mode_name)
            gesture = "MODE_SWITCH"
        else:
            result = self.mode_manager.handle(
                hands_landmarks, hands_gestures, frame_w, frame_h
            )
            gesture = result.gesture
            if result.status_text:
                self.status_text = result.status_text
                self.status_color = result.status_color
                self.status_timer = time.time()
            if result.action:
                self.execute_action(result.action)

        # 状态文本 1 秒后恢复默认
        if time.time() - self.status_timer > 1.0:
            self.status_text = "准备就绪" if hands_landmarks else "未检测到手"
            self.status_color = (0, 255, 0) if hands_landmarks else (0, 0, 255)

        if gesture == "COOLDOWN":
            cv2.putText(
                frame,
                "Cooldown...",
                (10, 220),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 165, 255),
                2,
            )

        # OpenCV -> PyQt
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(
            rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888
        ).copy()
        self.video_label.setPixmap(
            QPixmap.fromImage(qt_image).scaled(
                self.video_label.width(),
                self.video_label.height(),
                Qt.AspectRatioMode.KeepAspectRatio,
            )
        )
        self.mode_label.setText(self._mode_name_zh())
        self.hint_label.setText(self._mode_hint_zh())

        # 听写进行中时由 _set_voice_status 维护状态，跳过默认刷新避免抢占
        dictating = (
            hasattr(self, "voice_command")
            and self.voice_command is not None
            and self.voice_command.is_dictating
        )

        if not dictating:
            # 语音关键词闪烁显示（2秒后恢复）
            if hasattr(self, '_voice_keyword_flash') and hasattr(self, '_voice_keyword_time'):
                if time.time() - self._voice_keyword_time < 2.0:
                    self.voice_label.setText(f"🎤 {self._voice_keyword_flash}")
                    self.voice_label.setStyleSheet("""
                        QLabel {
                            color: #00ffff;
                            background-color: rgba(0, 60, 80, 200);
                            border-radius: 6px;
                            font-size: 10px;
                            padding: 1px 4px;
                        }
                    """)
                else:
                    self._update_voice_label()

    def _update_ui_state(self):
        """更新UI状态（由QTimer调用）"""
        # 这里可以添加其他需要定期更新的UI状态
        pass

    def _start_voice_dictation(self):
        """触发语音听写：持续录音，等待"结束板书"或超时后 ASR → 写到画布。"""
        if not hasattr(self, "voice_command") or not self.voice_command.is_running:
            logging.warning("语音服务未运行，无法听写")
            return

        # 只在板书模式才有意义
        current_mode = self.mode_manager.current_mode_name
        if current_mode != "draw":
            logging.info("非板书模式（%s）忽略听写请求", current_mode)
            return

        # 记录触发时的光标位置，避免回调时手已经离开导致定位漂移
        anchor_pos = None
        if self.overlay.cursor_pos is not None:
            anchor_pos = (self.overlay.cursor_pos.x(), self.overlay.cursor_pos.y())

        # 这三个回调会在 KWS 工作线程触发；不能直接改 UI，必须通过信号 marshal
        # 到主线程。signal 的 AutoConnection 会自动用 QueuedConnection。
        def on_status(phase, payload):
            self._dictation_status_signal.emit(phase, payload)

        def on_text(text):
            self._dictation_text_signal.emit(text, anchor_pos)

        def on_partial(text):
            self._dictation_partial_signal.emit(text or "")

        ok = self.voice_command.start_dictation(
            on_text=on_text,
            on_status=on_status,
            on_partial=on_partial,
        )
        if not ok:
            try:
                winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
            except Exception:
                pass
            self._set_voice_status("⚠️ 听写不可用（模型未安装？）")

    def _on_dictation_status(self, phase, payload):
        """主线程 slot：听写阶段状态变更（由 _dictation_status_signal 触发）。"""
        if phase == "started":
            self._set_voice_status('🎙️ 听写中（说"结束板书"停止）')
        elif phase == "tick":
            elapsed = payload if isinstance(payload, (int, float)) else 0.0
            self._set_voice_status(f"🎙️ 听写中... {elapsed:.0f}s")
        elif phase == "decoding":
            self._set_voice_status("🧠 识别中...")
        elif phase == "failed":
            reason = payload if isinstance(payload, str) else "no_text"
            self._set_voice_status(f"⚠️ 听写失败: {reason}")
        # phase == "done" 由 _on_dictation_text 处理（拿到 text 才能渲染）

    def _on_dictation_text(self, text, anchor_pos):
        """主线程 slot：听写文本就绪（由 _dictation_text_signal 触发）。"""
        self._render_dictation_text(text, anchor_pos)

    def _on_dictation_partial(self, text):
        """主线程 slot：partial ASR 结果到达，实时显示在 overlay 字幕区。"""
        if hasattr(self, "overlay") and self.overlay is not None:
            self.overlay.set_dictation_caption(text)

    def _on_caption_full(self):
        """实时字幕已写满屏幕，自动停止听写。"""
        if hasattr(self, "voice_command") and self.voice_command.is_dictating:
            logging.info("字幕已写满屏幕，自动停止听写")
            self.voice_command.stop_dictation()

    def _render_dictation_text(self, text, anchor_pos):
        """主线程：把识别结果写到画布。"""
        # 实时字幕先清掉，免得最终文字写到 canvas 后还叠着浮动字幕
        if hasattr(self, "overlay") and self.overlay is not None:
            self.overlay.clear_dictation_caption()
        if not text:
            self._set_voice_status("⚠️ 没听清，请再试一次")
            return
        x = y = None
        if anchor_pos is not None:
            x, y = anchor_pos
        try:
            self.overlay.draw_text(text, x=x, y=y)
        except Exception as e:
            logging.error("写文字到画布失败: %s", e, exc_info=True)
            self._set_voice_status("⚠️ 渲染失败")
            return
        self._set_voice_status(f"✍️ {text[:20]}")

    def _set_voice_status(self, text):
        """更新语音状态指示器（主线程调用）。"""
        if hasattr(self, "voice_label") and self.voice_label is not None:
            try:
                self.voice_label.setText(text)
            except Exception:
                pass

    def execute_action(self, action_name):
        if action_name == "next_slide":
            self.ppt.next_slide()
        elif action_name == "prev_slide":
            self.ppt.prev_slide()
        elif action_name == "start_presentation":
            self.ppt.start_presentation()
        elif action_name == "end_presentation":
            self.ppt.end_presentation()
        elif action_name == "switch_app":
            self.ppt.switch_app()
        elif action_name == "launch_voice_assistant":
            # "召唤豆包"：呼出豆包 AI 助手（外部程序）
            threading.Thread(target=self.voice_assistant.activate, daemon=True).start()
        elif action_name == "hang_up_voice_assistant":
            threading.Thread(target=self.voice_assistant.hang_up, daemon=True).start()
        # --- 语音指令专用 action ---
        elif action_name == "minimize_assistant":
            # "最小化助手"：最小化本程序
            self.showMinimized()
        elif action_name == "restore_assistant":
            # "显示助手"：把本程序从最小化恢复并置顶
            if self.isMinimized():
                self.showNormal()
            else:
                self.show()
            self.raise_()
            self.activateWindow()
        elif action_name == "left_click":
            self.mouse.left_click()
        elif action_name == "double_click":
            self.mouse.double_click()
        elif action_name == "right_click":
            self.mouse.right_click()
        elif action_name == "clear_canvas":
            self.overlay.clear_canvas()
            try:
                winsound.PlaySound(
                    "SystemExclamation", winsound.SND_ALIAS | winsound.SND_ASYNC
                )
            except RuntimeError:
                winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
        elif action_name == "start_dictation":
            self._start_voice_dictation()
        elif action_name == "stop_dictation":
            # "结束板书" 由 voice_command 内部处理，此分支仅做防御
            if hasattr(self, "voice_command"):
                self.voice_command.stop_dictation()
        elif action_name == "switch_to_draw":
            self._set_mode("draw")
        elif action_name == "switch_to_mouse":
            self._set_mode("mouse")
        elif action_name == "switch_to_presentation":
            self._set_mode("presentation")
        elif action_name == "toggle_shape_correction":
            if self.mode_manager.current_mode_name == "draw":
                enabled = self.overlay.toggle_shape_correction()
                self.toolbar.set_shape_correction(enabled)

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------

    # --- 拖动支持 ---

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drag_pos = event.globalPosition().toPoint()

    def mouseMoveEvent(self, event):
        if self.drag_pos is not None:
            delta = event.globalPosition().toPoint() - self.drag_pos
            self.move(self.x() + delta.x(), self.y() + delta.y())
            self.drag_pos = event.globalPosition().toPoint()

    def mouseReleaseEvent(self, event):
        self.drag_pos = None

    def moveEvent(self, event):
        """浮窗被拖动时，让已打开的指令面板按同样 delta 跟着走。

        moveEvent 在 Qt 内部移动后触发；event.pos()/oldPos() 给到屏幕坐标差，
        加给指令面板就行。指令面板若被用户单独拖到别处，自然偏移会变，但跟随
        逻辑只看 delta，跟随后两者保持新的相对位置——这就是用户期望。
        """
        super().moveEvent(event)
        dlg = getattr(self, "_voice_cheatsheet_dialog", None)
        if dlg is None or not dlg.isVisible():
            return
        try:
            delta = event.pos() - event.oldPos()
        except Exception:
            return
        if delta.x() == 0 and delta.y() == 0:
            return
        dlg.move(dlg.x() + delta.x(), dlg.y() + delta.y())

    def show(self):
        """首次显示时自动定位到屏幕左下角。"""
        if not self.isVisible():
            self.move(self._default_x, self._default_y)
        super().show()

    def keyPressEvent(self, event):
        """F1 切换调试覆盖层（开发/排查时用）。"""
        from PyQt6.QtCore import Qt
        if event.key() == Qt.Key.Key_F1:
            if hasattr(self, "inference_worker"):
                new_state = not self.inference_worker.debug_overlay
                self.inference_worker.set_debug_overlay(new_state)
            event.accept()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event):
        if self.mode_manager.current_mode:
            self.mode_manager.current_mode.on_exit()

        # 先断开信号，防止工作线程停止后仍有排队的信号访问已关闭的 widget
        if hasattr(self, 'inference_worker'):
            try:
                self.inference_worker.frame_ready.disconnect(self._on_frame_ready)
            except (TypeError, RuntimeError):
                pass  # 信号可能已断开或对象已销毁
            self.inference_worker.stop()

        # 停止语音指令服务
        if hasattr(self, 'voice_command'):
            self.voice_command.stop()

        # 再关闭 Qt 组件
        self.toolbar.close()
        self.overlay.close()
        self.cursor_overlay.close()
        self.camera.release()

        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    window = FloatingWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
