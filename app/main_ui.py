import logging
import os
import sys
import threading
import time
import winsound

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import cv2
from PyQt6.QtCore import Qt, QTimer
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
from services.camera import CameraService
from services.gesture_recognizer import GestureRecognizer
from services.hand_tracker import HandTracker
from services.inference_worker import InferenceWorker
from services.mouse_controller import MouseController
from services.ppt_controller import PptController
from services.voice_assistant import VoiceAssistantService
from services.voice_command import VoiceCommandService


class SettingsDialog(QDialog):
    def __init__(self, config_manager, parent=None):
        super().__init__(parent)
        self.config = config_manager
        self.setWindowTitle("设置")
        self.setMinimumWidth(300)
        self.init_ui()

    def init_ui(self):
        layout = QFormLayout()

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

    def save_settings(self):
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
    def __init__(self):
        super().__init__()
        self.config = ConfigManager()
        self.overlay = DrawingOverlay(self, pen_width=self.config.get("pen_width"))
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
        self.camera = CameraService(camera_index=self.config.get("camera_index"))
        self.camera.start()
        self.tracker = HandTracker(
            max_num_hands=2,
            min_detection_confidence=0.7,
            preferred_model_type=self.config.get("model_type"),
        )
        self.recognizer = GestureRecognizer(
            cooldown=self.config.get("cooldown"),
            swipe_threshold=self.config.get("swipe_threshold"),
        )
        self.ppt = PptController(target_app=self.config.get("target_app"))
        self.voice_assistant = VoiceAssistantService(
            assistant=self.config.get("voice_assistant")
        )
        self.voice_assistant.aircontrol_hwnd = int(self.winId())

        # 语音指令服务（KWS 离线关键词检测）
        self.voice_command = VoiceCommandService(self.config, action_callback=self.execute_action)
        self.voice_command.set_status_callback(self._on_voice_keyword_detected)
        if self.config.get("voice_command_enabled") is not False:
            try:
                self.voice_command.start()
            except Exception as e:
                logging.warning("语音指令服务启动失败: %s", e)

        # 启动推理工作线程
        self.inference_worker = InferenceWorker(self.camera, self.tracker, max_fps=30)
        self.inference_worker.frame_ready.connect(self._on_frame_ready)
        self.inference_worker.error_occurred.connect(self._on_inference_error)
        self.inference_worker.fps_updated.connect(self._on_fps_updated)
        self.inference_worker.start()

    def init_ui(self):
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedSize(320, 240)

        self.video_label = QLabel(self)
        self.video_label.resize(320, 240)
        self.video_label.setStyleSheet("background-color: black; border-radius: 10px;")

        self.mode_label = QLabel(self)
        self.mode_label.setGeometry(46, 12, 178, 42)
        self.mode_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.mode_label.setStyleSheet("""
            QLabel {
                color: white;
                background-color: rgba(0, 0, 0, 140);
                border-radius: 10px;
                font-size: 24px;
                font-weight: bold;
                padding: 2px 8px;
            }
        """)

        self.hint_label = QLabel(self)
        self.hint_label.setGeometry(10, 206, 300, 26)
        self.hint_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.hint_label.setStyleSheet("""
            QLabel {
                color: white;
                background-color: rgba(0, 0, 0, 150);
                border-radius: 8px;
                font-size: 11px;
                padding: 2px 6px;
            }
        """)

        # 语音指令状态指示器
        self.voice_label = QLabel(self)
        self.voice_label.setGeometry(10, 186, 80, 18)
        self.voice_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.voice_label.setStyleSheet("""
            QLabel {
                color: #00ff88;
                background-color: rgba(0, 80, 40, 160);
                border-radius: 6px;
                font-size: 10px;
                padding: 1px 4px;
            }
        """)
        self._update_voice_label()

        self.btn_settings = QPushButton("⚙", self)
        self.btn_settings.setGeometry(10, 10, 30, 30)
        self.btn_settings.setStyleSheet("""
            QPushButton {
                background-color: rgba(255, 255, 255, 150);
                border-radius: 15px;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: rgba(255, 255, 255, 200);
            }
        """)
        self.btn_settings.clicked.connect(self.open_settings)

        self.btn_minimize = QPushButton("─", self)
        self.btn_minimize.setGeometry(244, 10, 30, 30)
        self.btn_minimize.setStyleSheet("""
            QPushButton {
                background-color: rgba(255, 255, 255, 150);
                border-radius: 15px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: rgba(255, 255, 255, 200);
            }
        """)
        self.btn_minimize.clicked.connect(self.showMinimized)

        self.btn_close = QPushButton("X", self)
        self.btn_close.setGeometry(280, 10, 30, 30)
        self.btn_close.setStyleSheet("""
            QPushButton {
                background-color: rgba(255, 0, 0, 150);
                color: white;
                border-radius: 15px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: rgba(255, 0, 0, 200);
            }
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
            min_detection_confidence=0.7,
            preferred_model_type=self.config.get("model_type"),
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

    def _on_voice_keyword_detected(self, keyword):
        """语音关键词检测回调（从检测线程调用，需线程安全）"""
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

        switched = self.mode_manager.maybe_switch_by_two_fists(hands_landmarks)

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
            threading.Thread(target=self.voice_assistant.activate, daemon=True).start()
        elif action_name == "hang_up_voice_assistant":
            threading.Thread(target=self.voice_assistant.hang_up, daemon=True).start()
        # --- 语音指令专用 action ---
        elif action_name == "switch_mode":
            modes = list(self.modes.keys())
            current = self.mode_manager.current_mode_name
            idx = modes.index(current) if current in modes else 0
            next_mode = modes[(idx + 1) % len(modes)]
            self._set_mode(next_mode)
        elif action_name == "minimize_assistant":
            self.showMinimized()
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
        elif action_name == "dictate_to_screen":
            # TODO: 接入腾讯云在线 ASR
            logging.info("语音听写功能尚未接入在线 ASR")

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

    def show(self):
        """首次显示时自动定位到屏幕左下角。"""
        if not self.isVisible():
            self.move(self._default_x, self._default_y)
        super().show()

    def closeEvent(self, event):
        if self.mode_manager.current_mode:
            self.mode_manager.current_mode.on_exit()
        
        # 停止推理工作线程
        if hasattr(self, 'inference_worker'):
            self.inference_worker.stop()
        
        self.toolbar.close()
        self.overlay.close()
        self.cursor_overlay.close()
        self.camera.release()
        
        # 停止语音指令服务
        if hasattr(self, 'voice_command'):
            self.voice_command.stop()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    window = FloatingWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
