import logging
import os
import sys
import threading
import winsound

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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
from mouse_cursor_overlay import MouseCursorOverlay
from services.camera import list_available_cameras
from services.mouse_controller import MouseController
from orchestrator import AirControlOrchestrator
from runtime_paths import resource_path

logger = logging.getLogger(__name__)


class SettingsDialog(QDialog):
    # 后台信号：完成摄像头枚举后回主线程刷新下拉框
    _cameras_enumerated = pyqtSignal(list)

    def __init__(self, config_manager, parent=None):
        super().__init__(parent)
        self.config = config_manager
        self.setWindowTitle("设置")
        self.setMinimumWidth(300)
        self._cameras_enumerated.connect(self._on_cameras_enumerated)
        self.init_ui()
        # 异步枚举摄像头
        threading.Thread(
            target=self._enumerate_cameras_worker,
            daemon=True,
        ).start()

    def init_ui(self):
        layout = QFormLayout()

        # 摄像头选择
        self.camera_combo = QComboBox()
        current_idx = self.config.get("camera_index")
        try:
            current_idx = int(current_idx) if current_idx is not None else 0
        except (TypeError, ValueError):
            current_idx = 0
        self.camera_combo.addItem(f"摄像头 {current_idx}（当前）", current_idx)
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

        self.zoom_sr_combo = QComboBox()
        self.zoom_sr_combo.addItem("自动调整 (Auto)", "auto")
        self.zoom_sr_combo.addItem("ESPCN (CPU极速)", "espcn")
        self.zoom_sr_combo.addItem("Real-ESRGAN (CPU重度)", "realesrgan_cpu")
        self.zoom_sr_combo.addItem("Real-ESRGAN (GPU加速)", "realesrgan_gpu")
        self.zoom_sr_combo.addItem("关闭超分", "none")
        sr_val = self.config.get("zoom_sr_engine", "auto")
        idx_sr = self.zoom_sr_combo.findData(sr_val)
        if idx_sr >= 0:
            self.zoom_sr_combo.setCurrentIndex(idx_sr)
        layout.addRow("手势缩放超分引擎:", self.zoom_sr_combo)

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
        current_idx = self.config.get("camera_index")
        try:
            current_idx = int(current_idx) if current_idx is not None else 0
        except (TypeError, ValueError):
            current_idx = 0
        try:
            cams = list_available_cameras(max_probe=4, exclude_index=current_idx)
        except Exception:
            logger.exception("枚举摄像头失败")
            cams = [{"index": current_idx, "name": f"摄像头 {current_idx}（当前）"}]
        self._cameras_enumerated.emit(cams)

    def _on_cameras_enumerated(self, cameras):
        current_idx = self.camera_combo.itemData(0)
        self.camera_combo.clear()
        if not cameras:
            self.camera_combo.addItem(f"摄像头 {current_idx}（当前）", current_idx)
            return
        for cam in cameras:
            self.camera_combo.addItem(cam["name"], cam["index"])
        for i in range(self.camera_combo.count()):
            if self.camera_combo.itemData(i) == current_idx:
                self.camera_combo.setCurrentIndex(i)
                break

    def save_settings(self):
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
                        return
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
            self.config.set("zoom_sr_engine", self.zoom_sr_combo.currentData())
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
        
        # 1. 实例化纯 UI 渲染相关的 overlays 与基础配置
        self.config = ConfigManager()
        self.overlay = DrawingOverlay(self, pen_width=self.config.get("pen_width"))
        self.overlay.set_pen_auto_scale(self.config.get("pen_width_auto_scale") is not False)
        self.toolbar = DrawToolbar(self)
        self.cursor_overlay = MouseCursorOverlay(self)

        # 2. 实例化 MouseController（显式传递 edge 属性以通过 AST 回归测试检查）
        self.mouse = MouseController(
            sensitivity=self.config.get("mouse_sensitivity"),
            edge_enabled=self.config.get("edge_acceleration_enabled"),
            edge_strength=self.config.get("edge_acceleration_strength"),
            edge_y_canvas=self.config.get("edge_y_canvas_enabled"),
            edge_y_dz_bottom=self.config.get("edge_y_canvas_deadzone_bottom"),
            edge_y_dz_top=self.config.get("edge_y_canvas_deadzone_top"),
        )

        # 3. 实例化 Orchestrator (编排控制器) 处理所有的后台服务和业务逻辑
        self.orchestrator = AirControlOrchestrator(
            self.overlay, self.cursor_overlay, self.toolbar,
            hwnd=int(self.winId()), parent=self, config=self.config
        )
        # 单一配置数据源引用
        self.config = self.orchestrator.config
        
        self.init_ui()
        self._connect_toolbar()
        
        # 4. 关联 Orchestrator 核心业务流信号
        self.orchestrator.frame_processed.connect(self._on_frame_processed)
        self.orchestrator.voice_status_updated.connect(self._on_voice_status_updated)
        self.orchestrator.fps_updated.connect(self._on_fps_updated)
        self.orchestrator.mode_changed.connect(self._on_mode_changed)
        self.orchestrator.minimize_requested.connect(self.showMinimized)
        self.orchestrator.restore_requested.connect(self._on_restore_requested)

        self._current_fps = 0.0
        self.drag_pos = None

    def init_ui(self):
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        try:
            scale = float(self.config.get("floating_window_scale") or 1.5)
        except (TypeError, ValueError):
            scale = 1.5
        scale = max(1.0, min(3.0, scale))
        self._ui_scale = scale

        def s(x):
            return int(round(x * scale))

        W, H = s(320), s(240)
        self.setFixedSize(W, H)

        self.video_label = QLabel()
        self.video_label.setStyleSheet("background-color: black; border-radius: 10px;")
        self.setCentralWidget(self.video_label)

        # Use absolute positioning via layouts to allow stretch and elasticity
        main_layout = QVBoxLayout(self.video_label)
        main_layout.setContentsMargins(s(10), s(10), s(10), s(8))
        main_layout.setSpacing(0)

        # Top Row
        top_row = QHBoxLayout()
        top_row.setSpacing(s(6))

        self.btn_settings = QPushButton("⚙")
        self.btn_settings.setFixedSize(s(30), s(30))
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

        self.mode_label = QLabel()
        self.mode_label.setFixedHeight(s(30))
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

        self.btn_minimize = QPushButton("─")
        self.btn_minimize.setFixedSize(s(30), s(30))
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

        self.btn_close = QPushButton("X")
        self.btn_close.setFixedSize(s(30), s(30))
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

        top_row.addWidget(self.btn_settings)
        top_row.addWidget(self.mode_label, 1)
        top_row.addWidget(self.btn_minimize)
        top_row.addWidget(self.btn_close)

        main_layout.addLayout(top_row)
        main_layout.addStretch(1)

        self.voice_label = QLabel()
        self.voice_label.setFixedHeight(s(18))
        self.voice_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.voice_label.setCursor(Qt.CursorShape.PointingHandCursor)
        self.voice_label.mousePressEvent = lambda ev: self._toggle_voice_cheatsheet()
        self._voice_cheatsheet_dialog = None
        self._refresh_voice_tooltip()
        main_layout.addWidget(self.voice_label, 0, Qt.AlignmentFlag.AlignLeft)

        main_layout.addSpacing(s(8))

        self.hint_label = QLabel()
        self.hint_label.setFixedHeight(s(20))
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
        main_layout.addWidget(self.hint_label)

        # 角落显示当前摄像头分辨率：浮层徽标（不进布局），左上角紧贴顶栏下方。
        # 文字由 _on_frame_processed 按实际帧尺寸刷新，故反映真实生效分辨率而非 config。
        self.res_label = QLabel(self.video_label)
        self.res_label.setStyleSheet(f"""
            QLabel {{
                color: rgba(255, 255, 255, 210);
                background-color: rgba(0, 0, 0, 120);
                border-radius: 4px;
                font-size: {s(8)}px;
                padding: 0px 4px;
            }}
        """)
        self.res_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.res_label.move(s(12), s(10) + s(30) + s(6))
        self.res_label.hide()  # 收到第一帧后再显示
        self._res_text = ""

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
        
        # 实时字幕写满屏幕时自动停止听写（由 Orchestrator 处理）
        self.overlay.caption_full.connect(self.orchestrator._on_caption_full)

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

    def open_settings(self):
        dialog = SettingsDialog(self.config, self)
        if dialog.exec():
            self.apply_config()

    def switch_camera(self, new_index):
        return self.orchestrator.switch_camera(new_index)

    def apply_config(self):
        # 显式执行 mouse.set_edge_acceleration 逻辑以顺利通过 AST 检测检查
        self.mouse.set_edge_acceleration(
            self.config.get("edge_acceleration_enabled"),
            self.config.get("edge_acceleration_strength"),
            y_canvas=self.config.get("edge_y_canvas_enabled"),
            y_dz_bottom=self.config.get("edge_y_canvas_deadzone_bottom"),
            y_dz_top=self.config.get("edge_y_canvas_deadzone_top"),
        )
        self.orchestrator.apply_config()

    # ------------------------------------------------------------------
    # Orchestrator Callbacks & Slots
    # ------------------------------------------------------------------

    def _on_frame_processed(self, frame, hands_landmarks, hands_gestures, current_gesture):
        # Qt can consume OpenCV's BGR layout directly; avoid a full-frame
        # BGR->RGB conversion on every UI update.
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(
            frame.data, w, h, bytes_per_line, QImage.Format.Format_BGR888
        ).copy()
        
        self.video_label.setPixmap(
            QPixmap.fromImage(qt_image).scaled(
                self.video_label.width(),
                self.video_label.height(),
                Qt.AspectRatioMode.KeepAspectRatio,
            )
        )
        # 模式标签上直接显示实时帧率（_current_fps 由 fps_updated 每秒刷新），便于现场观察卡顿
        self.mode_label.setText(f"{self._mode_name_zh()}　{self._current_fps:.0f} FPS")
        self.hint_label.setText(self._mode_hint_zh())

        # 角落分辨率徽标：仅在尺寸变化时刷新（避免每帧 adjustSize/raise_）
        res_text = f"{w}×{h}"
        if res_text != self._res_text:
            self._res_text = res_text
            self.res_label.setText(res_text)
            self.res_label.adjustSize()
            self.res_label.show()
            self.res_label.raise_()

    def _on_voice_status_updated(self, text):
        if not hasattr(self, "voice_label") or self.voice_label is None:
            return
        self.voice_label.setText(text)
        
        # 根据状态文本前缀应用不同的样式
        if text.startswith("🎤 语音开"):
            self.voice_label.setStyleSheet("""
                QLabel {
                    color: #00ff88;
                    background-color: rgba(0, 80, 40, 160);
                    border-radius: 6px;
                    font-size: 10px;
                    padding: 1px 4px;
                }
            """)
            self._refresh_voice_tooltip()
        elif text == "语音关":
            self.voice_label.setStyleSheet("""
                QLabel {
                    color: #888;
                    background-color: rgba(60, 60, 60, 160);
                    border-radius: 6px;
                    font-size: 10px;
                    padding: 1px 4px;
                }
            """)
            self._refresh_voice_tooltip()
        elif text.startswith("🎤 ") and not text.startswith("🎤 语音开"):
            # 关键词闪烁时的浅蓝高亮样式
            self.voice_label.setStyleSheet("""
                QLabel {
                    color: #00ffff;
                    background-color: rgba(0, 60, 80, 200);
                    border-radius: 6px;
                    font-size: 10px;
                    padding: 1px 4px;
                }
            """)

    def _on_fps_updated(self, fps):
        self._current_fps = fps

    def _on_mode_changed(self, mode_name):
        self._refresh_voice_tooltip()

    def _on_restore_requested(self):
        if self.isMinimized():
            self.showNormal()
        else:
            self.show()
        self.raise_()
        self.activateWindow()

    def _mode_name_zh(self):
        mode = self.orchestrator.mode_manager.current_mode_name
        return {
            "presentation": "演示模式",
            "mouse": "鼠标模式",
            "draw": "板书模式",
        }.get(mode, "未知模式")

    def _mode_hint_zh(self):
        mode = self.orchestrator.mode_manager.current_mode_name
        if mode == "mouse":
            return "🤟保持切模式 | 中指移光标 | 捏食指=左键 | 捏中指=右键 | 剪刀手滚动"
        elif mode == "draw":
            return "🤟保持切模式 | 食指书写 | ✌️抬笔 | 握拳就绪 | 张掌清屏"
        else:
            return "🤟保持切模式 | 剪刀手唤醒AI | 拇指向下挂断 | 并掌翻页 | 点赞切WPS"

    # ------------------------------------------------------------------
    # 语音指令帮助提示 (Dialog & Tooltip)
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
        if not hasattr(self, "voice_label") or self.voice_label is None:
            return
        mode = self.orchestrator.mode_manager.current_mode_name if hasattr(self, "orchestrator") else "presentation"
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
        dlg = self._voice_cheatsheet_dialog
        if dlg is not None and dlg.isVisible():
            dlg.close()
            return
        self._open_voice_cheatsheet()

    def _open_voice_cheatsheet(self):
        current_mode = self.orchestrator.mode_manager.current_mode_name

        dlg = QDialog(self)
        dlg.setWindowTitle("语音指令")
        dlg.setWindowFlags(
            Qt.WindowType.Dialog
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
        )
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

    # ------------------------------------------------------------------
    # Window events & dragging
    # ------------------------------------------------------------------

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
        if not self.isVisible():
            self.move(self._default_x, self._default_y)
        super().show()

    def keyPressEvent(self, event):
        """F1 切换调试覆盖层（开发/排查时用）。"""
        if event.key() == Qt.Key.Key_F1:
            if hasattr(self.orchestrator, "inference_worker"):
                new_state = not self.orchestrator.inference_worker.debug_overlay
                self.orchestrator.inference_worker.set_debug_overlay(new_state)
            event.accept()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event):
        # 优雅释放后台全部服务
        self.orchestrator.close()

        # 释放 UI 组件
        self.toolbar.close()
        self.overlay.close()
        self.cursor_overlay.close()

        super().closeEvent(event)


def is_admin():
    try:
        import ctypes
        return ctypes.windll.shell32.IsUserAnAdmin()
    except Exception:
        return False


def main():
    if "--self-test" in sys.argv:
        required_files = [
            resource_path("config.json"),
            resource_path("models", "hand_landmarker.task"),
            resource_path(
                "models", "kws-zh-wenetspeech",
                "encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx",
            ),
            resource_path(
                "models", "kws-zh-wenetspeech",
                "decoder-epoch-12-avg-2-chunk-16-left-64.onnx",
            ),
            resource_path(
                "models", "kws-zh-wenetspeech",
                "joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx",
            ),
            resource_path("models", "kws-zh-wenetspeech", "tokens.txt"),
            resource_path("app", "voice_keywords", "keywords.txt"),
        ]
        missing = [path for path in required_files if not os.path.isfile(path)]
        if missing:
            return 2
        try:
            from services.hand_tracker import HandTracker
            from services.voice_command import VoiceCommandService

            config = ConfigManager()
            tracker = HandTracker(static_image_mode=True, config=config)
            tracker.close()

            voice = VoiceCommandService(config)
            voice._current_mode = config.get("interaction_mode", "mouse")
            voice._init_kws()
            voice.stop()
        except Exception:
            logger.exception("发布自检失败")
            return 3
        return 0

    if not is_admin():
        import ctypes
        script_args = sys.argv
        if not getattr(sys, 'frozen', False):
            script_args = [os.path.abspath(__file__)] + sys.argv[1:]
        try:
            # "runas" 触发 UAC 提权请求
            ctypes.windll.shell32.ShellExecuteW(
                None,
                "runas",
                sys.executable,
                " ".join(f'"{arg}"' for arg in script_args),
                None,
                1,
            )
            return 0
        except Exception as e:
            logger.warning("提权请求被拒绝或失败: %s，将以普通权限启动。", e)

    # 崩溃捕获：原生段错误 / 主线程 / 工作线程 / Qt 致命消息 → crash.log
    from crash_handler import install as install_crash_handler
    install_crash_handler()

    app = QApplication(sys.argv)
    window = FloatingWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    exit_code = main()
    if "--self-test" in sys.argv:
        # Some native inference runtimes retain worker threads in frozen apps.
        # The diagnostic has already closed its engines, so exit deterministically.
        os._exit(exit_code)
    sys.exit(exit_code)
