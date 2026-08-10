import logging
import os
import subprocess
import sys
import threading
import time
import winsound
from collections import deque

import cv2

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config_manager import ConfigManager, ConfigSaveError
from draw_toolbar import DrawToolbar
from drawing_overlay import DrawingOverlay
from modes import MODE_NAME_ZH, MODE_NAMES
from mouse_cursor_overlay import MouseCursorOverlay
from orchestrator import AirControlOrchestrator
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
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from runtime_paths import resource_path
from services.camera import list_available_cameras, probe_max_resolution
from services.mouse_controller import MouseController
from version import __version__

logger = logging.getLogger(__name__)


def _make_preview_pixmap(frame, target_width, target_height):
    """Create a display-sized pixmap without copying the full source frame."""
    source_height, source_width, channels = frame.shape
    target_width = max(1, int(target_width))
    target_height = max(1, int(target_height))
    scale = min(
        target_width / max(source_width, 1),
        target_height / max(source_height, 1),
    )
    preview_width = max(1, int(round(source_width * scale)))
    preview_height = max(1, int(round(source_height * scale)))
    if preview_width != source_width or preview_height != source_height:
        interpolation = (
            cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        )
        preview = cv2.resize(
            frame,
            (preview_width, preview_height),
            interpolation=interpolation,
        )
    else:
        preview = frame
    bytes_per_line = channels * preview_width
    image = QImage(
        preview.data,
        preview_width,
        preview_height,
        bytes_per_line,
        QImage.Format.Format_BGR888,
    )
    # QPixmap owns the converted pixels after this call, so the temporary
    # NumPy preview can be released without a full-resolution QImage.copy().
    return QPixmap.fromImage(image)


def camera_config_values(camera_data):
    """Return only complete camera fields safe to persist from a combo item."""
    if not isinstance(camera_data, dict):
        return {}
    camera_index = camera_data.get("index")
    if not isinstance(camera_index, int) or camera_index < 0:
        return {}
    values = {"camera_index": camera_index}
    width = camera_data.get("width")
    height = camera_data.get("height")
    if width is not None and height is not None:
        values.update(camera_width=width, camera_height=height)
    return values


class SettingsDialog(QDialog):
    # 后台信号：完成摄像头枚举后回主线程刷新下拉框
    _cameras_enumerated = pyqtSignal(list)
    _camera_probed = pyqtSignal(int, object, str)

    def __init__(self, config_manager, parent=None):
        super().__init__(parent)
        self.config = config_manager
        self.setWindowTitle("设置")
        self.setMinimumWidth(300)
        self._cameras_enumerated.connect(self._on_cameras_enumerated)
        self._camera_probed.connect(self._on_camera_probed)
        self._camera_probe_threads = {}
        self._camera_probe_lock = threading.Lock()
        self._camera_worker_cancel = threading.Event()
        self.init_ui()
        self.camera_combo.currentIndexChanged.connect(
            self._on_camera_selection_changed
        )
        # 异步枚举摄像头
        self._camera_enumeration_thread = self._start_camera_worker(
            self._enumerate_cameras_worker,
            "CameraEnumerationWorker",
        )

    def _start_camera_worker(self, target, name, args=()):
        """Start a cancellable camera helper and register it with the app owner."""
        parent = self.parent()
        start_owned = getattr(
            getattr(parent, "orchestrator", None),
            "_start_background_thread",
            None,
        )
        if callable(start_owned):
            return start_owned(target, name, args=args)
        thread = threading.Thread(
            target=target,
            args=args,
            name=name,
            daemon=True,
        )
        thread.start()
        return thread

    def done(self, result):
        # Native camera probes cannot be force-cancelled safely. Suppress late
        # Qt signals; the application owner tracks and joins them on shutdown.
        self._camera_worker_cancel.set()
        super().done(result)

    def init_ui(self):
        # ---- 创建所有控件（属性名保持不变以兼容 AST 测试与 save_settings） ----

        # 摄像头选择
        self.camera_combo = QComboBox()
        current_idx = self.config.get("camera_index")
        try:
            current_idx = int(current_idx) if current_idx is not None else 0
        except (TypeError, ValueError):
            current_idx = 0
        self.camera_combo.addItem(
            f"摄像头 {current_idx}（当前）",
            {"index": current_idx, "current": True},
        )
        self.camera_combo.addItem("正在检测其它摄像头…", -1)
        self.camera_combo.model().item(1).setEnabled(False)

        self.app_combo = QComboBox()
        self.app_combo.addItems(["PowerPoint", "WPS"])
        self.app_combo.setCurrentText(self.config.get("target_app"))

        self.model_combo = QComboBox()
        self.model_combo.addItems(["Lite", "Heavy"])
        self.model_combo.setCurrentText(self.config.get("model_type"))

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(list(MODE_NAMES))
        self.mode_combo.setCurrentText(self.config.get("interaction_mode"))

        self.profile_combo = QComboBox()
        self.profile_combo.addItem("稳定优先", "stable")
        self.profile_combo.addItem("平衡", "balanced")
        self.profile_combo.addItem("远距增强", "long_range")
        idx_profile = self.profile_combo.findData(
            self.config.get("stability_profile", "stable")
        )
        if idx_profile >= 0:
            self.profile_combo.setCurrentIndex(idx_profile)
        self.profile_combo.currentIndexChanged.connect(self._on_profile_changed)
        self.profile_combo.setToolTip("切换稳定性与远距离增强的默认策略")

        self.cd_spin = QSpinBox()
        self.cd_spin.setRange(500, 3000)
        self.cd_spin.setSingleStep(100)
        self.cd_spin.setValue(int(self.config.get("cooldown") * 1000))
        self.cd_spin.setSuffix(" ms")
        self.cd_spin.setToolTip("连续手势之间的最小间隔时间，避免一个动作被重复触发")

        # 手势映射
        gesture_actions = [
            "next_slide", "prev_slide", "start_presentation",
            "end_presentation", "switch_app",
            "launch_voice_assistant", "hang_up_voice_assistant", "none",
        ]

        self.right_combo = QComboBox()
        self.right_combo.addItems(gesture_actions)
        self.right_combo.setCurrentText(self.config.get_mapping("SWIPE_RIGHT"))

        self.left_combo = QComboBox()
        self.left_combo.addItems(gesture_actions)
        self.left_combo.setCurrentText(self.config.get_mapping("SWIPE_LEFT"))

        self.up_combo = QComboBox()
        self.up_combo.addItems(gesture_actions)
        self.up_combo.setCurrentText(self.config.get_mapping("SWIPE_UP"))

        self.down_combo = QComboBox()
        self.down_combo.addItems(gesture_actions)
        self.down_combo.setCurrentText(self.config.get_mapping("SWIPE_DOWN"))

        self.fist_combo = QComboBox()
        self.fist_combo.addItems(gesture_actions)
        self.fist_combo.setCurrentText(self.config.get_mapping("FIST"))

        self.thumb_combo = QComboBox()
        self.thumb_combo.addItems(gesture_actions)
        self.thumb_combo.setCurrentText(self.config.get_mapping("THUMB_UP"))

        self.scissor_combo = QComboBox()
        self.scissor_combo.addItems(gesture_actions)
        self.scissor_combo.setCurrentText(self.config.get_mapping("SCISSOR"))

        self.thumb_down_combo = QComboBox()
        self.thumb_down_combo.addItems(gesture_actions)
        self.thumb_down_combo.setCurrentText(self.config.get_mapping("THUMB_DOWN"))

        # 鼠标
        self.sensitivity_spin = QSpinBox()
        self.sensitivity_spin.setRange(1, 200)
        self.sensitivity_spin.setValue(int(self.config.get("mouse_sensitivity")))
        self.sensitivity_spin.setSuffix(" %")
        self.sensitivity_spin.setToolTip("鼠标模式下光标跟踪灵敏度，越高越跟手但越容易抖")

        self.edge_check = QCheckBox("边缘加速")
        self.edge_check.setChecked(bool(self.config.get("edge_acceleration_enabled")))
        self.edge_check.stateChanged.connect(self._on_edge_toggled)
        self.edge_check.setToolTip("光标靠近屏幕边缘时自动加速，便于访问任务栏和角落")

        self.edge_strength_spin = QSpinBox()
        self.edge_strength_spin.setRange(0, 100)
        self.edge_strength_spin.setValue(int(self.config.get("edge_acceleration_strength")))
        self.edge_strength_spin.setSuffix(" %")
        self.edge_strength_spin.setEnabled(self.edge_check.isChecked())
        self.edge_strength_spin.setToolTip("边缘加速的强度，越大加速越明显")

        self.y_canvas_check = QCheckBox("Y 轴虚拟画布（推荐用于任务栏）")
        self.y_canvas_check.setChecked(bool(self.config.get("edge_y_canvas_enabled")))
        self.y_canvas_check.setToolTip("在屏幕上下边缘扩展虚拟移动区域，让光标能到达任务栏")

        self.y_dz_bottom_spin = QSpinBox()
        self.y_dz_bottom_spin.setRange(0, 30)
        self.y_dz_bottom_spin.setValue(int(self.config.get("edge_y_canvas_deadzone_bottom")))
        self.y_dz_bottom_spin.setSuffix(" %")
        self.y_dz_bottom_spin.setToolTip("底部不响应虚拟画布的区域占比，避免误触任务栏")

        self.y_dz_top_spin = QSpinBox()
        self.y_dz_top_spin.setRange(0, 20)
        self.y_dz_top_spin.setValue(int(self.config.get("edge_y_canvas_deadzone_top")))
        self.y_dz_top_spin.setSuffix(" %")
        self.y_dz_top_spin.setToolTip("顶部不响应虚拟画布的区域占比")

        # 画笔
        self.pen_spin = QSpinBox()
        self.pen_spin.setRange(1, 100)
        self.pen_spin.setValue(int(self.config.get("pen_width")))
        self.pen_spin.setToolTip("板书模式下的笔触宽度（像素），可配合笔粗距离自适应使用")

        # 语音
        self.voice_combo = QComboBox()
        self.voice_combo.addItem("豆包", "doubao")
        self.voice_combo.addItem("通义千问", "qianwen")
        idx = self.voice_combo.findData(self.config.get("voice_assistant"))
        if idx >= 0:
            self.voice_combo.setCurrentIndex(idx)

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
        self.zoom_sr_combo.setToolTip("远距离手势识别的超分辨率引擎，Auto 自动按距离切换")

        # ---- 用 QTabWidget 分组 ----
        tabs = QTabWidget()

        # Tab 1: 基础
        tab_basic = QFormLayout()
        tab_basic.addRow("摄像头:", self.camera_combo)
        tab_basic.addRow("控制目标软件:", self.app_combo)
        tab_basic.addRow("手势模型精度:", self.model_combo)
        tab_basic.addRow("交互模式:", self.mode_combo)
        tab_basic.addRow("体验档位:", self.profile_combo)
        w_basic = QWidget()
        w_basic.setLayout(tab_basic)
        tabs.addTab(w_basic, "基础")

        # Tab 2: 手势
        tab_gesture = QFormLayout()
        tab_gesture.addRow("手势防抖(冷却):", self.cd_spin)
        tab_gesture.addRow("右挥动作映射:", self.right_combo)
        tab_gesture.addRow("左挥动作映射:", self.left_combo)
        tab_gesture.addRow("上挥动作映射:", self.up_combo)
        tab_gesture.addRow("下挥动作映射:", self.down_combo)
        tab_gesture.addRow("握拳动作映射:", self.fist_combo)
        tab_gesture.addRow("点赞动作映射:", self.thumb_combo)
        tab_gesture.addRow("剪刀手动作映射:", self.scissor_combo)
        tab_gesture.addRow("拇指向下动作映射:", self.thumb_down_combo)
        w_gesture = QWidget()
        w_gesture.setLayout(tab_gesture)
        tabs.addTab(w_gesture, "手势")

        # Tab 3: 鼠标
        tab_mouse = QFormLayout()
        tab_mouse.addRow("鼠标灵敏度:", self.sensitivity_spin)
        tab_mouse.addRow(self.edge_check)
        tab_mouse.addRow("边缘加速强度:", self.edge_strength_spin)
        tab_mouse.addRow(self.y_canvas_check)
        tab_mouse.addRow("Y 轴底部死区:", self.y_dz_bottom_spin)
        tab_mouse.addRow("Y 轴顶部死区:", self.y_dz_top_spin)
        w_mouse = QWidget()
        w_mouse.setLayout(tab_mouse)
        tabs.addTab(w_mouse, "鼠标")

        # Tab 4: 画笔
        tab_pen = QFormLayout()
        tab_pen.addRow("画笔粗细:", self.pen_spin)
        w_pen = QWidget()
        w_pen.setLayout(tab_pen)
        tabs.addTab(w_pen, "画笔")

        # Tab 5: 语音
        tab_voice = QFormLayout()
        tab_voice.addRow("语音助手:", self.voice_combo)
        tab_voice.addRow("手势缩放超分引擎:", self.zoom_sr_combo)
        w_voice = QWidget()
        w_voice.setLayout(tab_voice)
        tabs.addTab(w_voice, "语音")

        # ---- 底部按钮 ----
        btn_layout = QHBoxLayout()
        self.save_btn = QPushButton("保存")
        self.save_btn.clicked.connect(self.save_settings)
        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(self.reject)
        reset_btn = QPushButton("恢复默认")
        reset_btn.setToolTip("将所有设置恢复为默认值（需点击保存生效）")
        reset_btn.clicked.connect(self._reset_defaults)
        btn_layout.addWidget(reset_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(self.save_btn)
        btn_layout.addWidget(cancel_btn)

        main_layout = QVBoxLayout()
        main_layout.addWidget(tabs)
        main_layout.addLayout(btn_layout)
        self.setLayout(main_layout)

    def _reset_defaults(self):
        """将所有设置控件恢复为默认值（不立即写入 config，需点保存生效）。"""
        defaults = self.config.default_config
        self.app_combo.setCurrentText(defaults.get("target_app", "WPS"))
        self.model_combo.setCurrentText(defaults.get("model_type", "Heavy"))
        self.mode_combo.setCurrentText(defaults.get("interaction_mode", "mouse"))
        idx_profile = self.profile_combo.findData(defaults.get("stability_profile", "stable"))
        if idx_profile >= 0:
            self.profile_combo.setCurrentIndex(idx_profile)
        self.cd_spin.setValue(int(defaults.get("cooldown", 1.0) * 1000))
        self.sensitivity_spin.setValue(int(defaults.get("mouse_sensitivity", 40)))
        self.edge_check.setChecked(bool(defaults.get("edge_acceleration_enabled", False)))
        self.edge_strength_spin.setValue(int(defaults.get("edge_acceleration_strength", 35)))
        self.y_canvas_check.setChecked(bool(defaults.get("edge_y_canvas_enabled", True)))
        self.y_dz_bottom_spin.setValue(int(defaults.get("edge_y_canvas_deadzone_bottom", 18)))
        self.y_dz_top_spin.setValue(int(defaults.get("edge_y_canvas_deadzone_top", 10)))
        self.pen_spin.setValue(int(defaults.get("pen_width", 20)))
        idx = self.voice_combo.findData(defaults.get("voice_assistant", "doubao"))
        if idx >= 0:
            self.voice_combo.setCurrentIndex(idx)
        idx_sr = self.zoom_sr_combo.findData(defaults.get("zoom_sr_engine", "auto"))
        if idx_sr >= 0:
            self.zoom_sr_combo.setCurrentIndex(idx_sr)
        for combo, key in [
            (self.right_combo, "SWIPE_RIGHT"), (self.left_combo, "SWIPE_LEFT"),
            (self.up_combo, "SWIPE_UP"), (self.down_combo, "SWIPE_DOWN"),
            (self.fist_combo, "FIST"), (self.thumb_combo, "THUMB_UP"),
            (self.scissor_combo, "SCISSOR"), (self.thumb_down_combo, "THUMB_DOWN"),
        ]:
            default_action = defaults.get("gesture_mapping", {}).get(key, "none")
            combo.setCurrentText(default_action)

    def _on_edge_toggled(self, state):
        self.edge_strength_spin.setEnabled(state == Qt.CheckState.Checked.value)

    def _on_profile_changed(self, _index=None):
        preset = self.config.stability_profile_defaults(
            self.profile_combo.currentData()
        )
        self.edge_check.setChecked(bool(preset.get("edge_acceleration_enabled", False)))
        self.edge_strength_spin.setValue(int(preset.get("edge_acceleration_strength", 35)))

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
        for index, camera in enumerate(cams):
            if camera.get("index") != current_idx:
                continue
            camera = dict(camera)
            camera["current"] = True
            saved_width = self.config.get("camera_width")
            saved_height = self.config.get("camera_height")
            if saved_width is not None and saved_height is not None:
                camera["width"] = saved_width
                camera["height"] = saved_height
            cams[index] = camera
        if not self._camera_worker_cancel.is_set():
            self._cameras_enumerated.emit(cams)

    def _on_cameras_enumerated(self, cameras):
        current_data = self.camera_combo.itemData(0)
        current_idx = (
            current_data.get("index")
            if isinstance(current_data, dict)
            else current_data
        )
        self.camera_combo.clear()
        if not cameras:
            self.camera_combo.addItem(
                f"摄像头 {current_idx}（当前）",
                {"index": current_idx, "current": True},
            )
            return
        for cam in cameras:
            self.camera_combo.addItem(cam["name"], cam)
        for i in range(self.camera_combo.count()):
            data = self.camera_combo.itemData(i)
            index = data.get("index") if isinstance(data, dict) else data
            if index == current_idx:
                self.camera_combo.setCurrentIndex(i)
                break

    def _on_camera_selection_changed(self, _combo_index=None):
        camera = self.camera_combo.currentData()
        if not isinstance(camera, dict) or camera.get("current"):
            self.save_btn.setEnabled(True)
            return
        if camera.get("width") and camera.get("height"):
            self.save_btn.setEnabled(True)
            return
        if camera.get("probing"):
            self.save_btn.setEnabled(False)
            return

        camera = dict(camera)
        camera["probing"] = True
        combo_index = self.camera_combo.currentIndex()
        self.camera_combo.setItemData(combo_index, camera)
        camera_name = camera.get("name") or f"摄像头 {camera.get('index')}"
        self.camera_combo.setItemText(
            combo_index,
            f"{camera_name}（检测分辨率中…）",
        )
        self.save_btn.setEnabled(False)
        camera_index = camera["index"]
        with self._camera_probe_lock:
            thread = self._camera_probe_threads.get(camera_index)
            if thread is not None and thread.is_alive():
                return
            thread = self._start_camera_worker(
                self._probe_camera_worker,
                f"CameraResolutionProbe-{camera_index}",
                args=(camera,),
            )
            self._camera_probe_threads[camera_index] = thread

    def _probe_camera_worker(self, camera):
        camera_index = camera["index"]
        error = ""
        try:
            resolution = probe_max_resolution(
                camera_index,
                min_fps=self.config.get("camera_min_fps") or 20,
                force_mjpeg=self.config.get("camera_force_mjpeg") is not False,
                preferred_backend=camera.get("backend"),
            )
            if resolution is None:
                resolution = (1280, 720)
                error = "未能测得稳定分辨率，使用 1280×720"
            # Windows camera release is asynchronous; settle off the UI thread.
            self._camera_worker_cancel.wait(0.5)
        except Exception as exc:
            logger.exception("摄像头 %d 分辨率探测失败", camera_index)
            resolution = (1280, 720)
            error = str(exc)
        finally:
            with self._camera_probe_lock:
                self._camera_probe_threads.pop(camera_index, None)
        if not self._camera_worker_cancel.is_set():
            self._camera_probed.emit(camera_index, resolution, error)

    def _on_camera_probed(self, camera_index, resolution, error):
        for combo_index in range(self.camera_combo.count()):
            camera = self.camera_combo.itemData(combo_index)
            if not isinstance(camera, dict) or camera.get("index") != camera_index:
                continue
            camera = dict(camera)
            camera["probing"] = False
            camera["width"], camera["height"] = resolution
            if error:
                camera["probe_warning"] = error
            self.camera_combo.setItemData(combo_index, camera)
            camera_name = camera.get("name") or f"摄像头 {camera_index}"
            self.camera_combo.setItemText(
                combo_index,
                f"{camera_name} ({resolution[0]}×{resolution[1]})",
            )
            if combo_index == self.camera_combo.currentIndex():
                self.save_btn.setEnabled(True)
            return

    def save_settings(self):
        camera_data = self.camera_combo.currentData()
        new_camera_idx = (
            camera_data.get("index")
            if isinstance(camera_data, dict)
            else camera_data
        )
        camera_switched = False
        parent = self.parent()
        old_camera_idx = self.config.get("camera_index")
        try:
            old_camera_idx = int(old_camera_idx) if old_camera_idx is not None else 0
        except (TypeError, ValueError):
            old_camera_idx = 0
        old_camera_info = {
            "index": old_camera_idx,
            "width": self.config.get("camera_width"),
            "height": self.config.get("camera_height"),
        }
        old_camera = getattr(getattr(parent, "orchestrator", None), "camera", None)
        if old_camera is not None:
            old_camera_info.update({
                "backend": getattr(old_camera, "_backend", None),
                "width": getattr(old_camera, "width", None),
                "height": getattr(old_camera, "height", None),
            })

        if isinstance(new_camera_idx, int) and new_camera_idx >= 0:
            if new_camera_idx != old_camera_idx:
                if isinstance(camera_data, dict) and camera_data.get("probing"):
                    QMessageBox.information(
                        self,
                        "摄像头检测中",
                        "正在检测所选摄像头，请稍候再保存。",
                    )
                    return
                if parent is not None and hasattr(parent, "switch_camera"):
                    ok = parent.switch_camera(new_camera_idx, camera_data)
                    if not ok:
                        QMessageBox.warning(
                            self, "摄像头切换失败",
                            f"无法启用摄像头 {new_camera_idx}，已保留原摄像头。\n"
                            "请检查设备是否被其它程序占用，或换 USB 口重试。",
                        )
                        return
                    camera_switched = True

        try:
            with self.config.batch_update():
                if isinstance(new_camera_idx, int) and new_camera_idx >= 0:
                    for key, value in camera_config_values(camera_data).items():
                        self.config.set(key, value)
                self.config.apply_stability_profile(self.profile_combo.currentData())
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
        except ConfigSaveError as exc:
            logger.exception("设置保存失败")
            if camera_switched and parent is not None and hasattr(parent, "switch_camera"):
                if not parent.switch_camera(old_camera_idx, old_camera_info):
                    logger.error("配置保存失败后，摄像头回滚到 %d 失败", old_camera_idx)
            QMessageBox.critical(
                self,
                "保存设置失败",
                f"设置未保存，已保留原配置。\n\n{exc}",
            )
            return
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
        # 显式依赖注入：overlay/cursor_overlay/toolbar/hwnd/config/mouse 全部通过参数传入，
        # 不再通过 parent 反向读取，便于测试 mock 和生命周期管理。
        self.orchestrator = AirControlOrchestrator(
            self.overlay, self.cursor_overlay, self.toolbar,
            hwnd=int(self.winId()), parent=self, config=self.config, mouse=self.mouse
        )

        self.init_ui()
        self._connect_toolbar()

        # 4. 关联 Orchestrator 核心业务流信号
        self.orchestrator.frame_processed.connect(self._on_frame_processed)
        self.orchestrator.voice_status_updated.connect(self._on_voice_status_updated)
        self.orchestrator.fps_updated.connect(self._on_fps_updated)
        self.orchestrator.mode_changed.connect(self._on_mode_changed)
        self.orchestrator.status_updated.connect(self._on_status_updated)
        self.orchestrator.minimize_requested.connect(self.showMinimized)
        self.orchestrator.restore_requested.connect(self._on_restore_requested)

        self._current_fps = 0.0
        self._preview_ms = deque(maxlen=180)
        self._last_preview_perf_log = time.monotonic()
        self._last_mode_text = ""   # 缓存 mode_label 文本，避免每帧 setText
        self._last_hint_text = ""   # 缓存 hint_label 文本，避免每帧 setText
        self.drag_pos = None

        # 全局录像热键（F8）：F5 只在悬浮窗有键盘焦点时生效，站远操作够不到窗口。
        # 用 QTimer 轮询 GetAsyncKeyState，任何窗口焦点下都能启停原始帧录制。
        # 不选 F5：它是 PPT 放映键（ppt_controller 会注入 F5），全局占用会冲突。
        self._rec_hotkey_held = False
        self._rec_hotkey_timer = QTimer(self)
        self._rec_hotkey_timer.timeout.connect(self._poll_rec_hotkey)
        self._rec_hotkey_timer.start(150)

    def init_ui(self):
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setWindowTitle(f"AirControl v{__version__}")

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

        # 录制指示：无边框窗口没有标题栏，REC 状态必须画在窗体上（F8/F5 启停时显隐）。
        self.rec_label = QLabel("● REC", self)
        self.rec_label.setStyleSheet(
            "color: white; background-color: rgba(220, 40, 40, 200);"
            "padding: 2px 8px; border-radius: 4px; font-weight: bold;"
        )
        self.rec_label.adjustSize()
        self.rec_label.move(10, 10)
        self.rec_label.raise_()
        self.rec_label.hide()

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
        self.mode_label.setCursor(Qt.CursorShape.PointingHandCursor)
        self.mode_label.setToolTip("点击切换模式（或按 F2/F3/F4）")
        self.mode_label.mousePressEvent = lambda ev: self._cycle_mode()
        self.mode_label.setStyleSheet(f"""
            QLabel {{
                color: white;
                background-color: rgba(0, 0, 0, 140);
                border-radius: 8px;
                font-size: {s(15)}px;
                font-weight: bold;
                padding: 1px 6px;
            }}
            QLabel:hover {{
                background-color: rgba(0, 0, 0, 200);
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

        # 空格是真值标记键：悬浮窗按钮一律不吃键盘焦点，否则录像时敲空格
        # 会在松开瞬间激活焦点按钮（X = 直接退出程序，2026-07-22 实录"崩溃"）。
        for _btn in (self.btn_settings, self.btn_minimize, self.btn_close):
            _btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)

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

        # 动态状态标签：显示 Orchestrator 发射的实时状态（书写中/未检测到手/清屏进度/推理错误等）
        self.status_label = QLabel()
        self.status_label.setFixedHeight(s(22))
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet(f"""
            QLabel {{
                color: #39ff14;
                background-color: rgba(0, 0, 0, 210);
                border-radius: 6px;
                font-size: {s(12)}px;
                font-weight: bold;
                padding: 2px 6px;
            }}
        """)
        self._status_color_cache = (0, 255, 0)
        main_layout.addWidget(self.status_label)

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

        # 默认定位到鼠标光标所在屏幕的左下角（多显示器场景下跟随用户操作屏），
        # 找不到光标时回退主屏。
        from PyQt6.QtGui import QCursor
        cursor_screen = QApplication.screenAt(QCursor.pos())
        if cursor_screen is None:
            cursor_screen = QApplication.primaryScreen()
        screen = cursor_screen.geometry()
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

    def switch_camera(self, new_index, camera_info=None):
        return self.orchestrator.switch_camera(new_index, camera_info)

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
        preview_started = time.perf_counter()
        h, w, ch = frame.shape
        self.video_label.setPixmap(
            _make_preview_pixmap(
                frame,
                self.video_label.width(),
                self.video_label.height(),
            )
        )
        self._record_preview_performance(
            (time.perf_counter() - preview_started) * 1000.0
        )
        # 模式标签上直接显示实时帧率（_current_fps 由 fps_updated 每秒刷新），便于现场观察卡顿
        # 仅在文本变化时 setText，避免每帧触发 Qt 重绘
        mode_text = f"{self._mode_name_zh()}　{self._current_fps:.0f} FPS"
        if mode_text != self._last_mode_text:
            self._last_mode_text = mode_text
            self.mode_label.setText(mode_text)
        hint_text = self._mode_hint_zh()
        if hint_text != self._last_hint_text:
            self._last_hint_text = hint_text
            self.hint_label.setText(hint_text)

        # 角落分辨率徽标：仅在尺寸变化时刷新（避免每帧 adjustSize/raise_）
        res_text = f"{w}×{h}"
        if res_text != self._res_text:
            self._res_text = res_text
            self.res_label.setText(res_text)
            self.res_label.adjustSize()
            self.res_label.show()
            self.res_label.raise_()

    def _record_preview_performance(self, elapsed_ms):
        self._preview_ms.append(float(elapsed_ms))
        now = time.monotonic()
        if now - self._last_preview_perf_log < 5.0:
            return
        self._last_preview_perf_log = now
        values = sorted(self._preview_ms)
        if not values:
            return

        def percentile(fraction):
            index = min(
                len(values) - 1,
                int(round((len(values) - 1) * fraction)),
            )
            return values[index]

        logger.info(
            "[PERF UI] preview p50/p95=%.2f/%.2fms | samples=%d",
            percentile(0.50),
            percentile(0.95),
            len(values),
        )

    def _on_voice_status_updated(self, text):
        if not hasattr(self, "voice_label") or self.voice_label is None:
            return
        if self.voice_label.text() == text:
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

    def _on_status_updated(self, text, color):
        """显示 Orchestrator 发射的实时状态文字（带颜色）。

        信号每帧可能多次发射，仅在实际变化时刷新以避免无谓重绘。
        """
        if not hasattr(self, "status_label") or self.status_label is None:
            return
        if self.status_label.text() == text and self._status_color_cache == color:
            return
        self.status_label.setText(text)
        self._status_color_cache = color
        # 将 RGB 元组转为 hex 颜色
        try:
            r, g, b = int(color[0]), int(color[1]), int(color[2])
            hex_color = f"#{r:02x}{g:02x}{b:02x}"
        except (TypeError, IndexError, ValueError):
            hex_color = "#00ff88"
        self.status_label.setStyleSheet(f"""
            QLabel {{
                color: {hex_color};
                background-color: rgba(0, 0, 0, 210);
                border-radius: 6px;
                font-size: {self._ui_scale * 12:.0f}px;
                font-weight: bold;
                padding: 2px 6px;
            }}
        """)

    def _on_restore_requested(self):
        if self.isMinimized():
            self.showNormal()
        else:
            self.show()
        self.raise_()
        self.activateWindow()

    def _mode_name_zh(self):
        mode = self.orchestrator.current_mode_name
        return MODE_NAME_ZH.get(mode, "未知模式")

    def _mode_hint_zh(self):
        mode = self.orchestrator.current_mode_name
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
        mode = self.orchestrator.current_mode_name if hasattr(self, "orchestrator") else "presentation"
        mode_zh = MODE_NAME_ZH.get(mode, mode)

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
        current_mode = self.orchestrator.current_mode_name

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

    def _cycle_mode(self):
        """点击模式标签时循环切换 演示→鼠标→板书→演示。"""
        current = self.orchestrator.current_mode_name
        try:
            idx = MODE_NAMES.index(current)
        except ValueError:
            idx = 0
        next_mode = MODE_NAMES[(idx + 1) % len(MODE_NAMES)]
        self.orchestrator.set_mode(next_mode)

    # F2/F3/F4 → 模式名映射，与 MODE_NAMES 顺序一致
    _MODE_SHORTCUTS = {
        Qt.Key.Key_F2: "presentation",
        Qt.Key.Key_F3: "mouse",
        Qt.Key.Key_F4: "draw",
    }

    def _toggle_recording_ui(self):
        """启停原始帧录制并更新录制指示（F5 焦点热键与 F8 全局热键共用）。

        无边框窗口不显示标题栏，REC 红点画在窗体左上角——悬浮窗始终置顶，
        站远也能看到。
        """
        now_recording, path = self.orchestrator.toggle_recording()
        if now_recording:
            self.setWindowTitle(f"AirControl v{__version__} [REC: {os.path.basename(path)}]")
            self.rec_label.show()
            self.rec_label.raise_()
            logger.info("录制开始 -> %s", path)
        else:
            self.setWindowTitle(f"AirControl v{__version__}")
            self.rec_label.hide()
            logger.info("录制停止 -> %s", path)

    def _poll_rec_hotkey(self):
        """轮询 F8（VK 0x77）全局按键状态，按下沿触发录制启停。"""
        try:
            import win32api
            down = bool(win32api.GetAsyncKeyState(0x77) & 0x8000)
        except Exception:
            return
        if down and not self._rec_hotkey_held:
            self._toggle_recording_ui()
        self._rec_hotkey_held = down

    def keyPressEvent(self, event):
        """键盘快捷键：F1 调试覆盖层，F2/F3/F4 切换模式，F5 录制/停止（需窗口焦点）。"""
        key = event.key()
        if key == Qt.Key.Key_F1:
            if hasattr(self.orchestrator, "inference_worker"):
                new_state = not self.orchestrator.inference_worker.debug_overlay
                self.orchestrator.inference_worker.set_debug_overlay(new_state)
            event.accept()
            return
        if key in self._MODE_SHORTCUTS:
            self.orchestrator.set_mode(self._MODE_SHORTCUTS[key])
            event.accept()
            return
        if key == Qt.Key.Key_F5:
            self._toggle_recording_ui()
            event.accept()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event):
        # 优雅释放后台全部服务
        self._shutdown_incomplete = self.orchestrator.close() is False

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


def _request_admin_relaunch(argv=None):
    """Request an explicit elevated relaunch and report whether it started."""
    import ctypes

    argv = list(sys.argv if argv is None else argv)
    if getattr(sys, "frozen", False):
        executable = sys.executable
        relaunch_args = argv[1:]
    else:
        executable = sys.executable
        relaunch_args = [os.path.abspath(__file__), *argv[1:]]
    try:
        result = ctypes.windll.shell32.ShellExecuteW(
            None,
            "runas",
            executable,
            subprocess.list2cmdline(relaunch_args),
            None,
            1,
        )
    except Exception as exc:
        logger.warning("提权请求异常: %s，将以普通权限启动。", exc)
        return False
    if int(result) <= 32:
        logger.warning(
            "提权请求被拒绝或失败（ShellExecuteW=%s），将以普通权限启动。",
            result,
        )
        return False
    return True


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
            from services.hand_tracker_factory import create_hand_tracker
            from services.voice_command import VoiceCommandService

            config = ConfigManager()
            tracker = HandTracker(static_image_mode=True, config=config)
            tracker.close()

            # hagrid_yolo 引擎自检：模型为可选下载，缺失时跳过而非失败。
            yolo_model = resource_path("models", "hand_yolov8n.onnx")
            if os.path.isfile(yolo_model):
                yolo_tracker = create_hand_tracker(engine="hagrid_yolo", config=config)
                yolo_tracker.close()
            else:
                logger.info(
                    "自检: hand_yolov8n.onnx 未安装，hagrid_yolo 引擎"
                    "及 engine_auto_switch 远距自动切换不可用。"
                    "如需使用，请参考 README 下载模型。"
                )

            voice = VoiceCommandService(config)
            voice._current_mode = config.get("interaction_mode", "mouse")
            voice._init_kws()
            voice.stop()
        except Exception:
            logger.exception("发布自检失败")
            return 3
        return 0

    # 最小权限启动是默认行为。只有显式传入 --elevate 时才请求 UAC；
    # 用户拒绝或 ShellExecuteW 返回错误码时继续以普通权限运行。
    if "--elevate" in sys.argv and not is_admin():
        if _request_admin_relaunch():
            return 0

    # 统一日志配置：所有模块日志写入 gesture.log（在崩溃捕获之前，确保 critical 也能落盘）
    from log_config import setup_logging
    setup_logging()

    logger.info("AirControl v%s 启动", __version__)

    # 崩溃捕获：原生段错误 / 主线程 / 工作线程 / Qt 致命消息 → crash.log
    from crash_handler import install as install_crash_handler
    install_crash_handler()

    app = QApplication(sys.argv)
    window = FloatingWindow()
    window.show()
    exit_code = app.exec()
    if getattr(window, "_shutdown_incomplete", False) is True:
        # A native camera/ASR/codec call ignored its cooperative stop deadline.
        # Those resources were deliberately not released concurrently. Let the
        # OS reclaim the process instead of hanging or racing a live native call.
        logger.critical("原生后台任务未能按时退出，执行受控强制进程结束")
        logging.shutdown()
        os._exit(exit_code)
    return exit_code


if __name__ == "__main__":
    exit_code = main()
    if "--self-test" in sys.argv:
        # Some native inference runtimes retain worker threads in frozen apps.
        # The diagnostic has already closed its engines, so exit deterministically.
        os._exit(exit_code)
    sys.exit(exit_code)
