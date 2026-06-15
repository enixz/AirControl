import json
import logging
import os
import sys
import threading

from runtime_paths import data_path, project_root

logger = logging.getLogger(__name__)


# 关键字段校验 schema：(type, validator, default)
# validator 是 callable，返回 True 表示通过；类型不符或验证不过用默认值
def _is_num_in(lo, hi):
    return lambda v: (
        isinstance(v, (int, float))
        and not isinstance(v, bool)
        and lo <= v <= hi
    )


def _is_int_in(lo, hi):
    return lambda v: isinstance(v, int) and not isinstance(v, bool) and lo <= v <= hi


def _is_bool(value):
    return isinstance(value, bool)


def _is_optional_string(value):
    return value is None or isinstance(value, str)


_CONFIG_SCHEMA = {
    "target_app": (str, lambda v: v in ("WPS", "PowerPoint"), "WPS"),
    "detection_engine": (str, lambda v: v == "mediapipe", "mediapipe"),
    "camera_index": (int, _is_int_in(0, 9), 0),
    "camera_width": (
        (int, type(None)),
        lambda v: v is None or (isinstance(v, int) and 320 <= v <= 4096),
        None,
    ),
    "camera_height": (
        (int, type(None)),
        lambda v: v is None or (isinstance(v, int) and 240 <= v <= 2160),
        None,
    ),
    "camera_force_mjpeg": (bool, _is_bool, True),
    "camera_min_fps": (int, _is_int_in(5, 60), 10),
    "cooldown": ((int, float), _is_num_in(0.0, 10.0), 1.0),
    "swipe_threshold": (int, _is_int_in(10, 500), 60),
    "mouse_sensitivity": (int, _is_int_in(1, 200), 40),
    "pen_width": (int, _is_int_in(1, 100), 15),
    "pen_width_auto_scale": (bool, _is_bool, False),
    "edge_acceleration_enabled": (bool, _is_bool, False),
    "edge_acceleration_strength": (int, _is_int_in(0, 500), 30),
    "edge_y_canvas_enabled": (bool, _is_bool, True),
    "edge_y_canvas_deadzone_bottom": (int, _is_int_in(0, 100), 18),
    "edge_y_canvas_deadzone_top": (int, _is_int_in(0, 100), 10),
    "dominant_hand": (
        str,
        lambda v: v in ("Left", "Right", "Auto"),
        "Auto",
    ),
    "hand_detection_confidence": ((int, float), _is_num_in(0.1, 1.0), 0.4),
    "hand_presence_confidence": ((int, float), _is_num_in(0.1, 1.0), 0.5),
    "hand_tracking_confidence": ((int, float), _is_num_in(0.1, 1.0), 0.5),
    "model_type": (str, lambda v: v in ("Heavy", "Lite", "Full"), "Heavy"),
    "interaction_mode": (
        str,
        lambda v: v in ("mouse", "draw", "presentation"),
        "mouse",
    ),
    "voice_command_threshold": ((int, float), _is_num_in(0.0, 1.0), 0.25),
    "voice_command_enabled": (bool, _is_bool, True),
    "voice_assistant": (
        str,
        lambda v: v in ("doubao", "qianwen"),
        "doubao",
    ),
    "dictation_enabled": (bool, _is_bool, True),
    "dictation_model_dir": (str, lambda v: bool(v.strip()), "models/sense-voice"),
    "dictation_language": (str, lambda v: bool(v.strip()), "auto"),
    "dictation_use_itn": (bool, _is_bool, True),
    "dictation_num_threads": (int, _is_int_in(1, 16), 2),
    "dictation_partial_window_sec": (
        (int, float),
        _is_num_in(3.0, 60.0),
        12.0,
    ),
    "floating_window_scale": ((int, float), _is_num_in(1.0, 3.0), 1.5),
    "debug_overlay": (bool, _is_bool, False),
    "zoom_sr_engine": (
        str,
        lambda v: v in ("auto", "espcn", "realesrgan_cpu", "realesrgan_gpu", "none"),
        "auto",
    ),
    # crop-zoom 触发阈值（手 bbox 占全帧比）。far 越小→越远才放大（板书近距离更稳）。
    "zoom_far_threshold": ((int, float), _is_num_in(0.001, 0.05), 0.008),
    "zoom_near_threshold": ((int, float), _is_num_in(0.01, 0.20), 0.040),
    # 手部关键点一欧元滤波：min_cutoff 越小静止越不抖；beta 越大运动越跟手。
    "hand_smoothing_min_cutoff": ((int, float), _is_num_in(0.05, 5.0), 0.5),
    "hand_smoothing_beta": ((int, float), _is_num_in(0.0, 1.0), 0.015),
    # 远距离 ZOOM 鲁棒性：连续丢帧多少帧才断 ZOOM；人脸检测短边分辨率（越大越能找回远处的手）。
    "zoom_miss_frames": (int, _is_int_in(3, 60), 10),
    "face_detect_short": (int, _is_int_in(240, 1280), 400),
    "mode_switch_hold_sec": ((int, float), _is_num_in(0.4, 3.0), 1.0),
    "mode_switch_vote_ratio": ((int, float), _is_num_in(0.5, 1.0), 0.6),
    "mode_switch_release_sec": ((int, float), _is_num_in(0.1, 1.0), 0.25),
    "draw_frontality_gate": ((int, float), _is_num_in(0.0, 3.0), 0.55),
    "draw_record_trace": (bool, _is_bool, False),
    "draw_thumb_lift": (bool, _is_bool, False),
    "draw_two_finger_geom": (bool, _is_bool, False),
    "draw_vote_window_sec": ((int, float), _is_num_in(0.1, 1.0), 0.30),
    "draw_vote_ratio": ((int, float), _is_num_in(0.5, 1.0), 0.60),
    "record_raw_video": (bool, _is_bool, False),
    "record_raw_max_frames": (int, _is_int_in(1, 100000), 2000),
    "record_raw_max_seconds": (
        (int, float),
        _is_num_in(1.0, 3600.0),
        120.0,
    ),
    "wps_exe_path": ((str, type(None)), _is_optional_string, None),
    "powerpoint_exe_path": ((str, type(None)), _is_optional_string, None),
}


def _validate_config(cfg):
    """逐字段校验，错误用默认值兜底，返回修正后的 dict 和警告列表。"""
    warnings = []
    for key, (expected_type, validator, default) in _CONFIG_SCHEMA.items():
        if key not in cfg:
            continue
        value = cfg[key]
        # 类型检查 (bool 被排除在 int 之外)
        if not isinstance(value, expected_type) or (
            expected_type is int and isinstance(value, bool)
        ):
            warnings.append(
                f"{key}={value!r} 类型不符（期望 {expected_type}），使用默认值 {default!r}"
            )
            cfg[key] = default
            continue
        # 业务规则
        if not validator(value):
            warnings.append(
                f"{key}={value!r} 超出有效范围，使用默认值 {default!r}"
            )
            cfg[key] = default
    return cfg, warnings


class ConfigManager:
    def __init__(self, config_file="config.json"):
        if os.path.isabs(config_file):
            self.config_file = config_file
        elif getattr(sys, "frozen", False) and config_file == "config.json":
            self.config_file = data_path(config_file)
        else:
            self.config_file = os.path.join(project_root(), config_file)
        self.default_config = {
            "target_app": "WPS",
            "model_type": "Heavy",
            "interaction_mode": "mouse",
            "camera_index": 0,
            "cooldown": 1.0,
            "swipe_threshold": 60,
            "mouse_sensitivity": 40,
            "pen_width": 15,
            "pen_width_auto_scale": False,
            "camera_width": None,
            "camera_height": None,
            "camera_force_mjpeg": True,
            "camera_min_fps": 10,
            "edge_acceleration_enabled": False,
            "edge_acceleration_strength": 30,
            "edge_y_canvas_enabled": True,
            "edge_y_canvas_deadzone_bottom": 18,
            "edge_y_canvas_deadzone_top": 10,
            "voice_assistant": "doubao",
            "voice_command_enabled": True,
            "voice_command_threshold": 0.25,
            "zoom_sr_engine": "auto",
            "zoom_far_threshold": 0.008,
            "zoom_near_threshold": 0.040,
            "hand_smoothing_min_cutoff": 0.5,
            "hand_smoothing_beta": 0.015,
            "zoom_miss_frames": 10,
            "face_detect_short": 400,
            "detection_engine": "mediapipe",
            "dominant_hand": "Auto",
            "hand_detection_confidence": 0.4,
            "hand_presence_confidence": 0.5,
            "hand_tracking_confidence": 0.5,
            "mode_switch_hold_sec": 1.0,
            "mode_switch_vote_ratio": 0.6,
            "mode_switch_release_sec": 0.25,
            "draw_frontality_gate": 0.55,
            "draw_record_trace": False,
            "draw_thumb_lift": False,
            "draw_two_finger_geom": False,
            "draw_vote_window_sec": 0.30,
            "draw_vote_ratio": 0.60,
            "dictation_enabled": True,
            "dictation_model_dir": "models/sense-voice",
            "dictation_language": "auto",
            "dictation_use_itn": True,
            "dictation_num_threads": 2,
            "dictation_partial_window_sec": 12.0,
            "record_raw_video": False,
            "record_raw_max_frames": 2000,
            "record_raw_max_seconds": 120.0,
            "debug_overlay": False,
            "floating_window_scale": 1.5,
            "wps_exe_path": None,
            "powerpoint_exe_path": None,
            "gesture_mapping": {
                "SWIPE_RIGHT": "next_slide",
                "SWIPE_LEFT": "prev_slide",
                "SWIPE_UP": "start_presentation",
                "SWIPE_DOWN": "end_presentation",
                "FIST": "none",
                "THUMB_UP": "switch_app",
                "SCISSOR": "launch_voice_assistant",
                "THUMB_DOWN": "hang_up_voice_assistant"
            }
        }
        self.config = self.load_config()
        self._dirty = False
        self._batch_depth = 0
        self._save_lock = threading.Lock()

    def load_config(self):
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                    merged = self.default_config.copy()
                    merged.update(user_config)
                    merged_mapping = self.default_config["gesture_mapping"].copy()
                    merged_mapping.update(user_config.get("gesture_mapping", {}))
                    merged["gesture_mapping"] = merged_mapping
                    # Schema 校验：错误字段用默认值兜底，避免误编辑导致黑屏
                    merged, warnings = _validate_config(merged)
                    for w in warnings:
                        logger.warning("配置校验: %s", w)
                    return merged
            except Exception as e:
                logger.warning("读取配置失败，使用默认配置: %s", e)
        return self.default_config.copy()

    def save_config(self):
        if self._batch_depth:
            self._dirty = True
            return True
        return self._do_save()

    def _do_save(self):
        with self._save_lock:
            temp_path = self.config_file + ".tmp"
            try:
                os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
                with open(temp_path, 'w', encoding='utf-8') as f:
                    json.dump(self.config, f, indent=4, ensure_ascii=False)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(temp_path, self.config_file)
                return True
            except Exception as e:
                logger.error("保存配置失败: %s", e)
                try:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                except OSError:
                    pass
                return False

    def batch_update(self):
        return _BatchContext(self)

    def get(self, key, *args):
        """获取配置值。支持可选默认值：config.get("key") 或 config.get("key", default)。"""
        return self.config.get(key, *args)

    def set(self, key, value):
        self.config[key] = value
        self.save_config()

    def get_mapping(self, gesture):
        return self.config["gesture_mapping"].get(gesture, "none")

    def set_mapping(self, gesture, action):
        self.config["gesture_mapping"][gesture] = action
        self.save_config()


class _BatchContext:
    def __init__(self, manager: ConfigManager):
        self._manager = manager

    def __enter__(self):
        self._manager._batch_depth += 1
        return self._manager

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._manager._batch_depth = max(0, self._manager._batch_depth - 1)
        if self._manager._batch_depth == 0 and self._manager._dirty:
            self._manager._do_save()
            self._manager._dirty = False
