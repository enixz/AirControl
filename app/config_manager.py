import copy
import json
import logging
import os
import shutil
import sys
import threading
from datetime import datetime

from modes import MODE_NAMES
from runtime_paths import data_path, project_root, resource_path

logger = logging.getLogger(__name__)


class ConfigSaveError(OSError):
    """Raised when a configuration update cannot be persisted atomically."""

    def __init__(self, config_file, cause=None):
        message = f"无法保存配置文件: {config_file}"
        if cause is not None:
            message = f"{message} ({cause})"
        super().__init__(message)
        self.config_file = config_file
        self.cause = cause


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


# 与 services/truth_event_logger.py 的 MARKER_VK 保持同步。
_TRUTH_MARKERS = {
    "space", "enter", "shift", "ctrl", "alt", "tab", "x", "z",
    "rbutton", "mbutton", "xbutton1", "xbutton2", "pageup", "pagedown",
}


def _is_marker_list(value):
    """record_truth_marker：逗号分隔的标记键名列表（至少一个有效值）。"""
    if not isinstance(value, str):
        return False
    parts = [p.strip() for p in value.split(",") if p.strip()]
    return bool(parts) and all(p in _TRUTH_MARKERS for p in parts)


_STABILITY_PROFILE_PRESETS = {
    # v1.3.6 默认档：吸收 v1.3 的可预期手感，优先少误触、少断笔。
    # v1.4.0：三个 pinch 实验开关均不纳入默认档；离线录像没有点击/拖拽真值，
    # 不足以证明默认开启不会增加漏点或拖拽延迟。用户仍可在 config 中显式启用。
    "stable": {
        "edge_acceleration_enabled": False,
        "edge_acceleration_strength": 35,
        "long_range_enabled": False,
        "draw_thumb_lift": False,
        "draw_vote_ratio": 0.60,
        "adaptive_skip_enabled": False,
        "geometric_constraint_enabled": False,
        "temporal_voter_enabled": False,
        "pinch_freeze_enabled": False,
        "pinch_hysteresis_enabled": False,
    },
    # 保留 1.3.5 的触达增强，但把鼠标边缘加速降到温和强度。
    "balanced": {
        "edge_acceleration_enabled": True,
        "edge_acceleration_strength": 35,
        "long_range_enabled": True,
        "draw_thumb_lift": False,
        "draw_vote_ratio": 0.60,
        "adaptive_skip_enabled": False,
        "geometric_constraint_enabled": False,
        "temporal_voter_enabled": False,
        "pinch_freeze_enabled": False,
        "pinch_hysteresis_enabled": False,
    },
    # 远距演示/板书专项，保持 crop-zoom、人脸引导和预测补帧打开。
    "long_range": {
        "edge_acceleration_enabled": True,
        "edge_acceleration_strength": 60,
        "long_range_enabled": True,
        "draw_thumb_lift": False,
        "draw_vote_ratio": 0.60,
        "adaptive_skip_enabled": False,
        "geometric_constraint_enabled": False,
        "temporal_voter_enabled": False,
        "pinch_freeze_enabled": False,
        "pinch_hysteresis_enabled": False,
    },
}


_CONFIG_SCHEMA = {
    "target_app": (str, lambda v: v in ("WPS", "PowerPoint"), "WPS"),
    "stability_profile": (
        str,
        lambda v: v in _STABILITY_PROFILE_PRESETS,
        "stable",
    ),
    "detection_engine": (str, lambda v: v in ("mediapipe", "hagrid_yolo"), "mediapipe"),
    # 远距引擎三态自动切换（默认关闭，不打扰现有用户），闭环：
    #   NEAR(mediapipe 裸检) --连续无手--> CAPTURE(hagrid_yolo 全帧捕获)
    #   CAPTURE --连续稳定单手--> FAR_TRACK(mediapipe + ZOOM 运行时覆盖)
    #   FAR_TRACK --手变大/走近持续--> NEAR；--再次连续无手--> CAPTURE
    # 状态机见 services/engine_auto_switcher.py，仅主流程运行时生效。
    "engine_auto_switch": (bool, _is_bool, False),
    # 连续无手多少帧 → 进 CAPTURE（NEAR/FAR_TRACK 共用，丢手即让 YOLO 抓）
    "engine_auto_switch_no_hand_frames": (int, _is_int_in(5, 600), 60),
    # CAPTURE 态连续稳定单手多少帧 → 交接 FAR_TRACK（多手误检帧不计入）
    "engine_auto_switch_hand_frames": (int, _is_int_in(5, 600), 30),
    # FAR_TRACK→NEAR：单手 bbox 占全帧比 ≥ 该阈值视为"手变大/走近"。
    # 与 zoom_near_threshold 同量级（4%）；宁慢勿错，配合 near_frames 持续判据。
    "engine_auto_switch_near_bbox_ratio": ((int, float), _is_num_in(0.01, 0.30), 0.04),
    # "手变大/走近"需持续多少帧才回 NEAR（防瞬时大手误触发撤覆盖）
    "engine_auto_switch_near_frames": (int, _is_int_in(5, 600), 90),
    # 任意迁移后的冷却秒数：期间不计帧不迁移（新引擎/新链路预热 + 防抖）
    "engine_auto_switch_cooldown_sec": ((int, float), _is_num_in(0.5, 60.0), 5.0),
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
    "edge_acceleration_strength": (int, _is_int_in(0, 500), 35),
    # Freeze-on-pinch（实施方案 Phase 3.1）：捏合上升沿冻结光标，grace 期内
    # 锁定在瞄准点消除漂移；grace 结束后解冻允许 DRAG。
    # 录像只有观察性漂移指标、没有点击/拖拽真值，默认关闭，允许显式试用。
    "pinch_freeze_enabled": (bool, _is_bool, False),
    "pinch_freeze_grace_sec": ((int, float), _is_num_in(0.0, 2.0), 0.3),
    # Pinch 双阈值滞回（实施方案 Phase 3.2）：ENTER/EXIT 双阈值消除边界抖动。
    # 翻转减少不等同于准确率提高；缺少事件真值前默认关闭。
    "pinch_hysteresis_enabled": (bool, _is_bool, False),
    # 仅退出方向滞回：保留旧版 0.35 进入阈值，已捏合时以 0.40 阈值退出。
    # 带真值 A/B（14 组）未增加漏检/延迟，误报 9→4，故默认启用；可显式关闭。
    "pinch_exit_hysteresis_enabled": (bool, _is_bool, True),
    # thumb_extended 旋转不变判定（实施方案 Phase 3.3）：用拇指 tip 到掌心中轴的
    # 垂直距离/掌宽 替代旧的 thumb_tip_to_index_mcp 距离。暂默认关闭（A/B 显示
    # 阈值 0.50 偏低，perp_ratio 实测均值 1.106，需采集内收姿势标定后再开启）。
    "thumb_perp_ratio_enabled": (bool, _is_bool, False),
    "edge_y_canvas_enabled": (bool, _is_bool, True),
    "edge_y_canvas_deadzone_bottom": (int, _is_int_in(0, 100), 18),
    "edge_y_canvas_deadzone_top": (int, _is_int_in(0, 100), 10),
    "dominant_hand": (
        str,
        lambda v: v in ("Left", "Right", "Auto"),
        "Auto",
    ),
    "hand_detection_confidence": ((int, float), _is_num_in(0.1, 1.0), 0.5),
    "hand_presence_confidence": ((int, float), _is_num_in(0.1, 1.0), 0.5),
    "hand_tracking_confidence": ((int, float), _is_num_in(0.1, 1.0), 0.5),
    "model_type": (str, lambda v: v in ("Heavy", "Lite", "Full"), "Heavy"),
    "interaction_mode": (
        str,
        lambda v: v in MODE_NAMES,
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
    # 阶段 2.10：回到 0.5/0.015（D:\airControl 实测不拉扯的参数）。更强的低通
    # 吸收 MediaPipe 检测抖动，配合 handedness-keyed smoother 根治双手拉扯。
    "hand_smoothing_min_cutoff": ((int, float), _is_num_in(0.05, 5.0), 0.5),
    "hand_smoothing_beta": ((int, float), _is_num_in(0.0, 1.0), 0.015),
    # 推理降采样宽度：高分辨率帧（如 1080p）先缩到此宽度再喂 MediaPipe，坐标归一化无需补偿。
    # 1080p 整帧推理 ~42ms → ~640-720px 后 ~15ms，直接决定"快速移动跟不跟手"。0=不降采样。
    "inference_max_width": (int, _is_int_in(0, 1920), 720),
    # YOLO 手部检测器置信度阈值（仅 hagrid_yolo 引擎使用）。
    # 越低检出越多但误检也多；0.25 是 HaGRID v2 推荐值。
    "yolo_confidence": ((int, float), _is_num_in(0.05, 0.95), 0.25),
    # 远距 CAPTURE 面向单个主控手。限制 YOLO 只交给下游一个最高分候选，
    # 避免背景误检形成"多手"并阻塞自动切换；MediaPipe 仍支持双手。
    "yolo_max_hands": (int, _is_int_in(1, 2), 1),
    # 远距离 ZOOM 鲁棒性：连续丢帧多少帧才断 ZOOM；人脸检测短边分辨率（越大越能找回远处的手）。
    "zoom_miss_frames": (int, _is_int_in(3, 60), 10),
    "face_detect_short": (int, _is_int_in(240, 1280), 400),
    # === 投机式增强层总开关（阶段1：默认关闭，详见 docs/修复记录_阶段0-1.md）===
    # 这些层是 GLM5.2 强化时叠加的，互相打架反而降低识别率/产生闪烁/断笔/不跟手。
    # 默认关闭 = 回到接近原版 aircontrol 的直管线；需要时单独打开做 A/B 验证。
    "adaptive_skip_enabled": (bool, _is_bool, False),       # 自适应跳帧补帧（冻结→跳变）
    "long_range_enabled": (bool, _is_bool, False),          # 稳定档默认关闭，远距档位再开启
    "geometric_constraint_enabled": (bool, _is_bool, False),  # 骨长约束滤波（运动时冻结关键点）
    "hand_prediction_enabled": (bool, _is_bool, True),      # 幽灵手/丢手预测补帧（与老版一致，默认开启）
    "temporal_voter_enabled": (bool, _is_bool, False),      # 时序投票器+FSM（阶段2.11默认关闭，回到老版基线）
    "mode_switch_hold_sec": ((int, float), _is_num_in(0.4, 3.0), 1.0),
    "mode_switch_vote_ratio": ((int, float), _is_num_in(0.5, 1.0), 0.6),
    "mode_switch_release_sec": ((int, float), _is_num_in(0.1, 1.0), 0.25),
    "draw_frontality_gate": ((int, float), _is_num_in(0.0, 3.0), 0.65),
    "draw_record_trace": (bool, _is_bool, False),
    "draw_thumb_lift": (bool, _is_bool, False),
    "draw_two_finger_geom": (bool, _is_bool, False),
    "draw_vote_window_sec": ((int, float), _is_num_in(0.1, 1.0), 0.30),
    "draw_vote_ratio": ((int, float), _is_num_in(0.5, 1.0), 0.60),
    # 书写中"张掌立即抬笔"的连续帧去抖：单帧 is_open_palm 噪声不再断笔，
    # 需连续 N 帧才确认抬笔（真张掌清屏几乎无感延迟）。1=旧的单帧行为。
    "draw_open_palm_lift_frames": (int, _is_int_in(1, 10), 3),
    "record_raw_video": (bool, _is_bool, False),
    "record_raw_max_frames": (int, _is_int_in(1, 100000), 2000),
    "record_raw_max_seconds": (
        (int, float),
        _is_num_in(1.0, 3600.0),
        120.0,
    ),
    "record_raw_codec": (
        str,
        lambda v: v in ("mp4v", "ffv1"),
        "mp4v",
    ),
    "record_truth_events": (bool, _is_bool, True),
    "record_truth_marker": (
        str,
        lambda v: _is_marker_list(v),
        "space",
    ),
    "wps_exe_path": ((str, type(None)), _is_optional_string, None),
    "powerpoint_exe_path": ((str, type(None)), _is_optional_string, None),
}


def _validate_config(cfg, defaults=None):
    """逐字段校验，错误用默认值兜底，返回修正后的 dict 和警告列表。"""
    warnings = []
    defaults = defaults or {}
    for key, (expected_type, validator, schema_default) in _CONFIG_SCHEMA.items():
        if key not in cfg:
            continue
        default = defaults.get(key, schema_default)
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
            "stability_profile": "stable",
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
            "edge_acceleration_strength": 35,
            "pinch_freeze_enabled": False,
            "pinch_freeze_grace_sec": 0.3,
            "pinch_hysteresis_enabled": False,
            "pinch_exit_hysteresis_enabled": True,
            "thumb_perp_ratio_enabled": False,
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
            "inference_max_width": 720,
            "zoom_miss_frames": 10,
            "face_detect_short": 400,
            "yolo_max_hands": 1,
            # 投机式增强层总开关：稳定档默认关闭远距增强，需要时用 long_range 档位开启。
            "adaptive_skip_enabled": False,
            "long_range_enabled": False,
            "geometric_constraint_enabled": False,
            "hand_prediction_enabled": True,
            "temporal_voter_enabled": False,
            "detection_engine": "mediapipe",
            "engine_auto_switch": False,
            "engine_auto_switch_no_hand_frames": 60,
            "engine_auto_switch_hand_frames": 30,
            "engine_auto_switch_near_bbox_ratio": 0.04,
            "engine_auto_switch_near_frames": 90,
            "engine_auto_switch_cooldown_sec": 5.0,
            "dominant_hand": "Auto",
            "hand_detection_confidence": 0.5,
            "hand_presence_confidence": 0.5,
            "hand_tracking_confidence": 0.5,
            "mode_switch_hold_sec": 1.0,
            "mode_switch_vote_ratio": 0.6,
            "mode_switch_release_sec": 0.25,
            "draw_frontality_gate": 0.65,
            "draw_record_trace": False,
            "draw_thumb_lift": False,
            "draw_two_finger_geom": False,
            "draw_vote_window_sec": 0.30,
            "draw_vote_ratio": 0.60,
            "draw_open_palm_lift_frames": 3,
            "dictation_enabled": True,
            "dictation_model_dir": "models/sense-voice",
            "dictation_language": "auto",
            "dictation_use_itn": True,
            "dictation_num_threads": 2,
            "dictation_partial_window_sec": 12.0,
            "record_raw_video": False,
            "record_raw_max_frames": 2000,
            "record_raw_max_seconds": 120.0,
            "record_raw_codec": "mp4v",
            "record_truth_events": True,
            "record_truth_marker": "space",
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
        self.default_config = self._load_published_defaults(self.default_config)
        self._dirty = False
        self._batch_depth = 0
        self._batch_snapshot = None
        self._batch_dirty_before = False
        self._save_lock = threading.Lock()
        self.last_save_error = None
        self._needs_initial_save = False
        self.config = self.load_config()
        if self._needs_initial_save:
            self.save_config()

    @staticmethod
    def _load_published_defaults(builtin_defaults):
        """Merge the bundled config template over safe built-in fallbacks."""
        defaults = copy.deepcopy(builtin_defaults)
        template_path = resource_path("config.json")
        try:
            with open(template_path, encoding="utf-8") as stream:
                published = json.load(stream)
            if not isinstance(published, dict):
                raise TypeError("config.json 顶层必须是对象")
            published_mapping = published.pop("gesture_mapping", None)
            defaults.update(published)
            if isinstance(published_mapping, dict):
                defaults["gesture_mapping"].update(published_mapping)
            defaults, warnings = _validate_config(
                defaults,
                defaults=builtin_defaults,
            )
            for warning in warnings:
                logger.warning("发布默认配置校验: %s", warning)
        except Exception as exc:
            logger.warning(
                "读取发布默认配置失败，使用内置安全默认值 (%s): %s",
                template_path,
                exc,
            )
        return defaults

    def load_config(self):
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, encoding='utf-8') as f:
                    user_config = json.load(f)
                    if not isinstance(user_config, dict):
                        raise TypeError("配置顶层必须是对象")
                    merged = copy.deepcopy(self.default_config)
                    merged.update(user_config)
                    merged_mapping = copy.deepcopy(
                        self.default_config["gesture_mapping"]
                    )
                    merged_mapping.update(user_config.get("gesture_mapping", {}))
                    merged["gesture_mapping"] = merged_mapping
                    # Schema 校验：错误字段用默认值兜底，避免误编辑导致黑屏
                    merged, warnings = _validate_config(
                        merged,
                        defaults=self.default_config,
                    )
                    for w in warnings:
                        logger.warning("配置校验: %s", w)
                    return merged
            except Exception as e:
                logger.warning("读取配置失败，使用默认配置: %s", e)
                self._backup_corrupt_config(e)
                self._needs_initial_save = True
        else:
            self._needs_initial_save = True
        return copy.deepcopy(self.default_config)

    def _backup_corrupt_config(self, read_error):
        """Preserve an unreadable config before replacing it with defaults."""
        if not os.path.exists(self.config_file):
            return None
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        backup_path = f"{self.config_file}.corrupt-{timestamp}.bak"
        try:
            shutil.copy2(self.config_file, backup_path)
        except Exception as backup_error:
            logger.error(
                "损坏配置备份失败，拒绝覆盖原文件 (%s): %s",
                self.config_file,
                backup_error,
            )
            raise ConfigSaveError(self.config_file, backup_error) from read_error
        logger.warning("损坏配置已备份到: %s", backup_path)
        return backup_path

    def save_config(self):
        if self._batch_depth:
            self._dirty = True
            return True
        if not self._do_save():
            raise ConfigSaveError(self.config_file, self.last_save_error)
        self._dirty = False
        return True

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
                self.last_save_error = None
                return True
            except Exception as e:
                self.last_save_error = e
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
        missing = object()
        old_value = self.config.get(key, missing)
        self.config[key] = value
        try:
            self.save_config()
        except ConfigSaveError:
            if not self._batch_depth:
                if old_value is missing:
                    self.config.pop(key, None)
                else:
                    self.config[key] = old_value
            raise

    def apply_stability_profile(self, profile):
        """Apply a named experience profile to the related stability switches."""
        if profile not in _STABILITY_PROFILE_PRESETS:
            profile = "stable"
        with self.batch_update():
            self.set("stability_profile", profile)
            for key, value in _STABILITY_PROFILE_PRESETS[profile].items():
                self.set(key, value)

    def stability_profile_defaults(self, profile):
        """Return the config values controlled by a stability profile."""
        if profile not in _STABILITY_PROFILE_PRESETS:
            profile = "stable"
        return dict(_STABILITY_PROFILE_PRESETS[profile])

    def get_mapping(self, gesture):
        return self.config["gesture_mapping"].get(gesture, "none")

    def set_mapping(self, gesture, action):
        missing = object()
        old_value = self.config["gesture_mapping"].get(gesture, missing)
        self.config["gesture_mapping"][gesture] = action
        try:
            self.save_config()
        except ConfigSaveError:
            if not self._batch_depth:
                if old_value is missing:
                    self.config["gesture_mapping"].pop(gesture, None)
                else:
                    self.config["gesture_mapping"][gesture] = old_value
            raise


class _BatchContext:
    def __init__(self, manager: ConfigManager):
        self._manager = manager

    def __enter__(self):
        if self._manager._batch_depth == 0:
            self._manager._batch_snapshot = copy.deepcopy(self._manager.config)
            self._manager._batch_dirty_before = self._manager._dirty
        self._manager._batch_depth += 1
        return self._manager

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._manager._batch_depth = max(0, self._manager._batch_depth - 1)
        if self._manager._batch_depth:
            return False

        snapshot = self._manager._batch_snapshot
        dirty_before = self._manager._batch_dirty_before
        self._manager._batch_snapshot = None
        self._manager._batch_dirty_before = False

        if exc_type is not None:
            self._manager.config = snapshot
            self._manager._dirty = dirty_before
            return False

        if not self._manager._dirty:
            return False

        if not self._manager._do_save():
            self._manager.config = snapshot
            self._manager._dirty = dirty_before
            raise ConfigSaveError(
                self._manager.config_file,
                self._manager.last_save_error,
            )
        self._manager._dirty = False
        return False
