import json
import logging
import os
import sys
import threading

logger = logging.getLogger(__name__)


class ConfigManager:
    def __init__(self, config_file="config.json"):
        if getattr(sys, 'frozen', False):
            base_dir = os.path.dirname(sys.executable)
        else:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        self.config_file = os.path.join(base_dir, config_file)
        self.default_config = {
            "target_app": "WPS",
            "model_type": "Heavy",
            "interaction_mode": "mouse",
            "camera_index": 0,
            "cooldown": 1.0,
            "swipe_threshold": 60,
            "mouse_sensitivity": 40,
            "pen_width": 15,
            "edge_acceleration_enabled": False,
            "edge_acceleration_strength": 30,
            "edge_y_canvas_enabled": True,
            "edge_y_canvas_deadzone_bottom": 18,
            "edge_y_canvas_deadzone_top": 10,
            "voice_assistant": "doubao",
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
        self._batch_mode = False
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
                    return merged
            except Exception as e:
                logger.warning("读取配置失败，使用默认配置: %s", e)
        return self.default_config.copy()

    def save_config(self):
        if self._batch_mode:
            self._dirty = True
            return True
        return self._do_save()

    def _do_save(self):
        with self._save_lock:
            try:
                with open(self.config_file, 'w', encoding='utf-8') as f:
                    json.dump(self.config, f, indent=4, ensure_ascii=False)
                return True
            except Exception as e:
                logger.error("保存配置失败: %s", e)
                return False

    def batch_update(self):
        return _BatchContext(self)

    def get(self, key):
        return self.config.get(key)

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
        self._manager._batch_mode = True
        self._manager._dirty = False
        return self._manager

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._manager._batch_mode = False
        if self._manager._dirty:
            self._manager._do_save()
            self._manager._dirty = False
