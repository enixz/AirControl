"""
T4: Zoom Super-Resolution Configuration and Resolution Tests
Validates schema validation, defaults, and resolution logic for the Zoom SR engines.
"""
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'app'))

from config_manager import ConfigManager
from services.base_hand_tracker import BaseHandTracker


class DummyHandTracker(BaseHandTracker):
    """用于测试 BaseHandTracker 逻辑的哑实现。"""
    @property
    def engine_name(self) -> str:
        return "dummy"

    def _detect(self, frame):
        return [], [], []

    def _detect_crop_zoom(self, frame, hint_center, hint_size):
        return [], [], []


class TestZoomSR(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.tmpdir, 'config.json')

    def tearDown(self):
        if os.path.exists(self.config_path):
            os.remove(self.config_path)
        os.rmdir(self.tmpdir)

    def test_missing_sr_key_uses_default(self):
        """若 config.json 缺少 zoom_sr_engine，默认使用 auto"""
        old_config = {
            "target_app": "PowerPoint"
        }
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(old_config, f)

        cm = ConfigManager(config_file=self.config_path)
        self.assertIn("zoom_sr_engine", cm.config)
        self.assertEqual(cm.get("zoom_sr_engine"), "auto")

    def test_invalid_sr_key_resets_to_default(self):
        """若 config.json 中 zoom_sr_engine 范围无效，回退为 auto"""
        bad_config = {
            "zoom_sr_engine": "invalid_engine"
        }
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(bad_config, f)

        cm = ConfigManager(config_file=self.config_path)
        self.assertEqual(cm.get("zoom_sr_engine"), "auto")

    def test_valid_sr_engines(self):
        """支持的几个超分选项能够被正常读取"""
        for engine in ("auto", "espcn", "realesrgan_cpu", "realesrgan_gpu", "none"):
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump({"zoom_sr_engine": engine}, f)
            cm = ConfigManager(config_file=self.config_path)
            self.assertEqual(cm.get("zoom_sr_engine"), engine)

    @patch('os.path.exists')
    def test_lazy_loading_of_engines(self, mock_exists):
        """测试 lazy loading 能够检测模型文件的存在性"""
        # 强制使得模型文件不存，保证加载报错被捕获/优雅处理
        mock_exists.return_value = False

        cm = ConfigManager(config_file=self.config_path)
        tracker = DummyHandTracker(config=cm)

        # 初始时，未初始化
        self.assertFalse(hasattr(tracker, "_sr_initialized"))

        # 调用 _init_sr_engines 后应处于初始化完毕状态
        tracker._init_sr_engines()
        self.assertTrue(tracker._sr_initialized)
        self.assertIsNone(tracker._espcn_engine)
        self.assertIsNone(tracker._realesrgan_cpu_session)
        self.assertIsNone(tracker._realesrgan_gpu_session)

    def test_auto_enables_espcn_when_upscaling(self):
        """回归：auto 模式在裁剪框 < 目标尺寸（远距离放大）时必须启用 ESPCN。

        旧 bug：crop_size 在调用处被钳制到 >=240，而 auto 门控却用 `crop_size < 160`，
        两者互斥导致超分永远不触发（死代码）。此测试锁死该回归。
        """
        cm = ConfigManager(config_file=self.config_path)
        tracker = DummyHandTracker(config=cm)
        target = tracker._crop_target_size  # 384
        # 远距离：裁剪框被钳到下限 240 < 384 → 在放大 → 必须启用 espcn（而非 none）
        self.assertEqual(tracker._resolve_sr_engine("auto", 240, target), "espcn")
        self.assertEqual(tracker._resolve_sr_engine("auto", target - 1, target), "espcn")

    def test_auto_enables_espcn_for_tight_crops(self):
        """去掉 240 人为下限后：远距离的紧凑裁剪框（远小于 240）仍应启用 ESPCN。

        裁剪框可一路缩小到机械下限 _crop_min_size，超分一路放大，
        因此 64 这种小裁剪框也必须走 espcn 而非 none。
        """
        cm = ConfigManager(config_file=self.config_path)
        tracker = DummyHandTracker(config=cm)
        target = tracker._crop_target_size
        self.assertLess(tracker._crop_min_size, 240)  # 不再有 240 人为下限
        for cs in (tracker._crop_min_size, 64, 120, 239):
            self.assertEqual(tracker._resolve_sr_engine("auto", cs, target), "espcn")

    def test_auto_disables_sr_when_downscaling(self):
        """近距离关闭 SR：auto 模式下裁剪框 >= 目标尺寸（下采样/手较近）时退回普通插值。

        超分对"下采样"加不了任何细节，还白白吃 CPU（实测每段 ZOOM 的 crop 常达
        1000~1300px 远大于 384px 目标）。因此 crop_size >= target 时 auto 返回 none，
        把算力还给帧率；只有 crop_size < target（上采样放大小/远手）才用 ESPCN。
        """
        cm = ConfigManager(config_file=self.config_path)
        tracker = DummyHandTracker(config=cm)
        target = tracker._crop_target_size
        # 下采样（手较近）→ 关闭超分
        self.assertEqual(tracker._resolve_sr_engine("auto", target, target), "none")
        self.assertEqual(tracker._resolve_sr_engine("auto", target + 200, target), "none")
        # 上采样（手小/远）→ 仍用 ESPCN
        self.assertEqual(tracker._resolve_sr_engine("auto", target - 1, target), "espcn")

    def test_explicit_engine_is_respected(self):
        """显式选择的引擎不被 auto 逻辑改写（无论放大还是缩小）。"""
        cm = ConfigManager(config_file=self.config_path)
        tracker = DummyHandTracker(config=cm)
        target = tracker._crop_target_size
        for eng in ("espcn", "realesrgan_cpu", "realesrgan_gpu", "none"):
            self.assertEqual(tracker._resolve_sr_engine(eng, 240, target), eng)
            self.assertEqual(tracker._resolve_sr_engine(eng, target + 100, target), eng)

    @patch('os.path.exists')
    def test_sr_helpers_fallback_gracefully_when_models_missing(self, mock_exists):
        """模型文件缺失时 SR 助手返回 None（上层回退到普通插值），且不抛异常。"""
        import numpy as np
        mock_exists.return_value = False

        cm = ConfigManager(config_file=self.config_path)
        tracker = DummyHandTracker(config=cm)
        tracker._init_sr_engines()

        dummy = np.zeros((240, 240, 3), dtype=np.uint8)
        self.assertIsNone(tracker._sr_espcn(dummy, 384))
        self.assertIsNone(tracker._sr_realesrgan(dummy, 384, prefer_gpu=True))
        self.assertIsNone(tracker._sr_realesrgan(dummy, 384, prefer_gpu=False))


if __name__ == "__main__":
    unittest.main(verbosity=2)
