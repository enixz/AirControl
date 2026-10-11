"""
T4: Zoom Super-Resolution Configuration and Resolution Tests
Validates schema validation, defaults, and resolution logic for the Zoom SR engines.
"""
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

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

    def _make_tracker(self, gpu_available=False):
        """Create an auto-SR tracker without consulting host files/providers."""
        config = ConfigManager(config_file=self.config_path)
        tracker = DummyHandTracker(config=config)
        tracker._sr._realesrgan_gpu_available = bool(gpu_available)
        tracker._sr._realesrgan_gpu_provider = (
            "DmlExecutionProvider" if gpu_available else None
        )
        return tracker

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

        tracker = self._make_tracker()

        # 初始时，未初始化
        self.assertFalse(tracker._sr._sr_initialized)

        # 调用 init 后应处于初始化完毕状态
        tracker._sr.init()
        self.assertTrue(tracker._sr._sr_initialized)
        self.assertIsNone(tracker._sr._espcn_engine)
        self.assertIsNone(tracker._sr._realesrgan_cpu_session)
        self.assertIsNone(tracker._sr._realesrgan_gpu_session)

    def test_auto_enables_espcn_when_upscaling(self):
        """auto：crop < 192（检测器内部输入尺寸，真·远距小手）时启用 ESPCN。

        2026-10-07 P0 实测：crop ≥ 192 区间 SR 无益（内部重采样抵消一切放大），
        SR 介入阈值从 target*0.9(≈346) 收紧到 192。
        """
        tracker = self._make_tracker()
        target = tracker._crop_target_size  # 384
        self.assertEqual(tracker._sr.resolve("auto", 191, target), "espcn")
        self.assertEqual(tracker._sr.resolve("auto", 64, target), "espcn")
        self.assertEqual(tracker._sr.resolve("auto", 33, target), "espcn")

    def test_auto_skips_sr_between_192_and_target(self):
        """auto：crop 在 [192, target) 区间不再超分（改走直通/插值）。

        依据：MediaPipe 掌检测器内部 192×192，此区间放大被内部重采样抵消。
        """
        tracker = self._make_tracker()
        target = tracker._crop_target_size
        for cs in (192, 240, target - 1, target, target + 200):
            self.assertEqual(tracker._sr.resolve("auto", cs, target), "none")

    def test_auto_enables_espcn_for_tight_crops(self):
        """紧凑裁剪框（远小于 192）仍应启用 ESPCN。

        裁剪框可一路缩小到机械下限 _crop_min_size，超分一路放大，
        因此 64 这种小裁剪框也必须走 espcn 而非 none。
        """
        tracker = self._make_tracker()
        target = tracker._crop_target_size
        self.assertLess(tracker._crop_min_size, 192)
        for cs in (tracker._crop_min_size, 64, 120, 172):
            self.assertEqual(tracker._sr.resolve("auto", cs, target), "espcn")

    def test_auto_disables_sr_when_downscaling(self):
        """auto：crop >= target（下采样/手较近）时关闭超分。

        超分对"下采样"加不了任何细节，还白白吃 CPU（实测每段 ZOOM 的 crop 常达
        1000~1300px 远大于 384px 目标）。crop_size >= 192 即返回 none，
        只有真·远距小手（< 192px，上采样放大）才用 ESPCN。
        """
        tracker = self._make_tracker()
        target = tracker._crop_target_size
        # 下采样（手较近）→ 关闭超分
        self.assertEqual(tracker._sr.resolve("auto", target, target), "none")
        self.assertEqual(tracker._sr.resolve("auto", target + 200, target), "none")
        # 从关闭状态重新进入需要越过滞回下沿，避免在 192 附近抖动。
        self.assertEqual(
            tracker._sr.resolve("auto", int(192 * 0.89), target),
            "espcn",
        )

    def test_auto_prefers_realesrgan_when_gpu_model_is_available(self):
        tracker = self._make_tracker(gpu_available=True)
        target = tracker._crop_target_size

        self.assertEqual(
            tracker._sr.resolve("auto", 191, target),
            "realesrgan_gpu",
        )

    def test_explicit_engine_is_respected(self):
        """显式选择的引擎不被 auto 逻辑改写（无论放大还是缩小）。"""
        cm = ConfigManager(config_file=self.config_path)
        tracker = DummyHandTracker(config=cm)
        target = tracker._crop_target_size
        for eng in ("espcn", "realesrgan_cpu", "realesrgan_gpu", "none"):
            self.assertEqual(tracker._sr.resolve(eng, 240, target), eng)
            self.assertEqual(tracker._sr.resolve(eng, target + 100, target), eng)

    def test_auto_uses_hysteresis_near_target(self):
        tracker = self._make_tracker()
        target = tracker._crop_target_size

        # 全新状态（reset_tier 后）：192 边界 → 下方开启、上方关闭
        self.assertEqual(
            tracker._sr.resolve("auto", 191, target), "espcn"
        )
        tracker._sr.reset_tier()
        self.assertEqual(
            tracker._sr.resolve("auto", 192, target), "none"
        )
        # 已开启状态下，越过滞回上沿（192*1.1≈211）才关闭
        tracker._sr.reset_tier()
        self.assertEqual(
            tracker._sr.resolve("auto", int(192 * 0.89), target), "espcn"
        )
        self.assertEqual(
            tracker._sr.resolve("auto", 200, target), "espcn"
        )
        self.assertEqual(
            tracker._sr.resolve("auto", int(192 * 1.11), target),
            "none",
        )
        # 已关闭状态下，低于滞回下沿（192*0.9≈173）才重新开启
        self.assertEqual(
            tracker._sr.resolve("auto", 191, target), "none"
        )
        self.assertEqual(
            tracker._sr.resolve("auto", int(192 * 0.89), target),
            "espcn",
        )

    @patch('os.path.exists')
    def test_sr_helpers_fallback_gracefully_when_models_missing(self, mock_exists):
        """模型文件缺失时 SR 助手返回 None（上层回退到普通插值），且不抛异常。"""
        import numpy as np
        mock_exists.return_value = False

        cm = ConfigManager(config_file=self.config_path)
        tracker = DummyHandTracker(config=cm)
        tracker._sr.init()

        dummy = np.zeros((240, 240, 3), dtype=np.uint8)
        self.assertIsNone(tracker._sr.espcn(dummy, 384))
        self.assertIsNone(tracker._sr.realesrgan(dummy, 384, prefer_gpu=True))
        self.assertIsNone(tracker._sr.realesrgan(dummy, 384, prefer_gpu=False))

    def _make_frame(self, size=480):
        """全零帧（crop-zoom 对纯色不敏感，Dummy 子类直接返回空）。
        返回 (frame, None)；绘制留给调用方，便于直接断言喂给子类的图。"""
        import numpy as np
        return np.zeros((size, size, 3), dtype=np.uint8)

    @patch('os.path.exists')
    def test_direct_feed_skips_resize_for_large_crops(self, mock_exists):
        """crop ≥ 192（auto 判 none）时直通：子类收到的就是原 crop 本身。

        依据：MediaPipe/YOLO 内部会再重采样（192/640），先行插值放大是双重
        劳动。直通时 fed_size == crop_size（scale=1），坐标映射公式不变。
        """
        import numpy as np
        mock_exists.return_value = False
        cm = ConfigManager(config_file=self.config_path)
        tracker = DummyHandTracker(config=cm)

        frame = np.zeros((480, 480, 3), dtype=np.uint8)
        frame[60:300, 60:300] = 200  # crop 中心 (180,180)
        fed = []

        def run_sub_detect(zoomed):
            fed.append(zoomed)
            return ([], [], [])

        tracker._perform_crop_zoom(frame, (180, 180), 240, run_sub_detect)

        self.assertEqual(len(fed), 1)
        # 直通：无 resize，形状是 crop_size 而非 target 384
        self.assertEqual(fed[0].shape, (240, 240, 3))
        # fed 是 crop 本身（x0 = 180-120 = 60），证明未做任何重采样
        np.testing.assert_array_equal(fed[0], frame[60:300, 60:300])

    @patch('os.path.exists')
    def test_direct_feed_reports_tier_and_scale(self, mock_exists):
        """直通档位记为 none(direct)；viewport fed_size=crop_size、scale=1，
        归一化坐标映射回 crop 中心。"""
        import numpy as np
        mock_exists.return_value = False
        cm = ConfigManager(config_file=self.config_path)
        tracker = DummyHandTracker(config=cm)

        frame = np.zeros((480, 480, 3), dtype=np.uint8)
        frame[0:240, 0:240] = 127
        tiers = []
        orig_log_tier = tracker._sr.log_tier

        def spy_tier(tier, crop_size, target):
            tiers.append(tier)
            orig_log_tier(tier, crop_size, target)

        tracker._sr.log_tier = spy_tier

        def run_sub_detect(zoomed):
            return ([], [], {"fed_shape": zoomed.shape})

        detection_result, viewport, to_orig = tracker._perform_crop_zoom(
            frame, (120, 120), 240, run_sub_detect
        )
        self.assertIn("none(direct)", tiers)
        # viewport: (x0, y0, crop_size, fed_size, scale)；直通时 fed=crop、scale=1
        self.assertEqual(viewport[3], 240)
        self.assertAlmostEqual(viewport[4], 1.0)
        # fed 形状 = crop_size（240）而非 target 384，证明直通未重采样
        self.assertEqual(detection_result[2]["fed_shape"], (240, 240, 3))
        # 坐标映射：归一化中心 (0.5, 0.5) → crop 中心 (120, 120)
        cx, cy = to_orig(0.5, 0.5)
        self.assertAlmostEqual(cx, 120.0)
        self.assertAlmostEqual(cy, 120.0)

    @patch('os.path.exists')
    def test_small_crop_still_interpolates(self, mock_exists):
        """crop < 192 → auto 判 espcn；模型缺失回退普通插值（fed_size=target）。"""
        import numpy as np
        mock_exists.return_value = False
        cm = ConfigManager(config_file=self.config_path)
        tracker = DummyHandTracker(config=cm)

        frame = np.zeros((480, 480, 3), dtype=np.uint8)
        frame[0:64, 0:64] = 127
        fed = []

        def run_sub_detect(zoomed):
            fed.append(zoomed)
            return ([], [], [])

        tracker._perform_crop_zoom(frame, (32, 32), 64, run_sub_detect)

        self.assertEqual(len(fed), 1)
        # 小 crop：插值放大到 target 384
        self.assertEqual(fed[0].shape[0], tracker._crop_target_size)


if __name__ == "__main__":
    unittest.main(verbosity=2)
