import os
import sys
import time
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

import numpy as np
from services.camera import _RESOLUTION_CANDIDATES, probe_max_resolution


class TestCameraResolutionCandidates(unittest.TestCase):
    def test_automatic_probe_is_capped_at_1080p(self):
        self.assertLessEqual(max(height for _, height in _RESOLUTION_CANDIDATES), 1080)


class TestMjpgReapplyOnResolutionChange(unittest.TestCase):
    """回归：部分 Windows 驱动在 width/height 改变后会重置 FOURCC 回 YUY2，
    导致 1080p/720p 因 YUY2 带宽不足达不到 min_fps、最终回退到 480p。

    旧实现只在循环外设置一次 MJPG；新实现必须在每档分辨率变更后再补一次。
    此测试锁死该回归：探测到 1080p 时 MJPG 至少被设置 2 次（外 1 + 内 1）。
    """

    @patch("services.camera.cv2.VideoCapture")
    def test_probe_reapplies_mjpg_after_resolution_change(self, mock_vc):
        cap = MagicMock()
        cap.isOpened.return_value = True

        # 驱动接受所有候选分辨率（set 完后 get 回报相同值）
        state = {"w": 1920, "h": 1080}

        # 用真实 cv2 常量做分发，避免 MagicMock 比较问题
        import cv2

        def get_side_effect(prop):
            if prop == cv2.CAP_PROP_FRAME_WIDTH:
                return state["w"]
            if prop == cv2.CAP_PROP_FRAME_HEIGHT:
                return state["h"]
            if prop == cv2.CAP_PROP_FPS:
                return 30
            if prop == cv2.CAP_PROP_FOURCC:
                # 模拟"驱动复位 FOURCC"——每次 get 都报 YUY2，
                # 强制依赖 set 后立即 get 的旧测试会发现 FOURCC 不持久。
                # 但 probe 不校验 FOURCC，只校验 w/h，所以这里给什么都行。
                return 0x32595559  # 'YUY2'
            return 0

        def set_side_effect(prop, val):
            if prop == cv2.CAP_PROP_FRAME_WIDTH:
                state["w"] = val
            elif prop == cv2.CAP_PROP_FRAME_HEIGHT:
                state["h"] = val
            return True

        cap.get.side_effect = get_side_effect
        cap.set.side_effect = set_side_effect
        # read 15 帧全部成功 → real_fps 远超 min_fps，1080p 第一档即命中并 break。
        # 用 side_effect 而非 return_value，并 sleep 极小一段时间，确保
        # time.time() - t0 不为 0（否则 real_fps=0 会让所有分辨率都 fail fps）。
        def read_side_effect():
            time.sleep(0.001)
            return (True, np.zeros((1080, 1920, 3), dtype=np.uint8))
        cap.read.side_effect = read_side_effect
        mock_vc.return_value = cap

        result = probe_max_resolution(
            0, min_fps=20, force_mjpeg=True, use_cache=False
        )
        self.assertEqual(result, (1920, 1080))

        # 统计 CAP_PROP_FOURCC 的 set 调用次数
        fourcc_set_calls = [
            c for c in cap.set.call_args_list if c.args[0] == cv2.CAP_PROP_FOURCC
        ]
        # 旧代码：只在循环外设 1 次。新代码：循环外 1 + 循环内（1080p 命中）1 = 至少 2 次。
        # 若未来驱动复位行为更严重（每档都要重设），次数会更多——这里只锁下限。
        self.assertGreaterEqual(
            len(fourcc_set_calls),
            2,
            f"MJPG 应在每档分辨率变更后被重新设置，实际只调用了 "
            f"{len(fourcc_set_calls)} 次（旧 bug：仅在循环外设置 1 次）",
        )


if __name__ == "__main__":
    unittest.main()
