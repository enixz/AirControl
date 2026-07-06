"""T-Cam-Fallback: 摄像头索引自适应选择与后端回退测试

验证：
  1. 启动前探测可用摄像头索引
  2. 偏好索引不可用则回退到第一个可用索引
  3. DSHOW 后端打不开时回退到 OpenCV 默认后端
"""
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'app'))

import services.camera


def _find_available_cameras(exclude_index=None):
    """从 orchestrator.py 复制的探测逻辑，供测试。"""
    try:
        return services.camera.list_available_cameras(
            max_probe=4, exclude_index=exclude_index
        )
    except Exception:
        import logging
        logging.getLogger(__name__).exception("枚举摄像头失败")
        return []


def _choose_camera_index(requested_index, available):
    """从 orchestrator.py init_services 中提取的自适应选择逻辑，供测试。"""
    if available:
        available_indices = {c["index"] for c in available}
        if requested_index in available_indices:
            return requested_index
        return available[0]["index"]
    return requested_index


class TestFindAvailableCameras(unittest.TestCase):
    """测试 _find_available_cameras"""

    def test_returns_available_list(self):
        """返回可用摄像头列表"""
        with patch('services.camera.list_available_cameras', return_value=[
            {"index": 0, "name": "摄像头 0"},
            {"index": 2, "name": "摄像头 2"},
        ]):
            result = _find_available_cameras(exclude_index=1)
            self.assertEqual(len(result), 2)
            self.assertEqual(result[0]["index"], 0)

    def test_returns_empty_when_no_camera(self):
        """没有可用摄像头时返回空列表"""
        with patch('services.camera.list_available_cameras', return_value=[]):
            result = _find_available_cameras(exclude_index=1)
            self.assertEqual(result, [])

    def test_exception_returns_empty(self):
        """枚举异常时返回空列表不崩溃"""
        with patch('services.camera.list_available_cameras', side_effect=RuntimeError("boom")):
            result = _find_available_cameras(exclude_index=1)
            self.assertEqual(result, [])


class TestChooseCameraIndex(unittest.TestCase):
    """测试自适应选择逻辑"""

    def test_use_requested_when_available(self):
        """偏好索引可用时直接使用"""
        available = [{"index": 0}, {"index": 1}]
        idx = _choose_camera_index(1, available)
        self.assertEqual(idx, 1)

    def test_fallback_to_first_when_requested_unavailable(self):
        """偏好索引不可用时回退到第一个可用索引"""
        available = [{"index": 0}]
        idx = _choose_camera_index(1, available)
        self.assertEqual(idx, 0)

    def test_use_requested_when_no_camera_found(self):
        """探测失败时仍尝试使用偏好索引"""
        idx = _choose_camera_index(1, [])
        self.assertEqual(idx, 1)

    def test_config_index_zero_fallback_works(self):
        """config 为 0 但只有 1 时回退到 1"""
        available = [{"index": 1}, {"index": 2}]
        idx = _choose_camera_index(0, available)
        self.assertEqual(idx, 1)


class TestBackendFallback(unittest.TestCase):
    """测试 DSHOW 失败时回退到默认后端"""

    @patch('services.camera.cv2.VideoCapture')
    def test_dshow_falls_back_to_default_backend(self, mock_vc):
        """DSHOW 打不开时，list_available_cameras 会尝试 MSMF / 默认后端"""
        def side_effect(index, backend=None):
            cap = MagicMock()
            # DSHOW 后端失败；MSMF 后端失败；默认后端（backend=None）成功
            if backend == services.camera.cv2.CAP_DSHOW:
                cap.isOpened.return_value = False
            elif backend == getattr(services.camera.cv2, "CAP_MSMF", object()):
                cap.isOpened.return_value = False
            else:
                cap.isOpened.return_value = True
                cap.read.return_value = (True, None)
            return cap

        mock_vc.side_effect = side_effect

        result = services.camera.list_available_cameras(max_probe=1)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["index"], 0)
        self.assertIn("backend", result[0])
        # 每个索引会被调用 3 次：DSHOW + MSMF + 默认后端
        self.assertEqual(mock_vc.call_count, 3)

    @patch('services.camera.cv2.VideoCapture')
    def test_default_backend_also_fails_returns_empty(self, mock_vc):
        """DSHOW / MSMF / 默认后端 都失败时返回空"""
        cap = MagicMock()
        cap.isOpened.return_value = False
        mock_vc.return_value = cap

        result = services.camera.list_available_cameras(max_probe=1)

        self.assertEqual(result, [])
        # 每个索引 3 次尝试
        self.assertEqual(mock_vc.call_count, 3)

    @patch('services.camera.cv2.VideoCapture')
    def test_msmf_backend_used_when_dshow_fails(self, mock_vc):
        """DSHOW 失败但 MSMF 成功时，结果记录 MSMF 后端"""
        def side_effect(index, backend=None):
            cap = MagicMock()
            if backend == services.camera.cv2.CAP_DSHOW:
                cap.isOpened.return_value = False
            elif backend == getattr(services.camera.cv2, "CAP_MSMF", object()):
                cap.isOpened.return_value = True
                cap.read.return_value = (True, None)
            else:
                cap.isOpened.return_value = True
                cap.read.return_value = (True, None)
            return cap

        mock_vc.side_effect = side_effect

        result = services.camera.list_available_cameras(max_probe=1)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["index"], 0)
        self.assertEqual(result[0]["backend"], services.camera.cv2.CAP_MSMF)


if __name__ == "__main__":
    unittest.main(verbosity=2)
