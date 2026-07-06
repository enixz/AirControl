"""T-FSM-GPU: 手势状态机 + GPU自适应超分测试

验证：
  1. FSM 四状态转换（IDLE→DETECTING→CONFIRMED→RELEASING）
  2. GPU 自适应超分探测
"""
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'app'))

from services.base_hand_tracker import BaseHandTracker
from services.temporal_voter import (
    RELEASE_TIMEOUT_MS,
    GestureState,
    TemporalGestureVoter,
)


class _DummyTracker(BaseHandTracker):
    engine_name = "dummy"
    def _detect(self, frame):
        return [], [], []
    def _detect_crop_zoom(self, frame, cx, cy, size):
        return [], [], []


def _make_gesture(label, score=0.9):
    return [{"ml_label": label, "score": score, "label": label, "handedness": "Right"}]


class TestFSMStates(unittest.TestCase):
    """测试手势状态机四状态转换"""

    def setUp(self):
        self.voter = TemporalGestureVoter()

    def test_initial_state_is_idle(self):
        """初始状态为 IDLE"""
        self.assertEqual(self.voter.state, GestureState.IDLE)

    def test_idle_to_detecting(self):
        """IDLE → DETECTING：得分超过预阈值"""
        # Closed_Fist enter_th=0.55, pre_th=0.55*0.7=0.385
        # 第1帧高置信度 → 进入 DETECTING
        self.voter.update(_make_gesture("Closed_Fist", 0.9), hand_width=90.0)
        self.assertEqual(self.voter.state, GestureState.DETECTING)

    def test_detecting_to_confirmed(self):
        """DETECTING → CONFIRMED：得分达到 enter_threshold"""
        # 第1帧进入 DETECTING
        self.voter.update(_make_gesture("Closed_Fist", 0.95), hand_width=90.0)
        self.assertEqual(self.voter.state, GestureState.DETECTING)
        # 第2帧进入 CONFIRMED
        result = self.voter.update(_make_gesture("Closed_Fist", 0.95), hand_width=90.0)
        self.assertEqual(self.voter.state, GestureState.CONFIRMED)
        self.assertEqual(result, "Closed_Fist")

    def test_detecting_to_idle_on_low_score(self):
        """DETECTING → IDLE：得分低于预阈值"""
        # 先进入 DETECTING
        self.voter.update(_make_gesture("Closed_Fist", 0.9), hand_width=90.0)
        self.assertEqual(self.voter.state, GestureState.DETECTING)
        # 输入低分帧
        for _ in range(10):
            self.voter.update(_make_gesture("None", 0.0), hand_width=90.0)
        self.assertEqual(self.voter.state, GestureState.IDLE)

    def test_confirmed_to_releasing(self):
        """CONFIRMED → RELEASING：得分低于 exit_threshold"""
        # 到 CONFIRMED
        for _ in range(5):
            self.voter.update(_make_gesture("Closed_Fist", 0.95), hand_width=90.0)
        self.assertEqual(self.voter.state, GestureState.CONFIRMED)
        # 需要足够多 None 帧让窗口得分降到 exit_th(0.30) 以下
        for _ in range(6):
            self.voter.update(_make_gesture("None", 0.0), hand_width=90.0)
        self.assertEqual(self.voter.state, GestureState.RELEASING)

    def test_releasing_to_confirmed_recovery(self):
        """RELEASING → CONFIRMED：得分恢复"""
        # 到 CONFIRMED
        for _ in range(5):
            self.voter.update(_make_gesture("Closed_Fist", 0.95), hand_width=90.0)
        # 下降到 RELEASING
        for _ in range(6):
            self.voter.update(_make_gesture("None", 0.0), hand_width=90.0)
        self.assertEqual(self.voter.state, GestureState.RELEASING)
        # 恢复
        for _ in range(5):
            self.voter.update(_make_gesture("Closed_Fist", 0.9), hand_width=90.0)
        self.assertEqual(self.voter.state, GestureState.CONFIRMED)

    def test_releasing_to_idle(self):
        """RELEASING → IDLE：超时后退出"""
        # 到 CONFIRMED
        for _ in range(5):
            self.voter.update(_make_gesture("Closed_Fist", 0.95), hand_width=90.0)
        # 到 RELEASING
        for _ in range(6):
            self.voter.update(_make_gesture("None", 0.0), hand_width=90.0)
        self.assertEqual(self.voter.state, GestureState.RELEASING)
        # 模拟超时：设置 _state_enter_time 为过去
        self.voter._state_enter_time -= RELEASE_TIMEOUT_MS + 100
        # 下一帧应超时退出
        self.voter.update(_make_gesture("None", 0.0), hand_width=90.0)
        self.assertEqual(self.voter.state, GestureState.IDLE)

    def test_releasing_outputs_active_gesture(self):
        """RELEASING 状态仍输出当前手势（短暂抖动不中断）"""
        # 到 CONFIRMED
        for _ in range(5):
            self.voter.update(_make_gesture("Closed_Fist", 0.95), hand_width=90.0)
        # 进入 RELEASING（需要足够帧让得分降下来）
        for _ in range(6):
            result = self.voter.update(_make_gesture("None", 0.0), hand_width=90.0)
        self.assertEqual(self.voter.state, GestureState.RELEASING)
        self.assertEqual(result, "Closed_Fist")

    def test_reset_clears_state(self):
        """reset 清除状态"""
        for _ in range(15):
            self.voter.update(_make_gesture("Closed_Fist", 0.95), hand_width=90.0)
        self.voter.reset()
        self.assertEqual(self.voter.state, GestureState.IDLE)
        self.assertIsNone(self.voter._active_gesture)


class TestGPUAdaptiveSR(unittest.TestCase):
    """测试 GPU 自适应超分"""

    def setUp(self):
        self.tracker = _DummyTracker(config={})

    def test_detect_gpu_no_cuda(self):
        """无 CUDA 时 _realesrgan_gpu_available = False"""
        with patch('onnxruntime.get_available_providers', return_value=['CPUExecutionProvider']):
            self.tracker._sr._detect_gpu_availability()
        self.assertFalse(self.tracker._sr._realesrgan_gpu_available)

    def test_detect_gpu_with_cuda_no_model(self):
        """有 CUDA 但模型文件缺失 → False"""
        with patch('onnxruntime.get_available_providers', return_value=['CUDAExecutionProvider', 'CPUExecutionProvider']):
            with patch('os.path.exists', return_value=False):
                self.tracker._sr._detect_gpu_availability()
        self.assertFalse(self.tracker._sr._realesrgan_gpu_available)

    def test_detect_gpu_with_cuda_and_model(self):
        """有 CUDA 且模型存在 → True，provider=CUDA"""
        # 直接设置模型路径为存在的文件（本测试文件）
        self.tracker._sr._realesrgan_path = os.path.abspath(__file__)
        with patch('onnxruntime.get_available_providers', return_value=['CUDAExecutionProvider', 'CPUExecutionProvider']):
            self.tracker._sr._detect_gpu_availability()
        self.assertTrue(self.tracker._sr._realesrgan_gpu_available)
        self.assertEqual(self.tracker._sr._realesrgan_gpu_provider, 'CUDAExecutionProvider')

    def test_detect_gpu_with_directml_and_model(self):
        """无 CUDA 但有 DirectML 且模型存在 → True，provider=DirectML"""
        self.tracker._sr._realesrgan_path = os.path.abspath(__file__)
        with patch('onnxruntime.get_available_providers', return_value=['DmlExecutionProvider', 'CPUExecutionProvider']):
            self.tracker._sr._detect_gpu_availability()
        self.assertTrue(self.tracker._sr._realesrgan_gpu_available)
        self.assertEqual(self.tracker._sr._realesrgan_gpu_provider, 'DmlExecutionProvider')

    def test_detect_gpu_cuda_preferred_over_directml(self):
        """CUDA 和 DirectML 都可用时优先 CUDA"""
        self.tracker._sr._realesrgan_path = os.path.abspath(__file__)
        with patch('onnxruntime.get_available_providers', return_value=['CUDAExecutionProvider', 'DmlExecutionProvider', 'CPUExecutionProvider']):
            self.tracker._sr._detect_gpu_availability()
        self.assertEqual(self.tracker._sr._realesrgan_gpu_provider, 'CUDAExecutionProvider')

    def test_auto_uses_gpu_when_available(self):
        """auto 模式 + GPU 可用 → realesrgan_gpu"""
        self.tracker._sr._realesrgan_gpu_available = True
        self.tracker._sr._auto_sr_enabled = True  # 需要超分
        engine = self.tracker._sr.resolve("auto", crop_size=200, target=384)
        self.assertEqual(engine, "realesrgan_gpu")

    def test_auto_uses_espcn_when_no_gpu(self):
        """auto 模式 + 无 GPU → espcn"""
        self.tracker._sr._realesrgan_gpu_available = False
        self.tracker._sr._auto_sr_enabled = True
        engine = self.tracker._sr.resolve("auto", crop_size=200, target=384)
        self.assertEqual(engine, "espcn")

    def test_auto_none_when_no_upscale_needed(self):
        """auto 模式 + 不需要超分 → none"""
        self.tracker._sr._realesrgan_gpu_available = True
        self.tracker._sr._auto_sr_enabled = False  # 不需要超分
        engine = self.tracker._sr.resolve("auto", crop_size=500, target=384)
        self.assertEqual(engine, "none")

    def test_explicit_engine_not_overridden(self):
        """显式指定的引擎不被 auto 逻辑覆盖"""
        self.tracker._sr._realesrgan_gpu_available = True
        engine = self.tracker._sr.resolve("espcn", crop_size=200, target=384)
        self.assertEqual(engine, "espcn")


if __name__ == "__main__":
    unittest.main(verbosity=2)
