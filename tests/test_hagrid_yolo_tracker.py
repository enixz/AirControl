"""HaGRID YOLO 后处理的轻量单元测试（不加载原生模型）。"""
import os
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app"))

from services.hagrid_yolo_hand_tracker import HagridYoloHandTracker
from services.hand_tracker_factory import create_hand_tracker

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HAS_HYBRID_MODELS = all(
    (PROJECT_ROOT / "models" / name).is_file()
    for name in ("hand_yolov8n.onnx", "hand_landmarker.task")
)


class TestHagridYoloPostprocess(unittest.TestCase):
    def _bare_tracker(self, max_num_hands=2):
        tracker = object.__new__(HagridYoloHandTracker)
        tracker._yolo_conf = 0.25
        tracker.max_num_hands = max_num_hands
        return tracker

    def test_nms_suppresses_overlapping_center_xywh_candidates(self):
        """YOLO 的中心 xywh 必须先转成左上角 xywh 再交给 OpenCV NMS。"""
        tracker = self._bare_tracker()
        # 正常 YOLOv8 格式 [1, 5, A]：前两框高度重叠，第三框独立。
        # A 取 7，避免被误识别为端到端模型的 [1, N, 6] 输出。
        output = np.array([[
            [100.0, 104.0, 300.0, 20.0, 20.0, 20.0, 20.0],  # cx
            [100.0, 104.0, 300.0, 20.0, 20.0, 20.0, 20.0],  # cy
            [100.0, 100.0, 30.0, 10.0, 10.0, 10.0, 10.0],  # w
            [100.0, 100.0, 30.0, 10.0, 10.0, 10.0, 10.0],  # h
            [0.90, 0.80, 0.95, 0.10, 0.10, 0.10, 0.10],    # score
        ]], dtype=np.float32)

        boxes = tracker._parse_yolo_output([output], 1.0, 0, 0, 640, 640)

        self.assertEqual(len(boxes), 2)
        scores = [box[4] for box in boxes]
        self.assertTrue(any(abs(score - 0.95) < 1e-6 for score in scores))
        self.assertTrue(any(abs(score - 0.90) < 1e-6 for score in scores))
        self.assertFalse(any(abs(score - 0.80) < 1e-6 for score in scores))

    def test_respects_single_hand_capture_limit(self):
        tracker = self._bare_tracker(max_num_hands=1)
        output = np.array([[
            [100.0, 300.0, 20.0, 20.0, 20.0, 20.0, 20.0],
            [100.0, 300.0, 20.0, 20.0, 20.0, 20.0, 20.0],
            [80.0, 80.0, 10.0, 10.0, 10.0, 10.0, 10.0],
            [80.0, 80.0, 10.0, 10.0, 10.0, 10.0, 10.0],
            [0.80, 0.95, 0.10, 0.10, 0.10, 0.10, 0.10],
        ]], dtype=np.float32)

        boxes = tracker._parse_yolo_output([output], 1.0, 0, 0, 640, 640)

        self.assertEqual(len(boxes), 1)
        self.assertAlmostEqual(boxes[0][4], 0.95)


@unittest.skipUnless(HAS_HYBRID_MODELS, "requires bundled YOLO and HandLandmarker models")
class TestHagridYoloRealIntegration(unittest.TestCase):
    """真实 ONNX + HandLandmarker 引擎冒烟测试，防止仅 mock 的回归漏网。"""

    def test_factory_runs_real_hybrid_engine_on_blank_frame(self):
        tracker = create_hand_tracker(
            engine="hagrid_yolo",
            max_num_hands=1,
            config={"yolo_confidence": 0.25, "yolo_max_hands": 1},
        )
        try:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            output_frame, landmarks, gestures = tracker.find_hands(frame, draw=False)

            self.assertEqual(tracker.engine_name, "hagrid_yolo")
            self.assertEqual(output_frame.shape, frame.shape)
            self.assertIsInstance(landmarks, list)
            self.assertIsInstance(gestures, list)
            self.assertLessEqual(len(landmarks), 1)
            self.assertEqual(len(landmarks), len(gestures))
        finally:
            tracker.close()
            self.assertIsNone(tracker._yolo_session)
            self.assertIsNone(tracker._landmarker)


if __name__ == "__main__":
    unittest.main()
