"""person_pose_hand 引擎的轻量单元测试（解析/框手逻辑不加载原生模型）。

真实模型冒烟测试标记 skipUnless，仅在 models/yolov8n-pose.onnx 等
三个模型齐备时运行（该 ONNX 不入 git，见 MODEL_PROVENANCE.md）。
"""
import os
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app"))

from services.person_pose_hand_tracker import (
    PersonPoseHandTracker,
    _KP_RIGHT_WRIST,
    _KP_RIGHT_ELBOW,
)
from services.hand_tracker_factory import create_hand_tracker

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HAS_MODELS = all(
    (PROJECT_ROOT / "models" / name).is_file()
    for name in ("yolov8n-pose.onnx", "hand_yolov8n.onnx", "hand_landmarker.task")
)


def _make_pose_output(persons):
    """构造 yolov8-pose 输出 [1, 56, N]。

    persons: list of dict(cx, cy, w, h, conf, kps={kp_index:(x,y,c)})
    N 取 8，避免被 NMS 等误判；未填充的列 conf=0 会被阈值滤掉。
    """
    n = max(8, len(persons) + 1)
    out = np.zeros((1, 56, n), dtype=np.float32)
    for col, p in enumerate(persons):
        out[0, 0, col] = p["cx"]; out[0, 1, col] = p["cy"]
        out[0, 2, col] = p["w"];  out[0, 3, col] = p["h"]
        out[0, 4, col] = p.get("conf", 0.9)
        for kp_idx, (x, y, c) in p.get("kps", {}).items():
            base = 5 + 3 * kp_idx
            out[0, base, col] = x
            out[0, base + 1, col] = y
            out[0, base + 2, col] = c
    return out


class TestPoseOutputParsing(unittest.TestCase):
    def _bare(self, max_num_hands=2):
        t = object.__new__(PersonPoseHandTracker)
        t._pose_conf = 0.25
        t.max_num_hands = max_num_hands
        t._sr_enabled = False  # 解析/框手测试不触碰 SREngine
        t._sr_trigger = 96
        return t

    def test_parses_single_person_keypoints(self):
        t = self._bare()
        out = _make_pose_output([{
            "cx": 320, "cy": 240, "w": 100, "h": 200, "conf": 0.9,
            "kps": {_KP_RIGHT_WRIST: (360, 300, 0.95), _KP_RIGHT_ELBOW: (330, 260, 0.9)},
        }])
        persons = t._parse_pose_output([out], 1.0, 0, 0, 640, 480)
        self.assertEqual(len(persons), 1)
        body_box, kps = persons[0]
        self.assertEqual(body_box[:4], (270, 140, 370, 340))
        self.assertAlmostEqual(kps[_KP_RIGHT_WRIST][0], 360)
        self.assertAlmostEqual(kps[_KP_RIGHT_ELBOW][1], 260)

    def test_filters_low_confidence_person(self):
        t = self._bare()
        out = _make_pose_output([{"cx": 100, "cy": 100, "w": 50, "h": 100, "conf": 0.1}])
        persons = t._parse_pose_output([out], 1.0, 0, 0, 640, 480)
        self.assertEqual(persons, [])

    def test_letterbox_offset_maps_back_to_original(self):
        t = self._bare()
        # ratio=0.5, pad=(10,20)：letterbox 坐标 (110,140) → 原帧 ((110-10)/0.5, (140-20)/0.5)=(200,240)
        out = _make_pose_output([{
            "cx": 110, "cy": 140, "w": 40, "h": 80, "conf": 0.9,
            "kps": {_KP_RIGHT_WRIST: (110, 140, 0.9)},
        }])
        persons = t._parse_pose_output([out], 0.5, 10, 20, 640, 480)
        _, kps = persons[0]
        self.assertAlmostEqual(kps[_KP_RIGHT_WRIST][0], 200.0)
        self.assertAlmostEqual(kps[_KP_RIGHT_WRIST][1], 240.0)


class TestHandBoxFromPose(unittest.TestCase):
    def _bare(self, max_num_hands=2):
        t = object.__new__(PersonPoseHandTracker)
        t._pose_conf = 0.25
        t.max_num_hands = max_num_hands
        t._sr_enabled = False
        t._sr_trigger = 96
        return t

    def _stub_pose(self, t, persons):
        t._pose_detect = lambda frame: persons

    def test_wrist_anchor_and_forearm_size(self):
        """手框以手腕为中心，尺寸 = 肘腕距 × 系数。"""
        t = self._bare()
        kps = np.zeros((17, 3), dtype=np.float32)
        # 右腕 (400,300)，右肘 (300,300) → 肘腕距 100 → 框 ≈ 90 → half 45
        kps[_KP_RIGHT_WRIST] = (400, 300, 0.95)
        kps[_KP_RIGHT_ELBOW] = (300, 300, 0.95)
        self._stub_pose(t, [((0, 0, 200, 400, 0.9), kps)])
        boxes = t._hand_bboxes_from_pose(np.zeros((480, 640, 3), np.uint8))
        self.assertEqual(len(boxes), 1)
        x0, y0, x1, y1, score = boxes[0]
        # 中心应在手腕处
        self.assertAlmostEqual((x0 + x1) / 2, 400, delta=2)
        self.assertAlmostEqual((y0 + y1) / 2, 300, delta=2)
        # 尺寸 ≈ 100 * 0.9 = 90
        self.assertAlmostEqual(x1 - x0, 90, delta=4)

    def test_falls_back_to_body_scale_when_elbow_missing(self):
        """肘不可见时用手 = 人体高 × 比例 兜底。"""
        t = self._bare()
        kps = np.zeros((17, 3), dtype=np.float32)
        kps[_KP_RIGHT_WRIST] = (400, 300, 0.95)   # 只有手腕
        body = (0, 0, 200, 400, 0.9)               # 人体高 400 → 框 = 400*0.22=88
        self._stub_pose(t, [(body, kps)])
        boxes = t._hand_bboxes_from_pose(np.zeros((480, 640, 3), np.uint8))
        self.assertEqual(len(boxes), 1)
        self.assertAlmostEqual(boxes[0][2] - boxes[0][0], 88, delta=4)

    def test_low_conf_wrist_yields_no_box(self):
        t = self._bare()
        kps = np.zeros((17, 3), dtype=np.float32)
        kps[_KP_RIGHT_WRIST] = (400, 300, 0.1)   # 低于 _KP_MIN_CONF
        self._stub_pose(t, [((0, 0, 200, 400, 0.9), kps)])
        boxes = t._hand_bboxes_from_pose(np.zeros((480, 640, 3), np.uint8))
        self.assertEqual(boxes, [])

    def test_two_wrists_capped_by_max_num_hands(self):
        t = self._bare(max_num_hands=1)
        kps = np.zeros((17, 3), dtype=np.float32)
        kps[_KP_RIGHT_WRIST] = (400, 300, 0.9)
        kps[9] = (100, 100, 0.8)  # 左腕 _KP_LEFT_WRIST=9
        self._stub_pose(t, [((0, 0, 500, 400, 0.9), kps)])
        boxes = t._hand_bboxes_from_pose(np.zeros((480, 640, 3), np.uint8))
        self.assertEqual(len(boxes), 1)
        # 保留 conf 更高的右腕
        self.assertAlmostEqual(boxes[0][4], 0.9, places=5)

    def test_box_clamped_to_frame(self):
        """手腕贴近边缘时框不越界。"""
        t = self._bare()
        kps = np.zeros((17, 3), dtype=np.float32)
        kps[_KP_RIGHT_WRIST] = (5, 5, 0.95)   # 左上角
        self._stub_pose(t, [((0, 0, 200, 400, 0.9), kps)])
        boxes = t._hand_bboxes_from_pose(np.zeros((480, 640, 3), np.uint8))
        x0, y0, x1, y1, _ = boxes[0]
        self.assertGreaterEqual(x0, 0)
        self.assertGreaterEqual(y0, 0)


@unittest.skipUnless(HAS_MODELS, "requires yolov8n-pose + hand_yolov8n + hand_landmarker models")
class TestPersonPoseRealIntegration(unittest.TestCase):
    """真实引擎冒烟：模型加载、随机帧不崩、回退路径可用。"""

    def test_factory_builds_and_runs_blank_frame(self):
        tracker = create_hand_tracker(
            engine="person_pose_hand",
            max_num_hands=1,
            config={"person_pose_sr_enabled": False},
        )
        try:
            self.assertEqual(tracker.engine_name, "person_pose_hand")
            self.assertIsNotNone(tracker._pose_session)
            frame = (np.random.rand(240, 320, 3) * 255).astype(np.uint8)
            out_frame, landmarks, gestures = tracker.find_hands(frame, draw=False)
            self.assertEqual(out_frame.shape, frame.shape)
            self.assertEqual(len(landmarks), len(gestures))
        finally:
            tracker.close()
            self.assertIsNone(tracker._pose_session)


if __name__ == "__main__":
    unittest.main()
