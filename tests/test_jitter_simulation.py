#!/usr/bin/env python
"""自测脚本：模拟关键点抖动、handedness 翻转、幽灵手冲突等场景。

用法：python tests/test_jitter_simulation.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from app.services.base_hand_tracker import BaseHandTracker, OneEuroSmoother


class DummyTracker(BaseHandTracker):
    """可实例化的测试用 tracker，跳过真实 MediaPipe 检测。"""
    def _detect(self, frame, draw=False):
        return [], []
    def _detect_crop_zoom(self, frame, crop_center, crop_size, draw=False):
        return [], []
    @property
    def engine_name(self):
        return "dummy"


def make_landmarks(cx, cy, hand_size=50):
    """生成 21 个关键点，模拟一只手在 (cx, cy) 位置。"""
    lms = []
    # wrist
    lms.append((0, cx, cy, 0.9))
    # thumb CMC, MCP, IP, TIP
    for dx, dy in [(-hand_size*0.3, hand_size*0.1), (-hand_size*0.5, hand_size*0.2), (-hand_size*0.6, hand_size*0.3), (-hand_size*0.7, hand_size*0.4)]:
        lms.append((0, cx+dx, cy+dy, 0.8))
    # index MCP, PIP, DIP, TIP
    for dx, dy in [(-hand_size*0.1, -hand_size*0.3), (-hand_size*0.1, -hand_size*0.5), (-hand_size*0.1, -hand_size*0.65), (-hand_size*0.1, -hand_size*0.8)]:
        lms.append((0, cx+dx, cy+dy, 0.8))
    # middle MCP, PIP, DIP, TIP
    for dx, dy in [(hand_size*0.1, -hand_size*0.3), (hand_size*0.1, -hand_size*0.5), (hand_size*0.1, -hand_size*0.65), (hand_size*0.1, -hand_size*0.8)]:
        lms.append((0, cx+dx, cy+dy, 0.8))
    # ring MCP, PIP, DIP, TIP
    for dx, dy in [(hand_size*0.3, -hand_size*0.3), (hand_size*0.3, -hand_size*0.5), (hand_size*0.3, -hand_size*0.65), (hand_size*0.3, -hand_size*0.8)]:
        lms.append((0, cx+dx, cy+dy, 0.8))
    # pinky MCP, PIP, DIP, TIP
    for dx, dy in [(hand_size*0.5, -hand_size*0.2), (hand_size*0.5, -hand_size*0.4), (hand_size*0.5, -hand_size*0.55), (hand_size*0.5, -hand_size*0.7)]:
        lms.append((0, cx+dx, cy+dy, 0.8))
    return lms


def make_gesture(handedness="Right", label="Open_Palm"):
    return {
        "ml_label": label,
        "label": label,
        "score": 0.9,
        "handedness": handedness,
        "handedness_score": 0.9,
        "bbox_area": 1000.0,
    }


def test_handedness_flip_no_ghost():
    """测试：关闭幽灵手后，单手场景下 handedness 翻转不应产生幽灵手。"""
    tracker = DummyTracker(config={})
    tracker._hand_prediction_enabled = False  # 显式关闭幽灵手补帧

    # 直接调用平滑逻辑（绕过 _detect）
    # 帧1：右手
    lms = make_landmarks(320, 240, hand_size=50)
    ges = [make_gesture("Right")]
    # 模拟 find_hands 中检测后的平滑逻辑
    is_single = len(lms) == 21 and len(ges) == 1
    tracker._active_handedness = set()
    smoothed_all, gesture_all, seen = _run_smooth_loop(tracker, [lms], ges, is_single)
    print(f"帧1 (Right): seen={seen}, hands={len(smoothed_all)}")
    assert "Right" in seen, f"单手右手应使用 Right smoother，实际 {seen}"
    assert len(smoothed_all) == 1, f"不应有幽灵手，实际 {len(smoothed_all)} 只手"

    # 帧2：handedness 翻转
    lms2 = make_landmarks(325, 245, hand_size=50)
    ges2 = [make_gesture("Left")]
    smoothed_all, gesture_all, seen = _run_smooth_loop(tracker, [lms2], ges2, is_single)
    print(f"帧2 (Left flip): seen={seen}, hands={len(smoothed_all)}")
    assert "Left" in seen, f"handedness 翻转后应使用 Left smoother，实际 {seen}"
    assert len(smoothed_all) == 1, f"不应有幽灵手，实际 {len(smoothed_all)} 只手"

    print("PASS: handedness 翻转不产生幽灵手")


def _run_smooth_loop(tracker, hands_landmarks, hands_gestures, is_single_hand):
    """模拟 find_hands 中的平滑+幽灵手逻辑（与 base_hand_tracker.py 保持一致）。

    阶段 2.10：smoother 按 handedness 分配，不再有 Primary/Secondary 槽位。
    is_single_hand 参数保留以兼容调用方，但不再影响 smoother key 分配。
    """
    smoothed_all = []
    gesture_all = []
    seen = set()
    for landmarks, gesture in zip(hands_landmarks, hands_gestures, strict=True):
        # 按 MediaPipe handedness 分配 smoother
        key = gesture.get("handedness", "Unknown")
        if key not in tracker.smoothers:
            key = "Unknown"
        smoothed = tracker.smoothers[key].update(landmarks)
        geo_filter = tracker._geo_filters.get(key)
        if geo_filter is not None:
            smoothed = geo_filter.apply(smoothed)
        seen.add(key)
        smoothed_all.append(smoothed)
        gesture_all.append(gesture)

    # 幽灵手补帧（仅在 _hand_prediction_enabled 开启时生效，默认关闭）
    if getattr(tracker, "_hand_prediction_enabled", False):
        missing_keys = tracker._active_handedness - seen
        for key in missing_keys:
            smoother = tracker.smoothers.get(key)
            if smoother is None or smoother.lost_frames >= 3:
                continue
            ghost = smoother.predict()
            if ghost is None:
                continue
            smoothed_all.append(ghost)
            gesture_all.append({"handedness": key, "predicted": True})
            seen.add(key)

    tracker._active_handedness = seen
    return smoothed_all, gesture_all, seen


def test_multi_to_single_no_ghost():
    """测试：关闭幽灵手后，从多手场景切换到单手场景不应有幽灵手。"""
    tracker = DummyTracker(config={})
    tracker._hand_prediction_enabled = False  # 显式关闭幽灵手补帧
    tracker._active_handedness = set()

    # 帧1-3：两只手
    for i in range(3):
        lms_l = make_landmarks(160, 240, hand_size=40)
        lms_r = make_landmarks(480, 240, hand_size=40)
        ges_l = make_gesture("Left")
        ges_r = make_gesture("Right")
        smoothed_all, _, seen = _run_smooth_loop(tracker, [lms_l, lms_r], [ges_l, ges_r], is_single_hand=False)
        print(f"帧{i+1} (双手): seen={seen}, hands={len(smoothed_all)}")

    # 帧4：切换到单手（右手）
    lms = make_landmarks(320, 240, hand_size=50)
    ges = make_gesture("Right")
    smoothed_all, _, seen = _run_smooth_loop(tracker, [lms], [ges], is_single_hand=True)
    print(f"帧4 (切单手): seen={seen}, hands={len(smoothed_all)}")
    assert len(smoothed_all) == 1, f"切单手后不应有幽灵手，实际 {len(smoothed_all)} 只手"
    assert "Left" not in seen, f"切单手右手后 Left 不应在 seen 中，实际 {seen}"

    print("PASS: 多手切单手不产生幽灵手")


def test_jitter_stability():
    """测试：关键点抖动时 smoother 输出应平滑。"""
    smoother = OneEuroSmoother(min_cutoff=0.2, beta=0.01)
    np.random.seed(42)

    # 模拟 50 帧的抖动：手在 (320, 240) 附近抖动
    positions = []
    for _ in range(50):
        jitter_x = np.random.randn() * 5  # 5 像素标准差抖动
        jitter_y = np.random.randn() * 5
        lms = make_landmarks(320 + jitter_x, 240 + jitter_y, hand_size=50)
        smoothed = smoother.update(lms)
        wrist = smoothed[0]
        positions.append((wrist[0], wrist[1]))

    # 计算后 30 帧的位置标准差
    later = positions[20:]
    xs = [p[0] for p in later]
    ys = [p[1] for p in later]
    std_x = np.std(xs)
    std_y = np.std(ys)
    print(f"抖动测试: 输入 std=5.0, 输出 std_x={std_x:.2f}, std_y={std_y:.2f}")
    assert std_x < 3.0, f"x 方向平滑不足: std={std_x:.2f}"
    assert std_y < 3.0, f"y 方向平滑不足: std={std_y:.2f}"

    print("PASS: 关键点抖动被有效平滑")


def test_hand_width_stability():
    """测试：hand_width 在手大小不变时应稳定。"""
    smoother = OneEuroSmoother(min_cutoff=0.2, beta=0.01)
    np.random.seed(42)

    hand_widths = []
    for _ in range(50):
        jitter = np.random.randn() * 3
        lms = make_landmarks(320, 240, hand_size=50 + jitter)
        smoothed = smoother.update(lms)
        # 计算 hand_width: landmark 5 到 17 的距离
        import math
        hw = math.hypot(smoothed[5][0] - smoothed[17][0], smoothed[5][1] - smoothed[17][1])
        hand_widths.append(hw)

    later = hand_widths[20:]
    std_hw = np.std(later)
    mean_hw = np.mean(later)
    print(f"hand_width: mean={mean_hw:.2f}, std={std_hw:.2f}, cv={std_hw/mean_hw:.3f}")
    assert std_hw < 2.0, f"hand_width 平滑不足: std={std_hw:.2f}"

    print("PASS: hand_width 稳定")


if __name__ == "__main__":
    print("=" * 60)
    print("自测：关键点抖动与幽灵手问题")
    print("=" * 60)

    tests = [
        test_jitter_stability,
        test_hand_width_stability,
        test_handedness_flip_no_ghost,
        test_multi_to_single_no_ghost,
    ]

    passed = 0
    failed = 0
    for test in tests:
        print(f"\n--- {test.__name__} ---")
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"FAIL: {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR: {e}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"结果: {passed} passed, {failed} failed")
    print(f"{'=' * 60}")
    sys.exit(0 if failed == 0 else 1)
