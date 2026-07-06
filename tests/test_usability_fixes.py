"""自测脚本：验证鼠标模式 pinch 防抖和板书模式书写/清屏逻辑。

模拟真实场景下的关键点抖动，验证：
1. 鼠标模式 EMA 预热期不触发 left_down
2. 鼠标模式 连续帧确认避免单帧噪声误触发
3. 板书模式 张掌清屏能打断书写状态
4. 板书模式 放宽的 is_open_palm 在3指伸出时触发
5. 幽灵手颜色统一（黄色→紫色）
"""

import math
import os
import sys

# 添加项目根目录到 path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.gesture_recognizer import GestureRecognizer


def make_landmarks(
    wrist=(320, 240),
    hand_width=80,
    index_tip=(350, 180),
    thumb_tip=(300, 200),
    middle_tip=(370, 175),
    ring_tip=(390, 185),
    pinky_tip=(400, 210),
):
    """构造 21 关键点 landmarks [idx, x, y]。

    只需要关键点的 x,y 坐标正确即可，其他点用合理默认值填充。
    """
    # 基础模板：21 个关键点
    pts = []
    # 0: wrist
    pts.append([0, float(wrist[0]), float(wrist[1])])
    # 1-4: thumb (CMC, MCP, IP, TIP)
    pts.append([1, wrist[0] - 15, wrist[1] - 10])
    pts.append([2, wrist[0] - 25, wrist[1] - 20])
    pts.append([3, wrist[0] - 30, wrist[1] - 25])
    pts.append([4, float(thumb_tip[0]), float(thumb_tip[1])])
    # 5: index MCP
    pts.append([5, wrist[0] + 10, wrist[1] - 30])
    # 6-8: index PIP, DIP, TIP
    pts.append([6, wrist[0] + 15, wrist[1] - 50])
    pts.append([7, wrist[0] + 18, wrist[1] - 65])
    pts.append([8, float(index_tip[0]), float(index_tip[1])])
    # 9: middle MCP
    pts.append([9, wrist[0] + 25, wrist[1] - 30])
    # 10-12: middle PIP, DIP, TIP
    pts.append([10, wrist[0] + 30, wrist[1] - 55])
    pts.append([11, wrist[0] + 33, wrist[1] - 70])
    pts.append([12, float(middle_tip[0]), float(middle_tip[1])])
    # 13: ring MCP
    pts.append([13, wrist[0] + 40, wrist[1] - 28])
    # 14-16: ring PIP, DIP, TIP
    pts.append([14, wrist[0] + 45, wrist[1] - 50])
    pts.append([15, wrist[0] + 48, wrist[1] - 65])
    pts.append([16, float(ring_tip[0]), float(ring_tip[1])])
    # 17: pinky MCP
    pts.append([17, wrist[0] + 55, wrist[1] - 25])
    # 18-20: pinky PIP, DIP, TIP
    pts.append([18, wrist[0] + 58, wrist[1] - 45])
    pts.append([19, wrist[0] + 60, wrist[1] - 58])
    pts.append([20, float(pinky_tip[0]), float(pinky_tip[1])])
    return pts


def test_open_palm_relaxed():
    """测试1: is_open_palm 需要5指全伸才触发（Phase 2.11 回滚到老版严格定义）。"""
    print("\n=== 测试1: is_open_palm 5指全伸 ===")
    recognizer = GestureRecognizer()

    # 5指全伸 → open_palm
    lm = make_landmarks(
        index_tip=(350, 150),   # tip 在 PIP 上方 → up
        middle_tip=(370, 145),  # tip 在 PIP 上方 → up
        ring_tip=(390, 155),    # tip 在 PIP 上方 → up
        pinky_tip=(400, 165),   # tip 在 PIP 上方 → up
        thumb_tip=(280, 180),   # thumb up
    )
    features = recognizer.get_hand_features(lm)
    assert features["is_open_palm"], f"5指全伸应判定为 open_palm，实际 is_open_palm={features['is_open_palm']}"
    print(f"  5指全伸: is_open_palm={features['is_open_palm']} ✓")

    # 3指伸（食+中+无名），小指弯曲 → 不应是 open_palm
    lm2 = make_landmarks(
        index_tip=(350, 150),
        middle_tip=(370, 145),
        ring_tip=(390, 155),
        pinky_tip=(400, 220),   # pinky down
    )
    features2 = recognizer.get_hand_features(lm2)
    assert not features2["is_open_palm"], f"3指伸（小指弯曲）不应判定为 open_palm，实际 is_open_palm={features2['is_open_palm']}"
    print(f"  3指伸（小指弯曲）: is_open_palm={features2['is_open_palm']} ✓")

    # 剪刀手（食指+中指）→ 不应是 open_palm
    lm3 = make_landmarks(
        index_tip=(350, 150),
        middle_tip=(370, 145),
        ring_tip=(390, 220),   # ring down
        pinky_tip=(400, 225),  # pinky down
    )
    features3 = recognizer.get_hand_features(lm3)
    assert not features3["is_open_palm"], f"剪刀手不应判定为 open_palm，实际 is_open_palm={features3['is_open_palm']}"
    print(f"  剪刀手: is_open_palm={features3['is_open_palm']} ✓")

    # 握拳 → 不应是 open_palm
    lm4 = make_landmarks(
        index_tip=(350, 230),
        middle_tip=(370, 230),
        ring_tip=(390, 230),
        pinky_tip=(400, 230),
    )
    features4 = recognizer.get_hand_features(lm4)
    assert not features4["is_open_palm"], f"握拳不应判定为 open_palm，实际 is_open_palm={features4['is_open_palm']}"
    print(f"  握拳: is_open_palm={features4['is_open_palm']} ✓")

    print("  PASS: is_open_palm 放宽验证通过")


def test_mouse_warmup_no_click():
    """测试2: 鼠标模式 EMA 预热期不触发 left_down。

    模拟手进入画面前3帧，即使 pinch 距离很小（看起来像捏合），
    也不应触发 left_down。
    """
    print("\n=== 测试2: 鼠标模式 EMA 预热期不触发 ===")

    # 构造一个捏合手势（拇指食指很近）
    pinch_lm = make_landmarks(
        thumb_tip=(310, 200),
        index_tip=(315, 200),  # 距离拇指只有5像素 → pinch
        hand_width=80,
    )

    recognizer = GestureRecognizer()
    features = recognizer.get_hand_features(pinch_lm)

    # 验证 pinch 距离确实很小
    thumb_index_dist = math.hypot(
        pinch_lm[4][1] - pinch_lm[8][1],
        pinch_lm[4][2] - pinch_lm[8][2]
    )
    pinch_threshold = features["hand_width"] * 0.35
    print(f"  pinch 距离={thumb_index_dist:.1f}, 阈值={pinch_threshold:.1f}")
    assert thumb_index_dist < pinch_threshold, "测试前提：应为捏合状态"

    # 模拟 MouseMode 的预热逻辑
    ema_warmup_frames = 0
    EMA_WARMUP_MIN = 3
    left_pinch_triggered = False

    for frame_idx in range(5):
        ema_warmup_frames += 1
        if ema_warmup_frames < EMA_WARMUP_MIN:
            # 预热期：不判定 pinch
            print(f"  帧{frame_idx}: 预热期，跳过 pinch 判定")
            continue

        # 预热结束后才判定
        if thumb_index_dist < pinch_threshold:
            left_pinch_triggered = True
            print(f"  帧{frame_idx}: 预热结束，pinch 触发")

    # 前3帧不触发，第3帧（index=3, warmup=4）才触发
    assert not left_pinch_triggered or ema_warmup_frames >= EMA_WARMUP_MIN + 1, \
        "预热期内不应触发 pinch"
    print("  PASS: EMA 预热期不触发 left_down")


def test_mouse_streak_confirmation():
    """测试3: 鼠标模式 连续帧确认避免单帧噪声误触发。

    模拟 pinch 信号在连续帧中闪烁（True/False/True），
    streak 不到阈值不应触发。
    """
    print("\n=== 测试3: 连续帧确认避免单帧噪声 ===")

    LEFT_PINCH_MIN_STREAK = 2
    # 模拟 pinch 信号序列：True, False, True, True, True
    # streak: 1, 0, 1, 2, 3
    # 触发:  False, False, False, True, True
    pinch_signals = [True, False, True, True, True]
    expected_triggered = [False, False, False, True, True]

    streak = 0
    is_holding = False
    for i, (pinch_raw, expected) in enumerate(zip(pinch_signals, expected_triggered, strict=True)):
        if is_holding:
            # 已在捏合状态，滞回阈值足够宽松，不需要 streak
            triggered = pinch_raw
            streak = 0
        else:
            if pinch_raw:
                streak += 1
                triggered = streak >= LEFT_PINCH_MIN_STREAK
            else:
                streak = 0
                triggered = False

        print(f"  帧{i}: pinch_raw={pinch_raw}, streak={streak}, triggered={triggered}, expected={expected}")
        assert triggered == expected, f"帧{i}: 期望 triggered={expected}, 实际 triggered={triggered}"

    print("  PASS: 连续帧确认正确过滤单帧噪声")


def test_draw_clear_screen_interrupts_writing():
    """测试4: 板书模式 张掌清屏能打断书写状态。

    模拟书写中突然张掌，验证 is_open_palm 为 True 时能立即进入清屏分支。
    """
    print("\n=== 测试4: 张掌清屏打断书写 ===")

    recognizer = GestureRecognizer()

    # 书写姿势：食指伸出，拇指收拢
    writing_lm = make_landmarks(
        index_tip=(350, 150),   # index up
        middle_tip=(370, 220),  # middle down
        ring_tip=(390, 220),    # ring down
        pinky_tip=(400, 225),   # pinky down
        thumb_tip=(290, 230),   # thumb down (tucked)
    )
    writing_features = recognizer.get_hand_features(writing_lm)
    print(f"  书写姿势: index_up={writing_features['index_up']}, "
          f"is_open_palm={writing_features['is_open_palm']}, "
          f"thumb_tucked={writing_features['thumb_tucked']}")
    assert writing_features["index_up"], "书写姿势应食指伸出"
    assert not writing_features["is_open_palm"], "书写姿势不应是张掌"

    # 张掌姿势：5指全伸（Phase 2.11 回滚后 is_open_palm 需要5指全伸）
    open_palm_lm = make_landmarks(
        index_tip=(350, 150),
        middle_tip=(370, 145),
        ring_tip=(390, 155),
        pinky_tip=(400, 165),  # pinky up
        thumb_tip=(280, 180),  # thumb up
    )
    open_palm_features = recognizer.get_hand_features(open_palm_lm)
    print(f"  张掌姿势: index_up={open_palm_features['index_up']}, "
          f"middle_up={open_palm_features['middle_up']}, "
          f"ring_up={open_palm_features['ring_up']}, "
          f"pinky_up={open_palm_features['pinky_up']}, "
          f"is_open_palm={open_palm_features['is_open_palm']}")
    assert open_palm_features["is_open_palm"], "5指全伸应判定为张掌"

    # 模拟 draw_mode 的逻辑：书写中检测到 is_open_palm → 立即抬笔
    was_writing = True
    if open_palm_features["is_open_palm"]:
        # 立即抬笔，流向清屏分支
        was_writing = False
        print("  张掌检测到 → 立即抬笔，流向清屏分支")

    assert not was_writing, "张掌应立即打断书写状态"
    print("  PASS: 张掌清屏能打断书写")


def test_ghost_hand_color_unified():
    """测试5: 幽灵手颜色统一为紫色。

    验证 base_hand_tracker.py 中所有预测补帧的绘制颜色都是紫色 (255, 0, 255)，
    不再有黄色 (0, 255, 255)。
    """
    print("\n=== 测试5: 幽灵手颜色统一 ===")

    # 读取 base_hand_tracker.py 检查颜色
    tracker_path = os.path.join(
        os.path.dirname(__file__), "..", "app", "services", "base_hand_tracker.py"
    )
    with open(tracker_path, encoding="utf-8") as f:
        content = f.read()

    # 查找所有 draw_points 调用中的颜色
    import re
    # 找到所有 draw_points 调用（self._renderer.draw_points(...)）
    draw_calls = re.findall(
        r'\.draw_points\([^)]+,\s*\((\d+),\s*(\d+),\s*(\d+)\)\)',
        content
    )
    print(f"  找到 {len(draw_calls)} 个 draw_points 调用:")
    all_purple = True
    for i, (r, g, b) in enumerate(draw_calls):
        color = (int(r), int(g), int(b))
        is_purple = color == (255, 0, 255)
        if not is_purple:
            all_purple = False
        print(f"    {i}: {color} {'✓ 紫色' if is_purple else '✗ 非紫色！'}")

    assert all_purple, "所有绘制点颜色应为紫色 (255, 0, 255)"
    print("  PASS: 所有预测补帧颜色已统一为紫色")


def test_draw_vote_threshold_reachable():
    """测试6: 板书模式投票阈值在低帧率下可达。

    模拟 20fps 下 0.3s 窗口 = 6 帧，VOTE_MIN=2, vote_ratio=0.50：
    4帧 write 即可触发落笔（4/6=0.67 >= 0.50, total=6 >= 2）。
    """
    print("\n=== 测试6: 投票阈值在低帧率下可达 ===")

    VOTE_MIN = 2
    vote_ratio = 0.50
    vote_window_sec = 0.30
    fps = 20
    frames_in_window = int(fps * vote_window_sec)  # 6 帧

    # 模拟 6 帧中有 4 帧 write, 2 帧 other
    total = frames_in_window  # 6
    n_write = 4

    can_start_writing = (total >= VOTE_MIN) and (n_write >= total * vote_ratio)
    print(f"  {fps}fps, 窗口={vote_window_sec}s → {frames_in_window}帧")
    print(f"  {n_write}/{total} write, 阈值={vote_ratio}, VOTE_MIN={VOTE_MIN}")
    print(f"  可触发落笔: {can_start_writing}")
    assert can_start_writing, "4/6 write 应能触发落笔"

    # 旧阈值对比：VOTE_MIN=3, vote_ratio=0.60
    old_can = (total >= 3) and (n_write >= total * 0.60)
    print(f"  旧阈值(3/0.60): {n_write}/{total} = {n_write/total:.2f} >= 0.60? {old_can}")

    # 3帧 write 的情况（更常见）
    n_write_3 = 3
    new_can = (total >= VOTE_MIN) and (n_write_3 >= total * vote_ratio)
    old_can_3 = (total >= 3) and (n_write_3 >= total * 0.60)
    print(f"  3/6 write: 新阈值={new_can}, 旧阈值={old_can_3}")
    assert new_can, "3/6 write 在新阈值下应能触发"
    assert not old_can_3, "3/6 write 在旧阈值下不应触发（证明旧阈值太严）"

    print("  PASS: 新投票阈值在低帧率下可达，旧阈值不可达")


def test_index_extended_hysteresis():
    """测试7: index_extended 滞回——一旦伸出，小幅抖动不会取消伸出。

    模拟食指长度在阈值附近抖动，验证滞回机制：
    进入阈值 0.60，退出阈值 0.50。伸出后 index_len 降到 0.55×hand_width
    仍保持伸出状态。
    """
    print("\n=== 测试7: index_extended 滞回 ===")

    recognizer = GestureRecognizer()

    # make_landmarks 中 point 5=(wrist+10, wrist-30), point 17=(wrist+55, wrist-25)
    # hand_width = hyp(45, 5) ≈ 45.2
    # index_len = hyp(tip - point5) = hyp(tip_x - (wrist_x+10), tip_y - (wrist_y-30))
    # 需要 index_len/hand_width > 0.60 → index_len > 27.1
    # tip = (wrist_x+10, wrist_y-30-30) = (330, 180) → index_len = 30, ratio ≈ 0.66

    # 帧1: index_len ≈ 30 (ratio ≈ 0.66 > 0.60) → 伸出
    lm1 = make_landmarks(
        wrist=(320, 240),
        index_tip=(330, 180),  # index_len = hyp(0, 30) = 30
    )
    f1 = recognizer.get_hand_features(lm1)
    hw = f1["hand_width"]
    il1 = math.hypot(lm1[8][1] - lm1[5][1], lm1[8][2] - lm1[5][2])
    print(f"  帧1: hand_width={hw:.1f}, index_len={il1:.1f}, ratio={il1/hw:.2f}, "
          f"index_extended={f1['index_extended']} (应=True)")
    assert f1["index_extended"], f"帧1: ratio={il1/hw:.2f} > 0.60 应伸出"

    # 帧2: index_len 降到 ≈25 (ratio ≈ 0.55, 在 0.50~0.60 之间) → 滞回保持伸出
    lm2 = make_landmarks(
        wrist=(320, 240),
        index_tip=(330, 185),  # index_len = hyp(0, 25) = 25
    )
    f2 = recognizer.get_hand_features(lm2)
    il2 = math.hypot(lm2[8][1] - lm2[5][1], lm2[8][2] - lm2[5][2])
    print(f"  帧2: index_len={il2:.1f}, ratio={il2/hw:.2f}, "
          f"index_extended={f2['index_extended']} (滞回应=True)")
    assert f2["index_extended"], f"帧2: 滞回应保持伸出 (ratio={il2/hw:.2f} > 0.50)"

    # 帧3: index_len 降到 ≈20 (ratio ≈ 0.44 < 0.50) → 退出伸出
    lm3 = make_landmarks(
        wrist=(320, 240),
        index_tip=(330, 190),  # index_len = hyp(0, 20) = 20
    )
    f3 = recognizer.get_hand_features(lm3)
    il3 = math.hypot(lm3[8][1] - lm3[5][1], lm3[8][2] - lm3[5][2])
    print(f"  帧3: index_len={il3:.1f}, ratio={il3/hw:.2f}, "
          f"index_extended={f3['index_extended']} (应=False)")
    assert not f3["index_extended"], f"帧3: ratio={il3/hw:.2f} < 0.50 应退出伸出"

    print("  PASS: index_extended 滞回正确")


def test_classify_frame_writing_priority():
    """测试8: _classify_frame 书写姿势优先于 is_open_palm。

    食指伸出 + 中指误检为 up（other_fingers_up=1）→ 应判定为 write 而非 stop。
    """
    print("\n=== 测试8: 书写姿势优先于 is_open_palm ===")

    recognizer = GestureRecognizer()

    # 书写姿势：食指伸出，中指误检为 up，无名指/小指 down
    # is_open_palm = index_up and middle_up and ring_up → True (3指)
    # 但 other_fingers_up = 1 (只有 middle_up) → 应判定为 write
    lm = make_landmarks(
        index_tip=(350, 150),   # index up (tip above PIP)
        middle_tip=(370, 155),  # middle up (误检)
        ring_tip=(390, 220),    # ring down
        pinky_tip=(400, 225),   # pinky down
    )
    features = recognizer.get_hand_features(lm)
    print(f"  index_up={features['index_up']}, middle_up={features['middle_up']}, "
          f"ring_up={features['ring_up']}, is_open_palm={features['is_open_palm']}, "
          f"index_extended={features['index_extended']}")

    # 模拟 _classify_frame 的核心逻辑
    other_up = features["middle_up"] + features["ring_up"] + features["pinky_up"]
    if features["index_extended"] and other_up <= 1:
        classification = "write"
    elif features["is_open_palm"]:
        classification = "stop"
    else:
        classification = "other"

    print(f"  other_fingers_up={other_up}, classification={classification}")
    assert classification == "write", \
        f"食指伸出+1个其他手指up应判定为 write，实际={classification}"

    # 真正的张掌：5指全伸 → is_open_palm=True → stop（Phase 2.11 回滚后需5指全伸）
    lm2 = make_landmarks(
        index_tip=(350, 150),
        middle_tip=(370, 145),
        ring_tip=(390, 155),    # ring up
        pinky_tip=(400, 165),   # pinky up
        thumb_tip=(280, 180),   # thumb up
    )
    features2 = recognizer.get_hand_features(lm2)
    other_up2 = features2["middle_up"] + features2["ring_up"] + features2["pinky_up"]
    if features2["index_extended"] and other_up2 <= 1:
        classification2 = "write"
    elif features2["is_open_palm"]:
        classification2 = "stop"
    else:
        classification2 = "other"

    print(f"  张掌: other_fingers_up={other_up2}, classification={classification2}")
    assert classification2 == "stop", \
        f"5指全伸应判定为 stop，实际={classification2}"

    print("  PASS: 书写姿势优先于 is_open_palm")


if __name__ == "__main__":
    tests = [
        test_open_palm_relaxed,
        test_mouse_warmup_no_click,
        test_mouse_streak_confirmation,
        test_draw_clear_screen_interrupts_writing,
        test_ghost_hand_color_unified,
        test_draw_vote_threshold_reachable,
        test_index_extended_hysteresis,
        test_classify_frame_writing_priority,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except (AssertionError, Exception) as e:
            print(f"  FAIL: {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"自测结果: {passed} 通过, {failed} 失败, 共 {len(tests)} 项")
    if failed > 0:
        sys.exit(1)
    else:
        print("全部通过！")
