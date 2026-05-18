# AirControl Bug 分析报告

经过完整审查，项目已大幅重构（升级到 MediaPipe Gesture Recognizer Task API、加入卡尔曼平滑器、新增板书工具栏/鼠标光标叠加层等）。但一次性变更量过大，引入了一批相互耦合的 Bug。

---

## 🔴 P0 — 会导致程序异常或行为严重错误的 Bug

### B1. 系统鼠标从未隐藏 — MouseCursorOverlay 是空的壳

**位置**：`app/mouse_cursor_overlay.py:43-47`

```python
def _hide_system_cursor(self):
    pass    # ← 空函数！

def _show_system_cursor(self):
    pass    # ← 空函数！
```

**影响**：鼠标模式下，`MouseCursorOverlay` 会在屏幕上层画一个漂亮的光标环，但系统箭头光标**从未被隐藏**。两重光标同时显示，用户会看到系统箭头和自定义光环指针错位（光环位置是平滑后的，箭头是系统真实的），体验割裂且极度困惑。

**根因**：这两个方法留了 stub 但没有实现。需要调用 Windows API 隐藏/显示系统光标。

---

### B2. 卡尔曼平滑器预判可能产生幻影手部

**位置**：`app/services/hand_tracker.py:129, 249-252`

```python
self.smoothers = [KalmanSmoother(), KalmanSmoother()]  # 始终创建 2 个
...
for i, sm in enumerate(self.smoothers):
    predicted = sm.predict()
    if predicted is not None:
        predicted_all.append(predicted)
```

**触发场景**：`max_num_hands=2` 但只有 1 只手在画面中。当手部短暂丢失时，第二个 smoothers 会基于空状态预测出完全虚假的关键点坐标。

**影响**：可能误触发双拳切换模式、误触手势操作。这是**间歇性出现"幽灵手势"**的根因。

**修复方向**：只对 `last_gestures` 中有过有效数据的手使用 smoother 预测。

---

### B3. FIST 触发绕过 ML 模型置信度

**位置**：`app/services/gesture_recognizer.py:194-195`

```python
if ml_label == "FIST" or features["is_fist"]:
    logging.info("=> Trigger: FIST")
    self._reset_state()
    return "FIST"
```

**问题**：即使 ML 模型说"这不是 FIST"（比如返回 `Open_Palm`），只要手写规则 `features["is_fist"]` 返回 True，就仍然触发 FIST。**这完全绕过了 ML 模型**。

**影响**：ML 升级带来的识别精度收益几乎为零——旧规则一旦和 ML 冲突，旧规则仍会覆盖 ML 的判断。FIST 误触是最常被投诉的手势之一。

---

### B4. 板书书写手势允许剪刀手姿势误触

**位置**：`app/services/gesture_recognizer.py:70`

```python
"thumb_writing": index_extended and not ring_up and not pinky_up and not thumb_extended,
```

**问题**：条件只要求食指伸展、无名指和小指弯曲、拇指不收拢，但**不检查中指**。这意味着用户比 V 字（Victory/剪刀手）且拇指不收拢时，会被判定为"正在书写"——在演示中突然开始画线。

**实际场景**：用户做 V 字胜利手势时，板书模式开始写字。

**修复方向**：增加 `and not middle_up`。

---

### B5. `WindowTransparentForInput` 枚举名可能不存在

**位置**：`app/drawing_overlay.py:38`

```python
Qt.WindowType.WindowTransparentForInput  # 可能不存在的枚举
```

**问题**：PyQt6 中这个枚举的正确路径取决于版本。如果当前环境没有该枚举，初始化时就会 `AttributeError`，导致板书窗口创建失败。实际点击穿透靠后面的 `_make_click_through()` 处理，所以这个枚举是多余的但会造成崩溃风险。

**实际检查**：如果程序当前能启动，说明环境中存在该枚举，但这是版本依赖的——换个 PyQt6 版本就可能炸。

---

## 🟠 P1 — 功能异常或逻辑错误的 Bug

### B6. 双手平滑器数量与检测手数不匹配

**位置**：`app/services/hand_tracker.py:129 vs 178`

- `__init__` 中硬编码 2 个 KalmanSmoother
- 但 `HandTracker.__init__` 参数 `max_num_hands=1` 或 `2`
- `_detect` 根据 `num_hands` 参数返回对应数量的手
- `find_hands` 中用 `zip(hands_landmarks, raw_hand_lists)` 配对，但 smoother 总是 2 个

**影响**：当 `max_num_hands=2` 且画面有 2 只手，然后 1 只手移出画面，第 1 只手丢失时，两个 smoother 的 `predict` 返回长度不同的列表。如果 `smoothed_all` 长度和 `gesture_all` 长度不一致，`find_hands` 返回的列表会错位。

---

### B7. 鼠标滚轮检测与左键/右键点击竞争

**位置**：`app/main_ui.py:437-472`

```python
# 在 mouse_mode 处理中：
is_pinching = features["thumb_index_pinch"] or features["thumb_middle_pinch"]

if not is_pinching:
    scroll_dir = self.recognizer.check_scroll(...)  # 检查滚轮
```

**问题**：当用户用剪刀手（Victory）滚屏时，只要手稍微抖动导致 `thumb_index_pinch` 短暂为 True（接近阈值时），滚轮检测被跳过，同时左键点击条件也可能不满足，导致光标既不滚动也不点击——用户感觉"卡住了"。

**根因**：滚轮和点击的判定是互斥的，但实际上滚轮应该在"点击条件不满足时才尝试识别滚轮"，而不是"只要不是捏合就检测滚轮"。

---

### B8. 绘图画布桥接距离导致的断笔

**位置**：`app/drawing_overlay.py:123-126, 154`

```python
def _bridge_too_far(self, p1, p2):
    dx = p1.x() - p2.x()
    dy = p1.y() - p2.y()
    return dx * dx + dy * dy > self.MAX_BRIDGE_DISTANCE * self.MAX_BRIDGE_DISTANCE
```

**问题**：`MAX_BRIDGE_DISTANCE=120`，但经过卡尔曼平滑和 EMA 后，用户在正常书写速度下的笔迹跳变可能超过 120 像素（尤其在 4K 屏幕下），导致连续笔画被判为"新笔画"并推入 undo 快照。用户写一条长线可能需要按 3-4 次撤销才能清掉。

**修复方向**：桥接阈值应基于屏幕 DPI 缩放。

---

### B9. 手势识别中的 Open_Palm 条件重叠

**位置**：`app/services/gesture_recognizer.py:199-200`

```python
is_closed_palm = (ml_label == "OPEN" and features["fingers_close"]) or \
                 (ml_label in ("OTHER", "OPEN") and features["fingers_close"])
```

**问题**：第一个条件 `ml_label == "OPEN"` 完全被第二个条件包含（第二个已经覆盖了 `"OPEN"`），第一个是冗余条件。这本身不是运行时错误，但表明逻辑理解有偏差——如果意图是只在 ML 说 OPEN 且手指并拢时才允许挥动，那么第二个条件中的 `ml_label == "OTHER"` 就应该去掉。

**实际行为**：当前 `ml_label` 为 `None`（ML 无法分类）时也会允许挥动，这意味着 ML 不认识的姿势也能翻页。

---

### B10. 连续保存配置时 `pen_width` 被覆盖

**位置**：`app/main_ui.py:267-269`

```python
def _on_pen_width_changed(self, width):
    self.overlay.set_pen_width(width)
    self.config.set("pen_width", width)  # ← 每次拖动滑块都写磁盘
```

虽然 `batch_update()` 修复了设置面板的问题，但板书工具栏的滑块 **每次拖动都单独写一次 JSON 文件**，在快速调节画笔大小时产生高频 IO。

---

## 🟡 P2 — 设计缺陷与潜在隐患

### B11. `GestureRecognizer` 挥动检测使用 wrist 坐标但 ML 模型返回的手部可能在图像边缘

当手靠近图像边缘时，MediaPipe 关键点的置信度下降，wrist (landmark 0) 的位置可能跳跃。挥动检测基于 wrist 轨迹，这意味着在图像边缘挥手时可能误触发。

### B12. `PptController.switch_app()` 中 `os.system("start wpp")` 仍未改为 subprocess

原审查中已指出，但修改未覆盖此文件。虽然参数硬编码，`os.system` 会创建 cmd.exe 子进程，有轻微的安全和性能隐患。

### B13. 日志文件 `gesture.log` 无限增长

每帧都记录 `"Hand lost"`、`"Pose broken"`、挥动方向等，在一场 1 小时的演示中可能产生数万条日志。长期运行会占用磁盘空间。

### B14. `hand_tracker.py:244` 绘制的是原始关键点，返回的是平滑后关键点

```python
if draw:
    self._draw_landmarks(frame, raw_hand, landmarks)  # 绘制原始
return frame, smoothed_all, gesture_all  # 返回平滑后的
```

用户看到的视觉反馈是**未平滑的抖动关键点**，但系统响应的是**平滑后的坐标**。视觉和反馈不一致会让用户困惑——"屏幕上我的手在抖，为什么操作没抖？"或者反过来。

---

## 📋 总结：Bug 根因分布

| 根因类别 | 数量 | 占比 | 典型代表 |
|---------|------|------|---------|
| **半成品功能**（Stub/空壳） | 2 | 14% | B1 系统鼠标未隐藏 |
| **ML 与规则冲突** | 2 | 14% | B3 FIST 绕过 ML, B4 剪刀手误写 |
| **枚举/API 版本依赖** | 1 | 7% | B5 WindowTransparentForInput |
| **逻辑运算错误** | 3 | 21% | B2 幻影手, B6 平滑器数量, B9 条件重叠 |
| **设计竞争条件** | 2 | 14% | B7 滚轮vs点击, B8 桥接距离 |
| **未完成重构遗留** | 3 | 21% | B10 高频IO, B12 os.system, B13 日志膨胀 |
| **UX 不一致** | 1 | 7% | B14 绘制原始 vs 返回平滑 |

**核心结论**：大量 Bug 源于**一次性引入过多新功能**（Gesture Recognizer + Kalman + 工具栏 + 光标叠加层），修改散布在 10+ 个文件中，缺少系统性测试。ML 模型和手写规则并存但没有清晰的优先级策略（B3），新功能（MouseCursorOverlay）留有未实现的空壳（B1），平滑系统的代码优雅但多个组件之间协调不足（B2/B6/B14）。
