# AirControl 架构审查与改进建议

## 一、架构审查

### 1.1 总体评价

项目整体是一个**功能完整的原型级产品**，摄像头→手部追踪→手势识别→动作执行的流水线清晰。代码能跑、功能能用，但距离工程化产品还有明显的架构和科学性差距。

### 1.2 架构问题

#### 问题 P1：FloatingWindow 是 God Object（严重）

`main_ui.py` 中的 `FloatingWindow` 类 **539 行**，同时承担了：

- UI 初始化与渲染（窗口、按钮、标签、拖动）
- 三种模式的手势处理逻辑（`_handle_presentation_mode`, `_handle_mouse_mode`, `_handle_draw_mode`）
- 模式切换与调度（`_cycle_mode`, `_maybe_switch_mode_by_two_fists`）
- 手部特征提取（`_get_hand_features`，与 `GestureRecognizer.get_hand_pose` 重复）
- 配置同步（`apply_config`, `_sync_overlay_state`）
- 动作执行（`execute_action`）
- 状态管理（`status_text`, `status_color`, 各种计时器）

**问题**：模式逻辑散落在 UI 类中，新增模式需要改 FloatingWindow；UI 变动可能破坏手势逻辑；代码不可测试。

**建议**：引入策略模式，将每种模式独立为一个策略类，FloatingWindow 只负责 UI 和调度。

#### 问题 P2：手势识别逻辑重复（中等）

`GestureRecognizer.get_hand_pose()` 和 `FloatingWindow._get_hand_features()` 都在做手指伸展/握拳/捏合的判断，但判断标准**不一致**：

| 判断 | GestureRecognizer | FloatingWindow._get_hand_features |
|------|-------------------|----------------------------------|
| 握拳 | 四指弯曲 + 距手腕距离阈值 | 仅四指弯曲（更宽松） |
| 捏合 | 未实现 | thumb_index / thumb_middle 距离 |
| 点赞 | 拇指尖在掌根上方 | 未实现 |

这导致同一帧画面在两处得到不同的手势判定，行为不一致。

**建议**：统一到一个 `HandFeatures` 数据类中，由单一来源产出特征，两处消费。

#### 问题 P3：main.py 与 main_ui.py 存在大量重复（中等）

`main.py`（命令行版）和 `main_ui.py`（GUI 版）的核心循环逻辑几乎完全重写了一遍，手势→动作的映射逻辑在两处各实现了一遍。这违反 DRY 原则，且 `main.py` 不支持鼠标/板书模式。

**建议**：`main.py` 作为简化版本可以保留，但核心管道（追踪→识别→调度→执行）应从 UI 中抽离。

#### 问题 P4：线程安全与帧率问题（中等）

- `update_frame()` 在 QTimer 主线程中执行，包含 OpenCV 编解码 + MediaPipe 推理 + QImage 转换，全部阻塞 UI 线程
- `self.timestamp_ms += 33` 是硬编码的固定时间步，实际帧间隔并不固定，MediaPipe VIDEO 模式要求时间戳严格递增但应反映真实时间

**建议**：将摄像头读取和 MediaPipe 推理移到独立线程，用信号槽传递结果到 UI 线程。时间戳改为真实毫秒时间。

#### 问题 P5：配置管理每次 set 都写磁盘（轻微）

`ConfigManager.set()` 每次调用都 `json.dump` 到磁盘。设置面板保存时连续调用了 13 次 `set()` / `set_mapping()`，导致 13 次文件写入。

**建议**：增加 `batch_update()` 方法，或改为脏标记延迟写入。

### 1.3 模型选择评估

#### 当前：MediaPipe HandLandmarker + 自定义规则

项目使用 `hand_landmarker.task`（或 `heavy` / `full` 变体）获取 21 个关键点，然后通过手写的 Python if-else 规则做手势分类。

**优点**：
- MediaPipe HandLandmarker 是目前最成熟的手部关键点检测方案，GPU 加速、端侧推理、跨平台
- Heavy 模型精度高，适合桌面场景
- 光流补偿是很好的补充，增强了跟踪连续性

**问题**：
- 手势分类完全依赖手写规则（基于像素坐标的阈值判断），**不可泛化、不可学习、难以调优**
- 并拢判断 `dx1 < 60 or dx1 < hand_width * 0.8` 中 60 是硬编码像素值，与分辨率和手部距离强耦合
- 不支持连续手势（如"挥手"需要完整的轨迹判断，当前只是首尾帧差值）
- 缺少手指方向判断（指尖 y < 指节 y 的判断在手掌倾斜时失效）

#### 建议：升级到 MediaPipe Gesture Recognizer Task API

Google 已经提供了 `gesture_recognizer.task` 模型，内置 **8 种预定义手势**：

| 手势 | 说明 |
|------|------|
| Closed_Fist | 握拳 |
| Open_Palm | 张开手掌 |
| Pointing_Up | 食指指向上方 |
| Thumb_Down | 差评 |
| Thumb_Up | 点赞 |
| Victory | 剪刀手/V字 |
| ILoveYou | 摇滚手 |

**优势**：
- 基于深度学习的分类器（gesture_embedder + canned_gesture_classifier），比手写规则鲁棒得多
- 内置归一化和手部方向不变性，不怕手掌倾斜
- 一个模型文件同时输出关键点 + 手势类别，减少代码量
- 支持通过 `mediapipe-model-maker` 训练自定义手势分类器

**迁移成本**：低。`gesture_recognizer.task` 包含了 `hand_landmarker` 的全部功能，API 结构几乎一样，只需替换模型文件和初始化代码。

### 1.4 代码科学性审查

| 问题 | 位置 | 严重性 | 说明 |
|------|------|--------|------|
| 时间戳硬编码 | `hand_tracker.py:112` | 中 | `self.timestamp_ms += 33` 不反映真实帧时间，在帧率波动时会导致追踪质量下降 |
| 光流点选取过少 | `hand_tracker.py:71` | 中 | 只追踪 5 个点(0,5,9,13,17)，手部快速运动时估计偏差大 |
| 挥动检测过于简单 | `gesture_recognizer.py:138-139` | 中 | 只比较轨迹首尾帧的位置差，忽略了中间路径，容易被"画弧"动作误触发 |
| 握拳判定不含拇指 | `gesture_recognizer.py:61-68` | 低 | 握拳时拇指位置是关键特征，当前完全忽略拇指(landmark 1-4) |
| 异常吞没 | `mouse_controller.py:38-39` | 中 | `except Exception: pass` 会隐藏所有鼠标控制错误，包括权限问题 |
| 日志初始化在模块级 | `gesture_recognizer.py:7-18` | 低 | `logging.basicConfig` 在模块导入时执行，会覆盖调用方的日志配置 |
| QImage 数据引用 | `main_ui.py:506` | 中 | `QImage(rgb_image.data, ...)` 引用了 numpy 数组的原始内存，如果 numpy 释放了这块内存，QImage 会读取到无效数据 |
| os.system 注入风险 | `ppt_controller.py:44,53` | 低 | `os.system("start wpp")` 虽然参数是硬编码的，但使用 `subprocess.run` 更规范 |

---

## 二、改进路线图

### Phase 1：低风险高收益（1-2 天）

1. **替换为 MediaPipe Gesture Recognizer Task API**
   - 下载 `gesture_recognizer.task` 模型
   - 重构 `GestureRecognizer`，使用 `GestureRecognizer.create_from_options()` 替代手写规则
   - 删除 `get_hand_pose()` 中的所有手写判断逻辑
   - 保留挥动检测逻辑（GestureRecognizer 不含动态手势，需自行实现）

2. **修复 QImage 内存安全问题**
   - 改用 `.copy()` 确保 QImage 持有独立数据：
     ```python
     qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
     ```

3. **修复时间戳问题**
   - 改为真实时间：
     ```python
     self.timestamp_ms = int(time.time() * 1000)
     ```

4. **修复 os.system → subprocess.run**
   - `subprocess.run(["cmd", "/c", "start", "wpp"], shell=False)`

### Phase 2：架构重构（3-5 天）

5. **抽取模式策略类**
   ```
   app/modes/
   ├── __init__.py
   ├── base.py          # ModeBase 抽象类
   ├── presentation.py  # PresentationMode
   ├── mouse_mode.py    # MouseMode
   └── draw_mode.py     # DrawMode
   ```
   每个模式类实现 `handle(hands_landmarks, frame_w, frame_h) -> GestureResult`

6. **引入 ModeManager**
   - 管理模式切换、冷却、双手握拳检测
   - FloatingWindow 不再直接处理模式逻辑

7. **统一 HandFeatures**
   - 创建 `app/features.py`，将 `_get_hand_features` 和 `get_hand_pose` 合并为一个统一的手部特征提取器
   - 输出结构化的 `HandFeatures` 数据类

8. **异步推理管道**
   - 摄像头读取 + MediaPipe 推理放在 QThread
   - 通过 pyqtSignal 传递结果到 UI 线程
   - 预期提升：UI 不再卡顿，帧率从 ~25fps 提升到 ~30fps

### Phase 3：体验提升（5-7 天）

9. **板书模式增强**
   - 支持多色画笔（颜色选择器）
   - 支持橡皮擦模式（五指张开 + 移动）
   - 支持撤销/重做（保存每步操作的画布快照）
   - 画笔平滑改用 Catmull-Rom 样条插值替代线性平滑

10. **鼠标模式增强**
    - 支持滚轮（双指上下滑动）
    - 支持拖拽（捏合保持 + 移动）
    - 鼠标加速度曲线（快速移动时加速，慢速时精准）

11. **自定义手势训练**
    - 使用 `mediapipe-model-maker` 训练自定义手势分类器
    - 在设置界面提供"录制手势"功能
    - 导出为 `custom_gesture_classifier.tflite`

12. **多显示器支持**
    - MouseController 当前只用主显示器分辨率
    - 改为 `win32api.GetSystemMetrics(78/79)` 获取虚拟屏幕尺寸

### Phase 4：质量保障

13. **单元测试**
    - 为 GestureRecognizer、HandFeatures、ConfigManager 编写测试
    - 使用录制的 landmark 数据作为测试 fixture

14. **性能监控**
    - 添加 FPS 显示和推理延迟统计
    - 记录 MediaPipe 推理耗时和光流补偿触发率

15. **错误恢复**
    - 摄像头断开重连
    - MediaPipe 推理异常时自动降级（降低分辨率或切换 Lite 模型）

---

## 三、总结

| 维度 | 当前状态 | 改进后预期 |
|------|----------|-----------|
| 架构 | God Object，UI 与逻辑耦合 | 策略模式，职责分离 |
| 模型 | 手写规则，不可泛化 | MediaPipe Gesture Recognizer（ML 分类） |
| 帧率 | ~25fps，UI 卡顿 | ~30fps，推理异步化 |
| 可扩展性 | 新增模式需改 FloatingWindow | 新增 ModeBase 子类即可 |
| 可靠性 | 异常吞没，内存不安全 | 完善错误处理和恢复 |
| 手势识别率 | 中等（规则限制） | 高（ML 分类 + 改进挥动检测） |

**最高优先级建议**：先做 Phase 1 的 Gesture Recognizer 替换，这是性价比最高的改进——改动量小，但识别精度和鲁棒性会有质的提升。
