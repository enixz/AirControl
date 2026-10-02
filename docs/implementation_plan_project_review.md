# 实施方案：开源同类项目优点吸收

> 文档基于对 4 个 GitHub 同类项目的源码考察，结合 AC-trae 现有实现的逐文件对比分析得出。
> 所有借鉴点均已对照 AC-trae 实际代码验证，避免"为换新而换新"（参考项目 memory 中 Phase 2.5 教训）。

---

## 一、背景与目标

### 1.1 考察对象

| 项目 | 地址 | 语言/协议 | 定位 | 可参考度 |
|---|---|---|---|---|
| Air-Cursor | https://github.com/edic-23/Air-Cursor | Python / MIT / 2026-06 | 纯手势 Windows 鼠标控制，PyQt6 + MediaPipe HandLandmarker | 高（定位最接近） |
| opencv-example | https://github.com/ExpTechTW/opencv-example | Python / 2026-05 | OpenCV+MediaPipe 算法示例集（手部/头部/表情） | 高（算法参考） |
| PlayAbility | https://github.com/playAbilityTech/playAbility-desktop-releases | 闭源 / 仅二进制 | 多模态综合控制（语音+面部+头部） | 低（产品形态参考） |
| Gesture-Controlled-Virtual-Mouse | https://github.com/Viral-Doshi/Gesture-Controlled-Virtual-Mouse | Python / 2021-2023 | 早期 MediaPipe+语音练手项目 | 低（已过时） |

### 1.2 核心结论

**AC-trae 的算法层已显著领先这四个项目**。AC-trae 已有的先进特性：
- Kalman + OneEuro 双重平滑 + 几何约束滤镜（[smoothers.py](file:///d:/AC-trae/app/services/smoothers.py)）
- `hand_frontality` 门控 + 掌宽慢速 EMA（[gesture_recognizer.py:97-116](file:///d:/AC-trae/app/services/gesture_recognizer.py#L97-L116)）
- 动态自适应活动区映射 `ActiveRegionMapper`（[mouse_controller.py:91-178](file:///d:/AC-trae/app/services/mouse_controller.py#L91-L178)）
- crop-zoom + face_guide 长距离增强 + SR 引擎
- 四状态时序投票器 `TemporalGestureVoter`（[temporal_voter.py](file:///d:/AC-trae/app/services/temporal_voter.py)）
- MediaPipe GestureRecognizer Task（比 Air-Cursor 用的 HandLandmarker 更重）
- 多后端相机 fallback + MJPG 强制 + 分辨率自动探测（[camera.py](file:///d:/AC-trae/app/services/camera.py)）

**因此真正值得吸收的不是"更强的算法"，而是"更好的工程取舍和 UX 细节"**。

### 1.3 文档目标

- 记录每个借鉴点的对比分析，避免后人重复考察
- 明确标注哪些是"误判已收回"，防止二次踩坑
- 给出可执行的优先级排序和风险缓解方案

---

## 二、已纠正的误判（重要：避免后人重复）

在第一轮分析中，以下两个点曾被标为"强推"，但深入读 AC-trae 实际代码后发现 AC-trae 已有更完整的实现。**记录在此防止后人重新提出。**

### 2.1 误判一：多后端相机 fallback

| 维度 | Air-Cursor | AC-trae |
|---|---|---|
| 后端顺序 | DSHOW → MSMF → CAP_ANY | DSHOW → MSMF → 默认（[camera.py:61-74](file:///d:/AC-trae/app/services/camera.py#L61-L74)） |
| 打开失败回退 | 有 | 有（[camera.py:318-325](file:///d:/AC-trae/app/services/camera.py#L318-L325)） |
| MJPG 强制 | 无 | 有，且变更分辨率后二次设置 |
| 分辨率自动探测 | 无（固定 640×480） | 有，5 档候选 + 实测帧率 |
| FOURCC 校验日志 | 无 | 有 |
| 重连退避 | 无 | 指数退避（1.5s→30s 封顶） |

**结论**：AC-trae 是 Air-Cursor 的超集。**不吸收**。

### 2.2 误判二：Active Zone 映射

| 维度 | Air-Cursor | AC-trae |
|---|---|---|
| 区域类型 | 静态固定矩形（用户手动调 zone_x/y/w/h） | 动态自适应（[ActiveRegionMapper](file:///d:/AC-trae/app/services/mouse_controller.py#L91-L178)） |
| 适应方式 | 无 | 手移动时快速扩张+缓慢收缩，静止时冻结 |
| 距离分段 | 无 | `span_floor` 按掌宽/参考掌宽分段插值（[interp_tiers](file:///d:/AC-trae/app/services/mouse_controller.py#L181-L200)） |
| 书写冻结 | 无 | 有（[draw_mode.py:355-360](file:///d:/AC-trae/app/modes/draw_mode.py#L355-L360) `update=not freeze`） |
| 帧率无关 | 部分用时间戳 | 是（按秒计算收缩速率） |

**结论**：AC-trae 是 Air-Cursor 的超集。**不吸收**。

> 唯一可借鉴的细节：Air-Cursor 把 zone 参数作为 UI 滑块直接暴露，调试时容易理解"为什么光标这么灵敏"。AC-trae 的自动适应对用户友好但调试不透明。**建议：加一个 debug overlay 显示当前活动区边界，属于调优体验改进，非算法层。**

---

## 三、三个强推建议详细对比

### 3.1 建议一：Freeze-on-pinch（光标冻结）⭐⭐⭐

#### 借鉴对象
- **Air-Cursor** [tracker.py:324-339](https://github.com/edic-23/Air-Cursor/blob/main/air_cursor/tracker.py)
- 原始设计是 `freeze-on-fist`，AC-trae 本地化为 `freeze-on-pinch`（因为 AC-trae 的点击手势是拇指-食指捏合，不是握拳）

#### AC-trae 现状

| 文件 | 行号 | 现状 |
|---|---|---|
| [mouse_mode.py](file:///d:/AC-trae/app/modes/mouse_mode.py) | L18 | 注释："中指控制光标，拇指-食指捏合左键" |
| [mouse_mode.py](file:///d:/AC-trae/app/modes/mouse_mode.py) | L201-208 | 每帧调用 `move_to_normalized` 更新光标位置 |
| [mouse_mode.py](file:///d:/AC-trae/app/modes/mouse_mode.py) | L220 | `if features["thumb_index_pinch"]:` 触发 `left_down` |
| [mouse_mode.py](file:///d:/AC-trae/app/modes/mouse_mode.py) | L229-232 | pinch 上升沿调用 `left_down()` 但**不冻结光标** |

#### 问题分析

捏合瞬间（拇指快速接近食指）存在两类位置漂移源：
1. **手腕连带微动**：拇指快速屈曲时手腕会连带小幅移动
2. **landmark 本身抖动**：MediaPipe 在手指密集时关键点检测不稳定

结果：点击落点偏离用户瞄准位置。当前 [temporal_voter.py](file:///d:/AC-trae/app/services/temporal_voter.py) 只稳住"是否捏合"的标签判定，**没有稳住"光标位置"**。

#### 借鉴方案

Air-Cursor 在 fist 上升沿记录 `_frozen_pos` 和 `_freeze_start`，grace 期内（默认 2.0s）光标锁在冻结位置，grace 结束后解冻允许 DRAG。

#### AC-trae 已有的相似先例

[draw_mode.py:355-360](file:///d:/AC-trae/app/modes/draw_mode.py#L355-L360) 已经有 `freeze = is_drawing_now or self._was_writing`，传给 `ActiveRegionMapper.map(update=not freeze)`。这是"书写中冻结活动区映射"。**思路一致，只是没迁移到 mouse_mode 的 pinch-click 上。**

#### 吸收方案（不写代码，仅描述）

1. 在 [mouse_mode.py](file:///d:/AC-trae/app/modes/mouse_mode.py) 的 `MouseMode` 类中增加冻结状态字段：`_frozen_pos`、`_freeze_start`
2. 在 pinch 上升沿（`_is_left_holding` 从 False→True）记录当前 `screen_x, screen_y` 和 `time.time()`
3. 在 `move_to_normalized` 调用前判断：若处于 grace 期内且当前 pinch 中，跳过移动或直接 `set_cursor_pos` 到冻结位置
4. grace 结束后清空冻结状态，恢复正常移动（实现 DRAG）
5. grace 时长做成 config 项，建议默认 0.3s（比 Air-Cursor 的 2.0s 短，因为 pinch 动作比 fist 快）

#### 预期效果

| 维度 | 效果 |
|---|---|
| 点击精度 | 显著提升，落点=瞄准点 |
| 捏合抖动 | 消除捏合瞬间的光标漂移 |
| 拖拽体验 | grace 期内不能移动光标，但 left_down 已触发，grace 结束后可正常拖拽 |
| 现有逻辑影响 | 零，纯增量改动，不触动 `ActiveRegionMapper`、`temporal_voter`、`GestureRecognizer` |

#### 实施位置

- 主改动：[mouse_mode.py](file:///d:/AC-trae/app/modes/mouse_mode.py) 的 `MouseMode.handle()` 方法
- 配置项：[config.json](file:///d:/AC-trae/config.json) 增加 `pinch_freeze_grace_sec` 字段
- 不涉及其他文件

#### 风险与缓解

| 风险 | 等级 | 缓解 |
|---|---|---|
| grace 时长不当（太短不够用，太长影响拖拽手感） | 中 | 做成 config 项，提供 0.2/0.3/0.5s 三档实测对照 |
| 与 watchdog 死锁逻辑冲突 | 低 | watchdog 触发条件是"handle() 静默 + left_holding"，freeze 不影响 handle() 调用频率 |
| pinch 持续时间短于 grace | 低 | grace 是"上限"不是"固定时长"，pinch 释放时立即解冻 |

**总评：三个建议里风险最低、收益最直接。建议优先实施。**

---

### 3.2 建议二：Pinch 双阈值滞回 + 标定文档化 ⭐⭐⭐

#### 借鉴对象
- **Air-Cursor** [gestures.py:is_lateral_pinch](https://github.com/edic-23/Air-Cursor/blob/main/air_cursor/gestures.py) + [config.py](https://github.com/edic-23/Air-Cursor/blob/main/air_cursor/config.py)

#### AC-trae 现状

| 文件 | 行号 | 现状 |
|---|---|---|
| [gesture_recognizer.py](file:///d:/AC-trae/app/services/gesture_recognizer.py) | L35 | `PINCH_RATIO = 0.35` 单阈值，无标定注释 |
| [gesture_recognizer.py](file:///d:/AC-trae/app/services/gesture_recognizer.py) | L122-124 | `pinch_threshold = hand_width * self.PINCH_RATIO` |
| [gesture_recognizer.py](file:///d:/AC-trae/app/services/gesture_recognizer.py) | L212 | `"thumb_index_pinch": thumb_index < pinch_threshold` 单阈值无滞回 |
| [gesture_recognizer.py](file:///d:/AC-trae/app/services/gesture_recognizer.py) | L213 | `"thumb_middle_pinch": thumb_middle < pinch_threshold` 同上 |

下游 [mouse_mode.py:229-243](file:///d:/AC-trae/app/modes/mouse_mode.py#L229-L243) 用 `_is_left_holding` 做了状态机边沿检测（False→True 才 `left_down`），所以 pinch 抖动**不会导致重复点击**。但 pinch 特征本身在边界附近会逐帧跳变。

#### 问题分析

1. **边界抖动**：pinch 距离在阈值附近时，`thumb_index_pinch` 逐帧跳变
2. **下游脆弱**：虽然边沿检测能工作，但若某帧误判 pinch 消失→立即 `left_up`→下一帧又 pinch→`left_down`，理论上是隐患
3. **telemetry 噪声**：[mouse_mode.py:142-155](file:///d:/AC-trae/app/modes/mouse_mode.py#L142-L155) 每 15 帧记录的 `left_pinch` 字段在边界附近闪烁
4. **阈值无标定故事**：`PINCH_RATIO = 0.35` 没有注释说明为什么是 0.35，后人调参无依据

#### Air-Cursor 的关键教训（值得直接引用）

> The thumb-index distance is the reliable signal: a real pinch is much smaller than a fist, so `max_dist` (set between the two from live calibration) cleanly separates them. **We do NOT use the finger-extension guard — it proved unreliable on real webcam data; the distance threshold alone does the separation.**

Air-Cursor [config.py](https://github.com/edic-23/Air-Cursor/blob/main/air_cursor/config.py)：
- `pinch_on = 0.22`（进入滚动）
- `pinch_off = 0.32`（退出滚动，仍低于 fist 的 ~0.35）
- 滞回带 0.10
- 注释明确："a real pinch (~0.16) ... a fist (~0.35)"

**关键不是数值，是"阈值的标定锚点是 fist 距离"这个故事被文档化了。**

#### AC-trae 的现状对照

- AC-trae 的 pinch 判定里**没有**用 finger-extension guard（只用了距离）——这正好符合 Air-Cursor 的教训。**这点 AC-trae 已经做对了。**
- AC-trae 缺的是：(1) 滞回；(2) 标定文档。

#### 吸收方案

1. 在 [gesture_recognizer.py](file:///d:/AC-trae/app/services/gesture_recognizer.py) 的 `GestureRecognizer` 类常量区增加：
   - `PINCH_ENTER_RATIO`（进入捏合阈值）
   - `PINCH_EXIT_RATIO`（退出捏合阈值，> ENTER 形成滞回带）
2. 在 `__init__` 增加状态字段：`_was_thumb_index_pinch`、`_was_thumb_middle_pinch`
3. 在 `get_hand_features` 的 pinch 判定处加滞回：已捏合时用 EXIT（更宽松的保持），未捏合时用 ENTER（更严格的进入）
4. 在常量注释里文档化标定锚点："握拳时 thumb-index ≈ 掌宽×0.50+（fist 距离），所以 EXIT_RATIO 仍远低于 fist，握拳永远不会被误判为 pinch 释放"
5. 阈值具体数值需实测标定（参考 Air-Cursor 的归一化值，但 AC-trae 用掌宽比例，数值不同）

#### 预期效果

| 维度 | 效果 |
|---|---|
| pinch 状态稳定性 | 边界抖动消除，特征不再逐帧跳变 |
| 下游状态机 | 边沿检测更可靠（冗余防线） |
| telemetry 日志 | `left_pinch` 字段稳定，便于调试 |
| 阈值可维护性 | 有标定锚点，后人调参有依据 |

#### 实施位置

- 主改动：[gesture_recognizer.py](file:///d:/AC-trae/app/services/gesture_recognizer.py) 的 `GestureRecognizer` 类
- 不涉及 config.json（阈值是类常量，标定后固定）
- 可能涉及：[draw_mode.py](file:///d:/AC-trae/app/modes/draw_mode.py)、[orchestrator.py](file:///d:/AC-trae/app/orchestrator.py) 等下游（需先 grep 确认）

#### 风险与缓解

| 风险 | 等级 | 缓解 |
|---|---|---|
| `thumb_index_pinch` / `thumb_middle_pinch` 是 `get_hand_features` 返回的布尔特征，下游可能不止 mouse_mode 用 | **中** | 实施前先 grep 所有下游消费点，确认都希望滞回。若有不希望滞回的下游（如实时绘制），考虑在特征层加滞回、在绘制层用原始值 |
| 滞回带宽度需实测标定 | 中 | 参考 Air-Cursor 归一化值 0.10，AC-trae 用掌宽比例需重新标定。建议先加 telemetry 同时记录原始距离和滞回状态，对照真实手势标定 |
| 阈值标定需要采集 fist 距离数据 | 低 | 可以复用 [mouse_mode.py:142-155](file:///d:/AC-trae/app/modes/mouse_mode.py#L142-L155) 的 telemetry 机制扩展采集 |

---

### 3.3 建议三：`thumb_extended` 旋转不变判定 ⭐⭐⭐

#### 借鉴对象
- **opencv-example** [hand_direction.py:thumb_extended](https://github.com/ExpTechTW/opencv-example/blob/main/hand_direction.py)

#### AC-trae 现状

| 文件 | 行号 | 现状 | 问题 |
|---|---|---|---|
| [gesture_recognizer.py](file:///d:/AC-trae/app/services/gesture_recognizer.py) | L143 | `thumb_up = landmarks[4][2] < landmarks[3][2] and landmarks[4][2] < landmarks[2][2]` | 纯 y 坐标比较，手平放时失效 |
| [gesture_recognizer.py](file:///d:/AC-trae/app/services/gesture_recognizer.py) | L153 | `thumb_extended = thumb_tip_to_index_mcp > hand_width * self.THUMB_EXTEND_RATIO` | 距离判定，但混淆"向外伸"和"内收横跨掌心" |
| [gesture_recognizer.py](file:///d:/AC-trae/app/services/gesture_recognizer.py) | L162-166 | `is_thumbs_up` 依赖 `thumb_up`(y) + `thumb_tip[2] < thumb_mcp[2]`(z) + `THUMBS_UP_TIP_IP_DELTA = -15`(固定像素) | 强依赖 y+z，固定像素阈值有分辨率依赖 |
| [gesture_recognizer.py](file:///d:/AC-trae/app/services/gesture_recognizer.py) | L167-177 | `is_thumbs_down` 已改为掌宽比例（好），但仍依赖 y 坐标判定 | 部分改善 |

#### 问题分析

1. **`thumb_up` 强依赖 y 坐标**：手竖直向上时成立，手平放或侧对相机时失效
2. **`thumb_extended` 混淆内收与张开**：`thumb_tip_to_index_mcp` 距离在拇指内收横跨掌心时也大，误判为"伸出"
3. **`THUMBS_UP_TIP_IP_DELTA = -15` 固定像素**：注释里自己写了"避免固定 10px 在 1080p 下过松"，但 thumbs up 仍然是固定值

#### opencv-example 方案

```python
# 拇指 tip(4) 到「掌心中轴 (wrist 0 → middle MCP 9)」的垂直距离
# 以掌宽 (index MCP 5 ↔ pinky MCP 17) 当基准
perp = abs((t.x - w.x) * dy - (t.y - w.y) * dx) / axis_len
return perp > palm_width * 0.5
```

**为什么这个更好**：
- **旋转不变**：不管手怎么转，拇指张开向外时 tip 总是偏离中轴，perp 总是大
- **区分内收 vs 张开**：拇指内收横跨掌心时，tip 落在中轴附近，perp 小（正确判定为"未伸出"）；而 `thumb_tip_to_index_mcp` 距离在内收时也大（误判）
- **归一化到掌宽**：距离自适应，不依赖固定像素

#### 吸收方案

1. 在 [gesture_recognizer.py](file:///d:/AC-trae/app/services/gesture_recognizer.py) 的 `GestureRecognizer` 类增加辅助方法 `_thumb_perp_ratio(landmarks)`：计算拇指 tip 到掌心中轴（wrist→middle MCP）的垂直距离，归一化到掌宽
2. 在 `get_hand_features` 中：
   - **第一阶段（并存对照）**：同时计算旧特征（`thumb_tip_to_index_mcp` 距离）和新特征（`_thumb_perp_ratio`），都返回到 features dict
   - **第二阶段（切换）**：telemetry 日志对照确认新特征更稳定后，将 `thumb_extended` 改用新特征
3. 重新标定 `THUMB_EXTEND_RATIO` 阈值（perp_ratio 的数值范围和原距离比值不同）
4. 评估 `is_thumbs_up` / `is_thumbs_down` 的其他条件（z 比较、固定像素）是否一并替换

#### 预期效果

| 维度 | 效果 |
|---|---|
| thumbs up/down 识别场景 | 手平放、侧对相机时也能识别（原来这些场景失效） |
| thumb_extended 准确性 | 不再误判"内收横跨"为"伸出" |
| 分辨率依赖 | 消除 `THUMBS_UP_TIP_IP_DELTA = -15` 的固定像素依赖 |
| 旋转鲁棒性 | 显著提升，手部偏航时判定不再塌缩 |

#### 实施位置

- 主改动：[gesture_recognizer.py](file:///d:/AC-trae/app/services/gesture_recognizer.py) 的 `GestureRecognizer` 类
- 辅助：可能需要在 [mouse_mode.py](file:///d:/AC-trae/app/modes/mouse_mode.py) 的 telemetry 日志扩展（记录新旧特征对照）
- 不涉及其他文件

#### 风险与缓解

| 风险 | 等级 | 缓解 |
|---|---|---|
| 改的是核心特征提取，影响所有用 `thumb_extended` / `thumb_up` 的下游 | **高** | 先加新特征并存，用日志对照一段时间，确认更稳定后再切换。参考项目 memory "保留可回退的参考版本"原则 |
| 阈值需重新标定（perp_ratio 数值范围和原距离比值不同） | 中 | 复用 telemetry 机制采集真实手势数据标定。可以借鉴 [mouse_mode.py:142-155](file:///d:/AC-trae/app/modes/mouse_mode.py#L142-L155) 的 15 帧采样模式 |
| `is_thumbs_up` / `is_thumbs_down` 的其他条件是否一并替换需评估 | 中 | 分阶段实施：先替换 `thumb_extended`，观察效果后再评估 `is_thumbs_up/down` |
| 新算法在极端姿态（手完全侧对）下是否真的更稳 | 中 | 需实测对照，建议录制多种姿态的测试视频回放比对 |

**总评：三个建议里风险最高、收益最间接。建议最后实施，且必须并存对照。**

---

## 四、其他可借鉴但非强推的点

以下点有价值但优先级低于上述三个，记录备用。

### 4.1 Edge-triggered 一次性手势 + streak debounce（二星 ⭐⭐）

**借鉴**：Air-Cursor [tracker.py:298-302](https://github.com/edic-23/Air-Cursor/blob/main/air_cursor/tracker.py) 右键处理

**要点**：peace 手势需要连续 `gesture_debounce` 帧才触发，且**仅在 rising edge 发射一次**，peace 期间不重复触发。

**AC-trae 对照**：[mouse_mode.py:248-259](file:///d:/AC-trae/app/modes/mouse_mode.py#L248-L259) 右键用 `_is_right_pinching` 做了边沿检测（False→True 才 `right_click`），已基本等效。

**建议**：**对照检查即可，不一定需要改**。AC-trae 的边沿检测逻辑已经符合这个模式。

### 4.2 滚轮 sub-notch 累加（二星 ⭐⭐）

**借鉴**：Air-Cursor [tracker.py:_do_scroll](https://github.com/edic-23/Air-Cursor/blob/main/air_cursor/tracker.py) 用 `_scroll_residual` 累积小于一个 notch 的余数，慢速拖动也能平滑滚动。

**AC-trae 对照**：[gesture_recognizer.py](file:///d:/AC-trae/app/services/gesture_recognizer.py) 的 `SCROLL_THRESHOLD_RATIO = 1.2` 是阈值触发式，需确认是否有余数累加。

**建议**：检查 [mouse_mode.py:212-217](file:///d:/AC-trae/app/modes/mouse_mode.py#L212-L217) 的滚动实现，若慢速滚动卡顿则吸收。

### 4.3 Margin-to-margin sweep = Alt+Tab（二星 ⭐⭐）

**借鉴**：Air-Cursor [tracker.py:_detect_swipe](https://github.com/edic-23/Air-Cursor/blob/main/air_cursor/tracker.py) 挥动必须从画面边缘 ≤6% 起始，到达对侧 ≥94%，且在 0.9s 内完成。

**AC-trae 对照**：[gesture_recognizer.py:53-55](file:///d:/AC-trae/app/services/gesture_recognizer.py#L53-L55) 用 `EDGE_RATIO=0.12` + `SWIPE_DIR_CONSISTENCY=0.6`，是"方向一致性比例"。

**建议**：两者可叠加——边缘锚定防误触更强（中间区域挥手绝不触发），方向一致性防斜向误触发。**若当前 swipe 误触率高可考虑吸收，否则不动。**

### 4.4 Dwell-to-click 无障碍（一星 ⭐）

**借鉴**：Air-Cursor [tracker.py:_detect_dwell](https://github.com/edic-23/Air-Cursor/blob/main/air_cursor/tracker.py) 光标静止 1.0s 内 35px 范围 → 自动点击，配环形进度 UI。

**AC-trae 对照**：无，但 AC-trae 目前定位是健全人使用。

**建议**：**低优先级**，属于无障碍扩展范畴。若要做无障碍方向扩展再考虑。

### 4.5 curl_ratio 几何（参考 ⭐）

**借鉴**：Air-Cursor [gestures.py:curl_ratio](https://github.com/edic-23/Air-Cursor/blob/main/air_cursor/gestures.py) 每根手指的 `tip→palm / MCP→palm` 距离比，归一化到 hand_scale，0=全开、1+=全握。

**AC-trae 对照**：[gesture_recognizer.py:118-120](file:///d:/AC-trae/app/services/gesture_recognizer.py#L118-L120) 用 `landmarks[8][2] < landmarks[6][2]`（z 坐标比较），依赖深度通道且对手部旋转敏感。

**建议**：**可参考但不强推**。AC-trae 已有掌宽归一化和滞回，curl_ratio 是另一套思路，切换成本高、收益不确定。

### 4.6 架构层面：文件拆分（三星 ⭐⭐⭐ 但不强推）

**借鉴**：Air-Cursor 整个 `air_cursor/` 目录 14 个文件，最大 17.5KB，每个文件职责单一（`one_euro.py` 2.3KB、`gestures.py` 4.8KB、`mapping.py` 1.2KB）。

**AC-trae 对照**：
- [smoothers.py](file:///d:/AC-trae/app/services/smoothers.py) 已拆分（KalmanSmoother / OneEuroFilter / OneEuroSmoother / GeometricConstraintFilter 四个独立类，注释说"从 base_hand_tracker.py 拆出，降低上帝类体积"）
- [gesture_recognizer.py](file:///d:/AC-trae/app/services/gesture_recognizer.py) 529 行，但内部逻辑紧密耦合（`get_hand_features` 提取所有特征→返回 dict 给下游消费）

**建议**：**AC-trae 已经做过拆分工作（smoothers.py 就是明证），gesture_recognizer.py 虽然大但耦合紧密，强行拆分会增加状态共享成本**。这个建议部分成立但不强推——参考项目 memory "怀疑复杂度"原则，拆分本身也是复杂度，需评估净收益。

### 4.7 solvePnP 头部姿态解算（二星 ⭐⭐）

**借鉴**：opencv-example [head_direction.py](https://github.com/ExpTechTW/opencv-example/blob/main/head_direction.py) 取 FaceMesh 6 关键点 + 预设 3D 人脸模型 → `cv2.solvePnP` → Yaw/Pitch/Roll。

**AC-trae 对照**：[face_guide.py](file:///d:/AC-trae/app/services/face_guide.py) 用 `hand_frontality`（掌宽/食指长比值）作为正面度代理，是 2D 投影塌缩的间接测量。

**建议**：**值得 prototype，不一定值得落地**。若 `hand_frontality` 在长距离场景不够准，可考虑 PnP 解出头部姿态反推手部 3D 朝向。但代价是要跑 FaceMesh（4090D 算力没问题）+ 更复杂标定。**先 prototype 验证有真实收益再落地。**

### 4.8 Blendshape 表情识别（参考 ⭐）

**借鉴**：opencv-example [emotion_detect.py:classify_emotion](https://github.com/ExpTechTW/opencv-example/blob/main/emotion_detect.py) 用 ARKit blendshapes + 3 信号任 2 触发投票制识别 5 种表情。

**建议**：**当前不优先**。若以后要做"面部手势"（向 PlayAbility 方向扩展）这是现成配方。AC-trae 目前定位是手势控制，不在扩展方向上。

### 4.9 Proton 语音助手语法（一星 ⭐）

**借鉴**：Gesture-Controlled-Virtual-Mouse 的 Proton 命令清单：`Proton search {text}` / `Proton open {file_number}` / `Proton go back` / `Proton list files` / `Proton what is today's date` / `Proton Copy` / `Proton Paste` / `Proton bye` / `Proton wake up` / `Proton Exit`。

**AC-trae 对照**：[voice_command.py](file:///d:/AC-trae/app/services/voice_command.py) 用 doubao，主要是中文。

**建议**：**低优先级**。若 doubao 接入的语法已够用就不动。若想扩展系统操作类命令（文件导航、剪贴板），可参考这个命令清单。

---

## 五、明确不吸收的清单

为避免重蹈 Phase 2.5 覆辙，列出**看起来合理但不该吸收**的东西：

| 不吸收项 | 原因 |
|---|---|
| Air-Cursor 的 OneEuro 单滤波 | AC-trae 的 Kalman+OneEuro+GeometricConstraint 组合是实测可靠的，Air-Cursor 简单是因为它没做长距离增强。**简单不等于更好** |
| PlayAbility 的统一虚拟手柄抽象层 | 属于"听起来合理的架构改动"，需重构整个输入/输出管线。先做 A/B 对照再说 |
| Viral-Doshi 的彩色手套模块 | 4090D 算力足够，颜色检测是低配方案的妥协 |
| opencv-example 的 PnP 头部姿态（除非实测证明 `hand_frontality` 不够） | 你的 2D 代理已经工作，加 PnP 是引入新复杂度。先 prototype 验证有真实收益再落地 |
| Air-Cursor 的 `mouse_event` 替换 `SendInput` | AC-trae 的 [mouse_controller.py](file:///d:/AC-trae/app/services/mouse_controller.py) 已用 `mouse_event`，低延迟差异在毫秒级，除非实测有延迟问题否则不动 |
| Air-Cursor 的静态 Active Zone | AC-trae 的 `ActiveRegionMapper` 已是动态自适应超集（见第二节误判二） |
| Air-Cursor 的多后端相机 fallback | AC-trae 的 [camera.py](file:///d:/AC-trae/app/services/camera.py) 已完整实现且更先进（见第二节误判一） |

---

## 六、任务优先级与路线图

### 6.1 优先级矩阵

| 建议 | 收益 | 风险 | 实施难度 | 优先级 |
|---|---|---|---|---|
| **Freeze-on-pinch** | 高（直接解决点击漂移） | 低（纯增量，5-10 行） | 低 | **P0 立即** |
| **Pinch 双阈值滞回** | 中（稳定性提升，冗余防线） | 中（需确认下游，阈值标定） | 中 | **P1 短期** |
| **thumb_extended 旋转不变** | 中（扩展识别场景） | 高（改核心特征，需并存对照） | 中 | **P2 中期** |

### 6.2 实施路线图

#### Phase 3.1（P0，立即）：Freeze-on-pinch

**目标**：消除捏合瞬间的光标漂移，提升点击精度。

**任务清单**：
1. 在 [mouse_mode.py](file:///d:/AC-trae/app/modes/mouse_mode.py) 的 `MouseMode` 类增加冻结状态字段
2. 在 pinch 上升沿记录冻结位置和时间戳
3. 在 `move_to_normalized` 调用前判断 grace 期内是否冻结
4. 在 [config.json](file:///d:/AC-trae/config.json) 增加 `pinch_freeze_grace_sec` 配置项
5. 实测调优 grace 时长（建议测试 0.2/0.3/0.5s 三档）
6. A/B 对照：开启/关闭 freeze 的点击精度对比

**验证标准**：
- 连续 20 次瞄准同一目标点击，落点偏差从 X px 降到 Y px
- 拖拽体验：grace 期内能正常 `left_down`，grace 结束后能正常拖拽

#### Phase 3.2（P1，短期）：Pinch 双阈值滞回

**前置任务**：grep 确认 `thumb_index_pinch` / `thumb_middle_pinch` 的所有下游消费点。

**目标**：pinch 状态稳定，消除边界抖动，文档化阈值标定锚点。

**任务清单**：
1. grep 所有下游消费 `thumb_index_pinch` / `thumb_middle_pinch` 的代码
2. 评估每个下游是否希望滞回（若有不希望滞回的，考虑分层：特征层加滞回、特定下游用原始值）
3. 在 [gesture_recognizer.py](file:///d:/AC-trae/app/services/gesture_recognizer.py) 增加 `PINCH_ENTER_RATIO` / `PINCH_EXIT_RATIO` 常量
4. 增加 `_was_thumb_index_pinch` / `_was_thumb_middle_pinch` 状态字段
5. 在 `get_hand_features` 加滞回逻辑
6. 在常量注释文档化标定锚点（"握拳时 thumb-index ≈ 掌宽×0.50+，所以 EXIT_RATIO 仍远低于 fist"）
7. 扩展 telemetry 日志采集 fist 距离数据，标定阈值
8. A/B 对照：滞回前后 pinch 状态稳定性对比

**验证标准**：
- pinch 边界附近 100 帧内 `thumb_index_pinch` 跳变次数从 X 降到 0
- 握拳时绝不触发 pinch（fist 距离 > EXIT_RATIO）
- 阈值有注释说明标定依据

#### Phase 3.3（P2，中期）：thumb_extended 旋转不变判定

**目标**：thumbs up/down 在手平放/侧对时也能识别，thumb_extended 不再误判内收为伸出。

**任务清单**：
1. 在 [gesture_recognizer.py](file:///d:/AC-trae/app/services/gesture_recognizer.py) 增加 `_thumb_perp_ratio` 辅助方法
2. 在 `get_hand_features` 同时计算旧特征（`thumb_tip_to_index_mcp` 距离）和新特征（`_thumb_perp_ratio`），都返回到 features dict
3. 扩展 [mouse_mode.py](file:///d:/AC-trae/app/modes/mouse_mode.py) 的 telemetry 日志，同时记录新旧特征值
4. 录制多种姿态测试视频（手竖直、平放、侧对、拇指张开、拇指内收）
5. 对照日志分析新特征是否真的更稳定
6. 若验证通过：将 `thumb_extended` 改用新特征，重新标定 `THUMB_EXTEND_RATIO`
7. 评估 `is_thumbs_up` / `is_thumbs_down` 的其他条件是否一并替换
8. A/B 对照：旧特征 vs 新特征在极端姿态下的识别率

**验证标准**：
- 手平放时 thumbs up 识别率从 X% 提升到 Y%
- 拇指内收横跨掌心时 `thumb_extended` 不再误判为 True
- 录制测试视频回放，新旧特征对照日志显示新特征方差更小

---

## 七、风险评估矩阵

| 风险类别 | 具体风险 | 影响范围 | 发生概率 | 缓解措施 |
|---|---|---|---|---|
| **回归** | Freeze-on-pinch 的 grace 时长影响拖拽手感 | mouse_mode | 中 | 做成 config 项，提供多档实测对照 |
| **回归** | Pinch 滞回影响不希望滞回的下游 | 所有 pinch 消费方 | 中 | 实施前 grep 确认，必要时分层处理 |
| **回归** | thumb_extended 改动影响所有用该特征的下游 | gesture_recognizer + 所有下游 | 高 | 并存对照，先加新特征不删旧特征 |
| **标定** | Pinch 滞回阈值数值不当 | mouse_mode 稳定性 | 中 | 复用 telemetry 机制采集数据标定 |
| **标定** | thumb_perp_ratio 阈值数值不当 | thumbs up/down 识别 | 中 | 录制测试视频回放标定 |
| **架构** | gesture_recognizer.py 改动引入新 bug | 全局 | 中 | 充分单元测试 + A/B 实测对照 |
| **兼容性** | config.json 新增字段影响旧版本 | 配置加载 | 低 | 参照 Air-Cursor 的 clamp() 模式，缺失字段回退默认值 |

---

## 八、验证方法

### 8.1 单元测试

每个 Phase 实施后补充单元测试：
- Phase 3.1：测试 freeze 状态机的状态转换（IDLE→FROZEN→DRAGGING）
- Phase 3.2：测试滞回逻辑（ENTER/EXIT 阈值边界）
- Phase 3.3：测试 `_thumb_perp_ratio` 在不同姿态下的数值范围

**注意**：参考项目 memory 教训——"单元测试通过 ≠ 实际改进"。单元测试只测逻辑本身对不对，不测实际手势场景效果。**必须配合 A/B 实测对照。**

### 8.2 A/B 实测对照

每个 Phase 实施后做 A/B 对照：
- 录制固定测试动作序列（如"瞄准屏幕 5 个目标点击 20 次"）
- 开启/关闭新特性分别录制
- 对比指标：点击偏差、状态跳变次数、识别率

### 8.3 保留可回退的参考版本

参考项目 memory "保留可回退的参考版本"原则：
- 每个 Phase 实施前 git tag 当前稳定版本
- 新特性做成 config 开关（如 `pinch_freeze_enabled`），可一键关闭回退
- thumb_extended 改动保留旧特征计算，通过 config 切换新旧

---

## 九、参考索引

### 9.1 借鉴项目源码定位

| 项目 | 关键文件 | 用途 |
|---|---|---|
| Air-Cursor | [one_euro.py](https://github.com/edic-23/Air-Cursor/blob/main/air_cursor/one_euro.py) | One Euro 滤波实现（参考，AC-trae 已有更复杂版本） |
| Air-Cursor | [gestures.py](https://github.com/edic-23/Air-Cursor/blob/main/air_cursor/gestures.py) | curl_ratio + FistDetector 滞回（参考滞回设计） |
| Air-Cursor | [tracker.py](https://github.com/edic-23/Air-Cursor/blob/main/air_cursor/tracker.py) | freeze-on-fist + swipe + dwell（主要借鉴源） |
| Air-Cursor | [config.py](https://github.com/edic-23/Air-Cursor/blob/main/air_cursor/config.py) | 配置 schema + clamp() 模式 |
| Air-Cursor | [mapping.py](https://github.com/edic-23/Air-Cursor/blob/main/air_cursor/mapping.py) | Active Zone 映射（AC-trae 已有超集） |
| opencv-example | [hand_direction.py](https://github.com/ExpTechTW/opencv-example/blob/main/hand_direction.py) | thumb_extended 旋转不变判定（主要借鉴源） |
| opencv-example | [head_direction.py](https://github.com/ExpTechTW/opencv-example/blob/main/head_direction.py) | solvePnP 头部姿态（备用参考） |
| opencv-example | [emotion_detect.py](https://github.com/ExpTechTW/opencv-example/blob/main/emotion_detect.py) | Blendshape 表情识别（备用参考） |

### 9.2 AC-trae 对应文件

| 模块 | 文件 | 关键行号 |
|---|---|---|
| 鼠标模式 | [mouse_mode.py](file:///d:/AC-trae/app/modes/mouse_mode.py) | L201-260（pinch 处理）、L142-155（telemetry） |
| 板书模式（冻结先例） | [draw_mode.py](file:///d:/AC-trae/app/modes/draw_mode.py) | L355-360（freeze 模式） |
| 手势识别 | [gesture_recognizer.py](file:///d:/AC-trae/app/services/gesture_recognizer.py) | L35（PINCH_RATIO）、L122-124（pinch 阈值）、L143（thumb_up）、L153（thumb_extended）、L162-177（thumbs up/down） |
| 鼠标控制 | [mouse_controller.py](file:///d:/AC-trae/app/services/mouse_controller.py) | L91-178（ActiveRegionMapper）、L203-266（MouseController） |
| 时序投票器 | [temporal_voter.py](file:///d:/AC-trae/app/services/temporal_voter.py) | L62-237（四状态 FSM） |
| 平滑器 | [smoothers.py](file:///d:/AC-trae/app/services/smoothers.py) | L25-114（Kalman）、L121-285（OneEuro）、L308+（几何约束） |
| 相机 | [camera.py](file:///d:/AC-trae/app/services/camera.py) | L61-74（多后端）、L239-466（CameraService） |
| 配置 | [config.json](file:///d:/AC-trae/config.json) | 全文件 |

---

## 十、附录：反思原则（参考项目 memory）

本次实施方案严格遵循项目 memory 中的反思原则：

1. **先理解现有设计的理由，再考虑"改进"**——第二轮深入读 AC-trae 实际代码后纠正了 2 个误判（多后端 fallback、Active Zone），这正是"没先理解就下结论"的反面教材
2. **怀疑复杂度**——三个建议都是增量改动，不引入新架构（如统一虚拟手柄抽象层就被列为"不吸收"）
3. **单元测试通过 ≠ 实际改进**——方案强调 A/B 实测对照，不只依赖单元测试
4. **保留可回退的参考版本**——每个 Phase 都做成 config 开关，可一键回退
5. **警惕"听起来合理的设计目标"**——PlayAbility 的"统一虚拟手柄"听起来合理但需重构整个管线，列为"不吸收"先做 A/B 对照
