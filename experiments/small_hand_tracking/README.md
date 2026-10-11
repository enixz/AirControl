# 小手跟踪第二版离线评测

入口使用项目现有 `.venv`。无需安装依赖。实验只读生产配置和录像，配置加载器在输出目录的副本上运行，结果及源码快照写入 `artifacts/small_hand_tracking/<run_id>/`。同名 run_id 会拒绝覆盖。

```powershell
.venv\Scripts\python.exe -m pytest experiments/small_hand_tracking/test_replay.py -q
.venv\Scripts\python.exe -m experiments.small_hand_tracking.run_replay --clips 20260722_145659 20260722_164055 --variants current_tracker A B C D --run-id smoke_new --max-frames 120
.venv\Scripts\python.exe -m experiments.small_hand_tracking.run_replay --clips 20260722_145659 20260722_164055 --variants A A2 B C D --run-id full_new
.venv\Scripts\python.exe -m experiments.small_hand_tracking.report artifacts/small_hand_tracking/full_new
.venv\Scripts\python.exe -m experiments.small_hand_tracking.verify_clock --run-id clock_new --frames 120
```

`--rounds 3` 轮换每轮起始组，`--delay-ms 10` 在每次推理前增加真实等待，用来核验回放时钟。当前尚未实现模型预热，三轮回放也不能宣称为预热性能基准。

本运行器适用于独立的单线程离线进程。它只在单次推理期间替换 tracker、motion 和 OneEuro 所属模块的时间绑定，退出或异常时恢复；实际性能测量始终使用真实时钟。不要在运行桌面应用的同一进程里调用它。

## 输出含义

- `baseline.json` 保存生效配置、模型与源码哈希、环境、命令参数；`source/` 保存源码副本。
- 每个组的 `predictions.jsonl` 保存原始检测、处理后输出、预测标记、模型输入尺寸、实际 SR 和逐次搜索路径。
- `events.jsonl` 保存 ZOOM 变化及观察位置，可找到普通日志没有写出的退出。
- `summary.json` 中新鲜主手输出率的分母为所有帧，不是人工确认手可见的帧。它不是准确度。
- `current_tracker` 只保留当前 tracker 参数。自动引擎切换由 orchestrator 执行，本工具不执行；不能称为完整应用基线。
- A/A2/B/C/D 均关闭补帧和主动跳帧；B/C/D 仅改变放大引擎，在线框轨迹可以因此改变。
- C/D 显式指定 SR，包含较大裁剪的调用；Z_auto 才是当前 auto 尺寸滞回与 GPU 选择策略。不能把强制 SR 的开销直接归给 auto。
- 缺少预测标记或手与元数据数量不匹配时标 unknown；副手的新鲜输出不会覆盖主手的预测状态。
- `no_fresh_primary_intervals` 包括出画和遮挡，结尾没有恢复的区间标记为右截尾。没有人工标注时不能称为丢手或重捕事件。

原 `experiments/hand_feasibility` 的 MP_FULL/B0 输出属于旧口径，不与第二版混合聚合。第二版保留真实 `track_key`，使用 `ml_label` 记录原始模型标签，handedness_score 不作为关键点质量分数。

## 搜索框短暂保留实验

`--variants B B_grace` 比较普通插值链路与短失锁搜索保留候选。B_grace 在无预测补帧条件下保留搜索框，直到既有 miss 帧阈值或 400ms 超时先到者；手部输出仍为空，旧身份和动作状态不恢复。仅用于实验，尚未进入生产代码。

`B_native` 是另一项独立实验：普通全图检测失败后，用原分辨率重试，重试间隔至少 200ms。它与 B_grace 不组合，便于判断增量收益。补搜成功仍需核对真实手身份。

2026 年 10 月 8 日首轮结果：B_grace 的远距新鲜主手输出覆盖从 93.55% 降至 92.90%；B_native 仅升至 93.60% 且增加耗时。两项候选均不建议据此进入生产，代码保留用于复现负结果。

## 当前边界

已新增应用逻辑回放和假控制器。独立人工标注、固定人工裁剪 SR 对照、暖机性能和真实采集到消费的帧龄尚待完成。相应指标明确填 null。生产默认值采用仍按开发计划的验收门槛执行。

## 应用逻辑回放

`application_replay.py` 运行生产模型、InferenceWorker 的 tracker 交接和状态迁移、Orchestrator 的自动切换与模式判断。鼠标、画布和演示控制器只记录调用。模式切换可修改私有内存配置，输入配置不落盘改动。

```powershell
.venv\Scripts\python.exe -m experiments.small_hand_tracking.run_application --run-id app_far_new --clips 20260722_145659 --variants current no_sr always_zoom --mode draw
.venv\Scripts\python.exe -m experiments.small_hand_tracking.run_application --run-id app_mouse_new --clips 20260722_164055 --variants current no_sr responsive_crop --mode mouse
.venv\Scripts\python.exe -m experiments.small_hand_tracking.application_report artifacts/small_hand_tracking/app_far_new --review-clip 20260722_145659
```

current 保留参数，no_sr 只关闭 SR，always_zoom 在 no_sr 上从开始启用远距搜索。responsive_crop 在 no_sr 上实验性加快搜索框中心响应；responsive_auto 在 current 上加同样改动。实验不改变裁剪尺寸平滑或直接生成关键点。

早期 identity_continuity 改写手的编号，虽有局部改善但出现新增失锁，未采用。其后修订为保留编号和 smoother，仅清除明确对应新鲜手的重复预测别名；此项已接入生产 `_assign_smoother_keys`，正式回归请使用 current 组。identity_continuity 实验入口保留用于追溯，在当前生产代码上不再构成独立对照。各历史实验的完整实现保存在运行目录 source 快照中。加快框中心响应的候选仍未采用。

运行前必须移除强制引擎的 AIRCONTROL_ENGINE 环境变量。每次使用新 run-id。应用程序本身不要与回放器在同一进程运行，后者会临时替换类方法和模块时间绑定。

这是完整识别与操作判断逻辑的离线回放，模型构建在两帧之间同步完成；排除真实摄像头调度、后台预热、Qt 排队、鼠标 watchdog 和窗口行为。因此不能称为实机端到端时延或操作成功率测试。控制器调用次数和手势输出数量也不能作为成功率。审查图像只标注采样帧，不能充当整个区间的人工真值。

## 预测清屏回归

2026 年 10 月 9 日起，生产板书逻辑在预测主手出现时清空张掌累计，预测帧不能触发清屏。新回放的 `operation_events` 补充了 `clear_canvas`、`toggle_shape_correction`，旧回放可从逐帧 `operation_counts` 查到这两项，不宜直接比较完整事件列表。

```powershell
.venv\Scripts\python.exe -m experiments.small_hand_tracking.verify_fresh_clear artifacts/small_hand_tracking/alias_retirement_draw_20261008 artifacts/small_hand_tracking/fresh_clear_far_20261009
```

核验会检查帧对齐、模型和配置一致、只变更板书源码、跟踪与笔画事件保持一致，以及没有非新鲜主手清屏。它不会把清屏减少认定为操作成功率改善；独立动作意图与实机响应需另行验收。

## 全部现有录像与历史对照

应用回放支持原始 `frames.mkv`、`frames.mp4` 或 `frames.avi`；同目录只允许一个原始视频。旧录制因队列丢帧可能保留跳号的采集索引，因此应用入口按解码顺序与元数据行顺序对齐，保留原始时间戳和 `capture_frame_index`。重复或倒序索引、时间戳倒退仍会拒绝，完整回放逐帧检查尺寸及帧数；不补造漏录的画面。其他工具调用 `read_metadata` 时仍默认要求连续编号。

`before_fixes` 使用指定源码快照中的原方法：`BaseHandTracker._assign_smoother_keys`、`DrawMode.handle` 和 `MouseMode.handle`。加载前检查整个应用的其他文件哈希，并用 AST 确认这些文件的其他代码一致。方法仅在实验进程内替换，退出或异常即恢复，历史方法共享当前模块的回放时钟绑定。历史源码另存于 `before_fixes_source`，来源和哈希记在 `before_fixes_provenance.json`。工作区源码不回退。较早的运行只替换前两个方法，以各运行的 provenance 为准。

```powershell
$clips = @(Get-ChildItem D:/AC-trae/raw_capture -Directory | Where-Object { Test-Path (Join-Path $_.FullName 'meta.jsonl') } | Sort-Object Name | Select-Object -ExpandProperty Name)
.venv\Scripts\python.exe -m experiments.small_hand_tracking.run_application --run-id corpus_draw_new --clips $clips --variants before_fixes current --mode draw --baseline-snapshot artifacts/small_hand_tracking/app_far_development_20261008
.venv\Scripts\python.exe -m experiments.small_hand_tracking.corpus_report artifacts/small_hand_tracking/corpus_draw_new --output artifacts/small_hand_tracking/corpus_review_new
```

鼠标初始模式另起新 run-id，改为 `--mode mouse`。汇总入口可接收两个运行目录；运行中只纳入已有 summary.json 的完整对照对，其他明确记为待完成。动作手势仍可切换模式，不能把“从板书开始”解释成整段始终处于板书模式。姿态目录名也不等于逐帧精确角度标注。

如果中途修复了新发现的问题，可以只重跑最终版 current，保留已经完成且输入不变的 before_fixes 结果。汇总时用 `--current-runs <最终版运行目录>` 按初始模式替换 current 来源；汇总会校验模型、输入配置、生效配置、视频、元数据及逐帧时间戳一致，并记录每组前后目录。不要把中间版本的 current 结果当作最终版验收。

## 同手改编号与交接实验

2026 年 10 月 11 日起，新鲜、首位且无歧义匹配到旧主手的别名清理会携带 `previous_track_key`。鼠标仅在该证据匹配当前旧编号时保留同手捏合历史。预测或其他编号不能授权继承，新鲜松手照常释放。回放将当帧证明记为 `same_hand_rekey` 事件，并保存最后新鲜鼠标特征以解释释放原因；该字段可能是历史缓存，不能独立当作当前新鲜检测。

```powershell
.venv\Scripts\python.exe -m experiments.small_hand_tracking.verify_rekey_fix artifacts/small_hand_tracking/guarded_corpus_draw_20261011 artifacts/small_hand_tracking/corpus_mouse_20261011 artifacts/small_hand_tracking/rekey_corpus_draw_20261011 artifacts/small_hand_tracking/rekey_corpus_mouse_20261011 artifacts/small_hand_tracking/rekey_review_20261011
.venv\Scripts\python.exe -m experiments.small_hand_tracking.run_application --run-id capture_handoff_new --clips pose_yaw090_far pose_yaw180_far 20260705_174137 --variants capture_handoff --mode mouse
```

`capture_handoff` 是未采用的候选：只提前 CAPTURE→FAR_TRACK，要求至少 5 个新鲜主手样本、编号与位置连续 ≥0.3 秒、捕获态驻留 ≥0.5 秒；预测、相邻双手、位置/尺度跳变和过长采样间隔重置证据。其余冷却、模式、模型和 SR 原样运行。第一批 3 段已出现新增缺失，不能据此调低生产默认值。`handoff_report` 对比生效配置时只允许此实验标志不同，并校验生产源码、模型、录像和时间戳一致。

```powershell
.venv\Scripts\python.exe -m experiments.small_hand_tracking.handoff_report artifacts/small_hand_tracking/rekey_corpus_mouse_20261011 artifacts/small_hand_tracking/capture_handoff_focus_20261011 artifacts/small_hand_tracking/capture_handoff_review_20261011
```

`probe_capture_crop` 是独立的当帧裁剪诊断：读取已保存的捕获态新鲜主手位置，按生产 padding 和最小尺寸直接裁剪，调用现有 MediaPipe 与 SR；每个试识别帧都重新获得捕获位置，不代表独立跟踪。输出只检查 MP 手腕是否在捕获手跨度的 0.75 倍以内，不能当作身份真值或准确率。它不运行操作控制器，也不修改生产交接逻辑。

```powershell
.venv\Scripts\python.exe -m experiments.small_hand_tracking.probe_capture_crop artifacts/small_hand_tracking/corpus_mouse_20261011 artifacts/small_hand_tracking/capture_crop_probe_new --clips pose_yaw090_far pose_yaw180_far
```

首轮两段共 89 个试识别帧、86 个几何匹配；提前交接候选的 11 个短片新增缺失帧中，10 个在此试识别中匹配。由于模型历史和裁剪策略同时变化，该结果用于优先排查交接后的裁剪与跟踪历史，不作为 SR 无需改进或某单一原因已被证实的结论。
