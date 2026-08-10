# Handoff — person_pose_hand 侧位/远距手部识别引擎

> 2026-07-27 存档。下次接续直接读这份。
> **2026-07-27 二次更新：135° 短板已修复 + 延迟降 ~2.5×。见「实测数字」。**

## 一句话现状

新引擎 `person_pose_hand` 已完成并验证：**远距+侧位手部识别比 mediapipe 强很多，误检率最低**。全程在实验分支，未合主线，可回滚。

## 在哪

- **分支**：`experiment/person-pose-hand-crop`（master 停在 checkpoint `87b111f`）
- **回滚**：`git checkout master`（放弃实验：再 `git branch -D experiment/person-pose-hand-crop`）
- **核心文件**：
  - `app/services/person_pose_hand_tracker.py` — 引擎本体
  - `record_pose_matrix.py` — 录制助手（双击 `录制侧位视频.bat` / `录制远距视频.bat`）
  - `benchmark_pose_matrix.py` — 对比脚本（双击 `查看侧位对比.bat`）

## 引擎是什么（没训练新模型）

`框人(yolov8n-pose) → 拿手腕坐标 → 以手腕框手 → 小手超分 → MediaPipe HandLandmarker 出21点`
两个模型都是 2023 年的现成货，提升来自"先定位人再定位手"的策略，不是模型新。
- `models/yolov8n-pose.onnx` **不入 git**（Ultralytics AGPL），本地导出，见 MODEL_PROVENANCE.md

## 实测数字

**近距**：三引擎检出都 100%，无差距。但背对(180°)时 mediapipe/yolo 误检率 38%，person_pose **0%**。

**远距（检出率）**：
| 角度 | mediapipe | hagrid_yolo | person_pose |
|---|---|---|---|
| 0° | 60.6% | 98.0% | 100% |
| 90°侧 | 76.8% | 96.0% | **100%** |
| 135° | 35.4% | 100% | **100%**（原 91.9%，已修） |
| 180°背 | 68.7% | 93.9% | **100%** |

**结论**：mediapipe 远距崩；person_pose 全角度远距检出 100%（near 段 97–100%）。
**代价**：仍比 yolo 慢，但已大幅改善。P95 从 280–360ms 降到 **~100–130ms**（GPU DirectML）。

## 2026-07-27 二次改进（依据矩阵测试结果）

诊断（推翻了原"135° 是 pose 置信度掉"的猜测）：
- 135° 时 pose 层完全正常：人检出 100%、手腕关键点 conf 0.57–0.96、pose 仅 16ms。
- 真正根因：**小手提点前的 crop 太紧**。手腕框只包到腕/掌根，直接放大喂
  HandLandmarker 它看不到整只手 → 135°背侧位检出 48%。
- 延迟根因：Real-ESRGAN 每只远距小手都跑（框<96px 全触发），~55ms/手，占大头。

两处改动（都在 `person_pose_hand_tracker.py`）：
1. **crop 外扩上下文**：`_extract_landmarks_from_bboxes` 的小手路径，crop 前先以手腕
   框为中心外扩 `_HAND_CROP_CONTEXT=1.5` 倍再放大提点。135° 91.9%→100%。
2. **默认关掉 Real-ESRGAN**：新增 `person_pose_sr_engine`，默认 `"none"`（普通插值）。
   实测在外扩 crop 上 SR 与插值检出率完全一致（090/135/180/000 全平）但每帧省 ~54ms。
   要在极小手上试 SR：config 设 `person_pose_sr_engine="auto"`。

副作用（已知、待评估）：远距**多手率上升**（180_far 9%→36%）——外扩 crop 偶尔带进
第二只手/另一只腕。若双影/误锁影响体验需排查。

测试：504+180 全通过。A/B 结果存 `raw_capture/_improved_person_pose.json`（基线 `_baseline_person_pose.json`）。

## 已完成

1. ✅ 引擎骨架 + 工厂 + config 白名单（默认关闭）
2. ✅ yolov8n-pose.onnx 导出 + manifest
3. ✅ 9+5 项测试，全套 504+180 通过
4. ✅ 远距 A/B（91.6% 追平 yolo，误检 58%→30%）
5. ✅ CAPTURE 态可切：`engine_auto_switch_capture_engine=person_pose_hand`（orchestrator 已接好）
6. ✅ A4000/DirectML 修复（卸了冲突的标准 onnxruntime，重装 directml 1.24.4）
7. ✅ 录制助手（鼠标控制：左键开始/停止，右键下一段，C 切换摄像头）
8. ✅ 近距+远距姿态矩阵 A/B（数字见上）

## 下一步（你定，尚未做）

- **决定是否合主线**：person_pose 远距侧位最强，延迟已降到 ~100–130ms。选项：
  a) 设为 CAPTURE 态默认引擎（改 `engine_auto_switch_capture_engine` 默认值）
  b) 只作可选项，不动默认
  c) 还想更快再优化（见下）
- **多手率上升排查**：改进后远距多手率 9%→36%，确认是否引入双影/误锁再合主线
- **速度进一步优化**（可选）：P95 ~100–130ms 仍高于 yolo(125ms)。可减 pose 输入
  分辨率（_POSE_INPUT_SIZE 640→416）、或跳帧推理（adaptive_skip_enabled）
- **真实体验测试**：实际用三态闭环远距演示一遍手感

~~135° 偏低(91.9%)排查~~ → 已修复（见上「二次改进」），现 100%。
~~速度优化 P95 300ms~~ → 已降到 ~100–130ms。

## 环境注意

- 已把 `onnxruntime` 标准版卸了，只留 `onnxruntime-directml 1.24.4`（对齐 requirements.lock）。别再装标准版，会覆盖 DML。
- GPU 走 DirectML（A4000）。benchmark 报的延迟含 MediaPipe CPU 部分，实时使用更快。
