# Handoff — person_pose_hand 侧位/远距手部识别引擎

> 2026-07-27 存档。下次接续直接读这份。

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
| 0° | 60.6% | 98.0% | 97.0% |
| 90°侧 | 76.8% | 96.0% | **99.0%** |
| 135° | 35.4% | 100% | 91.9% |
| 180°背 | 68.7% | 93.9% | **98.0%** |

**结论**：mediapipe 远距崩；person_pose 在侧位/背对最强。135° 那段偏低(91.9%)是小瑕疵。
**代价**：慢。CPU P95 ~300ms（yolo 125ms）。GPU(DirectML)已修通，单帧 _detect 27ms@720p。

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

- **决定是否合主线**：person_pose 远距侧位最强但慢。选项：
  a) 设为 CAPTURE 态默认引擎（改 `engine_auto_switch_capture_engine` 默认值）
  b) 只作可选项，不动默认
  c) 先优化速度再合
- **135° 偏低(91.9%)排查**：可能那段手腕被身体遮挡，pose 置信度掉
- **速度优化**：P95 300ms 偏慢，可减 pose 输入分辨率 / 少跑 HandLandmarker crop
- **真实体验测试**：实际用三态闭环远距演示一遍手感

## 环境注意

- 已把 `onnxruntime` 标准版卸了，只留 `onnxruntime-directml 1.24.4`（对齐 requirements.lock）。别再装标准版，会覆盖 DML。
- GPU 走 DirectML（A4000）。benchmark 报的延迟含 MediaPipe CPU 部分，实时使用更快。
