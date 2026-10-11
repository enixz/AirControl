# AirControl 手部识别路线可行性论证

状态：P0/P1 完成，P2 冒烟通过。方案见 `docs/hand_model_feasibility_plan.md`。

## 目录结构

```text
protocol.json          统一评测协议（P1 评审稿，阈值为待冻结的工程目标）
adapters/              候选路线适配器（base_adapter.py 契约 + 各路线实现）
tools/                 run_adapter.py（统一运行器）、strict_scorer.py（严格评分器）
tests/                 test_strict_scorer.py（评测自检）
```

## 已交付（P0/P1）

- `artifacts/hand_feasibility/p0_baseline/environment.json` — 基线快照（git/硬件/配置/录像位置）
- `artifacts/hand_feasibility/p0_baseline/effective_config.json` — ConfigManager 合并后生效配置
- `artifacts/hand_feasibility/p0_baseline/code_audit.md` — 12 条调用链声明的逐条验证
- `artifacts/hand_feasibility/route_evidence.csv` — 五条路线资料核验表
- `experiments/hand_feasibility/protocol.json` — 评测协议
- `experiments/hand_feasibility/adapters/base_adapter.py` — 适配器契约 + ReplayClock
- `experiments/hand_feasibility/tools/strict_scorer.py` — 严格评分器（修正 truth_ab 0.5s 宽容口径）
- `experiments/hand_feasibility/tools/run_adapter.py` — 统一运行器

## 复现命令

```bash
# 评测自检（10 项）
D:/airControl/.venv/Scripts/python.exe D:/airControl/experiments/hand_feasibility/tests/test_strict_scorer.py

# MP_FULL 适配器冒烟（30 帧）
D:/airControl/.venv/Scripts/python.exe experiments/hand_feasibility/tools/run_adapter.py "D:/AC-trae/raw_capture/20260722_161024" --route MP_FULL --max-frames 30 --run-id p2_smoke
```

## 关键事实（与方案预期的差异）

1. **录像数据缺口不存在**：方案担心的 raw_capture 缺失，实际在 `D:/AC-trae/raw_capture`（7 个会话、4 个带 truth_events.jsonl、1920×1080、27FPS、meta.jsonl 带采集时间戳）。全部为单人单会话——满足先导集，不满足正式集 3 人要求。
2. **ReplayClock 已存在**：`artifacts/recognition_eval_20261003/run_eval.py:106-120`（diagnose_clicks.py 在用）。T0 不需要新造；正式副本已在 base_adapter.py 提供。
3. **engine_auto_switch 当前激活**（config.json engine_auto_switch=true）：FSM 的 CAPTURE 态会换 hagrid_yolo 引擎——B0 基线回放必须记录每帧实际引擎，不能全叫 MediaPipe。
4. **手势标签硬门槛确认**：Victory/ILoveYou/Thumb_Up 全部来自 MediaPipe gesture_recognizer.task；纯关键点模型（RTMPose/YOLO26）没有标签——方案 §6.4 标签适配是替换的前提条件。
5. **OneEuro predict() 是保持上帧**（非外推）：预测帧的实际行为是位置冻结，对"预测补帧"指标的解释有影响。
6. **MediaPipe 时间戳是墙钟毫秒**（time.time()*1000 强制单调），非帧号、非 monotonic——离线回放时传录像时间戳保持一致。

## 剩余工作（P2-P5）

- R1a/R1b/R3 权重下载与适配器（等资料核验 agent 返回后启动）
- R2 WiLoR：PyTorch 独立环境（不覆盖 onnxruntime-directml）
- R4 SAM 3D Body：HF 门控申请 + Detectron2 Windows 编译风险评估
- R5 Hand-4DGS：代码/权重可得性核实
- 先导集标注（需真人录制，本轮交付采集清单）
