# Changelog

## Unreleased

AirControl 2.0 cloud intelligence layer (M1–M6): an optional OpenAI-compatible
LLM layer on top of the fully-local v1.4 core, plus repo hygiene. All cloud
features are **off by default** and gated behind `cloud.enabled` + an
environment-variable API key; core gesture/voice features remain 100% local.

- **Cloud provider layer (M1)**: `app/services/cloud/` — provider presets
  (MoMA / custom OpenAI-compatible), key via env var only (never in
  config.json), httpx client with retry/backoff (429 minute-window aware),
  per-call token metering to `logs/token_usage.jsonl`, nested-schema config
  validation for the `cloud` section.
- **LLM smart command (M3)**: wake word `小助手` → one-sentence recording →
  local SenseVoice transcription → cloud LLM intent parsing → whitelisted
  action execution. Action whitelist derived from `voice_actions.py`,
  narrowed by current mode; confidence threshold + JSON validation guard
  against overreach.
- **Classroom AI (M4)**: `下课总结` (class summary) / `出三道题` (quiz) —
  accumulated dictation transcript → Markdown summary / practice questions
  via cloud LLM, exported to `data/notes/` and written to the canvas.
  Board-to-notes (vision model) implemented, UI wiring pending.
- **Local voice feedback (M5)**: Windows SAPI spoken confirmations (mode
  switches, generation results, refusals). Microphone frames are dropped
  while speaking to prevent self-triggering.
- **Cloud health monitor (M6)**: heartbeat ping (1-token chat), 3 consecutive
  failures → degraded; floating-window status dot (gray/green/yellow).
  Degraded state **blocks cloud features at the trigger** with a spoken hint
  instead of waiting through retries (`auto_fallback`).
- **Silero VAD for smart-command endpointing**: neural speech detection
  (MIT, 629KB, `models/silero_vad.onnx`, auto-downloaded, packaged) replaces
  the classroom-fragile RMS threshold; energy VAD remains the fallback when
  the model is absent. Measured: real speech closes the session 1.4s after
  speech end (RMS method hit the 8s cap).
- **Fast-channel model A/B (2026-10-02)**: intent-parsing latency p50
  1.06s on `deepseek-v4.1-flash` vs 2.38s (DeepSeek-V3) / 1.8–6.7s (GLM-5.2);
  flash stays the default. Measured pitfall documented: `max_tokens < 256`
  lets reasoning drain the budget and return empty content (see
  `docs/moma_usable_models.md`).
- **Repo hygiene**: dead ASR/TTS stub interfaces removed (MoMA free tier
  verified to exclude cloud voice; recoverable from git history); root-level
  experiment scripts moved to `scripts/`, handoff/release docs to `docs/`;
  `logs/` and 软著登记材料/ gitignored; `config.json` local drift reverted.
- **requirements.lock**: httpx chain (6 packages) pinned from the
  2026-10-01 verified venv snapshot — new installs no longer crash on the
  cloud import chain.

## v1.4.0 - 2026-08-10

AirControl 1.4 consolidates the locally iterated pinch-stability work, the
HaGRID YOLO far-range engine, ground-truth A/B validation, and the engine
auto-switch state machine into a single public minor release over v1.3.5.

The three new gesture layers are available for controlled trials. The
EXIT-only pinch hysteresis (`pinch_exit_hysteresis_enabled`) is now **on by
default** after a 14-event ground-truth A/B showed it preserves recall while
reducing false alarms. The freeze and dual-threshold modes remain off.

- **YOLO model policy change**: `hand_yolov8n.onnx` (AGPL-3.0) is no longer
  packaged into the release bundle. `build.py` verifies that `AirControl.spec`
  does not include the model; users who need the `hagrid_yolo` engine or
  `engine_auto_switch` feature download it separately (see README). The
  default `mediapipe` engine is unaffected.
- **Experimental `person_pose_hand` engine + A4000/DirectML GPU fix**
  (branch `experiment/person-pose-hand-crop`): new capture engine
  框人→yolov8-pose 拿手腕→框手→（小手超分）→HandLandmarker 关键点, selectable
  via `detection_engine=person_pose_hand` or as the CAPTURE-state engine via
  `engine_auto_switch_capture_engine`. Far-range A/B (`raw_capture/20260722_145659`,
  2000 frames): detect 91.6% (≈ hagrid_yolo 92.8%) but multi-hand 29.6% vs 58.1%
  (wrist anchoring rejects false hands); enabling SR on small hands improves all
  four metrics (detect↑/multi↓/jerk↓/latency↓). `models/yolov8n-pose.onnx` is
  NOT committed (Ultralytics AGPL; see MODEL_PROVENANCE.md) and must be exported
  locally. **GPU fix**: removed the conflicting plain `onnxruntime` package that
  shadowed `onnxruntime-directml`, restoring `DmlExecutionProvider` on the
  RTX A4000 — YOLO+pose now run on DirectML, cutting single-frame `_detect` to
  ~27 ms @720p. Experimental; defaults unchanged.
- **Long-range (3–5 m) engine A/B with ground truth**
  (`raw_capture/20260722_145659`, 2000 frames, 18 truth events): MediaPipe
  detects hands in 42.7% of frames vs HaGRID YOLO 92.8%, but YOLO costs ~2.5x
  latency (P95 81 ms vs 35 ms), ~4x jitter and a 58.1% multi-hand rate at
  conf 0.25. Conclusion: keep `mediapipe` as the default engine; pursue
  automatic switching instead of a global default.
- **Head-to-head: crop-zoom/SR vs YOLO at 3–5 m** (`benchmark_ab.py --set
  long_range_enabled=true`): with the long-range chain engaged, MediaPipe
  reaches 88.1% detection with only 1.4% multi-hand, half the jitter and
  ~10 ms lower P95 latency than YOLO — the better *tracker*. But zoom can
  only engage after a hand is already detected, so it structurally cannot
  solve initial acquisition at distance. YOLO is the better *catcher*; the
  two are complementary, not alternatives.
- **Engine auto-switch warm handoff** (`engine_auto_switch`, default off): a
  background HaGRID YOLO + Landmarker instance is prepared while MediaPipe is
  active and reused for CAPTURE, removing the observed 2.6–3.8 s cold start.
  After 60 no-hand frames it moves to CAPTURE; 30 stable single-hand frames
  hand off to FAR_TRACK. YOLO capture now keeps only the highest-confidence
  hand (`yolo_max_hands=1`) and uses corrected centre-xywh NMS coordinates.
  Multi-hand frames do not count toward handoff. The loop runs only when the
  configured engine is `mediapipe`; explicitly selecting `hagrid_yolo`
  remains authoritative.
- **Ground-truth A/B verdict for the three gesture features**:
  near-range B-group (`raw_capture/20260722_164055`, 1846 frames, 99% hand
  frames, 14 labeled click/drag groups) — the new EXIT-only pinch hysteresis
  preserves recall (64.3%) and onset P95 (1402 ms) while reducing false alarms
  9 → 4 and flips 35 → 25, so `pinch_exit_hysteresis_enabled` is now **on by
  default**. The older dual ENTER/EXIT switch remains off: it reduced false
  alarms further but recall fell to 57.1% and onset P95 rose to 4011 ms.
  `thumb_perp` still lacks labeled poses and remains off.
- **Fix: "app crashed during recording" was Space clicking the focused
  button** (`e5bb295`). Qt activates a focused QPushButton on Space release,
  and the floating window's close button had keyboard focus while the user
  tapped Space as the truth marker — an orderly `closeEvent` that looked like
  a crash. All floating-window buttons and every draw-toolbar button/slider
  are now `NoFocus`, with regression tests.
- **Ground-truth recording framework**: `TruthEventLogger`
  writes `truth_events.jsonl` from a configurable marker key (space for near
  range; wireless-mouse middle/side buttons or presenter PageUp/PageDown for
  3–5 m; comma-separated multi-key supported);
  `benchmark_gesture_ab.py` reports recall / miss rate / false alarms /
  onset-offset latency against truth with conservative default on/off
  criteria. F8 is a global record hotkey and a REC dot is painted on the
  frameless floating window while recording.
- **HaGRID YOLO engine finishing** (`ceed13b`): 12 MB
  `hand_yolov8n.onnx` stays in the repo for development/CI but is no longer
  packaged into the release bundle. Near-range A/B showed parity detection
  at 2.3–2.7x latency, so `mediapipe` stays default.
- `benchmark_ab.py` gained `--engines`, `--set key=value` (config overrides
  for offline experiments) and `--out`; new metrics include `zoom_on_frames`.
- **Freeze-on-pinch cursor stabilization**: `pinch_freeze_enabled` can lock the
  cursor for `pinch_freeze_grace_sec` (default 0.3s) after pinch begins. The
  corrected replay metric excludes the release frame and follows MouseMode's
  blended pointer. Recording `20260705_095523` had no evaluable continuing
  pinch event; `20260705_174137` had 6 evaluable events with event-maximum
  source-video drift mean 43.3px and P95 105.4px. These are observational
  values without click/drag ground truth, so the feature remains disabled.
- **Pinch dual-threshold hysteresis**: optional ENTER=0.30 / EXIT=0.40 ratios
  reduce boundary flicker. Corrected replay changed 1 frame in the first
  recording and 9 frames in the second; the pinch-heavy recording showed
  flips 22→18 (-18.2%). Fewer flips do not prove higher click accuracy, so
  `pinch_hysteresis_enabled` also remains disabled by default.
- **Rotation-invariant `thumb_extended` telemetry**: both the legacy feature
  and the perpendicular-distance ratio are emitted for calibration. The new
  path remains behind `thumb_perp_ratio_enabled=false`; the recordings lack
  labeled tucked/extended poses and therefore cannot select a safe threshold.
- Added stable / balanced / long-range experience profiles. The stable default
  keeps long-range prediction, adaptive skipping, geometric constraints, edge
  acceleration, and temporal voting conservative, while the long-range profile
  retains crop zoom, face guidance, and prediction for distant presentation use.
- Restored predictable board-writing and pointer defaults: single-finger pen
  lift, vote ratio 0.60, VOTE_MIN 3, and edge acceleration disabled at strength
  35 unless a user selects a profile that enables it.
- All stability profiles, schema defaults, the shipped config, and runtime
  configuration updates agree on these conservative defaults. Changing the
  hysteresis or perpendicular-ratio switch now takes effect without restart.
- Restored strict five-finger `is_open_palm` recognition so three-finger poses
  cannot accidentally enter destructive board clear behavior.
- Fixed the Windows PyInstaller version resource pipeline. The packaged EXE
  now carries FileVersion/ProductVersion 1.4.0 from `app/version.py`.
- Removed collection-time test module pollution and inference-worker leaks;
  the full suite can run repeatedly in one process without starting real
  camera or MediaPipe services from mocked orchestrator tests.
- Added focused tests for all three gesture phases, benchmark release-frame
  exclusion, build version resources, and runtime switch refresh.

## v1.3.5 - 2026-07-06

- Reverted to v1.3.0's handedness-keyed smoother architecture, eliminating
  the two-hand cursor flicker ("拉风箱") introduced during Phase 2.5-2.9.
  A/B on the same recording (1583 frames, 734 multi-hand): primary_switches
  11 → 0, switch_jerk 728.6px → 0, wrist_jerk_mean 19.9 → 14.6px,
  wrist_jerk_p95 66.6 → 50.5px.
- Hardened drawing stability: open-palm pen lift now requires N consecutive
  frames (draw_open_palm_lift_frames, default 3) instead of single-frame
  immediate lift; gesture thresholds use a slow EMA on hand_width to ignore
  palm-width jitter (log-observed 58↔208 px swings).
- Capped 1080p inference latency at ~15ms by downscaling frames to
  inference_max_width=720 before feeding MediaPipe (normalized output makes
  coordinate compensation unnecessary).
- Worked around Windows DSHOW/MSMF resetting FOURCC to YUY2 on resolution
  change: MJPEG is re-applied after every resolution switch, with a 0.5s
  device-settle delay on camera switch to avoid async-release crashes.
- Added 5 reversible config switches for speculative enhancement layers
  (adaptive_skip / long_range / geometric_constraint / hand_prediction /
  temporal_voter); rolled back 12 Phase 2.5-2.10 "improvements" that
  regressed stability (mouse-mode EMA overlay, local pinch recompute,
  3-finger open palm, ratio-only finger-close thresholds, wrist-velocity
  motion prior, oversized hint_label, etc.).
- Restored v1.3.0-baseline defaults: edge_acceleration on (strength 100),
  hand_prediction on, temporal_voter off, 5-finger open palm, 60px floor
  on finger-close/scroll thresholds, full _priority_score formula.
- Added meta.jsonl recording alongside raw video so replays reconstruct the
  original-runtime recognition point trajectory instead of re-detecting with
  current code; added mp4v/ffv1 codec selection (default mp4v, ~5-10x
  smaller files).
- Added analyze_primary_stability.py with --from-meta and --render-overlay
  modes for quantitative A/B diagnostics.
- Extracted base_hand_tracker into 6 focused service modules (smoothers,
  sr_engine, geometric_classifier, temporal_voter, face_guide, renderer)
  and expanded the test suite from 20 to 38 files (321 passed + 175
  subtests).

## v1.3.0 - 2026-06-15

- Improved near-range hand and fingertip stability with sub-pixel landmarks,
  corrected One Euro filtering, and weighted pointer landmarks.
- Reduced input latency with latest-frame capture, UI backpressure, and direct
  BGR-to-Qt frame conversion.
- Improved crop-zoom reacquisition and stabilized super-resolution switching.
- Hardened drawing state transitions and global mode-switch gesture handling.
- Added atomic configuration saves, runtime resource paths, safer tracker
  reloads, deterministic shutdown, and raw-frame diagnostics.
- Reworked Windows packaging, added a hardware-free package self-test, CI,
  linting, and broader regression tests.
