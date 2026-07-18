# Changelog

## v1.4.0 - 2026-07-18

AirControl 1.4 consolidates the locally iterated pinch-stability work into a
single public minor release over v1.3.5. The three new gesture layers are
available for controlled trials, but remain **off by default** until labeled
click/drag recordings demonstrate both accuracy and acceptable latency.

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
