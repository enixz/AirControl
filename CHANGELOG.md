# Changelog

## v1.3.6 - 2026-07-08

- Absorbed v1.3's conservative interaction defaults into the v1.3.5 tracker
  base: board writing now defaults to single-finger write stability
  (draw_thumb_lift=false), VOTE_MIN is back to 3, and the draw vote ratio is
  consistently 0.60.
- Reduced mouse-mode over-amplification: active-region mapping remains, but
  edge acceleration is only applied when the user enables it; the stable
  default is edge_acceleration_enabled=false with strength 35.
- Added stability_profile presets (stable / balanced / long_range) and exposed
  them in settings so far-distance enhancements remain available without
  affecting the default classroom/whiteboard feel.
- Stable profile now keeps speculative layers off by default, including
  long_range_enabled=false, while preserving the v1.3.5 handedness-keyed
  smoother, recording diagnostics, camera hardening, and expanded tests.

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
