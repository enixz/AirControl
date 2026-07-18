# Changelog

## v1.7.0 - 2026-07-18

- **Enable Phase 3.1 + 3.2 by default** after A/B validation on local
  recording `raw_capture/20260705_174137` (1583 frames, 1455 with hands):
  - `pinch_freeze_enabled`: now `true` by default. A/B showed 10 pinch
    rising-edges with grace-period fingertip drift mean 63.9px (P95 283px)
    — freeze eliminates this click-point drift.
  - `pinch_hysteresis_enabled`: now `true` by default. A/B showed pinch
    flips 20→18 (-10%), 16 frames in the 0.30-0.40 hysteresis band
    stabilized (32% of pinch frames).
- **Consistency fix**: both switches added to all three
  `stability_profile` presets (stable/balanced/long_range) as `true`, so
  the profile system remains the single source of truth for stability
  switches (previously they were unmanaged by profiles, causing an
  incomplete stability picture on profile switch). Schema defaults and
  `default_config` updated to match.
- `thumb_perp_ratio_enabled` remains `false` — A/B showed the 0.50
  threshold is far below the measured perp_ratio mean (1.106); needs
  thumb-tucked calibration recording before enabling.
- No code logic changes (Phase 3.1-3.3 implementations unchanged); this
  release only flips validated defaults and aligns the profile system.

## v1.6.0 - 2026-07-18

- **Phase 3.3 (实施方案)**: Rotation-invariant `thumb_extended` via
  perpendicular-distance ratio. Adds `_thumb_perp_ratio()` computing the
  thumb tip's perpendicular distance to the palm axis (wrist→middle MCP),
  normalized by palm width. This is rotation-invariant: the ratio stays
  stable when the hand rotates, unlike the old `thumb_tip_to_index_mcp`
  distance which varies with hand orientation.
- **Parallel coexistence**: both old (`thumb_tip_to_index_mcp > 0.9×hw`)
  and new (`thumb_perp_ratio > 0.5`) features are always computed and
  output to the features dict (`thumb_extended`, `thumb_extended_new`,
  `thumb_perp_ratio`). Telemetry logs all three for A/B calibration.
- New config switch `thumb_perp_ratio_enabled` (default off). When enabled,
  `thumb_extended` follows the new rotation-invariant logic; when disabled,
  the old distance-based logic is used (fully reversible).
- Threshold `THUMB_PERP_RATIO_THRESHOLD=0.5` is an initial value requiring
  real-world calibration (anchored: extended ≈ 0.5+×palm_width, tucked ≈
  0.2×palm_width). Rotation-invariance verified by unit test: ratio stays
  within 20% across 0°/45°/90°/135° rotations.
- 10 new unit tests: perp_ratio computation, rotation invariance, feature
  output, config switch behavior.

## v1.5.0 - 2026-07-18

- **Phase 3.2 (实施方案)**: Pinch dual-threshold hysteresis. Adds
  `PINCH_ENTER_RATIO=0.30` (stricter) and `PINCH_EXIT_RATIO=0.40` (lenient)
  to eliminate pinch-state flicker at the threshold boundary. When already
  pinching, the EXIT threshold keeps the state stable until the distance
  exceeds 0.40×hand_width; when not pinching, the ENTER threshold requires
  the distance to drop below 0.30×hand_width.
- Calibration anchors documented in code: real pinch ≈ 0.15–0.25×hand_width,
  fist ≈ 0.50+×hand_width, so EXIT=0.40 stays well below fist (fist never
  misfires as pinch-release). This matches Air-Cursor's lesson that
  finger-extension guards are unreliable and distance thresholds alone
  provide clean separation.
- New config switch `pinch_hysteresis_enabled` (default off for
  reversibility). Telemetry extended to log `idx_ratio`/`mid_ratio` for
  real-world calibration.
- 8 new unit tests: ENTER/EXIT boundaries, full hysteresis cycle, state reset.

## v1.4.0 - 2026-07-18

- **Phase 3.1 (实施方案)**: Freeze-on-pinch cursor stabilization. When the
  user pinches (thumb-index), the cursor locks at the aimed position for a
  configurable grace period (`pinch_freeze_grace_sec`, default 0.3s), then
  releases for normal drag. Eliminates click-point drift caused by wrist
  micro-motion and landmark jitter during the pinch gesture.
- Borrowed from Air-Cursor's `freeze-on-fist` design, localized to pinch
  (AC-trae's click gesture is thumb-index pinch, not fist). Follows the
  existing freeze precedent in `draw_mode.py` (freeze active-region mapping
  during writing).
- New config switches (default off for reversibility): `pinch_freeze_enabled`
  (bool) and `pinch_freeze_grace_sec` (0.0–2.0s). Set `pinch_freeze_enabled`
  to `true` in `config.json` to enable.
- 8 new unit tests covering the state machine: rising-edge record, grace-period
  lock, post-grace release, pinch-release clear, hand-lost clear, on_exit clear.

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
