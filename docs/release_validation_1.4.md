# AirControl 1.4 release validation

Validation date: 2026-08-10

## Automated gates

| Gate | Result |
| --- | --- |
| Python 3.10 | tests passed; `pip check` passed |
| Python 3.12.1 | compile + Ruff + tests passed; app coverage reported |
| Dependency integrity | `pip check` passed on Python 3.10 and 3.12 |
| Windows package | PyInstaller build passed on Python 3.12 |
| Package metadata | FileVersion and ProductVersion are both `1.4.0` |
| Package self-test | exit code 0; core models present |
| YOLO model exclusion | `build.py` verifies `AirControl.spec` does not package `hand_yolov8n.onnx` |

## YOLO model policy

The bundled `models/hand_yolov8n.onnx` reports an AGPL-3.0 licence string in
its embedded metadata, which conflicts with the repository's Apache-2.0
licence. Starting with v1.4.0 the model is **intentionally excluded** from
PyInstaller packaging. `build.py` reads `AirControl.spec` and fails the build
if an `add_data_if_present` call for the model reappears.

Users who need the `hagrid_yolo` engine or `engine_auto_switch` far-range
auto-switching feature download the model separately (see README and
`MODEL_PROVENANCE.md`). The default `mediapipe` engine is unaffected.

## Corrected replay evidence

The replay benchmark excludes the pinch-release frame from freeze drift,
uses MouseMode's blended pointer, distinguishes hysteresis-band candidates
from frames whose state actually changed, and does not recommend defaults
without click/drag ground truth.

| Recording | Frames / with hand | Hysteresis flips | Changed frames | Freeze evaluable events | Event-max drift mean / P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `20260705_095523` | 1079 / 1057 | 4 → 2 | 1 | 0 | not evaluable |
| `20260705_174137` | 1583 / 1455 | 22 → 18 | 9 | 6 | 43.3px / 105.4px |

A 14-event ground-truth replay (`20260722_164055`) showed that
`pinch_exit_hysteresis_enabled` preserves recall (64.3%) and onset P95
(1402 ms) while reducing false alarms 9→4, so it is enabled by default.
`pinch_hysteresis_enabled` (the older dual ENTER/EXIT mode),
`pinch_freeze_enabled`, and `thumb_perp_ratio_enabled` remain false by
default.

## Local experimental tag record

These local tags predated the public 1.4 candidate and must not be pushed as
release tags:

| Old local tag | Commit | Meaning |
| --- | --- | --- |
| `v1.3.6` | `5105f1234dae80807e5fdb22543098a62bf22343` | Unpublished stability-profile staging point |
| `v1.4.0` | `9f54a6d713a13c9635962bbf92de9d9b648cd0d8` | Phase 3.1 only |
| `v1.5.0` | `22002c0ab56631ebc9c519f21cba9a451fd8e5b6` | Phase 3.2 iteration |
| `v1.6.0` | `996b2dc211dd3946a4be39f24ef3d50bfe247382` | Phase 3.3 iteration |
| `v1.7.0` | `431f4a36880d38032f7656cfeca3da2d3c59d135` | Experimental defaults enabled |

The remote currently has release tags only through v1.3.5. After all
automated and manual gates pass, create one new `v1.4.0` tag on the final
release commit and push that tag explicitly; do not use `git push --tags`.

## Manual gates still required before publishing

- Confirm packaged EXE fallback when no camera is available.
- Exercise mouse move, short click, long hold, and intentional drag.
- Exercise board writing, pen lift, and deliberate five-finger clear.
- Confirm `--self-test` exits 0 without `hand_yolov8n.onnx` present.
