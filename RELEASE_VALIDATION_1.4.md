# AirControl 1.4 release validation

Validation date: 2026-07-18

## Automated gates

| Gate | Result |
| --- | --- |
| Python 3.10.7 | compile + Ruff + 353 tests passed; app coverage 40% |
| Python 3.12.1 | compile + Ruff + 353 tests passed; app coverage 40% |
| Dependency integrity | `pip check` passed on Python 3.10 and 3.12 |
| Windows package | PyInstaller 6.21.0 build passed on Python 3.12 |
| Package metadata | FileVersion and ProductVersion are both `1.4.0` |
| Package self-test | hand tracker and offline voice models initialized; exit code 0 |
| GUI startup smoke | remained alive for 10 seconds; camera 0 opened through DSHOW/MJPG at 1920x1080 and 24 fps; no crash marker |
| v1.3.5 config compatibility | all 64 shipped legacy keys merged in memory with zero schema normalization warnings |

Local candidate EXE: 9,623,573 bytes, SHA-256
`6CE90D23D9A45E10B94F78099285018FE83AF237018A9C6731887B940B01C559`.
CI now repeats the package build, version-resource check, and packaged self-test
on the Python 3.12 matrix job.

## Corrected replay evidence

The replay benchmark now excludes the pinch-release frame from freeze drift,
uses MouseMode's blended pointer, distinguishes hysteresis-band candidates from
frames whose state actually changed, and does not recommend defaults without
click/drag ground truth.

| Recording | Frames / with hand | Hysteresis flips | Changed frames | Freeze evaluable events | Event-max drift mean / P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `20260705_095523` | 1079 / 1057 | 4 → 2 | 1 | 0 | not evaluable |
| `20260705_174137` | 1583 / 1455 | 22 → 18 | 9 | 6 | 43.3px / 105.4px |

The drift values are source-video pixels, not screen cursor error. Until click
and intentional-drag intervals are labeled, `pinch_freeze_enabled`,
`pinch_hysteresis_enabled`, and `thumb_perp_ratio_enabled` remain false by
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

The remote currently has release tags only through v1.3.5. After all automated
and manual gates pass, create one new `v1.4.0` tag on the final release commit
and push that tag explicitly; do not use `git push --tags`.

## Manual gates still required before publishing

- Confirm packaged EXE fallback when no camera is available.
- Exercise mouse move, short click, long hold, and intentional drag.
- Exercise board writing, pen lift, and deliberate five-finger clear.
