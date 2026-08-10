# Model provenance and release policy

The repository code is licensed under Apache-2.0. That statement does **not**
automatically cover model weights or other third-party assets.

## `models/hand_yolov8n.onnx` — NOT bundled in releases

| Field | Current evidence |
| --- | --- |
| SHA-256 | `c0e22242413252175bdd4eab1f7f85624b1c5c388c8120fe08090e3f14302049` |
| Size | 12,265,534 bytes |
| ONNX metadata | Ultralytics YOLOv8n, detect task, 640×640 input, output `[1, 5, 8400]` |
| Embedded license string | `AGPL-3.0 License (https://ultralytics.com/license)` |
| Official licence guidance | <https://www.ultralytics.com/license> |
| Source URL | <https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt> |
| Bundled in release | **No** — excluded from `AirControl.spec` |
| Redistribution approval | Not required (model is not distributed) |

The model's AGPL-3.0 metadata conflicts with the repository's Apache-2.0
licence. Starting with v1.4.0 the model is **intentionally excluded** from
PyInstaller packaging. `build.py` verifies that `AirControl.spec` does not
contain an `add_data_if_present` call for this file; if someone re-adds it,
the build fails with an AGPL licence warning.

### How users obtain the model (optional)

The model is only needed for the `hagrid_yolo` detection engine and the
`engine_auto_switch` far-range auto-switching feature. The default `mediapipe`
engine works without it.

1. Download the YOLOv8n weights from
   [Ultralytics releases](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt).
2. Export to ONNX (requires `pip install ultralytics`):
   ```
   yolo export model=yolov8n.pt format=onnx opset=13 simplify imgsz=640
   ```
3. Rename the exported file to `hand_yolov8n.onnx` and place it in the
   `models/` directory next to the application.

Alternatively, download a HaGRID-trained hand detector:
<https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid_v2/models/YOLOv10n_hands.pt>
and export it the same way. The tracker's `_resolve_yolo_model` searches
multiple candidate filenames.

Local-only development, testing, and use are not blocked by this policy.

## `models/yolov8n-pose.onnx` (local experiment only, NOT in git)

Used by the experimental `person_pose_hand` engine
(`app/services/person_pose_hand_tracker.py`) for body-pose wrist anchoring.

| Field | Current evidence |
| --- | --- |
| SHA-256 | `09f7a631a4c0e1daa10d2c33a2d78346bbb1db016fbe899cf7eb55dcc4044611` |
| Size | 13,514,381 bytes |
| ONNX metadata | Ultralytics YOLOv8n, pose task, 640×640 input, output `[1, 56, 8400]` |
| Embedded license string | `AGPL-3.0 License (https://ultralytics.com/license)` |
| Source URL | <https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n-pose.pt> |
| Export | `yolo export model=yolov8n-pose.pt format=onnx opset=13 simplify imgsz=640` (ultralytics 8.4.92) |
| Bundled in release | **No** — excluded from version control and packaging |

Same Ultralytics AGPL situation as the hand detector. This file is
intentionally excluded from version control (`.gitignore`) and is **not**
packaged into any build. It is downloaded/exported locally on demand for
offline A/B experiments.
