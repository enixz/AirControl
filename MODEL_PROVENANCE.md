# Model provenance and release gate

The repository code is licensed under Apache-2.0. That statement does **not**
automatically cover model weights or other third-party assets bundled with the
application.

## `models/hand_yolov8n.onnx`

| Field | Current evidence |
| --- | --- |
| SHA-256 | `c0e22242413252175bdd4eab1f7f85624b1c5c388c8120fe08090e3f14302049` |
| Size | 12,265,534 bytes |
| ONNX metadata | Ultralytics YOLOv8n, detect task, 640×640 input, output `[1, 5, 8400]` |
| Embedded license string | `AGPL-3.0 License (https://ultralytics.com/license)` |
| Official licence guidance | <https://www.ultralytics.com/license> |
| Source URL / source revision | not recorded |
| Redistribution approval | not recorded |

This metadata conflicts with treating the bundled model as covered by the
repository's Apache-2.0 notice. It is not legal advice and does not establish
the model's provenance; it is a release blocker until the maintainer records
the original download/training source, the applicable model/license terms, and
an approval for the intended distribution.

Local-only development, testing, and use are not blocked by this project gate.
The warning applies only if a future build is intended for redistribution or
public release.

`models/model_manifest.json` is the machine-readable record. `python build.py`
and `python build.py --release` verify its checksum and refuse to package while
the model is not marked `redistribution_approved: true`. A local-only developer
bundle requires the explicit `python build.py --development` override, emits a
prominent warning, and must not be published.

Before changing the approval flag, retain a review reference that identifies
the model source, its licence, and the release channel/terms that were approved.
The gate currently covers this known YOLO asset only; it is not a substitute for
an audit of every third-party dependency and model in a release.
