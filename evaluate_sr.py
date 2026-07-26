import argparse
import hashlib
import os
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
import onnxruntime as ort
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Configure paths
SCRATCH_DIR = os.path.dirname(os.path.abspath(__file__))
HAND_IMG_PATH = os.path.join(SCRATCH_DIR, "hand_test.jpg")
FSRCNN_PATH = os.path.join(SCRATCH_DIR, "FSRCNN_x2.pb")
ESPCN_PATH = os.path.join(SCRATCH_DIR, "ESPCN_x2.pb")
REALESRGAN_PATH = os.path.join(SCRATCH_DIR, "Real-ESRGAN_x2plus.onnx")
OUTPUT_PATH = os.path.join(SCRATCH_DIR, "comparison_result.png")


# Pin every remote input to an immutable revision and verify both size and
# digest. Branch-based URLs silently changed benchmark inputs between runs and
# an interrupted download used to leave a corrupt file that was trusted later.
ASSETS = (
    (
        "hand image",
        "https://raw.githubusercontent.com/spmallick/learnopencv/"
        "500caba0a32443ede765ea96a2cf1851d4aa34fe/HandPose/hand.jpg",
        HAND_IMG_PATH,
        47_432,
        "87c8b7abbeee8b97a1d315b7c53bcf5bbcdb0d4dda22dc812080dfcc693be16a",
    ),
    (
        "FSRCNN x2",
        "https://raw.githubusercontent.com/Saafke/FSRCNN_Tensorflow/"
        "6a4812c4ef1c4f5947d79beafa32a05a6eb4a94d/models/FSRCNN_x2.pb",
        FSRCNN_PATH,
        38_973,
        "366b33f0084c7b3f2bf6724f0a2c77bca94fcec9d7b6d72389d330073b380d5c",
    ),
    (
        "ESPCN x2",
        "https://raw.githubusercontent.com/fannymonori/TF-ESPCN/"
        "5c628eca82028161a53e1265cc3a5b571ab8625f/export/ESPCN_x2.pb",
        ESPCN_PATH,
        86_446,
        "59f77351e1d7c0057bf6fe088b4a8a07e42c468c8c8aebb674a6b4ea1823221d",
    ),
    (
        "Real-ESRGAN x2plus",
        "https://huggingface.co/tidus2102/Real-ESRGAN/resolve/"
        "610d2a0128178c67407fbad3b9a90b8849ddd6c2/Real-ESRGAN_x2plus.onnx",
        REALESRGAN_PATH,
        67_075_476,
        "735f42fd172779c9776606298b9f744d9d488f8aa1fe90fe6c6470186020a754",
    ),
)


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _asset_is_valid(path, expected_size, expected_sha256):
    return (
        os.path.isfile(path)
        and os.path.getsize(path) == expected_size
        and _file_sha256(path) == expected_sha256
    )


def download_file(
    url,
    path,
    expected_size,
    expected_sha256,
    timeout_sec=30.0,
    offline=False,
):
    if os.path.exists(path):
        if _asset_is_valid(path, expected_size, expected_sha256):
            print(f"[Info] Verified cached file: {os.path.basename(path)}")
            return True
        print(f"[Warning] Cached file failed verification: {os.path.basename(path)}")
    if offline:
        print(f"[Error] Offline mode requires a valid cache: {os.path.basename(path)}")
        return False
    print(f"[Download] Fetching {url} ...")
    temp_path = path + ".part"
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "AirControl-SR-Evaluator/1.0"},
        )
        digest = hashlib.sha256()
        written = 0
        with (
            urllib.request.urlopen(req, timeout=max(0.1, float(timeout_sec))) as response,
            open(temp_path, "wb") as out_file,
        ):
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > expected_size:
                    raise ValueError(
                        f"download exceeds expected size {expected_size}"
                    )
                digest.update(chunk)
                out_file.write(chunk)
            out_file.flush()
            os.fsync(out_file.fileno())
        if written != expected_size or digest.hexdigest() != expected_sha256:
            raise ValueError(
                f"verification failed: size={written}, sha256={digest.hexdigest()}"
            )
        os.replace(temp_path, path)
        print(f"[Download] Success -> {os.path.basename(path)}")
        return True
    except Exception as e:
        print(f"[Download] Error downloading {url}: {e}")
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except OSError:
            pass
        return False


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate hand detection after several 2x super-resolution methods."
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="do not download; require verified assets in the project directory",
    )
    parser.add_argument(
        "--download-timeout",
        type=float,
        default=30.0,
        help="socket timeout in seconds for each asset download (default: 30)",
    )
    parser.add_argument(
        "--output",
        default=OUTPUT_PATH,
        help="comparison image path",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    print("=== Super-Resolution Evaluation Tool v2 (Real-ESRGAN Included) ===")

    # 1. Download assets
    for name, url, path, expected_size, expected_sha256 in ASSETS:
        if not download_file(
            url,
            path,
            expected_size,
            expected_sha256,
            timeout_sec=args.download_timeout,
            offline=args.offline,
        ):
            print(f"[Error] Required asset unavailable: {name}")
            return 2

    # 2. Initialize MediaPipe Task API
    print("[Init] Initializing MediaPipe HandLandmarker Task...")
    model_path = os.path.join(SCRATCH_DIR, "models", "hand_landmarker.task")
    if not os.path.exists(model_path):
        model_path = os.path.join(SCRATCH_DIR, "hand_landmarker.task")
    if not os.path.isfile(model_path):
        print(f"[Error] MediaPipe model not found: {model_path}")
        return 2

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.3
    )
    detector = vision.HandLandmarker.create_from_options(options)

    # 3. Initialize ONNX Runtime for Real-ESRGAN
    print("[Init] Loading Real-ESRGAN x2plus ONNX model on GPU/CPU...")
    available_providers = set(ort.get_available_providers())
    requested_providers = [
        provider for provider in ("DmlExecutionProvider", "CPUExecutionProvider")
        if provider in available_providers
    ]
    ort_session = ort.InferenceSession(
        REALESRGAN_PATH,
        providers=requested_providers or None,
    )
    active_provider = ort_session.get_providers()[0]
    realesrgan_input_name = ort_session.get_inputs()[0].name

    # 4. Read and crop hand from original image
    print("[Process] Reading test image and locating hand...")
    img = cv2.imread(HAND_IMG_PATH)
    if img is None:
        print("[Error] Could not read hand_test.jpg")
        return

    h, w, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    results = detector.detect(mp_image)

    if results and len(results.hand_landmarks) > 0:
        print("[Process] Hand located by MediaPipe in original image.")
        landmarks = results.hand_landmarks[0]
        xs = [lm.x * w for lm in landmarks]
        ys = [lm.y * h for lm in landmarks]
        xmin, xmax = int(min(xs)), int(max(xs))
        ymin, ymax = int(min(ys)), int(max(ys))

        box_w = xmax - xmin
        box_h = ymax - ymin
        size = int(max(box_w, box_h) * 1.4)
        cx, cy = (xmin + xmax) // 2, (ymin + ymax) // 2

        x0 = max(0, cx - size // 2)
        y0 = max(0, cy - size // 2)
        x1 = min(w, x0 + size)
        y1 = min(h, y0 + size)
        crop = img[y0:y1, x0:x1]
    else:
        print("[Warning] MediaPipe failed to detect hand in full image. Using default crop.")
        sz = min(w, h) // 2
        crop = img[h//2 - sz//2 : h//2 + sz//2, w//2 - sz//2 : w//2 + sz//2]

    # Ensure crop is a clean 160x160 square for standardizing
    crop = cv2.resize(crop, (160, 160), interpolation=cv2.INTER_AREA)

    # 5. Generate low-resolution image (40x40) - simulates hand at 3+ meters
    print("[Process] Downsampling hand to 40x40 (simulating distance)...")
    lr = cv2.resize(crop, (40, 40), interpolation=cv2.INTER_AREA)

    # 6. Load Super-Resolution Models
    print("[Init] Loading FSRCNN and ESPCN models in OpenCV DNN...")
    sr_fsrcnn = cv2.dnn_superres.DnnSuperResImpl_create()
    sr_fsrcnn.readModel(FSRCNN_PATH)
    sr_fsrcnn.setModel("fsrcnn", 2)  # upscale by 2x

    sr_espcn = cv2.dnn_superres.DnnSuperResImpl_create()
    sr_espcn.readModel(ESPCN_PATH)
    sr_espcn.setModel("espcn", 2)  # upscale by 2x

    # 7. Perform Upscaling (to 80x80)
    print("[Process] Upscaling 2x using various methods...")

    # Traditional
    up_bilinear = cv2.resize(lr, (80, 80), interpolation=cv2.INTER_LINEAR)
    up_bicubic = cv2.resize(lr, (80, 80), interpolation=cv2.INTER_CUBIC)

    # AI 2016
    up_fsrcnn = sr_fsrcnn.upsample(lr)
    up_espcn = sr_espcn.upsample(lr)

    # Real-ESRGAN requires 64x64 input and yields 128x128 output.
    print("[Process] Running Real-ESRGAN inference (including pre/post-resizing)...")
    lr_64 = cv2.resize(lr, (64, 64), interpolation=cv2.INTER_CUBIC)
    re_in = lr_64.astype(np.float32) / 255.0
    re_in = np.transpose(re_in, (2, 0, 1))
    re_in = np.expand_dims(re_in, axis=0)

    re_out = ort_session.run(None, {realesrgan_input_name: re_in})[0][0]
    re_out = np.clip(re_out * 255.0, 0, 255).astype(np.uint8)
    re_out = np.transpose(re_out, (1, 2, 0))
    up_realesrgan = cv2.resize(re_out, (80, 80), interpolation=cv2.INTER_AREA)

    # 8. Benchmarking Latency (GPU/CPU)
    print("\n--- Latency Benchmark (runs: 100 for light, 20 for Real-ESRGAN on GPU) ---")

    def test_bilinear():
        cv2.resize(lr, (80, 80), interpolation=cv2.INTER_LINEAR)
    def test_bicubic():
        cv2.resize(lr, (80, 80), interpolation=cv2.INTER_CUBIC)
    def test_fsrcnn():
        sr_fsrcnn.upsample(lr)
    def test_espcn():
        sr_espcn.upsample(lr)
    def test_realesrgan():
        lr_64 = cv2.resize(lr, (64, 64), interpolation=cv2.INTER_CUBIC)
        re_in = lr_64.astype(np.float32) / 255.0
        re_in = np.transpose(re_in, (2, 0, 1))
        re_in = np.expand_dims(re_in, axis=0)
        re_out = ort_session.run(None, {realesrgan_input_name: re_in})[0][0]
        re_out = np.clip(re_out * 255.0, 0, 255).astype(np.uint8)
        re_out = np.transpose(re_out, (1, 2, 0))
        cv2.resize(re_out, (80, 80), interpolation=cv2.INTER_AREA)

    def run_benchmark(name, fn, runs=100):
        # Warm up
        for _ in range(5):
            fn()
        start = time.perf_counter()
        for _ in range(runs):
            fn()
        end = time.perf_counter()
        avg_ms = (end - start) / runs * 1000
        print(f"| {name:<12} | Avg Time: {avg_ms:6.3f} ms | Frame Rate: {int(1000/avg_ms) if avg_ms > 0 else 0:4} FPS |")
        return avg_ms

    latencies = {}
    latencies["Bilinear"] = run_benchmark("Bilinear", test_bilinear, 100)
    latencies["Bicubic"] = run_benchmark("Bicubic", test_bicubic, 100)
    latencies["ESPCN (AI)"] = run_benchmark("ESPCN (AI)", test_espcn, 100)
    latencies["FSRCNN (AI)"] = run_benchmark("FSRCNN (AI)", test_fsrcnn, 100)
    latencies["Real-ESRGAN"] = run_benchmark("Real-ESRGAN", test_realesrgan, 20)  # Heavy model, 20 runs is enough

    # 9. MediaPipe Hand Landmark Detection Test on Upscaled Outputs
    print("\n--- MediaPipe Hand Detection Results ---")

    def test_detection(img_80x80, name):
        # MediaPipe requires standard sizing, let's upscale to 256x256 for fair detection
        test_img = cv2.resize(img_80x80, (256, 256), interpolation=cv2.INTER_LINEAR)
        test_rgb = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
        mp_test_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=test_rgb)

        det_results = detector.detect(mp_test_image)
        detected = det_results and len(det_results.hand_landmarks) > 0
        score = 0.0
        if detected:
            score = det_results.handedness[0][0].score
            print(f"| {name:<12} | SUCCESS | Confidence Score: {score:.4f} |")
        else:
            print(f"| {name:<12} | FAILED  | - |")
        return detected, score

    d_gt, s_gt = test_detection(cv2.resize(crop, (80, 80)), "Ground Truth")
    d_lr, s_lr = test_detection(cv2.resize(lr, (80, 80), interpolation=cv2.INTER_NEAREST), "Low-Res (40x40)")
    d_bilinear, s_bilinear = test_detection(up_bilinear, "Bilinear")
    d_bicubic, s_bicubic   = test_detection(up_bicubic, "Bicubic")
    d_fsrcnn, s_fsrcnn     = test_detection(up_fsrcnn, "FSRCNN (AI)")
    d_espcn, s_espcn       = test_detection(up_espcn, "ESPCN (AI)")
    d_re, s_re             = test_detection(up_realesrgan, "Real-ESRGAN")

    # 10. Generate Comparison Visual Grid (2x4 cells, each 320x320)
    print("\n[Output] Generating visual comparison grid...")

    def make_cell(img, label, detected, score):
        # Resize to 320x320 using Nearest Neighbor to show pixels clearly
        cell = cv2.resize(img, (320, 320), interpolation=cv2.INTER_NEAREST)

        # Draw transparent black overlay for text readability
        overlay = cell.copy()
        cv2.rectangle(overlay, (0, 0), (320, 45), (0, 0, 0), -1)
        cv2.rectangle(overlay, (0, 280), (320, 320), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, cell, 0.6, 0, cell)

        # Render text
        color = (0, 255, 0) if detected else (0, 0, 255)
        status_text = f"MediaPipe: DETECTED ({score:.2%})" if detected else "MediaPipe: FAILED"

        cv2.putText(cell, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(cell, status_text, (10, 305), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

        # Draw border
        cv2.rectangle(cell, (0, 0), (319, 319), (100, 100, 100), 2)
        return cell

    def make_summary_cell(lats):
        cell = np.zeros((320, 320, 3), dtype=np.uint8)
        cv2.rectangle(cell, (0, 0), (319, 319), (100, 100, 100), 2)

        cv2.putText(cell, f"8. Latency Summary ({active_provider})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        y = 80
        for name, lat in lats.items():
            text = f"{name}: {lat:.3f} ms"
            cv2.putText(cell, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
            y += 35

        fps_re = int(1000 / lats.get("Real-ESRGAN", 1.0))
        cv2.putText(cell, f"Real-ESRGAN: ~{fps_re} FPS", (15, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)
        return cell

    # Prep cells
    gt_cell = make_cell(cv2.resize(crop, (80, 80)), "1. Ground Truth (160x160 -> 80x80)", d_gt, s_gt)
    lr_cell = make_cell(cv2.resize(lr, (80, 80), interpolation=cv2.INTER_NEAREST), "2. Low-Res Sim (40x40)", d_lr, s_lr)
    bi_cell = make_cell(up_bilinear, "3. Bilinear Interpolation (CPU)", d_bilinear, s_bilinear)
    bc_cell = make_cell(up_bicubic, "4. Bicubic Interpolation (CPU)", d_bicubic, s_bicubic)
    fs_cell = make_cell(up_fsrcnn, "5. FSRCNN (AI, 2016)", d_fsrcnn, s_fsrcnn)
    es_cell = make_cell(up_espcn, "6. ESPCN (AI, 2016)", d_espcn, s_espcn)
    re_cell = make_cell(up_realesrgan, "7. Real-ESRGAN (AI, 2026)", d_re, s_re)
    sum_cell = make_summary_cell(latencies)

    # Stack into a 2x4 grid
    row1 = np.hstack([gt_cell, lr_cell, bi_cell, bc_cell])
    row2 = np.hstack([fs_cell, es_cell, re_cell, sum_cell])
    grid = np.vstack([row1, row2])

    output_path = os.path.abspath(args.output)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    if not cv2.imwrite(output_path, grid):
        print(f"[Error] Failed to write comparison image: {output_path}")
        return 3
    print(f"[Output] Success! Results comparison image saved to: {output_path}")
    print("========================================")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
