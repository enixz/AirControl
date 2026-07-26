import logging
import threading
import time
from collections import deque

import cv2
from PyQt6.QtCore import QThread, pyqtSignal

logger = logging.getLogger(__name__)


class InferenceWorker(QThread):
    """
    后台推理线程，负责摄像头读取和MediaPipe推理。
    通过信号将结果发送到主线程处理模式逻辑和UI更新。
    """

    # 信号：帧数据、手部关键点、手势识别结果、worker本身引用
    frame_ready = pyqtSignal(object, list, list, object)
    error_occurred = pyqtSignal(str)
    fps_updated = pyqtSignal(float)
    performance_updated = pyqtSignal(object)
    # worker, active tracker, caller context, migration/seed result
    tracker_swapped = pyqtSignal(object, object, object, object)

    def __init__(self, camera, tracker, max_fps=30, parent=None, debug_overlay=False,
                 frame_recorder=None):
        super().__init__(parent)
        self.camera = camera
        self.tracker = tracker
        self._pending_tracker = None
        self._pending_tracker_context = None
        self._pending_tracker_seed_crop_zoom = False
        self._pending_tracker_lock = threading.Lock()
        self._retired_pending_trackers = []
        self._accept_tracker_updates = True
        self.frame_recorder = frame_recorder  # 可选：原始帧无损录制（默认 None）
        self.running = False
        self.lock = threading.Lock()
        self._capture_stop = threading.Event()
        self._capture_condition = threading.Condition()
        self._capture_thread = None
        self._latest_frame = None
        self._latest_sequence = 0
        self._processed_sequence = 0
        self._result_lock = threading.Lock()
        self._result_pending = False
        self._recorder_lock = threading.Lock()
        self._frame_count = 0
        self._fps_start_time = time.time()
        self._current_fps = 0.0
        self.max_fps = max_fps
        self._frame_interval = 1.0 / max_fps  # 最小帧间隔
        self.debug_overlay = debug_overlay  # 调试覆盖层开关
        self._capture_ms = deque(maxlen=180)
        self._queue_ms = deque(maxlen=180)
        self._inference_ms = deque(maxlen=180)
        self._total_ms = deque(maxlen=180)
        self._last_performance_emit = time.monotonic()

    def run(self):
        """线程主循环"""
        self.running = True
        self._capture_stop.clear()
        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            name="CameraCaptureWorker",
            daemon=True,
        )
        self._capture_thread.start()
        logger.info("InferenceWorker 启动")

        try:
            while self.running:
                item = self._wait_for_latest_frame()
                if item is None:
                    continue
                try:
                    self._process_frame(*item)
                except Exception as e:
                    logger.error("InferenceWorker 错误: %s", e, exc_info=True)
                    self.error_occurred.emit(str(e))
                    time.sleep(0.1)
        finally:
            self._capture_stop.set()
            with self._capture_condition:
                self._capture_condition.notify_all()
            if self._capture_thread and self._capture_thread.is_alive():
                # VideoCapture.read() may still be inside a native backend call.
                # Returning from run() while that thread is alive lets the caller
                # release the same native handle concurrently, which can crash
                # OpenCV/COM.  The capture thread owns camera access, so wait for
                # it to leave read_frame() and observe _capture_stop before the
                # QThread is allowed to finish.
                logger.info("等待摄像头采集线程退出")
                self._capture_thread.join()
            logger.info("InferenceWorker 停止")

    def _capture_loop(self):
        """Continuously capture frames and retain only the newest one."""
        while not self._capture_stop.is_set():
            started = time.perf_counter()
            success, frame = self.camera.read_frame()
            capture_ms = (time.perf_counter() - started) * 1000.0
            if not success:
                # 处于冷却期或摄像头彻底未打开时，增加休眠避免 100Hz 的极速盲等死循环
                sleep_time = 0.01 if self.camera.cap is not None else 0.5
                self._capture_stop.wait(sleep_time)
                continue
            frame = cv2.flip(frame, 1)
            captured_at = time.perf_counter()
            with self._capture_condition:
                self._latest_sequence += 1
                self._latest_frame = (
                    self._latest_sequence,
                    frame,
                    capture_ms,
                    captured_at,
                )
                self._capture_condition.notify()

    def _wait_for_latest_frame(self):
        with self._capture_condition:
            self._capture_condition.wait_for(
                lambda: (
                    not self.running
                    or (
                        self._latest_frame is not None
                        and self._latest_frame[0] > self._processed_sequence
                    )
                ),
                timeout=0.25,
            )
            if not self.running or self._latest_frame is None:
                return None
            sequence, frame, capture_ms, captured_at = self._latest_frame
            if sequence <= self._processed_sequence:
                return None
            self._processed_sequence = sequence
            return frame, capture_ms, captured_at

    def _process_frame(self, frame, capture_ms, captured_at):
        """处理单帧（带节流）"""
        start_time = time.perf_counter()
        queue_ms = max(0.0, (start_time - captured_at) * 1000.0)

        # 原始帧录制：find_hands(draw=True) 会原地绘制叠层污染 frame，故先 copy 一份
        # 干净帧；待 find_hands 完成后再写入（带检测元数据），保证 meta 与 frame 对齐。
        with self._recorder_lock:
            recorder = self.frame_recorder
        record_frame = frame.copy() if recorder is not None else None

        inference_started = time.perf_counter()
        # Hold the lock through native inference so an old detector cannot be
        # closed while MediaPipe is still using it.
        with self._pending_tracker_lock:
            pending_tracker = self._pending_tracker
            pending_context = self._pending_tracker_context
            seed_crop_zoom = self._pending_tracker_seed_crop_zoom
            self._pending_tracker = None
            self._pending_tracker_context = None
            self._pending_tracker_seed_crop_zoom = False
            retired_pending = self._retired_pending_trackers
            self._retired_pending_trackers = []
        with self.lock:
            # 在执行推理前安全替换 _pending_tracker，消除主线程的 lock 阻塞
            for retired in retired_pending:
                self._close_tracker(retired, "关闭被替换的 pending tracker 失败")
            if pending_tracker is not None:
                old_tracker = self.tracker
                migrate = getattr(pending_tracker, "migrate_state_from", None)
                if callable(migrate) and old_tracker is not None:
                    try:
                        migrate(old_tracker)
                    except Exception:
                        logger.exception(
                            "tracker 状态迁移失败（非致命，继续使用新 tracker）"
                        )
                seeded = False
                if seed_crop_zoom:
                    seed = getattr(
                        pending_tracker,
                        "seed_crop_zoom_from_hint",
                        None,
                    )
                    try:
                        seeded = bool(seed()) if callable(seed) else False
                    except Exception:
                        logger.exception("tracker crop-zoom 种子播种失败")
                self.tracker = pending_tracker
                if old_tracker is not self.tracker:
                    self._close_tracker(old_tracker, "关闭旧 tracker 失败")
                # This signal is emitted by the same worker that emits frame_ready.
                # Therefore queued old-frame results are delivered before the swap
                # commit, and new-frame results after it.
                self.tracker_swapped.emit(
                    self,
                    pending_tracker,
                    pending_context,
                    {
                        "seed_requested": bool(seed_crop_zoom),
                        "seeded": seeded,
                    },
                )
                logger.info("InferenceWorker: tracker 已异步更新完成")

            frame, hands_landmarks, hands_gestures = self.tracker.find_hands(
                frame, draw=True
            )
        inference_ms = (time.perf_counter() - inference_started) * 1000.0

        # 原始帧 + 检测元数据一起写入（meta 让 replay/analyze 能重建原始运行时识别点轨迹）
        if recorder is not None and record_frame is not None:
            meta = self._collect_record_meta(hands_landmarks, hands_gestures)
            recorder.write(record_frame, meta=meta)

        # 计算FPS
        self._update_fps()

        # 调试覆盖层（F1 切换）
        if self.debug_overlay:
            self._draw_debug_overlay(frame, hands_landmarks, hands_gestures)

        # Qt 的 queued signal 没有天然背压。主线程忙时只保留一个待处理结果，
        # 避免旧帧持续堆积并放大端到端延迟。
        if self._claim_result_slot():
            self.frame_ready.emit(frame, hands_landmarks, hands_gestures, self)

        total_ms = (time.perf_counter() - start_time) * 1000.0
        self._record_performance(capture_ms, queue_ms, inference_ms, total_ms)

        # 节流：确保不超过最大帧率
        elapsed = time.perf_counter() - start_time
        sleep_time = self._frame_interval - elapsed
        if sleep_time > 0:
            self._capture_stop.wait(sleep_time)

    @staticmethod
    def _percentile(samples, fraction):
        values = sorted(samples)
        if not values:
            return 0.0
        index = min(len(values) - 1, int(round((len(values) - 1) * fraction)))
        return values[index]

    def _record_performance(self, capture_ms, queue_ms, inference_ms, total_ms):
        self._capture_ms.append(capture_ms)
        self._queue_ms.append(queue_ms)
        self._inference_ms.append(inference_ms)
        self._total_ms.append(total_ms)
        now = time.monotonic()
        if now - self._last_performance_emit < 5.0:
            return
        self._last_performance_emit = now
        metrics = {}
        for name, values in (
            ("capture", self._capture_ms),
            ("queue", self._queue_ms),
            ("inference", self._inference_ms),
            ("total", self._total_ms),
        ):
            metrics[f"{name}_p50_ms"] = self._percentile(values, 0.50)
            metrics[f"{name}_p95_ms"] = self._percentile(values, 0.95)
        self.performance_updated.emit(metrics)

    def _update_fps(self):
        """更新FPS计算"""
        self._frame_count += 1
        current_time = time.time()
        elapsed = current_time - self._fps_start_time

        if elapsed >= 1.0:  # 每秒更新一次FPS
            self._current_fps = self._frame_count / elapsed
            self.fps_updated.emit(self._current_fps)
            self._frame_count = 0
            self._fps_start_time = current_time

    def update_tracker(self, new_tracker, context=None, seed_crop_zoom=False):
        """Queue an atomic worker-owned migration and tracker swap."""
        close_immediately = None
        with self._pending_tracker_lock:
            if not self._accept_tracker_updates:
                close_immediately = new_tracker
            else:
                # A rapid A -> B -> A sequence must not leave A in the retired
                # list and then promote the already-closed instance.
                self._retired_pending_trackers = [
                    tracker for tracker in self._retired_pending_trackers
                    if tracker is not new_tracker
                ]
                if (
                    self._pending_tracker is not None
                    and self._pending_tracker is not new_tracker
                    and all(
                        tracker is not self._pending_tracker
                        for tracker in self._retired_pending_trackers
                    )
                ):
                    self._retired_pending_trackers.append(
                        self._pending_tracker
                    )
            if self._accept_tracker_updates:
                self._pending_tracker = new_tracker
                self._pending_tracker_context = context
                self._pending_tracker_seed_crop_zoom = bool(seed_crop_zoom)
        if close_immediately is not None:
            self._close_tracker(
                close_immediately,
                "worker 已停止，关闭新 tracker 失败",
            )
            return False
        logger.info("InferenceWorker: tracker 异步更新已挂起")
        return True

    @staticmethod
    def _close_tracker(tracker, error_message):
        close = getattr(tracker, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                logger.exception(error_message)

    def _claim_result_slot(self):
        with self._result_lock:
            if self._result_pending:
                return False
            self._result_pending = True
            return True

    def mark_result_consumed(self):
        """Allow the next inference result to be queued to the Qt main thread."""
        with self._result_lock:
            self._result_pending = False

    def set_debug_overlay(self, enabled):
        """切换调试覆盖层（F1 或 config 控制）"""
        self.debug_overlay = bool(enabled)
        logger.info("调试覆盖层: %s", "开" if self.debug_overlay else "关")

    def set_frame_recorder(self, recorder):
        """线程安全替换原始帧录制器（F5 热切换）。

        每帧开始时在 _recorder_lock 下取得一致快照；后续该帧始终写入同一实例。
        """
        with self._recorder_lock:
            self.frame_recorder = recorder

    def _collect_record_meta(self, hands_landmarks, hands_gestures):
        """收集用于回放分析的元数据：主手 wrist、手数、crop 视口等。

        让 analyze_primary_stability.py 能直接读 meta.jsonl 重建原始运行时识别点
        轨迹，而不是用当前代码重新跑 find_hands（那只能看当前代码的表现，看不到原始
        运行时的"拉扯"行为）。

        注意：hands_landmarks 是 find_hands 返回的 smoothed 结果（按 _priority_score
        降序排列，index 0 是分数最高的手），即用户实际看到的识别点位置。
        """
        tracker = self.tracker
        meta = {
            "hands": len(hands_landmarks),
            "zoom_on": bool(getattr(tracker, "_crop_zoom_mode", False)),
        }
        if hands_landmarks:
            # smoothed wrists（用户看到的识别点位置）；index 0 是分数最高的手
            meta["wrists"] = [
                [round(float(h[0][1]), 2), round(float(h[0][2]), 2)]
                for h in hands_landmarks
            ]
            # primary_wrist 单独冗余一份，方便分析脚本直接取
            pw = hands_landmarks[0][0]
            meta["primary_wrist"] = [round(float(pw[1]), 2), round(float(pw[2]), 2)]
        cc = getattr(tracker, "_current_crop_center", None)
        cs = getattr(tracker, "_current_crop_size", None)
        if cc is not None:
            meta["crop_center"] = [round(float(cc[0]), 2), round(float(cc[1]), 2)]
        if cs is not None:
            meta["crop_size"] = round(float(cs), 2)
        return meta

    def _draw_debug_overlay(self, frame, hands_landmarks, hands_gestures):
        """在画面上叠加 FPS、手数、handedness、predicted 标记等调试信息。

        左上角块：FPS + 帧数
        每只手旁边：handedness + 是否 predicted + 运动 EMA（如果有）
        """
        h, w = frame.shape[:2]
        # 半透明黑底，让文字始终可读
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (260, 70), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        cv2.putText(
            frame, f"FPS: {self._current_fps:.1f}",
            (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
        )
        cv2.putText(
            frame, f"Hands: {len(hands_landmarks)}",
            (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
        )

        # 每只手的元数据
        for i, (landmarks, gesture) in enumerate(zip(hands_landmarks, hands_gestures, strict=True)):
            if not landmarks:
                continue
            wrist = landmarks[0]
            x, y = int(wrist[1]), int(wrist[2])
            label = f"#{i} {gesture.get('handedness', '?')[:1]}"
            if gesture.get("predicted"):
                label += " [predict]"
                # 统一用紫色：预测补帧与真实检测视觉一致，避免黄紫交替闪烁
                color = (255, 0, 255) if i == 0 else (200, 200, 200)
            else:
                color = (255, 0, 255) if i == 0 else (200, 200, 200)
            cv2.putText(
                frame, label, (max(x + 10, 10), max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1,
            )

    def stop(self, timeout_ms=3000):
        """停止线程。

        若 update_tracker() 提交了 _pending_tracker 但 worker 在下次推理前就被
        停掉，该 pending tracker 永远不会被 swap-in、也不会被关闭，原生
        MediaPipe 句柄与 SR ONNX session 会泄漏。这里在停掉线程后 flush 它：
        关闭 pending tracker。active tracker 不在此关闭——它由 orchestrator
        持有并负责释放（camera 切换时还会复用同一只 tracker）。
        """
        self.running = False
        self._capture_stop.set()
        with self._pending_tracker_lock:
            self._accept_tracker_updates = False
        with self._capture_condition:
            self._capture_condition.notify_all()
        if self.isRunning():
            # A native camera backend can block forever in VideoCapture.read().
            # Keep shutdown bounded, but never let the caller release the camera
            # or active tracker unless this wait succeeds.
            timeout_ms = max(0, int(timeout_ms))
            if not self.wait(timeout_ms):
                logger.error(
                    "InferenceWorker 未能在 %.1f 秒内停止；"
                    "保留摄像头和 tracker 句柄，禁止并发释放",
                    timeout_ms / 1000.0,
                )
                return False
        # flush 未被消费的 pending tracker，避免句柄泄漏
        with self._pending_tracker_lock:
            pending = self._pending_tracker
            self._pending_tracker = None
            self._pending_tracker_context = None
            self._pending_tracker_seed_crop_zoom = False
            retired_pending = self._retired_pending_trackers
            self._retired_pending_trackers = []
        for tracker in [*retired_pending, pending]:
            if tracker is not None:
                self._close_tracker(tracker, "关闭 pending tracker 失败")
        if pending is not None or retired_pending:
            logger.info("InferenceWorker: 已 flush 未消费的 pending tracker")
        logger.info("InferenceWorker: 已停止")
        return True

    def get_fps(self):
        """获取当前FPS"""
        return self._current_fps
