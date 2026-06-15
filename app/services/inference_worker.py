import logging
import threading
import time
from collections import deque

import cv2
from PyQt6.QtCore import QThread, pyqtSignal


class InferenceWorker(QThread):
    """
    后台推理线程，负责摄像头读取和MediaPipe推理。
    通过信号将结果发送到主线程处理模式逻辑和UI更新。
    """

    # 信号：帧数据、手部关键点、手势识别结果
    frame_ready = pyqtSignal(object, list, list)
    error_occurred = pyqtSignal(str)
    fps_updated = pyqtSignal(float)
    performance_updated = pyqtSignal(object)

    def __init__(self, camera, tracker, max_fps=30, parent=None, debug_overlay=False,
                 frame_recorder=None):
        super().__init__(parent)
        self.camera = camera
        self.tracker = tracker
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
        logging.info("InferenceWorker 启动")

        try:
            while self.running:
                item = self._wait_for_latest_frame()
                if item is None:
                    continue
                try:
                    self._process_frame(*item)
                except Exception as e:
                    logging.error("InferenceWorker 错误: %s", e, exc_info=True)
                    self.error_occurred.emit(str(e))
                    time.sleep(0.1)
        finally:
            self._capture_stop.set()
            with self._capture_condition:
                self._capture_condition.notify_all()
            if self._capture_thread and self._capture_thread.is_alive():
                self._capture_thread.join(timeout=3.0)
            logging.info("InferenceWorker 停止")

    def _capture_loop(self):
        """Continuously capture frames and retain only the newest one."""
        while not self._capture_stop.is_set():
            started = time.perf_counter()
            success, frame = self.camera.read_frame()
            capture_ms = (time.perf_counter() - started) * 1000.0
            if not success:
                self._capture_stop.wait(0.01)
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

        # 原始帧录制：必须在 find_hands(draw=True) 绘制叠层之前，录的是干净输入。
        if self.frame_recorder is not None:
            self.frame_recorder.write(frame)

        inference_started = time.perf_counter()
        # Hold the lock through native inference so an old detector cannot be
        # closed while MediaPipe is still using it.
        with self.lock:
            frame, hands_landmarks, hands_gestures = self.tracker.find_hands(
                frame, draw=True
            )
        inference_ms = (time.perf_counter() - inference_started) * 1000.0

        # 计算FPS
        self._update_fps()

        # 调试覆盖层（F1 切换）
        if self.debug_overlay:
            self._draw_debug_overlay(frame, hands_landmarks, hands_gestures)

        # Qt 的 queued signal 没有天然背压。主线程忙时只保留一个待处理结果，
        # 避免旧帧持续堆积并放大端到端延迟。
        if self._claim_result_slot():
            self.frame_ready.emit(frame, hands_landmarks, hands_gestures)

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

    def update_tracker(self, new_tracker):
        """线程安全地更新tracker"""
        with self.lock:
            old_tracker = self.tracker
            self.tracker = new_tracker
        if old_tracker is not new_tracker:
            close = getattr(old_tracker, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    logging.exception("关闭旧 tracker 失败")
        logging.info("InferenceWorker: tracker 已更新")

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
        logging.info("调试覆盖层: %s", "开" if self.debug_overlay else "关")

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
        for i, (landmarks, gesture) in enumerate(zip(hands_landmarks, hands_gestures)):
            if not landmarks:
                continue
            wrist = landmarks[0]
            x, y = int(wrist[1]), int(wrist[2])
            label = f"#{i} {gesture.get('handedness', '?')[:1]}"
            if gesture.get("predicted"):
                label += " [predict]"
                color = (0, 255, 255)
            else:
                color = (255, 0, 255) if i == 0 else (200, 200, 200)
            cv2.putText(
                frame, label, (max(x + 10, 10), max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1,
            )

    def stop(self):
        """停止线程"""
        self.running = False
        self._capture_stop.set()
        with self._capture_condition:
            self._capture_condition.notify_all()
        if not self.wait(5000):
            logging.warning("InferenceWorker 5 秒内未能停止")
        logging.info("InferenceWorker: 已停止")

    def get_fps(self):
        """获取当前FPS"""
        return self._current_fps
