import logging
import threading
import time

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

    def __init__(self, camera, tracker, max_fps=30, parent=None, debug_overlay=False,
                 frame_recorder=None):
        super().__init__(parent)
        self.camera = camera
        self.tracker = tracker
        self.frame_recorder = frame_recorder  # 可选：原始帧无损录制（默认 None）
        self.running = False
        self.lock = threading.Lock()
        self._frame_count = 0
        self._fps_start_time = time.time()
        self._current_fps = 0.0
        self.max_fps = max_fps
        self._frame_interval = 1.0 / max_fps  # 最小帧间隔
        self.debug_overlay = debug_overlay  # 调试覆盖层开关

    def run(self):
        """线程主循环"""
        self.running = True
        logging.info("InferenceWorker 启动")

        while self.running:
            try:
                self._process_frame()
            except Exception as e:
                logging.error("InferenceWorker 错误: %s", e, exc_info=True)
                self.error_occurred.emit(str(e))
                # 短暂暂停避免错误循环
                time.sleep(0.1)

        logging.info("InferenceWorker 停止")

    def _process_frame(self):
        """处理单帧（带节流）"""
        start_time = time.time()
        
        # 读取摄像头帧
        success, frame = self.camera.read_frame()
        if not success:
            time.sleep(0.01)  # 短暂等待避免忙循环
            return

        # 水平翻转（镜像）
        frame = cv2.flip(frame, 1)

        # 原始帧录制：必须在 find_hands(draw=True) 绘制叠层之前，录的是干净输入。
        if self.frame_recorder is not None:
            self.frame_recorder.write(frame)

        # 获取当前tracker（线程安全）
        with self.lock:
            current_tracker = self.tracker

        # 执行MediaPipe推理
        frame, hands_landmarks, hands_gestures = current_tracker.find_hands(
            frame, draw=True
        )

        # 计算FPS
        self._update_fps()

        # 调试覆盖层（F1 切换）
        if self.debug_overlay:
            self._draw_debug_overlay(frame, hands_landmarks, hands_gestures)

        # 发送结果到主线程
        self.frame_ready.emit(frame, hands_landmarks, hands_gestures)
        
        # 节流：确保不超过最大帧率
        elapsed = time.time() - start_time
        sleep_time = self._frame_interval - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

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
            self.tracker = new_tracker
            logging.info("InferenceWorker: tracker 已更新")

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
        self.wait()  # 等待线程结束
        logging.info("InferenceWorker: 已停止")

    def get_fps(self):
        """获取当前FPS"""
        return self._current_fps