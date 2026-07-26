"""原始相机帧录制器。

把推理线程实际喂给 find_hands 的帧（已水平翻转、尚未绘制叠层）落盘到
raw_capture/<时间戳>/，供 replay_video.py 离线回放：用同一段真实画面客观对比
任何检测/缩放/超分/参数改动的效果，免去每次真人实测。

编码默认 mp4v（MPEG-4 Part 2，有损但体积约 FFV1 的 1/5~1/10），足够回放测试。
需要无损对比（如测 JPEG 压缩对识别的影响）时设 record_raw_codec="ffv1"。
均不可用时回退 PNG 帧序列。

默认关闭：仅当 config record_raw_video=true 时由 orchestrator 创建。写入永不抛异常
打断推理线程；自带帧数/时长上限防止填满磁盘；atexit 保证进程退出时正确收尾
（mkv 索引需要 release() 才写入）。

可选真值采集（评估报告 P1-1）：record_truth=True 时同步启动 TruthEventLogger，
把"意图标记键"（默认空格，config record_truth_marker 可换）的 down/up 跳变写入
truth_events.jsonl（与 meta.jsonl 同一 epoch 时钟）。录制时边做捏合边用另一只手
点按/按住标记键，离线回放即可算出检出率/漏检率/误报/延迟，替代纯观察性指标。
"""
import atexit
import json
import logging
import os
import queue
import threading
import time

import cv2
from runtime_paths import data_path

logger = logging.getLogger(__name__)


class FrameRecorder:
    def __init__(self, out_root="raw_capture", max_frames=2000, max_seconds=120.0,
                 codec="mp4v", record_truth=False, truth_marker="space"):
        if not os.path.isabs(out_root):
            out_root = data_path(out_root)
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.dir = os.path.join(out_root, ts)
        os.makedirs(self.dir, exist_ok=True)
        self.max_frames = int(max_frames)
        self.max_seconds = float(max_seconds)
        self._codec = str(codec).lower()
        self._count = 0
        self._submitted = 0
        self._dropped = 0
        self._start = None
        self._writer = None        # cv2.VideoWriter 或 None→PNG 回退
        self._use_png = False
        self._meta = open(
            os.path.join(self.dir, "meta.jsonl"), "w", encoding="utf-8", buffering=1
        )
        self._closed = False
        self._finalized = False
        self._sentinel_queued = False
        self._truth_closed = False
        self._state_lock = threading.Lock()
        self._queue = queue.Queue(maxsize=8)
        self._thread = threading.Thread(
            target=self._writer_loop,
            name="FrameRecorderWriter",
            daemon=True,
        )
        self._thread.start()
        atexit.register(self.close)
        logger.info(
            "原始帧录制 -> %s (上限 %d 帧 / %.0fs, codec=%s)",
            self.dir, self.max_frames, self.max_seconds, self._codec,
        )
        self._truth_logger = None
        if record_truth:
            try:
                from .truth_event_logger import TruthEventLogger

                self._truth_logger = TruthEventLogger(
                    self.dir, markers=truth_marker,
                )
            except Exception as e:
                logger.warning("真值事件采集启动失败（不影响录帧）: %s", e)

    def _ensure_writer(self, w, h):
        if self._writer is not None or self._use_png:
            return
        # 按优先级尝试：用户指定 codec → mp4v（最通用）→ FFV1（无损兜底）
        if self._codec == "ffv1":
            candidates = [("FFV1", ".mkv"), ("mp4v", ".mp4")]
        else:
            candidates = [("mp4v", ".mp4"), ("FFV1", ".mkv")]
        for fourcc_str, cext in candidates:
            cpath = os.path.join(self.dir, f"frames{cext}")
            try:
                fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
                writer = cv2.VideoWriter(cpath, fourcc, 30.0, (int(w), int(h)))
                if writer.isOpened():
                    self._writer = writer
                    if fourcc_str != self._codec.upper():
                        logger.info("录帧编码回退：%s → %s", self._codec, fourcc_str)
                    return
            except Exception as e:
                logger.warning("%s 初始化异常: %s", fourcc_str, e)
        # 回退到 PNG 帧序列（始终可用、真正无损，但体积大）
        self._use_png = True
        os.makedirs(os.path.join(self.dir, "frames"), exist_ok=True)
        logger.warning("mp4v/FFV1 均不可用，回退 PNG 帧序列（体积更大）")

    def write(self, frame, meta=None):
        """Queue a clean frame without blocking the inference thread.

        meta: 可选 dict，记录该帧的检测元数据（Primary wrist、手数、是否切换等），
              会合并到 meta.jsonl 的对应行。用于离线回放重建原始运行时识别点轨迹。
        """
        if frame is None:
            return
        try:
            frame_copy = frame.copy()
            meta_copy = dict(meta) if meta else None
            now = time.time()
            with self._state_lock:
                if self._closed:
                    return
                if self._start is None:
                    self._start = now
                if (
                    self._submitted >= self.max_frames
                    or (now - self._start) >= self.max_seconds
                ):
                    return
                item = (self._submitted, now, frame_copy, meta_copy)
                self._submitted += 1
                try:
                    self._queue.put_nowait(item)
                except queue.Full:
                    self._dropped += 1
        except Exception:
            logger.exception("录帧入队失败")

    def _writer_loop(self):
        while True:
            item = self._queue.get()
            if item is None:
                return
            index, now, frame, meta = item
            try:
                h, w = frame.shape[:2]
                self._ensure_writer(w, h)
                if self._use_png:
                    cv2.imwrite(
                        os.path.join(self.dir, "frames", f"{index:06d}.png"),
                        frame,
                    )
                else:
                    self._writer.write(frame)
                row = {"i": index, "t": round(now, 4), "w": int(w), "h": int(h)}
                if meta:
                    row.update(meta)
                self._meta.write(json.dumps(row, separators=(",", ":")) + "\n")
                self._count += 1
            except Exception:
                logger.exception("录帧失败")
                return

    def close(self, timeout_sec=3.0):
        deadline = time.monotonic() + max(0.0, float(timeout_sec))
        with self._state_lock:
            if self._finalized:
                return True
            # Holding the same lock used by write() guarantees that no frame can
            # be queued after the shutdown sentinel.
            self._closed = True
        if not self._truth_closed:
            try:
                if self._truth_logger is not None:
                    truth_result = self._truth_logger.close(
                        timeout_sec=max(0.0, deadline - time.monotonic())
                    )
                    self._truth_closed = truth_result is not False
                else:
                    self._truth_closed = True
            except Exception:
                logger.exception("真值事件采集收尾失败")
        if self._thread.is_alive() and not self._sentinel_queued:
            try:
                self._queue.put(
                    None,
                    timeout=max(0.0, deadline - time.monotonic()),
                )
                self._sentinel_queued = True
            except queue.Full:
                # Never evict a submitted frame merely to make room for the
                # sentinel. A later close() retry can enqueue it after the
                # writer has drained the backlog.
                logger.error(
                    "录像队列未能在关闭期限内排空；保留全部待写帧并延后收尾"
                )
                return False
        if self._thread.is_alive():
            # The writer owns _writer and _meta until it has consumed the
            # sentinel. Closing those resources after a timeout caused the live
            # thread to write into already-closed handles.
            self._thread.join(max(0.0, deadline - time.monotonic()))
        if self._thread.is_alive():
            logger.error(
                "录像写入线程未能在 %.1f 秒内退出；保留编码器和元数据句柄",
                max(0.0, float(timeout_sec)),
            )
            return False
        if not self._truth_closed:
            logger.error("真值事件采集仍在退出，录像资源延后统一收尾")
            return False
        try:
            if self._writer is not None:
                self._writer.release()
        except Exception:
            pass
        try:
            self._meta.close()
        except Exception:
            pass
        with self._state_lock:
            self._finalized = True
        try:
            atexit.unregister(self.close)
        except Exception:
            pass
        logger.info(
            "原始帧录制结束：%d 帧（丢弃 %d）-> %s",
            self._count,
            self._dropped,
            self.dir,
        )
        return True
