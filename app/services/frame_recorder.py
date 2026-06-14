"""原始相机帧无损录制器。

把推理线程实际喂给 find_hands 的帧（已水平翻转、尚未绘制叠层）无损落盘到
raw_capture/<时间戳>/，供 replay_video.py 离线回放：用同一段真实画面客观对比
任何检测/缩放/超分/参数改动的效果，免去每次真人实测。

无损（FFV1 视频，不可用则回退 PNG 帧序列）是刻意的——不引入任何压缩痕迹，
这样回放时还能单独做 JPEG 压缩来测「压缩对识别的影响」（replay_video --jpeg-quality）。

默认关闭：仅当 config record_raw_video=true 时由 orchestrator 创建。写入永不抛异常
打断推理线程；自带帧数/时长上限防止填满磁盘；atexit 保证进程退出时正确收尾
（FFV1 的 mkv 索引需要 release() 才写入）。
"""
import atexit
import json
import logging
import os
import time

import cv2

logger = logging.getLogger(__name__)


class FrameRecorder:
    def __init__(self, out_root="raw_capture", max_frames=2000, max_seconds=120.0):
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.dir = os.path.join(out_root, ts)
        os.makedirs(self.dir, exist_ok=True)
        self.max_frames = int(max_frames)
        self.max_seconds = float(max_seconds)
        self._count = 0
        self._start = None
        self._writer = None        # cv2.VideoWriter(FFV1) 或 None→PNG 回退
        self._use_png = False
        self._meta = open(
            os.path.join(self.dir, "meta.jsonl"), "w", encoding="utf-8", buffering=1
        )
        self._closed = False
        atexit.register(self.close)
        logger.info(
            "原始帧录制 -> %s (上限 %d 帧 / %.0fs)",
            self.dir, self.max_frames, self.max_seconds,
        )

    def _ensure_writer(self, w, h):
        if self._writer is not None or self._use_png:
            return
        path = os.path.join(self.dir, "frames.mkv")
        try:
            fourcc = cv2.VideoWriter_fourcc(*"FFV1")  # 无损
            writer = cv2.VideoWriter(path, fourcc, 30.0, (int(w), int(h)))
            if writer.isOpened():
                self._writer = writer
                return
        except Exception as e:
            logger.warning("FFV1 初始化异常: %s", e)
        # 回退到 PNG 帧序列（始终可用、真正无损）
        self._use_png = True
        os.makedirs(os.path.join(self.dir, "frames"), exist_ok=True)
        logger.warning("FFV1 不可用，回退 PNG 帧序列（体积更大）")

    def write(self, frame):
        """录一帧。永不向调用方抛异常（推理线程不能被录制拖垮）。"""
        if self._closed or frame is None:
            return
        try:
            now = time.time()
            if self._start is None:
                self._start = now
            if self._count >= self.max_frames or (now - self._start) >= self.max_seconds:
                self.close()
                return
            h, w = frame.shape[:2]
            self._ensure_writer(w, h)
            if self._use_png:
                cv2.imwrite(
                    os.path.join(self.dir, "frames", f"{self._count:06d}.png"), frame
                )
            else:
                self._writer.write(frame)
            self._meta.write(
                json.dumps(
                    {"i": self._count, "t": round(now, 4), "w": int(w), "h": int(h)},
                    separators=(",", ":"),
                )
                + "\n"
            )
            self._count += 1
        except Exception as e:
            logger.error("录帧失败，停止录制: %s", e)
            self.close()

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            if self._writer is not None:
                self._writer.release()
        except Exception:
            pass
        try:
            self._meta.close()
        except Exception:
            pass
        logger.info("原始帧录制结束：%d 帧 -> %s", self._count, self.dir)
