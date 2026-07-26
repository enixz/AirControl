"""侧位姿态矩阵录制助手 —— 引导式逐段录制，省去手动开 app + 改名目录。

为什么需要它：benchmark_pose_matrix.py 需要"每个姿态角度一个目录"的录像，
手动用 app 的 F8 录制再逐段改名很繁琐。本助手直接开摄像头、实时预览，
按键切换下一段，自动按角度命名落盘成 benchmark 可直接消费的格式。

用法：
    python record_pose_matrix.py
    # 或自定义角度序列 / 输出根目录 / 摄像头：
    python record_pose_matrix.py --segments yaw000,yaw045,yaw090,yaw135,yaw180
    python record_pose_matrix.py --out raw_capture --camera 0 --width 1280 --height 720

按键（预览窗口聚焦时）：
    空格   开始 / 停止当前段录制
    N      下一段（停止当前段并前进）
    B      重录上一段（删除该段已录内容）
    Q/ESC  结束并退出

录制流程：每段先"准备"（对准角度）→ 空格开始 → 保持姿势 → 空格停止 →
自动进入下一段准备。全部段录完后，会打印一键对比命令。

每段目录命名：<out>/pose_<segment>/，内含 frames.mp4 + meta.jsonl，
与 replay_video.py / benchmark_ab.py / benchmark_pose_matrix.py 完全兼容。
"""

import argparse
import logging
import os
import shutil
import sys
import time

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "app"))
from services.frame_recorder import FrameRecorder  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("pose_recorder")

# 默认角度序列：先近距 5 档，再远距 5 档
DEFAULT_SEGMENTS = [
    "yaw000_near", "yaw045_near", "yaw090_near", "yaw135_near", "yaw180_near",
    "yaw000_far", "yaw045_far", "yaw090_far", "yaw135_far", "yaw180_far",
]

_SEGMENT_HINTS = {
    "yaw000": "掌心正对镜头",
    "yaw045": "手转约 45°",
    "yaw090": "手完全侧对镜头（手背/侧面）[瓶颈点]",
    "yaw135": "手转约 135°",
    "yaw180": "手背完全朝镜头",
}


def _hint(segment):
    for key, text in _SEGMENT_HINTS.items():
        if segment.startswith(key):
            dist = "远距 3–5m" if segment.endswith("_far") else "近距"
            return f"{text}（{dist}）"
    return segment


class PoseMatrixRecorder:
    def __init__(self, out_root, segments, camera, width, height, seconds_per_seg):
        self.out_root = out_root
        self.segments = segments
        self.camera_index = camera
        self.width = width
        self.height = height
        self.seconds_per_seg = seconds_per_seg
        self.idx = 0
        self.recorder = None
        self.recording = False
        self.seg_start = None
        self.done = []

    def _seg_dir(self, segment):
        return os.path.join(self.out_root, f"pose_{segment}")

    def start_segment(self):
        segment = self.segments[self.idx]
        seg_dir = self._seg_dir(segment)
        # FrameRecorder 用时间戳建子目录；我们要固定目录名，故先录到临时名再改名。
        self.recorder = FrameRecorder(out_root=self.out_root, max_seconds=self.seconds_per_seg)
        self._tmp_dir = self.recorder.dir
        self.recording = True
        self.seg_start = time.time()
        logger.info("● 开始录制 [%s]  %s", segment, _hint(segment))

    def stop_segment(self):
        if not self.recording:
            return
        segment = self.segments[self.idx]
        self.recording = False
        try:
            self.recorder.close()
        except Exception:
            logger.exception("段落收尾异常")
        # 改名时间戳目录 → pose_<segment>
        target = self._seg_dir(segment)
        try:
            if os.path.isdir(target):
                shutil.rmtree(target)
            shutil.move(self._tmp_dir, target)
        except Exception:
            logger.exception("段落改名失败，保留原始目录 %s", self._tmp_dir)
            target = self._tmp_dir
        n = self.recorder._count
        logger.info("■ 已保存 [%s]  %d 帧 → %s", segment, n, target)
        self.done.append((segment, target))
        self.recorder = None

    def next_segment(self):
        if self.recording:
            self.stop_segment()
        self.idx += 1
        if self.idx >= len(self.segments):
            return False
        return True

    def redo_segment(self):
        if self.recording:
            self.stop_segment()
            self.done.pop()  # 去掉刚保存的
        elif self.done and self.idx > 0:
            self.idx -= 1
            segment = self.segments[self.idx]
            target = self._seg_dir(segment)
            if os.path.isdir(target):
                shutil.rmtree(target, ignore_errors=True)
            if self.done and self.done[-1][0] == segment:
                self.done.pop()
        logger.info("↺ 重录 [%s]", self.segments[self.idx])

    def _open_camera(self, index):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            return None
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        return cap

    @staticmethod
    def _detect_cameras(max_index=6):
        """探测可用摄像头索引列表。"""
        found = []
        for i in range(max_index + 1):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                found.append(i)
            cap.release()
        return found

    def run(self):
        cams = self._detect_cameras()
        if self.camera_index not in cams:
            cams.append(self.camera_index)
            cams = sorted(set(cams))
        cam_pos = cams.index(self.camera_index) if self.camera_index in cams else 0

        cap = self._open_camera(self.camera_index)
        if cap is None:
            logger.error("打不开摄像头 index=%d", self.camera_index)
            return 2

        win = "Pose Matrix Recorder"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        logger.info("就绪。共 %d 段。空格=开始/停止  N=下一段  B=重录  C=切换摄像头  Q=退出", len(self.segments))
        logger.info("检测到摄像头: %s（当前 %d，按 C 切换）", cams, self.camera_index)

        while self.idx < len(self.segments):
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            frame = cv2.flip(frame, 1)  # 镜像，与 app 一致

            segment = self.segments[self.idx]
            if self.recording:
                self.recorder.write(frame)
                elapsed = time.time() - self.seg_start
                status = f"REC {segment}  {elapsed:.0f}s / {self.seconds_per_seg:.0f}s"
                color = (0, 0, 255)
                if elapsed >= self.seconds_per_seg:
                    self.stop_segment()
                    if not self.next_segment():
                        break
                    continue
            else:
                status = f"[{self.idx+1}/{len(self.segments)}] 准备: {segment}  空格开始"
                color = (0, 255, 0)

            disp = frame.copy()
            cv2.rectangle(disp, (0, 0), (disp.shape[1], 44), (0, 0, 0), -1)
            cv2.putText(disp, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(disp, f"CAM {self.camera_index}", (disp.shape[1] - 110, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(disp, _hint(segment), (10, disp.shape[0] - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.imshow(win, disp)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            elif key == ord(" "):
                if self.recording:
                    self.stop_segment()
                    if not self.next_segment():
                        break
                else:
                    self.start_segment()
            elif key == ord("n"):
                if not self.next_segment():
                    break
            elif key == ord("b"):
                self.redo_segment()
            elif key == ord("c"):
                # 循环切换摄像头（录制中也允许：换源后继续当前段）
                cam_pos = (cam_pos + 1) % len(cams)
                self.camera_index = cams[cam_pos]
                new_cap = self._open_camera(self.camera_index)
                if new_cap is not None:
                    cap.release()
                    cap = new_cap
                    logger.info("切换到摄像头 %d", self.camera_index)
                else:
                    logger.warning("摄像头 %d 打不开，仍用当前", self.camera_index)

        if self.recording:
            self.stop_segment()
        cap.release()
        cv2.destroyAllWindows()

        self._print_summary()
        return 0

    def _print_summary(self):
        if not self.done:
            logger.info("未录制任何段。")
            return
        logger.info("\n=== 录制完成，共 %d 段 ===", len(self.done))
        for segment, path in self.done:
            logger.info("  %s → %s", segment, path)
        logger.info("\n一键侧位对比（三引擎）：")
        logger.info('  python benchmark_pose_matrix.py --glob "%s"', os.path.join(self.out_root, "pose_*"))


def main(argv=None):
    ap = argparse.ArgumentParser(description="侧位姿态矩阵录制助手")
    ap.add_argument("--segments", default=",".join(DEFAULT_SEGMENTS),
                    help="逗号分隔段名（默认近/远各 5 档角度）")
    ap.add_argument("--out", default="raw_capture", help="输出根目录")
    ap.add_argument("--camera", type=int, default=0, help="摄像头索引")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--seconds", type=float, default=10.0, help="每段最长秒数（到时自动停）")
    args = ap.parse_args(argv)

    segments = [s.strip() for s in args.segments.split(",") if s.strip()]
    if not segments:
        logger.error("段序列为空")
        return 2
    rec = PoseMatrixRecorder(args.out, segments, args.camera,
                             args.width, args.height, args.seconds)
    return rec.run()


if __name__ == "__main__":
    sys.exit(main())
