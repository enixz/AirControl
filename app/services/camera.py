"""摄像头采集服务 — 自动探测设备最高可用分辨率。

设计目标：在各种设备上零配置工作。
  • HD-3000 (720p 上限) → 自动选 1280×720
  • 笔记本内置 (1080p 但只能 15fps) → 自动降到 720p 保证流畅
  • 4K 直播摄像头 → 自动选 1920×1080 (再高 MediaPipe 跑不动)
  • 旧摄像头只支持 480p → 自动选 640×480

策略：从高到低试候选分辨率，第一个满足 "(a) 驱动接受 (b) 实测帧率达标" 的胜出。
"""

import logging
import time

import cv2

logger = logging.getLogger(__name__)

# 候选分辨率（高 → 低）。不放 4K：MediaPipe 在 4K 下推理成本爆炸，且消费级摄像头几乎没有
_RESOLUTION_CANDIDATES = [
    (1920, 1080),
    (1280, 720),
    (960, 540),
    (640, 480),
    (640, 360),
]

# 同进程多个 CameraService 共用探测结果
_PROBE_CACHE = {}


def list_available_cameras(max_probe=4, exclude_index=None):
    """枚举系统上可用的摄像头索引，返回 [{"index": int, "name": str}, ...]。

    实现策略：试开 0..max_probe-1，能 `open()` 且 `read()` 到一帧的算可用。
    每个不存在的索引在 Windows 上 cv2.VideoCapture 会阻塞 0.5-2s，因此函数本身耗时不小，
    调用方应放到后台线程里跑。

    Args:
        max_probe: 探测到哪个索引为止（不含），默认 0..3 共 4 个
        exclude_index: 当前正在被本进程使用的摄像头索引（独占设备无法同时再开），
                       跳过探测但仍计入结果，name 后缀加 "（当前）"
    """
    results = []
    for idx in range(max_probe):
        if exclude_index is not None and idx == exclude_index:
            results.append({"index": idx, "name": f"摄像头 {idx}（当前）"})
            continue
        cap = None
        try:
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if not cap.isOpened():
                continue
            ok, _ = cap.read()
            if not ok:
                continue
            results.append({"index": idx, "name": f"摄像头 {idx}"})
        except Exception:
            logger.debug("枚举摄像头 %d 异常", idx, exc_info=True)
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
    return results


def probe_max_resolution(camera_index, min_fps=20, force_mjpeg=True, use_cache=True):
    """探测摄像头能稳定跑到 min_fps 的最高分辨率，返回 (w, h) 或 None。

    返回的元组保证：(a) 驱动确实接受这个分辨率，(b) 实测帧率 ≥ min_fps。
    若所有候选都达不到帧率要求，回退到"驱动接受的最低分辨率"作 best-effort。
    """
    if use_cache and camera_index in _PROBE_CACHE:
        return _PROBE_CACHE[camera_index]

    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        logger.warning("摄像头 %d 无法打开，跳过分辨率探测", camera_index)
        return None

    if force_mjpeg:
        # 必须在设分辨率前设 fourcc——HD-3000 等老摄像头在 720p YUY2 下只能 10fps，
        # 切到 MJPG 才能上 30fps
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    chosen = None
    last_accepted = None  # 用作 "没有任何分辨率达到帧率" 时的兜底

    for w, h in _RESOLUTION_CANDIDATES:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_FPS, 30)

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if actual_w != w or actual_h != h:
            logger.debug("  探测 %dx%d 驱动拒绝（实际 %dx%d）", w, h, actual_w, actual_h)
            continue

        # 驱动接受了——抓 15 帧测真实帧率
        ok_frames = 0
        t0 = time.time()
        for _ in range(15):
            ok, _ = cap.read()
            if ok:
                ok_frames += 1
        elapsed = time.time() - t0
        real_fps = ok_frames / elapsed if elapsed > 0 else 0

        logger.info("  探测 %dx%d → 实测 %.1f fps (%d/15 帧)", w, h, real_fps, ok_frames)

        last_accepted = (w, h)
        if real_fps >= min_fps and ok_frames >= 12:
            chosen = (w, h)
            break  # 找到最高合格分辨率，提前结束

    cap.release()

    result = chosen or last_accepted
    if chosen:
        logger.info("摄像头 %d 选定: %dx%d (满足 ≥%d fps)", camera_index, *result, min_fps)
    elif last_accepted:
        logger.warning(
            "摄像头 %d 没有分辨率达到 %d fps，回退到 %dx%d",
            camera_index, min_fps, *result,
        )
    else:
        logger.error("摄像头 %d 探测失败，所有候选分辨率都被驱动拒绝", camera_index)

    if use_cache:
        _PROBE_CACHE[camera_index] = result
    return result


class CameraService:
    """摄像头采集。

    width/height 任一为 None → 启动时自动探测最高可用分辨率。
    显式指定数值则使用指定值（用户想强制时可在 config.json 设 camera_width/height）。
    """

    def __init__(
        self,
        camera_index=0,
        width=None,
        height=None,
        force_mjpeg=True,
        min_fps=20,
        reconnect_after_failures=15,
        reconnect_cooldown_sec=1.5,
    ):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.force_mjpeg = force_mjpeg
        self.min_fps = min_fps
        self.cap = None
        # 重连逻辑：连续 N 帧读取失败触发自动重新打开摄像头
        # 适用场景：USB 被意外拔出、驱动崩溃、被其他程序短暂抢占
        self._consecutive_failures = 0
        self._reconnect_threshold = reconnect_after_failures
        self._reconnect_base_cooldown = reconnect_cooldown_sec
        self._reconnect_backoff = 1.0  # 指数退避系数（每次失败 ×2，上限 20）
        self._last_reconnect_attempt = 0.0
        self._reconnecting = False
        self._disconnected = False  # 安静期标志——进入断连后只打一次 warning

    def start(self):
        # 1. 未指定分辨率 → 自动探测
        if not self.width or not self.height:
            logger.info("正在探测摄像头 %d 的最高可用分辨率…", self.camera_index)
            detected = probe_max_resolution(
                self.camera_index,
                min_fps=self.min_fps,
                force_mjpeg=self.force_mjpeg,
            )
            if detected:
                self.width, self.height = detected
            else:
                self.width, self.height = 1280, 720
                logger.warning("探测失败，回退到 1280×720")

        # 2. 正式打开
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if self.force_mjpeg:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开摄像头 {self.camera_index}")

        # 3. 回报实际生效参数
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        fourcc_val = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        fourcc_str = "".join(chr((fourcc_val >> 8 * i) & 0xFF) for i in range(4))
        logger.info(
            "摄像头 %d 启动: %dx%d@%.1ffps (%s)",
            self.camera_index, actual_w, actual_h, actual_fps, fourcc_str,
        )

    def read_frame(self):
        if self.cap is None or not self.cap.isOpened():
            return self._try_reconnect()

        ok, frame = self.cap.read()
        if ok and frame is not None:
            # 读到帧——若处于断连状态，标记恢复并打一条 info
            if self._disconnected:
                logger.info("摄像头已恢复连接 (%dx%d)", self.width or 0, self.height or 0)
                self._disconnected = False
                self._reconnect_backoff = 1.0
            self._consecutive_failures = 0
            return ok, frame

        self._consecutive_failures += 1
        if self._consecutive_failures >= self._reconnect_threshold:
            return self._try_reconnect()
        return False, None

    def _try_reconnect(self):
        """安静的指数退避重连：不打日志洪水，不重做分辨率探测。

        进入断连状态时只打一次 warning，恢复时只打一次 info。
        每次失败后冷却时间翻倍（1.5s → 3 → 6 → 12 → 24 → 30s 封顶），
        避免摄像头被拔走 1 分钟期间疯狂尝试 + 刷日志 + 触发 OpenCV C++ stderr 警告。
        """
        now = time.time()
        current_cooldown = min(self._reconnect_base_cooldown * self._reconnect_backoff, 30.0)
        if self._reconnecting or (now - self._last_reconnect_attempt) < current_cooldown:
            return False, None

        self._reconnecting = True
        self._last_reconnect_attempt = now

        # 进入断连状态：只在首次进入时打一条 warning，后续静默重试
        if not self._disconnected:
            logger.warning(
                "摄像头断开（连续 %d 帧失败），后台自动重连中…",
                self._consecutive_failures,
            )
            self._disconnected = True

        try:
            # 释放旧句柄
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None

            # 关键：跳过完整探测，直接用上次成功的分辨率重开
            # 探测会跑 5×15=75 次 cap.read()，每次失败都让 OpenCV C++ 喷 stderr
            if not (self.width and self.height):
                # 极少数情况下首次启动就失败：兜底 720p
                self.width, self.height = 1280, 720

            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            if self.force_mjpeg:
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not self.cap.isOpened():
                # 静默失败 + 退避
                self.cap.release() if self.cap else None
                self.cap = None
                self._reconnect_backoff = min(self._reconnect_backoff * 2.0, 20.0)
                return False, None

            # 试读一帧验证（驱动可能"假打开"）
            ok, frame = self.cap.read()
            if ok and frame is not None:
                # 真的恢复了——read_frame 下次成功时会打 "已恢复连接" 并复位状态
                self._consecutive_failures = 0
                return ok, frame

            # cap 假打开：再退避
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
            self._reconnect_backoff = min(self._reconnect_backoff * 2.0, 20.0)
            return False, None

        except Exception:
            # 静默吃异常 + 退避，断连期间不刷日志
            try:
                if self.cap:
                    self.cap.release()
            except Exception:
                pass
            self.cap = None
            self._reconnect_backoff = min(self._reconnect_backoff * 2.0, 20.0)
            return False, None
        finally:
            self._reconnecting = False

    def release(self):
        if self.cap:
            self.cap.release()
