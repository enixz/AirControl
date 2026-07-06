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

# 候选分辨率（高 → 低）
_RESOLUTION_CANDIDATES = [
    (1920, 1080),
    (1280, 720),
    (960, 540),
    (640, 480),
    (640, 360),
]

# 同进程多个 CameraService 共用探测结果
_PROBE_CACHE = {}

# 探测（开/关同一设备）后、正式重开同一设备前的"沉淀"时间。Windows DSHOW 的设备
# 释放是异步的：探测很快命中（如 1080p 首档即达标，仅 ~1s）时，probe 的 release()
# 底层拆除可能尚未完成，紧接着重开会句柄争用 → 原生崩溃/卡死（faulthandler 抓不到，
# 日志停在探测末尾）。给设备一点时间彻底释放再重开。0.5s 仅在切换/启动各付一次。
_DEVICE_SETTLE_SEC = 0.5


def _try_open_camera(index, backend=None):
    """尝试用指定后端打开摄像头并读取一帧。

    backend 为 None 时使用 OpenCV 默认后端。
    返回 (cap, ok) 元组；ok 为 True 表示成功打开并能读到帧。
    """
    cap = None
    try:
        if backend is None:
            cap = cv2.VideoCapture(index)
        else:
            cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            return None, False
        ok, _ = cap.read()
        if not ok:
            return None, False
        return cap, True
    except Exception:
        logger.debug("枚举摄像头 %d (backend=%s) 异常", index, backend, exc_info=True)
        return None, False


def _preferred_backends():
    """返回 Windows 上推荐的摄像头后端探测顺序。

    顺序：DSHOW（通常延迟最低）→ MSMF（Media Foundation，兼容性最好）
    → 默认后端（CAP_ANY）。某些摄像头/虚拟摄像头在 DSHOW 下无法打开，
    但在 MSMF 或默认后端下可以。
    """
    backends = []
    if hasattr(cv2, "CAP_DSHOW"):
        backends.append(cv2.CAP_DSHOW)
    if hasattr(cv2, "CAP_MSMF"):
        backends.append(cv2.CAP_MSMF)
    backends.append(None)  # CAP_ANY / 默认后端
    return backends


def list_available_cameras(max_probe=4, exclude_index=None):
    """枚举系统上可用的摄像头索引，返回 [{"index": int, "name": str, "backend": int|None}, ...]。

    实现策略：试开 0..max_probe-1，能 `open()` 且 `read()` 到一帧的算可用。
    先在 Windows 上按 DSHOW → MSMF → 默认后端顺序尝试；记录每个可用索引
    实际能打开的后端，供 CameraService 启动时复用。
    每个不存在的索引在 Windows 上 cv2.VideoCapture 会阻塞 0.5-2s，因此函数本身耗时不小，
    调用方应放到后台线程里跑。

    Args:
        max_probe: 探测到哪个索引为止（不含），默认 0..3 共 4 个
        exclude_index: 当前正在被本进程使用的摄像头索引（独占设备无法同时再开），
                       跳过探测但仍计入结果，name 后缀加 "（当前）"
    """
    results = []
    backends = _preferred_backends()
    for idx in range(max_probe):
        if exclude_index is not None and idx == exclude_index:
            results.append({"index": idx, "name": f"摄像头 {idx}（当前）", "backend": None})
            continue

        cap = None
        ok = False
        used_backend = None
        for backend in backends:
            cap, ok = _try_open_camera(idx, backend)
            if ok:
                used_backend = backend
                break

        if ok:
            backend_name = _backend_name(used_backend)
            results.append({
                "index": idx,
                "name": f"摄像头 {idx} ({backend_name})",
                "backend": used_backend,
            })

        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
    return results


def _backend_name(backend):
    """将后端常量转为可读的短名称，用于日志/菜单显示。"""
    if backend == getattr(cv2, "CAP_DSHOW", None):
        return "DSHOW"
    if backend == getattr(cv2, "CAP_MSMF", None):
        return "MSMF"
    if backend is None:
        return "默认"
    return str(backend)


def probe_max_resolution(camera_index, min_fps=20, force_mjpeg=True, use_cache=True, preferred_backend=None):
    """探测摄像头能稳定跑到 min_fps 的最高分辨率，返回 (w, h) 或 None。

    返回的元组保证：(a) 驱动确实接受这个分辨率，(b) 实测帧率 ≥ min_fps。
    若所有候选都达不到帧率要求，回退到"驱动接受的最低分辨率"作 best-effort。
    优先使用调用方指定的后端；未指定时按 DSHOW → MSMF → 默认后端顺序尝试。
    """
    if use_cache and camera_index in _PROBE_CACHE:
        return _PROBE_CACHE[camera_index]

    cap = None
    opened_backend = None
    if preferred_backend is not None:
        cap = cv2.VideoCapture(camera_index, preferred_backend)
        if cap.isOpened():
            opened_backend = preferred_backend

    if cap is None or not cap.isOpened():
        for backend in _preferred_backends():
            cap = cv2.VideoCapture(camera_index, backend) if backend is not None else cv2.VideoCapture(camera_index)
            if cap.isOpened():
                opened_backend = backend
                break

    if cap is None or not cap.isOpened():
        logger.warning("摄像头 %d 所有后端均无法打开，跳过分辨率探测", camera_index)
        return None

    logger.info("摄像头 %d 分辨率探测使用后端: %s", camera_index, _backend_name(opened_backend))

    if force_mjpeg:
        # 在所有后端下都尝试设置 MJPG。DSHOW 下行为最稳定；MSMF/默认后端下
        # 某些驱动也会接受 MJPG，能显著提升高分辨率帧率（YUY2 在 1080p 下带宽不够）。
        # 即使设置失败也不会报错，只是保持驱动默认格式。
        mjpg_fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        cap.set(cv2.CAP_PROP_FOURCC, mjpg_fourcc)
        actual_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        actual_fourcc_str = "".join(chr((actual_fourcc >> 8 * i) & 0xFF) for i in range(4))
        logger.info("  尝试设置 MJPG，实际 fourcc=%s", actual_fourcc_str)

    chosen = None
    last_accepted = None  # 用作 "没有任何分辨率达到帧率" 时的兜底

    for w, h in _RESOLUTION_CANDIDATES:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_FPS, 30)
        # 变更分辨率后再次设置 MJPG：部分 Windows 驱动（DSHOW/MSMF）在 width/height
        # 改变时会复位 FOURCC 为默认值（通常 YUY2），导致 1080p/720p 因带宽不足
        # 达不到 min_fps、一路回退到 480p。这里在每档分辨率后补一次 MJPG，并校验
        # 实际生效的 FOURCC 用于日志诊断。
        if force_mjpeg:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

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

        # 读取实际生效的 FOURCC：用于诊断"驱动接受了分辨率但实际仍以 YUY2 跑"
        # 的情况（MJPG 没设上 → 高分辨率带宽不够 → fps 不达标）。
        actual_fourcc_val = int(cap.get(cv2.CAP_PROP_FOURCC))
        actual_fourcc_str = "".join(
            chr((actual_fourcc_val >> 8 * i) & 0xFF) for i in range(4)
        )
        logger.info(
            "  探测 %dx%d → 实测 %.1f fps (%d/15 帧, fourcc=%s)",
            w, h, real_fps, ok_frames, actual_fourcc_str,
        )

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
        preferred_backend=None,
    ):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.force_mjpeg = force_mjpeg
        self.min_fps = min_fps
        self.cap = None
        # 实际生效的 OpenCV 后端：DSHOW / MSMF / 默认（None）。
        # 在 Windows 上某些摄像头/驱动只能用特定后端打开，因此启动时动态探测
        # 并记录，重连时复用同一后端。preferred_backend 由 list_available_cameras
        # 传入时优先使用。
        self._backend = preferred_backend if preferred_backend is not None else cv2.CAP_DSHOW
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
        # 1. 未指定分辨率 → 自动探测（复用 list_available_cameras 选定的后端）
        probed = False
        if not self.width or not self.height:
            logger.info(
                "正在探测摄像头 %d 的最高可用分辨率（后端: %s）…",
                self.camera_index, _backend_name(self._backend),
            )
            detected = probe_max_resolution(
                self.camera_index,
                min_fps=self.min_fps,
                force_mjpeg=self.force_mjpeg,
                preferred_backend=self._backend,
            )
            probed = True
            if detected:
                self.width, self.height = detected
            else:
                self.width, self.height = 1280, 720
                logger.warning("探测失败，回退到 1280×720")

        # 刚探测过同一设备 → 等它彻底释放再重开，避免 DSHOW 异步释放争用导致的
        # 原生崩溃（详见 _DEVICE_SETTLE_SEC 注释）。仅探测路径需要。
        if probed:
            time.sleep(_DEVICE_SETTLE_SEC)

        # 2. 正式打开（优先指定后端，失败按 DSHOW → MSMF → 默认后端回退）
        # 这条日志把"探测结束 → 设备重开"之间的盲区补上：若崩溃仍发生，日志会
        # 停在此行之后、"启动:"之前，即可确认死在设备重开/参数设置这段原生调用。
        logger.info(
            "打开摄像头 %d @ %dx%d（后端: %s）…",
            self.camera_index, self.width or 0, self.height or 0,
            _backend_name(self._backend),
        )
        self.cap = cv2.VideoCapture(self.camera_index, self._backend) if self._backend is not None else cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            logger.warning(
                "摄像头 %d 后端 %s 打开失败，尝试其他后端",
                self.camera_index, _backend_name(self._backend),
            )
            for backend in _preferred_backends():
                if backend == self._backend:
                    continue
                self.cap = cv2.VideoCapture(self.camera_index, backend) if backend is not None else cv2.VideoCapture(self.camera_index)
                if self.cap.isOpened():
                    self._backend = backend
                    logger.info("摄像头 %d 使用回退后端: %s", self.camera_index, _backend_name(backend))
                    break

        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开摄像头 {self.camera_index}")

        if self.force_mjpeg:
            # 在所有后端下都尝试设置 MJPG（与 probe_max_resolution 保持一致）。
            # DSHOW 下行为最稳定；MSMF/默认后端下某些驱动也会接受，能显著提升
            # 高分辨率帧率。设置失败不会报错，保持驱动默认格式。
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # 变更分辨率后再次设置 MJPG：部分驱动在 width/height 改变时会复位 FOURCC，
        # 导致实际仍以 YUY2 跑、高分辨率帧率上不去。与 probe_max_resolution 保持一致。
        if self.force_mjpeg:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

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

            self.cap = cv2.VideoCapture(self.camera_index, self._backend)
            if self.force_mjpeg:
                # 与 start() 保持一致：在所有后端下都尝试设置 MJPG（不再只限 DSHOW）。
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            # 变更分辨率后再次设置 MJPG：驱动可能复位 FOURCC，与 start/probe 保持一致。
            if self.force_mjpeg:
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

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
        # 复位状态，避免 release 后 read_frame 误判为"cap 存在但未打开"而触发重连
        self.cap = None
        self._disconnected = False
        self._consecutive_failures = 0
        self._reconnect_backoff = 1.0
