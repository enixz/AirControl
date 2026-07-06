"""超分辨率引擎调度 — 从 base_hand_tracker.py 拆出的独立职责。

集中管理 ESPCN / Real-ESRGAN(GPU/CPU) 的按需加载、GPU 自适应探测、
auto 档位解析。BaseHandTracker 持有 self._sr = SREngine(...)，crop-zoom
路径通过它放大裁剪区域；find_hands 在 ZOOM OFF/丢手时调 reset_tier()
清空档位记忆。

状态全部封装在本类实例内，BaseHandTracker 不再直接持有 _sr_*/_espcn_*/
_realesrgan_* 属性。
"""

import logging
import os

import cv2
import numpy as np
from runtime_paths import resource_path


class SREngine:
    """超分引擎调度：ESPCN / Real-ESRGAN(GPU/CPU) 按需加载 + GPU 自适应。"""

    def __init__(self, logger=None):
        self._logger = logger or logging.getLogger("gesture")
        self._sr_initialized = False
        self._espcn_engine = None
        self._espcn_path = resource_path("ESPCN_x2.pb")
        self._realesrgan_path = resource_path("Real-ESRGAN_x2plus.onnx")
        self._realesrgan_cpu_session = None
        self._realesrgan_gpu_session = None
        self._realesrgan_input_name = None
        self._realesrgan_gpu_available = None  # None=未探测, True/False=已探测
        self._realesrgan_gpu_provider = None   # 实际可用的 GPU provider 名称
        self._auto_sr_enabled = None           # auto 档位滞回判定状态
        self._last_sr_tier = None              # 上次生效的放大档位（日志去重）

    def init(self):
        """初始化超分辨率引擎状态（不在此处真正加载模型）。

        仅准备模型路径与占位变量；具体引擎在首次被选中时由
        _ensure_espcn() / _ensure_realesrgan() **按需加载**，避免一次性把
        ESPCN + Real-ESRGAN(GPU) + Real-ESRGAN(CPU) 三份模型全部常驻内存，
        同时支持运行时切换引擎时再加载所需的那一个。
        """
        if self._sr_initialized:
            return

        self._sr_initialized = True
        self._espcn_engine = None
        self._realesrgan_cpu_session = None
        self._realesrgan_gpu_session = None
        self._realesrgan_input_name = None
        self._realesrgan_gpu_available = None
        self._realesrgan_gpu_provider = None

        self._espcn_path = resource_path("ESPCN_x2.pb")
        self._realesrgan_path = resource_path("Real-ESRGAN_x2plus.onnx")

    def release(self):
        """释放超分辨率引擎持有的 ONNX session 和 OpenCV DNN 资源。

        ONNX session 持有 GPU 内存或线程池，反复创建/销毁 tracker 时若不释放
        会泄漏显存。ESPCN 的 DnnSuperResImpl 由 Python GC 回收，无需显式释放。
        """
        if not self._sr_initialized:
            return
        for attr in ("_realesrgan_cpu_session", "_realesrgan_gpu_session"):
            sess = getattr(self, attr, None)
            if sess is not None:
                try:
                    del sess
                except Exception:
                    pass
                setattr(self, attr, None)
        self._espcn_engine = None
        self._sr_initialized = False

    def reset_tier(self):
        """清空 auto 档位与日志去重状态。

        find_hands 在 ZOOM OFF / 丢手 / ZOOM miss 超阈值时调用，使下一段
        crop-zoom 重新决策 SR 档位并重新打印日志。
        """
        self._last_sr_tier = None
        self._auto_sr_enabled = None

    def _ensure_espcn(self):
        """按需加载 ESPCN（OpenCV dnn_superres，CPU）。返回引擎或 None。"""
        if self._espcn_engine is not None:
            return self._espcn_engine
        if not os.path.exists(self._espcn_path):
            return None
        try:
            sr = cv2.dnn_superres.DnnSuperResImpl_create()
            sr.readModel(self._espcn_path)
            sr.setModel("espcn", 2)
            self._espcn_engine = sr
            self._logger.info("[SR] ESPCN model loaded successfully.")
        except Exception as e:
            self._logger.error("[SR] Failed to load ESPCN: %s", e)
            self._espcn_engine = None
        return self._espcn_engine

    def _ensure_realesrgan(self, prefer_gpu):
        """按需加载 Real-ESRGAN ONNX session。

        返回 (session, input_name)，不可用时返回 (None, None)。仅加载当前需要的
        provider；input_name 取自**真正加载成功**的 session（修复此前只在 CPU
        分支赋值导致 GPU-only 时被静默跳过的问题）。

        GPU 支持：优先 CUDAExecutionProvider，其次 DirectMLExecutionProvider
        （Windows 常见），最后回退 CPU。通过 _detect_gpu_availability() 统一探测。
        """
        if not os.path.exists(self._realesrgan_path):
            return None, None

        import onnxruntime as ort

        if prefer_gpu:
            if self._realesrgan_gpu_session is not None:
                return self._realesrgan_gpu_session, self._realesrgan_input_name
            if self._realesrgan_gpu_available is None:
                self._detect_gpu_availability()
            if self._realesrgan_gpu_available and self._realesrgan_gpu_provider:
                try:
                    sess = ort.InferenceSession(
                        self._realesrgan_path,
                        providers=[self._realesrgan_gpu_provider, "CPUExecutionProvider"],
                    )
                    self._realesrgan_gpu_session = sess
                    self._realesrgan_input_name = sess.get_inputs()[0].name
                    self._logger.info(
                        "[SR] Real-ESRGAN loaded on GPU (%s).",
                        self._realesrgan_gpu_provider,
                    )
                    return sess, self._realesrgan_input_name
                except Exception as e:
                    self._logger.warning(
                        "[SR] Failed to init Real-ESRGAN on GPU, fallback to CPU: %s", e
                    )
                    self._realesrgan_gpu_available = False
                    self._realesrgan_gpu_provider = None
            # GPU 不可用 → 回退 CPU

        if self._realesrgan_cpu_session is not None:
            return self._realesrgan_cpu_session, self._realesrgan_input_name
        try:
            sess = ort.InferenceSession(
                self._realesrgan_path, providers=["CPUExecutionProvider"]
            )
            self._realesrgan_cpu_session = sess
            self._realesrgan_input_name = sess.get_inputs()[0].name
            self._logger.info("[SR] Real-ESRGAN loaded on CPU.")
            return sess, self._realesrgan_input_name
        except Exception as e:
            self._logger.error("[SR] Failed to load Real-ESRGAN session: %s", e)
            return None, None

    def resolve(self, sr_engine, crop_size, target):
        """将配置（含 auto）解析为具体引擎名。

        auto 模式下的 GPU 自适应：
          1. 检测 GPU 是否可用（ONNX Runtime CUDA EP）
          2. GPU 可用 + 需要超分 → realesrgan_gpu（质量更好）
          3. GPU 不可用 + 需要超分 → espcn（CPU 轻量）
          4. 不需要超分（crop >= target）→ none（纯插值）

        例外（近距离关闭 SR）：当 crop_size >= target 时，裁剪框本就 ≥ 目标分辨率，
        这是"下采样"场景——超分加不了任何细节，还白白吃算力。此时直接用普通插值。
        """
        if sr_engine == "auto":
            # 滞回判定：是否需要超分
            if self._auto_sr_enabled is None:
                self._auto_sr_enabled = crop_size < target
            elif self._auto_sr_enabled and crop_size >= target * 1.10:
                self._auto_sr_enabled = False
            elif not self._auto_sr_enabled and crop_size <= target * 0.90:
                self._auto_sr_enabled = True

            if not self._auto_sr_enabled:
                return "none"

            # GPU 自适应：首次进入超分时探测 GPU 可用性
            if getattr(self, '_realesrgan_gpu_available', None) is None:
                self._detect_gpu_availability()

            if self._realesrgan_gpu_available:
                return "realesrgan_gpu"
            return "espcn"
        return sr_engine

    def _detect_gpu_availability(self):
        """探测 GPU 是否可用于 ONNX Runtime 推理。

        支持 CUDA（NVIDIA）和 DirectML（Windows 通用 GPU）。
        只探测一次，结果缓存到 _realesrgan_gpu_available 和 _realesrgan_gpu_provider。
        """
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            # 优先 CUDA，其次 DirectML；两者都可用时选 CUDA（通常推理效率更高）
            gpu_provider = None
            if 'CUDAExecutionProvider' in providers:
                gpu_provider = 'CUDAExecutionProvider'
            elif 'DmlExecutionProvider' in providers:
                gpu_provider = 'DmlExecutionProvider'

            if gpu_provider and os.path.exists(self._realesrgan_path):
                self._realesrgan_gpu_available = True
                self._realesrgan_gpu_provider = gpu_provider
                self._logger.info(
                    "[SR] GPU 自适应：检测到 %s，将使用 Real-ESRGAN GPU 超分",
                    gpu_provider,
                )
            else:
                self._realesrgan_gpu_available = False
                self._realesrgan_gpu_provider = None
                if not gpu_provider:
                    self._logger.info("[SR] GPU 自适应：未检测到 GPU provider，使用 ESPCN CPU 超分")
                else:
                    self._logger.info("[SR] GPU 自适应：GPU 可用但模型文件缺失，使用 ESPCN CPU 超分")
        except Exception as e:
            self._realesrgan_gpu_available = False
            self._realesrgan_gpu_provider = None
            self._logger.info("[SR] GPU 自适应：探测失败(%s)，使用 ESPCN CPU 超分", e)

    def log_tier(self, tier, crop_size, target):
        """记录本次 crop-zoom 实际生效的放大档位（ESPCN / Real-ESRGAN / none 插值）。

        仅在档位变化时打印，避免逐帧刷屏；每次 ZOOM OFF 会复位 _last_sr_tier，
        因此每段 ZOOM 会重新记录一次。
        """
        if tier == self._last_sr_tier:
            return
        self._last_sr_tier = tier
        self._logger.info(
            "[SR] zoom upscaler -> %s (crop=%dpx, target=%dpx)", tier, crop_size, target
        )

    def espcn(self, crop, target):
        """ESPCN 超分：限幅输入到约 target/2，2x 放大后重采样到 target。

        ESPCN 开销随输入像素数增长（240² 在 CPU 上约 30ms）。由于其放大倍率固定为
        2，把输入限制到约 target/2 既能让输出≈target、又能把单帧开销压到固定上限内
        （≈target/2 输入约 18ms）；当 crop 本就更小则直接喂入不再下采样。
        失败/不可用返回 None。
        """
        engine = self._ensure_espcn()
        if engine is None:
            return None
        try:
            cap = max(64, target // 2)
            src = crop
            longest = max(crop.shape[0], crop.shape[1])
            if longest > cap:
                s = cap / float(longest)
                src = cv2.resize(
                    crop,
                    (max(1, int(round(crop.shape[1] * s))), max(1, int(round(crop.shape[0] * s)))),
                    interpolation=cv2.INTER_AREA,
                )
            out = engine.upsample(src)  # 2x
            if out.shape[0] == target and out.shape[1] == target:
                return out
            interp = cv2.INTER_AREA if out.shape[0] > target else cv2.INTER_LINEAR
            return cv2.resize(out, (target, target), interpolation=interp)
        except Exception as e:
            self._logger.error("[SR] ESPCN upsampling failed: %s", e)
            return None

    def realesrgan(self, crop, target, prefer_gpu):
        """Real-ESRGAN 超分（固定 64x64 输入的导出）。

        该 ONNX 导出的空间输入被写死为 64x64，若像旧实现那样把整张 crop 全局
        下采样到 64 再 2x，会先丢掉已有分辨率、效果常不如双线性。这里改用
        **分块批量推理**绕开 64 的天花板：把 crop 缩放到 grid*64 的方形后切成
        grid*grid 个 64x64 tile，一次 batch 推理（模型 batch 维为动态），再把
        各 128x128 输出拼接成方形并重采样到 target。模型实际"看到"的有效分辨率
        提升到 grid*64。失败/不可用返回 None。
        """
        session, input_name = self._ensure_realesrgan(prefer_gpu)
        if session is None or input_name is None:
            return None
        try:
            TILE = 64
            # 选择网格数，使输出 grid*128 尽量贴近 target（模型放大倍率为 2）
            grid = max(1, int(round(target / (TILE * 2))))
            side_in = grid * TILE

            interp_in = cv2.INTER_AREA if crop.shape[0] > side_in else cv2.INTER_CUBIC
            crop_sq = cv2.resize(crop, (side_in, side_in), interpolation=interp_in)

            tiles = []
            for gy in range(grid):
                for gx in range(grid):
                    tiles.append(crop_sq[gy * TILE:(gy + 1) * TILE, gx * TILE:(gx + 1) * TILE])
            batch = np.stack(tiles, axis=0).astype(np.float32) / 255.0  # (N,64,64,3)
            batch = np.transpose(batch, (0, 3, 1, 2))                   # (N,3,64,64)

            out = session.run(None, {input_name: batch})[0]            # (N,3,h,w)
            out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
            out = np.transpose(out, (0, 2, 3, 1))                      # (N,h,w,3)

            th, tw = out.shape[1], out.shape[2]
            canvas = np.empty((grid * th, grid * tw, 3), dtype=np.uint8)
            k = 0
            for gy in range(grid):
                for gx in range(grid):
                    canvas[gy * th:(gy + 1) * th, gx * tw:(gx + 1) * tw] = out[k]
                    k += 1

            if canvas.shape[0] == target and canvas.shape[1] == target:
                return canvas
            interp_out = cv2.INTER_AREA if canvas.shape[0] > target else cv2.INTER_LINEAR
            return cv2.resize(canvas, (target, target), interpolation=interp_out)
        except Exception as e:
            self._logger.error("[SR] Real-ESRGAN upsampling failed: %s", e)
            return None
