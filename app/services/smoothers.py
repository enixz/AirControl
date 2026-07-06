"""关键点平滑器与几何约束 — 引擎无关的纯算法层。

从 base_hand_tracker.py 拆出，降低上帝类体积。包含：
  - KalmanSmoother: 卡尔曼 + EMA 双重平滑
  - OneEuroFilter / OneEuroSmoother: 自适应低通滤波（含置信度加权）
  - GeometricConstraintFilter: 骨长突变 + z 深度遮挡的后处理约束
  - _pack_landmarks: 平滑后 (x,y) 与原始 z 打包回标准格式
  - _BONE_CONNECTIONS: MediaPipe 21 关键点骨骼连接表

base_hand_tracker.py 仍 re-export 这些名字，保持 `from services.base_hand_tracker
import KalmanSmoother` 等历史 import 路径可用。
"""

import math
import time

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# KalmanSmoother — 卡尔曼滤波 + EMA 双重平滑器
# ---------------------------------------------------------------------------

class KalmanSmoother:
    """为 21 个关键点各自维护 [x,y,vx,vy] 状态的卡尔曼 + EMA 平滑器。"""

    def __init__(
        self,
        num_keypoints=21,
        process_noise=0.5,
        measurement_noise=3.0,
        ema_alpha=0.4,
        max_lost_frames=8,
    ):
        self.num_kp = num_keypoints
        self.ema_alpha = ema_alpha
        self.max_lost_frames = max_lost_frames

        self.filters = []
        for _ in range(num_keypoints):
            kf = cv2.KalmanFilter(4, 2)
            kf.measurementMatrix = np.eye(2, 4, dtype=np.float32)
            kf.transitionMatrix = np.array(
                [[1, 0, 1, 0],
                 [0, 1, 0, 1],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]], dtype=np.float32,
            )
            kf.processNoiseCov = np.array(
                [[0.25, 0, 0.5, 0],
                 [0, 0.25, 0, 0.5],
                 [0.5, 0, 1, 0],
                 [0, 0.5, 0, 1]], dtype=np.float32,
            ) * process_noise
            kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise
            kf.errorCovPost = np.eye(4, dtype=np.float32) * 10.0
            self.filters.append(kf)

        self.ema = None
        self.lost_frames = 0
        self.initialized = False

    def update(self, landmarks):
        raw = np.array([[lm[1], lm[2]] for lm in landmarks], dtype=np.float32)

        if not self.initialized:
            self.ema = raw.copy()
            for i, kf in enumerate(self.filters):
                kf.statePost = np.array(
                    [[raw[i, 0]], [raw[i, 1]], [0], [0]], dtype=np.float32,
                )
            self.initialized = True
        else:
            self.ema = self.ema_alpha * raw + (1 - self.ema_alpha) * self.ema
            for i, kf in enumerate(self.filters):
                kf.predict()
                kf.correct(self.ema[i].reshape(2, 1))

        self.lost_frames = 0
        return _pack_landmarks(landmarks, self.ema)

    def predict(self):
        if not self.initialized or self.lost_frames >= self.max_lost_frames:
            return None
        self.lost_frames += 1
        pred = np.zeros((self.num_kp, 2), dtype=np.float32)
        for i, kf in enumerate(self.filters):
            p = kf.predict()
            pred[i] = [p[0, 0], p[1, 0]]
        return [
            [i, float(pred[i, 0]), float(pred[i, 1])]
            for i in range(self.num_kp)
        ]

    def reset(self):
        self.ema = None
        self.lost_frames = 0
        self.initialized = False


def _pack_landmarks(landmarks, smoothed_xy):
    """把平滑后的 (x,y) 与原始 landmark 的额外维度（z 等）打包回标准格式。

    平滑器只对 x/y 做低通滤波；z（深度）按原值透传，留给下游几何约束做
    遮挡判定。输入只有 [idx,x,y] 时输出仍是 3 元组，保持向后兼容。
    """
    packed = []
    for i, lm in enumerate(landmarks):
        base = [lm[0], float(smoothed_xy[i, 0]), float(smoothed_xy[i, 1])]
        if len(lm) > 3:
            base.extend(float(v) for v in lm[3:])
        packed.append(base)
    return packed


# ---------------------------------------------------------------------------
# OneEuroFilter & OneEuroSmoother — 自适应低通平滑滤波器
# ---------------------------------------------------------------------------

class OneEuroFilter:
    """一欧元自适应低通滤波器。

    能够根据信号的变化速度自动调整截止频率：
      - 慢速时降低截止频率以消除抖动；
      - 快速时提高截止频率以消除延迟。
    """
    def __init__(self, t0, x0, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = float(x0)
        self.x_raw_prev = float(x0)
        self.dx_prev = 0.0
        self.t_prev = float(t0)

    def __call__(self, t, x):
        t = float(t)
        x = float(x)
        dt = t - self.t_prev
        if dt <= 0.0:
            return self.x_prev

        # 1. 计算一阶导数（速度）并应用低通滤波
        # Derivative must use consecutive raw samples. Using the previously
        # filtered value feeds filter lag back into the speed estimate and can
        # make a nearly stationary fingertip look faster than it really is.
        dx = (x - self.x_raw_prev) / dt
        r_d = 2.0 * math.pi * self.d_cutoff * dt
        a_d = r_d / (r_d + 1.0)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev

        # 2. 根据运动速度自适应计算截止频率
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)

        # 3. 对位置进行低通滤波
        r_x = 2.0 * math.pi * cutoff * dt
        a_x = r_x / (r_x + 1.0)
        x_hat = a_x * x + (1.0 - a_x) * self.x_prev

        # 4. 保存状态
        self.x_prev = x_hat
        self.x_raw_prev = x
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat


class OneEuroSmoother:
    """为手部 21 个关键点各自维护 X 和 Y 轴一欧元滤波器的平滑器。

    接口设计与原 KalmanSmoother 完全一致，实现无缝替换。

    置信度加权：为每个关键点维护帧间位移 EMA。位移大的点（噪声/遮挡）
    自动降低 min_cutoff（更强平滑），位移小的点保持原参数。
    """
    def __init__(
        self,
        num_keypoints=21,
        min_cutoff=1.5,     # 手部微抖动截止频率（静态时）
        beta=0.01,          # 速度响应系数（运动时防延迟）
        d_cutoff=1.0,       # 速度低通滤波截止频率
        max_lost_frames=8,
        # 置信度加权参数
        jitter_ema_alpha=0.3,    # 帧间位移 EMA 平滑系数
        jitter_penalty=0.8,      # 位移越大，min_cutoff 最多降到原值的 (1-penalty)
        jitter_threshold=3.0,    # 像素，超过此位移开始施加惩罚
    ):
        self.num_kp = num_keypoints
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.max_lost_frames = max_lost_frames

        # 置信度加权：每个关键点的帧间位移 EMA
        self._jitter_alpha = float(jitter_ema_alpha)
        self._jitter_penalty = float(jitter_penalty)
        self._jitter_threshold = float(jitter_threshold)
        self._jitter_ema = np.zeros(num_keypoints, dtype=np.float32)  # 每点的位移 EMA
        self._prev_raw = None  # 上一帧原始位置，用于计算位移

        self.filters_x = []
        self.filters_y = []
        self.initialized = False
        self.lost_frames = 0
        self.last_landmarks = None

    def _compute_effective_cutoff(self, kp_idx):
        """根据关键点的帧间位移 EMA 计算有效 min_cutoff。

        位移小（稳定）→ 保持原 min_cutoff
        位移大（噪声/遮挡）→ 降低 min_cutoff，增强平滑
        """
        jitter = self._jitter_ema[kp_idx]
        if jitter <= self._jitter_threshold:
            return self.min_cutoff
        # 线性衰减：位移越大，cutoff 越低，最低降到 min_cutoff * (1 - penalty)
        excess = jitter - self._jitter_threshold
        # 用对数衰减避免 cutoff 降到 0
        decay = 1.0 / (1.0 + excess * 0.1)
        min_factor = 1.0 - self._jitter_penalty
        factor = max(min_factor, decay)
        return self.min_cutoff * factor

    def _update_jitter(self, raw):
        """更新每个关键点的帧间位移 EMA。"""
        if self._prev_raw is not None:
            disp = np.sqrt(np.sum((raw - self._prev_raw) ** 2, axis=1))
            self._jitter_ema = (
                self._jitter_alpha * disp
                + (1.0 - self._jitter_alpha) * self._jitter_ema
            )
        self._prev_raw = raw.copy()

    def update(self, landmarks):
        t = time.perf_counter()
        raw = np.array([[lm[1], lm[2]] for lm in landmarks], dtype=np.float32)

        # 更新帧间位移 EMA（置信度估计）
        self._update_jitter(raw)

        if not self.initialized:
            self.filters_x = [
                OneEuroFilter(t, raw[i, 0], min_cutoff=self.min_cutoff, beta=self.beta, d_cutoff=self.d_cutoff)
                for i in range(self.num_kp)
            ]
            self.filters_y = [
                OneEuroFilter(t, raw[i, 1], min_cutoff=self.min_cutoff, beta=self.beta, d_cutoff=self.d_cutoff)
                for i in range(self.num_kp)
            ]
            self.initialized = True
            smoothed = raw.copy()
        else:
            smoothed = np.zeros((self.num_kp, 2), dtype=np.float32)
            for i in range(self.num_kp):
                # 置信度加权：动态调整该点的 min_cutoff
                eff_cutoff = self._compute_effective_cutoff(i)
                self.filters_x[i].min_cutoff = eff_cutoff
                self.filters_y[i].min_cutoff = eff_cutoff
                smoothed[i, 0] = self.filters_x[i](t, raw[i, 0])
                smoothed[i, 1] = self.filters_y[i](t, raw[i, 1])

        self.lost_frames = 0
        self.last_landmarks = _pack_landmarks(landmarks, smoothed)
        return self.last_landmarks

    def predict(self):
        if not self.initialized or self.lost_frames >= self.max_lost_frames:
            return None
        self.lost_frames += 1
        return self.last_landmarks

    def reset(self):
        self.initialized = False
        self.lost_frames = 0
        self.last_landmarks = None
        self.filters_x.clear()
        self.filters_y.clear()
        self._jitter_ema.fill(0.0)
        self._prev_raw = None

    def get_jitter_stats(self):
        """返回各关键点的抖动 EMA，供外部（如 temporal_voter）使用。"""
        return self._jitter_ema.copy()


# ---------------------------------------------------------------------------
# GeometricConstraintFilter — 几何约束后处理
# ---------------------------------------------------------------------------

# MediaPipe 21 关键点的骨骼连接（相邻关节对）
_BONE_CONNECTIONS = [
    # 拇指
    (1, 2), (2, 3), (3, 4),
    # 食指
    (5, 6), (6, 7), (7, 8),
    # 中指
    (9, 10), (10, 11), (11, 12),
    # 无名指
    (13, 14), (14, 15), (15, 16),
    # 小指
    (17, 18), (18, 19), (19, 20),
    # 手掌
    (0, 1), (0, 5), (0, 17), (5, 9), (9, 13), (13, 17),
]


class GeometricConstraintFilter:
    """对平滑后的关键点应用几何约束，抑制"手指乱飞"。

    核心原理：手部骨骼长度在帧间不应突变。如果某根骨骼长度相对上一帧
    变化超过 max_bone_length_change（默认 50%），判定为关键点异常跳变，
    用上一帧的位置替代异常端点。

    此外利用 MediaPipe 的 z（深度，相对手腕、按图像宽归一）做遮挡判定：
    整只手在深度上的平移会让所有点 z 同步漂移；当单个关键点的 z 偏离
    "整体深度漂移"超过 z_occlusion_threshold 时，视作该点被遮挡/翻面，
    回退到上一帧位置。这对指尖在掌心后方的遮挡抖动特别有效。

    远距离下"小指/无名指瞬间跳到错误位置"也由骨长突变检测覆盖。
    """

    def __init__(
        self,
        max_bone_length_change=0.5,  # 骨骼长度最大允许变化比例
        max_correction_per_frame=5,  # 每帧最多修正几个关键点（避免过度修正）
        z_occlusion_threshold=0.06,  # 单点 z 偏离整体漂移的最大允许值（归一化）
    ):
        self._max_change = float(max_bone_length_change)
        self._max_corrections = int(max_correction_per_frame)
        self._z_threshold = float(z_occlusion_threshold)
        self._prev_bone_lengths = None  # 上一帧各骨骼长度
        self._prev_landmarks = None     # 上一帧完整关键点（含 z，用于回退）
        self._prev_z = None             # 上一帧 z 数组（用于漂移对比）
        self._has_z = False             # 输入是否带 z

    def apply(self, landmarks):
        """对关键点应用几何约束，返回修正后的关键点。

        Args:
            landmarks: [[idx, x, y, (z)], ...] 21 个关键点

        Returns:
            修正后的 landmarks，格式与输入一致（带 z 时保留 z）
        """
        if len(landmarks) != 21:
            return landmarks

        pts = np.array([[lm[1], lm[2]] for lm in landmarks], dtype=np.float32)
        has_z = any(len(lm) > 3 for lm in landmarks)
        z_curr = None
        if has_z:
            z_curr = np.array([float(lm[3]) if len(lm) > 3 else 0.0 for lm in landmarks], dtype=np.float32)

        # 首帧：记录骨骼长度与 z，不做修正
        if self._prev_bone_lengths is None or self._prev_landmarks is None:
            self._prev_bone_lengths = self._compute_bone_lengths(pts)
            self._prev_landmarks = [list(lm) for lm in landmarks]
            if has_z:
                self._prev_z = z_curr.copy()
                self._has_z = True
            return landmarks

        # 计算当前帧骨骼长度
        curr_lengths = self._compute_bone_lengths(pts)

        # 找出长度突变超过阈值的骨骼
        violations = []
        for i, (prev_len, curr_len) in enumerate(zip(self._prev_bone_lengths, curr_lengths, strict=True)):
            if prev_len < 1e-6:
                continue
            change_ratio = abs(curr_len - prev_len) / prev_len
            if change_ratio > self._max_change:
                violations.append((i, change_ratio))

        # 统计每个关键点被多少根异常骨骼涉及
        violation_count = np.zeros(21, dtype=np.int32)
        for bone_idx, _ in violations:
            a, b = _BONE_CONNECTIONS[bone_idx]
            violation_count[a] += 1
            violation_count[b] += 1

        # z 遮挡判定：单点 z 偏离整体深度漂移 → 该点被遮挡/翻面，计入候选
        z_outliers = np.zeros(21, dtype=bool)
        if has_z and self._has_z and self._prev_z is not None:
            z_delta = z_curr - self._prev_z
            # 整只手的深度漂移用中位数估计（对单点异常鲁棒）
            median_delta = float(np.median(z_delta))
            z_dev = np.abs(z_delta - median_delta)
            z_outliers = z_dev > self._z_threshold
            for kp_idx in range(21):
                if z_outliers[kp_idx]:
                    violation_count[kp_idx] += 1

        if not violations and not z_outliers.any():
            self._prev_bone_lengths = curr_lengths
            self._prev_landmarks = [list(lm) for lm in landmarks]
            if has_z:
                self._prev_z = z_curr.copy()
            return landmarks

        # 修正涉及异常最多的关键点（最多修正 max_corrections 个）
        candidates = np.argsort(violation_count)[::-1]
        corrected_xy = pts.copy()
        corrected_z = z_curr.copy() if has_z else None
        num_corrected = 0
        for kp_idx in candidates:
            if violation_count[kp_idx] == 0 or num_corrected >= self._max_corrections:
                break
            prev_lm = self._prev_landmarks[kp_idx]
            corrected_xy[kp_idx, 0] = float(prev_lm[1])
            corrected_xy[kp_idx, 1] = float(prev_lm[2])
            if has_z and len(prev_lm) > 3:
                corrected_z[kp_idx] = float(prev_lm[3])
            num_corrected += 1

        # 更新状态
        self._prev_bone_lengths = self._compute_bone_lengths(corrected_xy)
        self._prev_landmarks = []
        for i, lm in enumerate(landmarks):
            entry = [lm[0], float(corrected_xy[i, 0]), float(corrected_xy[i, 1])]
            if has_z:
                entry.append(float(corrected_z[i]))
            self._prev_landmarks.append(entry)
        if has_z:
            self._prev_z = corrected_z.copy()
            self._has_z = True

        # 返回修正后的 landmarks（保持原格式，带 z 时保留 z）
        result = []
        for i, lm in enumerate(landmarks):
            entry = [lm[0], float(corrected_xy[i, 0]), float(corrected_xy[i, 1])]
            if has_z:
                entry.append(float(corrected_z[i]))
            result.append(entry)
        return result

    @staticmethod
    def _compute_bone_lengths(pts):
        """计算所有骨骼的长度。"""
        lengths = np.zeros(len(_BONE_CONNECTIONS), dtype=np.float32)
        for i, (a, b) in enumerate(_BONE_CONNECTIONS):
            lengths[i] = np.linalg.norm(pts[a] - pts[b])
        return lengths

    def reset(self):
        self._prev_bone_lengths = None
        self._prev_landmarks = None
        self._prev_z = None
        self._has_z = False
