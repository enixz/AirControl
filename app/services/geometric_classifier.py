"""几何特征提取器 + 加权投票手势分类器 — 纯 CPU 算法，无需训练数据。

核心思路：
  1. 从 21 关键点提取 30+ 维连续特征（手指弯曲度、指间角度、掌心比例）
  2. 每个手势类别用加权置信度评分（0-1），替代硬阈值布尔判定
  3. 融合 MediaPipe ML 标签和几何置信度，输出最终手势

为什么能优于 MediaPipe 原生分类器：
  - MediaPipe 输出离散标签，远距离下关键点抖动导致标签逐帧跳变
  - 本分类器输出连续置信度，下游可设阈值滞回，抗抖动
  - 几何特征对手部偏航/距离不变（用比例而非绝对值），远距离更稳定
"""

import logging

import numpy as np

logger = logging.getLogger('gesture')


# MediaPipe 21 关键点索引
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

# 每根手指的关节链（MCP → PIP → DIP → TIP）
FINGER_CHAINS = {
    "thumb":  [THUMB_MCP, THUMB_IP, THUMB_TIP],
    "index":  [INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP],
    "middle": [MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP],
    "ring":   [RING_MCP, RING_PIP, RING_DIP, RING_TIP],
    "pinky":  [PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP],
}


class GeometricFeatureExtractor:
    """从 21 关键点提取距离/比例不变的连续特征向量。

    所有特征都用比例（相对掌宽）或角度，确保对手部距离、偏航不变。
    """

    def __init__(self):
        self._feature_names = None  # 延迟初始化

    def extract(self, landmarks):
        """提取特征向量。

        Args:
            landmarks: [[idx, x, y], ...] 21 个关键点

        Returns:
            dict: 特征名 → 连续值（大部分在 [0, 1] 或弧度）
        """
        pts = np.array([[lm[1], lm[2]] for lm in landmarks], dtype=np.float32)

        # 掌宽（5↔17），作为归一化基准
        palm_width = max(20.0, float(np.linalg.norm(pts[INDEX_MCP] - pts[PINKY_MCP])))

        # 手掌中心（0, 5, 9, 13, 17 的质心）
        palm_center = pts[[WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]].mean(axis=0)

        features = {}

        # === 1. 每根手指的伸出程度（0=完全弯曲, 1=完全伸出）===
        for name, chain in FINGER_CHAINS.items():
            if name == "thumb":
                # 拇指：用 tip 到 wrist 的距离 / 掌宽
                tip_dist = float(np.linalg.norm(pts[THUMB_TIP] - pts[WRIST]))
                features[f"{name}_extension"] = tip_dist / palm_width
            else:
                # 其他手指：tip 到 palm_center 的距离 / 掌宽
                mcp, _, _, tip = chain
                tip_to_palm = float(np.linalg.norm(pts[tip] - palm_center))
                mcp_to_palm = float(np.linalg.norm(pts[mcp] - palm_center))
                # extension = (tip_to_palm - mcp_to_palm) / palm_width
                # 弯曲时 tip 靠近 palm，伸出时远离
                features[f"{name}_extension"] = (tip_to_palm - mcp_to_palm) / palm_width

        # === 2. 每根手指的弯曲度（MCP-PIP-TIP 角度，0=直, 1=完全弯曲）===
        for name, chain in FINGER_CHAINS.items():
            if len(chain) < 3:
                continue
            # 取 MCP → PIP → TIP 的角度
            angle = self._joint_angle(pts[chain[0]], pts[chain[1]], pts[chain[-1]])
            # 180度=直, 0度=完全弯曲 → 归一化到 [0, 1]
            features[f"{name}_curl"] = 1.0 - (angle / 180.0)

        # === 3. 手指间张开度（相邻指尖距离 / 掌宽）===
        finger_tips = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
        for i in range(len(finger_tips) - 1):
            t1, t2 = finger_tips[i], finger_tips[i + 1]
            spread = float(np.linalg.norm(pts[t1] - pts[t2])) / palm_width
            features[f"spread_{i}_{i+1}"] = spread

        # === 4. 拇指特殊特征 ===
        # 拇指 ↔ 食指 MCP 距离（拇指内收/外展）
        thumb_to_index_mcp = float(np.linalg.norm(pts[THUMB_TIP] - pts[INDEX_MCP])) / palm_width
        features["thumb_to_index_mcp"] = thumb_to_index_mcp

        # 拇指 ↔ 食指尖距离（捏合）
        thumb_index_pinch = float(np.linalg.norm(pts[THUMB_TIP] - pts[INDEX_TIP])) / palm_width
        features["thumb_index_pinch"] = thumb_index_pinch

        # 拇指方向（上/下）：tip 相对于 IP 的 y 差 / 掌宽
        # 负值=拇指朝上，正值=拇指朝下
        features["thumb_vertical"] = float(pts[THUMB_TIP][1] - pts[THUMB_IP][1]) / palm_width

        # === 5. 手掌朝向（正面度）===
        # 掌宽 / 食指长：正对相机 ≈0.8，侧对 ≈0.4
        index_len = float(np.linalg.norm(pts[INDEX_TIP] - pts[INDEX_MCP]))
        features["hand_frontality"] = palm_width / max(index_len, 1e-6)

        # === 6. 整体手型紧凑度 ===
        # 所有关键点到 palm_center 的平均距离 / 掌宽
        all_dists = np.linalg.norm(pts - palm_center, axis=1)
        features["compactness"] = float(np.mean(all_dists)) / palm_width

        # === 7. 手指一致性（伸出手指数量，连续值）===
        # 用 extension > 0.3 作为"伸出"判据，统计伸出手指数
        extended_count = sum(
            1 for name in ["index", "middle", "ring", "pinky"]
            if features.get(f"{name}_extension", 0) > 0.3
        )
        features["extended_finger_count"] = float(extended_count)

        # === 8. 手指相对长度（各指长 / 掌宽，距离不变）===
        for name, chain in FINGER_CHAINS.items():
            if name == "thumb":
                finger_len = float(np.linalg.norm(pts[THUMB_TIP] - pts[THUMB_MCP]))
            else:
                mcp, _, _, tip = chain
                finger_len = float(np.linalg.norm(pts[tip] - pts[mcp]))
            features[f"{name}_length"] = finger_len / palm_width

        # === 9. 手掌面积比例（掌心四边形面积 / 掌宽²）===
        palm_quad = np.array([pts[INDEX_MCP], pts[MIDDLE_MCP], pts[RING_MCP], pts[PINKY_MCP]])
        x = palm_quad[:, 0]
        y = palm_quad[:, 1]
        palm_area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        features["palm_area_ratio"] = palm_area / (palm_width ** 2)

        # === 10. 手腕到各 MCP 的距离（手掌骨架比例）===
        for name, mcp_idx in [("index", INDEX_MCP), ("middle", MIDDLE_MCP),
                               ("ring", RING_MCP), ("pinky", PINKY_MCP)]:
            dist = float(np.linalg.norm(pts[mcp_idx] - pts[WRIST]))
            features[f"wrist_to_{name}_mcp"] = dist / palm_width

        # === 11. 食指方向角（相对垂直方向，用于指向判定）===
        index_dir = pts[INDEX_TIP] - pts[INDEX_MCP]
        vertical = np.array([0.0, -1.0], dtype=np.float32)
        cos_dir = np.dot(index_dir, vertical) / (np.linalg.norm(index_dir) + 1e-8)
        features["index_vertical_angle"] = float(np.degrees(np.arccos(np.clip(cos_dir, -1, 1))))

        if self._feature_names is None:
            self._feature_names = list(features.keys())

        return features

    @staticmethod
    def _joint_angle(a, b, c):
        """计算三点 a-b-c 在 b 处的关节角度（度）。"""
        ba = a - b
        bc = c - b
        cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_angle)))

    @property
    def feature_names(self):
        return self._feature_names or []


class WeightedVoteClassifier:
    """加权投票手势分类器。

    每个手势类别用一组特征规则计算置信度（0-1），最终取置信度最高的类别。
    与 MediaPipe ML 标签融合：ML 标签作为先验，几何置信度作为证据。

    不需要训练数据：规则基于手部解剖学先验知识。
    """

    def __init__(self, ml_weight=0.4, geo_weight=0.6):
        """
        Args:
            ml_weight: MediaPipe ML 标签的融合权重
            geo_weight: 几何特征的融合权重
        """
        self._ml_weight = float(ml_weight)
        self._geo_weight = float(geo_weight)
        self._extractor = GeometricFeatureExtractor()

    def classify(self, landmarks, ml_label="None", ml_score=0.0):
        """对手部关键点进行手势分类。

        Args:
            landmarks: [[idx, x, y], ...] 21 个关键点
            ml_label: MediaPipe 的手势标签（如 "Closed_Fist"）
            ml_score: MediaPipe 的置信度 [0, 1]

        Returns:
            dict: {
                "label": 最终手势标签（内部名，如 "FIST"）,
                "confidence": 综合置信度 [0, 1],
                "geo_scores": 各手势类别的几何置信度,
                "features": 提取的特征向量,
            }
        """
        features = self._extractor.extract(landmarks)
        geo_scores = self._compute_geo_scores(features)

        # 融合 ML 标签和几何置信度
        ml_label_internal = ML_LABEL_TO_INTERNAL.get(ml_label, "OTHER")
        fused_scores = {}

        for gesture, geo_score in geo_scores.items():
            ml_component = ml_score if gesture == ml_label_internal else 0.0
            fused_scores[gesture] = (
                self._ml_weight * ml_component
                + self._geo_weight * geo_score
            )

        # 取最高分
        best_gesture = max(fused_scores, key=fused_scores.get)
        best_score = fused_scores[best_gesture]

        return {
            "label": best_gesture,
            "confidence": best_score,
            "geo_scores": geo_scores,
            "fused_scores": fused_scores,
            "features": features,
        }

    def _compute_geo_scores(self, f):
        """根据特征向量计算每个手势类别的几何置信度。

        每个手势返回 [0, 1] 的置信度，基于特征匹配程度。
        """
        scores = {}

        # --- 握拳：所有手指弯曲，拇指内收 ---
        curl_sum = (
            f.get("index_curl", 0) + f.get("middle_curl", 0)
            + f.get("ring_curl", 0) + f.get("pinky_curl", 0)
        ) / 4.0
        thumb_tucked = 1.0 - min(1.0, f.get("thumb_to_index_mcp", 1.0))
        scores["FIST"] = 0.5 * curl_sum + 0.3 * thumb_tucked + 0.2 * (1.0 - f.get("extended_finger_count", 0) / 4.0)

        # --- 张掌：所有手指伸出，拇指伸出 ---
        ext_sum = (
            f.get("index_extension", 0) + f.get("middle_extension", 0)
            + f.get("ring_extension", 0) + f.get("pinky_extension", 0)
        ) / 4.0
        thumb_ext = min(1.0, f.get("thumb_extension", 0) / 1.5)
        scores["OPEN"] = 0.5 * ext_sum + 0.3 * thumb_ext + 0.2 * min(1.0, f.get("extended_finger_count", 0) / 4.0)

        # --- 指向：只有食指伸出 ---
        index_ext = min(1.0, f.get("index_extension", 0) / 0.5)
        others_curled = (
            f.get("middle_curl", 0) + f.get("ring_curl", 0) + f.get("pinky_curl", 0)
        ) / 3.0
        scores["POINTING_UP"] = 0.5 * index_ext + 0.4 * others_curled + 0.1 * (1.0 - abs(f.get("extended_finger_count", 0) - 1.0) / 4.0)

        # --- 点赞：拇指朝上，其他手指弯曲 ---
        thumb_up = max(0.0, -f.get("thumb_vertical", 0))  # 负值=朝上
        thumb_up_norm = min(1.0, thumb_up / 0.5)
        scores["THUMB_UP"] = 0.4 * thumb_up_norm + 0.4 * curl_sum + 0.2 * (1.0 - f.get("thumb_to_index_mcp", 0))

        # --- 倒赞：拇指朝下，其他手指弯曲 ---
        thumb_down = max(0.0, f.get("thumb_vertical", 0))  # 正值=朝下
        thumb_down_norm = min(1.0, thumb_down / 0.5)
        scores["THUMB_DOWN"] = 0.4 * thumb_down_norm + 0.4 * curl_sum + 0.2 * (1.0 - f.get("thumb_to_index_mcp", 0))

        # --- 比耶：食指+中指伸出，无名指+小指弯曲 ---
        index_mid_ext = (
            min(1.0, f.get("index_extension", 0) / 0.5)
            + min(1.0, f.get("middle_extension", 0) / 0.5)
        ) / 2.0
        ring_pinky_curl = (f.get("ring_curl", 0) + f.get("pinky_curl", 0)) / 2.0
        spread_ok = min(1.0, f.get("spread_1_2", 0) / 0.3)  # 食指↔中指张开
        scores["VICTORY"] = 0.4 * index_mid_ext + 0.3 * ring_pinky_curl + 0.3 * spread_ok

        # --- ILY：拇指+食指+小指伸出，中指+无名指弯曲 ---
        ily_ext = (
            min(1.0, f.get("thumb_extension", 0) / 1.5)
            + min(1.0, f.get("index_extension", 0) / 0.5)
            + min(1.0, f.get("pinky_extension", 0) / 0.5)
        ) / 3.0
        mid_ring_curl = (f.get("middle_curl", 0) + f.get("ring_curl", 0)) / 2.0
        scores["I_LOVE_YOU"] = 0.5 * ily_ext + 0.5 * mid_ring_curl

        # --- OTHER：不匹配任何特定手势 ---
        max_specific = max(scores.values())
        scores["OTHER"] = max(0.0, 1.0 - max_specific - 0.2)

        # 归一化到 [0, 1]
        total = sum(scores.values())
        if total > 0:
            for k in scores:
                scores[k] /= total

        return scores


# MediaPipe 标签到内部标签的映射（与 hand_tracker.py 一致）
ML_LABEL_TO_INTERNAL = {
    "Closed_Fist": "FIST",
    "Open_Palm": "OPEN",
    "Pointing_Up": "POINTING_UP",
    "Thumb_Up": "THUMB_UP",
    "Thumb_Down": "THUMB_DOWN",
    "Victory": "VICTORY",
    "ILoveYou": "I_LOVE_YOU",
    "None": "OTHER",
}
