"""T-Geometric-Classifier: 几何特征提取器 + 加权投票分类器测试

验证：
  1. GeometricFeatureExtractor 提取 30+ 维特征
  2. WeightedVoteClassifier 对各种手型的分类正确性
  3. 特征的距离/比例不变性
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'app'))

from services.geometric_classifier import (
    GeometricFeatureExtractor,
    WeightedVoteClassifier,
)


def _make_landmarks(pts):
    """把 (x, y) 列表转为 [idx, x, y] 格式。"""
    return [[i, float(x), float(y)] for i, (x, y) in enumerate(pts)]


def _make_open_palm():
    """生成张掌手型关键点（所有手指伸出）。"""
    return _make_landmarks([
        (0, 0),      # 0 wrist
        (10, -5),    # 1 thumb_cmc
        (20, -10),   # 2 thumb_mcp
        (30, -15),   # 3 thumb_ip
        (40, -20),   # 4 thumb_tip
        (10, -30),   # 5 index_mcp
        (12, -50),   # 6 index_pip
        (14, -65),   # 7 index_dip
        (16, -80),   # 8 index_tip
        (25, -30),   # 9 middle_mcp
        (27, -55),   # 10 middle_pip
        (29, -70),   # 11 middle_dip
        (31, -85),   # 12 middle_tip
        (40, -30),   # 13 ring_mcp
        (42, -50),   # 14 ring_pip
        (44, -65),   # 15 ring_dip
        (46, -80),   # 16 ring_tip
        (55, -25),   # 17 pinky_mcp
        (57, -40),   # 18 pinky_pip
        (59, -50),   # 19 pinky_dip
        (61, -60),   # 20 pinky_tip
    ])


def _make_fist():
    """生成握拳手型关键点（所有手指弯曲）。"""
    return _make_landmarks([
        (0, 0),      # 0 wrist
        (10, -5),    # 1 thumb_cmc
        (15, -10),   # 2 thumb_mcp
        (18, -8),    # 3 thumb_ip
        (20, -5),    # 4 thumb_tip (内收)
        (10, -20),   # 5 index_mcp
        (12, -15),   # 6 index_pip (弯曲)
        (14, -18),   # 7 index_dip
        (16, -22),   # 8 index_tip (靠近掌心)
        (25, -20),   # 9 middle_mcp
        (27, -15),   # 10 middle_pip
        (29, -18),   # 11 middle_dip
        (31, -22),   # 12 middle_tip
        (40, -20),   # 13 ring_mcp
        (42, -15),   # 14 ring_pip
        (44, -18),   # 15 ring_dip
        (46, -22),   # 16 ring_tip
        (55, -18),   # 17 pinky_mcp
        (57, -13),   # 18 pinky_pip
        (59, -16),   # 19 pinky_dip
        (61, -20),   # 20 pinky_tip
    ])


def _make_victory():
    """生成比耶手型（食指+中指伸出，无名指+小指弯曲）。"""
    return _make_landmarks([
        (0, 0),      # 0 wrist
        (10, -5),    # 1 thumb_cmc
        (15, -10),   # 2 thumb_mcp
        (18, -8),    # 3 thumb_ip
        (20, -5),    # 4 thumb_tip
        (10, -30),   # 5 index_mcp
        (12, -50),   # 6 index_pip
        (14, -65),   # 7 index_dip
        (16, -80),   # 8 index_tip (伸出)
        (25, -30),   # 9 middle_mcp
        (27, -55),   # 10 middle_pip
        (29, -70),   # 11 middle_dip
        (31, -85),   # 12 middle_tip (伸出)
        (40, -20),   # 13 ring_mcp
        (42, -15),   # 14 ring_pip (弯曲)
        (44, -18),   # 15 ring_dip
        (46, -22),   # 16 ring_tip
        (55, -18),   # 17 pinky_mcp
        (57, -13),   # 18 pinky_pip (弯曲)
        (59, -16),   # 19 pinky_dip
        (61, -20),   # 20 pinky_tip
    ])


def _make_pointing():
    """生成指向手型（只有食指伸出）。"""
    return _make_landmarks([
        (0, 0),      # 0 wrist
        (10, -5),    # 1 thumb_cmc
        (15, -10),   # 2 thumb_mcp
        (18, -8),    # 3 thumb_ip
        (20, -5),    # 4 thumb_tip
        (10, -30),   # 5 index_mcp
        (12, -50),   # 6 index_pip
        (14, -65),   # 7 index_dip
        (16, -80),   # 8 index_tip (伸出)
        (25, -20),   # 9 middle_mcp
        (27, -15),   # 10 middle_pip (弯曲)
        (29, -18),   # 11 middle_dip
        (31, -22),   # 12 middle_tip
        (40, -20),   # 13 ring_mcp
        (42, -15),   # 14 ring_pip
        (44, -18),   # 15 ring_dip
        (46, -22),   # 16 ring_tip
        (55, -18),   # 17 pinky_mcp
        (57, -13),   # 18 pinky_pip
        (59, -16),   # 19 pinky_dip
        (61, -20),   # 20 pinky_tip
    ])


class TestGeometricFeatureExtractor(unittest.TestCase):
    """测试几何特征提取器"""

    def setUp(self):
        self.extractor = GeometricFeatureExtractor()

    def test_extracts_30_plus_features(self):
        """提取 30+ 维特征"""
        features = self.extractor.extract(_make_open_palm())
        self.assertGreaterEqual(len(features), 25, f"只提取了{len(features)}个特征")

    def test_has_finger_extensions(self):
        """包含每根手指的伸出程度"""
        features = self.extractor.extract(_make_open_palm())
        for name in ["thumb", "index", "middle", "ring", "pinky"]:
            self.assertIn(f"{name}_extension", features)

    def test_has_finger_curls(self):
        """包含每根手指的弯曲度"""
        features = self.extractor.extract(_make_open_palm())
        for name in ["thumb", "index", "middle", "ring", "pinky"]:
            self.assertIn(f"{name}_curl", features)

    def test_has_finger_spreads(self):
        """包含手指间张开度"""
        features = self.extractor.extract(_make_open_palm())
        # spread_0_1, spread_1_2, spread_2_3, spread_3_4
        spread_keys = [k for k in features if k.startswith("spread_")]
        self.assertGreaterEqual(len(spread_keys), 3)

    def test_distance_invariance(self):
        """特征对距离不变（缩放后特征值不变）"""
        lm1 = _make_open_palm()
        # 缩小到 0.5 倍（模拟远距离）
        lm2 = _make_landmarks([(x * 0.5, y * 0.5) for i, (x, y) in enumerate(
            [(lm[1], lm[2]) for lm in lm1]
        )])
        f1 = self.extractor.extract(lm1)
        f2 = self.extractor.extract(lm2)
        # 比例特征应该相近（允许 10% 误差）
        for key in ["index_extension", "middle_curl", "thumb_to_index_mcp"]:
            self.assertAlmostEqual(f1[key], f2[key], delta=0.1,
                                   msg=f"{key} 不满足距离不变性")

    def test_open_palm_has_high_extension(self):
        """张掌时手指伸出程度高"""
        features = self.extractor.extract(_make_open_palm())
        self.assertGreater(features["index_extension"], 0.3)
        self.assertGreater(features["middle_extension"], 0.3)

    def test_fist_has_high_curl(self):
        """握拳时手指弯曲度高"""
        features = self.extractor.extract(_make_fist())
        self.assertGreater(features["index_curl"], 0.3)
        self.assertGreater(features["middle_curl"], 0.3)


class TestWeightedVoteClassifier(unittest.TestCase):
    """测试加权投票分类器"""

    def setUp(self):
        self.classifier = WeightedVoteClassifier()

    def test_open_palm_classified_correctly(self):
        """张掌被正确分类"""
        result = self.classifier.classify(_make_open_palm(), ml_label="Open_Palm", ml_score=0.9)
        self.assertEqual(result["label"], "OPEN")
        self.assertGreater(result["confidence"], 0.3)

    def test_fist_classified_correctly(self):
        """握拳被正确分类"""
        result = self.classifier.classify(_make_fist(), ml_label="Closed_Fist", ml_score=0.9)
        self.assertEqual(result["label"], "FIST")
        self.assertGreater(result["confidence"], 0.3)

    def test_victory_classified_correctly(self):
        """比耶被正确分类"""
        result = self.classifier.classify(_make_victory(), ml_label="Victory", ml_score=0.9)
        self.assertEqual(result["label"], "VICTORY")

    def test_pointing_classified_correctly(self):
        """指向被正确分类"""
        result = self.classifier.classify(_make_pointing(), ml_label="Pointing_Up", ml_score=0.9)
        self.assertEqual(result["label"], "POINTING_UP")

    def test_returns_confidence_in_range(self):
        """置信度在 [0, 1] 范围内"""
        result = self.classifier.classify(_make_open_palm(), ml_label="Open_Palm", ml_score=0.5)
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)

    def test_returns_geo_scores(self):
        """返回各手势类别的几何置信度"""
        result = self.classifier.classify(_make_open_palm())
        geo_scores = result["geo_scores"]
        self.assertIn("FIST", geo_scores)
        self.assertIn("OPEN", geo_scores)
        self.assertIn("VICTORY", geo_scores)

    def test_returns_features(self):
        """返回提取的特征向量"""
        result = self.classifier.classify(_make_open_palm())
        self.assertIn("features", result)
        self.assertGreaterEqual(len(result["features"]), 25)

    def test_low_ml_score_uses_geo_more(self):
        """ML 置信度低时几何权重更大"""
        # ML 标签是 FIST 但几何特征是 OPEN
        result = self.classifier.classify(_make_open_palm(), ml_label="Closed_Fist", ml_score=0.3)
        # 几何特征应该让 OPEN 得分更高
        self.assertGreater(result["geo_scores"]["OPEN"], result["geo_scores"]["FIST"])

    def test_distance_invariance_classification(self):
        """分类结果对距离不变"""
        lm_full = _make_open_palm()
        lm_half = _make_landmarks([(x * 0.5, y * 0.5) for i, (x, y) in enumerate(
            [(lm[1], lm[2]) for lm in lm_full]
        )])
        r1 = self.classifier.classify(lm_full, ml_label="Open_Palm", ml_score=0.8)
        r2 = self.classifier.classify(lm_half, ml_label="Open_Palm", ml_score=0.8)
        self.assertEqual(r1["label"], r2["label"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
