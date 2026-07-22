"""benchmark_ab.py 新增 CLI 开关（--set/--engines/--out）的单元测试。

只测纯解析逻辑，不跑视频（避免重型 cv2/mediapipe 推理进测试）。
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark_ab import _coerce_value, _parse_overrides


class TestCoerceValue(unittest.TestCase):
    def test_bool(self):
        self.assertIs(_coerce_value("true"), True)
        self.assertIs(_coerce_value("FALSE"), False)

    def test_none(self):
        self.assertIsNone(_coerce_value("none"))
        self.assertIsNone(_coerce_value("null"))

    def test_int_and_float(self):
        self.assertEqual(_coerce_value("60"), 60)
        self.assertIsInstance(_coerce_value("60"), int)
        self.assertEqual(_coerce_value("0.008"), 0.008)
        self.assertIsInstance(_coerce_value("0.008"), float)

    def test_str_fallback(self):
        self.assertEqual(_coerce_value("espcn"), "espcn")


class TestParseOverrides(unittest.TestCase):
    def test_multiple_pairs(self):
        result = _parse_overrides([
            "long_range_enabled=true",
            "zoom_sr_engine=espcn",
            "zoom_far_threshold=0.006",
        ])
        self.assertEqual(result, {
            "long_range_enabled": True,
            "zoom_sr_engine": "espcn",
            "zoom_far_threshold": 0.006,
        })

    def test_empty_input(self):
        self.assertEqual(_parse_overrides([]), {})
        self.assertEqual(_parse_overrides(None), {})

    def test_value_containing_equals(self):
        self.assertEqual(_parse_overrides(["k=a=b"]), {"k": "a=b"})

    def test_invalid_pair_raises(self):
        with self.assertRaises(ValueError):
            _parse_overrides(["no_equals_sign"])
        with self.assertRaises(ValueError):
            _parse_overrides(["=v"])


if __name__ == "__main__":
    unittest.main()
