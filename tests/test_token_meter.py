"""TokenMeter 计量测试：jsonl 追加写 + daily_summary 聚合。"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'app'))

from services.cloud.token_meter import TokenMeter


def test_record_appends_jsonl(tmp_path):
    meter = TokenMeter(log_dir=str(tmp_path))
    meter.record("deepseek-v4.1-flash", 10, 5, 123.4, True)
    meter.record("DeepSeek-V3.2", 20, 30, 456.7, True)
    meter.record("deepseek-v4.1-flash", 0, 0, 50.0, False, error="http_429")

    usage_file = tmp_path / "token_usage.jsonl"
    assert usage_file.is_file()
    lines = usage_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3

    first = json.loads(lines[0])
    assert first["model"] == "deepseek-v4.1-flash"
    assert first["prompt_tokens"] == 10
    assert first["completion_tokens"] == 5
    assert first["total_tokens"] == 15
    assert first["latency_ms"] == 123.4
    assert first["success"] is True
    assert first["error"] is None
    assert "T" in first["timestamp"]  # ISO 格式

    third = json.loads(lines[2])
    assert third["success"] is False
    assert third["error"] == "http_429"
    assert third["total_tokens"] == 0


def test_daily_summary_aggregates(tmp_path):
    meter = TokenMeter(log_dir=str(tmp_path))
    meter.record("deepseek-v4.1-flash", 10, 5, 100, True)
    meter.record("deepseek-v4.1-flash", 20, 10, 100, True)
    meter.record("DeepSeek-V3.2", 100, 200, 100, True)

    summary = meter.daily_summary()
    assert len(summary) == 1  # 同一天
    day = next(iter(summary.values()))
    assert day["calls"] == 3
    assert day["tokens"] == 345
    assert day["by_model"]["deepseek-v4.1-flash"] == {"calls": 2, "tokens": 45}
    assert day["by_model"]["DeepSeek-V3.2"] == {"calls": 1, "tokens": 300}


def test_daily_summary_empty(tmp_path):
    meter = TokenMeter(log_dir=str(tmp_path))
    assert meter.daily_summary() == {}


def test_daily_summary_skips_corrupt_lines(tmp_path):
    meter = TokenMeter(log_dir=str(tmp_path))
    meter.record("m", 1, 2, 10, True)
    with open(meter.usage_file, "a", encoding="utf-8") as f:
        f.write("not-json\n\n")
    summary = meter.daily_summary()
    day = next(iter(summary.values()))
    assert day["calls"] == 1
    assert day["tokens"] == 3
