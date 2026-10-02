"""Token 用量计量 — 追加写 logs/token_usage.jsonl，按日聚合

每次云端调用（含失败）记录一行 JSON。日志目录沿用 log_config 的约定：
writable_data_dir() 下（开发态即项目根目录），文件为 logs/token_usage.jsonl。
写入用锁保证线程安全。
"""

import json
import logging
import os
import threading
from datetime import datetime

from runtime_paths import writable_data_dir

logger = logging.getLogger(__name__)

_USAGE_FILE_NAME = "token_usage.jsonl"


class TokenMeter:
    """云端调用计量器。

    Args:
        log_dir: 日志目录；默认 writable_data_dir()/logs
    """

    def __init__(self, log_dir=None):
        self._log_dir = log_dir or os.path.join(writable_data_dir(), "logs")
        self._lock = threading.Lock()

    @property
    def usage_file(self):
        return os.path.join(self._log_dir, _USAGE_FILE_NAME)

    def record(self, model, prompt_tokens, completion_tokens, latency_ms,
               success, error=None):
        """追加一条调用记录（每行一个 JSON 对象）。"""
        entry = {
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "model": model,
            "prompt_tokens": int(prompt_tokens or 0),
            "completion_tokens": int(completion_tokens or 0),
            "total_tokens": int(prompt_tokens or 0) + int(completion_tokens or 0),
            "latency_ms": round(float(latency_ms or 0), 1),
            "success": bool(success),
            "error": error,
        }
        line = json.dumps(entry, ensure_ascii=False)
        try:
            with self._lock:
                os.makedirs(self._log_dir, exist_ok=True)
                with open(self.usage_file, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
        except OSError as e:
            # 计量失败不影响主流程
            logger.warning("token 计量写入失败: %s", e)

    def daily_summary(self):
        """按日聚合：{date: {"calls": n, "tokens": n, "by_model": {model: {...}}}}"""
        summary = {}
        path = self.usage_file
        if not os.path.isfile(path):
            return summary
        try:
            with open(path, encoding="utf-8") as f:
                lines = f.readlines()
        except OSError as e:
            logger.warning("token 计量读取失败: %s", e)
            return summary

        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            date = str(entry.get("timestamp", ""))[:10]
            model = entry.get("model") or "unknown"
            day = summary.setdefault(
                date, {"calls": 0, "tokens": 0, "by_model": {}}
            )
            day["calls"] += 1
            day["tokens"] += int(entry.get("total_tokens") or 0)
            per_model = day["by_model"].setdefault(
                model, {"calls": 0, "tokens": 0}
            )
            per_model["calls"] += 1
            per_model["tokens"] += int(entry.get("total_tokens") or 0)
        return summary
