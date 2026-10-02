"""云端健康监控 — 启动 ping + 心跳，连续失败触发降级/恢复回调

保持 Qt 无关：心跳用 daemon 线程 + threading.Event，便于单测。
回调用 subscribe 模式（注册列表），不依赖 Qt 信号。

状态：
  online      — 最近一次探测成功
  degraded    — 连续失败达到阈值（默认 3 次），已触发降级回调
  unconfigured — 未启用或未配置，监控不启动
"""

import logging
import threading

logger = logging.getLogger(__name__)

_FAILURE_THRESHOLD = 3


class CloudHealthMonitor:
    """云端可用性监控。

    Args:
        llm_client: LLMClient 实例
        interval_sec: 心跳间隔秒数（默认 60，取 cloud.health_check_interval_sec）
        failure_threshold: 连续失败多少次触发降级（默认 3）
    """

    STATUS_ONLINE = "online"
    STATUS_DEGRADED = "degraded"
    STATUS_UNCONFIGURED = "unconfigured"

    def __init__(self, llm_client, interval_sec=60, failure_threshold=_FAILURE_THRESHOLD):
        self._llm_client = llm_client
        self._interval_sec = max(1, int(interval_sec))
        self._failure_threshold = max(1, int(failure_threshold))
        self._consecutive_failures = 0
        self._status = (
            self.STATUS_UNCONFIGURED
            if not llm_client.is_configured()
            else self.STATUS_ONLINE
        )
        self._degraded_callbacks = []
        self._recovered_callbacks = []
        self._stop_event = threading.Event()
        self._thread = None
        self._lock = threading.Lock()

    @property
    def status(self):
        return self._status

    def subscribe_degraded(self, callback):
        """注册降级回调：callable()，连续失败达到阈值时触发一次。"""
        self._degraded_callbacks.append(callback)

    def subscribe_recovered(self, callback):
        """注册恢复回调：callable()，降级后探测再次成功时触发一次。"""
        self._recovered_callbacks.append(callback)

    def start(self):
        """启动 ping + 后台心跳。未配置时保持 unconfigured 并返回 False。"""
        if not self._llm_client.is_configured():
            self._status = self.STATUS_UNCONFIGURED
            logger.info("云端未配置，健康监控不启动")
            return False
        self.check_once()  # 启动 ping
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._heartbeat_loop, name="CloudHealthMonitor", daemon=True
        )
        self._thread.start()
        return True

    def stop(self, timeout_sec=3.0):
        """停止心跳线程。"""
        self._stop_event.set()
        thread = self._thread
        self._thread = None
        if thread is not None and thread.is_alive():
            thread.join(max(0.0, float(timeout_sec)))
        return thread is None or not thread.is_alive()

    def check_once(self):
        """执行一次探测（1-token chat 请求），更新状态并触发回调。"""
        try:
            self._llm_client.call_fast(
                [{"role": "user", "content": "ping"}], max_tokens=1
            )
        except Exception as e:
            logger.warning("云端健康探测失败: %s", e)
            self._on_failure()
            return False
        self._on_success()
        return True

    def _heartbeat_loop(self):
        while not self._stop_event.wait(self._interval_sec):
            self.check_once()

    def _on_success(self):
        with self._lock:
            self._consecutive_failures = 0
            was_degraded = self._status == self.STATUS_DEGRADED
            self._status = self.STATUS_ONLINE
        if was_degraded:
            logger.info("云端已恢复")
            self._notify(self._recovered_callbacks)

    def _on_failure(self):
        with self._lock:
            self._consecutive_failures += 1
            if (
                self._status != self.STATUS_DEGRADED
                and self._consecutive_failures >= self._failure_threshold
            ):
                self._status = self.STATUS_DEGRADED
                should_notify = True
            else:
                should_notify = False
        if should_notify:
            logger.warning("云端连续失败 %d 次，进入降级", self._consecutive_failures)
            self._notify(self._degraded_callbacks)

    @staticmethod
    def _notify(callbacks):
        for callback in callbacks:
            try:
                callback()
            except Exception:
                logger.exception("云端健康回调异常")
