"""CloudHealthMonitor 测试：mock LLMClient，验证降级/恢复回调与状态机。"""
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'app'))

from services.cloud.cloud_health import CloudHealthMonitor
from services.cloud.moma_client import CloudAPIError


def _make_llm_client(configured=True):
    client = mock.MagicMock()
    client.is_configured.return_value = configured
    return client


class TestCloudHealth(unittest.TestCase):
    def test_unconfigured_status(self):
        monitor = CloudHealthMonitor(_make_llm_client(configured=False))
        self.assertEqual(monitor.status, "unconfigured")
        self.assertFalse(monitor.start())

    def test_successful_ping_online(self):
        llm = _make_llm_client()
        monitor = CloudHealthMonitor(llm)
        self.assertTrue(monitor.check_once())
        self.assertEqual(monitor.status, "online")

    def test_degraded_after_three_failures(self):
        """连续失败 3 次触发降级回调，且只触发一次。"""
        llm = _make_llm_client()
        llm.call_fast.side_effect = CloudAPIError("boom")
        monitor = CloudHealthMonitor(llm)
        degraded = mock.MagicMock()
        recovered = mock.MagicMock()
        monitor.subscribe_degraded(degraded)
        monitor.subscribe_recovered(recovered)

        for _ in range(2):
            self.assertFalse(monitor.check_once())
            self.assertEqual(monitor.status, "online")
            degraded.assert_not_called()

        self.assertFalse(monitor.check_once())
        self.assertEqual(monitor.status, "degraded")
        degraded.assert_called_once()

        # 继续失败不重复触发降级回调
        monitor.check_once()
        degraded.assert_called_once()
        recovered.assert_not_called()

    def test_recovery_triggers_callback(self):
        """降级后探测成功 → 触发恢复回调，状态回 online。"""
        llm = _make_llm_client()
        llm.call_fast.side_effect = CloudAPIError("boom")
        monitor = CloudHealthMonitor(llm)
        degraded = mock.MagicMock()
        recovered = mock.MagicMock()
        monitor.subscribe_degraded(degraded)
        monitor.subscribe_recovered(recovered)

        for _ in range(3):
            monitor.check_once()
        self.assertEqual(monitor.status, "degraded")

        llm.call_fast.side_effect = None
        self.assertTrue(monitor.check_once())
        self.assertEqual(monitor.status, "online")
        recovered.assert_called_once()

        # 恢复后不重复触发
        monitor.check_once()
        recovered.assert_called_once()

    def test_failure_count_resets_on_success(self):
        """中途成功会重置连续失败计数。"""
        llm = _make_llm_client()
        llm.call_fast.side_effect = CloudAPIError("boom")
        monitor = CloudHealthMonitor(llm)
        degraded = mock.MagicMock()
        monitor.subscribe_degraded(degraded)

        monitor.check_once()
        monitor.check_once()
        llm.call_fast.side_effect = None
        monitor.check_once()
        llm.call_fast.side_effect = CloudAPIError("boom")
        monitor.check_once()
        monitor.check_once()
        degraded.assert_not_called()
        self.assertEqual(monitor.status, "online")

    def test_start_and_stop_heartbeat(self):
        """start 做启动 ping 并启动心跳线程，stop 正常退出。"""
        llm = _make_llm_client()
        monitor = CloudHealthMonitor(llm, interval_sec=3600)
        self.assertTrue(monitor.start())
        llm.call_fast.assert_called_once()
        self.assertTrue(monitor.stop())


if __name__ == "__main__":
    unittest.main(verbosity=2)
