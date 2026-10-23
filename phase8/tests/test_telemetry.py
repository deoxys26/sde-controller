import unittest
import time
from phase8.orchestrator.telemetry_buffer import TelemetryBuffer

class TestTelemetryBuffer(unittest.TestCase):
    def setUp(self):
        self.buffer = TelemetryBuffer(capacity_bps=100_000_000)
        self.t0 = time.time()
        
    def test_first_sample(self):
        res = self.buffer.process_port_stats(
            dpid=1, port=2, timestamp=self.t0,
            rx_bytes=1000, tx_bytes=2000,
            rx_packets=10, tx_packets=20,
            rx_dropped=0, tx_dropped=0,
            rx_errors=0, tx_errors=0,
            src_node="s1", dst_node="s2"
        )
        self.assertIsNone(res, "First sample should return None (no rate available)")
        
    def test_counter_delta_and_utilization(self):
        self.buffer.process_port_stats(
            dpid=1, port=2, timestamp=self.t0,
            rx_bytes=1000, tx_bytes=1_000_000,
            rx_packets=10, tx_packets=20,
            rx_dropped=0, tx_dropped=0,
            rx_errors=0, tx_errors=0,
            src_node="s1", dst_node="s2"
        )
        
        # 1 second later, 1,000,000 bytes transmitted
        t1 = self.t0 + 1.0
        res = self.buffer.process_port_stats(
            dpid=1, port=2, timestamp=t1,
            rx_bytes=1000, tx_bytes=2_000_000,
            rx_packets=10, tx_packets=40,
            rx_dropped=0, tx_dropped=0,
            rx_errors=0, tx_errors=0,
            src_node="s1", dst_node="s2"
        )
        
        self.assertIsNotNone(res)
        self.assertEqual(res["traffic"], 8_000_000) # 1MB/s * 8 bits
        self.assertAlmostEqual(res["utilization"], 0.08) # 8M / 100M = 0.08
        self.assertEqual(res["src"], "s1")
        self.assertEqual(res["dst"], "s2")
        
    def test_counter_reset(self):
        self.buffer.process_port_stats(
            dpid=1, port=2, timestamp=self.t0,
            rx_bytes=1000, tx_bytes=5000,
            rx_packets=10, tx_packets=20,
            rx_dropped=0, tx_dropped=0,
            rx_errors=0, tx_errors=0,
            src_node="s1", dst_node="s2"
        )
        
        # Counter resets to lower value
        t1 = self.t0 + 1.0
        res = self.buffer.process_port_stats(
            dpid=1, port=2, timestamp=t1,
            rx_bytes=0, tx_bytes=1000,
            rx_packets=0, tx_packets=5,
            rx_dropped=0, tx_dropped=0,
            rx_errors=0, tx_errors=0,
            src_node="s1", dst_node="s2"
        )
        
        self.assertIsNone(res, "Should return None on counter reset")
        
        # Next sample should work
        t2 = t1 + 1.0
        res2 = self.buffer.process_port_stats(
            dpid=1, port=2, timestamp=t2,
            rx_bytes=0, tx_bytes=2000,
            rx_packets=0, tx_packets=10,
            rx_dropped=0, tx_dropped=0,
            rx_errors=0, tx_errors=0,
            src_node="s1", dst_node="s2"
        )
        self.assertIsNotNone(res2)
        self.assertEqual(res2["traffic"], 8000) # 1000 bytes/s -> 8000 bps
        
if __name__ == "__main__":
    unittest.main()
