import time
from collections import defaultdict

class TelemetryBuffer:
    def __init__(self, capacity_bps=100_000_000):
        # Store recent observations for each directed link: (src_switch, dst_switch) -> list of records
        self.link_history = defaultdict(list)
        
        # Store raw cumulative counters for rate calculation: (dpid, port) -> last record
        self._last_raw_counters = {}
        
        # We assume 100 Mbps for all links by default, matching topology bw=100
        self.capacity_bps = capacity_bps

    def process_port_stats(self, dpid, port, timestamp, rx_bytes, tx_bytes, rx_packets, tx_packets, rx_dropped, tx_dropped, rx_errors, tx_errors, src_node, dst_node):
        """
        Process a new port statistics sample, calculate rates, and store in the buffer.
        """
        port_key = (dpid, port)
        
        current_raw = {
            "timestamp": timestamp,
            "rx_bytes": rx_bytes,
            "tx_bytes": tx_bytes,
            "rx_packets": rx_packets,
            "tx_packets": tx_packets,
            "rx_dropped": rx_dropped,
            "tx_dropped": tx_dropped,
            "rx_errors": rx_errors,
            "tx_errors": tx_errors
        }
        
        if port_key not in self._last_raw_counters:
            self._last_raw_counters[port_key] = current_raw
            return None # First sample, cannot calculate rate
            
        prev_raw = self._last_raw_counters[port_key]
        
        dt = timestamp - prev_raw["timestamp"]
        if dt <= 0:
            return None # Invalid time delta
            
        # Handle counter reset (e.g., switch reboot)
        if rx_bytes < prev_raw["rx_bytes"] or tx_bytes < prev_raw["tx_bytes"]:
            self._last_raw_counters[port_key] = current_raw
            return None
            
        rx_byte_rate = (rx_bytes - prev_raw["rx_bytes"]) / dt
        tx_byte_rate = (tx_bytes - prev_raw["tx_bytes"]) / dt
        
        # The link is directed. If this is the TX side (e.g., s1 -> s2 on s1 port), we care about TX bytes.
        # But wait, we measure from both ends if we want. For simplicity, if we map (switch, port) -> directed edge,
        # then the traffic LEAVING this switch on this port is TX.
        
        # We will use tx_bytes for traffic leaving the switch towards dst_node
        traffic_bps = tx_byte_rate * 8
        utilization = traffic_bps / self.capacity_bps
        
        telemetry_record = {
            "timestamp": timestamp,
            "src": src_node,
            "dst": dst_node,
            "capacity": self.capacity_bps,
            "traffic": traffic_bps,
            "utilization": utilization,
            "rx_bytes": rx_bytes,
            "tx_bytes": tx_bytes,
            "rx_packets": rx_packets,
            "tx_packets": tx_packets,
            "rx_dropped": rx_dropped,
            "tx_dropped": tx_dropped,
            "rx_errors": rx_errors,
            "tx_errors": tx_errors
        }
        
        self.link_history[(src_node, dst_node)].append(telemetry_record)
        
        # Limit history size to prevent memory leak
        if len(self.link_history[(src_node, dst_node)]) > 1000:
            self.link_history[(src_node, dst_node)].pop(0)
            
        self._last_raw_counters[port_key] = current_raw
        
        return telemetry_record

    def get_latest_telemetry(self):
        """Return the most recent telemetry record for all directed links."""
        latest = {}
        for link, history in self.link_history.items():
            if history:
                key = f"{link[0]}->{link[1]}"
                latest[key] = history[-1]
        return latest
