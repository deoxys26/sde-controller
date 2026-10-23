# Phase 8 Milestone 2 Results

## Telemetry Collection
Ryu periodically queries OpenFlow port statistics and processes them in a `TelemetryBuffer`.

### Exact Measurement Definitions
* `traffic` (bps): Measured by `(tx_bytes_current - tx_bytes_previous) / dt * 8`. It represents the rate of bits leaving a switch port heading toward the connected adjacent switch.
* `utilization`: Measured by `traffic / capacity_bps`. Capacity is statically configured to 100Mbps per link.
* `rx_dropped` and `tx_dropped`: Raw cumulative drop counts reported by Open vSwitch per port. (Note: These are switch-port drops, not end-to-end drops).
* `delay` and `jitter`: Intentionally **deferred** per Milestone 2 guidelines, to prevent fabricating them from controller event timings.

### Raw Telemetry Example
The raw measurements tracked in state include:
```json
{
  "timestamp": 1790327281.8752253,
  "rx_bytes": 10240,
  "tx_bytes": 456780,
  "rx_packets": 128,
  "tx_packets": 320,
  "rx_dropped": 0,
  "tx_dropped": 0,
  "rx_errors": 0,
  "tx_errors": 0
}
```

### Derived Telemetry Example
When queried mid-iperf via `/phase8/telemetry`, the system reported the following for the `s1 -> s2` edge (Path A):
```json
{
  "timestamp": 1790327273.861783,
  "src": "s1",
  "dst": "s2",
  "capacity": 100000000,
  "traffic": 49953931.332308695,
  "utilization": 0.49953931332308694,
  "rx_bytes": 586,
  "tx_bytes": 12513222,
  "rx_packets": 7,
  "tx_packets": 8274,
  "rx_dropped": 0,
  "tx_dropped": 0,
  "rx_errors": 0,
  "tx_errors": 0
}
```
At the same time, `s1 -> s3` showed 0 traffic/utilization because it was not the active path.

### Fixes
* Fixed `pingall` drops from Milestone 1: Mininet `addLink` was misaligning switch-port indices with the controller's inter-switch map. This caused `qos_controller.py` to instruct `s4` to output toward `s3` instead of `h2`. Port indices in Mininet are deterministically ordered, so `diamond_topo.py` was adjusted. `pingall` now succeeds with 0% packet loss.

### Code
* `phase8/controller/qos_controller.py`
* `phase8/orchestrator/telemetry_buffer.py`
* `phase8/tests/test_telemetry.py`
