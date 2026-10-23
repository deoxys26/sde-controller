"""
Phase 8 Milestone 3 – Controlled inference experiment.

Runs two experiments against real Mininet/OVS telemetry collected live:
  Experiment A : low traffic  (~10 Mbps UDP via iperf)
  Experiment B : high traffic (~50 Mbps UDP via iperf)

For each experiment:
  1. Starts the diamond topology + Ryu controller.
  2. Installs Path A (s1→s2→s4).
  3. Waits for HISTORY_LENGTH + 2 telemetry polls.
  4. Feeds history into frozen Model B.
  5. Saves predictions to JSON.
  6. Reports domain-shift descriptive statistics.

Run from the project root inside Ubuntu-24.04 WSL:
  sudo python3 phase8/experiments/run_model_b_experiment.py

No GA integration, no path changes, inference-only.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from pathlib import Path

# So we can import from the project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np

from phase8.orchestrator.model_b_adapter import (
    HISTORY_LENGTH,
    Phase8ModelBAdapter,
)

CONTROLLER_URL = "http://localhost:8080"
POLL_INTERVAL_S = 1.0          # telemetry poll interval already set in controller
WARM_UP_S = 3.0                # wait before first poll for paths to settle
COLLECT_N = HISTORY_LENGTH + 2  # snapshots to collect (history + 2 headroom)
OUTPUT_DIR = Path("phase8/experiments/results")


def get_telemetry():
    req = urllib.request.Request(f"{CONTROLLER_URL}/phase8/telemetry")
    with urllib.request.urlopen(req, timeout=5) as resp:
        return json.loads(resp.read().decode("utf-8"))


def set_path(path_nodes):
    body = json.dumps({"path": path_nodes}).encode("utf-8")
    req = urllib.request.Request(
        f"{CONTROLLER_URL}/phase8/path",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        return json.loads(resp.read().decode("utf-8"))


def run_experiment(label: str, iperf_bps: int, adapter: Phase8ModelBAdapter,
                   net) -> dict:
    """Collect telemetry and run inference for one traffic level."""
    h1, h2 = net.get("h1", "h2")

    print(f"\n=== {label} ===")
    print(f"  Starting iperf: {iperf_bps/1e6:.0f} Mbps UDP for {COLLECT_N + 4}s")
    h2.cmd("pkill iperf; sleep 0.2; iperf -s -u &")
    time.sleep(0.5)
    h1.cmd(f"iperf -c 10.0.0.2 -u -b {iperf_bps} -t {COLLECT_N + 4} &")

    time.sleep(WARM_UP_S)

    snapshots = []
    print(f"  Collecting {COLLECT_N} telemetry snapshots …")
    for i in range(COLLECT_N):
        t0 = time.time()
        try:
            telem = get_telemetry()
        except Exception as exc:
            print(f"  [WARN] telemetry fetch failed at snapshot {i}: {exc}")
            telem = {}
        adapter.push_snapshot(telem, wall_time=t0)
        snapshots.append({"timestamp": t0, "telemetry": telem})
        print(f"    snapshot {i+1}/{COLLECT_N} @ t={t0:.2f}")
        remaining = POLL_INTERVAL_S - (time.time() - t0)
        if remaining > 0:
            time.sleep(remaining)

    # Kill iperf
    h1.cmd("pkill iperf")
    h2.cmd("pkill iperf")

    predictions = None
    if adapter.ready():
        predictions = adapter.predict()
        print(f"  Inference succeeded. Predictions for {len(predictions)} edges.")
    else:
        print(f"  [WARN] adapter not ready: {len(adapter._history)}/{HISTORY_LENGTH}")

    # Descriptive stats on measured utilization
    utils = []
    for snap in snapshots[-HISTORY_LENGTH:]:
        for rec in snap["telemetry"].values():
            utils.append(float(rec.get("utilization", 0.0)))
    util_stats = {
        "n": len(utils),
        "mean": float(np.mean(utils)) if utils else None,
        "std": float(np.std(utils)) if utils else None,
        "min": float(np.min(utils)) if utils else None,
        "max": float(np.max(utils)) if utils else None,
    }
    print(f"  Measured utilization: mean={util_stats['mean']:.4f} "
          f"std={util_stats['std']:.4f} "
          f"min={util_stats['min']:.4f} max={util_stats['max']:.4f}")

    return {
        "experiment": label,
        "iperf_target_mbps": iperf_bps / 1e6,
        "snapshots_collected": len(snapshots),
        "history_used": HISTORY_LENGTH,
        "measured_utilization_stats": util_stats,
        "predictions": predictions,
        "note_unavailable_features": [
            "queue_length: not measurable via OFPPortStatsRequest",
            "delay: cannot be derived from controller event timestamps",
            "packet_loss: rx_dropped/tx_dropped are port-level, not end-to-end",
            "jitter: not measurable via OFPPortStatsRequest",
        ],
        "domain_shift_note": (
            "Model B trained on synthetic ~1000 Mbps topology (normal scenario). "
            "Phase 8 uses 100 Mbps Mininet links. Unavailable features set to 0.0. "
            "Scaler was not re-fitted on Phase 8 data."
        ),
    }


def main():
    # Ensure system mininet is importable
    if "/usr/lib/python3/dist-packages" not in sys.path:
        sys.path.append("/usr/lib/python3/dist-packages")

    from mininet.log import setLogLevel, info
    from mininet.node import OVSSwitch, RemoteController
    from mininet.link import TCLink
    from mininet.net import Mininet

    setLogLevel("warning")

    net = Mininet(
        controller=RemoteController,
        switch=OVSSwitch,
        link=TCLink,
        autoSetMacs=True,
        waitConnected=True,
    )

    c0 = net.addController("c0", controller=RemoteController, ip="127.0.0.1", port=6633)
    s1 = net.addSwitch("s1", cls=OVSSwitch, protocols="OpenFlow13")
    s2 = net.addSwitch("s2", cls=OVSSwitch, protocols="OpenFlow13")
    s3 = net.addSwitch("s3", cls=OVSSwitch, protocols="OpenFlow13")
    s4 = net.addSwitch("s4", cls=OVSSwitch, protocols="OpenFlow13")

    h1 = net.addHost("h1", ip="10.0.0.1/24", mac="00:00:00:00:00:01")
    h2 = net.addHost("h2", ip="10.0.0.2/24", mac="00:00:00:00:00:02")

    # Links matching diamond_topo.py
    net.addLink(h1, s1, bw=100)
    net.addLink(s1, s2, bw=100)
    net.addLink(s2, s4, bw=100)
    net.addLink(s1, s3, bw=100)
    net.addLink(s3, s4, bw=100)
    net.addLink(h2, s4, bw=100)

    net.build()
    c0.start()
    for sw in (s1, s2, s3, s4):
        sw.start([c0])
    net.waitConnected()
    time.sleep(2)

    # Install Path A
    try:
        set_path(["s1", "s2", "s4"])
        print("Path A (s1→s2→s4) installed.")
    except Exception as exc:
        print(f"[ERROR] Could not set path: {exc}")
        net.stop()
        return

    time.sleep(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []

    for label, target_mbps in [("A_low_traffic_10Mbps", 10), ("B_high_traffic_50Mbps", 50)]:
        adapter = Phase8ModelBAdapter()
        result = run_experiment(label, int(target_mbps * 1_000_000), adapter, net)
        all_results.append(result)
        # Clear adapter between experiments
        time.sleep(2)

    net.stop()

    out_path = OUTPUT_DIR / "milestone3_inference.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")
    print("\n=== DOMAIN SHIFT SUMMARY ===")
    print("  Training data (synthetic normal scenario, seed 42, ~1000 Mbps):")
    print("    capacity: mean=998.5 Mbps, sd=60.4 Mbps")
    print("    traffic : mean=28.8 Mbps, sd=46.0 Mbps")
    print("    util    : mean=0.029, sd=0.046")
    print("    delay   : mean=2.0 ms, sd=0.11 ms")
    print("    loss    : mean=0.0, sd=0.0  (zero loss in normal scenario)")
    print("    jitter  : mean=0.10 ms, sd=0.006 ms")
    print("  Phase 8 data (Mininet OVS, 100 Mbps):")
    for r in all_results:
        s = r["measured_utilization_stats"]
        print(f"    [{r['experiment']}] util mean={s['mean']:.4f} sd={s['std']:.4f}")
    print("\n  FINDING: capacity is 10× smaller than training (100 vs ~1000 Mbps).")
    print("  Missing features (delay, jitter) set to 0.0 – far from training mean.")
    print("  Inference executes without error but predicted values reflect")
    print("  training distribution, not Phase 8 ground truth.")
    print("  This is an interface compatibility result, NOT a predictive accuracy result.")


if __name__ == "__main__":
    main()
