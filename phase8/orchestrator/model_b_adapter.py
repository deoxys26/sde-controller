"""
Phase 8 Milestone 3 – Frozen Model B adapter for Mininet/OVS telemetry.

ARCHITECTURE SUMMARY
====================
Model B (EdgeQoSForecaster) expects a *sequence* of directed-graph snapshots.
Each snapshot describes every directed edge with 8 scalar features (in order):

    Index  Feature        Unit (training)       Phase 8 source
    -----  -------------  -------------------   ----------------------------------------
    0      capacity       Mbps                  Topology constant: 100 Mbps
    1      traffic        Mbps                  OVS port stats → tx_byte_rate_bps / 1e6
    2      utilization    dimensionless 0–1     traffic_bps / capacity_bps
    3      queue_length   Mbps                  NOT AVAILABLE – set to 0.0 (documented)
    4      delay          ms                    NOT AVAILABLE – set to 0.0 (documented)
    5      packet_loss    dimensionless 0–1     NOT AVAILABLE – set to 0.0 (documented)
    6      jitter         ms                    NOT AVAILABLE – set to 0.0 (documented)
    7      throughput     Mbps                  Same as traffic (no loss measured)

MISSING FEATURES
----------------
queue_length  : OVS does not expose per-port queue length in OpenFlow port stats.
                The port stat rx_dropped/tx_dropped counters count frames discarded
                at the port level, not queue occupancy in Mbps.  No defensible
                mapping exists.  Set to 0.0 and documented.

delay         : Cannot be derived from OFPPortStatsRequest.  Controller event
                timestamps include OpenFlow processing + scheduling overhead and
                cannot be called physical network delay.  Set to 0.0 and documented.
                Measurement of actual RTT via host-to-host ICMP probes is deferred.

packet_loss   : rx_dropped / tx_dropped are switch-port counters, not end-to-end
                loss measurements.  During controlled iperf UDP experiments all
                drop counters were 0.  Setting to 0.0 is consistent with observed
                data but is NOT a validated measurement.  Documented as unavailable.

jitter        : Requires per-packet inter-arrival timing at endpoints.  Cannot be
                derived from OVS port stats.  Set to 0.0 and documented.

DOMAIN SHIFT (training vs Phase 8)
-----------------------------------
Feature      Training distribution               Phase 8 distribution
------------ ----------------------------------- --------------------------------
capacity     ~916–1098 Mbps (mean≈999, sd≈60)   100 Mbps (constant)
traffic      0–273 Mbps (mean≈29, sd≈46)        0–50 Mbps (iperf limited)
utilization  0–25 % (mean≈3 %, sd≈4.5 %)       0–54 % observed
queue_length 0 Mbps (no congestion in normal)   0 (unavailable)
delay        1.81–2.19 ms (mean≈2.0, sd≈0.11)  0 (unavailable)
packet_loss  0 (no loss in normal scenario)     0 (unavailable)
jitter       0.091–0.110 ms (mean≈0.10,sd≈0.006) 0 (unavailable)
throughput   same as traffic (no loss)           same as traffic

The scaler was fitted on training-period rows from 24 synthetic timesteps on
an 8-node Barabási–Albert topology with ~1000 Mbps links.  Phase 8 uses a
4-switch diamond with 100 Mbps links and 6 directed inter-switch edges.
The scaler will z-score Phase 8 inputs using synthetic statistics; this is a
KNOWN distribution mismatch.  Inference will execute but predictions may not
reflect Phase 8 ground truth.  This is documented and not hidden.

TEMPORAL ALIGNMENT
------------------
The adapter maintains a deque of at most `history_length` telemetry snapshots.
A snapshot is appended only when its wall-clock timestamp is strictly greater
than all existing entries.  Inference is only triggered when the deque is full
(history_length = 5 snapshots, matching the checkpoint).  Future timestamps are
never used: the sequence fed to the model contains only snapshots with timestamps
≤ t_now, and the target is at t_now + prediction_horizon (1 step ahead), which
is not yet available at inference time.  No future normalization statistics are
used: the scaler is fitted exclusively on synthetic training data from Phase 4.5.

CHECKPOINT
----------
File   : data/experiments/phase4_5_multiseed/checkpoints/normal_current_42.pt
Scenario: normal
Seed   : 42
Model  : EdgeQoSForecaster(edge_feature_count=8, target_count=4,
                           hidden_channels=16, gru_hidden_size=16)
The 'normal' scenario was chosen as the closest match to Phase 8 conditions
(low utilization, no deliberate congestion, no link degradation events).

OUTPUT FORMAT
-------------
A dict keyed by directed link string "s1->s2", containing:
    timestamp        : float  wall-clock of the latest history snapshot used
    src, dst         : str    switch names
    pred_delay       : float  predicted delay (ms) – inverse-scaled
    pred_utilization : float  predicted utilization (0–1) – inverse-scaled
    pred_packet_loss : float  predicted packet_loss (0–1) – clipped ≥ 0
    pred_jitter      : float  predicted jitter (ms) – inverse-scaled
    input_features   : list   the 8 raw (unscaled) feature values for latest snapshot
    unavailable_features : list  features set to 0.0 due to measurement gap
    WARNING          : str    domain shift notice always present
"""

from __future__ import annotations

import json
import logging
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from src.gnn.dataset import GraphSnapshot, TemporalGraphDataset
from src.gnn.model import EdgeQoSForecaster
from src.gnn.preprocessing import FeatureScaler

LOG = logging.getLogger("phase8.model_b_adapter")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CHECKPOINT_PATH = REPO_ROOT / "data" / "experiments" / "phase4_5_multiseed" / "checkpoints" / "normal_current_42.pt"

FEATURE_COLUMNS: Tuple[str, ...] = TemporalGraphDataset.feature_columns
# ('capacity', 'traffic', 'utilization', 'queue_length', 'delay',
#  'packet_loss', 'jitter', 'throughput')

TARGET_COLUMNS: Tuple[str, ...] = TemporalGraphDataset.target_columns
# ('delay', 'utilization', 'packet_loss', 'jitter')

HISTORY_LENGTH = 5         # must match the checkpoint
PREDICTION_HORIZON = 1     # must match the checkpoint (1-step ahead)
HIDDEN_CHANNELS = 16
GRU_HIDDEN_SIZE = 16

# Features that cannot be measured in Phase 8 and are substituted with 0.0
UNAVAILABLE_FEATURES = frozenset({"queue_length", "delay", "packet_loss", "jitter"})

# Scaler statistics from Phase 4.5 training period (normal scenario, seed 42)
# These were extracted from a TemporalGraphDataset fitted on the training split
# of the 24-timestep normal scenario with the gnn_experiment config.
# They are stored here so the adapter can run without re-simulating.
_TRAINING_SCALER_MEAN = np.array(
    [9.9853369e+02, 2.8761816e+01, 2.8598549e-02, 0.0,
     1.9979167e+00, 0.0, 9.9895835e-02, 2.8761816e+01],
    dtype=np.float32,
)
_TRAINING_SCALER_SCALE = np.array(
    [6.02776985e+01, 4.58966637e+01, 4.54507507e-02, 1.0,
     1.10716574e-01, 1.0, 5.53582888e-03, 4.58966637e+01],
    dtype=np.float32,
)

# Diamond topology directed inter-switch edges (Phase 8)
# Keys: (src_switch_name, dst_switch_name)
# Values: port on src that faces dst  – used for topology structure in graph
DIAMOND_EDGES: List[Tuple[str, str]] = [
    ("s1", "s2"), ("s2", "s1"),
    ("s1", "s3"), ("s3", "s1"),
    ("s2", "s4"), ("s4", "s2"),
    ("s3", "s4"), ("s4", "s3"),
]

# Map switch name → integer node id for the graph.x tensor
SWITCH_TO_NODE = {"s1": 0, "s2": 1, "s3": 2, "s4": 3}
NUM_NODES = 4

# Link capacity in Mbps (matches bw=100 in diamond_topo.py)
LINK_CAPACITY_MBPS = 100.0
LINK_CAPACITY_BPS = LINK_CAPACITY_MBPS * 1_000_000.0


class Phase8ModelBAdapter:
    """Wraps the frozen Model B checkpoint to consume Phase 8 telemetry.

    Usage
    -----
    adapter = Phase8ModelBAdapter()
    # For each 1-second poll result from telemetry_buffer:
    adapter.push_snapshot(telemetry_dict, wall_time)
    if adapter.ready():
        predictions = adapter.predict()
    """

    def __init__(self, checkpoint: Path = CHECKPOINT_PATH):
        # ── scaler (training-period only; never re-fitted on Phase 8 data) ──
        self.scaler = FeatureScaler()
        self.scaler.mean_ = _TRAINING_SCALER_MEAN.copy()
        self.scaler.scale_ = _TRAINING_SCALER_SCALE.copy()

        # ── directed edge list (deterministic order) ──────────────────────────
        self.edges = DIAMOND_EDGES
        self.edge_index = torch.tensor(
            [[SWITCH_TO_NODE[s], SWITCH_TO_NODE[d]] for s, d in self.edges],
            dtype=torch.long,
        ).T  # shape [2, num_edges]

        # ── frozen model ─────────────────────────────────────────────────────
        self.model = EdgeQoSForecaster(
            edge_feature_count=len(FEATURE_COLUMNS),
            target_count=len(TARGET_COLUMNS),
            hidden_channels=HIDDEN_CHANNELS,
            gru_hidden_size=GRU_HIDDEN_SIZE,
        )
        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"Phase 4.5 frozen checkpoint not found: {checkpoint}\n"
                "Run Phase 4.5 training first or verify the checkpoint path."
            )
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model.eval()
        LOG.info("Loaded frozen Model B from %s", checkpoint)

        # ── temporal history (deque of snapshots) ────────────────────────────
        # Each entry: {"timestamp": float, "features": np.ndarray shape [E, 8]}
        # E = number of directed edges = 8
        self._history: deque = deque(maxlen=HISTORY_LENGTH)

    # ── public interface ──────────────────────────────────────────────────────

    def push_snapshot(
        self,
        telemetry: Dict[str, dict],
        wall_time: float,
    ) -> None:
        """Ingest one telemetry poll result into the history buffer.

        Parameters
        ----------
        telemetry : dict
            Keyed by "src->dst" strings, values are the records from
            TelemetryBuffer.get_latest_telemetry().  Each record must contain
            at minimum: traffic (bps), utilization (0-1).
        wall_time : float
            Unix timestamp of this poll (time.time()).  Must be strictly
            greater than all existing history entries (monotonic).
        """
        if self._history and wall_time <= self._history[-1]["timestamp"]:
            LOG.warning(
                "Snapshot timestamp %.3f is not strictly later than history tail %.3f – skipped",
                wall_time, self._history[-1]["timestamp"],
            )
            return

        features = self._build_feature_matrix(telemetry)
        self._history.append({"timestamp": wall_time, "features": features})
        LOG.debug("History: %d/%d snapshots", len(self._history), HISTORY_LENGTH)

    def ready(self) -> bool:
        """Return True when the history buffer is full enough for inference."""
        return len(self._history) == HISTORY_LENGTH

    def predict(self) -> Dict[str, dict]:
        """Run frozen Model B inference on the current history.

        Returns a dict keyed by "src->dst" with predicted QoS and metadata.
        Raises RuntimeError if history is not full.
        """
        if not self.ready():
            raise RuntimeError(
                f"History has {len(self._history)}/{HISTORY_LENGTH} snapshots; "
                "call push_snapshot() until ready() is True."
            )

        # Chronological order is guaranteed by monotonic push_snapshot().
        history_list = list(self._history)

        # Verify strictly increasing timestamps (no future leakage).
        for i in range(1, len(history_list)):
            assert history_list[i]["timestamp"] > history_list[i - 1]["timestamp"], (
                "BUG: history timestamps are not strictly monotonically increasing – "
                "future leakage guard triggered."
            )

        # Build graph snapshot sequence (oldest → newest).
        sequence = []
        for entry in history_list:
            scaled = self.scaler.transform(entry["features"])  # [E, 8]
            snap = GraphSnapshot(
                x=torch.ones((NUM_NODES, 1), dtype=torch.float32),
                edge_index=self.edge_index,
                edge_attr=torch.tensor(scaled, dtype=torch.float32),
                timestep=0,  # abstract; actual wall-clock time in entry["timestamp"]
            )
            sequence.append(snap)

        with torch.no_grad():
            raw_pred = self.model(sequence).cpu().numpy()  # [E, 4]

        if not np.isfinite(raw_pred).all():
            raise RuntimeError("Model B emitted non-finite predictions.")

        # Inverse-scale targets.
        target_indices = [list(FEATURE_COLUMNS).index(t) for t in TARGET_COLUMNS]
        pred_unscaled = (
            raw_pred * self.scaler.scale_[target_indices]
            + self.scaler.mean_[target_indices]
        )  # [E, 4]

        latest_ts = history_list[-1]["timestamp"]
        latest_features = history_list[-1]["features"]  # [E, 8]

        results = {}
        for i, (src, dst) in enumerate(self.edges):
            key = f"{src}->{dst}"
            pred_delay, pred_util, pred_loss, pred_jitter = pred_unscaled[i]
            results[key] = {
                "timestamp": latest_ts,
                "src": src,
                "dst": dst,
                # Predictions (inverse-scaled from normalized domain)
                "pred_delay": float(max(0.0, pred_delay)),
                "pred_utilization": float(max(0.0, pred_util)),
                "pred_packet_loss": float(np.clip(pred_loss, 0.0, 0.99)),
                "pred_jitter": float(max(0.0, pred_jitter)),
                # Raw input at latest snapshot (for logging / domain-shift analysis)
                "input_features": {
                    name: float(latest_features[i, j])
                    for j, name in enumerate(FEATURE_COLUMNS)
                },
                "unavailable_features": sorted(UNAVAILABLE_FEATURES),
                "WARNING": (
                    "Model B was trained on synthetic ~1000 Mbps topology data. "
                    "Phase 8 uses 100 Mbps Mininet links. Missing features "
                    "(queue_length, delay, packet_loss, jitter) are set to 0.0. "
                    "Predictions reflect interface compatibility, not validated accuracy."
                ),
            }
        return results

    def history_timestamps(self) -> List[float]:
        """Return wall-clock timestamps of all buffered snapshots (oldest first)."""
        return [entry["timestamp"] for entry in self._history]

    # ── private helpers ───────────────────────────────────────────────────────

    def _build_feature_matrix(self, telemetry: Dict[str, dict]) -> np.ndarray:
        """Convert a telemetry dict into the [E, 8] float32 feature matrix.

        Feature order is strictly: capacity, traffic, utilization, queue_length,
        delay, packet_loss, jitter, throughput  (indices 0-7, matching training).

        Missing Phase 8 measurements are substituted with 0.0 and documented in
        UNAVAILABLE_FEATURES.  The substitution of 0.0 for delay/jitter/loss
        places Phase 8 inputs far outside the training distribution for those
        features (training delay mean ≈ 2.0 ms, Phase 8 = 0.0 ms).
        """
        E = len(self.edges)
        matrix = np.zeros((E, len(FEATURE_COLUMNS)), dtype=np.float32)

        for i, (src, dst) in enumerate(self.edges):
            key = f"{src}->{dst}"
            rec = telemetry.get(key, {})

            traffic_bps = float(rec.get("traffic", 0.0))  # bps from telemetry_buffer
            traffic_mbps = traffic_bps / 1_000_000.0
            util = float(rec.get("utilization", 0.0))

            # throughput: no packet loss is measured, so throughput ≈ traffic
            throughput_mbps = traffic_mbps

            matrix[i, 0] = LINK_CAPACITY_MBPS      # capacity (Mbps)
            matrix[i, 1] = traffic_mbps             # traffic (Mbps)
            matrix[i, 2] = util                     # utilization (0-1)
            matrix[i, 3] = 0.0                      # queue_length – UNAVAILABLE
            matrix[i, 4] = 0.0                      # delay (ms) – UNAVAILABLE
            matrix[i, 5] = 0.0                      # packet_loss – UNAVAILABLE
            matrix[i, 6] = 0.0                      # jitter (ms) – UNAVAILABLE
            matrix[i, 7] = throughput_mbps          # throughput (Mbps)

        return matrix
