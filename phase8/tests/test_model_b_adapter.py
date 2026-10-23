"""
Phase 8 Milestone 3 – Model B adapter tests.

All tests are purely unit-level and use no Mininet/WSL/network.
They exercise the adapter's logic, tensor construction, scaler,
temporal history, and leakage guards.
"""

from __future__ import annotations

import math
import time
import unittest
from collections import deque
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from phase8.orchestrator.model_b_adapter import (
    DIAMOND_EDGES,
    FEATURE_COLUMNS,
    HISTORY_LENGTH,
    LINK_CAPACITY_MBPS,
    NUM_NODES,
    TARGET_COLUMNS,
    UNAVAILABLE_FEATURES,
    Phase8ModelBAdapter,
    _TRAINING_SCALER_MEAN,
    _TRAINING_SCALER_SCALE,
)

CHECKPOINT = Path("data/experiments/phase4_5_multiseed/checkpoints/normal_current_42.pt")


def _make_telemetry(traffic_bps=1_000_000.0, util=0.01):
    """Build a minimal telemetry dict covering all 8 diamond directed edges."""
    return {
        f"{src}->{dst}": {
            "src": src, "dst": dst,
            "traffic": traffic_bps,
            "utilization": util,
            "capacity": 100_000_000,
            "rx_bytes": 1000, "tx_bytes": 1000,
            "rx_packets": 10, "tx_packets": 10,
            "rx_dropped": 0, "tx_dropped": 0,
            "rx_errors": 0, "tx_errors": 0,
        }
        for src, dst in DIAMOND_EDGES
    }


class TestCheckpointLoading(unittest.TestCase):
    def test_checkpoint_loads_without_error(self):
        if not CHECKPOINT.is_file():
            self.skipTest("Checkpoint not available in this environment")
        adapter = Phase8ModelBAdapter(checkpoint=CHECKPOINT)
        self.assertIsInstance(adapter.model, torch.nn.Module)

    def test_model_is_set_to_eval_mode(self):
        if not CHECKPOINT.is_file():
            self.skipTest("Checkpoint not available in this environment")
        adapter = Phase8ModelBAdapter(checkpoint=CHECKPOINT)
        self.assertFalse(adapter.model.training, "Model must be in eval mode")

    def test_missing_checkpoint_raises_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            Phase8ModelBAdapter(checkpoint=Path("/nonexistent/path/model.pt"))


class TestFeatureOrdering(unittest.TestCase):
    def test_feature_columns_exact_order(self):
        expected = (
            "capacity", "traffic", "utilization", "queue_length",
            "delay", "packet_loss", "jitter", "throughput",
        )
        self.assertEqual(FEATURE_COLUMNS, expected)

    def test_target_columns_exact_order(self):
        expected = ("delay", "utilization", "packet_loss", "jitter")
        self.assertEqual(TARGET_COLUMNS, expected)

    def test_unavailable_features_documented(self):
        for feat in ("queue_length", "delay", "packet_loss", "jitter"):
            self.assertIn(feat, UNAVAILABLE_FEATURES)


class TestTensorConstruction(unittest.TestCase):
    def setUp(self):
        if not CHECKPOINT.is_file():
            self.skipTest("Checkpoint not available in this environment")
        self.adapter = Phase8ModelBAdapter(checkpoint=CHECKPOINT)
        self.telemetry = _make_telemetry(traffic_bps=10_000_000.0, util=0.10)

    def test_feature_matrix_shape(self):
        mat = self.adapter._build_feature_matrix(self.telemetry)
        self.assertEqual(mat.shape, (len(DIAMOND_EDGES), len(FEATURE_COLUMNS)))

    def test_capacity_column_is_constant(self):
        mat = self.adapter._build_feature_matrix(self.telemetry)
        np.testing.assert_array_equal(mat[:, 0], LINK_CAPACITY_MBPS)

    def test_traffic_converted_from_bps_to_mbps(self):
        mat = self.adapter._build_feature_matrix(self.telemetry)
        # 10_000_000 bps / 1e6 = 10 Mbps
        np.testing.assert_array_almost_equal(mat[:, 1], 10.0)

    def test_unavailable_features_are_zero(self):
        mat = self.adapter._build_feature_matrix(self.telemetry)
        # queue_length=3, delay=4, packet_loss=5, jitter=6
        for col in (3, 4, 5, 6):
            np.testing.assert_array_equal(mat[:, col], 0.0,
                err_msg=f"Column {col} ({FEATURE_COLUMNS[col]}) must be 0.0")

    def test_throughput_equals_traffic_when_no_loss(self):
        mat = self.adapter._build_feature_matrix(self.telemetry)
        np.testing.assert_array_equal(mat[:, 1], mat[:, 7])

    def test_missing_edge_defaults_to_zero_traffic(self):
        # Partial telemetry – only one edge present
        partial = {f"{DIAMOND_EDGES[0][0]}->{DIAMOND_EDGES[0][1]}": {
            "traffic": 5_000_000.0, "utilization": 0.05}}
        mat = self.adapter._build_feature_matrix(partial)
        # Row 0 should have traffic=5 Mbps; remaining rows should have 0
        self.assertAlmostEqual(mat[0, 1], 5.0)
        np.testing.assert_array_equal(mat[1:, 1], 0.0)


class TestScalerApplication(unittest.TestCase):
    def test_scaler_mean_shape(self):
        self.assertEqual(_TRAINING_SCALER_MEAN.shape, (len(FEATURE_COLUMNS),))

    def test_scaler_scale_positive(self):
        self.assertTrue((np.asarray(_TRAINING_SCALER_SCALE) > 0).all())

    def test_scaler_transform_known_value(self):
        """capacity = mean → z-score = 0."""
        from src.gnn.preprocessing import FeatureScaler
        s = FeatureScaler()
        s.mean_ = _TRAINING_SCALER_MEAN.copy()
        s.scale_ = _TRAINING_SCALER_SCALE.copy()
        row = _TRAINING_SCALER_MEAN.reshape(1, -1)
        transformed = s.transform(row)
        np.testing.assert_array_almost_equal(transformed, np.zeros((1, len(FEATURE_COLUMNS))), decimal=5)


class TestHistoryConstruction(unittest.TestCase):
    def setUp(self):
        if not CHECKPOINT.is_file():
            self.skipTest("Checkpoint not available in this environment")
        self.adapter = Phase8ModelBAdapter(checkpoint=CHECKPOINT)

    def _fill_history(self, n=HISTORY_LENGTH, base_t=1000.0, dt=1.0,
                      traffic_bps=5_000_000.0):
        for i in range(n):
            telem = _make_telemetry(traffic_bps=traffic_bps)
            self.adapter.push_snapshot(telem, wall_time=base_t + i * dt)

    def test_not_ready_with_empty_history(self):
        self.assertFalse(self.adapter.ready())

    def test_not_ready_with_partial_history(self):
        self._fill_history(n=HISTORY_LENGTH - 1)
        self.assertFalse(self.adapter.ready())

    def test_ready_when_history_full(self):
        self._fill_history(n=HISTORY_LENGTH)
        self.assertTrue(self.adapter.ready())

    def test_history_length_bounded(self):
        # Push extra snapshots; deque must cap at HISTORY_LENGTH
        self._fill_history(n=HISTORY_LENGTH + 3)
        self.assertEqual(len(self.adapter._history), HISTORY_LENGTH)

    def test_timestamps_monotonically_increasing(self):
        self._fill_history(n=HISTORY_LENGTH)
        ts = self.adapter.history_timestamps()
        for i in range(1, len(ts)):
            self.assertGreater(ts[i], ts[i - 1])


class TestTimestampOrdering(unittest.TestCase):
    def setUp(self):
        if not CHECKPOINT.is_file():
            self.skipTest("Checkpoint not available in this environment")
        self.adapter = Phase8ModelBAdapter(checkpoint=CHECKPOINT)

    def test_non_monotonic_snapshot_is_rejected(self):
        """A snapshot with the same or earlier timestamp must be dropped."""
        t0 = 1000.0
        self.adapter.push_snapshot(_make_telemetry(), wall_time=t0)
        # Same timestamp – should be silently rejected
        self.adapter.push_snapshot(_make_telemetry(), wall_time=t0)
        self.assertEqual(len(self.adapter._history), 1)

    def test_earlier_timestamp_is_rejected(self):
        self.adapter.push_snapshot(_make_telemetry(), wall_time=1000.0)
        self.adapter.push_snapshot(_make_telemetry(), wall_time=999.0)
        self.assertEqual(len(self.adapter._history), 1)


class TestNoFutureLeakage(unittest.TestCase):
    """Verify that inference uses only past/present snapshots."""

    def setUp(self):
        if not CHECKPOINT.is_file():
            self.skipTest("Checkpoint not available in this environment")
        self.adapter = Phase8ModelBAdapter(checkpoint=CHECKPOINT)
        # Fill history with strictly ascending timestamps
        for i in range(HISTORY_LENGTH):
            self.adapter.push_snapshot(_make_telemetry(), wall_time=float(1000 + i))

    def test_latest_history_timestamp_is_most_recent_push(self):
        ts = self.adapter.history_timestamps()
        self.assertEqual(ts[-1], float(1000 + HISTORY_LENGTH - 1))

    def test_predict_does_not_consume_future_samples(self):
        """After predict(), the history is unchanged (no future sample consumed)."""
        ts_before = list(self.adapter.history_timestamps())
        self.adapter.predict()
        ts_after = list(self.adapter.history_timestamps())
        self.assertEqual(ts_before, ts_after)

    def test_internal_assert_detects_nonmonotonic_history(self):
        """Tamper with the internal deque then confirm the assertion fires."""
        # Directly inject a non-monotonic entry at the end
        self.adapter._history.append(
            {"timestamp": -999.0,
             "features": self.adapter._history[-1]["features"].copy()})
        with self.assertRaises(AssertionError):
            self.adapter.predict()


class TestDirectedEdgeMapping(unittest.TestCase):
    def test_all_eight_edges_present(self):
        expected = {
            ("s1", "s2"), ("s2", "s1"), ("s1", "s3"), ("s3", "s1"),
            ("s2", "s4"), ("s4", "s2"), ("s3", "s4"), ("s4", "s3"),
        }
        self.assertEqual(set(DIAMOND_EDGES), expected)

    def test_edge_index_shape(self):
        if not Path(CHECKPOINT).is_file():
            self.skipTest("Checkpoint not available")
        adapter = Phase8ModelBAdapter(checkpoint=CHECKPOINT)
        self.assertEqual(adapter.edge_index.shape, (2, len(DIAMOND_EDGES)))

    def test_edge_index_values_in_range(self):
        if not Path(CHECKPOINT).is_file():
            self.skipTest("Checkpoint not available")
        adapter = Phase8ModelBAdapter(checkpoint=CHECKPOINT)
        self.assertTrue((adapter.edge_index >= 0).all())
        self.assertTrue((adapter.edge_index < NUM_NODES).all())


class TestOutputShape(unittest.TestCase):
    def setUp(self):
        if not CHECKPOINT.is_file():
            self.skipTest("Checkpoint not available in this environment")
        self.adapter = Phase8ModelBAdapter(checkpoint=CHECKPOINT)
        for i in range(HISTORY_LENGTH):
            self.adapter.push_snapshot(
                _make_telemetry(traffic_bps=float(i + 1) * 1_000_000),
                wall_time=float(1000 + i),
            )

    def test_output_has_all_edges(self):
        results = self.adapter.predict()
        self.assertEqual(len(results), len(DIAMOND_EDGES))

    def test_output_keys_match_edges(self):
        results = self.adapter.predict()
        expected_keys = {f"{s}->{d}" for s, d in DIAMOND_EDGES}
        self.assertEqual(set(results.keys()), expected_keys)

    def test_output_contains_required_fields(self):
        results = self.adapter.predict()
        for key, rec in results.items():
            for field in ("timestamp", "src", "dst", "pred_delay",
                          "pred_utilization", "pred_packet_loss", "pred_jitter",
                          "input_features", "unavailable_features", "WARNING"):
                self.assertIn(field, rec, f"Missing field '{field}' in {key}")

    def test_predictions_are_finite(self):
        results = self.adapter.predict()
        for key, rec in results.items():
            for field in ("pred_delay", "pred_utilization",
                          "pred_packet_loss", "pred_jitter"):
                self.assertTrue(
                    math.isfinite(rec[field]),
                    f"{field} is not finite for edge {key}",
                )

    def test_predictions_non_negative(self):
        results = self.adapter.predict()
        for key, rec in results.items():
            self.assertGreaterEqual(rec["pred_delay"], 0.0)
            self.assertGreaterEqual(rec["pred_utilization"], 0.0)
            self.assertGreaterEqual(rec["pred_packet_loss"], 0.0)
            self.assertGreaterEqual(rec["pred_jitter"], 0.0)


class TestDeterministicInference(unittest.TestCase):
    """Same inputs must always produce identical predictions."""

    def test_inference_is_deterministic(self):
        if not CHECKPOINT.is_file():
            self.skipTest("Checkpoint not available in this environment")
        results_a, results_b = [], []
        for run in (results_a, results_b):
            adapter = Phase8ModelBAdapter(checkpoint=CHECKPOINT)
            for i in range(HISTORY_LENGTH):
                adapter.push_snapshot(
                    _make_telemetry(traffic_bps=5_000_000.0), wall_time=float(1000 + i)
                )
            run.extend(adapter.predict().items())
        for (k_a, v_a), (k_b, v_b) in zip(sorted(results_a), sorted(results_b)):
            self.assertEqual(k_a, k_b)
            self.assertAlmostEqual(v_a["pred_utilization"], v_b["pred_utilization"], places=6)
            self.assertAlmostEqual(v_a["pred_delay"], v_b["pred_delay"], places=6)


if __name__ == "__main__":
    unittest.main()
