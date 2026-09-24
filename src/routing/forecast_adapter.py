"""Frozen Phase 4.5 checkpoint adapter for causal Phase 7 route decisions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from src.gnn.dataset import GraphSnapshot, TemporalGraphDataset
from src.gnn.run_experiment import _dataset_for_scenario, build_model


class FrozenQoSForecaster:
    """Load a pre-trained forecaster and its training-period-only scaler.

    The scaler is reconstructed by the original Phase 4.5 dataset protocol:
    it fits only timestamps in that rollout's chronological training period.
    No Phase 7 evaluation observations participate in fitting.
    """

    def __init__(self, scenario, gnn_config, checkpoint_dir, model_name="current", training_seed=42,
                 scaler_output_dir=None):
        self.scenario = scenario
        self.dataset = _dataset_for_scenario(scenario, gnn_config)
        self._assert_protocol(gnn_config)
        self.model = build_model(model_name, self.dataset, gnn_config)
        self.checkpoint = Path(checkpoint_dir) / f"{scenario}_{model_name}_{training_seed}.pt"
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"Required frozen Phase 4.5 checkpoint is missing: {self.checkpoint}")
        try:
            self.model.load_state_dict(torch.load(self.checkpoint, map_location="cpu", weights_only=True), strict=True)
        except RuntimeError as error:
            raise RuntimeError(f"Checkpoint is incompatible with {model_name} / Phase 4.5 feature protocol: {self.checkpoint}") from error
        self.model.eval()
        if scaler_output_dir is not None:
            self.serialize_scaler(Path(scaler_output_dir) / f"{scenario}_{model_name}_{training_seed}_scaler.json")

    def _assert_protocol(self, config):
        if tuple(self.dataset.feature_columns) != TemporalGraphDataset.feature_columns:
            raise ValueError("Unexpected Phase 4.5 feature ordering")
        if tuple(self.dataset.target_columns) != TemporalGraphDataset.target_columns:
            raise ValueError("Unexpected Phase 4.5 target ordering")
        if self.dataset.history_length != int(config["history_length"]):
            raise ValueError("Checkpoint history length differs from configured Phase 4.5 protocol")
        if self.dataset.prediction_horizon != int(config["prediction_horizon"]):
            raise ValueError("Phase 7 requires the Phase 4.5 one-step prediction horizon")

    def serialize_scaler(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"feature_columns": list(self.dataset.feature_columns),
                   "target_columns": list(self.dataset.target_columns),
                   "mean": self.dataset.scaler.mean_.astype(float).tolist(),
                   "scale": self.dataset.scaler.scale_.astype(float).tolist(),
                   "train_end_timestep": int(self.dataset.train_end_timestep),
                   "fit_scope": "Phase 4.5 chronological training period only"}
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def predict(self, history_rows):
        """Forecast next-link targets from exactly ``history_length`` past snapshots."""
        if len(history_rows) != self.dataset.history_length:
            raise ValueError("Forecast history must match the checkpoint's configured history length")
        snapshots = []
        for rows in history_rows:
            by_edge = {(row["source"], row["destination"]): row for row in rows}
            if any(edge not in by_edge for edge in self.dataset.edges):
                raise ValueError("Forecast history is missing a topology edge")
            values = np.asarray([[by_edge[edge][name] for name in self.dataset.feature_columns]
                                 for edge in self.dataset.edges], dtype=np.float32)
            snapshots.append(GraphSnapshot(
                x=torch.ones((self.dataset.graph.number_of_nodes(), 1), dtype=torch.float32),
                edge_index=self.dataset.edge_index,
                edge_attr=torch.tensor(self.dataset.scaler.transform(values), dtype=torch.float32),
                timestep=int(rows[0]["timestep"])))
        with torch.no_grad():
            normalized = self.model(snapshots).cpu().numpy()
        if not np.isfinite(normalized).all():
            raise RuntimeError("Frozen GNN emitted a non-finite QoS prediction")
        predicted_targets = self.dataset.inverse_targets(normalized)
        latest = {(row["source"], row["destination"]): dict(row) for row in history_rows[-1]}
        output = []
        for edge, values in zip(self.dataset.edges, predicted_targets):
            row = latest[edge]
            row.update(dict(zip(self.dataset.target_columns, map(float, values))))
            row["timestep"] = int(row["timestep"]) + self.dataset.prediction_horizon
            # Forecasting can slightly exceed physical bounds; routing costs are
            # bounded without changing the raw prediction artifact.
            row["packet_loss"] = float(np.clip(row["packet_loss"], 0.0, 0.99))
            row["utilization"] = float(max(0.0, row["utilization"]))
            row["delay"] = float(max(0.0, row["delay"]))
            row["jitter"] = float(max(0.0, row["jitter"]))
            output.append(row)
        return output
