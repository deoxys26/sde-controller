"""Leakage-safe conversion of Phase 3 link telemetry into graph sequences."""

from __future__ import annotations

import numpy as np
import torch
from dataclasses import dataclass

from src.gnn.preprocessing import FeatureScaler


@dataclass
class GraphSnapshot:
    """Minimal PyTorch-compatible directed graph snapshot."""
    x: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    timestep: int


class TemporalGraphDataset(torch.utils.data.Dataset):
    """Returns (historical directed graph snapshots, next-link-QoS target, metadata)."""

    feature_columns = ("capacity", "traffic", "utilization", "queue_length", "delay",
                       "packet_loss", "jitter", "throughput")
    target_columns = ("delay", "utilization", "packet_loss", "jitter")

    def __init__(self, link_telemetry, graph, history_length=5, prediction_horizon=1,
                 train_ratio=.70, validation_ratio=.15):
        if history_length < 1 or prediction_horizon < 1:
            raise ValueError("history_length and prediction_horizon must be positive")
        self.history_length, self.prediction_horizon = history_length, prediction_horizon
        self.graph = graph
        self.edges = tuple(sorted(graph.edges()))
        self.edge_index = torch.tensor(np.array(self.edges).T, dtype=torch.long)
        self.timesteps = tuple(sorted(link_telemetry["timestep"].unique()))
        if len(self.timesteps) < history_length + prediction_horizon:
            raise ValueError("Not enough timesteps for the requested history and horizon")
        required = {"timestep", "source", "destination", *self.feature_columns}
        missing = required - set(link_telemetry.columns)
        if missing:
            raise ValueError(f"Link telemetry lacks required columns: {sorted(missing)}")
        self._rows = {}
        for timestep in self.timesteps:
            rows = link_telemetry[link_telemetry.timestep == timestep].set_index(["source", "destination"])
            if any(edge not in rows.index for edge in self.edges):
                raise ValueError(f"Timestep {timestep} does not include every topology edge")
            self._rows[timestep] = rows.loc[list(self.edges), list(self.feature_columns)].to_numpy(dtype=np.float32)

        train_periods = max(1, int(len(self.timesteps) * train_ratio))
        validation_periods = max(1, int(len(self.timesteps) * validation_ratio))
        self.train_end_timestep = self.timesteps[train_periods - 1]
        self.validation_end_timestep = self.timesteps[min(len(self.timesteps) - 1, train_periods + validation_periods - 1)]
        # This fit deliberately excludes every validation/test timestep.
        train_values = np.concatenate([self._rows[t] for t in self.timesteps if t <= self.train_end_timestep])
        self.scaler = FeatureScaler().fit(train_values)
        self.target_indices = [self.feature_columns.index(name) for name in self.target_columns]
        self.samples = []
        for end_index in range(history_length - 1, len(self.timesteps) - prediction_horizon):
            history_times = self.timesteps[end_index - history_length + 1:end_index + 1]
            target_time = self.timesteps[end_index + prediction_horizon]
            split = "train" if target_time <= self.train_end_timestep else (
                "validation" if target_time <= self.validation_end_timestep else "test")
            self.samples.append((history_times, target_time, split))

    def __len__(self):
        return len(self.samples)

    def _snapshot(self, timestep):
        edge_attr = torch.tensor(self.scaler.transform(self._rows[timestep]), dtype=torch.float32)
        # Constant node features keep topology connectivity explicit; link attributes carry QoS state.
        x = torch.ones((self.graph.number_of_nodes(), 1), dtype=torch.float32)
        return GraphSnapshot(x=x, edge_index=self.edge_index, edge_attr=edge_attr, timestep=int(timestep))

    def __getitem__(self, index):
        history_times, target_time, split = self.samples[index]
        sequence = [self._snapshot(timestep) for timestep in history_times]
        target = torch.tensor(self.scaler.transform(self._rows[target_time])[:, self.target_indices], dtype=torch.float32)
        return sequence, target, {"history_timesteps": history_times, "target_timestep": target_time, "split": split}

    def indices(self, split):
        return [index for index, sample in enumerate(self.samples) if sample[2] == split]

    def inverse_targets(self, values):
        values = np.asarray(values, dtype=np.float32)
        return values * self.scaler.scale_[self.target_indices] + self.scaler.mean_[self.target_indices]
