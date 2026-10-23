"""Experimental edge-centric multi-horizon utilization forecaster.

Inputs are per-directed-link utilization histories in the caller-supplied edge
order. Optional daily/weekly residual features are derived from that same
history; externally supplied seasonal features are accepted only when they
match the history-derived values. Graph modes differ only in their fixed
adjacency buffer and use the same message-passing operator and parameters.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Literal

import torch
from torch import nn


Edge = tuple[str, str]
AdjacencyMode = Literal["topology", "identity", "shuffled"]
SHUFFLE_SEED = 20261019
SEASONAL_CHANNELS = ("same_time_yesterday_minus_ewma", "same_time_last_week_minus_ewma")


def _binary_line_graph(edge_order: Sequence[Edge]) -> torch.Tensor:
    edges = tuple(tuple(edge) for edge in edge_order)
    if not edges:
        raise ValueError("edge_order must contain at least one directed link")
    if any(len(edge) != 2 or not all(isinstance(node, str) and node for node in edge) for edge in edges):
        raise ValueError("Every directed link must contain two nonempty node identifiers")
    if any(source == destination for source, destination in edges):
        raise ValueError("Self-links are not supported")
    if len(set(edges)) != len(edges):
        raise ValueError("edge_order contains duplicate directed links")

    count = len(edges)
    adjacency = torch.eye(count, dtype=torch.float32)
    endpoints = [set(edge) for edge in edges]
    for receiving in range(count):
        for informing in range(count):
            if endpoints[receiving].intersection(endpoints[informing]):
                adjacency[receiving, informing] = 1.0
    return adjacency


def build_binary_link_adjacency(edge_order: Sequence[Edge]) -> torch.Tensor:
    """Return the unnormalized symmetric line graph, including self-loops."""
    return _binary_line_graph(edge_order)


def _validate_binary_adjacency(adjacency: torch.Tensor) -> torch.Tensor:
    matrix = torch.as_tensor(adjacency, dtype=torch.float32).detach().cpu()
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("adjacency must be a nonempty square matrix")
    if not torch.isfinite(matrix).all() or not torch.all((matrix == 0) | (matrix == 1)):
        raise ValueError("adjacency must contain only finite binary values")
    if not torch.equal(matrix, matrix.T):
        raise ValueError("adjacency must be symmetric for undirected edge swaps")
    if not torch.all(torch.diagonal(matrix) == 1):
        raise ValueError("adjacency must contain a self-loop for every link")
    return matrix.clone()


def degree_preserving_shuffle(
    adjacency: torch.Tensor,
    *,
    seed: int = SHUFFLE_SEED,
) -> torch.Tensor:
    """Deterministically rewire an undirected binary adjacency by edge swaps.

    The diagonal self-loops stay fixed. Undirected off-diagonal edges undergo
    double-edge swaps ``(a,b),(c,d) -> (a,d),(c,b)`` (or the alternate pairing)
    only when four distinct endpoints and absent replacement edges permit it.
    This preserves node set, symmetry, edge count, and each node's degree, but
    intentionally does not preserve link-endpoint meaning, communities, or
    higher-order motifs. At least one successful swap is required.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("shuffle seed must be an integer")
    result = _validate_binary_adjacency(adjacency)
    node_count = result.shape[0]
    edges = {
        (left, right)
        for left in range(node_count)
        for right in range(left + 1, node_count)
        if result[left, right] == 1
    }
    if len(edges) < 2:
        raise ValueError("adjacency has too few off-diagonal edges to rewire")

    original = result.clone()
    rng = random.Random(seed)
    target_swaps = len(edges)
    max_attempts = max(100, target_swaps * 100)
    successful_swaps = 0
    attempts = 0
    while successful_swaps < target_swaps and attempts < max_attempts:
        attempts += 1
        current_edges = sorted(edges)
        (a, b), (c, d) = rng.sample(current_edges, 2)
        if len({a, b, c, d}) != 4:
            continue
        if rng.randrange(2):
            proposed = ((min(a, d), max(a, d)), (min(c, b), max(c, b)))
        else:
            proposed = ((min(a, c), max(a, c)), (min(b, d), max(b, d)))
        if proposed[0] == proposed[1] or any(edge in edges for edge in proposed):
            continue
        edges.remove((a, b))
        edges.remove((c, d))
        edges.update(proposed)
        successful_swaps += 1

    if successful_swaps == 0:
        raise ValueError("no valid degree-preserving edge swap was possible")

    result.zero_()
    result.fill_diagonal_(1.0)
    for left, right in edges:
        result[left, right] = 1.0
        result[right, left] = 1.0
    if torch.equal(result, original):
        raise RuntimeError("edge-swap procedure completed without changing adjacency")
    if not torch.equal(result.sum(dim=1), original.sum(dim=1)):
        raise RuntimeError("edge-swap procedure changed the degree sequence")
    return result


def build_link_adjacency(
    edge_order: Sequence[Edge],
    *,
    mode: AdjacencyMode = "topology",
    shuffle_seed: int = SHUFFLE_SEED,
) -> torch.Tensor:
    """Build row-normalized adjacency; matrix index ``i`` remains edge ``i``.

    ``topology`` connects directed links whose endpoint sets intersect,
    including reverse directions. ``identity`` disables cross-link messages.
    ``shuffled`` uses seeded degree-preserving rewiring of the same undirected
    binary line graph. Every mode is normalized and consumed by the same model
    operator; no mode changes parameters or link/target ordering.
    """
    topology = _binary_line_graph(edge_order)
    if mode == "topology":
        selected = topology
    elif mode == "identity":
        selected = torch.eye(topology.shape[0], dtype=topology.dtype)
    elif mode == "shuffled":
        selected = degree_preserving_shuffle(topology, seed=shuffle_seed)
    else:
        raise ValueError(f"Unsupported adjacency mode: {mode!r}")
    return selected / selected.sum(dim=1, keepdim=True)


class SpatioTemporalLinkForecaster(nn.Module):
    """Forecast EWMA plus learned residuals at each requested horizon.

    ``history`` has shape ``[B, 672, E]`` by default. Seasonal residual
    channels are, for each horizon, ``u[t+h-96,e]-S[t,e]`` and
    ``u[t+h-672,e]-S[t,e]`` in that order, where ``S`` is the normalized
    alpha-0.8 EWMA over the most recent five observations. The source offsets
    are read from ``history`` only. If supplied, ``seasonal_features`` must
    equal these computed features and have shape ``[B,H,E,2]``; this prevents
    arbitrary future-derived covariates from entering the model.
    """

    def __init__(
        self,
        edge_order: Sequence[Edge],
        *,
        history_length: int = 672,
        horizons: Sequence[int] = (4, 96),
        hidden_size: int = 16,
        alpha: float = 0.8,
        ewma_steps: int = 5,
        include_seasonal_features: bool = True,
        adjacency_mode: AdjacencyMode = "topology",
        shuffle_seed: int = SHUFFLE_SEED,
    ) -> None:
        super().__init__()
        self.edge_order = tuple(tuple(edge) for edge in edge_order)
        if history_length < ewma_steps or ewma_steps <= 0:
            raise ValueError("history_length must be at least the positive ewma_steps")
        if not horizons or any(isinstance(h, bool) or not isinstance(h, int) or h <= 0 for h in horizons):
            raise ValueError("horizons must be a nonempty sequence of positive integers")
        if len(set(horizons)) != len(horizons):
            raise ValueError("horizons must not contain duplicates")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be strictly between zero and one")
        if not isinstance(include_seasonal_features, bool):
            raise ValueError("include_seasonal_features must be a bool")
        if adjacency_mode not in ("topology", "identity", "shuffled"):
            raise ValueError(f"Unsupported adjacency mode: {adjacency_mode!r}")
        if include_seasonal_features:
            minimum_source = min(
                history_length - 1 + horizon - seasonal_lag
                for horizon in horizons
                for seasonal_lag in (96, 672)
            )
            if minimum_source < 0:
                raise ValueError("history_length is too short for the requested seasonal sources")

        self.history_length = int(history_length)
        self.horizons = tuple(int(h) for h in horizons)
        self.link_count = len(self.edge_order)
        self.ewma_steps = int(ewma_steps)
        self.alpha = float(alpha)
        self.include_seasonal_features = include_seasonal_features
        self.adjacency_mode = adjacency_mode
        self.shuffle_seed = shuffle_seed

        raw_weights = torch.tensor(
            [alpha * (1.0 - alpha) ** (ewma_steps - 1 - k) for k in range(ewma_steps)],
            dtype=torch.float64,
        )
        self.register_buffer("ewma_weights", (raw_weights / raw_weights.sum()).to(torch.float32))
        self.register_buffer("topology_adjacency", build_link_adjacency(self.edge_order))
        self.register_buffer("identity_adjacency", build_link_adjacency(self.edge_order, mode="identity"))
        self.register_buffer(
            "shuffled_adjacency",
            build_link_adjacency(self.edge_order, mode="shuffled", shuffle_seed=shuffle_seed),
        )

        self.temporal_encoder = nn.GRU(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.seasonal_projection = nn.Linear(len(SEASONAL_CHANNELS), hidden_size, bias=False)
        self.self_projection = nn.Linear(hidden_size, hidden_size)
        self.message_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.residual_head = nn.Linear(hidden_size, len(self.horizons))
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def _ewma(self, history: torch.Tensor) -> torch.Tensor:
        recent = history[:, -self.ewma_steps :, :]
        return torch.einsum("bte,t->be", recent, self.ewma_weights)

    def seasonal_residual_features(self, history: torch.Tensor) -> torch.Tensor:
        """Derive ``[B,H,E,2]`` seasonal-minus-EWMA features from history only."""
        if history.ndim != 3 or history.shape[1] != self.history_length:
            raise ValueError("history has the wrong shape for seasonal feature construction")
        baseline = self._ewma(history)
        origin = self.history_length - 1
        per_horizon = []
        for horizon in self.horizons:
            yesterday_index = origin + horizon - 96
            last_week_index = origin + horizon - 672
            if min(yesterday_index, last_week_index) < 0 or max(yesterday_index, last_week_index) > origin:
                raise ValueError("seasonal source is outside the observed history")
            yesterday = history[:, yesterday_index, :] - baseline
            last_week = history[:, last_week_index, :] - baseline
            per_horizon.append(torch.stack((yesterday, last_week), dim=-1))
        return torch.stack(per_horizon, dim=1)

    def _adjacency(self, mode: AdjacencyMode) -> torch.Tensor:
        return {
            "topology": self.topology_adjacency,
            "identity": self.identity_adjacency,
            "shuffled": self.shuffled_adjacency,
        }[mode]

    def forward(
        self,
        history: torch.Tensor,
        *,
        seasonal_features: torch.Tensor | None = None,
        adjacency_mode: AdjacencyMode | None = None,
    ) -> torch.Tensor:
        """Return ``[B,H,E]`` forecasts; targets are never accepted as inputs."""
        if history.ndim != 3:
            raise ValueError("history must have shape [batch, history, directed_links]")
        batch, steps, links = history.shape
        if batch == 0 or links != self.link_count or steps != self.history_length:
            raise ValueError(
                f"Expected [B, {self.history_length}, {self.link_count}] with B>0, got {tuple(history.shape)}"
            )
        if not history.is_floating_point():
            raise TypeError("history must have a floating-point dtype")
        if not torch.isfinite(history).all():
            raise ValueError("history contains nonfinite utilization")

        if seasonal_features is not None and not self.include_seasonal_features:
            raise ValueError("seasonal_features supplied while seasonal features are disabled")
        computed_features = (
            self.seasonal_residual_features(history)
            if self.include_seasonal_features
            else None
        )
        if seasonal_features is not None:
            assert computed_features is not None
            if seasonal_features.shape != computed_features.shape:
                raise ValueError(
                    f"seasonal_features must have shape {tuple(computed_features.shape)}"
                )
            if not seasonal_features.is_floating_point() or not torch.isfinite(seasonal_features).all():
                raise ValueError("seasonal_features must be finite floating-point values")
            if not torch.allclose(
                seasonal_features.to(computed_features), computed_features, rtol=1e-5, atol=1e-7
            ):
                raise ValueError("seasonal_features must be derived from the supplied history")

        if self.include_seasonal_features:
            assert computed_features is not None
            selected_features = computed_features if seasonal_features is None else seasonal_features.to(history)
        else:
            selected_features = history.new_zeros(
                batch, len(self.horizons), links, len(SEASONAL_CHANNELS)
            )

        per_link_sequences = history.permute(0, 2, 1).reshape(batch * links, steps, 1)
        temporal, _ = self.temporal_encoder(per_link_sequences)
        states = temporal[:, -1, :].reshape(batch, links, -1)
        states = states.unsqueeze(1) + self.seasonal_projection(selected_features)

        mode = self.adjacency_mode if adjacency_mode is None else adjacency_mode
        if mode not in ("topology", "identity", "shuffled"):
            raise ValueError(f"Unsupported adjacency mode: {mode!r}")
        adjacency = self._adjacency(mode).to(device=states.device, dtype=states.dtype)
        neighbor_state = torch.matmul(adjacency, states)
        states = torch.relu(self.self_projection(states) + self.message_projection(neighbor_state))

        all_corrections = self.residual_head(states)
        corrections = all_corrections.diagonal(dim1=1, dim2=3).permute(0, 2, 1)
        return self._ewma(history).unsqueeze(1) + corrections


__all__ = [
    "AdjacencyMode",
    "Edge",
    "SEASONAL_CHANNELS",
    "SHUFFLE_SEED",
    "SpatioTemporalLinkForecaster",
    "build_binary_link_adjacency",
    "build_link_adjacency",
    "degree_preserving_shuffle",
]
