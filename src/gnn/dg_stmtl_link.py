"""Single-target, edge-centric adaptation of the DG-STMTL architecture.

This is not a literal reproduction of DG-STMTL. It retains its hybrid prior /
input-conditioned adjacency, multiplicative task gate, and two-stage grouped
spatio-temporal graph convolution. GEANT supplies one link-utilization target,
so cross-task exchange and multi-task loss aggregation are intentionally absent.
See ``docs/geant_dg_stmtl_h96_v1.md`` for the adaptation contract.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
import torch.nn.functional as F

from src.gnn.spatiotemporal_link import Edge, build_binary_link_adjacency


GROUP_WIDTH = 3
TOKEN_COUNT = 12
HISTORY_LENGTH = 672
PATCH_LENGTH = HISTORY_LENGTH // TOKEN_COUNT


def build_hybrid_prior(edge_order: Sequence[Edge]) -> torch.Tensor:
    """Construct the documented binary 3E-node synchronous prior.

    Node order is ``(temporal_slot, caller_edge_index)``. The prior is the
    union of: (1) same-slot line-graph neighbors, (2) same directed link in
    adjacent temporal slots, and (3) line-graph neighbors across adjacent
    temporal slots. Self-loops are retained. This is a transparent adaptation
    of the paper's unspecified ``F(A_S, A_T, A_ST)`` construction.
    """
    line = build_binary_link_adjacency(edge_order).to(torch.float32)
    links = int(line.shape[0])
    eye_links = torch.eye(links, dtype=torch.float32)
    time_eye = torch.eye(GROUP_WIDTH, dtype=torch.float32)
    adjacent_time = torch.zeros_like(time_eye)
    for slot in range(GROUP_WIDTH - 1):
        adjacent_time[slot, slot + 1] = 1.0
        adjacent_time[slot + 1, slot] = 1.0
    same_slot_spatial = torch.kron(time_eye, line)
    adjacent_slot_temporal = torch.kron(adjacent_time, eye_links)
    adjacent_slot_spatiotemporal = torch.kron(adjacent_time, line - eye_links)
    prior = (
        same_slot_spatial + adjacent_slot_temporal + adjacent_slot_spatiotemporal
    ).gt(0).to(torch.float32)
    prior.fill_diagonal_(1.0)
    if prior.shape != (GROUP_WIDTH * links, GROUP_WIDTH * links):
        raise RuntimeError("hybrid prior has an unexpected node ordering or shape")
    if not torch.equal(prior, prior.T):
        raise RuntimeError("hybrid prior must be symmetric")
    return prior


class _GroupGraphBlock(nn.Module):
    """Three GCN-AGG layers plus normalized dense residual fusion."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.self_layers = nn.ModuleList(nn.Linear(hidden_size, hidden_size) for _ in range(3))
        self.message_layers = nn.ModuleList(
            nn.Linear(hidden_size, hidden_size, bias=False) for _ in range(3)
        )
        self.fusion_logits = nn.Parameter(torch.zeros(4))

    def forward(self, grouped: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        # grouped: [B, G, 3E, C]; each sample has its own dynamic adjacency.
        if grouped.ndim != 4 or adjacency.ndim != 3:
            raise ValueError("group graph expects [B,G,3E,C] and [B,3E,3E]")
        batch, groups, vertices, _channels = grouped.shape
        if adjacency.shape != (batch, vertices, vertices):
            raise ValueError("adjacency and graph-batch dimensions do not align")
        states = [grouped]
        current = grouped
        for self_layer, message_layer in zip(self.self_layers, self.message_layers):
            # Row j aggregates source i using A[j,i]; batch dimensions never mix.
            neighbors = torch.einsum("bji,bgih->bgjh", adjacency, current)
            current = F.relu(self_layer(current) + message_layer(neighbors))
            states.append(current)
        weights = torch.softmax(self.fusion_logits, dim=0)
        fused = sum(weight * state for weight, state in zip(weights, states))
        return fused.reshape(batch, groups, GROUP_WIDTH, vertices // GROUP_WIDTH, -1)[:, :, 1]


class DGSTMTLLinkForecaster(nn.Module):
    """Forecast the h=96 link target as alpha-0.8 EWMA plus one residual.

    Input is only ``history: [B,672,E]`` in caller-supplied directed-link
    order. Twelve learned, non-overlapping 56-sample temporal patches feed
    temporal grouping (four adjacent 3-token groups), then a second overlapping
    feature grouping (groups 1:3 and 2:4). The input-conditioned adjacency is
    generated independently for each sample from the input representation.
    No target tensor is accepted by ``forward``.
    """

    def __init__(
        self,
        edge_order: Sequence[Edge],
        *,
        history_length: int = HISTORY_LENGTH,
        token_count: int = TOKEN_COUNT,
        hidden_size: int = 16,
        dynamic_dim: int = 8,
        alpha: float = 0.8,
        ewma_steps: int = 5,
    ) -> None:
        super().__init__()
        self.edge_order = tuple(tuple(edge) for edge in edge_order)
        if not self.edge_order or len(set(self.edge_order)) != len(self.edge_order):
            raise ValueError("edge_order must be nonempty and unique")
        if history_length <= 0 or token_count != TOKEN_COUNT or history_length % token_count:
            raise ValueError("history must divide into exactly twelve temporal patches")
        if hidden_size <= 0 or dynamic_dim <= 0 or ewma_steps <= 0:
            raise ValueError("hidden_size, dynamic_dim, and ewma_steps must be positive")
        if history_length < ewma_steps or not 0.0 < alpha < 1.0:
            raise ValueError("invalid EWMA history or alpha")
        self.history_length = int(history_length)
        self.token_count = int(token_count)
        self.link_count = len(self.edge_order)
        self.patch_length = history_length // token_count
        self.hidden_size = int(hidden_size)
        self.ewma_steps = int(ewma_steps)
        self.alpha = float(alpha)

        weights = torch.tensor(
            [alpha * (1.0 - alpha) ** (ewma_steps - 1 - i) for i in range(ewma_steps)],
            dtype=torch.float64,
        )
        self.register_buffer("ewma_weights", (weights / weights.sum()).to(torch.float32))
        self.register_buffer("static_prior", build_hybrid_prior(self.edge_order))
        self.temporal_patch = nn.Conv1d(
            1, hidden_size, kernel_size=self.patch_length, stride=self.patch_length
        )
        self.temporal_embedding = nn.Parameter(torch.zeros(1, token_count, 1, hidden_size))
        self.link_embedding = nn.Parameter(torch.zeros(1, 1, self.link_count, hidden_size))
        self.dynamic_projection = nn.Linear(hidden_size, GROUP_WIDTH * dynamic_dim)
        # Single available target => one task-specific gating matrix.
        vertex_count = GROUP_WIDTH * self.link_count
        self.task_gate_logits = nn.Parameter(torch.zeros(vertex_count, vertex_count))
        self.temporal_group_block = _GroupGraphBlock(hidden_size)
        self.feature_group_block = _GroupGraphBlock(hidden_size)
        self.residual_head = nn.Linear(hidden_size, 1)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def ewma(self, history: torch.Tensor) -> torch.Tensor:
        if history.ndim != 3 or history.shape[1:] != (self.history_length, self.link_count):
            raise ValueError("history shape does not match the model contract")
        return torch.einsum("bte,t->be", history[:, -self.ewma_steps :, :], self.ewma_weights)

    def _input_tokens(self, history: torch.Tensor) -> torch.Tensor:
        batch, steps, links = history.shape
        per_link = history.permute(0, 2, 1).reshape(batch * links, 1, steps)
        encoded = self.temporal_patch(per_link).transpose(1, 2)
        if encoded.shape[1] != self.token_count:
            raise RuntimeError("temporal patch encoder emitted an unexpected token count")
        encoded = encoded.reshape(batch, links, self.token_count, self.hidden_size).permute(0, 2, 1, 3)
        return encoded + self.temporal_embedding + self.link_embedding

    def dynamic_adjacency(self, tokens: torch.Tensor) -> torch.Tensor:
        """Input-only CTKE-style correlation adjacency for each batch sample."""
        pooled = tokens.max(dim=1).values  # [B,E,C], no target or future inputs
        projected = F.relu(self.dynamic_projection(pooled))
        batch, links, _ = projected.shape
        dynamic_dim = projected.shape[-1] // GROUP_WIDTH
        projected = projected.reshape(batch, links, GROUP_WIDTH, dynamic_dim)
        projected = projected.permute(0, 2, 1, 3).reshape(batch, GROUP_WIDTH * links, dynamic_dim)
        similarity = torch.bmm(projected, projected.transpose(1, 2)) / (dynamic_dim ** 0.5)
        return torch.softmax(similarity, dim=-1)

    def hybrid_adjacency(self, tokens: torch.Tensor) -> torch.Tensor:
        dynamic = self.dynamic_adjacency(tokens)
        gate = torch.sigmoid(self.task_gate_logits).unsqueeze(0)
        hybrid = gate * (self.static_prior.unsqueeze(0) + dynamic)
        row_sums = hybrid.sum(dim=-1, keepdim=True)
        if not torch.isfinite(hybrid).all() or torch.any(row_sums <= 0):
            raise ValueError("hybrid adjacency is nonfinite or has an empty row")
        return hybrid / row_sums

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        if history.ndim != 3 or history.shape[1:] != (self.history_length, self.link_count):
            raise ValueError(
                f"expected [B,{self.history_length},{self.link_count}], got {tuple(history.shape)}"
            )
        if history.shape[0] == 0 or not history.is_floating_point() or not torch.isfinite(history).all():
            raise ValueError("history must be a nonempty finite floating-point tensor")
        tokens = self._input_tokens(history)
        adjacency = self.hybrid_adjacency(tokens)
        batch = history.shape[0]
        temporal_groups = tokens.reshape(batch, 4, GROUP_WIDTH, self.link_count, self.hidden_size)
        temporal_groups = temporal_groups.reshape(batch, 4, GROUP_WIDTH * self.link_count, self.hidden_size)
        centered_groups = self.temporal_group_block(temporal_groups, adjacency)  # [B,4,E,C]
        feature_groups = torch.stack((centered_groups[:, :3], centered_groups[:, 1:]), dim=1)
        feature_groups = feature_groups.reshape(batch, 2, GROUP_WIDTH * self.link_count, self.hidden_size)
        feature_output = self.feature_group_block(feature_groups, adjacency)  # [B,2,E,C]
        link_state = feature_output.max(dim=1).values
        correction = self.residual_head(link_state).squeeze(-1)
        return self.ewma(history) + correction


__all__ = ["DGSTMTLLinkForecaster", "GROUP_WIDTH", "HISTORY_LENGTH", "TOKEN_COUNT", "build_hybrid_prior"]
