"""A compact GCN + GRU baseline that predicts QoS for each directed link."""

from __future__ import annotations

import torch
from torch import nn


class DirectedGraphConv(nn.Module):
    """Mean aggregation over directed incoming neighbours, with a self transform."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.self_linear = nn.Linear(in_channels, out_channels)
        self.neighbor_linear = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x, edge_index):
        source, destination = edge_index
        messages = x[source]
        aggregated = torch.zeros_like(x).index_add(0, destination, messages)
        degrees = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
        degrees.index_add_(0, destination, torch.ones_like(destination, dtype=x.dtype))
        return self.self_linear(x) + self.neighbor_linear(aggregated / degrees.clamp(min=1).unsqueeze(1))


class EdgeQoSForecaster(nn.Module):
    def __init__(self, edge_feature_count, target_count, hidden_channels=16, gru_hidden_size=16):
        super().__init__()
        self.gcn1 = DirectedGraphConv(1, hidden_channels)
        self.gcn2 = DirectedGraphConv(hidden_channels, hidden_channels)
        self.edge_encoder = nn.Sequential(nn.Linear(hidden_channels * 2 + edge_feature_count, hidden_channels), nn.ReLU())
        self.gru = nn.GRU(hidden_channels, gru_hidden_size, batch_first=True)
        self.head = nn.Linear(gru_hidden_size, target_count)

    def forward(self, sequence):
        encoded = []
        for graph in sequence:
            node_state = torch.relu(self.gcn1(graph.x, graph.edge_index))
            node_state = torch.relu(self.gcn2(node_state, graph.edge_index))
            source, destination = graph.edge_index
            edge_state = torch.cat((node_state[source], node_state[destination], graph.edge_attr), dim=1)
            encoded.append(self.edge_encoder(edge_state))
        temporal_edges = torch.stack(encoded, dim=1)  # [directed links, history, embedding]
        output, _ = self.gru(temporal_edges)
        return self.head(output[:, -1, :])


def _link_adjacency(edge_index, directed):
    """Build a static link graph from directed topology edges at runtime."""
    source, destination = edge_index
    if directed:
        # Row is the receiving link and column is the informing link:
        # (u,v) -> (v,w), so source(receiving) == destination(informing).
        adjacent = source[:, None].eq(destination[None, :])
    else:
        endpoints = torch.stack((source, destination), dim=1)
        adjacent = (endpoints[:, None, :, None] == endpoints[None, :, None, :]).any(dim=(2, 3))
    adjacent = adjacent.float() + torch.eye(len(source), device=edge_index.device)
    return adjacent / adjacent.sum(dim=1, keepdim=True).clamp(min=1)


class DGSTMTLReferenceForecaster(nn.Module):
    """DG-STMTL-inspired reference adapted from road nodes to SDN links.

    It retains multi-task task encoders, dense input-derived adjacency, task
    gates, three graph-convolution residual stages, and a shared output head.
    The paper's 3N synchronous adjacency is adapted because our targets live on
    directed links, not road-detector nodes.
    """
    def __init__(self, edge_feature_count, target_count, hidden_channels=16):
        super().__init__()
        self.target_count = target_count
        self.task_inputs = nn.ModuleList([nn.Linear(edge_feature_count, hidden_channels) for _ in range(target_count)])
        self.query = nn.ModuleList([nn.Linear(hidden_channels, hidden_channels, bias=False) for _ in range(target_count)])
        self.key = nn.ModuleList([nn.Linear(hidden_channels, hidden_channels, bias=False) for _ in range(target_count)])
        self.gate = nn.ModuleList([nn.Linear(hidden_channels, hidden_channels, bias=False) for _ in range(target_count)])
        self.layers = nn.ModuleList([nn.ModuleList([nn.Linear(hidden_channels, hidden_channels) for _ in range(3)]) for _ in range(target_count)])
        self.task_weights = nn.Parameter(torch.ones(target_count, 4))
        self.head = nn.Sequential(nn.Linear(hidden_channels * target_count, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, target_count))

    def _hybrid_adjacency(self, representation, static, task):
        q, k = self.query[task](representation), self.key[task](representation)
        dynamic = torch.softmax(q @ k.T / (representation.size(1) ** .5), dim=1)
        gate = torch.sigmoid(self.gate[task](representation) @ representation.T / (representation.size(1) ** .5))
        hybrid = gate * (static + dynamic)
        return hybrid / hybrid.sum(dim=1, keepdim=True).clamp(min=1e-6)

    def forward(self, sequence):
        # The reference paper groups triples; pooling all complete/partial groups
        # preserves that short-window principle for configurable histories.
        static = _link_adjacency(sequence[-1].edge_index, directed=False)
        task_outputs = []
        for task in range(self.target_count):
            encoded = [torch.relu(self.task_inputs[task](snapshot.edge_attr)) for snapshot in sequence]
            groups = [torch.stack(encoded[start:start + 3]).mean(0) for start in range(0, len(encoded), 3)]
            group_outputs = []
            for state in groups:
                adjacency = self._hybrid_adjacency(state, static, task)
                states = [state]
                for layer in self.layers[task]:
                    states.append(torch.relu(adjacency @ states[-1] @ layer.weight.T + layer.bias))
                weights = torch.softmax(self.task_weights[task], dim=0)
                group_outputs.append(sum(weight * value for weight, value in zip(weights, states)))
            task_outputs.append(torch.stack(group_outputs).amax(0))
        return self.head(torch.cat(task_outputs, dim=1))


class DirectedCongestionDGSTMTLForecaster(DGSTMTLReferenceForecaster):
    """Proposed model: dynamic links are limited to directed path continuity.

    Dynamic correlation is scaled by recent utilization/queue congestion similarity,
    rather than allowing every link pair to exchange state indiscriminately.
    """
    def _hybrid_adjacency(self, representation, static, task):
        q, k = self.query[task](representation), self.key[task](representation)
        dynamic = torch.softmax(q @ k.T / (representation.size(1) ** .5), dim=1)
        gate = torch.sigmoid(self.gate[task](representation) @ representation.T / (representation.size(1) ** .5))
        hybrid = gate * (static + dynamic * static)
        return hybrid / hybrid.sum(dim=1, keepdim=True).clamp(min=1e-6)

    def _forward_with_prior(self, sequence, directed, use_congestion):
        """Shared ablation implementation; output dimensions stay unchanged."""
        static = _link_adjacency(sequence[-1].edge_index, directed=directed)
        if use_congestion:
            # Feature positions 2/3 are documented utilization and queue length.
            congestion = sequence[-1].edge_attr[:, 2:4]
            similarity = torch.exp(-torch.cdist(congestion, congestion, p=1))
            static = static * (1 + similarity)
            static = static / static.sum(dim=1, keepdim=True).clamp(min=1e-6)
        task_outputs = []
        for task in range(self.target_count):
            encoded = [torch.relu(self.task_inputs[task](snapshot.edge_attr)) for snapshot in sequence]
            groups = [torch.stack(encoded[start:start + 3]).mean(0) for start in range(0, len(encoded), 3)]
            group_outputs = []
            for state in groups:
                adjacency = self._hybrid_adjacency(state, static, task)
                states = [state]
                for layer in self.layers[task]:
                    states.append(torch.relu(adjacency @ states[-1] @ layer.weight.T + layer.bias))
                weights = torch.softmax(self.task_weights[task], dim=0)
                group_outputs.append(sum(weight * value for weight, value in zip(weights, states)))
            task_outputs.append(torch.stack(group_outputs).amax(0))
        return self.head(torch.cat(task_outputs, dim=1))

    def forward(self, sequence):
        # (u,v) informs downstream (v,w), then congestion weights the prior.
        return self._forward_with_prior(sequence, directed=True, use_congestion=True)


class DirectedOnlyDGSTMTLForecaster(DirectedCongestionDGSTMTLForecaster):
    """Ablation: directed route continuity without congestion conditioning."""
    def forward(self, sequence):
        return self._forward_with_prior(sequence, directed=True, use_congestion=False)


class CongestionOnlyDGSTMTLForecaster(DirectedCongestionDGSTMTLForecaster):
    """Ablation: congestion-conditioned adjacency without directed continuity."""
    def forward(self, sequence):
        return self._forward_with_prior(sequence, directed=False, use_congestion=True)
