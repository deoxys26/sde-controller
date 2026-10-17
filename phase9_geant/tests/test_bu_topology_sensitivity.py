"""Synthetic tests for B-U graph topology sensitivity; no GEANT artifacts."""

from collections import Counter

import numpy as np
import torch

from phase9_geant.diagnose_bu_v2_topology_sensitivity import (
    _degree_counts,
    deterministic_degree_preserving_rewire,
)
from phase9_geant.train_bu_v2 import Normalization, build_edge_index, make_bu_sequence
from src.gnn.model import EdgeQoSForecaster


def _synthetic_topology():
    nodes = ("a", "b", "c", "d", "e", "f")
    links = (
        ("a", "b"), ("b", "c"), ("c", "d"),
        ("d", "e"), ("e", "f"), ("f", "a"),
        ("a", "d"), ("b", "e"), ("c", "f"),
    )
    return nodes, links


def test_rewire_is_deterministic_degree_preserving_and_bidirectional():
    nodes, links = _synthetic_topology()
    rewired_a = deterministic_degree_preserving_rewire(nodes, links, accepted_swaps=3)
    rewired_b = deterministic_degree_preserving_rewire(nodes, links, accepted_swaps=3)
    assert rewired_a == rewired_b
    assert set(map(frozenset, rewired_a)) != set(map(frozenset, links))
    assert _degree_counts(nodes, rewired_a) == _degree_counts(nodes, links)
    edge_index, node_order, directed = build_edge_index(nodes, rewired_a)
    assert set(node_order) == set(nodes)
    assert len(directed) == 2 * len(links)
    for index in range(0, len(directed), 2):
        assert directed[index] == directed[index + 1][::-1]
    assert edge_index.shape == (2, len(directed))


def test_real_bu_model_is_invariant_to_noop_and_degree_preserving_rewire():
    nodes, links = _synthetic_topology()
    original_index, node_order, original_edges = build_edge_index(nodes, links)
    rewired_links = deterministic_degree_preserving_rewire(nodes, links, accepted_swaps=3)
    rewired_index, rewired_node_order, rewired_edges = build_edge_index(nodes, rewired_links)
    assert rewired_node_order == node_order
    assert len(original_edges) == len(rewired_edges)
    assert original_edges != rewired_edges
    assert Counter(dst for _, dst in original_edges) == Counter(dst for _, dst in rewired_edges)
    assert Counter(src for src, _ in original_edges) == Counter(src for src, _ in rewired_edges)

    # Two synthetic samples make sure equality is not an artifact of a single
    # edge sequence. Their edge histories remain in the exact same column order.
    history = np.linspace(0.0, 1.0, num=2 * 5 * len(original_edges), dtype=np.float32)
    history = history.reshape(2, 5, len(original_edges))
    norm = Normalization(mean=0.5, scale=0.25)
    original_sequence = make_bu_sequence(history, original_index, len(node_order), norm)
    noop_sequence = make_bu_sequence(history, original_index.clone(), len(node_order), norm)
    rewired_sequence = make_bu_sequence(history, rewired_index, len(node_order), norm)
    for graph in original_sequence:
        assert torch.all(graph.x == 1)
        assert graph.edge_index.shape[1] == 2 * len(original_edges)

    torch.manual_seed(2026)
    model = EdgeQoSForecaster(edge_feature_count=1, target_count=1, hidden_channels=8, gru_hidden_size=7)
    model.eval()
    with torch.no_grad():
        original = model(original_sequence)
        noop = model(noop_sequence)
        rewired = model(rewired_sequence)

    assert original.shape == (2 * len(original_edges), 1)
    assert torch.isfinite(original).all()
    torch.testing.assert_close(noop, original, rtol=0.0, atol=0.0)
    torch.testing.assert_close(rewired, original, rtol=0.0, atol=1e-7)

