import torch

from src.gnn.dataset import TemporalGraphDataset
from src.gnn.model import (CongestionOnlyDGSTMTLForecaster, DGSTMTLReferenceForecaster,
                           DirectedCongestionDGSTMTLForecaster, DirectedOnlyDGSTMTLForecaster,
                           _link_adjacency)
from src.simulator.network_simulator import NetworkSimulator
from src.topology.graph_builder import build_topology


def test_reference_and_proposed_models_emit_finite_link_targets():
    graph = build_topology(num_nodes=5, seed=4)
    _, links = NetworkSimulator(graph, {"seed": 4, "simulation_timesteps": 12, "traffic_flows_per_timestep": 4}).run()
    dataset = TemporalGraphDataset(links, graph, history_length=5)
    sequence, target, _ = dataset[0]
    for model_class in (DGSTMTLReferenceForecaster, DirectedOnlyDGSTMTLForecaster,
                        CongestionOnlyDGSTMTLForecaster, DirectedCongestionDGSTMTLForecaster):
        output = model_class(len(dataset.feature_columns), len(dataset.target_columns), hidden_channels=8)(sequence)
        assert output.shape == target.shape
        assert torch.isfinite(output).all()


def test_proposed_link_prior_follows_directed_route_continuity():
    # (0, 1) informs (1, 2), while unrelated (3, 4) remains disconnected.
    edges = torch.tensor([[0, 1, 3], [1, 2, 4]])
    adjacency = _link_adjacency(edges, directed=True)
    assert adjacency[1, 0] > 0
    assert adjacency[0, 1] == 0
    assert adjacency[2, 0] == 0
