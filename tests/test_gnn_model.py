import torch

from src.gnn.dataset import TemporalGraphDataset
from src.gnn.model import EdgeQoSForecaster
from src.simulator.network_simulator import NetworkSimulator
from src.topology.graph_builder import build_topology


def test_model_forward_shape_and_finite_values():
    graph = build_topology(num_nodes=5, seed=5)
    _, links = NetworkSimulator(graph, {"seed": 5, "simulation_timesteps": 12, "traffic_flows_per_timestep": 4}).run()
    dataset = TemporalGraphDataset(links, graph, history_length=3)
    sequence, _, _ = dataset[0]
    model = EdgeQoSForecaster(len(dataset.feature_columns), len(dataset.target_columns), hidden_channels=8, gru_hidden_size=8)
    output = model(sequence)
    assert output.shape == (len(dataset.edges), len(dataset.target_columns))
    assert torch.isfinite(output).all()
