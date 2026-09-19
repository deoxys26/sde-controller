import numpy as np

from src.gnn.dataset import TemporalGraphDataset
from src.simulator.network_simulator import NetworkSimulator
from src.topology.graph_builder import build_topology


def _dataset():
    graph = build_topology(num_nodes=6, seed=11, default_capacity_mbps=250)
    _, links = NetworkSimulator(graph, {"seed": 11, "simulation_timesteps": 32, "scenario": "increasing_load",
                                        "traffic_flows_per_timestep": 6, "traffic_base_demand_mbps": 120}).run()
    return TemporalGraphDataset(links, graph, history_length=4, prediction_horizon=1), links


def test_snapshots_windows_splits_and_temporal_ordering():
    dataset, _ = _dataset()
    sequence, target, metadata = dataset[0]
    assert len(sequence) == 4
    assert sequence[0].edge_attr.shape == (len(dataset.edges), len(dataset.feature_columns))
    assert target.shape == (len(dataset.edges), len(dataset.target_columns))
    assert max(metadata["history_timesteps"]) < metadata["target_timestep"]
    assert dataset.indices("train") and dataset.indices("validation") and dataset.indices("test")
    for index in range(len(dataset)):
        _, _, meta = dataset[index]
        assert max(meta["history_timesteps"]) < meta["target_timestep"]


def test_scaler_uses_training_period_only_and_future_does_not_change_inputs():
    dataset, links = _dataset()
    train_rows = links[links.timestep <= dataset.train_end_timestep][list(dataset.feature_columns)].to_numpy()
    np.testing.assert_allclose(dataset.scaler.mean_, train_rows.mean(axis=0), rtol=1e-6)
    future = links.copy()
    future.loc[future.timestep > dataset.train_end_timestep, "delay"] += 1_000_000
    changed = TemporalGraphDataset(future, dataset.graph, history_length=4, prediction_horizon=1)
    original_index = dataset.indices("train")[0]
    for before, after in zip(dataset[original_index][0], changed[original_index][0]):
        assert before.edge_attr.equal(after.edge_attr)
