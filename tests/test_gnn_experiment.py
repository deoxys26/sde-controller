import pytest

from src.gnn.run_experiment import ABLATION_MODELS, MAIN_MODELS, SCENARIOS, build_model, summarize
from src.gnn.dataset import TemporalGraphDataset
from src.simulator.network_simulator import NetworkSimulator
from src.topology.graph_builder import build_topology


def test_experiment_model_selection_and_result_aggregation():
    graph = build_topology(num_nodes=5, seed=3)
    _, links = NetworkSimulator(graph, {"seed": 3, "simulation_timesteps": 12, "traffic_flows_per_timestep": 3}).run()
    dataset = TemporalGraphDataset(links, graph, history_length=4)
    for name in set(MAIN_MODELS + ABLATION_MODELS):
        assert build_model(name, dataset, {"hidden_channels": 8, "gru_hidden_size": 8, "dgst_mtl_hidden_channels": 8})
    records = [{"metrics": {"aggregate": {"mae": .2, "rmse": .3}}, "parameter_count": 10,
                "training_seconds": 1., "inference_seconds": .1},
               {"metrics": {"aggregate": {"mae": .4, "rmse": .5}}, "parameter_count": 10,
                "training_seconds": 3., "inference_seconds": .3}]
    summary = summarize(records)
    assert summary["aggregate"]["mae"]["mean"] == pytest.approx(.3)
    assert summary["aggregate"]["mae"]["std"] == pytest.approx(.1)
    assert len(SCENARIOS) == 10
