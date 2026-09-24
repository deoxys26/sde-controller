import copy

import networkx as nx

from src.routing.phase7_evaluator import Phase7Evaluator, summarize_realized, synthetic_security_interface_check


class _FakeDataset:
    history_length = 2

    def indices(self, split):
        return [0] if split == "test" else []

    def __getitem__(self, index):
        return None, None, {"target_timestep": 3}


class _CausalFakeForecaster:
    calls = []

    def __init__(self, *args, **kwargs):
        self.dataset = _FakeDataset()

    def predict(self, history):
        self.calls.append([row["timestep"] for rows in history for row in rows])
        # Preserve the latest snapshot, as a frozen one-step forecaster would.
        return [dict(row, timestep=row["timestep"] + 1) for row in history[-1]]


def _config():
    return {"ga_routing": {"random_seed": 42, "population_size": 4, "generations": 1,
                             "mutation_rate": .2, "crossover_rate": .8, "elite_count": 1,
                             "tournament_size": 2, "max_path_length": 5, "objective_weights": {}},
            "reactive_routing_weights": {},
            "gnn_experiment": {"simulator_seed": 42, "topology_seed": 42, "num_nodes": 5,
                               "simulation_timesteps": 5, "traffic_flows_per_timestep": 3,
                               "traffic_base_demand_mbps": 20, "queue_capacity_mbps": 100,
                               "history_length": 2, "prediction_horizon": 1},
            "phase7": {"forecast_model": "current", "forecast_training_seed": 42,
                       "default_capacity_mbps": 350}}


def test_evaluator_is_causal_fair_and_deterministic(monkeypatch, tmp_path):
    monkeypatch.setattr("src.routing.phase7_evaluator.FrozenQoSForecaster", _CausalFakeForecaster)
    _CausalFakeForecaster.calls = []
    first = Phase7Evaluator(_config(), tmp_path, tmp_path).run(["normal"])
    second = Phase7Evaluator(_config(), tmp_path, tmp_path).run(["normal"])
    methods = first["scenario_results"][0]["methods"]
    assert set(methods) == {"shortest_path", "reactive", "current_qos_ga", "predicted_qos_ga"}
    assert len({item["flow_samples"] for item in methods.values()}) == 1
    assert len({item["link_samples"] for item in methods.values()}) == 1
    assert first == second
    # Every forecast receives exactly the two contiguous, already completed snapshots.
    assert all(len(set(times)) == 2 and max(times) - min(times) == 1
               for times in _CausalFakeForecaster.calls)
    assert first["scenario_results"][0]["evaluation_start_timestep"] == 3


def test_no_path_is_accounted_for_without_crashing(tmp_path):
    evaluator = Phase7Evaluator(_config(), tmp_path, tmp_path)
    graph = nx.DiGraph()
    graph.add_nodes_from([0, 1])
    paths = evaluator._select_paths("current_qos_ga", graph, [{"source": 0, "destination": 1, "demand_mbps": 1}], [], None)
    assert paths == [None]
    summary = summarize_realized([{"selected_path": None, "throughput": 0}], [])
    assert summary["failed_flows"] == 1 and summary["failure_rate"] == 1


def test_synthetic_security_check_is_explicitly_separate_and_passes():
    result = synthetic_security_interface_check()
    assert result["passed"]
    assert "not UNSW-NB15" in result["label"]
