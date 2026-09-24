"""Causal, side-by-side Phase 7 routing evaluation over frozen simulator data."""

from __future__ import annotations

import copy
import json
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import yaml

from src.gnn.run_experiment import SCENARIOS
from src.routing.baseline_methods import ReactiveRouting, StaticShortestPath
from src.routing.forecast_adapter import FrozenQoSForecaster
from src.routing.ga_routing import GeneticRouter, TelemetryPathCostProvider
from src.simulator.network_simulator import NetworkSimulator
from src.topology.graph_builder import build_topology

METHODS = ("shortest_path", "reactive", "current_qos_ga", "predicted_qos_ga")


def _mean_std(values):
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()), "std": float(values.std())}


def summarize_realized(flow_rows, link_rows):
    """Summarize realized QoS; failed paths remain explicit, never imputed."""
    successful = [row for row in flow_rows if row["selected_path"]]
    return {
        "link_samples": len(link_rows), "flow_samples": len(flow_rows),
        "failed_flows": len(flow_rows) - len(successful),
        "failure_rate": float((len(flow_rows) - len(successful)) / len(flow_rows)) if flow_rows else 0.0,
        "mean_link_delay": float(np.mean([row["delay"] for row in link_rows])) if link_rows else 0.0,
        "mean_link_utilization": float(np.mean([row["utilization"] for row in link_rows])) if link_rows else 0.0,
        "mean_link_packet_loss": float(np.mean([row["packet_loss"] for row in link_rows])) if link_rows else 0.0,
        "mean_link_jitter": float(np.mean([row["jitter"] for row in link_rows])) if link_rows else 0.0,
        "mean_link_throughput": float(np.mean([row["throughput"] for row in link_rows])) if link_rows else 0.0,
        "mean_successful_flow_delay": float(np.mean([row["delay"] for row in successful])) if successful else None,
        "mean_flow_throughput": float(np.mean([row["throughput"] for row in flow_rows])) if flow_rows else 0.0,
    }


class Phase7Evaluator:
    """Run identical causal demands through isolated routing-policy simulators."""

    def __init__(self, config, checkpoint_dir, scaler_output_dir):
        self.config = config
        self.gnn_config = config["gnn_experiment"]
        self.phase7 = config["phase7"]
        self.checkpoint_dir = checkpoint_dir
        self.scaler_output_dir = scaler_output_dir

    def _simulation_config(self, scenario):
        keys = ("simulator_seed", "simulation_timesteps", "traffic_flows_per_timestep",
                "traffic_base_demand_mbps", "queue_capacity_mbps")
        result = {key.replace("simulator_", ""): self.gnn_config[key] for key in keys}
        result["scenario"] = scenario
        return result

    def _graph(self):
        return build_topology(num_nodes=int(self.gnn_config["num_nodes"]),
                              seed=int(self.gnn_config["topology_seed"]),
                              default_capacity_mbps=float(self.phase7.get("default_capacity_mbps", 350.0)))

    def _select_paths(self, method, graph, demands, prior_telemetry, forecast):
        if method == "shortest_path":
            router = StaticShortestPath()
            return [router.select_path(graph, item["source"], item["destination"], prior_telemetry) for item in demands]
        if method == "reactive":
            router = ReactiveRouting(self.config.get("reactive_routing_weights", {}))
            return [router.select_path(graph, item["source"], item["destination"], prior_telemetry) for item in demands]
        telemetry = forecast if method == "predicted_qos_ga" else prior_telemetry
        provider = TelemetryPathCostProvider(graph, telemetry)
        result = []
        for demand in demands:
            try:
                result.append(GeneticRouter(graph, provider, self.config["ga_routing"]).optimize(
                    demand["source"], demand["destination"], demand["demand_mbps"])["path"])
            except nx.NetworkXNoPath:
                result.append(None)
        return result

    def run_scenario(self, scenario):
        graph, sim_config = self._graph(), self._simulation_config(scenario)
        simulators = {method: NetworkSimulator(graph, copy.deepcopy(sim_config)) for method in METHODS}
        forecaster = FrozenQoSForecaster(
            scenario, self.gnn_config, self.checkpoint_dir,
            model_name=self.phase7["forecast_model"], training_seed=int(self.phase7["forecast_training_seed"]),
            scaler_output_dir=self.scaler_output_dir)
        history_length = forecaster.dataset.history_length
        evaluation_start = int(min(metadata["target_timestep"] for index in forecaster.dataset.indices("test")
                                   for _, _, metadata in [forecaster.dataset[index]]))
        records = {method: {"flows": [], "links": [], "history": [], "decision_log": []} for method in METHODS}
        demand_source = NetworkSimulator(graph, copy.deepcopy(sim_config))
        for timestep in range(int(sim_config["simulation_timesteps"])):
            demands = demand_source.traffic_generator.generate(timestep)
            for method, simulator in simulators.items():
                prior = simulator.previous_telemetry
                if timestep < history_length:
                    paths = self._select_paths("shortest_path", graph, demands, prior, None)
                    information = "warmup_static_past_only"
                else:
                    forecast = (forecaster.predict(records[method]["history"][-history_length:])
                                if method == "predicted_qos_ga" else None)
                    paths = self._select_paths(method, graph, demands, prior, forecast)
                    information = "forecast_from_completed_history" if forecast is not None else "completed_previous_timestep"
                flows, links = simulator.evaluate_selected_paths(timestep, demands, paths)
                records[method]["history"].append(links)
                if timestep >= evaluation_start:
                    records[method]["flows"].extend(flows)
                    records[method]["links"].extend(links)
                    records[method]["decision_log"].append({"timestep": timestep, "information": information,
                                                             "forecast_timestep": forecast[0]["timestep"] if method == "predicted_qos_ga" and forecast else None})
        return {"scenario": scenario, "evaluation_start_timestep": evaluation_start,
                "evaluation_timesteps": int(sim_config["simulation_timesteps"]) - evaluation_start,
                "methods": {method: summarize_realized(value["flows"], value["links"])
                            for method, value in records.items()},
                "causality": "All evaluated decisions use only telemetry completed before the decision timestep; realized QoS is computed after selected routes are applied."}

    def run(self, scenarios=SCENARIOS):
        scenario_records = [self.run_scenario(scenario) for scenario in scenarios]
        aggregate = {}
        for method in METHODS:
            aggregate[method] = {metric: _mean_std([row["methods"][method][metric] for row in scenario_records])
                                 for metric in ("mean_link_delay", "mean_link_utilization", "mean_link_packet_loss",
                                                "mean_link_jitter", "mean_link_throughput", "failure_rate")}
            aggregate[method]["link_samples"] = int(sum(row["methods"][method]["link_samples"] for row in scenario_records))
            aggregate[method]["flow_samples"] = int(sum(row["methods"][method]["flow_samples"] for row in scenario_records))
            aggregate[method]["failed_flows"] = int(sum(row["methods"][method]["failed_flows"] for row in scenario_records))
        return {"protocol": {"scenarios": list(scenarios), "methods": list(METHODS),
                              "forecast_model": self.phase7["forecast_model"],
                              "forecast_training_seed": int(self.phase7["forecast_training_seed"]),
                              "evaluation_window": "Phase 4.5 held-out chronological test target period",
                              "selection": "No retraining, tuning, or test-set model selection."},
                "scenario_results": scenario_records, "aggregate_across_scenarios": aggregate}


def synthetic_security_interface_check():
    """Controlled GA interface test; risks are synthetic and have no UNSW meaning."""
    graph = nx.DiGraph()
    graph.add_edges_from([(0, 1, {"capacity_mbps": 100, "base_delay_ms": 1}),
                          (1, 3, {"capacity_mbps": 100, "base_delay_ms": 1}),
                          (0, 2, {"capacity_mbps": 100, "base_delay_ms": 1}),
                          (2, 3, {"capacity_mbps": 100, "base_delay_ms": 1})])
    telemetry = [{"source": u, "destination": v, "capacity": 100, "traffic": 0,
                  "delay": 1, "utilization": 0, "packet_loss": 0, "jitter": 0} for u, v in graph.edges]
    risk = {(0, 1): .95, (1, 3): .95, (0, 2): .01, (2, 3): .01}
    router = GeneticRouter(graph, TelemetryPathCostProvider(graph, telemetry, risk),
                           {"random_seed": 42, "population_size": 4, "generations": 2, "elite_count": 1,
                            "objective_weights": {"delay": 0, "utilization": 0, "packet_loss": 0,
                                                  "jitter": 0, "security_risk": 1}})
    selected = router.optimize(0, 3, 1)["path"]
    return {"label": "Synthetic per-edge risk interface check; not UNSW-NB15 data or security-performance evidence.",
            "selected_path": selected, "expected_low_risk_path": [0, 2, 3], "passed": selected == [0, 2, 3]}


def main(config_path="configs/simulation.yaml", output_dir="data/experiments/phase7"):
    with open(config_path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    output = Path(output_dir)
    evaluator = Phase7Evaluator(config, config["phase7"]["checkpoint_dir"], output / "scalers")
    report = evaluator.run()
    report["synthetic_security_interface_check"] = synthetic_security_interface_check()
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output / "results.json"


if __name__ == "__main__":
    print(main())
