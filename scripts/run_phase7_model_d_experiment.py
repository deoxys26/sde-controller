"""Phase 7 evaluation extension for Model D.

Runs the same causal end-to-end routing experiment as Phase 7, but adds a
``predicted_qos_ga_model_d`` method that uses the frozen Model D checkpoint
from ``data/experiments/model_d/``.

IMPORTANT:
- This script does NOT overwrite the original Phase 7 results.
- It writes to ``data/experiments/phase7_model_d/``.
- Run this ONLY after confirming Model D Phase 4.5 results are complete.
- The original Shortest Path / Reactive / Current-QoS GA results remain the
  reference baseline.

Usage:
    python -m scripts.run_phase7_model_d_experiment
"""

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
from src.routing.phase7_evaluator import summarize_realized
from src.simulator.network_simulator import NetworkSimulator
from src.topology.graph_builder import build_topology

CONFIG_PATH = Path("configs/simulation.yaml")
OUTPUT_DIR = Path("data/experiments/phase7_model_d")
ORIGINAL_PHASE7_DIR = Path("data/experiments/phase7")
MODEL_D_CHECKPOINT_DIR = Path("data/experiments/model_d/checkpoints")

# The four original methods plus the Model D predicted GA.
METHODS = (
    "shortest_path",
    "reactive",
    "current_qos_ga",
    "predicted_qos_ga_current",   # Original: Model B seed 42
    "predicted_qos_ga_model_d",   # New: Model D seed 42
)


def _mean_std(values):
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()), "std": float(values.std())}


class Phase7ModelDEvaluator:
    """Like Phase7Evaluator but adds a Model D predicted-QoS GA method."""

    def __init__(self, config, current_checkpoint_dir, model_d_checkpoint_dir,
                 scaler_output_dir):
        self.config = config
        self.gnn_config = config["gnn_experiment"]
        self.phase7 = config["phase7"]
        self.current_ckpt_dir = current_checkpoint_dir
        self.model_d_ckpt_dir = model_d_checkpoint_dir
        self.scaler_output_dir = scaler_output_dir

    def _simulation_config(self, scenario):
        keys = ("simulator_seed", "simulation_timesteps", "traffic_flows_per_timestep",
                "traffic_base_demand_mbps", "queue_capacity_mbps")
        result = {key.replace("simulator_", ""): self.gnn_config[key] for key in keys}
        result["scenario"] = scenario
        return result

    def _graph(self):
        return build_topology(
            num_nodes=int(self.gnn_config["num_nodes"]),
            seed=int(self.gnn_config["topology_seed"]),
            default_capacity_mbps=float(self.phase7.get("default_capacity_mbps", 350.0)),
        )

    def _select_paths(self, method, graph, demands, prior, forecast):
        if method == "shortest_path":
            router = StaticShortestPath()
            return [router.select_path(graph, d["source"], d["destination"], prior)
                    for d in demands]
        if method == "reactive":
            router = ReactiveRouting(self.config.get("reactive_routing_weights", {}))
            return [router.select_path(graph, d["source"], d["destination"], prior)
                    for d in demands]
        telemetry = forecast if forecast is not None else prior
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
        simulators = {m: NetworkSimulator(graph, copy.deepcopy(sim_config)) for m in METHODS}

        # Load the two frozen forecasters (Model B and Model D), both seed 42.
        forecaster_current = FrozenQoSForecaster(
            scenario, self.gnn_config, self.current_ckpt_dir,
            model_name="current", training_seed=42,
            scaler_output_dir=self.scaler_output_dir / "current")
        forecaster_model_d = FrozenQoSForecaster(
            scenario, self.gnn_config, self.model_d_ckpt_dir,
            model_name="model_d", training_seed=42,
            scaler_output_dir=self.scaler_output_dir / "model_d")

        history_length = forecaster_current.dataset.history_length
        evaluation_start = int(min(
            metadata["target_timestep"]
            for index in forecaster_current.dataset.indices("test")
            for _, _, metadata in [forecaster_current.dataset[index]]
        ))

        records = {m: {"flows": [], "links": [], "history": [], "decision_log": []}
                   for m in METHODS}
        demand_source = NetworkSimulator(graph, copy.deepcopy(sim_config))

        for timestep in range(int(sim_config["simulation_timesteps"])):
            demands = demand_source.traffic_generator.generate(timestep)
            for method, simulator in simulators.items():
                prior = simulator.previous_telemetry
                if timestep < history_length:
                    paths = self._select_paths("shortest_path", graph, demands, prior, None)
                    information = "warmup_static_past_only"
                    forecast = None
                elif method == "predicted_qos_ga_current":
                    forecast = forecaster_current.predict(
                        records[method]["history"][-history_length:])
                    paths = self._select_paths(method, graph, demands, prior, forecast)
                    information = "forecast_from_completed_history"
                elif method == "predicted_qos_ga_model_d":
                    forecast = forecaster_model_d.predict(
                        records[method]["history"][-history_length:])
                    paths = self._select_paths(method, graph, demands, prior, forecast)
                    information = "forecast_from_completed_history"
                else:
                    forecast = None
                    paths = self._select_paths(method, graph, demands, prior, None)
                    information = "completed_previous_timestep"

                flows, links = simulator.evaluate_selected_paths(timestep, demands, paths)
                records[method]["history"].append(links)
                if timestep >= evaluation_start:
                    records[method]["flows"].extend(flows)
                    records[method]["links"].extend(links)
                    records[method]["decision_log"].append({
                        "timestep": timestep, "information": information,
                        "forecast_timestep": (
                            forecast[0]["timestep"]
                            if forecast and method.startswith("predicted_") else None),
                    })

        return {
            "scenario": scenario,
            "evaluation_start_timestep": evaluation_start,
            "evaluation_timesteps": int(sim_config["simulation_timesteps"]) - evaluation_start,
            "methods": {m: summarize_realized(records[m]["flows"], records[m]["links"])
                        for m in METHODS},
            "causality": "All evaluated decisions use only telemetry completed before the "
                         "decision timestep; realized QoS is computed after selected routes "
                         "are applied.",
        }

    def run(self, scenarios=SCENARIOS):
        scenario_records = [self.run_scenario(scenario) for scenario in scenarios]
        aggregate = {}
        for method in METHODS:
            aggregate[method] = {
                metric: _mean_std([row["methods"][method][metric]
                                   for row in scenario_records])
                for metric in ("mean_link_delay", "mean_link_utilization",
                                "mean_link_packet_loss", "mean_link_jitter",
                                "mean_link_throughput", "failure_rate")
            }
            aggregate[method]["link_samples"] = int(sum(
                row["methods"][method]["link_samples"] for row in scenario_records))
            aggregate[method]["flow_samples"] = int(sum(
                row["methods"][method]["flow_samples"] for row in scenario_records))
            aggregate[method]["failed_flows"] = int(sum(
                row["methods"][method]["failed_flows"] for row in scenario_records))
        return {
            "protocol": {
                "scenarios": list(scenarios),
                "methods": list(METHODS),
                "model_b_checkpoint_dir": str(self.current_ckpt_dir),
                "model_d_checkpoint_dir": str(self.model_d_ckpt_dir),
                "forecast_training_seed": 42,
                "note": ("predicted_qos_ga_current uses Model B (Phase 4.5 baseline). "
                         "predicted_qos_ga_model_d uses Model D (temporal attention). "
                         "Original Phase 7 results preserved in data/experiments/phase7/."),
            },
            "scenario_results": scenario_records,
            "aggregate_across_scenarios": aggregate,
        }


def main():
    if not MODEL_D_CHECKPOINT_DIR.exists():
        raise FileNotFoundError(
            f"Model D checkpoint directory not found: {MODEL_D_CHECKPOINT_DIR}\n"
            "Run run_model_d_experiment.py first and confirm results."
        )

    with open(CONFIG_PATH, encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    scaler_dir = OUTPUT_DIR / "scalers"
    scaler_dir.mkdir(parents=True, exist_ok=True)
    evaluator = Phase7ModelDEvaluator(
        config,
        current_checkpoint_dir=config["phase7"]["checkpoint_dir"],
        model_d_checkpoint_dir=str(MODEL_D_CHECKPOINT_DIR),
        scaler_output_dir=scaler_dir,
    )
    report = evaluator.run()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "results.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Written: {out}")
    return str(out)


if __name__ == "__main__":
    print(main())
