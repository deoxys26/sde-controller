"""Reproducible Phase 4.5 multi-seed QoS experiment runner.

Example: python -m src.gnn.run_experiment --epochs 10 --seeds 42 123 2026
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml

from src.gnn.dataset import TemporalGraphDataset
from src.gnn.evaluate import evaluate_model
from src.gnn.model import (CongestionOnlyDGSTMTLForecaster, DirectedCongestionDGSTMTLForecaster,
                           DirectedOnlyDGSTMTLForecaster, DGSTMTLReferenceForecaster, EdgeQoSForecaster,
                           TemporalAttentionEdgeForecaster)
from src.gnn.train import train_model
from src.topology.graph_builder import build_topology
from src.simulator.network_simulator import NetworkSimulator

SCENARIOS = ("normal", "low_load", "medium_load", "high_load", "increasing_load", "congestion",
             "traffic_spike", "link_degradation", "recovery", "anomalous_traffic")
MAIN_MODELS = ("reference", "current", "proposed")
ABLATION_MODELS = ("reference", "directed_only", "congestion_only", "proposed")
# Model D comparison: B (current), C (proposed), D (model_d).
MODEL_D_MODELS = ("current", "proposed", "model_d")


def build_model(name, dataset, config):
    features, targets = len(dataset.feature_columns), len(dataset.target_columns)
    hidden = int(config.get("hidden_channels", 16))
    if name == "current":
        return EdgeQoSForecaster(features, targets, hidden_channels=hidden,
                                 gru_hidden_size=int(config.get("gru_hidden_size", hidden)))
    if name == "model_d":
        return TemporalAttentionEdgeForecaster(features, targets, hidden_channels=hidden,
                                              gru_hidden_size=int(config.get("gru_hidden_size", hidden)))
    classes = {"reference": DGSTMTLReferenceForecaster,
               "directed_only": DirectedOnlyDGSTMTLForecaster,
               "congestion_only": CongestionOnlyDGSTMTLForecaster,
               "proposed": DirectedCongestionDGSTMTLForecaster}
    if name not in classes:
        raise ValueError(f"Unknown model: {name}")
    return classes[name](features, targets, hidden_channels=int(config.get("dgst_mtl_hidden_channels", hidden)))


def summarize(records):
    """Mean and population standard deviation for records with identical metric keys."""
    summary = {}
    for target in records[0]["metrics"]:
        summary[target] = {}
        for metric in ("mae", "rmse"):
            values = np.array([record["metrics"][target][metric] for record in records], dtype=float)
            summary[target][metric] = {"mean": float(values.mean()), "std": float(values.std())}
    for field in ("parameter_count", "training_seconds", "inference_seconds"):
        values = np.array([record[field] for record in records if record[field] is not None], dtype=float)
        summary[field] = ({"mean": float(values.mean()), "std": float(values.std())}
                          if len(values) else {"mean": None, "std": None})
    return summary


def _dataset_for_scenario(scenario, config):
    simulation = {"seed": int(config["simulator_seed"]), "simulation_timesteps": int(config["simulation_timesteps"]),
                  "scenario": scenario, "traffic_flows_per_timestep": int(config["traffic_flows_per_timestep"]),
                  "traffic_base_demand_mbps": float(config["traffic_base_demand_mbps"]),
                  "queue_capacity_mbps": float(config["queue_capacity_mbps"])}
    graph = build_topology(num_nodes=int(config["num_nodes"]), seed=int(config["topology_seed"]),
                           default_capacity_mbps=float(config.get("default_capacity_mbps", 350.0)))
    _, links = NetworkSimulator(graph, simulation).run()
    return TemporalGraphDataset(links, graph, history_length=int(config["history_length"]),
                                prediction_horizon=int(config["prediction_horizon"]))


def run_configuration(scenario, model_name, training_seed, config, output_dir):
    dataset = _dataset_for_scenario(scenario, config)
    model = build_model(model_name, dataset, config)
    checkpoint = Path(output_dir) / "checkpoints" / f"{scenario}_{model_name}_{training_seed}.pt"
    training = {"seed": int(training_seed), "epochs": int(config["epochs"]),
                "early_stopping_patience": config.get("early_stopping_patience"),
                "learning_rate": float(config["learning_rate"]), "batch_size": int(config["batch_size"]),
                "checkpoint_path": str(checkpoint)}
    if checkpoint.exists():
        # The interrupted study has model weights but no aggregate artifact.
        # Recover by test evaluation only; never retrain completed tuples.
        model.load_state_dict(torch.load(checkpoint, weights_only=True))
        started = time.perf_counter()
        metrics, _ = evaluate_model(model, dataset, dataset.indices("test"))
        inference_seconds = time.perf_counter() - started
        return {"scenario": scenario, "model": model_name, "training_seed": int(training_seed),
                "metrics": metrics, "parameter_count": int(sum(p.numel() for p in model.parameters())),
                "training_seconds": None, "inference_seconds": inference_seconds,
                "best_validation_epoch": None, "history": [], "recovered_checkpoint": True}
    started = time.perf_counter()
    history, metrics, _ = train_model(model, dataset, training)
    training_seconds = time.perf_counter() - started
    test_indices = dataset.indices("test")
    started = time.perf_counter()
    with torch.no_grad():
        model.eval()
        for index in test_indices:
            model(dataset[index][0])
    inference_seconds = time.perf_counter() - started
    best_epoch = history[-1].get("best_validation_epoch") if history else None
    return {"scenario": scenario, "model": model_name, "training_seed": int(training_seed),
            "metrics": metrics, "parameter_count": int(sum(p.numel() for p in model.parameters())),
            "training_seconds": training_seconds, "inference_seconds": inference_seconds,
            "best_validation_epoch": best_epoch, "history": history, "recovered_checkpoint": False}


def run_study(config, output_dir, seeds=None, scenarios=None, models=None):
    """Run all requested configurations; each uses the same frozen simulator data."""
    output_dir = Path(output_dir)
    selected_seeds = list(config["seeds"] if seeds is None else seeds)
    selected_scenarios = list(SCENARIOS if scenarios is None else scenarios)
    selected_models = list(MAIN_MODELS if models is None else models)
    records = [run_configuration(scenario, model, seed, config, output_dir)
               for scenario in selected_scenarios for seed in selected_seeds for model in selected_models]
    grouped = defaultdict(list)
    for record in records:
        grouped[f"{record['scenario']}::{record['model']}"].append(record)
    return {"protocol": {"simulator_seed": config["simulator_seed"], "topology_seed": config["topology_seed"],
                         "epochs": config["epochs"], "seeds": selected_seeds, "scenarios": selected_scenarios,
                         "models": selected_models},
            "records": records, "summary_by_scenario_model": {key: summarize(value) for key, value in grouped.items()}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/simulation.yaml")
    parser.add_argument("--output-dir", default="data/experiments/phase4_5")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--scenarios", nargs="+")
    parser.add_argument("--models", nargs="+")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)["gnn_experiment"]
    if args.epochs is not None:
        config["epochs"] = args.epochs
    report = run_study(config, args.output_dir, seeds=args.seeds, scenarios=args.scenarios, models=args.models)
    output = Path(args.output_dir) / "results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
