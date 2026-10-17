"""Reduced-feature B/C/D forecasting of computed GEANT link utilization."""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from phase9_geant.parse_geant import parse_topology
from phase9_geant.run_baselines import compute_metrics
from src.gnn.dataset import GraphSnapshot
from src.gnn.model import (
    DirectedCongestionDGSTMTLForecaster,
    EdgeQoSForecaster,
    TemporalAttentionEdgeForecaster,
)

FEATURES = ("utilization",)
MODELS = ("B-U", "C-U", "D-U")
SEEDS = (42, 123, 2026)
CONFIG = {
    "history_length": 5,
    "prediction_horizon": 1,
    "epochs": 5,
    "early_stopping_patience": 2,
    "learning_rate": 0.005,
    "batch_size": 32,
    "optimizer": "Adam",
    "loss": "SmoothL1Loss",
    "hidden_channels": 16,
    "gru_hidden_size": 16,
    "initialization": "from_scratch",
}


class GeantUtilDataset:
    """Chronological windows and a single global scaler fitted on train steps."""

    def __init__(self, root=Path("data/processed/geant")):
        self.root = Path(root)
        self.util = np.load(self.root / "link_series_utilization.npy").astype(np.float64)
        self.valid = np.load(self.root / "valid_mask.npy")
        self.demand = np.load(self.root / "total_demand_float64.npy").astype(np.float64)
        self.split = json.loads((self.root / "split.json").read_text(encoding="utf-8"))["blocks"]
        nodes, links = parse_topology("data/raw/geant_topology/geant.txt")
        node_order = tuple(sorted(nodes))
        self.node_count = len(node_order)
        self.edges = tuple([(u, v) for u, v in links] + [(v, u) for u, v in links])
        self.edge_names = [f"{u}->{v}" for u, v in self.edges]
        node_index = {node: i for i, node in enumerate(node_order)}
        self.edge_index = torch.tensor(
            [[node_index[u] for u, _ in self.edges], [node_index[v] for _, v in self.edges]],
            dtype=torch.long,
        )
        train = self.split["train"]
        train_start, train_end = train["start_index"], train["end_index"]
        train_steps = np.arange(train_start, train_end)[self.valid[train_start:train_end]]
        train_values = self.util[train_steps]
        self.mean = float(train_values.mean(dtype=np.float64))
        self.scale = float(train_values.std(dtype=np.float64))
        if not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("Training utilization scaler has invalid scale")
        self.indices = {}
        self.window_audit = {}
        for name, block in self.split.items():
            starts = np.asarray(block["window_starts"], dtype=np.int64)
            block_start, block_end = block["start_index"], block["end_index"]
            candidates = max(0, block_end - block_start - CONFIG["history_length"])
            checks = [
                block_start <= int(start)
                and int(start) + CONFIG["history_length"] < block_end
                and self.valid[int(start):int(start) + CONFIG["history_length"] + 1].all()
                for start in starts
            ]
            if not all(checks):
                raise ValueError(f"Invalid or cross-block window found in {name}")
            self.indices[name] = starts
            self.window_audit[name] = {
                "candidate_windows": int(candidates),
                "valid_windows": int(len(starts)),
                "invalid_windows": int(candidates - len(starts)),
                "target_index_first": int(starts[0] + CONFIG["history_length"]),
                "target_index_last": int(starts[-1] + CONFIG["history_length"]),
                "target_index_ranges": [[
                    int(starts[0] + CONFIG["history_length"]),
                    int(starts[-1] + CONFIG["history_length"]),
                ]],
            }
        self.busy_threshold = float(np.percentile(self.demand[train_steps], 90))

    def sequence(self, starts):
        """Build a batch of disconnected copies of the same directed graph."""
        starts = np.asarray(starts, dtype=np.int64)
        batch_size = len(starts)
        x = torch.ones((batch_size * self.node_count, 1), dtype=torch.float32)
        offsets = torch.arange(batch_size, dtype=torch.long) * self.node_count
        edge_index = torch.cat([self.edge_index + offset for offset in offsets], dim=1)
        sequence = []
        for offset in range(CONFIG["history_length"]):
            rows = np.concatenate([self.util[int(start) + offset] for start in starts])
            attrs = torch.tensor(((rows - self.mean) / self.scale)[:, None], dtype=torch.float32)
            sequence.append(GraphSnapshot(x=x, edge_index=edge_index, edge_attr=attrs, timestep=int(starts[0] + offset)))
        return sequence

    def c_sequence(self, starts):
        """Batched edge attributes for C-U's sample-isolated dense link attention."""
        starts = np.asarray(starts, dtype=np.int64)
        return [
            GraphSnapshot(
                x=torch.ones((self.node_count, 1), dtype=torch.float32),
                edge_index=self.edge_index,
                edge_attr=torch.tensor(
                    np.stack([(self.util[int(start) + step] - self.mean) / self.scale for start in starts])[:, :, None],
                    dtype=torch.float32,
                ),
                timestep=int(starts[0] + step),
            )
            for step in range(CONFIG["history_length"])
        ]

    def target(self, starts):
        return np.stack([self.util[int(start) + CONFIG["history_length"]] for start in starts])

    def demand_target(self, starts):
        return self.demand[np.asarray(starts, dtype=np.int64) + CONFIG["history_length"]]


def build_model(name):
    if name == "B-U":
        return EdgeQoSForecaster(1, 1, CONFIG["hidden_channels"], CONFIG["gru_hidden_size"])
    if name == "C-U":
        return DirectedCongestionDGSTMTLForecaster(
            1, 1, CONFIG["hidden_channels"], feature_names=FEATURES
        )
    if name == "D-U":
        return TemporalAttentionEdgeForecaster(1, 1, CONFIG["hidden_channels"], CONFIG["gru_hidden_size"])
    raise ValueError(name)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def predict_batches(model, name, data, starts, batch_size=None):
    batch_size = batch_size or CONFIG["batch_size"]
    model.eval()
    predictions = []
    with torch.no_grad():
        for begin in range(0, len(starts), batch_size):
            batch_starts = starts[begin:begin + batch_size]
            sequence = data.c_sequence(batch_starts) if name == "C-U" else data.sequence(batch_starts)
            output = model.forward_batched(sequence) if name == "C-U" else model(sequence)
            output = output.reshape(len(batch_starts), len(data.edges), 1)[..., 0]
            raw = output.cpu().numpy().astype(np.float64) * data.scale + data.mean
            predictions.append(raw)
    return np.concatenate(predictions, axis=0)


def metric_bundle(prediction, target, target_demand, data):
    busy = target_demand >= data.busy_threshold
    return {
        "overall": compute_metrics(target, prediction, float(target.mean(dtype=np.float64))),
        "busy": compute_metrics(target[busy], prediction[busy], float(target[busy].mean(dtype=np.float64))),
        "busy_target_count": int(busy.sum()),
        "per_link_mae": [
            {"link": data.edge_names[index], "edge_idx": index,
             "mae": float(np.abs(target[:, index] - prediction[:, index]).mean(dtype=np.float64))}
            for index in range(len(data.edges))
        ],
    }


def _load_existing(output_path, expected_config):
    if not output_path.is_file():
        return {"protocol": expected_config, "models": {name: {"seed_results": []} for name in MODELS}}
    existing = json.loads(output_path.read_text(encoding="utf-8"))
    if existing.get("protocol") != expected_config:
        raise ValueError("Existing Prompt 6 results use a different protocol; refusing to merge")
    return existing


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temporary.replace(path)


def run():
    torch.set_num_threads(1)
    root = Path("data/processed/geant")
    out = root / "neural_prompt6"
    checkpoint_dir = out / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    data = GeantUtilDataset(root)
    baseline = json.loads((root / "baselines.json").read_text(encoding="utf-8"))
    protocol = {
        **CONFIG,
        "features": list(FEATURES),
        "target_order": ["utilization"],
        "seeds": list(SEEDS),
        "primary_series": "distance",
        "split_file": "data/processed/geant/split.json",
        "scaler": {"method": "global standardization", "fit_scope": "valid timesteps in train block only",
                   "mean": data.mean, "scale": data.scale},
        "window_audit": data.window_audit,
        "busy_target_threshold_train_p90_total_demand": data.busy_threshold,
        "checkpoint_selection": "minimum validation SmoothL1Loss on standardized utilization target",
        "initialization": "from_scratch; original synthetic B checkpoint and interrupted MAE-selected checkpoints not used",
        "source_checkpoint": None,
        "batching": "disconnected graph copies for B/D; independent per-sample link attention for C; mathematically sample-isolated",
    }
    result_path = out / "results.json"
    results = _load_existing(result_path, protocol)
    config_path = out / "experiment_config.json"
    _write_json(config_path, protocol)
    train_indices = data.indices["train"]
    loss_fn = nn.SmoothL1Loss(reduction="mean")

    for name in MODELS:
        results["models"].setdefault(name, {"seed_results": []})
        completed = {int(row["seed"]) for row in results["models"][name]["seed_results"]}
        for seed in SEEDS:
            if seed in completed:
                print(f"Skipping completed {name} seed {seed}")
                continue
            set_seed(seed)
            model = build_model(name)
            optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
            best_val_loss, best_epoch, stale = float("inf"), 0, 0
            history = []
            checkpoint = checkpoint_dir / f"{name}_seed{seed}.pt"
            started = time.perf_counter()
            for epoch in range(CONFIG["epochs"]):
                model.train()
                epoch_loss = []
                order = np.random.permutation(train_indices)
                for begin in range(0, len(order), CONFIG["batch_size"]):
                    starts = order[begin:begin + CONFIG["batch_size"]]
                    sequence = data.c_sequence(starts) if name == "C-U" else data.sequence(starts)
                    output = model.forward_batched(sequence) if name == "C-U" else model(sequence)
                    output = output.reshape(len(starts), len(data.edges), 1)
                    target = torch.tensor(
                        ((data.target(starts) - data.mean) / data.scale)[:, :, None],
                        dtype=torch.float32,
                    )
                    optimizer.zero_grad(set_to_none=True)
                    loss = loss_fn(output, target)
                    loss.backward()
                    optimizer.step()
                    epoch_loss.append(float(loss.detach()))

                model.eval()
                val_loss_total, val_count = 0.0, 0
                val_starts = data.indices["val"]
                with torch.no_grad():
                    for begin in range(0, len(val_starts), CONFIG["batch_size"]):
                        starts = val_starts[begin:begin + CONFIG["batch_size"]]
                        sequence = data.c_sequence(starts) if name == "C-U" else data.sequence(starts)
                        output = model.forward_batched(sequence) if name == "C-U" else model(sequence)
                        output = output.reshape(len(starts), len(data.edges), 1)
                        target = torch.tensor(
                            ((data.target(starts) - data.mean) / data.scale)[:, :, None], dtype=torch.float32
                        )
                        val_loss_total += float(loss_fn(output, target)) * target.numel()
                        val_count += target.numel()
                val_loss = val_loss_total / val_count
                val_prediction = predict_batches(model, name, data, val_starts)
                val_target = data.target(val_starts)
                history.append({"epoch": epoch + 1, "train_loss": float(np.mean(epoch_loss)),
                                "validation_loss": val_loss,
                                "validation_mae": float(np.abs(val_prediction - val_target).mean(dtype=np.float64)),
                                "validation_rmse": float(np.sqrt(np.mean((val_prediction - val_target) ** 2, dtype=np.float64)))})
                if val_loss < best_val_loss:
                    best_val_loss, best_epoch, stale = val_loss, epoch + 1, 0
                    torch.save(model.state_dict(), checkpoint)
                else:
                    stale += 1
                if stale >= CONFIG["early_stopping_patience"]:
                    break

            model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
            metrics = {}
            for split in ("val", "test"):
                starts = data.indices[split]
                prediction = predict_batches(model, name, data, starts)
                target = data.target(starts)
                metrics[split] = metric_bundle(prediction, target, data.demand_target(starts), data)
            row = {
                "seed": int(seed), "checkpoint": str(checkpoint), "best_epoch": best_epoch,
                "epochs_completed": len(history), "best_validation_loss": best_val_loss,
                "initialization": "fresh seeded initialization; no warm start",
                "training_seconds": float(time.perf_counter() - started), "history": history,
                "metrics": metrics,
            }
            results["models"][name]["seed_results"].append(row)
            results["models"][name]["seed_results"].sort(key=lambda item: item["seed"])
            _write_json(result_path, results)
            print(f"Completed {name} seed {seed}: best validation loss {best_val_loss:.6g}")

        rows = results["models"][name]["seed_results"]
        if len(rows) == len(SEEDS):
            test_mae = np.asarray([row["metrics"]["test"]["overall"]["mae"] for row in rows], dtype=np.float64)
            persistence = baseline["distance"]["blocks"]["test"]["baselines"]["persistence"]["overall"]["mae"]
            gain = persistence - test_mae
            results["models"][name]["summary"] = {
                "test_mae_mean": float(test_mae.mean()), "test_mae_seed_std": float(test_mae.std(ddof=1)),
                "persistence_test_mae": float(persistence), "mean_mae_gain_vs_persistence": float(gain.mean()),
                "gain_seed_std": float(gain.std(ddof=1)), "beats_persistence_seed_count": int((gain > 0).sum()),
                "acceptance_useful": bool((gain > 0).sum() >= 2 and gain.mean() > gain.std(ddof=1)),
            }
            _write_json(result_path, results)

    # Keep the audited Prompt 5 non-neural evaluations alongside the neural
    # results. Seasonal naive metrics retain their own covered subset metadata.
    results["baseline_comparison"] = {
        "series": "distance-shortest-path utilization (primary)",
        "source": "data/processed/geant/baselines.json",
        "blocks": {
            split: {
                "target_count": int(len(data.indices[split]) * len(data.edges)),
                "methods": {
                    method: {
                        "overall": values["overall"],
                        **({"coverage": values["coverage"],
                            "valid_count": values["valid_count"],
                            "total_count": values["total_count"],
                            "persistence_same_subset": values["persistence_same_subset"]}
                           if method == "seasonal_naive_96" else {}),
                    }
                    for method, values in baseline["distance"]["blocks"][split]["baselines"].items()
                },
            }
            for split in ("val", "test")
        },
    }
    for name in MODELS:
        rows = results["models"][name]["seed_results"]
        if len(rows) != len(SEEDS):
            continue
        results["models"][name]["summary"]["non_neural_test_mae_comparison"] = {
            method: {
                "baseline_test_mae": values["overall"]["mae"],
                "neural_mean_test_mae_minus_baseline": float(
                    np.mean([row["metrics"]["test"]["overall"]["mae"] for row in rows])
                    - values["overall"]["mae"]
                ),
            }
            for method, values in baseline["distance"]["blocks"]["test"]["baselines"].items()
        }
    _write_json(result_path, results)

    return result_path


if __name__ == "__main__":
    print(run())
