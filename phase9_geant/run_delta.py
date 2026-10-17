"""Train/evaluate D-Delta without modifying the Prompt 6 D-U experiment."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from phase9_geant.run_baselines import compute_metrics
from phase9_geant.run_neural import CONFIG, SEEDS, GeantUtilDataset, build_model, set_seed


def residual_targets(data: GeantUtilDataset, starts) -> np.ndarray:
    """Return u[t+1]-u[t] for windows ending at t (never future history)."""
    starts = np.asarray(starts, dtype=np.int64)
    final_history = np.stack([
        data.util[int(start) + CONFIG["history_length"] - 1] for start in starts
    ])
    return data.target(starts) - final_history


def fit_residual_scaler(data: GeantUtilDataset, train_starts) -> tuple[float, float]:
    """Fit mean/std on training-window residual targets only."""
    values = residual_targets(data, train_starts).astype(np.float64)
    mean = float(values.mean(dtype=np.float64))
    scale = float(values.std(dtype=np.float64))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Training residual scaler has invalid scale")
    return mean, scale


def reconstruct_utilization(last_observed, predicted_delta):
    """Reconstruct u_hat[t+1] = u[t] + predicted_delta[t+1]."""
    return np.asarray(last_observed, dtype=np.float64) + np.asarray(predicted_delta, dtype=np.float64)


def last_observed(data: GeantUtilDataset, starts) -> np.ndarray:
    return np.stack([
        data.util[int(start) + CONFIG["history_length"] - 1]
        for start in np.asarray(starts, dtype=np.int64)
    ])


def predict_batches(model, data, starts, residual_mean, residual_scale, batch_size=None):
    """Return raw-unit reconstructed utilization forecasts; no clipping/imputation."""
    batch_size = batch_size or CONFIG["batch_size"]
    model.eval()
    forecasts = []
    with torch.no_grad():
        for begin in range(0, len(starts), batch_size):
            batch_starts = np.asarray(starts[begin:begin + batch_size], dtype=np.int64)
            standardized_delta = model(data.sequence(batch_starts)).reshape(
                len(batch_starts), len(data.edges)
            ).cpu().numpy().astype(np.float64)
            delta = standardized_delta * residual_scale + residual_mean
            forecasts.append(reconstruct_utilization(last_observed(data, batch_starts), delta))
    return np.concatenate(forecasts, axis=0)


def forecast_diagnostics(prediction, target, last):
    actual_change = np.asarray(target, dtype=np.float64) - last
    predicted_change = np.asarray(prediction, dtype=np.float64) - last
    actual_flat, predicted_flat = actual_change.ravel(), predicted_change.ravel()
    if np.std(actual_flat) == 0 or np.std(predicted_flat) == 0:
        correlation = None
    else:
        correlation = float(np.corrcoef(actual_flat, predicted_flat)[0, 1])
    nonzero = actual_flat != 0
    return {
        "signed_bias": float((prediction - target).mean(dtype=np.float64)),
        "change_correlation": correlation,
        "change_direction_accuracy": float(
            (np.sign(predicted_flat[nonzero]) == np.sign(actual_flat[nonzero])).mean()
        ),
        "actual_change_std": float(np.std(actual_flat, dtype=np.float64)),
        "predicted_change_std": float(np.std(predicted_flat, dtype=np.float64)),
        "change_std_ratio": float(np.std(predicted_flat, dtype=np.float64) / np.std(actual_flat, dtype=np.float64)),
    }


def metric_bundle(prediction, target, last, edge_names):
    return {
        "overall": compute_metrics(target, prediction, float(target.mean(dtype=np.float64))),
        "diagnostics": forecast_diagnostics(prediction, target, last),
        "per_link_mae": [
            {"link": edge_names[i], "edge_idx": i,
             "mae": float(np.abs(target[:, i] - prediction[:, i]).mean(dtype=np.float64))}
            for i in range(len(edge_names))
        ],
    }


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _load_or_initialize(path, protocol):
    if not path.exists():
        return {"protocol": protocol, "seed_results": []}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol") != protocol:
        raise ValueError("Existing D-Delta results use a different protocol; refusing to merge")
    for row in payload.get("seed_results", []):
        if not Path(row["checkpoint"]).is_file():
            raise ValueError(f"Completed D-Delta seed is missing checkpoint: {row['checkpoint']}")
    return payload


def run():
    torch.set_num_threads(1)
    root = Path("data/processed/geant")
    out = root / "neural_prompt8_delta"
    checkpoint_dir = out / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    data = GeantUtilDataset(root)
    baseline_archive = np.load(root / "baseline_predictions.npz")
    train_starts = data.indices["train"]
    residual_mean, residual_scale = fit_residual_scaler(data, train_starts)
    baseline_ts = {split: baseline_archive[f"distance_{split}_target_ts"] for split in ("val", "test")}
    baseline_targets = {split: baseline_archive[f"distance_{split}_y"] for split in ("val", "test")}
    for split in ("val", "test"):
        starts = data.indices[split]
        if not np.array_equal(starts + CONFIG["history_length"], baseline_ts[split]):
            raise ValueError(f"{split} target timestamps do not match audited baseline archive")
        if not np.array_equal(data.target(starts), baseline_targets[split]):
            raise ValueError(f"{split} targets do not match audited baseline archive")

    protocol = {
        "experiment": "D-Delta residual/change prediction",
        "architecture": "TemporalAttentionEdgeForecaster (same D-U architecture, new target parameterization)",
        "history_length": CONFIG["history_length"],
        "prediction_horizon": CONFIG["prediction_horizon"],
        "features": ["utilization"],
        "target": "delta_utilization = u[t+1] - u[t], where u[t] is final historical observation",
        "reconstruction": "u_hat[t+1] = u[t] + delta_hat[t+1]",
        "seeds": list(SEEDS),
        "split_file": "data/processed/geant/split.json",
        "window_audit": data.window_audit,
        "input_scaler": {
            "method": "global standardization",
            "fit_scope": "valid timesteps in train block only",
            "mean": data.mean,
            "scale": data.scale,
        },
        "residual_scaler": {
            "method": "global standardization",
            "fit_scope": "residual targets from valid train windows only",
            "mean": residual_mean,
            "scale": residual_scale,
        },
        "training": {k: CONFIG[k] for k in (
            "epochs", "early_stopping_patience", "learning_rate", "batch_size",
            "optimizer", "loss", "hidden_channels", "gru_hidden_size", "initialization"
        )},
        "checkpoint_selection": "minimum validation SmoothL1Loss in standardized residual-target space; test not used",
        "evaluation": "reconstruct utilization; compare identical valid timestamps and directed links with persistence, EWMA, AR(5)",
        "initialization": "fresh seeded initialization; no warm start",
        "source_checkpoint": None,
        "baseline_source": "data/processed/geant/baseline_predictions.npz",
    }
    results_path = out / "results.json"
    results = _load_or_initialize(results_path, protocol)
    _write_json(out / "experiment_config.json", protocol)
    loss_fn = nn.SmoothL1Loss(reduction="mean")
    completed = {int(row["seed"]) for row in results["seed_results"]}

    for seed in SEEDS:
        if seed in completed:
            print(f"Skipping completed D-Delta seed {seed}")
            continue
        set_seed(seed)
        model = build_model("D-U")
        optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
        best_val_loss, best_epoch, stale = float("inf"), 0, 0
        history = []
        checkpoint = checkpoint_dir / f"D-delta_seed{seed}.pt"
        started = time.perf_counter()
        for epoch in range(CONFIG["epochs"]):
            model.train()
            losses = []
            order = np.random.permutation(train_starts)
            for begin in range(0, len(order), CONFIG["batch_size"]):
                starts = order[begin:begin + CONFIG["batch_size"]]
                prediction = model(data.sequence(starts)).reshape(len(starts), len(data.edges), 1)
                target = torch.as_tensor(
                    ((residual_targets(data, starts) - residual_mean) / residual_scale)[:, :, None],
                    dtype=torch.float32,
                )
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(prediction, target)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()))

            val_starts = data.indices["val"]
            model.eval()
            val_total, val_count = 0.0, 0
            with torch.no_grad():
                for begin in range(0, len(val_starts), CONFIG["batch_size"]):
                    starts = val_starts[begin:begin + CONFIG["batch_size"]]
                    prediction = model(data.sequence(starts)).reshape(len(starts), len(data.edges), 1)
                    target = torch.as_tensor(
                        ((residual_targets(data, starts) - residual_mean) / residual_scale)[:, :, None],
                        dtype=torch.float32,
                    )
                    val_total += float(loss_fn(prediction, target)) * target.numel()
                    val_count += target.numel()
            val_loss = val_total / val_count
            val_forecast = predict_batches(model, data, val_starts, residual_mean, residual_scale)
            val_target = data.target(val_starts)
            history.append({
                "epoch": epoch + 1,
                "train_loss": float(np.mean(losses)),
                "validation_loss_standardized_residual": val_loss,
                "validation_utilization_mae": float(np.abs(val_forecast - val_target).mean(dtype=np.float64)),
                "validation_utilization_rmse": float(np.sqrt(np.mean((val_forecast - val_target) ** 2, dtype=np.float64))),
            })
            if val_loss < best_val_loss:
                best_val_loss, best_epoch, stale = val_loss, epoch + 1, 0
                torch.save(model.state_dict(), checkpoint)
            else:
                stale += 1
            if stale >= CONFIG["early_stopping_patience"]:
                break

        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        split_metrics = {}
        for split in ("val", "test"):
            starts = data.indices[split]
            target = data.target(starts)
            last = last_observed(data, starts)
            prediction = predict_batches(model, data, starts, residual_mean, residual_scale)
            metrics = metric_bundle(prediction, target, last, data.edge_names)
            metrics["baseline_comparison"] = {}
            for method in ("persistence", "ewma", "ar5"):
                baseline_prediction = baseline_archive[f"distance_{split}_{method}"]
                metrics["baseline_comparison"][method] = compute_metrics(
                    target, baseline_prediction, float(target.mean(dtype=np.float64))
                )
            split_metrics[split] = metrics
        row = {
            "seed": int(seed),
            "checkpoint": str(checkpoint),
            "best_epoch": best_epoch,
            "epochs_completed": len(history),
            "best_validation_loss_standardized_residual": best_val_loss,
            "initialization": "fresh seeded initialization; no warm start",
            "training_seconds": float(time.perf_counter() - started),
            "history": history,
            "metrics": split_metrics,
        }
        results["seed_results"].append(row)
        results["seed_results"].sort(key=lambda item: item["seed"])
        _write_json(results_path, results)
        print(f"Completed D-Delta seed {seed}: best validation residual loss {best_val_loss:.6g}")

    if len(results["seed_results"]) == len(SEEDS):
        rows = results["seed_results"]
        maes = np.asarray([row["metrics"]["test"]["overall"]["mae"] for row in rows], dtype=np.float64)
        persistence_mae = float(np.abs(
            baseline_archive["distance_test_persistence"] - baseline_targets["test"]
        ).mean(dtype=np.float64))
        gains = persistence_mae - maes
        results["summary"] = {
            "test_mae_mean": float(maes.mean()),
            "test_mae_seed_std": float(maes.std(ddof=1)),
            "test_rmse_mean": float(np.mean([row["metrics"]["test"]["overall"]["rmse"] for row in rows])),
            "test_rmse_seed_std": float(np.std([row["metrics"]["test"]["overall"]["rmse"] for row in rows], ddof=1)),
            "persistence_test_mae": persistence_mae,
            "mean_mae_gain_vs_persistence": float(gains.mean()),
            "mae_gain_seed_std": float(gains.std(ddof=1)),
            "beats_persistence_seed_count": int((gains > 0).sum()),
            "preliminary_usefulness_gate_passed": bool((gains > 0).sum() >= 2 and gains.mean() > gains.std(ddof=1)),
            "baseline_test_mae": {
                method: float(np.abs(baseline_archive[f"distance_test_{method}"] - baseline_targets["test"]).mean())
                for method in ("persistence", "ewma", "ar5")
            },
        }

        # Save raw-unit, timestamp/link-aligned forecasts for reproducible audits.
        pred_arrays = {
            "link_names": np.asarray(data.edge_names),
            "val_target_indices": baseline_ts["val"],
            "test_target_indices": baseline_ts["test"],
            "val_targets": baseline_targets["val"],
            "test_targets": baseline_targets["test"],
            "val_last_observed": last_observed(data, data.indices["val"]),
            "test_last_observed": last_observed(data, data.indices["test"]),
        }
        for row in rows:
            model = build_model("D-U")
            model.load_state_dict(torch.load(row["checkpoint"], map_location="cpu", weights_only=True))
            pred_arrays[f"val_prediction_seed{row['seed']}"] = predict_batches(
                model, data, data.indices["val"], residual_mean, residual_scale
            )
            pred_arrays[f"test_prediction_seed{row['seed']}"] = predict_batches(
                model, data, data.indices["test"], residual_mean, residual_scale
            )
        temporary = out / "predictions.npz.tmp"
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **pred_arrays)
        temporary.replace(out / "predictions.npz")
        _write_json(results_path, results)
    return results_path


if __name__ == "__main__":
    print(run())
