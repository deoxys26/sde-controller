"""Validation-only B-U and baseline diagnostics for GEANT validity-v2.

The script reads only the four v2 train/validation arrays, the three explicit
B-U full-run manifests/checkpoints, topology text, and validation-only baseline
and split metadata. It never discovers files by glob and never loads test data.
All new outputs are written to one fresh diagnostic directory.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from phase9_geant.parse_geant import parse_topology
from phase9_geant.train_bu_v2 import (
    HISTORY_LENGTH,
    PROTOCOL_ID,
    TRAINING,
    V2_DATA_ROOT,
    Normalization,
    build_edge_index,
    load_v2_windows,
    make_bu_sequence,
)
from src.gnn.model import EdgeQoSForecaster


SCRIPT_VERSION = "1.0.0"
SEEDS = (42, 123, 2026)
EWMA_ALPHA = 0.8
MOVING_AVERAGE_WINDOW = 3
SEASONAL_PERIOD_STEPS = 4  # One hour at 15-minute sampling; available inside history.
NEAR_ZERO_THRESHOLD = 0.001  # 0.1% utilization ratio, fixed before validation scoring.
HIGH_LINK_FRACTION = 0.10
TOP_DECILE_LINK_COUNT = math.ceil(72 * HIGH_LINK_FRACTION)
TARGET_QUANTILES = (0.50, 0.90, 0.99)
SEEDS_RUN_DIR = {
    seed: V2_DATA_ROOT / f"neural_bu_v2_full_seed_{seed}" for seed in SEEDS
}
BASELINE_REPORT_PATH = V2_DATA_ROOT / "baselines_validation.json"
METADATA_PATH = V2_DATA_ROOT / "metadata.json"
WINDOW_STARTS_PATH = V2_DATA_ROOT / "window_starts.json"
TOPOLOGY_PATH = Path(__file__).resolve().parents[1] / "data/raw/geant_topology/geant.txt"
OUTPUT_DIR = V2_DATA_ROOT / "diagnostics_baseline_error_20261017_final"


def build_baselines(history: np.ndarray) -> dict[str, np.ndarray]:
    """Return causal forecasts from [sample, five historical steps, edge]."""
    if history.ndim != 3 or history.shape[1] != HISTORY_LENGTH:
        raise ValueError("Expected histories with shape [samples, 5, directed_links]")
    if not np.isfinite(history).all():
        raise ValueError("History contains nonfinite values; refusing to impute")
    weights = EWMA_ALPHA * (1.0 - EWMA_ALPHA) ** (4 - np.arange(5, dtype=np.float64))
    weights /= weights.sum(dtype=np.float64)
    return {
        "persistence": history[:, -1, :].astype(np.float64),
        "ewma_alpha_0.8": np.tensordot(history.astype(np.float64), weights, axes=([1], [0])),
        "moving_average_3": history[:, -MOVING_AVERAGE_WINDOW:, :].astype(np.float64).mean(axis=1),
        # Seasonal-naive period 4 predicts target t+1 from observation t-3,
        # which is history index 1 of [t-4, ..., t]. It cannot cross a gap
        # because each accepted v2 sample is an all-valid consecutive window.
        "seasonal_naive_4": history[:, -SEASONAL_PERIOD_STEPS, :].astype(np.float64),
    }


def training_thresholds(y_train: np.ndarray) -> dict[str, float]:
    """Compute frozen utilization thresholds from training targets only."""
    if y_train.ndim != 2 or y_train.size == 0 or not np.isfinite(y_train).all():
        raise ValueError("Training targets must be a nonempty finite [samples, links] array")
    values = y_train.astype(np.float64).reshape(-1)
    return {
        f"p{int(q * 100)}": float(np.quantile(values, q, method="linear"))
        for q in TARGET_QUANTILES
    }


def subgroup_masks(
    target: np.ndarray, thresholds: dict[str, float]
) -> dict[str, np.ndarray]:
    """Build fixed cell-level utilization bands from training-only quantiles."""
    if target.ndim != 2 or not np.isfinite(target).all():
        raise ValueError("Targets must be a finite [samples, links] array")
    p50, p90, p99 = thresholds["p50"], thresholds["p90"], thresholds["p99"]
    return {
        "low_le_train_p50": target <= p50,
        "mid_train_p50_to_p90": (target > p50) & (target < p90),
        "high_ge_train_p90": target >= p90,
        "burst_cells_ge_train_p99": target >= p99,
    }


def burst_window_mask(target: np.ndarray, thresholds: dict[str, float]) -> np.ndarray:
    if target.ndim != 2 or not np.isfinite(target).all():
        raise ValueError("Targets must be finite before burst-window selection")
    return np.any(target >= thresholds["p99"], axis=1)


def metric_bundle(
    target: np.ndarray,
    prediction: np.ndarray,
    mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute float64 MAE/RMSE on identical finite target/prediction pairs."""
    if target.shape != prediction.shape:
        raise ValueError(f"Target/prediction shape mismatch: {target.shape} != {prediction.shape}")
    if not np.isfinite(target).all():
        raise ValueError("Target contains nonfinite values; no target may be silently dropped")
    if not np.isfinite(prediction).all():
        raise ValueError("Prediction contains nonfinite values; no prediction may be dropped")
    if mask is None:
        selected = np.ones(target.shape, dtype=bool)
    else:
        if mask.shape != target.shape or mask.dtype != np.dtype(bool):
            raise ValueError("Metric mask must be boolean and match target/prediction dimensions")
        selected = mask
    count = int(selected.sum())
    errors = prediction[selected].astype(np.float64) - target[selected].astype(np.float64)
    return {
        "evaluated_pairs": count,
        "available_pairs": count,
        "coverage": float(count / count) if count else 0.0,
        "full_split_pairs": int(target.size),
        "mae": float(np.mean(np.abs(errors), dtype=np.float64)) if count else None,
        "rmse": float(np.sqrt(np.mean(np.square(errors), dtype=np.float64))) if count else None,
    }


def per_link_metrics(target: np.ndarray, prediction: np.ndarray) -> list[dict[str, Any]]:
    if target.shape != prediction.shape or target.ndim != 2:
        raise ValueError("Per-link target and prediction must share [samples, links] shape")
    if not np.isfinite(target).all() or not np.isfinite(prediction).all():
        raise ValueError("Per-link metrics refuse nonfinite values")
    rows = []
    for edge_index in range(target.shape[1]):
        result = metric_bundle(target[:, edge_index], prediction[:, edge_index])
        rows.append({"edge_index": edge_index, **result})
    return rows


def _assert_v2_metadata() -> list[int]:
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    if metadata.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("v2 metadata protocol ID mismatch")
    if metadata.get("history_steps") != HISTORY_LENGTH or metadata.get("prediction_horizon") != 1:
        raise ValueError("v2 metadata does not specify five-history/one-step forecasting")
    bounds = metadata.get("split_bounds", {})
    if bounds.get("train") != [0, 8020] or bounds.get("validation") != [8020, 9740]:
        raise ValueError("Train/validation split bounds differ from the frozen v2 protocol")
    starts = json.loads(WINDOW_STARTS_PATH.read_text(encoding="utf-8"))
    validation_starts = starts.get("validation")
    if validation_starts != list(range(8020, 9740 - HISTORY_LENGTH)):
        raise ValueError("Validation window starts are not consecutive and split-local")
    return validation_starts


def _load_run(seed: int, common_edges: list[dict[str, Any]]) -> tuple[dict[str, Any], Path]:
    run_dir = SEEDS_RUN_DIR[seed]
    manifest_path = run_dir / "run_metadata.json"
    checkpoint = run_dir / "best_validation.pt"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("protocol_id") != PROTOCOL_ID
        or manifest.get("status") != "complete"
        or manifest.get("mode") != "full"
        or manifest.get("candidate") != "B-U"
        or manifest.get("seed") != seed
    ):
        raise ValueError(f"Seed {seed} manifest is incomplete or has wrong identity")
    if manifest.get("model_settings") != TRAINING:
        raise ValueError(f"Seed {seed} training settings differ from the fixed protocol")
    if manifest.get("directed_edge_order") != common_edges:
        raise ValueError(f"Seed {seed} directed-edge order differs from runtime topology")
    expected_norm = "all scalar values in X_train.npy and y_train.npy only"
    norm = manifest.get("normalization", {})
    if norm.get("fit_scope") != expected_norm:
        raise ValueError(f"Seed {seed} scaler provenance is not training-only")
    if manifest.get("data_split", {}).get("test") != "not loaded or evaluated":
        raise ValueError(f"Seed {seed} manifest does not affirm test isolation")
    expected_checkpoint = (run_dir / "best_validation.pt").resolve()
    if Path(str(manifest.get("checkpoint", ""))).resolve() != expected_checkpoint:
        raise ValueError(f"Seed {seed} checkpoint path does not match its run directory")
    if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
        raise FileNotFoundError(f"Seed {seed} checkpoint is missing or empty")
    return manifest, checkpoint


def _predict_gnn(
    checkpoint: Path,
    manifest: dict[str, Any],
    x_validation: np.ndarray,
    edge_index: torch.Tensor,
    node_count: int,
) -> np.ndarray:
    normalization = Normalization(
        mean=float(manifest["normalization"]["mean"]),
        scale=float(manifest["normalization"]["scale"]),
    )
    if not np.isfinite(normalization.mean) or not np.isfinite(normalization.scale) or normalization.scale <= 0:
        raise ValueError("Invalid recorded training-only normalization")
    model = EdgeQoSForecaster(
        edge_feature_count=1,
        target_count=1,
        hidden_channels=TRAINING["hidden_channels"],
        gru_hidden_size=TRAINING["gru_hidden_size"],
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    batches = []
    with torch.no_grad():
        for start in range(0, len(x_validation), TRAINING["batch_size"]):
            history = x_validation[start : start + TRAINING["batch_size"]]
            sequence = make_bu_sequence(history, edge_index, node_count, normalization)
            standardized = model(sequence).reshape(len(history), edge_index.shape[1])
            if not torch.isfinite(standardized).all():
                raise ValueError("B-U produced a nonfinite validation prediction")
            raw = standardized.cpu().numpy().astype(np.float64) * normalization.scale + normalization.mean
            batches.append(raw)
    result = np.concatenate(batches, axis=0)
    if result.shape != (len(x_validation), edge_index.shape[1]) or not np.isfinite(result).all():
        raise ValueError("B-U prediction shape or finiteness check failed")
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty diagnostic table: {path.name}")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(output_dir: Path = OUTPUT_DIR) -> Path:
    """Run diagnostics; output must be a previously nonexistent directory."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite diagnostic output: {output_dir}")
    if not output_dir.resolve().is_relative_to(V2_DATA_ROOT.resolve()):
        raise ValueError("Diagnostic output must be contained under the v2 data root")

    validation_starts = _assert_v2_metadata()
    arrays = load_v2_windows(V2_DATA_ROOT)
    x_train, y_train = arrays.x_train, arrays.y_train
    x_val, y_val = arrays.x_validation, arrays.y_validation
    if len(validation_starts) != len(x_val):
        raise ValueError("Validation array sample count does not match split-local window-start metadata")
    if x_train.shape[1:] != (5, 72) or y_train.shape != (len(x_train), 72):
        raise ValueError("Unexpected v2 training array shape")
    if x_val.shape[1:] != (5, 72) or y_val.shape != (len(x_val), 72):
        raise ValueError("Unexpected v2 validation array shape")

    nodes, links = parse_topology(str(TOPOLOGY_PATH))
    edge_index, node_order, edges = build_edge_index(nodes, links)
    edge_order = [
        {"index": i, "source": source, "destination": destination}
        for i, (source, destination) in enumerate(edges)
    ]
    if len(node_order) != 22 or len(links) != 36 or len(edges) != 72:
        raise ValueError("Runtime topology does not satisfy the v2 22/36/72 contract")
    if x_val.shape[2] != len(edges):
        raise ValueError("Validation edge columns do not match topology directed-edge order")

    manifests: dict[int, dict[str, Any]] = {}
    checkpoints: dict[int, Path] = {}
    predictions: dict[str, np.ndarray] = build_baselines(x_val)
    for seed in SEEDS:
        manifest, checkpoint = _load_run(seed, edge_order)
        manifests[seed] = manifest
        checkpoints[seed] = checkpoint
        predictions[f"B-U_seed_{seed}"] = _predict_gnn(
            checkpoint, manifest, x_val, edge_index, len(node_order)
        )

    thresholds = training_thresholds(y_train)
    masks = subgroup_masks(y_val, thresholds)
    burst_windows = burst_window_mask(y_val, thresholds)
    # Trained-data-only link ranking: top decile by mean target utilization.
    train_link_means = y_train.astype(np.float64).mean(axis=0, dtype=np.float64)
    high_links = sorted(range(y_train.shape[1]), key=lambda i: (-train_link_means[i], i))[:TOP_DECILE_LINK_COUNT]

    baseline_reference = json.loads(BASELINE_REPORT_PATH.read_text(encoding="utf-8"))
    if baseline_reference.get("protocol_id") != PROTOCOL_ID or baseline_reference.get("split") != "validation":
        raise ValueError("Existing baseline reference is not the same v2 validation protocol")
    if baseline_reference.get("history_shape") != list(x_val.shape) or baseline_reference.get("target_shape") != list(y_val.shape):
        raise ValueError("Existing validation baseline does not cover identical array dimensions")

    overall = {name: metric_bundle(y_val, pred) for name, pred in predictions.items()}
    for name in ("persistence", "ewma_alpha_0.8"):
        key = "persistence" if name == "persistence" else "ewma"
        recorded = baseline_reference["metrics"][key]
        actual = overall[name]
        if actual["evaluated_pairs"] != recorded["evaluated_scalar_targets"]:
            raise ValueError(f"Rebuilt {key} coverage differs from existing validation baseline")
        for metric in ("mae", "rmse"):
            if not np.isclose(actual[metric], recorded[metric], rtol=0.0, atol=1e-12):
                raise ValueError(f"Rebuilt {key} {metric} differs from existing validation baseline")

    seed_mae = [overall[f"B-U_seed_{seed}"]["mae"] for seed in SEEDS]
    seed_rmse = [overall[f"B-U_seed_{seed}"]["rmse"] for seed in SEEDS]
    aggregate = {
        metric: {
            "mean_across_seeds": float(np.mean(values, dtype=np.float64)),
            "sample_std_across_seeds": float(np.std(values, ddof=1, dtype=np.float64)),
            "min_across_seeds": float(np.min(values)),
            "max_across_seeds": float(np.max(values)),
            "range_across_seeds": float(np.max(values) - np.min(values)),
        }
        for metric, values in (("mae", seed_mae), ("rmse", seed_rmse))
    }

    pairwise: dict[str, Any] = {}
    per_link_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    for method, prediction in predictions.items():
        for link_row in per_link_metrics(y_val, prediction):
            index = link_row["edge_index"]
            per_link_rows.append({"method": method, "edge_index": index,
                                  "source": edges[index][0], "destination": edges[index][1],
                                  "mae": link_row["mae"], "rmse": link_row["rmse"],
                                  "evaluated_pairs": link_row["evaluated_pairs"]})
    per_link_lookup = {(row["method"], row["edge_index"]): row for row in per_link_rows}
    top_link_metrics = []
    for index in high_links:
        ewma_row = per_link_lookup[("ewma_alpha_0.8", index)]
        seed_rows = [per_link_lookup[(f"B-U_seed_{seed}", index)] for seed in SEEDS]
        top_link_metrics.append({
            "edge_index": index,
            "source": edges[index][0],
            "destination": edges[index][1],
            "training_mean_utilization": float(train_link_means[index]),
            "ewma_mae": ewma_row["mae"],
            "ewma_rmse": ewma_row["rmse"],
            "b_u_mean_seed_mae": float(np.mean([row["mae"] for row in seed_rows], dtype=np.float64)),
            "b_u_mean_seed_rmse": float(np.mean([row["rmse"] for row in seed_rows], dtype=np.float64)),
            "per_seed": {
                str(seed): {"mae": row["mae"], "rmse": row["rmse"]}
                for seed, row in zip(SEEDS, seed_rows, strict=True)
            },
        })
    baseline_names = ("persistence", "ewma_alpha_0.8", "moving_average_3", "seasonal_naive_4")
    for seed in SEEDS:
        model_name = f"B-U_seed_{seed}"
        pairwise[str(seed)] = {}
        model_link = per_link_metrics(y_val, predictions[model_name])
        for baseline_name in baseline_names:
            base_link = per_link_metrics(y_val, predictions[baseline_name])
            mae_diffs = []
            rmse_diffs = []
            for model_row, baseline_row in zip(model_link, base_link, strict=True):
                mae_diff = model_row["mae"] - baseline_row["mae"]
                rmse_diff = model_row["rmse"] - baseline_row["rmse"]
                mae_diffs.append(mae_diff)
                rmse_diffs.append(rmse_diff)
                paired_rows.append({
                    "seed": seed, "model": model_name, "baseline": baseline_name,
                    "edge_index": model_row["edge_index"],
                    "model_mae": model_row["mae"], "baseline_mae": baseline_row["mae"],
                    "mae_difference_model_minus_baseline": mae_diff,
                    "model_rmse": model_row["rmse"], "baseline_rmse": baseline_row["rmse"],
                    "rmse_difference_model_minus_baseline": rmse_diff,
                })
            pairwise[str(seed)][baseline_name] = {
                "mae": {"mean_per_link_difference": float(np.mean(mae_diffs)),
                        "links_won": int(np.sum(np.asarray(mae_diffs) < 0)),
                        "links_lost": int(np.sum(np.asarray(mae_diffs) > 0)),
                        "links_tied": int(np.sum(np.asarray(mae_diffs) == 0))},
                "rmse": {"mean_per_link_difference": float(np.mean(rmse_diffs)),
                         "links_won": int(np.sum(np.asarray(rmse_diffs) < 0)),
                         "links_lost": int(np.sum(np.asarray(rmse_diffs) > 0)),
                         "links_tied": int(np.sum(np.asarray(rmse_diffs) == 0))},
            }

    subgroup_rows: list[dict[str, Any]] = []
    for method, prediction in predictions.items():
        for group, mask in masks.items():
            result = metric_bundle(y_val, prediction, mask)
            subgroup_rows.append({"method": method, "group": group,
                                  "windows": int(np.any(mask, axis=1).sum()),
                                  "selected_links": "", **result})
        window_mask = np.broadcast_to(burst_windows[:, None], y_val.shape)
        subgroup_rows.append({"method": method, "group": "burst_windows_any_target_ge_train_p99",
                              "windows": int(burst_windows.sum()), "selected_links": "",
                              **metric_bundle(y_val, prediction, window_mask)})
        subgroup_rows.append({"method": method, "group": "top_8_links_by_training_mean",
                              "windows": int(len(y_val)), "selected_links": ";".join(map(str, high_links)),
                              **metric_bundle(y_val[:, high_links], prediction[:, high_links])})

    distribution = {
        "validation_target_pairs": int(y_val.size),
        "minimum": float(np.min(y_val)),
        "median": float(np.quantile(y_val.astype(np.float64), 0.50)),
        "p90": float(np.quantile(y_val.astype(np.float64), 0.90)),
        "p95": float(np.quantile(y_val.astype(np.float64), 0.95)),
        "p99": float(np.quantile(y_val.astype(np.float64), 0.99)),
        "maximum": float(np.max(y_val)),
        "zero_count": int(np.count_nonzero(y_val == 0)),
        "zero_proportion": float(np.mean(y_val == 0)),
        "near_zero_threshold_ratio": NEAR_ZERO_THRESHOLD,
        "near_zero_count": int(np.count_nonzero(y_val <= NEAR_ZERO_THRESHOLD)),
        "near_zero_proportion": float(np.mean(y_val <= NEAR_ZERO_THRESHOLD)),
    }
    subgroup_summary = {
        name: {"target_pairs": int(mask.sum()), "proportion_of_all_pairs": float(mask.mean())}
        for name, mask in masks.items()
    }
    subgroup_summary["burst_windows_any_target_ge_train_p99"] = {
        "windows": int(burst_windows.sum()),
        "total_validation_windows": int(len(burst_windows)),
        "target_pairs": int(burst_windows.sum() * y_val.shape[1]),
    }

    results = {
        "script_version": SCRIPT_VERSION,
        "protocol_id": PROTOCOL_ID,
        "scope": "validation diagnostics only; no training, test access, or model selection",
        "inputs": {
            "arrays": ["X_train.npy", "y_train.npy", "X_validation.npy", "y_validation.npy"],
            "baseline_reference": str(BASELINE_REPORT_PATH),
            "topology": str(TOPOLOGY_PATH),
            "checkpoint_paths": {str(seed): str(checkpoints[seed]) for seed in SEEDS},
        },
        "validation_alignment": {
            "target_indices": [8025, 9739],
            "history_length": HISTORY_LENGTH,
            "prediction_horizon": 1,
            "windows": int(len(x_val)),
            "directed_links": int(y_val.shape[1]),
            "directed_edge_order": edge_order,
            "finite_target_pairs": int(np.isfinite(y_val).sum()),
            "total_target_pairs": int(y_val.size),
        },
        "definitions": {
            "metrics": "MAE=mean(abs(pred-target)); RMSE=sqrt(mean((pred-target)^2)); float64; dimensionless computed utilization ratio",
            "persistence": "last of five historical observations",
            "ewma": {"alpha": EWMA_ALPHA, "weights_oldest_to_newest": (EWMA_ALPHA * (1-EWMA_ALPHA) ** (4-np.arange(5, dtype=np.float64)) / np.sum(EWMA_ALPHA * (1-EWMA_ALPHA) ** (4-np.arange(5, dtype=np.float64)))).tolist()},
            "moving_average": {"window_steps": MOVING_AVERAGE_WINDOW, "definition": "unweighted mean of latest three history observations"},
            "seasonal_naive": {"period_steps": SEASONAL_PERIOD_STEPS, "minutes": 60, "definition": "history t-3 predicts target t+1; history index 1 of 5; v2 valid consecutive windows only"},
            "thresholds_fitted_on_training_y_only": thresholds,
            "utilization_bands": {"low": "target <= training p50", "mid": "training p50 < target < training p90", "high": "target >= training p90", "burst_cell": "target >= training p99", "burst_window": "any target link in window >= training p99"},
            "highest_utilization_links": {"rule": "top ceil(10% of 72)=8 directed links by training-target mean; ties by edge index", "edge_indices": high_links, "link_order": [{"edge_index": i, "source": edges[i][0], "destination": edges[i][1]} for i in high_links]},
        },
        "target_distribution": distribution,
        "subgroup_coverage": subgroup_summary,
        "overall_metrics": overall,
        "b_u_seed_aggregate": aggregate,
        "per_link_paired_win_loss": pairwise,
        "checkpoint_selection": {str(seed): {"best_epoch": manifests[seed]["best_validation_epoch"], "validation_smooth_l1_standardized": manifests[seed]["best_validation_loss"], "checkpoint": str(checkpoints[seed])} for seed in SEEDS},
        "top_utilization_link_metrics": top_link_metrics,
        "spatial_architecture_note": "The B-U adapter assigns an all-ones feature to every node, and the topology has no zero-in-degree nodes. DirectedGraphConv uses mean aggregation, so every node receives the same neighbor-feature mean; node embeddings are identical and do not encode node-specific topology. Each directed edge is encoded from those repeated endpoint embeddings plus only that edge's own utilization value. The GRU runs independently per directed edge; neighboring links' historical utilization is not propagated. Thus this implementation does not meaningfully exploit link-to-link spatial utilization information.",
        "limitations": [
            "The sample standard deviation/range across three seeds describes run variability, not a confidence interval.",
            "Validation data selected checkpoints; validation metrics are diagnostic, not held-out generalization evidence.",
            "Seasonal naive uses a one-hour lag because a daily lag of 96 steps is unavailable in a five-step input; no prediction is substituted across missing windows.",
            "Burst windows are target-defined periods within each validation sample, not independently reconstructed contiguous time episodes.",
        ],
    }

    # All data computation succeeded before the fresh destination is created.
    output_dir.mkdir(parents=False, exist_ok=False)
    (output_dir / "results.json").write_text(json.dumps(results, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    _write_csv(output_dir / "per_link_metrics.csv", per_link_rows)
    _write_csv(output_dir / "paired_link_differences.csv", paired_rows)
    _write_csv(output_dir / "subgroup_metrics.csv", subgroup_rows)
    report = _render_report(results)
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "overall_metrics": overall, "seed_aggregate": aggregate}, indent=2))
    return output_dir


def _render_report(results: dict[str, Any]) -> str:
    lines = [
        "# GEANT v2 baseline and error diagnostics",
        "",
        "**Scope:** diagnostics on the v2 validation split only; no training or test access.",
        "",
        f"Protocol: `{results['protocol_id']}`. Validation targets: {results['validation_alignment']['windows']} windows × {results['validation_alignment']['directed_links']} directed links ({results['validation_alignment']['total_target_pairs']} pairs), target indices {results['validation_alignment']['target_indices']} inclusive.",
        "Metrics are float64 MAE/RMSE in dimensionless computed-utilization-ratio units, on identical complete target-link pairs. Training-only normalization is applied for B-U inputs and reversed for its output; classical baselines predict directly in utilization units.",
        "",
        "## Overall validation results",
        "",
        "| Method | MAE | RMSE | evaluated pairs |",
        "|---|---:|---:|---:|",
    ]
    for method, values in results["overall_metrics"].items():
        lines.append(f"| {method} | {values['mae']:.10g} | {values['rmse']:.10g} | {values['evaluated_pairs']} |")
    for metric, values in results["b_u_seed_aggregate"].items():
        lines.append(f"| B-U {metric} across seeds | {values['mean_across_seeds']:.10g} ± {values['sample_std_across_seeds']:.10g} | range {values['min_across_seeds']:.10g}–{values['max_across_seeds']:.10g} | 3 seeds |")
    lines.extend(["", "Seed SD/range are descriptive run variability, not confidence intervals.", "", "## Per-seed B-U metrics", "", "| Seed | Best epoch | Validation SmoothL1 (standardized) | MAE | RMSE |", "|---:|---:|---:|---:|---:|"])
    for seed, detail in results["checkpoint_selection"].items():
        metrics = results["overall_metrics"][f"B-U_seed_{seed}"]
        lines.append(f"| {seed} | {detail['best_epoch']} | {detail['validation_smooth_l1_standardized']:.10g} | {metrics['mae']:.10g} | {metrics['rmse']:.10g} |")
    lines.extend(["", "## Baseline definitions and paired link diagnostics", "", "- Persistence is the last observed input step.", "- EWMA uses fixed α=0.8 and the existing normalized weights.", "- Moving average is the unweighted latest three steps.", "- Seasonal naive uses an available four-step (one-hour) lag inside each five-step history; daily naive-96 is unavailable.", "- Per-link paired MAE/RMSE differences are B-U minus baseline; a negative value is a B-U win. Directed-link wins/losses are reported in `results.json` and row-level values in `paired_link_differences.csv`.", "- Existing persistence and EWMA validation metrics were reproduced and matched against the existing validation-only baseline report at 1e-12 absolute tolerance.", "", "## Utilization distribution and predeclared subgroups", "", f"Validation target median={results['target_distribution']['median']:.8g}, p90={results['target_distribution']['p90']:.8g}, p95={results['target_distribution']['p95']:.8g}, p99={results['target_distribution']['p99']:.8g}, maximum={results['target_distribution']['maximum']:.8g}; exact zero share={results['target_distribution']['zero_proportion']:.3%}; near-zero (≤{NEAR_ZERO_THRESHOLD}) share={results['target_distribution']['near_zero_proportion']:.3%}.", f"Training-only thresholds: p50={results['definitions']['thresholds_fitted_on_training_y_only']['p50']:.8g}, p90={results['definitions']['thresholds_fitted_on_training_y_only']['p90']:.8g}, p99={results['definitions']['thresholds_fitted_on_training_y_only']['p99']:.8g}. Subgroup metrics, eligible counts, and prediction coverage are in `subgroup_metrics.csv`.", "", "### Training-defined top 8 directed links", "", "Links are ranked by training-target mean utilization (ties by edge index). B-U values are the mean of the three seed-specific per-link metrics; all seed-specific values remain in `per_link_metrics.csv`.", "", "| Edge | Link | EWMA MAE | B-U mean MAE | EWMA RMSE | B-U mean RMSE |", "|---:|---|---:|---:|---:|---:|"])
    for item in results["top_utilization_link_metrics"]:
        lines.append(f"| {item['edge_index']} | {item['source']}→{item['destination']} | {item['ewma_mae']:.8g} | {item['b_u_mean_seed_mae']:.8g} | {item['ewma_rmse']:.8g} | {item['b_u_mean_seed_rmse']:.8g} |")
    lines.extend(["", "## Interpretation and limitations", "", results["spatial_architecture_note"], "", "Validation was used to select the retained checkpoints; these results are diagnostics, not independent test generalization. Per-link win counts do not establish statistical significance. Burst windows are operational target-defined windows, not independent contiguous time episodes. No clipping or imputation is applied.", "", "## Artifacts", "", "- `results.json`: run identities, metrics, thresholds, masks, pairwise summaries, and limitations.", "- `per_link_metrics.csv`: per-method directed-link metrics.", "- `paired_link_differences.csv`: per-seed/model-baseline per-link differences.", "- `subgroup_metrics.csv`: subgroup metrics and coverage.", ""])
    return "\n".join(lines)


if __name__ == "__main__":
    run()
