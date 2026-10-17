"""Read-only Stage 8 diagnostics over GEANT v2 training windows only.

This script explicitly opens X_train.npy and y_train.npy, the GEANT topology,
and the train/validation provenance sidecar (train fields only). It does not
load validation arrays, test artifacts, manifests, checkpoints, or model code.
The JSON output is an aggregate diagnostic, not a model evaluation or tuning
result.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from phase9_geant.parse_geant import parse_topology


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = (
    REPOSITORY_ROOT
    / "data"
    / "processed"
    / "geant_validity_v2_rebuild_trainval_20261017_v1"
)
TOPOLOGY_PATH = REPOSITORY_ROOT / "data" / "raw" / "geant_topology" / "geant.txt"
PROVENANCE_PATH = DATA_ROOT / "window_provenance_train_validation.json"
ALLOWED_ARRAYS = ("X_train.npy", "y_train.npy")
EWMA_ALPHA = 0.8
NEAR_ZERO_EPSILON = 1e-6
SHUFFLE_SEED = 20261017
SHUFFLE_REPLICATES = 20


def _load_train_array(name: str) -> np.ndarray:
    if name not in ALLOWED_ARRAYS:
        raise ValueError(f"Refusing non-training array name: {name!r}")
    path = DATA_ROOT / name
    values = np.load(path, allow_pickle=False)
    if not isinstance(values, np.ndarray):
        raise TypeError(f"{name} did not load as an ndarray")
    return values


def _ewma_weights(alpha: float, history_length: int) -> np.ndarray:
    raw = alpha * (1.0 - alpha) ** (
        history_length - 1 - np.arange(history_length, dtype=np.float64)
    )
    return raw / raw.sum(dtype=np.float64)


def _metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    target64 = np.asarray(target, dtype=np.float64)
    prediction64 = np.asarray(prediction, dtype=np.float64)
    if target64.shape != prediction64.shape:
        raise ValueError("Diagnostic prediction/target shapes differ")
    if not np.isfinite(target64).all() or not np.isfinite(prediction64).all():
        raise ValueError("Train diagnostic contains nonfinite values")
    error = prediction64 - target64
    return {
        "mae": float(np.mean(np.abs(error), dtype=np.float64)),
        "rmse": float(np.sqrt(np.mean(np.square(error), dtype=np.float64))),
        "scalar_targets": int(target64.size),
    }


def _gini(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if np.any(values < 0) or not np.isfinite(values).all():
        raise ValueError("Gini inputs must be finite and nonnegative")
    total = values.sum(dtype=np.float64)
    if total == 0:
        return 0.0
    ordered = np.sort(values)
    n = ordered.size
    ranks = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * np.sum(ranks * ordered) / (n * total)) - (n + 1) / n)


def _pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    left64 = np.asarray(left, dtype=np.float64).reshape(-1)
    right64 = np.asarray(right, dtype=np.float64).reshape(-1)
    if left64.shape != right64.shape or left64.size < 2:
        return None
    if not np.isfinite(left64).all() or not np.isfinite(right64).all():
        return None
    left_centered = left64 - left64.mean(dtype=np.float64)
    right_centered = right64 - right64.mean(dtype=np.float64)
    denom = np.sqrt(
        np.sum(left_centered**2, dtype=np.float64)
        * np.sum(right_centered**2, dtype=np.float64)
    )
    if denom == 0:
        return None
    return float(np.sum(left_centered * right_centered, dtype=np.float64) / denom)


def _top_error_share(error: np.ndarray, fraction: float) -> dict[str, float | int]:
    absolute = np.abs(np.asarray(error, dtype=np.float64)).reshape(-1)
    count = max(1, int(np.ceil(fraction * absolute.size)))
    total = absolute.sum(dtype=np.float64)
    share = (
        float(np.partition(absolute, absolute.size - count)[-count:].sum(dtype=np.float64) / total)
        if total > 0
        else 0.0
    )
    return {"pair_count": count, "share_of_total_absolute_error": share}


def _neighbor_indices(
    edges: Sequence[tuple[str, str]],
) -> list[np.ndarray]:
    rows: list[np.ndarray] = []
    for edge_index, (source, destination) in enumerate(edges):
        endpoints = {source, destination}
        rows.append(
            np.asarray(
                [
                    other_index
                    for other_index, other in enumerate(edges)
                    if other_index != edge_index and endpoints.intersection(other)
                ],
                dtype=np.int64,
            )
        )
    return rows


def _neighbor_mean(values: np.ndarray, neighbors: Sequence[np.ndarray]) -> np.ndarray:
    windows, links = values.shape
    result = np.empty((windows, links), dtype=np.float64)
    for link, indices in enumerate(neighbors):
        if indices.size == 0:
            raise ValueError(f"Directed link {link} has no topology neighbors")
        result[:, link] = values[:, indices].mean(axis=1, dtype=np.float64)
    return result


def _shuffle_neighbor_rows(
    neighbors: Sequence[np.ndarray], *, rng: np.random.Generator
) -> list[np.ndarray]:
    link_count = len(neighbors)
    shuffled: list[np.ndarray] = []
    for target, row in enumerate(neighbors):
        candidates = np.concatenate(
            (np.arange(target, dtype=np.int64), np.arange(target + 1, link_count, dtype=np.int64))
        )
        shuffled.append(
            np.sort(rng.choice(candidates, size=len(row), replace=False)).astype(np.int64)
        )
    return shuffled


def _reconstruct_training_series(
    x_train: np.ndarray,
    y_train: np.ndarray,
    starts: Sequence[int],
) -> tuple[list[tuple[int, np.ndarray]], dict[str, Any]]:
    if len(starts) != len(x_train):
        return [], {"available": False, "reason": "train provenance starts do not align with X_train rows"}
    by_index: dict[int, np.ndarray] = {}
    for row, start in enumerate(starts):
        for offset in range(5):
            index = int(start) + offset
            observation = np.asarray(x_train[row, offset], dtype=np.float64)
            existing = by_index.get(index)
            if existing is not None and not np.array_equal(existing, observation):
                return [], {"available": False, "reason": f"overlapping training windows disagree at index {index}"}
            by_index[index] = observation
        target_index = int(start) + 5
        target = np.asarray(y_train[row], dtype=np.float64)
        existing_target = by_index.get(target_index)
        if existing_target is not None and not np.array_equal(existing_target, target):
            return [], {"available": False, "reason": f"history/target windows disagree at index {target_index}"}
        by_index[target_index] = target

    sorted_indices = sorted(by_index)
    runs: list[tuple[int, np.ndarray]] = []
    run_start = sorted_indices[0] if sorted_indices else 0
    run_values: list[np.ndarray] = []
    previous: int | None = None
    for index in sorted_indices:
        if previous is not None and index != previous + 1:
            runs.append((run_start, np.stack(run_values, axis=0)))
            run_start, run_values = index, []
        run_values.append(by_index[index])
        previous = index
    if run_values:
        runs.append((run_start, np.stack(run_values, axis=0)))
    return runs, {
        "available": bool(runs),
        "reconstructed_observation_count": int(len(by_index)),
        "contiguous_run_count": len(runs),
        "run_lengths": [int(values.shape[0]) for _, values in runs],
    }


def _training_autocorrelation(
    runs: Sequence[tuple[int, np.ndarray]], *, max_lag: int
) -> list[dict[str, Any]]:
    if not runs:
        return []
    link_count = runs[0][1].shape[1]
    results: list[dict[str, Any]] = []
    for lag in range(1, max_lag + 1):
        per_link: list[float | None] = []
        valid_pair_counts: list[int] = []
        for link in range(link_count):
            left_parts: list[np.ndarray] = []
            right_parts: list[np.ndarray] = []
            for _, values in runs:
                if values.shape[0] > lag:
                    left_parts.append(values[:-lag, link])
                    right_parts.append(values[lag:, link])
            if not left_parts:
                per_link.append(None)
                valid_pair_counts.append(0)
                continue
            left = np.concatenate(left_parts)
            right = np.concatenate(right_parts)
            per_link.append(_pearson(left, right))
            valid_pair_counts.append(int(left.size))
        finite = [value for value in per_link if value is not None]
        results.append(
            {
                "lag_steps": lag,
                "lag_hours": lag * 900 / 3600,
                "mean_link_autocorrelation": float(np.mean(finite)) if finite else None,
                "median_link_autocorrelation": float(np.median(finite)) if finite else None,
                "links_with_defined_correlation": len(finite),
                "total_valid_pairs": int(sum(valid_pair_counts)),
            }
        )
    return results


def run_diagnostics(output_path: Path) -> dict[str, Any]:
    # Explicit allowlist: do not replace these reads with globbing or discovery.
    x_train = _load_train_array("X_train.npy")
    y_train = _load_train_array("y_train.npy")
    if (
        x_train.dtype != np.float32
        or y_train.dtype != np.float32
        or x_train.ndim != 3
        or x_train.shape[1] != 5
        or y_train.shape != (x_train.shape[0], x_train.shape[2])
        or not np.isfinite(x_train).all()
        or not np.isfinite(y_train).all()
    ):
        raise ValueError("Training arrays do not satisfy the finite float32 five-history schema")

    sidecar = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))
    step_seconds = sidecar.get("step_seconds")
    train_block = sidecar.get("splits", {}).get("train", {})
    starts = train_block.get("window_starts")
    if type(step_seconds) is not int or step_seconds <= 0:
        raise ValueError("Train/validation provenance has no valid step_seconds")
    if not isinstance(starts, list) or any(type(value) is not int for value in starts):
        raise ValueError("Train provenance has no valid window-start indices")

    nodes, undirected = parse_topology(str(TOPOLOGY_PATH))
    edges = tuple(
        directed
        for source, destination in undirected
        for directed in ((source, destination), (destination, source))
    )
    if len(nodes) != 22 or len(undirected) != 36 or len(edges) != x_train.shape[2]:
        raise ValueError("Training link columns do not match the 22/36/72 topology contract")

    target = y_train.astype(np.float64)
    history = x_train.astype(np.float64)
    near_zero = np.abs(target) <= NEAR_ZERO_EPSILON
    zero_fraction = float(np.count_nonzero(target == 0) / target.size)
    near_zero_fraction = float(np.count_nonzero(near_zero) / target.size)
    quantiles = (0.0, 0.5, 0.9, 0.95, 0.99, 1.0)
    target_distribution = {
        "scalar_target_count": int(target.size),
        "exact_zero_count": int(np.count_nonzero(target == 0)),
        "exact_zero_fraction": zero_fraction,
        "near_zero_threshold_absolute_utilization": NEAR_ZERO_EPSILON,
        "near_zero_count_including_exact_zero": int(np.count_nonzero(near_zero)),
        "near_zero_fraction_including_exact_zero": near_zero_fraction,
        "overall_quantiles": {
            str(q): float(np.quantile(target, q)) for q in quantiles
        },
        "overall_mean": float(target.mean(dtype=np.float64)),
    }
    per_link_target = []
    for link, edge in enumerate(edges):
        values = target[:, link]
        per_link_target.append(
            {
                "link_index": link,
                "edge": list(edge),
                "exact_zero_fraction": float(np.mean(values == 0)),
                "near_zero_fraction": float(np.mean(np.abs(values) <= NEAR_ZERO_EPSILON)),
                "quantiles": {str(q): float(np.quantile(values, q)) for q in quantiles},
            }
        )

    predictions: dict[str, np.ndarray] = {
        "persistence_last_value": history[:, -1, :],
        "mean_of_five": history.mean(axis=1, dtype=np.float64),
    }
    for alpha in (0.2, 0.4, 0.6, 0.8, 0.95):
        weights = _ewma_weights(alpha, history.shape[1])
        predictions[f"ewma_alpha_{alpha:.2f}"] = np.tensordot(
            history, weights, axes=([1], [0])
        )
    baseline_metrics = {name: _metrics(target, value) for name, value in predictions.items()}
    ewma_prediction = predictions[f"ewma_alpha_{EWMA_ALPHA:.2f}"]
    ewma_error = ewma_prediction - target
    absolute_error = np.abs(ewma_error)
    per_link_abs_error = absolute_error.sum(axis=0, dtype=np.float64)
    total_abs_error = float(per_link_abs_error.sum(dtype=np.float64))
    per_link_diagnostics = []
    for link, edge in enumerate(edges):
        e = ewma_error[:, link]
        per_link_diagnostics.append(
            {
                "link_index": link,
                "edge": list(edge),
                "ewma_mae": float(np.mean(np.abs(e), dtype=np.float64)),
                "ewma_rmse": float(np.sqrt(np.mean(e**2, dtype=np.float64))),
                "absolute_error_sum": float(per_link_abs_error[link]),
                "share_of_total_ewma_absolute_error": float(per_link_abs_error[link] / total_abs_error),
            }
        )
    top_links = sorted(
        per_link_diagnostics,
        key=lambda item: item["share_of_total_ewma_absolute_error"],
        reverse=True,
    )[:10]
    sorted_link_error = np.sort(per_link_abs_error)
    cumulative_link_error = float(
        np.cumsum(sorted_link_error, dtype=np.float64)[int(np.ceil(0.8 * len(sorted_link_error))) - 1]
        / total_abs_error
    )
    error_concentration = {
        "top_1_percent_window_link_pairs": _top_error_share(ewma_error, 0.01),
        "top_5_percent_window_link_pairs": _top_error_share(ewma_error, 0.05),
        "link_error_gini": _gini(per_link_abs_error),
        "top_10_link_error_share": float(sum(item["share_of_total_ewma_absolute_error"] for item in top_links)),
        "top_10_links": top_links,
        "per_link_ewma_metrics": per_link_diagnostics,
        "share_from_lowest_80_percent_of_links": cumulative_link_error,
    }

    neighbors = _neighbor_indices(edges)
    last_utilization = history[:, -1, :]
    neighbor_last_mean = _neighbor_mean(last_utilization, neighbors)
    neighbor_residual_mean = _neighbor_mean(ewma_error, neighbors)
    spatial_results: dict[str, Any] = {}
    rng = np.random.default_rng(SHUFFLE_SEED)
    for label, signal in (
        ("neighbor_mean_last_step_utilization", neighbor_last_mean),
        ("same_window_neighbor_mean_ewma_residual", neighbor_residual_mean),
    ):
        observed = _pearson(ewma_error, signal)
        shuffled_correlations = []
        for _ in range(SHUFFLE_REPLICATES):
            randomized = _shuffle_neighbor_rows(neighbors, rng=rng)
            shuffled_signal = _neighbor_mean(
                last_utilization if label == "neighbor_mean_last_step_utilization" else ewma_error,
                randomized,
            )
            shuffled_correlations.append(_pearson(ewma_error, shuffled_signal))
        numeric_shuffles = [value for value in shuffled_correlations if value is not None]
        q95 = float(np.quantile(numeric_shuffles, 0.95)) if numeric_shuffles else None
        spatial_results[label] = {
            "real_neighbor_pearson_r": observed,
            "randomized_neighbor_assignments": numeric_shuffles,
            "shuffle_mean_r": float(np.mean(numeric_shuffles)) if numeric_shuffles else None,
            "shuffle_sd_r": float(np.std(numeric_shuffles, ddof=1)) if len(numeric_shuffles) > 1 else None,
            "shuffle_95th_percentile_r": q95,
            "real_exceeds_shuffle_95th_percentile": bool(
                observed is not None and q95 is not None and observed > q95
            ),
            "interpretation_rule": "descriptive only; real correlation must exceed the 20-shuffle 95th percentile to be called clearly above this control",
        }

    runs, reconstruction = _reconstruct_training_series(history, target, starts)
    max_lag = int((2 * 24 * 3600) // step_seconds)
    autocorrelation = (
        _training_autocorrelation(runs, max_lag=max_lag)
        if reconstruction["available"]
        else []
    )
    history_span_seconds = (history.shape[1] - 1) * step_seconds
    full_window_span_seconds = history.shape[1] * step_seconds

    result: dict[str, Any] = {
        "experiment": "Stage 8 training-only diagnostics; no model training or validation/test evaluation",
        "data_root": str(DATA_ROOT),
        "input_files_opened": [str(DATA_ROOT / name) for name in ALLOWED_ARRAYS]
        + [str(TOPOLOGY_PATH), str(PROVENANCE_PATH)],
        "split": "train only",
        "window_count": int(x_train.shape[0]),
        "history_steps": int(x_train.shape[1]),
        "directed_links": int(x_train.shape[2]),
        "units": "dimensionless computed link-utilization ratio",
        "threshold_definitions": {
            "near_zero_absolute_utilization": NEAR_ZERO_EPSILON,
            "ewma_top_error_concentration": "largest absolute per-(window,link) prediction errors by sum",
        },
        "target_distribution": target_distribution,
        "per_link_target_distribution": per_link_target,
        "training_baseline_metrics": baseline_metrics,
        "ewma_error_concentration": error_concentration,
        "spatial_signal": {
            "neighbor_definition": "all other directed topology links sharing either endpoint; includes each link's reverse direction",
            "neighbor_counts": {
                "minimum": min(len(row) for row in neighbors),
                "median": float(np.median([len(row) for row in neighbors])),
                "maximum": max(len(row) for row in neighbors),
            },
            "randomization": f"{SHUFFLE_REPLICATES} independent per-link neighbor sets sampled without replacement, preserving each target link's neighbor count and excluding itself; RNG seed {SHUFFLE_SEED}",
            "statistics": spatial_results,
        },
        "temporal": {
            "step_seconds": step_seconds,
            "history_span_minutes": history_span_seconds / 60,
            "history_span_hours": history_span_seconds / 3600,
            "history_plus_target_span_minutes": full_window_span_seconds / 60,
            "history_plus_target_span_hours": full_window_span_seconds / 3600,
            "reconstruction": reconstruction,
            "target_autocorrelation_through_two_days": autocorrelation,
            "maximum_lag_steps": max_lag,
            "maximum_lag_hours": max_lag * step_seconds / 3600,
            "gap_rule": "correlations use pairs contained in reconstructed contiguous training runs only; no pair crosses a missing-index gap",
        },
        "interpretation_limits": [
            "This is a descriptive train-only diagnostic; it is not a model comparison on validation or test data.",
            "Random-neighbor correlations are controls, not formal inferential p-values.",
            "Training baselines are not replacements for the frozen evaluation baselines.",
        ],
    }
    output_path = Path(output_path)
    resolved_output = Path(os.path.abspath(output_path))
    if resolved_output.is_relative_to(REPOSITORY_ROOT):
        raise ValueError("Diagnostic output must be outside the repository")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite diagnostic report: {output_path}")
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_diagnostics(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "window_count": result["window_count"],
                "target_distribution": result["target_distribution"],
                "baseline_metrics": result["training_baseline_metrics"],
                "top_10_links": result["ewma_error_concentration"]["top_10_links"],
                "spatial_signal": result["spatial_signal"],
                "temporal_summary": {
                    key: value
                    for key, value in result["temporal"].items()
                    if key not in {"target_autocorrelation_through_two_days"}
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
