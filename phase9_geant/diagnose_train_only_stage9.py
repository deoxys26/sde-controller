"""Stage 9 training-only residual-signal audit (no model training).

Only the explicitly allow-listed X_train.npy/y_train.npy arrays and the
train-only fields needed from window_provenance_train_validation.json are
read. No validation/test arrays, run manifests, checkpoints, or predictions
are part of this diagnostic.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = (
    REPOSITORY_ROOT
    / "data"
    / "processed"
    / "geant_validity_v2_rebuild_trainval_20261017_v1"
)
DEFAULT_PROVENANCE = DEFAULT_DATA_ROOT / "window_provenance_train_validation.json"
DEFAULT_TOPOLOGY = REPOSITORY_ROOT / "data" / "raw" / "geant_topology" / "geant.txt"
ALLOWED_ARRAY_NAMES = frozenset({"X_train.npy", "y_train.npy"})
HISTORY_LENGTH = 5
EWMA_ALPHA = 0.8
RIDGE_ALPHA = 1.0
SHUFFLE_SEED = 20261017
SHUFFLE_REPLICATES = 20
LAGS = (1, 4, 96, 192, 672)
DIAGNOSTIC_EWMA_ALPHAS = (0.2, 0.4, 0.6, 0.8, 0.95)
PREDECLARED_THRESHOLDS = {
    "candidate_a_unsupported_below_percent": 0.5,
    "candidate_a_supported_at_or_above_percent": 1.0,
}


def _load_train_array(data_root: Path, filename: str) -> np.ndarray:
    if filename not in ALLOWED_ARRAY_NAMES:
        raise ValueError(f"Refusing array outside explicit train allowlist: {filename!r}")
    value = np.load(data_root / filename, allow_pickle=False)
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{filename} did not load as a NumPy array")
    return value


def _ewma_weights(alpha: float = EWMA_ALPHA) -> np.ndarray:
    raw = alpha * (1.0 - alpha) ** (
        HISTORY_LENGTH - 1 - np.arange(HISTORY_LENGTH, dtype=np.float64)
    )
    return raw / raw.sum(dtype=np.float64)


def _corr(left: np.ndarray, right: np.ndarray) -> float | None:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    if left.shape != right.shape or left.size < 2:
        return None
    left = left - left.mean(dtype=np.float64)
    right = right - right.mean(dtype=np.float64)
    denominator = np.sqrt(
        np.sum(left * left, dtype=np.float64)
        * np.sum(right * right, dtype=np.float64)
    )
    if denominator == 0.0:
        return None
    return float(np.sum(left * right, dtype=np.float64) / denominator)


def _contiguous_runs(indices: Iterable[int]) -> list[list[int]]:
    ordered = sorted(set(int(index) for index in indices))
    if not ordered:
        return []
    runs: list[list[int]] = []
    current = [ordered[0]]
    for index in ordered[1:]:
        if index != current[-1] + 1:
            runs.append(current)
            current = []
        current.append(index)
    runs.append(current)
    return runs


def _reconstruct(
    x_train: np.ndarray,
    y_train: np.ndarray,
    starts: Sequence[int],
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[int, np.ndarray], list[list[int]]]:
    if len(starts) != len(x_train):
        raise ValueError("Train provenance window starts do not align with X_train rows")
    if any(type(start) is not int for start in starts):
        raise ValueError("Training window starts must be integer indices")
    if list(starts) != sorted(set(starts)):
        raise ValueError("Training window starts must be unique and chronological")

    observations: dict[int, np.ndarray] = {}
    residuals: dict[int, np.ndarray] = {}
    ewma_by_target: dict[int, np.ndarray] = {}
    weights = _ewma_weights()
    for row, start in enumerate(starts):
        for offset in range(HISTORY_LENGTH):
            index = start + offset
            value = np.asarray(x_train[row, offset], dtype=np.float64)
            old = observations.get(index)
            if old is not None and not np.array_equal(old, value):
                raise ValueError(f"Overlapping train histories disagree at index {index}")
            observations[index] = value
        target_index = start + HISTORY_LENGTH
        target = np.asarray(y_train[row], dtype=np.float64)
        old_target = observations.get(target_index)
        if old_target is not None and not np.array_equal(old_target, target):
            raise ValueError(f"Overlapping train target/history disagree at {target_index}")
        observations[target_index] = target
        forecast = weights @ np.asarray(x_train[row], dtype=np.float64)
        if target_index in residuals:
            raise ValueError(f"Duplicate training target index {target_index}")
        ewma_by_target[target_index] = forecast
        residuals[target_index] = target - forecast

    return observations, residuals, ewma_by_target, _contiguous_runs(observations)


def _lag_correlations(
    values_by_index: dict[int, np.ndarray], lags: Sequence[int], link_count: int
) -> list[dict[str, Any]]:
    value_runs = _contiguous_runs(values_by_index)
    run_matrices = [
        np.stack([values_by_index[index] for index in run], axis=0).astype(
            np.float64, copy=False
        )
        for run in value_runs
    ]
    results: list[dict[str, Any]] = []
    for lag in lags:
        count = 0
        sum_x = np.zeros(link_count, dtype=np.float64)
        sum_y = np.zeros(link_count, dtype=np.float64)
        sum_xx = np.zeros(link_count, dtype=np.float64)
        sum_yy = np.zeros(link_count, dtype=np.float64)
        sum_xy = np.zeros(link_count, dtype=np.float64)
        pooled_n = 0
        pooled_x = pooled_y = pooled_xx = pooled_yy = pooled_xy = 0.0
        for matrix in run_matrices:
            if matrix.shape[0] <= lag:
                continue
            left, right = matrix[:-lag], matrix[lag:]
            count += left.shape[0]
            sum_x += left.sum(axis=0, dtype=np.float64)
            sum_y += right.sum(axis=0, dtype=np.float64)
            sum_xx += np.square(left).sum(axis=0, dtype=np.float64)
            sum_yy += np.square(right).sum(axis=0, dtype=np.float64)
            sum_xy += (left * right).sum(axis=0, dtype=np.float64)
            pooled_n += left.size
            pooled_x += float(left.sum(dtype=np.float64))
            pooled_y += float(right.sum(dtype=np.float64))
            pooled_xx += float(np.square(left).sum(dtype=np.float64))
            pooled_yy += float(np.square(right).sum(dtype=np.float64))
            pooled_xy += float((left * right).sum(dtype=np.float64))

        per_link_values: list[float | None] = []
        if count:
            covariance = sum_xy - sum_x * sum_y / count
            variance_x = np.maximum(sum_xx - sum_x * sum_x / count, 0.0)
            variance_y = np.maximum(sum_yy - sum_y * sum_y / count, 0.0)
            denominator = np.sqrt(variance_x * variance_y)
            for index in range(link_count):
                per_link_values.append(
                    float(covariance[index] / denominator[index])
                    if denominator[index] > 0.0
                    else None
                )
            pooled_cov = pooled_xy - pooled_x * pooled_y / pooled_n
            pooled_vx = max(pooled_xx - pooled_x * pooled_x / pooled_n, 0.0)
            pooled_vy = max(pooled_yy - pooled_y * pooled_y / pooled_n, 0.0)
            pooled_den = np.sqrt(pooled_vx * pooled_vy)
            pooled_corr = float(pooled_cov / pooled_den) if pooled_den > 0.0 else None
        else:
            per_link_values = [None] * link_count
            pooled_corr = None
        defined = [value for value in per_link_values if value is not None]
        results.append(
            {
                "lag_steps": int(lag),
                "per_link_mean_correlation": float(np.mean(defined)) if defined else None,
                "per_link_median_correlation": float(np.median(defined)) if defined else None,
                "pooled_scalar_correlation": pooled_corr,
                "link_lag_pairs": int(count * link_count),
                "pairs_per_link": [int(count)] * link_count,
                "defined_link_correlations": len(defined),
            }
        )
    return results


def _hour_bin(index: int, anchor: datetime, step_seconds: int) -> int:
    stamp = anchor + timedelta(seconds=index * step_seconds)
    return stamp.hour


def _directed_neighbors(
    undirected_links: Sequence[tuple[str, str]], link_count: int
) -> tuple[list[tuple[str, str]], list[list[int]]]:
    """Match the builder's interleaved edge order and construct link neighbors."""
    directed = [edge for u, v in undirected_links for edge in ((u, v), (v, u))]
    if len(directed) != link_count:
        raise ValueError(
            f"Topology has {len(directed)} directed links but arrays have {link_count} features"
        )
    neighbors: list[list[int]] = []
    for i, (src, dst) in enumerate(directed):
        adjacent = [
            j
            for j, (other_src, other_dst) in enumerate(directed)
            if j != i and ({src, dst} & {other_src, other_dst})
        ]
        reverse = next(
            (j for j, edge in enumerate(directed) if edge == (dst, src)), None
        )
        if reverse is None:
            raise ValueError(f"Topology is missing reverse directed edge {(dst, src)!r}")
        if reverse != i and reverse not in adjacent:
            adjacent.append(reverse)
        neighbors.append(sorted(adjacent))
    if any(not row for row in neighbors):
        raise ValueError("Every directed link must have at least one neighboring directed link")
    return directed, neighbors


def _spatial_signal_summary(
    observations: dict[int, np.ndarray],
    residuals: dict[int, np.ndarray],
    neighbors: Sequence[Sequence[int]],
    *,
    rng_seed: int = SHUFFLE_SEED,
    shuffle_replicates: int = SHUFFLE_REPLICATES,
) -> dict[str, Any]:
    """Descriptive train-only correlations; same-step residuals are noncausal."""
    if shuffle_replicates < 1:
        raise ValueError("At least one deterministic shuffle replicate is required")
    link_count = len(neighbors)
    last_use: list[np.ndarray] = []
    own_resid: list[np.ndarray] = []
    same_step_resid: list[np.ndarray] = []
    retained_targets: list[int] = []
    for q in sorted(residuals):
        if q - 1 not in observations:
            continue
        if residuals[q].shape != (link_count,) or observations[q - 1].shape != (link_count,):
            raise ValueError("Spatial arrays and topology link dimensions do not agree")
        if not np.isfinite(residuals[q]).all() or not np.isfinite(observations[q - 1]).all():
            continue
        last_use.append(observations[q - 1])
        own_resid.append(residuals[q])
        same_step_resid.append(residuals[q])
        retained_targets.append(q)
    if not retained_targets:
        raise ValueError("No aligned train-only observations for spatial signal summary")

    history = np.stack(last_use)
    residual = np.stack(own_resid)
    temporal_neighbor_values = np.column_stack(
        [history[:, row].mean(axis=1, dtype=np.float64) for row in neighbors]
    )
    same_step_neighbor_values = np.column_stack(
        [residual[:, row].mean(axis=1, dtype=np.float64) for row in neighbors]
    )
    rng = np.random.default_rng(rng_seed)
    shuffled_temporal: list[float | None] = []
    shuffled_same_step: list[float | None] = []
    for _ in range(shuffle_replicates):
        shuffled_neighbors = [
            sorted(rng.choice(link_count - 1, size=len(row), replace=False).astype(int).tolist())
            for row in neighbors
        ]
        # The draw above is over [0, E-2]; shift around each target index so
        # self-links remain excluded while each target degree is preserved.
        shuffled_neighbors = [
            sorted(j if j < i else j + 1 for j in row)
            for i, row in enumerate(shuffled_neighbors)
        ]
        shuf_temporal = np.column_stack(
            [history[:, row].mean(axis=1, dtype=np.float64) for row in shuffled_neighbors]
        )
        shuf_same = np.column_stack(
            [residual[:, row].mean(axis=1, dtype=np.float64) for row in shuffled_neighbors]
        )
        shuffled_temporal.append(_corr(residual, shuf_temporal))
        shuffled_same_step.append(_corr(residual, shuf_same))

    def distribution(values: Sequence[float | None]) -> dict[str, Any]:
        defined = np.asarray([value for value in values if value is not None], dtype=np.float64)
        return {
            "defined_replicates": int(defined.size),
            "mean": float(defined.mean(dtype=np.float64)) if defined.size else None,
            "p05": float(np.quantile(defined, 0.05)) if defined.size else None,
            "median": float(np.median(defined)) if defined.size else None,
            "p95": float(np.quantile(defined, 0.95)) if defined.size else None,
            "values": [None if value is None else float(value) for value in values],
        }

    return {
        "definition": (
            "For each directed link, neighbors are other directed links sharing either endpoint, "
            "including its reverse; each shuffle independently samples the same number of "
            "non-self directed links per target link. Correlations pool aligned (target, link) pairs."
        ),
        "target_indices": {"count": len(retained_targets), "first": retained_targets[0], "last": retained_targets[-1]},
        "directed_edge_count": link_count,
        "neighbor_count_per_link": [len(row) for row in neighbors],
        "rng_seed": rng_seed,
        "shuffle_replicates": shuffle_replicates,
        "causal_history_utilization": {
            "statistic": "corr(r(q), mean_neighbor(u(q-1)))",
            "real_neighbor_correlation": _corr(residual, temporal_neighbor_values),
            "shuffled_neighbor_distribution": distribution(shuffled_temporal),
            "interpretation": "historical neighbor utilization is available at forecast time; correlation is descriptive, not a trained predictive gain",
        },
        "same_window_residual_control": {
            "statistic": "corr(r(q), mean_neighbor(r(q)))",
            "real_neighbor_correlation": _corr(residual, same_step_neighbor_values),
            "shuffled_neighbor_distribution": distribution(shuffled_same_step),
            "interpretation": "same-target residual is unavailable at forecast time and is reported only as a contemporaneous co-movement control",
        },
    }


def _time_of_day_summary(
    residuals: dict[int, np.ndarray], anchor: datetime, step_seconds: int
) -> dict[str, Any]:
    bins: dict[int, list[np.ndarray]] = {hour: [] for hour in range(24)}
    all_values: list[np.ndarray] = []
    for index, values in residuals.items():
        bins[_hour_bin(index, anchor, step_seconds)].append(values)
        all_values.append(values)
    flat = np.concatenate(all_values).astype(np.float64, copy=False)
    grand_mean = float(flat.mean(dtype=np.float64))
    total_ss = float(np.sum((flat - grand_mean) ** 2, dtype=np.float64))
    between_ss = 0.0
    hour_rows: list[dict[str, Any]] = []
    for hour, parts in bins.items():
        if not parts:
            hour_rows.append({"hour": hour, "pair_count": 0, "mean_residual": None})
            continue
        values = np.concatenate(parts).astype(np.float64, copy=False)
        mean = float(values.mean(dtype=np.float64))
        between_ss += values.size * (mean - grand_mean) ** 2
        hour_rows.append(
            {"hour": hour, "pair_count": int(values.size), "mean_residual": mean}
        )
    return {
        "bins": hour_rows,
        "residual_scalar_count": int(flat.size),
        "residual_variance_explained_by_hour_means": (
            float(between_ss / total_ss) if total_ss > 0.0 else None
        ),
    }


def _ridge_prediction(
    x_fit: np.ndarray, y_fit: np.ndarray, x_eval: np.ndarray
) -> np.ndarray:
    """Ridge with train-only standardization, unpenalized intercept, alpha=1."""
    x_fit = np.asarray(x_fit, dtype=np.float64)
    y_fit = np.asarray(y_fit, dtype=np.float64).reshape(-1)
    x_eval = np.asarray(x_eval, dtype=np.float64)
    x_mean = x_fit.mean(axis=0, dtype=np.float64)
    x_scale = x_fit.std(axis=0, dtype=np.float64)
    x_scale[x_scale == 0.0] = 1.0
    y_mean = float(y_fit.mean(dtype=np.float64))
    y_scale = float(y_fit.std(dtype=np.float64))
    if y_scale == 0.0:
        return np.full(x_eval.shape[0], y_mean, dtype=np.float64)
    fit_x = (x_fit - x_mean) / x_scale
    eval_x = (x_eval - x_mean) / x_scale
    fit_y = (y_fit - y_mean) / y_scale
    gram = fit_x.T @ fit_x + RIDGE_ALPHA * np.eye(fit_x.shape[1], dtype=np.float64)
    coefficients = np.linalg.solve(gram, fit_x.T @ fit_y)
    return y_mean + y_scale * (eval_x @ coefficients)


def _regression_rows(
    observations: dict[int, np.ndarray],
    residuals: dict[int, np.ndarray],
    ewma_by_target: dict[int, np.ndarray],
    anchor: datetime,
    step_seconds: int,
) -> tuple[list[tuple[int, int]], np.ndarray, np.ndarray, np.ndarray]:
    """Create complete, causal feature rows; unavailable rows are dropped, never imputed."""
    keys: list[tuple[int, int]] = []
    features: list[list[float]] = []
    targets: list[float] = []
    ewma_values: list[float] = []
    for q in sorted(residuals):
        # Full q-96..q continuity is required; all features are strictly prior to q.
        if any(index not in observations for index in range(q - 96, q + 1)):
            continue
        if q - 96 not in residuals or q - 1 not in residuals:
            continue
        tod_angle = 2.0 * np.pi * _hour_bin(q, anchor, step_seconds) / 24.0
        for link in range(residuals[q].size):
            seasonal_change = float(observations[q - 95][link] - observations[q - 96][link])
            row = [
                seasonal_change,
                float(residuals[q - 96][link]),
                float(np.sin(tod_angle)),
                float(np.cos(tod_angle)),
                float(residuals[q - 1][link]),
            ]
            if not np.isfinite(row).all() or not np.isfinite(residuals[q][link]):
                continue
            keys.append((q, link))
            features.append(row)
            targets.append(float(residuals[q][link]))
            ewma_values.append(float(ewma_by_target[q][link]))
    return (
        keys,
        np.asarray(features, dtype=np.float64),
        np.asarray(targets, dtype=np.float64),
        np.asarray(ewma_values, dtype=np.float64),
    )


def _metric_pair(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    return {
        "mae": float(np.mean(np.abs(error), dtype=np.float64)),
        "rmse": float(np.sqrt(np.mean(error * error, dtype=np.float64))),
    }


def _chronological_ridge(
    keys: Sequence[tuple[int, int]],
    features: np.ndarray,
    residual_targets: np.ndarray,
    ewma_predictions: np.ndarray,
    reconstructed_indices: Sequence[int],
) -> dict[str, Any]:
    if not keys:
        raise ValueError("No complete train-only feature rows are available for ridge diagnostic")
    split_position = int(np.floor(0.70 * len(reconstructed_indices)))
    if split_position < 1:
        raise ValueError("Insufficient reconstructed training timeline")
    cutoff_index = int(reconstructed_indices[split_position - 1])
    holdout_start = cutoff_index + 97  # 96 excluded steps, then the first held-out index.
    fit_rows = np.asarray([i for i, (q, _) in enumerate(keys) if q <= cutoff_index], dtype=np.int64)
    eval_rows = np.asarray([i for i, (q, _) in enumerate(keys) if q >= holdout_start], dtype=np.int64)
    if not fit_rows.size or not eval_rows.size:
        raise ValueError("Chronological 70/30 train holdout lacks fit or evaluation rows")

    group_columns = {
        "seasonal_change_only": [0],
        "lag96_residual_only": [1],
        "time_of_day_only": [2, 3],
        "previous_step_residual_only": [4],
        "combined_all_features": [0, 1, 2, 3, 4],
    }
    # Utilization targets are reconstructed in the residual map; the caller
    # supplies residual targets, so evaluate reconstructed forecast errors.
    results: dict[str, Any] = {}
    ewma_target = ewma_predictions[eval_rows] + residual_targets[eval_rows]
    baseline = _metric_pair(ewma_target, ewma_predictions[eval_rows])
    for name, columns in group_columns.items():
        correction = _ridge_prediction(
            features[np.ix_(fit_rows, columns)],
            residual_targets[fit_rows],
            features[np.ix_(eval_rows, columns)],
        )
        forecast = ewma_predictions[eval_rows] + correction
        metrics = _metric_pair(ewma_target, forecast)
        metrics["mae_relative_reduction_vs_ewma_percent"] = (
            100.0 * (baseline["mae"] - metrics["mae"]) / baseline["mae"]
            if baseline["mae"] > 0.0
            else 0.0
        )
        metrics["rmse_relative_reduction_vs_ewma_percent"] = (
            100.0 * (baseline["rmse"] - metrics["rmse"]) / baseline["rmse"]
            if baseline["rmse"] > 0.0
            else 0.0
        )
        metrics["evaluated_window_link_pairs"] = int(eval_rows.size)
        metrics["fit_window_link_pairs"] = int(fit_rows.size)
        results[name] = metrics
    return {
        "split_rule": "first 70% of reconstructed timeline for fit; last 30% for evaluation with 96-step embargo",
        "fit_cutoff_index": cutoff_index,
        "first_heldout_target_index": int(min(keys[index][0] for index in eval_rows)),
        "embargo_steps": 96,
        "ridge_alpha": RIDGE_ALPHA,
        "fit_scaling": "feature and residual mean/std fitted on fit rows only; constant feature scales set to 1",
        "ewma_reference": baseline,
        "feature_set_results": results,
    }


def run_diagnostics(
    data_root: Path,
    provenance_path: Path,
    output_path: Path,
    topology_path: Path = DEFAULT_TOPOLOGY,
    topology_parser: Any | None = None,
) -> dict[str, Any]:
    data_root = Path(data_root)
    provenance_path = Path(provenance_path)
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite diagnostic output: {output_path}")

    # These are the only NumPy array reads in this module.
    x_train = _load_train_array(data_root, "X_train.npy")
    y_train = _load_train_array(data_root, "y_train.npy")
    if (
        x_train.ndim != 3
        or x_train.shape[1] != HISTORY_LENGTH
        or y_train.shape != (x_train.shape[0], x_train.shape[2])
        or not np.issubdtype(x_train.dtype, np.floating)
        or not np.issubdtype(y_train.dtype, np.floating)
        or not np.isfinite(x_train).all()
        or not np.isfinite(y_train).all()
    ):
        raise ValueError("Training arrays must be finite float arrays with [N,5,E] and [N,E] shapes")

    # Consume only the anchor, step, and training window starts from the
    # permitted train/validation JSON sidecar; no validation fields are used.
    sidecar = json.loads(provenance_path.read_text(encoding="utf-8"))
    anchor_value = sidecar.get("anchor_timestamp")
    step_seconds = sidecar.get("step_seconds")
    train = sidecar.get("splits", {}).get("train", {})
    starts = train.get("window_starts")
    if not isinstance(anchor_value, str):
        raise ValueError("Training provenance is missing anchor_timestamp")
    try:
        anchor = datetime.fromisoformat(anchor_value)
    except ValueError as exc:
        raise ValueError("Training provenance anchor_timestamp is invalid ISO format") from exc
    if type(step_seconds) is not int or step_seconds <= 0:
        raise ValueError("Training provenance is missing a positive step_seconds")
    if not isinstance(starts, list) or any(type(value) is not int for value in starts):
        raise ValueError("Training provenance is missing integer train window_starts")
    if not np.isfinite(x_train).all() or not np.isfinite(y_train).all():
        raise ValueError("Nonfinite train arrays are not imputed or dropped")

    observations, residuals, ewma_by_target, runs = _reconstruct(x_train, y_train, starts)
    if not residuals:
        raise ValueError("No training residual targets were reconstructed")
    link_count = int(x_train.shape[2])
    if topology_parser is None:
        from phase9_geant.parse_geant import parse_topology

        topology_parser = parse_topology
    topology_nodes, undirected_links = topology_parser(str(topology_path))
    if len(topology_nodes) != 22 or len(undirected_links) != 36:
        raise ValueError(
            f"Expected the documented topology structure (22 nodes/36 links); "
            f"found {len(topology_nodes)} nodes/{len(undirected_links)} links"
        )
    directed_edges, neighbors = _directed_neighbors(undirected_links, link_count)
    spatial = _spatial_signal_summary(observations, residuals, neighbors)
    residual_lags = _lag_correlations(residuals, LAGS, link_count)
    target_autocorrelation = _lag_correlations(
        observations, tuple(range(1, 2 * 24 * 4 + 1)), link_count
    )

    seasonal_residuals: list[np.ndarray] = []
    seasonal_changes: list[np.ndarray] = []
    for q, residual in residuals.items():
        if q - 96 not in observations or q - 95 not in observations:
            continue
        # Only pairs inside one fully contiguous reconstructed run are retained.
        if any(index not in observations for index in range(q - 96, q + 1)):
            continue
        seasonal_changes.append(observations[q - 95] - observations[q - 96])
        seasonal_residuals.append(residual)
    seasonal_residual_flat = np.concatenate(seasonal_residuals) if seasonal_residuals else np.array([])
    seasonal_change_flat = np.concatenate(seasonal_changes) if seasonal_changes else np.array([])
    seasonal_corr_per_link: list[float | None] = []
    if seasonal_residuals:
        residual_matrix = np.stack(seasonal_residuals)
        change_matrix = np.stack(seasonal_changes)
        for link in range(link_count):
            seasonal_corr_per_link.append(_corr(residual_matrix[:, link], change_matrix[:, link]))
    else:
        seasonal_corr_per_link = [None] * link_count

    time_summary = _time_of_day_summary(residuals, anchor, step_seconds)
    keys, features, residual_target_rows, ewma_rows = _regression_rows(
        observations, residuals, ewma_by_target, anchor, step_seconds
    )
    reconstructed_indices = sorted(observations)
    ridge = _chronological_ridge(
        keys,
        features,
        residual_target_rows,
        ewma_rows,
        reconstructed_indices,
    )

    # Per-link target distribution and EWMA absolute-error concentration.
    targets = y_train.astype(np.float64, copy=False)
    history = x_train.astype(np.float64, copy=False)
    ewma_pred = _ewma_weights() @ history
    train_baseline_predictions: dict[str, np.ndarray] = {
        "last_value_persistence": history[:, -1, :],
        "five_step_simple_mean": history.mean(axis=1, dtype=np.float64),
    }
    for alpha in DIAGNOSTIC_EWMA_ALPHAS:
        train_baseline_predictions[f"ewma_alpha_{alpha:g}"] = _ewma_weights(alpha) @ history
    train_baselines = {
        name: {
            **_metric_pair(targets, prediction),
            "evaluated_window_link_pairs": int(targets.size),
        }
        for name, prediction in train_baseline_predictions.items()
    }
    residual_matrix = targets - ewma_pred
    all_abs = np.abs(residual_matrix).reshape(-1)
    total_abs = float(all_abs.sum(dtype=np.float64))
    per_link_mae = np.mean(np.abs(residual_matrix), axis=0, dtype=np.float64)
    link_error = np.sum(np.abs(residual_matrix), axis=0, dtype=np.float64)
    top_links = np.argsort(-link_error)[: min(10, link_count)]
    top_error = {
        f"top_{percent}_percent": {
            "pair_count": int(max(1, np.ceil((percent / 100.0) * all_abs.size))),
            "absolute_error_share": float(
                np.partition(all_abs, all_abs.size - max(1, int(np.ceil((percent / 100.0) * all_abs.size))))[
                    -max(1, int(np.ceil((percent / 100.0) * all_abs.size))) :
                ].sum(dtype=np.float64)
                / total_abs
            ) if total_abs else 0.0,
        }
        for percent in (1, 5)
    }
    target_quantiles = {
        str(q): float(value)
        for q, value in zip(
            (0.0, 0.5, 0.9, 0.95, 0.99, 1.0),
            np.quantile(targets.reshape(-1), (0.0, 0.5, 0.9, 0.95, 0.99, 1.0)),
        )
    }
    per_link_quantiles = [
        {
            "link_index": link,
            "q50": float(np.quantile(targets[:, link], 0.50)),
            "q90": float(np.quantile(targets[:, link], 0.90)),
            "q95": float(np.quantile(targets[:, link], 0.95)),
            "q99": float(np.quantile(targets[:, link], 0.99)),
        }
        for link in range(link_count)
    ]
    sorted_link_error = np.sort(link_error)
    ranks = np.arange(1, link_count + 1, dtype=np.float64)
    gini = (
        float(2.0 * np.sum(ranks * sorted_link_error) / (link_count * total_abs) - (link_count + 1) / link_count)
        if total_abs > 0.0
        else 0.0
    )

    result: dict[str, Any] = {
        "protocol": "Stage 9 train-only residual feature signal audit; diagnostic only, no model training",
        "read_allowlist": [
            "X_train.npy",
            "y_train.npy",
            "window_provenance_train_validation.json (anchor, step_seconds, train window_starts only)",
            str(Path(topology_path)),
        ],
        "split": "train only",
        "window_count": int(x_train.shape[0]),
        "history_length": HISTORY_LENGTH,
        "directed_link_count": link_count,
        "topology_path": str(Path(topology_path)),
        "topology_node_count": len(topology_nodes),
        "topology_undirected_link_count": len(undirected_links),
        "topology_directed_edge_order": "undirected topology record i expands immediately to (u,v),(v,u), matching build_validity_v2.py",
        "directed_edges": [list(edge) for edge in directed_edges],
        "step_seconds": step_seconds,
        "anchor_timestamp": anchor_value,
        "ewma_alpha": EWMA_ALPHA,
        "training_baselines": {
            "scope": "train arrays only; context diagnostics, not baseline selection or a protocol change",
            "sample_count_definition": "all aligned finite (training window, directed-link) targets; no imputation or exclusions",
            "ewma_alphas": list(DIAGNOSTIC_EWMA_ALPHAS),
            "metrics": train_baselines,
        },
        "ewma_weight_order": "oldest to newest; normalized alpha*(1-alpha)^(age from newest)",
        "target_distribution": {
            "target_count": int(targets.size),
            "exact_zero_fraction": float(np.mean(targets == 0.0)),
            "near_zero_abs_le_1e-6_fraction": float(np.mean(np.abs(targets) <= 1e-6)),
            "pooled_quantiles": target_quantiles,
            "per_link_quantiles": per_link_quantiles,
        },
        "residual_autocorrelation": {
            "definition": "Pearson correlation of EWMA residual r(q) with r(q-lag), within contiguous reconstructed residual runs only",
            "lags": residual_lags,
        },
        "temporal_context": {
            "step_seconds": step_seconds,
            "history_point_to_point_span_minutes": (HISTORY_LENGTH - 1) * step_seconds / 60.0,
            "history_start_to_target_span_minutes": HISTORY_LENGTH * step_seconds / 60.0,
            "history_point_to_point_span_hours": (HISTORY_LENGTH - 1) * step_seconds / 3600.0,
            "history_start_to_target_span_hours": HISTORY_LENGTH * step_seconds / 3600.0,
            "target_utilization_autocorrelation": {
                "definition": "Pearson correlation of reconstructed utilization u(q) with u(q-lag), within contiguous observation runs only",
                "lags_through_two_days": target_autocorrelation,
            },
        },
        "seasonal_change_correlation": {
            "feature": "c(q)=u(q-95)-u(q-96), the one-step utilization change one day before target q",
            "pooled_correlation": _corr(seasonal_residual_flat, seasonal_change_flat) if seasonal_residual_flat.size else None,
            "per_link_mean_correlation": float(np.mean([v for v in seasonal_corr_per_link if v is not None])) if any(v is not None for v in seasonal_corr_per_link) else None,
            "per_link_correlations": seasonal_corr_per_link,
            "scalar_pairs": int(seasonal_residual_flat.size),
            "gap_rule": "only targets with a complete contiguous reconstructed interval from q-96 through q are used",
        },
        "spatial_signal": spatial,
        "hour_of_day_residuals": time_summary,
        "ewma_error_concentration": {
            "total_absolute_error": total_abs,
            "largest_pair_shares": top_error,
            "per_link_gini": gini,
            "top_10_links_by_absolute_error_share": [
                {
                    "link_index": int(link),
                    "absolute_error_share": float(link_error[link] / total_abs) if total_abs else 0.0,
                    "mae": float(per_link_mae[link]),
                    "rmse": float(np.sqrt(np.mean(residual_matrix[:, link] ** 2, dtype=np.float64))),
                }
                for link in top_links
            ],
        },
        "chronological_ridge_signal_check": ridge,
        "missing_data": "No imputation, padding, or zero replacement; incomplete feature rows are unavailable and omitted from paired diagnostic rows.",
        "predeclared_decision_rule": PREDECLARED_THRESHOLDS,
        "interpretation_limit": "Training-only signal diagnostic; not validation performance, a trained model, or a change to frozen baselines.",
        "reconstructed_timeline": {
            "observation_count": int(len(observations)),
            "contiguous_run_count": len(runs),
            "run_lengths": [len(run) for run in runs],
            "no_pair_crosses_index_gap": True,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--provenance", type=Path, default=DEFAULT_PROVENANCE)
    parser.add_argument("--topology", type=Path, default=DEFAULT_TOPOLOGY)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(r"C:\Users\Admin\DN_audit\stage9_train_only_feature_audit.json"),
    )
    args = parser.parse_args(argv)
    result = run_diagnostics(args.data_root, args.provenance, args.output, args.topology)
    print(json.dumps({
        "output": str(args.output),
        "window_count": result["window_count"],
        "target_count": result["target_distribution"]["target_count"],
        "ridge": result["chronological_ridge_signal_check"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
