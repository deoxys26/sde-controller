"""Causal Ridge screen and disabled-by-default horizon-v2 training utilities.

All file reads go through ``horizon_v2_data.load_horizon_array`` and its exact
train/validation allowlist. This module contains no test split path or loader.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import random
import argparse
import time
import tempfile
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from phase9_geant.horizon_v2_data import (
    BUILD_CONFIGURATION,
    EXPECTED_START,
    EXPECTED_START_SOURCE,
    HORIZONS,
    HISTORY_LENGTH,
    OFFICIAL_SPLIT_BOUNDS,
    PROTOCOL_ID,
    CAPACITY_MBPS,
    RAW_DIRECTORY_RELATIVE,
    TOPOLOGY_RELATIVE,
    _topology_snapshot,
    _edge_digest,
    _canonical_horizon_root,
    _assert_artifact_inventory,
    _current_build_source_hashes,
    _routing_paths,
    expected_split_paths,
    load_horizon_array,
    validate_artifact_name,
)
from src.gnn.spatiotemporal_link import SpatioTemporalLinkForecaster
from phase9_geant.safe_paths import canonical_repository_root, checked_absolute
from phase9_geant import validity_v2
from phase9_geant.horizon_windows import enumerate_horizon_windows


SEEDS = (42, 123, 2026)
RIDGE_ALPHA = 1.0
SCREEN_FIT_FRACTION = 0.70
SCREEN_EMBARGO_STEPS = 768
SCREEN_THRESHOLDS = {"stop_below": 0.005, "review_at_or_above": 0.01}
SOURCE_IDENTITY_FILES = (
    "phase9_geant/__init__.py",
    "phase9_geant/horizon_v2_experiment.py",
    "phase9_geant/horizon_v2_data.py",
    "phase9_geant/horizon_windows.py",
    "phase9_geant/safe_paths.py",
    "phase9_geant/validity_v2.py",
    "phase9_geant/build_link_series.py",
    "phase9_geant/parse_geant.py",
    # Importing src.gnn.spatiotemporal_link first runs src.gnn.__init__.
    "src/gnn/__init__.py",
    "src/gnn/dataset.py",
    "src/gnn/preprocessing.py",
    "src/gnn/model.py",
    "src/gnn/spatiotemporal_link.py",
)
TRAINING_CONFIG = {
    "epochs": 20,
    "patience": 4,
    "learning_rate": 0.005,
    "batch_size": 32,
    "optimizer": "Adam",
    "loss": "SmoothL1(beta=0.001, original residual utilization units)",
    "hidden_size": 16,
    "alpha": 0.8,
    "ewma_steps": 5,
    "seeds": list(SEEDS),
}


@dataclass(frozen=True)
class Standardizer:
    mean: float
    scale: float
    fit_scope: str


@dataclass(frozen=True)
class RidgeScreenResult:
    horizon: int
    fit_origins: int
    holdout_origins: int
    holdout_pairs: int
    embargo_steps: int
    ewma_mae: float
    ridge_ewma_mae: float
    ewma_rmse: float
    ridge_ewma_rmse: float
    relative_mae_reduction: float
    residual_r2: float
    decision: str


def _ewma(history: np.ndarray, *, alpha: float = 0.8, steps: int = 5) -> np.ndarray:
    if history.ndim != 3 or history.shape[1] < steps:
        raise ValueError("history must have shape [N, T>=steps, E]")
    weights = np.asarray([alpha * (1.0 - alpha) ** (steps - 1 - i) for i in range(steps)], dtype=np.float64)
    weights /= weights.sum()
    return np.einsum("nte,t->ne", history[:, -steps:, :].astype(np.float64), weights)


def baseline_forecasts(history: np.ndarray, horizon: int) -> dict[str, np.ndarray]:
    """Return four fixed causal forecasts for aligned 672-step histories."""
    x = np.asarray(history, dtype=np.float64)
    if x.ndim != 3 or x.shape[1] != HISTORY_LENGTH or horizon not in HORIZONS:
        raise ValueError("baseline history/horizon does not match the frozen protocol")
    if not np.isfinite(x).all():
        raise ValueError("baseline history contains invalid values")
    origin = HISTORY_LENGTH - 1
    return {
        "ewma": _ewma(x, alpha=0.8, steps=5),
        "persistence": x[:, -1, :],
        "same_time_yesterday": x[:, origin + horizon - 96, :],
        "same_time_last_week": x[:, origin + horizon - 672, :],
    }


def metric_summary(target: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    """Compute dimensionless utilization MAE/RMSE on exact aligned finite pairs."""
    y = np.asarray(target, dtype=np.float64)
    p = np.asarray(prediction, dtype=np.float64)
    if y.shape != p.shape or y.size == 0:
        raise ValueError("metric target and prediction shapes must match and be nonempty")
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("metric inputs contain missing/nonfinite values; no implicit pair dropping")
    error = p - y
    return {
        "evaluated_target_link_pairs": int(y.size),
        "coverage": 1.0,
        "mae": float(np.mean(np.abs(error), dtype=np.float64)),
        "rmse": float(np.sqrt(np.mean(np.square(error), dtype=np.float64))),
        "units": "computed utilization ratio",
    }


def validation_loss_improved(candidate: float, best: float) -> bool:
    """Checkpoint rule: strict lower validation objective only."""
    return bool(np.isfinite(candidate) and candidate < best)


def build_ridge_features(
    history: np.ndarray,
    edge_order: Sequence[tuple[str, str]],
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return fixed protocol features and flat EWMA for train-only origins.

    Features, in order: own same-time-yesterday residual, own same-time-last-
    week residual, mean neighboring current residual, mean neighboring
    four-step change. Neighbor sets use shared endpoints (reverse direction is
    included) but exclude the target link itself; values use history only.
    """
    x = np.asarray(history)
    if x.ndim != 3 or x.shape[1] != HISTORY_LENGTH or x.shape[2] != len(edge_order):
        raise ValueError(f"history must be [N,{HISTORY_LENGTH},len(edge_order)]")
    if horizon not in HORIZONS or not np.isfinite(x).all():
        raise ValueError("unsupported horizon or nonfinite history")
    baseline = _ewma(x)
    origin = HISTORY_LENGTH - 1
    daily_index = origin + horizon - 96
    weekly_index = origin + horizon - 672
    if min(daily_index, weekly_index) < 0 or max(daily_index, weekly_index) > origin:
        raise ValueError("seasonal reference is not available inside the history")
    own_daily = x[:, daily_index, :] - baseline
    own_weekly = x[:, weekly_index, :] - baseline
    own_change = x[:, -1, :] - x[:, -5, :]

    if len(set(edge_order)) != len(edge_order):
        raise ValueError("edge_order must be unique and ordered")
    endpoint_sets = [set(edge) for edge in edge_order]
    neighbor_indices = [
        [j for j, endpoints in enumerate(endpoint_sets) if j != i and endpoints & endpoint_sets[i]]
        for i in range(len(edge_order))
    ]
    neighbor_residual = np.empty_like(baseline)
    neighbor_change = np.empty_like(baseline)
    for link, neighbors in enumerate(neighbor_indices):
        if not neighbors:
            raise ValueError(f"Link {link} has no neighboring directed links")
        neighbor_residual[:, link] = np.mean(x[:, -1, neighbors] - baseline[:, neighbors], axis=1)
        neighbor_change[:, link] = np.mean(own_change[:, neighbors], axis=1)
    features = np.stack((own_daily, own_weekly, neighbor_residual, neighbor_change), axis=-1)
    if not np.isfinite(features).all() or not np.isfinite(baseline).all():
        raise ValueError("feature construction produced nonfinite values")
    return features.astype(np.float64), baseline


def ridge_features_from_training_series(
    utilization: np.ndarray,
    origins: Sequence[int],
    edge_order: Sequence[tuple[str, str]],
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized causal Ridge features from one training-only series.

    Each origin indexes a fully valid 672-step history. No target values are
    accepted; callers construct residuals separately from the training-only
    target indices.
    """
    series = np.asarray(utilization, dtype=np.float64)
    origin_values = np.asarray(origins, dtype=np.int64)
    if series.ndim != 2 or series.shape[1] != len(edge_order):
        raise ValueError("training utilization must be [time, ordered_links]")
    if horizon not in HORIZONS or not len(origin_values):
        raise ValueError("unsupported horizon or empty origin set")
    if np.any(np.diff(origin_values) <= 0) or origin_values[0] < HISTORY_LENGTH - 1:
        raise ValueError("origins must be chronological and have full history")
    if origin_values[-1] + horizon >= len(series):
        raise ValueError("training target index exceeds supplied training series")
    ewma_weights = np.asarray([0.8 * (1 - 0.8) ** (4 - i) for i in range(5)], dtype=np.float64)
    ewma_weights /= ewma_weights.sum()
    recent = series[origin_values[:, None] - np.arange(4, -1, -1)]
    required_indices = np.concatenate(
        (origin_values[:, None] - np.arange(HISTORY_LENGTH - 1, -1, -1), (origin_values + horizon)[:, None]),
        axis=1,
    )
    if not np.isfinite(series[required_indices]).all():
        raise ValueError("a selected history/target window contains invalid values")
    baseline = np.einsum("nte,t->ne", recent, ewma_weights)
    own_daily = series[origin_values + horizon - 96] - baseline
    own_weekly = series[origin_values + horizon - 672] - baseline
    current_residual = series[origin_values] - baseline
    four_step_change = series[origin_values] - series[origin_values - 4]
    if len(set(edge_order)) != len(edge_order):
        raise ValueError("edge_order must be unique and ordered")
    endpoint_sets = [set(edge) for edge in edge_order]
    features = np.empty((len(origin_values), len(edge_order), 4), dtype=np.float64)
    for link, endpoints in enumerate(endpoint_sets):
        neighbors = [index for index, candidate in enumerate(endpoint_sets) if index != link and endpoints & candidate]
        if not neighbors:
            raise ValueError(f"Link {link} has no adjacent directed links")
        features[:, link, 0] = own_daily[:, link]
        features[:, link, 1] = own_weekly[:, link]
        features[:, link, 2] = current_residual[:, neighbors].mean(axis=1)
        features[:, link, 3] = four_step_change[:, neighbors].mean(axis=1)
    residual = series[origin_values + horizon] - baseline
    if not np.isfinite(features).all() or not np.isfinite(residual).all():
        raise ValueError("causal training feature construction produced nonfinite values")
    return features, residual


def screen_training_utilization(
    utilization: np.ndarray,
    valid_mask: Sequence[bool],
    timestamps_seconds: Sequence[int],
    edge_order: Sequence[tuple[str, str]],
) -> tuple[dict[int, RidgeScreenResult], dict[int, dict[str, int]]]:
    """Run the fixed horizon screens on one official training-only series."""
    series = np.asarray(utilization, dtype=np.float64)
    if series.shape != (8020, len(edge_order)) or len(valid_mask) != 8020 or len(timestamps_seconds) != 8020:
        raise ValueError("Ridge screen requires exactly the official training interval")
    results: dict[int, RidgeScreenResult] = {}
    counts: dict[int, dict[str, int]] = {}
    for horizon in HORIZONS:
        windows = enumerate_horizon_windows(
            "train", valid_mask, timestamps_seconds, history_length=HISTORY_LENGTH,
            horizons=(horizon,), step_seconds=900,
        )[horizon]
        origins = np.asarray([window.origin for window in windows], dtype=np.int64)
        if not len(origins):
            raise ValueError(f"No valid train windows for horizon {horizon}")
        features, residual = ridge_features_from_training_series(series, origins, edge_order, horizon)
        results[horizon] = ridge_screen_from_features(features, residual, origins, horizon)
        potential = max(0, 8020 - HISTORY_LENGTH + 1 - horizon)
        counts[horizon] = {
            "potential_origins": potential,
            "accepted_origins": len(windows),
            "rejected_origins": potential - len(windows),
            "fit_origins": results[horizon].fit_origins,
            "embargoed_eligible_origins": len(windows) - results[horizon].fit_origins - results[horizon].holdout_origins,
            "holdout_origins": results[horizon].holdout_origins,
        }
    return results, counts


def fit_train_only_standardizer(values: np.ndarray) -> Standardizer:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("cannot fit scaler on empty or nonfinite training values")
    scale = float(array.std(ddof=0))
    if scale <= np.finfo(np.float64).eps:
        scale = 1.0
    return Standardizer(float(array.mean()), scale, "training arrays only")


def chronological_ridge_screen(
    history: np.ndarray,
    target: np.ndarray,
    origins: Sequence[int],
    edge_order: Sequence[tuple[str, str]],
    horizon: int,
) -> RidgeScreenResult:
    """Fixed Ridge screen with 70%-prefix fit, 768-step embargo, later holdout.

    Origins are supplied in strictly increasing order. The fit prefix is the
    first floor(70%); holdout rows begin no earlier than 768 raw steps after
    the last fit origin. Feature/target standardizers are fitted on the fit
    prefix only. Validation and test data are not accepted by this function.
    """
    x = np.asarray(history, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    origin_values = np.asarray(origins, dtype=np.int64)
    if x.ndim != 3 or x.shape[1:] != (HISTORY_LENGTH, len(edge_order)):
        raise ValueError("history shape does not match the protocol")
    if y.shape != (len(x), len(edge_order)) or len(origin_values) != len(x):
        raise ValueError("target/origin alignment mismatch")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("screen inputs must be finite valid training windows")
    if np.any(np.diff(origin_values) <= 0):
        raise ValueError("origins must be strictly chronological")
    fit_count = int(np.floor(len(x) * SCREEN_FIT_FRACTION))
    if fit_count < 2 or fit_count >= len(x):
        raise ValueError("not enough eligible origins for the fixed 70/30 screen")

    feature, baseline = build_ridge_features(x, edge_order, horizon)
    residual = y - baseline
    fit_mask = np.arange(len(x)) < fit_count
    holdout_mask = (np.arange(len(x)) >= fit_count) & (
        origin_values >= origin_values[fit_count - 1] + SCREEN_EMBARGO_STEPS
    )
    if not holdout_mask.any():
        raise ValueError("768-step embargo leaves no chronological holdout origins")

    return ridge_screen_from_features(feature, residual, origin_values, horizon)


def ridge_screen_from_features(
    feature: np.ndarray,
    residual: np.ndarray,
    origins: Sequence[int],
    horizon: int,
) -> RidgeScreenResult:
    """Fit the same fixed Ridge screen from precomputed train-only features."""
    feature = np.asarray(feature, dtype=np.float64)
    residual = np.asarray(residual, dtype=np.float64)
    origin_values = np.asarray(origins, dtype=np.int64)
    if horizon not in HORIZONS or feature.ndim != 3 or residual.shape != feature.shape[:2]:
        raise ValueError("feature/residual shape or horizon mismatch")
    if len(origin_values) != len(feature) or np.any(np.diff(origin_values) <= 0):
        raise ValueError("screen origins must align and be strictly chronological")
    if not np.isfinite(feature).all() or not np.isfinite(residual).all():
        raise ValueError("screen features and residuals must be finite")
    fit_count = int(np.floor(len(feature) * SCREEN_FIT_FRACTION))
    if fit_count < 2 or fit_count >= len(feature):
        raise ValueError("not enough eligible origins for the fixed 70/30 screen")
    fit_mask = np.arange(len(feature)) < fit_count
    holdout_mask = (np.arange(len(feature)) >= fit_count) & (
        origin_values >= origin_values[fit_count - 1] + SCREEN_EMBARGO_STEPS
    )
    if not holdout_mask.any():
        raise ValueError("768-step embargo leaves no chronological holdout origins")

    fit_features = feature[fit_mask].reshape(-1, feature.shape[-1])
    fit_residual = residual[fit_mask].reshape(-1)
    x_mean = fit_features.mean(axis=0)
    x_scale = fit_features.std(axis=0)
    x_scale[x_scale <= np.finfo(np.float64).eps] = 1.0
    y_mean = float(fit_residual.mean())
    y_scale = float(fit_residual.std())
    if y_scale <= np.finfo(np.float64).eps:
        y_scale = 1.0
    zfit = (fit_features - x_mean) / x_scale
    rfit = (fit_residual - y_mean) / y_scale
    design = np.column_stack((np.ones(len(zfit)), zfit))
    penalty = np.eye(design.shape[1], dtype=np.float64) * RIDGE_ALPHA
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ rfit)

    hold_features = feature[holdout_mask].reshape(-1, feature.shape[-1])
    hold_residual = residual[holdout_mask].reshape(-1)
    hold_design = np.column_stack((np.ones(len(hold_features)), (hold_features - x_mean) / x_scale))
    predicted_residual = (hold_design @ coefficients) * y_scale + y_mean
    absolute = np.abs(hold_residual)
    ridge_absolute = np.abs(hold_residual - predicted_residual)
    total_sum_squares = float(np.square(hold_residual - hold_residual.mean()).sum())
    residual_r2 = (
        1.0 - float(np.square(hold_residual - predicted_residual).sum()) / total_sum_squares
        if total_sum_squares
        else 0.0
    )
    ewma_mae = float(absolute.mean())
    ridge_mae = float(ridge_absolute.mean())
    reduction = (ewma_mae - ridge_mae) / ewma_mae if ewma_mae else 0.0
    decision = "stop" if reduction < 0.005 else ("candidate-design-review" if reduction >= 0.01 else "pause-for-review")
    return RidgeScreenResult(
        horizon=horizon,
        fit_origins=int(fit_mask.sum()),
        holdout_origins=int(holdout_mask.sum()),
        holdout_pairs=int(holdout_mask.sum() * residual.shape[1]),
        embargo_steps=SCREEN_EMBARGO_STEPS,
        ewma_mae=ewma_mae,
        ridge_ewma_mae=ridge_mae,
        ewma_rmse=float(np.sqrt(np.mean(np.square(hold_residual)))),
        ridge_ewma_rmse=float(np.sqrt(np.mean(np.square(hold_residual - predicted_residual)))),
        relative_mae_reduction=reduction,
        residual_r2=residual_r2,
        decision=decision,
    )


def load_train_arrays_for_screen(
    data_root: str | Path,
    *,
    repository_root: str | Path,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], tuple[tuple[str, str], ...], dict]:
    """Load only train horizon arrays and their metadata/index sidecar."""
    return _load_split_arrays(data_root, "train", repository_root=repository_root)


def _load_split_arrays(
    data_root: str | Path,
    split: str,
    *,
    repository_root: str | Path,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], tuple[tuple[str, str], ...], dict]:
    if split not in ("train", "validation"):
        raise ValueError("Only train and validation horizon artifacts are supported")
    root = _canonical_horizon_root(data_root, repository_root)
    _assert_artifact_inventory(root)
    from phase9_geant.safe_paths import require_regular_file

    metadata_path = require_regular_file(root / validate_artifact_name("metadata.json"), root)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("protocol_id") != PROTOCOL_ID or metadata.get("status") != "complete":
        raise ValueError("Horizon-v2 metadata identity/status mismatch")
    expected_scope = {name: list(OFFICIAL_SPLIT_BOUNDS[name]) for name in ("train", "validation")}
    if metadata.get("source_scope") != expected_scope:
        raise ValueError("Horizon-v2 source split provenance mismatch")
    if metadata.get("source_hashes_at_build") != _current_build_source_hashes(Path(repository_root)):
        raise ValueError("Horizon-v2 preprocessing source identity drift")
    build_config = metadata.get("build_configuration")
    if build_config != BUILD_CONFIGURATION or hashlib.sha256(
        json.dumps(build_config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest() != metadata.get("build_configuration_sha256"):
        raise ValueError("Horizon-v2 build configuration identity mismatch")
    if metadata.get("window") != {"history_length": HISTORY_LENGTH, "horizons": list(HORIZONS), "step_seconds": 900}:
        raise ValueError("Horizon-v2 window configuration mismatch")
    if metadata.get("splits", {}).get(split, {}).get("bounds") != list(OFFICIAL_SPLIT_BOUNDS[split]):
        raise ValueError(f"Horizon-v2 {split} bounds mismatch")
    split_record = metadata["splits"][split]
    expected_filenames = [
        f"demandMatrix-geant-uhlig-15min-{(EXPECTED_START + index * timedelta(seconds=900)):%Y%m%d-%H%M}.txt"
        for index in range(*OFFICIAL_SPLIT_BOUNDS[split])
    ]
    if split_record.get("source_file_names") != expected_filenames:
        raise ValueError(f"Horizon-v2 {split} source filename provenance mismatch")
    semantic_digest = split_record.get("parsed_source_semantic_sha256")
    if not isinstance(semantic_digest, str) or len(semantic_digest) != 64:
        raise ValueError(f"Horizon-v2 {split} source semantic fingerprint is malformed")
    edges = tuple(tuple(edge) for edge in metadata.get("topology", {}).get("directed_edges_interleaved", ()))
    topology_metadata = metadata.get("topology", {})
    if (
        len(edges) != 72
        or len(set(edges)) != 72
        or len(topology_metadata.get("nodes", [])) != 22
        or topology_metadata.get("undirected_links") != 36
        or topology_metadata.get("directed_links") != 72
        or _edge_digest(edges) != topology_metadata.get("ordered_edge_sha256")
    ):
        raise ValueError("Horizon-v2 directed-edge identity mismatch")
    actual_nodes, actual_links, actual_edges, actual_topology_hash = _topology_snapshot(
        Path(repository_root) / TOPOLOGY_RELATIVE, repository_root
    )
    del actual_nodes, actual_links
    if actual_edges != edges or actual_topology_hash != topology_metadata.get("sha256_local_file_identity_only"):
        raise ValueError("Stored topology does not match the actual ordered GEANT topology")
    expected_names = {
        f"{kind}_{current_split}_h{horizon}.npy"
        for current_split in ("train", "validation")
        for horizon in HORIZONS
        for kind in ("history", "target")
    } | {"metadata.json", "window_index_train.json", "window_index_validation.json"}
    recorded = metadata.get("artifact_names")
    if not isinstance(recorded, list) or len(recorded) != len(expected_names) or set(recorded) != expected_names:
        raise ValueError("Horizon-v2 artifact allowlist mismatch")
    histories: dict[int, np.ndarray] = {}
    targets: dict[int, np.ndarray] = {}
    for horizon in HORIZONS:
        history_name = validate_artifact_name(f"history_{split}_h{horizon}.npy")
        target_name = validate_artifact_name(f"target_{split}_h{horizon}.npy")
        histories[horizon] = load_horizon_array(root, history_name, repository_root=repository_root)
        targets[horizon] = load_horizon_array(root, target_name, repository_root=repository_root)
        entry = metadata["splits"][split]["horizons"][str(horizon)]
        if histories[horizon].shape != tuple(entry["history_shape"]) or targets[horizon].shape != tuple(entry["target_shape"]):
            raise ValueError(f"Horizon-v2 {split} array shape mismatch")
        if hashlib.sha256(histories[horizon].tobytes()).hexdigest() != entry["history_sha256"]:
            raise ValueError(f"Horizon-v2 {split} history fingerprint mismatch")
        if hashlib.sha256(targets[horizon].tobytes()).hexdigest() != entry["target_sha256"]:
            raise ValueError(f"Horizon-v2 {split} target fingerprint mismatch")
        if histories[horizon].shape[1:] != (HISTORY_LENGTH, len(edges)) or targets[horizon].shape != (len(histories[horizon]), len(edges)):
            raise ValueError(f"Horizon-v2 {split} tensor dimensions mismatch")
    index_name = validate_artifact_name(f"window_index_{split}.json")
    index_path = require_regular_file(root / index_name, root)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    lower, upper = OFFICIAL_SPLIT_BOUNDS[split]
    for horizon in HORIZONS:
        rows = index.get(str(horizon))
        if not isinstance(rows, list) or len(rows) != len(histories[horizon]):
            raise ValueError(f"Horizon-v2 {split} window index is misaligned")
        prior = lower - 1
        for row in rows:
            origin = int(row["origin"])
            if not (
                int(row["history_start"]) == origin - HISTORY_LENGTH + 1
                and int(row["target"]) == origin + horizon
                and lower <= int(row["history_start"]) <= origin < int(row["target"]) < upper
                and origin > prior
            ):
                raise ValueError(f"Horizon-v2 {split} window violates target/split alignment")
            prior = origin
    return histories, targets, edges, index


def run_train_only_ridge_screen(
    data_root: str | Path,
    *,
    repository_root: str | Path,
) -> dict[str, object]:
    """Execute the predeclared Ridge screen from train artifacts only."""
    histories, targets, edges, index = load_train_arrays_for_screen(data_root, repository_root=repository_root)
    results: dict[str, object] = {}
    for horizon in HORIZONS:
        origins = [int(item["origin"]) for item in index[str(horizon)]]
        result = chronological_ridge_screen(histories[horizon], targets[horizon], origins, edges, horizon)
        results[str(horizon)] = asdict(result)
    return {
        "protocol_id": PROTOCOL_ID,
        "scope": "training block only; no validation/test arrays or observations",
        "fit_fraction": SCREEN_FIT_FRACTION,
        "ridge_alpha": RIDGE_ALPHA,
        "standardization_fit_scope": "first chronological fit prefix only",
        "screen_thresholds": SCREEN_THRESHOLDS,
        "results": results,
    }


def run_raw_training_ridge_screen(
    *,
    repository_root: str | Path,
    output_root: str | Path,
) -> tuple[dict[str, object], Path, str]:
    """Run the one allowed real-data operation: parse training sources only.

    Exact training filenames are selected before parsing. This function never
    discovers, constructs, or opens validation/test matrix paths or arrays.
    It writes only aggregate Ridge-screen results and provenance.
    """
    repo = canonical_repository_root(repository_root)
    processed = repo / "data" / "processed"
    if not processed.is_dir() or processed.resolve(strict=True) != processed:
        raise FileNotFoundError(f"Canonical processed-data directory is unavailable: {processed}")
    output = checked_absolute(output_root, allow_missing=True)
    prefix = "geant_horizons_v2_train_only_screen_"
    suffix = output.name[len(prefix) :] if output.name.startswith(prefix) else ""
    if output.parent != processed or not suffix or not all(char.isalnum() or char in "-_" for char in suffix):
        raise ValueError(f"Screen output must be a new direct child of {processed} with prefix {prefix}")
    if output.exists():
        raise FileExistsError(f"Refusing existing train-only screen output: {output}")

    raw = repo / RAW_DIRECTORY_RELATIVE
    source_access_audit: dict[str, object] = {}
    training_paths = expected_split_paths(
        raw, "train", repository_root=repo, access_audit=source_access_audit
    )
    nodes, undirected, edges, topology_hash = _topology_snapshot(repo / TOPOLOGY_RELATIVE, repo)
    ordered_nodes = tuple(sorted(nodes))
    node_index = {node: index for index, node in enumerate(ordered_nodes)}
    routes = _routing_paths(nodes, edges)
    training_read = validity_v2.read_training_period(
        raw,
        ordered_nodes,
        node_index,
        expected_start_timestamp=EXPECTED_START,
        expected_start_timestamp_source=EXPECTED_START_SOURCE,
        discovery=lambda _directory: training_paths,
    )
    if len(training_read.observations) != OFFICIAL_SPLIT_BOUNDS["train"][1]:
        raise ValueError("Training-only reader returned an unexpected number of observations")
    traffic, utilization, reasons = validity_v2.build_link_series_v2(
        training_read.observations,
        edges,
        routes,
        training_read.cutoff_mbps,
        nodes=ordered_nodes,
    )
    del traffic
    valid = np.asarray([not reason for reason in reasons], dtype=np.bool_)
    timestamps = tuple(item.timestamp for item in training_read.observations)
    timestamp_seconds = tuple(int((stamp - timestamps[0]).total_seconds()) for stamp in timestamps)
    screen_results, window_counts = screen_training_utilization(utilization, valid, timestamp_seconds, edges)
    source_fingerprint = hashlib.sha256()
    for item in training_read.observations:
        source_fingerprint.update(item.filename.encode("utf-8") + b"\0")
        source_fingerprint.update(item.timestamp.isoformat().encode("ascii"))
        source_fingerprint.update(item.demand_count.to_bytes(8, "big", signed=False))
        source_fingerprint.update(np.asarray(item.matrix_mbps, dtype="<f8").tobytes(order="C"))
    result: dict[str, object] = {
        "protocol_id": PROTOCOL_ID,
        "scope": "raw training interval [0,8020) only; no validation/test source or arrays",
        "source_directory": str(raw.relative_to(repo)),
        "source_file_count": len(training_paths),
        "source_access_audit": source_access_audit,
        "source_filenames": [path.name for path in training_paths],
        "parsed_training_semantic_sha256": source_fingerprint.hexdigest(),
        "expected_start_timestamp": EXPECTED_START.strftime("%Y%m%d-%H%M"),
        "anchor_provenance": "project-cited official SNDlib documentation; local archive checksum and upstream dataset identity unverified",
        "training_cutoff_mbps": training_read.cutoff_mbps,
        "training_cutoff_nonempty_count": training_read.metadata["cutoff_fit_nonempty_count"],
        "training_cutoff_filenames": list(training_read.cutoff_fit_filenames),
        "validity_counts": {
            "total": len(reasons),
            "valid": int(valid.sum()),
            "invalid": int((~valid).sum()),
            "empty": sum("empty" in reason for reason in reasons),
            "total_demand": sum("total_demand" in reason for reason in reasons),
            "individual_demand": sum("individual_demand" in reason for reason in reasons),
        },
        "topology": {
            "path": str((repo / TOPOLOGY_RELATIVE).relative_to(repo)),
            "sha256_local_file_identity_only": topology_hash,
            "upstream_provenance": "unverified",
            "node_count": len(nodes),
            "undirected_link_count": len(undirected),
            "directed_edge_count": len(edges),
            "directed_edge_order": "each undirected link (u,v) expands immediately to (u,v),(v,u)",
            "ordered_edge_sha256": _edge_digest(edges),
            "capacity_mbps_assumed": CAPACITY_MBPS,
            "utilization": "computed ratio; not measured telemetry",
        },
        "window": {"history_length": HISTORY_LENGTH, "horizons": list(HORIZONS), "step_seconds": 900},
        "screen_protocol": {
            "ridge_alpha": RIDGE_ALPHA,
            "fit_fraction": SCREEN_FIT_FRACTION,
            "embargo_steps": SCREEN_EMBARGO_STEPS,
            "features": ["own daily residual", "own weekly residual", "mean neighboring current residual", "mean neighboring four-step change"],
            "feature_and_target_scaling": "fit prefix only",
            "gate": SCREEN_THRESHOLDS,
        },
        "window_counts": {str(horizon): values for horizon, values in window_counts.items()},
        "results": {str(horizon): asdict(value) for horizon, value in screen_results.items()},
        "source_hashes_at_run": _source_identity(),
        "source_hash_limit": "file hashes do not prove which Python source was imported/executed",
        "command": "python -m phase9_geant.horizon_v2_experiment --run-raw-ridge-screen --repository-root <repo> --output-root <new-screen-root>",
    }
    output.mkdir(parents=False, exist_ok=False)
    if checked_absolute(output) != output or output.resolve(strict=True) != output or output.parent.resolve(strict=True) != processed:
        raise ValueError("Created train-only screen directory failed containment verification")
    result_path = output / "results.json"
    _atomic_json(result_path, result)
    output_hash = hashlib.sha256(result_path.read_bytes()).hexdigest()
    return result, result_path, output_hash


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def fit_training_scaler(histories: Mapping[int, np.ndarray], targets: Mapping[int, np.ndarray]) -> Standardizer:
    """Fit a train-only affine scaler using bounded temporary memory.

    Chunked parallel-variance aggregation preserves population standardization
    without making full-size float64 copies of all overlapping histories.
    """
    count = 0
    mean = 0.0
    m2 = 0.0
    chunk_size = 1_000_000
    for source in (*histories.values(), *targets.values()):
        array = np.asarray(source)
        if array.size == 0:
            raise ValueError("cannot fit scaler on empty training arrays")
        flat = array.flat
        for start in range(0, array.size, chunk_size):
            values = np.asarray(flat[start : start + chunk_size], dtype=np.float64)
            if not np.isfinite(values).all():
                raise ValueError("cannot fit scaler on nonfinite training values")
            chunk_count = int(values.size)
            chunk_mean = float(values.mean(dtype=np.float64))
            centered = values - chunk_mean
            chunk_m2 = float(np.dot(centered, centered))
            if count == 0:
                mean, m2, count = chunk_mean, chunk_m2, chunk_count
                continue
            total = count + chunk_count
            delta = chunk_mean - mean
            m2 += chunk_m2 + delta * delta * count * chunk_count / total
            mean += delta * chunk_count / total
            count = total
    if count == 0:
        raise ValueError("cannot fit scaler on empty training values")
    scale = float(np.sqrt(max(0.0, m2 / count)))
    if scale <= np.finfo(np.float64).eps:
        scale = 1.0
    return Standardizer(mean, scale, "training arrays only")


def _atomic_torch_save(value: object, destination: Path) -> None:
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    os.close(fd)
    try:
        torch.save(value, temporary)
        # Windows requires a writable handle for os.fsync; reopening read-only
        # can raise EBADF even though torch.save completed successfully.
        with open(temporary, "rb+") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _atomic_json(path: Path, value: object) -> None:
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n"
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _source_identity() -> dict[str, str]:
    repository = Path(__file__).resolve().parents[1]
    result: dict[str, str] = {}
    for relative_name in SOURCE_IDENTITY_FILES:
        path = checked_absolute(repository / Path(relative_name))
        if not path.is_file() or path.resolve(strict=True) != path:
            raise ValueError(f"Required first-party source identity file is unavailable: {relative_name}")
        result[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def _validate_planned_run_directory(root: Path, mode: str, seed: int) -> Path:
    if mode not in ("topology", "identity", "shuffled") or seed not in SEEDS:
        raise ValueError("run mode or seed is not part of the frozen protocol")
    run_parent = root / "runs"
    if run_parent.exists():
        checked_absolute(run_parent)
        if not run_parent.is_dir() or run_parent.resolve(strict=True) != run_parent:
            raise ValueError(f"Run parent is redirected or not a directory: {run_parent}")
    output = checked_absolute(run_parent / f"horizon_v2_{mode}_seed_{seed}", allow_missing=True)
    if not output.is_relative_to(root):
        raise ValueError("Run directory escapes the validated data root")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing existing horizon-v2 run directory: {output}")
    return output


def _train_arrays(
    *,
    histories: Mapping[int, np.ndarray],
    targets: Mapping[int, np.ndarray],
    validation_histories: Mapping[int, np.ndarray],
    validation_targets: Mapping[int, np.ndarray],
    edge_order: Sequence[tuple[str, str]],
    output_directory: str | Path,
    seed: int,
    adjacency_mode: str,
    run_provenance: Mapping[str, object],
    device: str = "cpu",
) -> dict[str, object]:
    """Train one topology control on supplied train/validation arrays only.

    This callable is intentionally not invoked here. The caller must first
    load via the versioned allowlist and independently approve a run.
    """
    if 4 in HORIZONS:
        raise PermissionError(
            "Candidate training is blocked: h=4 is STOP and this fixed trainer includes it in the joint objective."
        )
    if seed not in SEEDS or adjacency_mode not in ("topology", "identity", "shuffled"):
        raise ValueError("seed or adjacency mode is not part of the frozen protocol")
    if set(histories) != set(HORIZONS) or set(targets) != set(HORIZONS):
        raise ValueError("training arrays must supply both fixed horizons")
    if set(validation_histories) != set(HORIZONS) or set(validation_targets) != set(HORIZONS):
        raise ValueError("validation arrays must supply both fixed horizons")
    for horizon in HORIZONS:
        if histories[horizon].ndim != 3 or histories[horizon].shape[1:] != (HISTORY_LENGTH, len(edge_order)):
            raise ValueError("history dimensions do not match the fixed protocol")
        if targets[horizon].shape != (len(histories[horizon]), len(edge_order)):
            raise ValueError("target dimensions do not align with histories")
        if validation_histories[horizon].shape[1:] != (HISTORY_LENGTH, len(edge_order)):
            raise ValueError("validation history dimensions do not match protocol")
        if validation_targets[horizon].shape != (len(validation_histories[horizon]), len(edge_order)):
            raise ValueError("validation targets do not align with histories")
        if not all(np.isfinite(array).all() for array in (histories[horizon], targets[horizon], validation_histories[horizon], validation_targets[horizon])):
            raise ValueError("training/validation arrays must contain valid finite windows")

    output = checked_absolute(output_directory, allow_missing=True)
    data_root = Path(str(run_provenance["data_root"]))
    run_parent = data_root / "runs"
    if output.parent != run_parent:
        raise ValueError("Run output must be a direct child of the horizon-v2 runs directory")
    if output.exists():
        raise FileExistsError(f"Refusing existing horizon-v2 run directory: {output}")
    scaler = fit_training_scaler(histories, targets)
    _seed_all(seed)
    if not run_parent.exists():
        run_parent.mkdir(parents=False, exist_ok=False)
    output.mkdir(parents=False, exist_ok=False)
    if (
        checked_absolute(output) != output
        or output.resolve(strict=True) != output
        or output.parent.resolve(strict=True) != run_parent.resolve(strict=True)
        or not output.parent.resolve(strict=True).is_relative_to(data_root)
    ):
        raise ValueError("Created run directory failed containment validation")
    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    history_log: list[dict[str, object]] = []
    checkpoint = output / "best.pt"
    manifest = {
        "protocol_id": PROTOCOL_ID, "status": "incomplete", "seed": seed,
        "adjacency_mode": adjacency_mode, "configuration": TRAINING_CONFIG,
        "normalization": asdict(scaler), "selection": "validation SmoothL1 only",
        "validation_split": list(OFFICIAL_SPLIT_BOUNDS["validation"]),
        "run_provenance": dict(run_provenance), "checkpoint": "best.pt",
    }
    _atomic_json(output / "run.json", manifest)
    started = time.perf_counter()
    try:
        model = SpatioTemporalLinkForecaster(
            edge_order, history_length=HISTORY_LENGTH, horizons=HORIZONS,
            hidden_size=int(TRAINING_CONFIG["hidden_size"]), alpha=0.8, ewma_steps=5,
            include_seasonal_features=True, adjacency_mode=adjacency_mode,  # type: ignore[arg-type]
        ).to(device)
        adjacency = model._adjacency(adjacency_mode).detach().cpu().contiguous().numpy()
        manifest["adjacency_identity"] = {
            "mode": adjacency_mode,
            "normalized_matrix_sha256": hashlib.sha256(adjacency.tobytes(order="C")).hexdigest(),
            "shuffle_seed": getattr(model, "shuffle_seed", None),
            "link_count": len(edge_order),
            "ordered_edge_sha256": run_provenance.get("ordered_edge_sha256"),
            "operator": "shared self_projection + message_projection; only fixed adjacency buffer varies",
        }
        # Persist realized control identity while the run is still explicitly
        # incomplete, before optimizer updates or checkpoint creation.
        _atomic_json(output / "run.json", manifest)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(TRAINING_CONFIG["learning_rate"]))
        for epoch in range(1, int(TRAINING_CONFIG["epochs"]) + 1):
            model.train()
            train_sum_by_horizon: list[float] = []
            train_count_by_horizon: list[int] = []
            rng = np.random.default_rng(seed + epoch)
            for horizon_index, horizon in enumerate(HORIZONS):
                horizon_loss_sum = 0.0
                horizon_count = 0
                order = rng.permutation(len(histories[horizon]))
                for start in range(0, len(order), int(TRAINING_CONFIG["batch_size"])):
                    batch_indices = order[start : start + int(TRAINING_CONFIG["batch_size"])]
                    hx = torch.as_tensor((histories[horizon][batch_indices] - scaler.mean) / scaler.scale, dtype=torch.float32, device=device)
                    ty = torch.as_tensor((targets[horizon][batch_indices] - scaler.mean) / scaler.scale, dtype=torch.float32, device=device)
                    prediction = model(hx)[:, horizon_index, :]
                    pred_original = prediction * scaler.scale + scaler.mean
                    target_original = ty * scaler.scale + scaler.mean
                    baseline_original = model._ewma(hx) * scaler.scale + scaler.mean
                    loss = F.smooth_l1_loss(
                        pred_original - baseline_original, target_original - baseline_original,
                        beta=0.001,
                    )
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    count = int(ty.numel())
                    horizon_loss_sum += float(loss.detach().cpu()) * count
                    horizon_count += count
                train_sum_by_horizon.append(horizon_loss_sum)
                train_count_by_horizon.append(horizon_count)
            validation_loss_by_horizon: list[float] = []
            model.eval()
            with torch.no_grad():
                for horizon_index, horizon in enumerate(HORIZONS):
                    loss_sum = 0.0
                    loss_count = 0
                    for start in range(0, len(validation_histories[horizon]), int(TRAINING_CONFIG["batch_size"])):
                        end = start + int(TRAINING_CONFIG["batch_size"])
                        hx_batch = torch.as_tensor((validation_histories[horizon][start:end] - scaler.mean) / scaler.scale, dtype=torch.float32, device=device)
                        ty_batch = torch.as_tensor((validation_targets[horizon][start:end] - scaler.mean) / scaler.scale, dtype=torch.float32, device=device)
                        pred = model(hx_batch)[:, horizon_index, :]
                        pred_original = pred * scaler.scale + scaler.mean
                        target_original = ty_batch * scaler.scale + scaler.mean
                        base = model._ewma(hx_batch) * scaler.scale + scaler.mean
                        loss = F.smooth_l1_loss(pred_original - base, target_original - base, beta=0.001)
                        count = int(ty_batch.numel())
                        loss_sum += float(loss.cpu()) * count
                        loss_count += count
                    validation_loss_by_horizon.append(loss_sum / loss_count)
            val_loss = float(np.mean(validation_loss_by_horizon))
            train_by_horizon = [total / count for total, count in zip(train_sum_by_horizon, train_count_by_horizon)]
            epoch_record = {
                "epoch": epoch,
                "train_smooth_l1_by_horizon": dict(zip(map(str, HORIZONS), train_by_horizon)),
                "validation_smooth_l1_by_horizon": dict(zip(map(str, HORIZONS), validation_loss_by_horizon)),
                "validation_smooth_l1_equal_horizon_mean": val_loss,
            }
            history_log.append(epoch_record)
            if validation_loss_improved(val_loss, best_loss):
                best_loss, best_epoch, stale = val_loss, epoch, 0
                _atomic_torch_save({"state_dict": model.state_dict(), "seed": seed, "best_epoch": epoch}, checkpoint)
            else:
                stale += 1
            if stale >= int(TRAINING_CONFIG["patience"]):
                break
        if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
            raise RuntimeError("No nonempty best checkpoint was finalized")
        checkpoint_bytes = checkpoint.read_bytes()
        checkpoint_sha = hashlib.sha256(checkpoint_bytes).hexdigest()
        checkpoint_data = torch.load(io.BytesIO(checkpoint_bytes), map_location=device, weights_only=True)
        model.load_state_dict(checkpoint_data["state_dict"], strict=True)
        validation_metrics: dict[str, object] = {}
        model.eval()
        with torch.no_grad():
            for horizon_index, horizon in enumerate(HORIZONS):
                prediction_chunks = []
                for start in range(0, len(validation_histories[horizon]), int(TRAINING_CONFIG["batch_size"])):
                    end = start + int(TRAINING_CONFIG["batch_size"])
                    hx = torch.as_tensor((validation_histories[horizon][start:end] - scaler.mean) / scaler.scale, dtype=torch.float32, device=device)
                    pred_norm = model(hx)[:, horizon_index, :]
                    prediction_chunks.append(pred_norm.detach().cpu().numpy().astype(np.float64))
                prediction = np.concatenate(prediction_chunks, axis=0) * scaler.scale + scaler.mean
                target = validation_targets[horizon].astype(np.float64)
                baseline_predictions = baseline_forecasts(validation_histories[horizon], horizon)
                validation_metrics[str(horizon)] = {
                    "b_u": metric_summary(target, prediction),
                    **{
                        name: metric_summary(target, baseline_prediction)
                        for name, baseline_prediction in baseline_predictions.items()
                    },
                }
        manifest.update({
            "status": "complete", "best_epoch": best_epoch, "best_validation_loss": best_loss,
            "epochs_completed": len(history_log), "history": history_log,
            "checkpoint_sha256": checkpoint_sha, "checkpoint_reload_strict": True,
            "validation_metrics": validation_metrics,
            "early_stopping": {"patience": TRAINING_CONFIG["patience"], "stopped_early": len(history_log) < int(TRAINING_CONFIG["epochs"])},
            "duration_seconds": time.perf_counter() - started,
        })
        _atomic_json(output / "run.json", manifest)
    except BaseException as exc:
        manifest.update({"status": "failed", "failure": f"{type(exc).__name__}: {exc}"})
        try:
            _atomic_json(output / "run.json", manifest)
        except BaseException:
            pass
        raise
    return manifest


def train_from_horizon_root(
    data_root: str | Path,
    *,
    repository_root: str | Path,
    seed: int,
    adjacency_mode: str,
    device: str = "cpu",
    enabled: bool = False,
) -> dict[str, object]:
    """Validated explicit training entry; disabled unless ``enabled=True``."""
    if not enabled:
        raise PermissionError("Horizon-v2 training gate is disabled; explicit authorization is required")
    if 4 in HORIZONS:
        raise PermissionError(
            "Candidate training is blocked: the fixed Ridge screen classified h=4 as STOP, but the current "
            "trainer jointly optimizes h=4 and h=96. A separately reviewed horizon-specific protocol and "
            "implementation are required before training."
        )
    repo = canonical_repository_root(repository_root)
    root = _canonical_horizon_root(data_root, repo)
    planned = _validate_planned_run_directory(root, adjacency_mode, seed)
    from phase9_geant.safe_paths import require_regular_file

    dataset_metadata_path = require_regular_file(root / "metadata.json", root)
    dataset_metadata = json.loads(dataset_metadata_path.read_text(encoding="utf-8"))
    if dataset_metadata.get("protocol_id") != PROTOCOL_ID or dataset_metadata.get("status") != "complete":
        raise ValueError("Horizon-v2 dataset metadata is not complete or has the wrong protocol")
    expected_scope = {split: list(OFFICIAL_SPLIT_BOUNDS[split]) for split in ("train", "validation")}
    if dataset_metadata.get("source_scope") != expected_scope:
        raise ValueError("Dataset source scope is not the frozen train/validation pair")
    from phase9_geant.horizon_v2_data import _edge_digest, _topology_snapshot

    topology_path = repo / TOPOLOGY_RELATIVE
    nodes, _links, actual_edges, topology_hash = _topology_snapshot(topology_path, repo)
    del nodes, _links
    metadata_edges = tuple(tuple(edge) for edge in dataset_metadata.get("topology", {}).get("directed_edges_interleaved", ()))
    if actual_edges != metadata_edges or topology_hash != dataset_metadata["topology"].get("sha256_local_file_identity_only"):
        raise ValueError("Current topology identity/order differs from the built dataset")
    train_x, train_y, edges, train_index = _load_split_arrays(root, "train", repository_root=repo)
    validation_x, validation_y, validation_edges, validation_index = _load_split_arrays(root, "validation", repository_root=repo)
    if edges != validation_edges:
        raise ValueError("Train/validation directed-edge order differs")
    if actual_edges != edges:
        raise ValueError("Train arrays do not match the verified topology edge ordering")
    expected_bounds = {split: list(OFFICIAL_SPLIT_BOUNDS[split]) for split in ("train", "validation")}
    if dataset_metadata.get("source_scope") != expected_bounds:
        raise ValueError("Dataset split provenance mismatch")
    config = {**TRAINING_CONFIG, "seed": seed, "adjacency_mode": adjacency_mode}
    source_hashes = _source_identity()
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    run_provenance = {
        "data_root": str(root),
        "dataset_protocol_id": dataset_metadata.get("protocol_id"),
        "data_fingerprints": {
            split: {
                str(horizon): {
                    "history_sha256": dataset_metadata["splits"][split]["horizons"][str(horizon)]["history_sha256"],
                    "target_sha256": dataset_metadata["splits"][split]["horizons"][str(horizon)]["target_sha256"],
                }
                for horizon in HORIZONS
            }
            for split in ("train", "validation")
        },
        "ordered_edge_sha256": _edge_digest(edges),
        "topology_sha256": topology_hash,
        "source_hashes_at_run_start": source_hashes,
        "configuration_sha256": config_hash,
        "training_window_counts": {str(h): len(train_index[str(h)]) for h in HORIZONS},
        "validation_window_counts": {str(h): len(validation_index[str(h)]) for h in HORIZONS},
        "test_data_access": "no test array name, path, loader, or evaluation branch is implemented",
        "source_hash_limit": (
            "first-party source hashes describe listed file bytes at run start; they do not prove imported/executed "
            "Python identity or pin third-party/native runtime libraries"
        ),
    }
    # Preflight is complete before run-directory creation or model construction.
    return _train_arrays(
        histories=train_x, targets=train_y,
        validation_histories=validation_x, validation_targets=validation_y,
        edge_order=edges, output_directory=planned, seed=seed,
        adjacency_mode=adjacency_mode, run_provenance=run_provenance, device=device,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root")
    parser.add_argument("--output-root")
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--adjacency", choices=("topology", "identity", "shuffled"))
    parser.add_argument("--run-training", action="store_true", help="explicitly enable one train/validation run")
    parser.add_argument("--run-ridge-screen", action="store_true", help="explicitly enable the permitted train-only screen")
    parser.add_argument("--run-raw-ridge-screen", action="store_true", help="explicitly enable the train-source-only Ridge screen")
    parser.add_argument("--device", default="cpu")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    actions = [args.run_training, args.run_ridge_screen, args.run_raw_ridge_screen]
    if sum(actions) != 1:
        parser.error("choose exactly one of --run-training, --run-ridge-screen or --run-raw-ridge-screen")
    if args.run_training:
        if args.seed is None or args.adjacency is None or args.data_root is None:
            parser.error("--run-training requires --data-root, --seed and --adjacency")
        result = train_from_horizon_root(
            args.data_root, repository_root=args.repository_root, seed=args.seed,
            adjacency_mode=args.adjacency, device=args.device, enabled=True,
        )
    elif args.run_ridge_screen:
        if args.data_root is None or args.output_root is not None:
            parser.error("--run-ridge-screen requires --data-root and does not accept --output-root")
        if args.seed is not None or args.adjacency is not None:
            parser.error("Ridge screen does not accept seed or adjacency options")
        result = run_train_only_ridge_screen(args.data_root, repository_root=args.repository_root)
    else:
        if args.data_root is not None or args.output_root is None:
            parser.error("--run-raw-ridge-screen requires --output-root and does not accept --data-root")
        if args.seed is not None or args.adjacency is not None:
            parser.error("Raw Ridge screen does not accept seed or adjacency options")
        result, result_path, result_hash = run_raw_training_ridge_screen(
            repository_root=args.repository_root, output_root=args.output_root
        )
        print(json.dumps({"result_path": str(result_path), "result_sha256": result_hash}, indent=2))
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "RIDGE_ALPHA", "RidgeScreenResult", "SEEDS", "Standardizer", "build_ridge_features",
    "chronological_ridge_screen", "fit_train_only_standardizer", "fit_training_scaler",
    "load_train_arrays_for_screen", "run_train_only_ridge_screen", "train_from_horizon_root",
]
