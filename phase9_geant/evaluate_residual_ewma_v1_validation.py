"""Read-only validation comparison for completed residual, EWMA, and B-U runs.

The evaluator is opt-in through a function call and loads only the fixed
validation arrays, approved validation provenance, and explicitly named run
manifests/checkpoints. It contains no training or test-data loader.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from phase9_geant import train_bu_v2 as bu
from phase9_geant import train_residual_ewma_v1 as residual
from phase9_geant.parse_geant import parse_topology
from phase9_geant.safe_paths import (
    require_regular_file,
    validate_existing_child,
    validate_fresh_child,
    validate_v2_root,
)
from phase9_geant.split_access import load_permitted_split_array
from phase9_geant.validity_v2 import PROTOCOL_ID, SPLIT_BOUNDS
from phase9_geant.window_provenance_v2 import (
    PROVENANCE_FILENAME,
    array_fingerprint,
    canonical_json_sha256,
    validate_train_validation_provenance,
)
from src.gnn.model import EdgeQoSForecaster


SEEDS = (42, 123, 2026)
OUTPUT_NAME = "neural_ewma_residual_v1_validation_eval"
BOOTSTRAP_SEED = 20261017
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_CONFIDENCE = 0.95
EXPECTED_RESIDUAL_PREFIX = "neural_ewma_residual_v1_seed_"
EXPECTED_BU_PREFIX = "neural_bu_v2_full_seed_"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    target64 = np.asarray(target, dtype=np.float64)
    prediction64 = np.asarray(prediction, dtype=np.float64)
    if target64.shape != prediction64.shape:
        raise ValueError("Prediction shape differs from validation target shape")
    if target64.ndim != 2 or not np.isfinite(target64).all() or not np.isfinite(prediction64).all():
        raise ValueError("Metrics require aligned, entirely finite validation values")
    error = prediction64 - target64
    return {
        "mae": float(np.mean(np.abs(error), dtype=np.float64)),
        "rmse": float(np.sqrt(np.mean(np.square(error), dtype=np.float64))),
        "evaluated_scalar_targets": int(target64.size),
        "windows": int(target64.shape[0]),
        "directed_links": int(target64.shape[1]),
    }


def paired_absolute_error_difference(
    target: np.ndarray, candidate: np.ndarray, baseline: np.ndarray
) -> np.ndarray:
    target64 = np.asarray(target, dtype=np.float64)
    candidate64 = np.asarray(candidate, dtype=np.float64)
    baseline64 = np.asarray(baseline, dtype=np.float64)
    if target64.shape != candidate64.shape or target64.shape != baseline64.shape:
        raise ValueError("Paired methods must have exactly the same window/link shape")
    if not all(np.isfinite(value).all() for value in (target64, candidate64, baseline64)):
        raise ValueError("Paired comparison contains nonfinite values; no intersection mask is allowed")
    # Negative values mean the candidate has lower absolute error.
    return np.abs(candidate64 - target64) - np.abs(baseline64 - target64)


def validate_edge_order(
    recorded: object,
    runtime_edges: Sequence[tuple[str, str]],
    *,
    format_name: str,
) -> None:
    if format_name == "residual":
        expected: object = [list(edge) for edge in runtime_edges]
    elif format_name == "B-U":
        expected = [
            {"index": index, "source": source, "destination": destination}
            for index, (source, destination) in enumerate(runtime_edges)
        ]
    else:
        raise ValueError("Unknown edge-order manifest format")
    if recorded != expected:
        raise ValueError(f"{format_name} directed-edge order differs from the runtime topology")


def paired_window_bootstrap_ci(
    differences: np.ndarray,
    *,
    seed: int = BOOTSTRAP_SEED,
    replicates: int = BOOTSTRAP_REPLICATES,
    confidence: float = BOOTSTRAP_CONFIDENCE,
) -> dict[str, float | int]:
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("Bootstrap input must be a nonempty [windows, links] matrix")
    if not np.isfinite(values).all() or replicates < 1 or not 0.0 < confidence < 1.0:
        raise ValueError("Invalid bootstrap values or settings")
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sampled_windows = rng.integers(0, values.shape[0], size=values.shape[0])
        estimates[index] = values[sampled_windows].mean(dtype=np.float64)
    tail = (1.0 - confidence) / 2.0
    return {
        "mean_absolute_error_difference": float(values.mean(dtype=np.float64)),
        "per_link_mean_absolute_error_difference": values.mean(axis=0, dtype=np.float64).tolist(),
        "window_count": int(values.shape[0]),
        "link_count": int(values.shape[1]),
        "lower": float(np.quantile(estimates, tail)),
        "upper": float(np.quantile(estimates, 1.0 - tail)),
        "confidence": confidence,
        "replicates": replicates,
        "rng_seed": seed,
        "resampling_unit": "validation windows; all directed-link errors for each sampled window retained",
    }


def aggregate_seed_metrics(per_seed: Mapping[str, Mapping[str, float]]) -> dict[str, dict[str, float]]:
    if tuple(sorted(int(seed) for seed in per_seed)) != tuple(sorted(SEEDS)):
        raise ValueError("Aggregation requires exactly seeds 42, 123, and 2026")
    output: dict[str, dict[str, float]] = {}
    for metric_name in ("mae", "rmse"):
        values = np.asarray([float(per_seed[str(seed)][metric_name]) for seed in SEEDS], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("Seed-level metrics must be finite")
        output[metric_name] = {
            "mean": float(values.mean(dtype=np.float64)),
            "sample_std_ddof_1": float(values.std(ddof=1, dtype=np.float64)),
        }
    return output


def usefulness_gate(residual_maes: Sequence[float], ewma_maes: Sequence[float]) -> dict[str, Any]:
    residual_values = np.asarray(residual_maes, dtype=np.float64)
    ewma_values = np.asarray(ewma_maes, dtype=np.float64)
    if residual_values.shape != (3,) or ewma_values.shape != (3,):
        raise ValueError("The gate requires exactly three residual and EWMA seed metrics")
    if not np.isfinite(residual_values).all() or not np.isfinite(ewma_values).all() or np.any(ewma_values <= 0):
        raise ValueError("Gate metrics must be finite and EWMA MAE must be positive")
    wins = residual_values < ewma_values  # strict; ties are not wins
    relative = (ewma_values - residual_values) / ewma_values
    mean_reduction = float(relative.mean(dtype=np.float64))
    condition_wins = int(wins.sum()) >= 2
    condition_reduction = mean_reduction >= 0.01
    return {
        "status": "PASS" if condition_wins and condition_reduction else "FAIL",
        "wins": int(wins.sum()),
        "required_wins": 2,
        "strict_per_seed_wins": [bool(value) for value in wins],
        "mean_seed_relative_mae_reduction": mean_reduction,
        "required_mean_reduction": 0.01,
        "condition_majority": condition_wins,
        "condition_reduction": condition_reduction,
    }


def build_seed_comparison(
    target: np.ndarray,
    residual_prediction: np.ndarray,
    ewma_prediction: np.ndarray,
    bu_prediction: np.ndarray,
    *,
    bootstrap_seed: int,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    predictions = {
        "residual_ewma": residual_prediction,
        "ewma": ewma_prediction,
        "B-U": bu_prediction,
    }
    summaries = {name: metrics(target, values) for name, values in predictions.items()}
    return {
        "metrics": summaries,
        "paired_absolute_error_difference_vs_ewma": {
            name: paired_window_bootstrap_ci(
                paired_absolute_error_difference(target, predictions[name], ewma_prediction),
                seed=bootstrap_seed,
                replicates=bootstrap_replicates,
            )
            for name in ("residual_ewma", "B-U")
        },
    }


def _load_json(path: Path, root: Path) -> dict[str, Any]:
    checked = require_regular_file(path, root)
    value = json.loads(checked.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object at {path}")
    return value


def _preflight_completed_artifacts(
    data_root: Path,
    repository_root: Path,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    provenance_hashes: Mapping[str, str],
    provenance_file_hash: str,
    runtime_edge_order: Sequence[tuple[str, str]],
    expected_array_fingerprints: Mapping[str, str],
    expected_array_shapes: Mapping[str, Sequence[int]],
    expected_sample_counts: Mapping[str, int],
    *,
    allow_source_drift: bool = False,
) -> list[str]:
    """Check all six run identities before loading any checkpoint state."""
    warnings: list[str] = []
    current_residual_sources = residual._source_hashes(repository_root)
    for seed in SEEDS:
        for kind, prefix in (("residual", EXPECTED_RESIDUAL_PREFIX), ("B-U", EXPECTED_BU_PREFIX)):
            name = f"{prefix}{seed}"
            _, run_dir = validate_existing_child(data_root, data_root / name, repository_root, expected_name=name)
            manifest = _load_json(require_regular_file(run_dir / "run_metadata.json", run_dir), run_dir)
            if manifest.get("seed") != seed or manifest.get("protocol_id") != PROTOCOL_ID or manifest.get("status") != "complete":
                raise ValueError(f"{kind} seed {seed} manifest is incomplete or has wrong identity")
            if manifest.get("window_provenance_sha256", {}).get("validation") != provenance_hashes["validation"]:
                raise ValueError(f"{kind} seed {seed} validation provenance is absent or mismatched")
            if kind == "residual" and manifest.get("window_provenance_sha256", {}).get("file") != provenance_file_hash:
                raise ValueError(f"Residual seed {seed} sidecar identity mismatch")
            if manifest.get("array_fingerprints", {}).get("X_validation.npy") != array_fingerprint(x_validation):
                raise ValueError(f"{kind} seed {seed} validation input fingerprint mismatch")
            if manifest.get("array_fingerprints", {}).get("y_validation.npy") != array_fingerprint(y_validation):
                raise ValueError(f"{kind} seed {seed} validation target fingerprint mismatch")
            edge_value = manifest.get("edge_order") if kind == "residual" else manifest.get("directed_edge_order")
            validate_edge_order(edge_value, runtime_edge_order, format_name=kind)
            if kind == "residual":
                warnings.extend(_validate_residual_run_metadata(
                    manifest,
                    seed=seed,
                    run_dir=run_dir,
                    data_root=data_root,
                    repository_root=repository_root,
                    provenance_hashes=provenance_hashes,
                    provenance_file_hash=provenance_file_hash,
                    runtime_edge_order=runtime_edge_order,
                    expected_array_fingerprints=expected_array_fingerprints,
                    expected_array_shapes=expected_array_shapes,
                    expected_sample_counts=expected_sample_counts,
                    current_source_hashes=current_residual_sources,
                    allow_source_drift=allow_source_drift,
                ))
            else:
                warnings.extend(_validate_bu_run_metadata(
                    manifest,
                    seed=seed,
                    data_root=data_root,
                    provenance_hashes=provenance_hashes,
                    provenance_file_hash=provenance_file_hash,
                    repository_root=repository_root,
                    runtime_edge_order=runtime_edge_order,
                    expected_array_fingerprints=expected_array_fingerprints,
                    expected_array_shapes=expected_array_shapes,
                    expected_sample_counts=expected_sample_counts,
                    allow_source_drift=allow_source_drift,
                ))
            shapes = manifest.get("array_shapes", {})
            if shapes.get("X_validation") != list(x_validation.shape) or shapes.get("y_validation") != list(y_validation.shape):
                raise ValueError(f"{kind} seed {seed} validation shape mismatch")
            if manifest.get("sample_counts", {}).get("validation") != len(x_validation):
                raise ValueError(f"{kind} seed {seed} validation window count mismatch")
            checkpoint = require_regular_file(run_dir / "best_validation.pt", run_dir)
            if checkpoint.stat().st_size <= 0 or Path(str(manifest.get("checkpoint", ""))).resolve() != checkpoint.resolve():
                raise ValueError(f"{kind} seed {seed} checkpoint path/size mismatch")
            if manifest.get("checkpoint_sha256") != _sha256_file(checkpoint):
                raise ValueError(f"{kind} seed {seed} checkpoint hash is missing or mismatched")
    return sorted(set(warnings))


def _normalized_run_path(path_value: str | Path, repository_root: Path) -> str:
    """Normalize a recorded path without reading its target."""
    if not isinstance(path_value, (str, Path)) or not str(path_value).strip():
        raise ValueError("input-array path entries must be nonempty strings")
    candidate = Path(path_value)
    if not candidate.is_absolute():
        candidate = repository_root / candidate
    return os.path.normcase(os.path.abspath(os.fspath(candidate)))


def validate_input_array_path_allowlist(
    recorded_paths: object,
    *,
    data_root: Path,
    repository_root: Path,
    array_names: Sequence[str],
    field_name: str = "input array paths",
) -> None:
    """Require exactly the four permitted files, independent of path spelling."""
    if not isinstance(recorded_paths, list) or len(recorded_paths) != len(array_names):
        raise ValueError(f"{field_name} must list exactly the permitted train/validation arrays")
    recorded_names = [re.split(r"[\\/]", str(path))[-1] for path in recorded_paths]
    if any(re.search(r"(^|[_-])test([_.-]|$)", name, re.IGNORECASE) for name in recorded_names):
        raise ValueError(f"{field_name} contains a test-split token")

    expected_paths = {
        _normalized_run_path(data_root / filename, repository_root)
        for filename in array_names
    }
    normalized_recorded = [
        _normalized_run_path(path, repository_root) for path in recorded_paths
    ]
    if len(set(normalized_recorded)) != len(array_names) or set(normalized_recorded) != expected_paths:
        raise ValueError(f"{field_name} do not resolve to the exact data-root allowlist")


def diagnose_completed_metadata(
    data_root: Path,
    *,
    repository_root: Path = bu.REPOSITORY_ROOT,
) -> dict[str, Any]:
    """Collect manifest identity/path issues without loading arrays or checkpoints.

    This diagnostic is read-only and cannot publish evaluation results. It checks
    the six fixed run manifests, their basic identity fields, and the exact
    train/validation array path allowlist; artifact-content checks remain the
    responsibility of the full preflight.
    """
    root = validate_v2_root(data_root, repository_root)
    mismatches: list[dict[str, Any]] = []
    for kind, prefix, array_names in (
        ("residual", EXPECTED_RESIDUAL_PREFIX, residual.INPUT_ARRAY_NAMES),
        ("B-U", EXPECTED_BU_PREFIX, bu.ARRAY_NAMES),
    ):
        for seed in SEEDS:
            run_name = f"{prefix}{seed}"
            run_record: dict[str, Any] = {"run": run_name, "mismatches": []}
            try:
                _, run_dir = validate_existing_child(
                    root, root / run_name, repository_root, expected_name=run_name
                )
                manifest = _load_json(
                    require_regular_file(run_dir / "run_metadata.json", run_dir), run_dir
                )
            except (OSError, ValueError, TypeError) as exc:
                run_record["mismatches"].append(f"manifest: {type(exc).__name__}: {exc}")
                mismatches.append(run_record)
                continue

            expected_identity = {
                "status": "complete",
                "seed": seed,
                "protocol_id": PROTOCOL_ID,
                "mode": "full",
            }
            for field, expected in expected_identity.items():
                if manifest.get(field) != expected:
                    run_record["mismatches"].append(
                        f"{field}: expected {expected!r}, got {manifest.get(field)!r}"
                    )
            path_field = "input_artifacts" if kind == "residual" else "array_paths"
            try:
                validate_input_array_path_allowlist(
                    manifest.get(path_field),
                    data_root=root,
                    repository_root=repository_root,
                    array_names=array_names,
                    field_name=path_field,
                )
            except ValueError as exc:
                run_record["mismatches"].append(f"{path_field}: {exc}")
            if run_record["mismatches"]:
                mismatches.append(run_record)
    return {
        "data_root": str(root),
        "runs_checked": len(SEEDS) * 2,
        "arrays_loaded": False,
        "checkpoints_loaded": False,
        "results_published": False,
        "mismatches": mismatches,
    }


def _validate_residual_run_metadata(
    manifest: Mapping[str, Any],
    *,
    seed: int,
    run_dir: Path,
    data_root: Path,
    repository_root: Path,
    provenance_hashes: Mapping[str, str],
    provenance_file_hash: str,
    runtime_edge_order: Sequence[tuple[str, str]],
    expected_array_fingerprints: Mapping[str, str],
    expected_array_shapes: Mapping[str, Sequence[int]],
    expected_sample_counts: Mapping[str, int],
    current_source_hashes: Mapping[str, str],
    allow_source_drift: bool,
) -> list[str]:
    """Validate full train/validation identity before residual checkpoint use."""
    if manifest.get("status") != "complete" or manifest.get("seed") != seed or manifest.get("protocol_id") != PROTOCOL_ID:
        raise ValueError(f"Residual seed {seed} status/seed/protocol identity mismatch")
    if manifest.get("mode") != "full" or manifest.get("experiment") != "Neural Residual-to-EWMA Forecasting, Version 1":
        raise ValueError(f"Residual seed {seed} mode/candidate identity mismatch")
    if manifest.get("model") != {
        "class": "PerLinkResidualMLP", "architecture": "shared 5 -> 16 -> ReLU -> 1"
    }:
        raise ValueError(f"Residual seed {seed} model identity mismatch")
    if manifest.get("ewma") != {
        "alpha": residual.EWMA_ALPHA,
        "weights_oldest_to_newest": residual.ewma_weights().tolist(),
    }:
        raise ValueError(f"Residual seed {seed} EWMA identity mismatch")
    if manifest.get("configuration") != residual.TRAINING:
        raise ValueError(f"Residual seed {seed} configuration mismatch")
    if manifest.get("configuration_sha256") != residual._configuration_hash():
        raise ValueError(f"Residual seed {seed} configuration digest mismatch")
    if manifest.get("split_bounds") != {name: list(bounds) for name, bounds in SPLIT_BOUNDS.items()}:
        raise ValueError(f"Residual seed {seed} split bounds mismatch")
    if manifest.get("history_length") != 5 or manifest.get("prediction_horizon") != 1:
        raise ValueError(f"Residual seed {seed} history/horizon mismatch")
    validate_input_array_path_allowlist(
        manifest.get("input_artifacts"),
        data_root=data_root,
        repository_root=repository_root,
        array_names=residual.INPUT_ARRAY_NAMES,
        field_name=f"Residual seed {seed} input_artifacts",
    )
    if manifest.get("window_provenance_sha256") != {
        "file": provenance_file_hash,
        "train": provenance_hashes["train"],
        "validation": provenance_hashes["validation"],
    }:
        raise ValueError(f"Residual seed {seed} train/validation provenance mismatch")
    if manifest.get("array_fingerprints") != dict(expected_array_fingerprints):
        raise ValueError(f"Residual seed {seed} train/validation array fingerprints mismatch")
    shapes = manifest.get("array_shapes")
    if not isinstance(shapes, dict) or {
        name: shapes.get(name) for name in expected_array_shapes
    } != {name: list(shape) for name, shape in expected_array_shapes.items()}:
        raise ValueError(f"Residual seed {seed} train/validation array shapes mismatch")
    if manifest.get("sample_counts") != dict(expected_sample_counts):
        raise ValueError(f"Residual seed {seed} train/validation sample counts mismatch")
    preprocessing = manifest.get("preprocessing_provenance")
    expected_preprocessing = {
        "protocol_id": PROTOCOL_ID,
        "source": "existing GEANT validity-v2 prebuilt train/validation windows",
        "window_rule": "five consecutive valid history observations plus one valid next-step target within the same official split",
        "link_order": "72 directed columns in v2 interleaved forward/reverse order, checked against the topology parser",
        "utilization_units": "dimensionless computed link-utilization ratio",
        "window_provenance_file": residual.PROVENANCE_FILENAME,
        "window_provenance_sha256": provenance_file_hash,
        "train_window_provenance_sha256": provenance_hashes["train"],
        "validation_window_provenance_sha256": provenance_hashes["validation"],
    }
    if preprocessing != expected_preprocessing:
        raise ValueError(f"Residual seed {seed} preprocessing provenance mismatch")
    validate_edge_order(manifest.get("edge_order"), runtime_edge_order, format_name="residual")

    source_before = manifest.get("source_sha256_pretraining")
    source_after = manifest.get("source_sha256")
    expected_source_paths = set(current_source_hashes)
    if not isinstance(source_before, dict) or set(source_before) != expected_source_paths:
        raise ValueError(f"Residual seed {seed} pretraining source inventory mismatch")
    if not isinstance(source_after, dict) or set(source_after) != expected_source_paths:
        raise ValueError(f"Residual seed {seed} post-training source inventory mismatch")
    if any(not bu._is_sha256(value) for value in (*source_before.values(), *source_after.values())):
        raise ValueError(f"Residual seed {seed} source hashes are malformed")
    if source_before != source_after:
        raise ValueError(f"Residual seed {seed} source changed during training")
    warning: str | None = None
    if source_after != dict(current_source_hashes):
        if not allow_source_drift:
            raise ValueError(f"Residual seed {seed} source hashes differ from current source")
        warning = (
            f"source-drift: residual seed {seed} source hashes differ from current source; "
            "this is not a strict-integrity evaluation"
        )

    checkpoint = require_regular_file(run_dir / "best_validation.pt", run_dir)
    if checkpoint.stat().st_size <= 0 or Path(str(manifest.get("checkpoint", ""))).resolve() != checkpoint.resolve():
        raise ValueError(f"Residual seed {seed} checkpoint path/size mismatch")
    if manifest.get("checkpoint_sha256") != _sha256_file(checkpoint):
        raise ValueError(f"Residual seed {seed} checkpoint hash mismatch")

    normalization = manifest.get("normalization")
    residual_scaler = manifest.get("residual_scaler")
    if not isinstance(normalization, dict) or normalization.get("fit_scope") != "all scalar values in X_train.npy and y_train.npy only":
        raise ValueError(f"Residual seed {seed} utilization normalization provenance mismatch")
    if not isinstance(residual_scaler, dict) or residual_scaler.get("fit_scope") != "training residuals from X_train.npy and y_train.npy only":
        raise ValueError(f"Residual seed {seed} residual normalization provenance mismatch")
    try:
        input_mean, input_scale = float(normalization["mean"]), float(normalization["scale"])
        target_mean, target_scale = float(residual_scaler["mean"]), float(residual_scaler["scale"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Residual seed {seed} normalization parameters are malformed") from exc
    if not np.isfinite([input_mean, input_scale, target_mean, target_scale]).all() or input_scale <= 0 or target_scale <= 0:
        raise ValueError(f"Residual seed {seed} normalization parameters are invalid")
    return [warning] if warning else []


def _validate_bu_run_metadata(
    manifest: Mapping[str, Any],
    *,
    seed: int,
    data_root: Path,
    provenance_hashes: Mapping[str, str],
    provenance_file_hash: str,
    repository_root: Path,
    runtime_edge_order: Sequence[tuple[str, str]],
    expected_array_fingerprints: Mapping[str, str],
    expected_array_shapes: Mapping[str, Sequence[int]],
    expected_sample_counts: Mapping[str, int],
    allow_source_drift: bool = False,
) -> list[str]:
    """Enforce the complete future B-U metadata contract before checkpoint use."""
    if manifest.get("mode") != "full" or manifest.get("candidate") != "B-U":
        raise ValueError(f"B-U seed {seed} is not a full B-U run")
    if manifest.get("model") != "EdgeQoSForecaster (unchanged)" or manifest.get("model_settings") != bu.TRAINING:
        raise ValueError(f"B-U seed {seed} architecture/settings mismatch")
    if manifest.get("history_length") != 5 or manifest.get("prediction_horizon") != 1:
        raise ValueError(f"B-U seed {seed} history/horizon mismatch")
    if manifest.get("run_id") is not None:
        raise ValueError(f"B-U seed {seed} run path does not match the fixed allow-list")
    validate_input_array_path_allowlist(
        manifest.get("array_paths"),
        data_root=data_root,
        repository_root=repository_root,
        array_names=bu.ARRAY_NAMES,
        field_name=f"B-U seed {seed} array_paths",
    )
    if manifest.get("data_split") != {
        "training": "X_train.npy and y_train.npy",
        "validation": "X_validation.npy and y_validation.npy",
        "test": "not loaded or evaluated",
    }:
        raise ValueError(f"B-U seed {seed} split declaration mismatch")

    bu.validate_required_run_metadata(
        manifest,
        expected_edges=runtime_edge_order,
        expected_fingerprints=dict(expected_array_fingerprints),
    )
    expected_provenance = {
        "file": provenance_file_hash,
        "train": provenance_hashes["train"],
        "validation": provenance_hashes["validation"],
    }
    if manifest.get("window_provenance_sha256") != expected_provenance:
        raise ValueError(f"B-U seed {seed} train/validation provenance identity mismatch")

    source_hashes = manifest.get("source_sha256")
    if set(source_hashes) != set(bu.RUN_IDENTITY_SOURCE_FILES):
        raise ValueError(f"B-U seed {seed} source-hash inventory mismatch")
    if manifest.get("source_sha256_posttraining") != source_hashes:
        raise ValueError(f"B-U seed {seed} source hashes changed during its run")
    current_source_hashes = bu._source_sha256(repository_root)
    warning: str | None = None
    if source_hashes != current_source_hashes:
        if not allow_source_drift:
            raise ValueError(f"B-U seed {seed} source hashes do not match the reviewed current source files")
        warning = (
            f"source-drift: B-U seed {seed} source hashes differ from current source; "
            "this is not a strict-integrity evaluation"
        )
    if not isinstance(expected_sample_counts.get("train"), int):
        raise ValueError("Expected train sample count is malformed")
    if manifest.get("sample_counts") != dict(expected_sample_counts):
        raise ValueError(f"B-U seed {seed} sample counts mismatch")
    shapes = manifest.get("array_shapes")
    if not isinstance(shapes, dict) or {
        name: shapes.get(name) for name in expected_array_shapes
    } != {name: list(shape) for name, shape in expected_array_shapes.items()}:
        raise ValueError(f"B-U seed {seed} train/validation array shapes mismatch")

    expected_identity = bu._preflight_run_identity(
        mode="full",
        seed=seed,
        run_id=None,
        provenance_file_sha256=provenance_file_hash,
        provenance_hashes=dict(provenance_hashes),
        edges=runtime_edge_order,
        source_sha256=dict(source_hashes),
    )
    for field in (
        "configuration_sha256", "edge_order_sha256", "directed_edge_order",
    ):
        if manifest.get(field) != expected_identity[field]:
            raise ValueError(f"B-U seed {seed} {field} identity mismatch")

    normalization = manifest.get("normalization")
    if not isinstance(normalization, dict) or normalization.get("fit_scope") != "all scalar values in X_train.npy and y_train.npy only":
        raise ValueError(f"B-U seed {seed} normalization provenance is missing or invalid")
    try:
        mean, scale = float(normalization["mean"]), float(normalization["scale"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"B-U seed {seed} normalization parameters are malformed") from exc
    if not np.isfinite([mean, scale]).all() or scale <= 0:
        raise ValueError(f"B-U seed {seed} normalization parameters are invalid")
    return [warning] if warning else []


def _load_validation_arrays(data_root: Path, repository_root: Path) -> tuple[np.ndarray, np.ndarray]:
    root = validate_v2_root(data_root, repository_root)
    x = load_permitted_split_array(root, "X_validation.npy")
    y = load_permitted_split_array(root, "y_validation.npy")
    if x.dtype != np.float32 or y.dtype != np.float32 or x.ndim != 3 or x.shape[1] != 5:
        raise ValueError("Validation arrays do not match the v2 float32 five-history schema")
    if y.shape != (x.shape[0], x.shape[2]) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Validation arrays are misaligned or nonfinite")
    return x, y


def _load_residual_model(
    seed: int,
    data_root: Path,
    repository_root: Path,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    provenance_hashes: Mapping[str, str],
    provenance_file_hash: str,
    runtime_edge_order: Sequence[tuple[str, str]],
    expected_array_fingerprints: Mapping[str, str],
    expected_array_shapes: Mapping[str, Sequence[int]],
    expected_sample_counts: Mapping[str, int],
    *,
    allow_source_drift: bool = False,
) -> tuple[np.ndarray, dict[str, Any], Path]:
    name = f"{EXPECTED_RESIDUAL_PREFIX}{seed}"
    _, run_dir = validate_existing_child(data_root, data_root / name, repository_root, expected_name=name)
    manifest_path = require_regular_file(run_dir / "run_metadata.json", run_dir)
    manifest = _load_json(manifest_path, run_dir)
    _validate_residual_run_metadata(
        manifest,
        seed=seed,
        run_dir=run_dir,
        data_root=data_root,
        repository_root=repository_root,
        provenance_hashes=provenance_hashes,
        provenance_file_hash=provenance_file_hash,
        runtime_edge_order=runtime_edge_order,
        expected_array_fingerprints=expected_array_fingerprints,
        expected_array_shapes=expected_array_shapes,
        expected_sample_counts=expected_sample_counts,
        current_source_hashes=residual._source_hashes(repository_root),
        allow_source_drift=allow_source_drift,
    )
    if manifest.get("status") != "complete" or manifest.get("seed") != seed or manifest.get("protocol_id") != PROTOCOL_ID:
        raise ValueError(f"Residual seed {seed} manifest is incomplete or incompatible")
    if manifest.get("model") != {"class": "PerLinkResidualMLP", "architecture": "shared 5 -> 16 -> ReLU -> 1"}:
        raise ValueError(f"Residual seed {seed} model identity mismatch")
    if manifest.get("ewma", {}).get("alpha") != 0.8:
        raise ValueError(f"Residual seed {seed} EWMA setting mismatch")
    if manifest.get("configuration") != residual.TRAINING:
        raise ValueError(f"Residual seed {seed} configuration mismatch")
    if manifest.get("split_bounds") != {name: list(bounds) for name, bounds in SPLIT_BOUNDS.items()}:
        raise ValueError(f"Residual seed {seed} split bounds mismatch")
    if manifest.get("history_length") != 5 or manifest.get("prediction_horizon") != 1:
        raise ValueError(f"Residual seed {seed} history/horizon mismatch")
    if manifest.get("window_provenance_sha256", {}).get("file") != provenance_file_hash:
        raise ValueError(f"Residual seed {seed} sidecar provenance mismatch")
    if manifest.get("window_provenance_sha256", {}).get("validation") != provenance_hashes["validation"]:
        raise ValueError(f"Residual seed {seed} validation provenance mismatch")
    validate_edge_order(manifest.get("edge_order"), runtime_edge_order, format_name="residual")
    expected_shapes = {
        "X_validation": list(x_validation.shape),
        "y_validation": list(y_validation.shape),
    }
    if any(manifest.get("array_shapes", {}).get(key) != shape for key, shape in expected_shapes.items()):
        raise ValueError(f"Residual seed {seed} validation array shape mismatch")
    if manifest.get("sample_counts", {}).get("validation") != len(x_validation):
        raise ValueError(f"Residual seed {seed} validation sample count mismatch")
    if manifest.get("array_fingerprints", {}).get("X_validation.npy") != array_fingerprint(x_validation):
        raise ValueError(f"Residual seed {seed} validation input fingerprint mismatch")
    if manifest.get("array_fingerprints", {}).get("y_validation.npy") != array_fingerprint(y_validation):
        raise ValueError(f"Residual seed {seed} validation target fingerprint mismatch")
    checkpoint = require_regular_file(run_dir / "best_validation.pt", run_dir)
    if Path(str(manifest.get("checkpoint", ""))).resolve() != checkpoint.resolve():
        raise ValueError(f"Residual seed {seed} checkpoint path mismatch")
    checkpoint_hash = _sha256_file(checkpoint)
    if manifest.get("checkpoint_sha256") != checkpoint_hash:
        raise ValueError(f"Residual seed {seed} checkpoint identity mismatch")
    normalization_record = manifest.get("normalization", {})
    residual_record = manifest.get("residual_scaler", {})
    if normalization_record.get("fit_scope") != "all scalar values in X_train.npy and y_train.npy only":
        raise ValueError("Residual utilization normalization is not training-only")
    if residual_record.get("fit_scope") != "training residuals from X_train.npy and y_train.npy only":
        raise ValueError("Residual target normalization is not training-only")
    scaler = bu.Normalization(float(normalization_record["mean"]), float(normalization_record["scale"]))
    residual_scaler = residual.ResidualScaler(float(residual_record["mean"]), float(residual_record["scale"]))
    if not np.isfinite(scaler.mean) or not np.isfinite(scaler.scale) or scaler.scale <= 0:
        raise ValueError("Invalid residual model input scaler")
    if not np.isfinite(residual_scaler.mean) or not np.isfinite(residual_scaler.scale) or residual_scaler.scale <= 0:
        raise ValueError("Invalid residual-target scaler")
    model = residual.PerLinkResidualMLP()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    raw_residual = residual._predict(
        model,
        x_validation,
        scaler,
        residual_scaler,
        batch_size=residual.TRAINING["batch_size"],
    )
    return residual.reconstruct_forecast(x_validation, raw_residual), manifest, checkpoint


def _load_bu_model(
    seed: int,
    data_root: Path,
    repository_root: Path,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    provenance_hashes: Mapping[str, str],
    provenance_file_hash: str,
    runtime_edge_order: Sequence[tuple[str, str]],
    edge_index: torch.Tensor,
    node_count: int,
    expected_array_fingerprints: Mapping[str, str],
    expected_array_shapes: Mapping[str, Sequence[int]],
    expected_sample_counts: Mapping[str, int],
) -> tuple[np.ndarray, dict[str, Any], Path]:
    name = f"{EXPECTED_BU_PREFIX}{seed}"
    _, run_dir = validate_existing_child(data_root, data_root / name, repository_root, expected_name=name)
    manifest = _load_json(require_regular_file(run_dir / "run_metadata.json", run_dir), run_dir)
    if manifest.get("status") != "complete" or manifest.get("mode") != "full" or manifest.get("seed") != seed:
        raise ValueError(f"B-U seed {seed} is not a completed full run")
    if manifest.get("protocol_id") != PROTOCOL_ID or manifest.get("candidate") != "B-U":
        raise ValueError(f"B-U seed {seed} protocol/model mismatch")
    _validate_bu_run_metadata(
        manifest,
        seed=seed,
        data_root=data_root,
        provenance_hashes=provenance_hashes,
        provenance_file_hash=provenance_file_hash,
        repository_root=repository_root,
        runtime_edge_order=runtime_edge_order,
        expected_array_fingerprints=expected_array_fingerprints,
        expected_array_shapes=expected_array_shapes,
        expected_sample_counts=expected_sample_counts,
    )
    checkpoint = require_regular_file(run_dir / "best_validation.pt", run_dir)
    if Path(str(manifest.get("checkpoint", ""))).resolve() != checkpoint.resolve() or checkpoint.stat().st_size <= 0:
        raise ValueError(f"B-U seed {seed} checkpoint identity/size mismatch")
    checkpoint_hash = _sha256_file(checkpoint)
    if manifest.get("checkpoint_sha256") != checkpoint_hash:
        raise ValueError(f"B-U seed {seed} checkpoint hash is absent or mismatched")
    normalization_record = manifest.get("normalization", {})
    if normalization_record.get("fit_scope") != "all scalar values in X_train.npy and y_train.npy only":
        raise ValueError("B-U normalization is not documented as training-only")
    normalization = bu.Normalization(float(normalization_record["mean"]), float(normalization_record["scale"]))
    if normalization.scale <= 0 or not np.isfinite([normalization.mean, normalization.scale]).all():
        raise ValueError("Invalid B-U normalization")
    model = EdgeQoSForecaster(edge_feature_count=1, target_count=1, hidden_channels=16, gru_hidden_size=16)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for begin in range(0, len(x_validation), bu.TRAINING["batch_size"]):
            end = min(begin + bu.TRAINING["batch_size"], len(x_validation))
            sequence = bu.make_bu_sequence(x_validation[begin:end], edge_index, node_count, normalization)
            standardized = model(sequence).reshape(end - begin, edge_index.shape[1])
            predictions.append((standardized.cpu().numpy().astype(np.float64) * normalization.scale) + normalization.mean)
    prediction = np.concatenate(predictions, axis=0)
    if prediction.shape != y_validation.shape or not np.isfinite(prediction).all():
        raise ValueError("B-U validation prediction is not aligned and finite")
    return prediction, manifest, checkpoint


def run_validation_evaluation(
    *,
    data_root: Path = bu.V2_DATA_ROOT,
    repository_root: Path = bu.REPOSITORY_ROOT,
    output_path: Path | None = None,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    allow_source_drift: bool = False,
) -> Path:
    output = output_path or (data_root / OUTPUT_NAME)
    root, output = validate_fresh_child(
        data_root, output, repository_root, expected_name=OUTPUT_NAME
    )
    x_validation, y_validation = _load_validation_arrays(data_root, repository_root)
    provenance_path = require_regular_file(root / PROVENANCE_FILENAME, root)
    provenance_raw = provenance_path.read_bytes()
    provenance = json.loads(provenance_raw.decode("utf-8"))
    provenance_file_hash = hashlib.sha256(provenance_raw).hexdigest()
    provenance_hashes = validate_train_validation_provenance(
        provenance,
        {"validation": (x_validation, y_validation)},
        required_splits=("validation",),
    )
    train_block = provenance["splits"]["train"]
    validation_block = provenance["splits"]["validation"]
    provenance_hashes = {
        "train": canonical_json_sha256(train_block),
        **provenance_hashes,
    }
    # The train array fingerprints come from the exact train-only sidecar
    # covered by provenance_file_hash. Validation fingerprints are checked
    # directly against the arrays loaded for this evaluation.
    expected_array_fingerprints = {
        **train_block["array_fingerprints"],
        "X_validation.npy": array_fingerprint(x_validation),
        "y_validation.npy": array_fingerprint(y_validation),
    }
    nodes, undirected = bu.parse_topology(str(bu.TOPOLOGY_PATH))
    if len(nodes) != 22 or len(undirected) != 36:
        raise ValueError("Runtime topology differs from the frozen v2 structure")
    edge_index, node_order, edge_order = bu.build_edge_index(tuple(nodes), undirected)
    if len(edge_order) != 72 or x_validation.shape[2] != len(edge_order):
        raise ValueError("Validation link columns do not match the directed topology")
    train_count = train_block["array_rows"]
    expected_array_shapes = {
        "X_train": [train_count, 5, len(edge_order)],
        "y_train": [train_count, len(edge_order)],
        "X_validation": list(x_validation.shape),
        "y_validation": list(y_validation.shape),
    }
    expected_sample_counts = {
        "train": train_count,
        "validation": len(x_validation),
    }
    integrity_warnings = _preflight_completed_artifacts(
        root, repository_root, x_validation, y_validation,
        provenance_hashes, provenance_file_hash, edge_order,
        expected_array_fingerprints, expected_array_shapes, expected_sample_counts,
        allow_source_drift=allow_source_drift,
    )
    residual_predictions: dict[int, np.ndarray] = {}
    residual_manifests: dict[int, dict[str, Any]] = {}
    residual_checkpoints: dict[int, Path] = {}
    bu_predictions: dict[int, np.ndarray] = {}
    bu_manifests: dict[int, dict[str, Any]] = {}
    bu_checkpoints: dict[int, Path] = {}
    for seed in SEEDS:
        rp, rm, rc = _load_residual_model(
            seed, root, repository_root, x_validation, y_validation,
            provenance_hashes, provenance_file_hash, edge_order,
            expected_array_fingerprints, expected_array_shapes, expected_sample_counts,
            allow_source_drift=allow_source_drift,
        )
        bp, bm, bc = _load_bu_model(
            seed, root, repository_root, x_validation, y_validation,
            provenance_hashes, provenance_file_hash,
            edge_order, edge_index, len(node_order),
            expected_array_fingerprints, expected_array_shapes, expected_sample_counts,
        )
        residual_predictions[seed], residual_manifests[seed], residual_checkpoints[seed] = rp, rm, rc
        bu_predictions[seed], bu_manifests[seed], bu_checkpoints[seed] = bp, bm, bc

    weights = residual.ewma_weights(alpha=0.8)
    ewma_prediction = np.tensordot(x_validation.astype(np.float64), weights, axes=([1], [0]))
    per_seed: dict[str, Any] = {}
    for seed in SEEDS:
        per_seed[str(seed)] = build_seed_comparison(
            y_validation,
            residual_predictions[seed],
            ewma_prediction,
            bu_predictions[seed],
            bootstrap_seed=bootstrap_seed,
            bootstrap_replicates=bootstrap_replicates,
        )
    seed_metrics = {
        method: {str(seed): per_seed[str(seed)]["metrics"][method] for seed in SEEDS}
        for method in ("residual_ewma", "ewma", "B-U")
    }
    aggregate = {
        method: aggregate_seed_metrics({seed: {metric: values[metric] for metric in ("mae", "rmse")} for seed, values in metrics_by_seed.items()})
        for method, metrics_by_seed in seed_metrics.items()
    }
    gate = usefulness_gate(
        [seed_metrics["residual_ewma"][str(seed)]["mae"] for seed in SEEDS],
        [seed_metrics["ewma"][str(seed)]["mae"] for seed in SEEDS],
    )
    result = {
        "protocol_id": PROTOCOL_ID,
        "split": "validation",
        "validation_bounds_half_open": list(SPLIT_BOUNDS["validation"]),
        "history_length": 5,
        "prediction_horizon": 1,
        "units": "dimensionless computed link-utilization ratio",
        "metric_rules": {
            "mae": "mean absolute error over every finite window-link pair, float64",
            "rmse": "square root mean squared error over every finite window-link pair, float64",
            "coverage": "all methods must match full target shape and be finite; no intersection/drop mask",
        },
        "sample_count": int(len(y_validation)),
        "link_count": int(y_validation.shape[1]),
        "scalar_target_count": int(y_validation.size),
        "validation_provenance_sha256": {
            "sidecar": provenance_file_hash,
            "validation": provenance_hashes["validation"],
        },
        "methods": {
            "residual_ewma": "fixed per-link MLP residual forecast added to alpha=0.8 EWMA",
            "ewma": {"formula": "normalized alpha*(1-alpha)^(4-k), oldest-to-newest", "alpha": 0.8},
            "B-U": "fixed EdgeQoSForecaster checkpoint; contextual comparison only",
        },
        "per_seed": per_seed,
        "aggregate_across_seeds_mean_and_sample_std_ddof_1": aggregate,
        "aggregate_absolute_mae_difference_residual_minus_ewma": float(np.mean([
            per_seed[str(seed)]["paired_absolute_error_difference_vs_ewma"]["residual_ewma"]["mean_absolute_error_difference"]
            for seed in SEEDS
        ], dtype=np.float64)),
        "usefulness_gate": gate,
        "interpretation": "Validation evidence only; checkpoint selection also used validation. The gate is not statistical significance or held-out generalization. B-U does not influence the gate.",
        "integrity_warnings": integrity_warnings,
        "strict_integrity_evaluation": not bool(integrity_warnings),
        "artifacts": {
            "residual_checkpoints": {str(seed): str(residual_checkpoints[seed]) for seed in SEEDS},
            "residual_checkpoint_sha256": {str(seed): residual_manifests[seed]["checkpoint_sha256"] for seed in SEEDS},
            "bu_checkpoints": {str(seed): str(bu_checkpoints[seed]) for seed in SEEDS},
            "bu_checkpoint_sha256": {str(seed): _sha256_file(bu_checkpoints[seed]) for seed in SEEDS},
            "input_arrays": ["X_validation.npy", "y_validation.npy"],
        },
        "bootstrap": {
            "method": "paired window bootstrap percentile interval; resample windows with replacement and retain each selected window's complete link-error vector",
            "confidence": BOOTSTRAP_CONFIDENCE,
            "replicates": bootstrap_replicates,
            "rng_seed": bootstrap_seed,
        },
        "test_access": "No test arrays, predictions, or metrics are loaded by this evaluator",
    }
    # Fresh output is created only after inputs, provenance, manifests, and
    # checkpoints have passed validation. No existing result is overwritten.
    output.mkdir(parents=False, exist_ok=False)
    try:
        _, checked_output = validate_existing_child(root, output, repository_root, expected_name=OUTPUT_NAME)
        content = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
        target = checked_output / "results.json"
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite result: {target}")
        descriptor, temp_name = tempfile.mkstemp(prefix=".results.", suffix=".tmp", dir=checked_output)
        temp_path = Path(temp_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, target)
        except BaseException:
            temp_path.unlink(missing_ok=True)
            raise
        return checked_output
    except BaseException:
        # Preserve diagnostic partials. The directory was created exclusively
        # by this invocation; do not remove it or retry automatically.
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-validation-only", action="store_true", required=True)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=bu.V2_DATA_ROOT,
        help="canonical v2 root or explicitly versioned rebuild root",
    )
    parser.add_argument(
        "--allow-source-drift",
        action="store_true",
        help="explicitly permit source-hash drift; output will be marked non-strict-integrity",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(run_validation_evaluation(
        data_root=args.data_root,
        allow_source_drift=args.allow_source_drift,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

