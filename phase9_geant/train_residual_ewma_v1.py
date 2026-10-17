"""Isolated train/validation runner for GEANT v2 residual-to-EWMA forecasting.

This entry point is deliberately opt-in. It reads only the four fixed v2
training/validation arrays and never implements a test-data loading path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn

from phase9_geant import train_bu_v2 as bu
import phase9_geant.parse_geant as topology_parser
from phase9_geant.validity_v2 import HISTORY_STEPS, PREDICTION_HORIZON, PROTOCOL_ID, SPLIT_BOUNDS, STEP
from phase9_geant.safe_paths import (
    canonical_repository_root,
    checked_absolute,
    require_regular_file,
    validate_fresh_child,
    validate_v2_root,
    verify_new_directory_containment,
)
from phase9_geant.split_access import load_permitted_split_array
from phase9_geant.window_provenance_v2 import (
    PROVENANCE_FILENAME,
    array_fingerprint,
    canonical_json_sha256,
    validate_train_validation_provenance,
)


SCRIPT_VERSION = "1.0.0"
EWMA_ALPHA = 0.8
SEEDS = (42, 123, 2026)
TRAINING = {
    "max_epochs": 5,
    "early_stopping_patience": 2,
    "learning_rate": 0.005,
    "batch_size": 32,
    "optimizer": "Adam",
    "loss": "SmoothL1Loss",
    "initialization": "from_scratch",
    "mlp_input_features": 5,
    "mlp_hidden_features": 16,
    "mlp_activation": "ReLU",
    "mlp_output_features": 1,
}
INPUT_ARRAY_NAMES = ("X_train.npy", "y_train.npy", "X_validation.npy", "y_validation.npy")
V2_DATA_ROOT = bu.V2_DATA_ROOT
TOPOLOGY_PATH = bu.TOPOLOGY_PATH


@dataclass(frozen=True)
class ResidualScaler:
    mean: float
    scale: float
    fit_scope: str = "training residuals from X_train.npy and y_train.npy only"


class PerLinkResidualMLP(nn.Module):
    """Shared temporal MLP applied independently to each directed link."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(5, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, per_link_history: torch.Tensor) -> torch.Tensor:
        if per_link_history.ndim != 2 or per_link_history.shape[1] != HISTORY_STEPS:
            raise ValueError("MLP input must have shape [samples * links, 5]")
        return self.network(per_link_history)


def ewma_weights(*, alpha: float = EWMA_ALPHA) -> np.ndarray:
    if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("EWMA alpha must be finite and between zero and one")
    raw = alpha * (1.0 - alpha) ** (HISTORY_STEPS - 1 - np.arange(HISTORY_STEPS, dtype=np.float64))
    return raw / raw.sum(dtype=np.float64)


def ewma_forecast(history: np.ndarray, *, alpha: float = EWMA_ALPHA) -> np.ndarray:
    history = np.asarray(history)
    if history.ndim != 3 or history.shape[1] != HISTORY_STEPS or history.shape[2] == 0:
        raise ValueError("History must have shape [windows, 5, directed_links]")
    if not np.isfinite(history).all():
        raise ValueError("History contains nonfinite utilization values")
    weights = ewma_weights(alpha=alpha)
    return np.tensordot(history.astype(np.float64), weights, axes=([1], [0]))


def residual_targets(history: np.ndarray, target: np.ndarray) -> np.ndarray:
    target = np.asarray(target)
    baseline = ewma_forecast(history)
    if target.shape != baseline.shape:
        raise ValueError(f"Target shape {target.shape} does not align with EWMA shape {baseline.shape}")
    if not np.isfinite(target).all():
        raise ValueError("Target contains nonfinite utilization values")
    return target.astype(np.float64) - baseline


def fit_residual_scaler(training_residuals: np.ndarray) -> ResidualScaler:
    values = np.asarray(training_residuals, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Training residuals must be nonempty and finite")
    scale = float(values.std(dtype=np.float64))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Training residual scale must be finite and positive")
    return ResidualScaler(mean=float(values.mean(dtype=np.float64)), scale=scale)


def make_per_link_inputs(history: np.ndarray, utilization_scaler: bu.Normalization) -> torch.Tensor:
    """Return [windows * links, 5] inputs, preserving each link's own history."""
    values = np.asarray(history)
    if values.ndim != 3 or values.shape[1] != HISTORY_STEPS or values.shape[2] == 0:
        raise ValueError("History must have shape [windows, 5, directed_links]")
    if not np.isfinite(values).all():
        raise ValueError("History contains nonfinite utilization values")
    per_link = values.astype(np.float64).transpose(0, 2, 1)
    standardized = (per_link - utilization_scaler.mean) / utilization_scaler.scale
    return torch.as_tensor(standardized.reshape(-1, HISTORY_STEPS).astype(np.float32))


def standardize_residuals(values: np.ndarray, scaler: ResidualScaler) -> torch.Tensor:
    residual = np.asarray(values, dtype=np.float64)
    if not np.isfinite(residual).all():
        raise ValueError("Residual values contain nonfinite entries")
    return torch.as_tensor(((residual - scaler.mean) / scaler.scale).astype(np.float32).reshape(-1, 1))


def inverse_scale_residual(values: np.ndarray | torch.Tensor, scaler: ResidualScaler) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    result = np.asarray(values, dtype=np.float64) * scaler.scale + scaler.mean
    if not np.isfinite(result).all():
        raise ValueError("Inverse-scaled residual prediction is nonfinite")
    return result


def reconstruct_forecast(history: np.ndarray, predicted_residual: np.ndarray) -> np.ndarray:
    baseline = ewma_forecast(history)
    residual = np.asarray(predicted_residual, dtype=np.float64)
    if residual.shape != baseline.shape or not np.isfinite(residual).all():
        raise ValueError("Residual predictions must be finite and match the EWMA forecast shape")
    return baseline + residual


def validation_loss_improved(candidate: float, best: float) -> bool:
    """Checkpoint decision is based only on standardized validation residual loss."""
    if not np.isfinite(candidate):
        raise ValueError("Validation residual loss must be finite")
    return candidate < best


def validate_window_starts(
    starts: Sequence[int],
    timestamps: Sequence[datetime],
    valid_mask: Sequence[bool],
    *,
    split_name: str,
) -> None:
    """Check synthetic or supplied starts against v2's official split rules."""
    if split_name not in SPLIT_BOUNDS:
        raise ValueError("Unknown official split")
    start, end = SPLIT_BOUNDS[split_name]
    if end > len(timestamps) or len(timestamps) != len(valid_mask):
        raise ValueError("Timestamp and validity metadata do not cover the declared split")
    normalized_starts = tuple(int(item) for item in starts)
    if tuple(sorted(set(normalized_starts))) != normalized_starts:
        raise ValueError("Window starts must be unique and chronological")
    for first in normalized_starts:
        last = first + HISTORY_STEPS + PREDICTION_HORIZON
        if first < start or last > end:
            raise ValueError("Window crosses an official split boundary")
        if not all(bool(value) for value in valid_mask[first:last]):
            raise ValueError("Window includes an invalid observation")
        if any(timestamps[i] - timestamps[i - 1] != STEP for i in range(first + 1, last)):
            raise ValueError("Window crosses a chronology gap")


def validate_directed_edge_order(
    nodes: Sequence[str], undirected_links: Sequence[tuple[str, str]], *, expected_links: int = 72
) -> tuple[tuple[str, str], ...]:
    edge_index, node_order, edge_order = bu.build_edge_index(tuple(nodes), undirected_links)
    if len(edge_order) != expected_links:
        raise ValueError(f"Expected {expected_links} directed links, found {len(edge_order)}")
    expected = bu.directed_edges_v2(undirected_links)
    bu.assert_edge_order(expected, edge_order)
    reconstructed = tuple((node_order[int(src)], node_order[int(dst)]) for src, dst in edge_index.T.tolist())
    bu.assert_edge_order(expected, reconstructed)
    return edge_order


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
        raise


def _atomic_torch_save(state: dict[str, torch.Tensor], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(state, temporary)
        with temporary.open("rb+") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_paths(repository_root: Path) -> tuple[Path, ...]:
    return tuple(repository_root / item for item in (
        "phase9_geant/train_residual_ewma_v1.py",
        "phase9_geant/train_bu_v2.py",
        "phase9_geant/validity_v2.py",
        "phase9_geant/build_validity_v2.py",
        "phase9_geant/window_provenance_v2.py",
        "phase9_geant/safe_paths.py",
        "phase9_geant/split_access.py",
        "phase9_geant/parse_geant.py",
        "phase9_geant/run_baselines_v2_validation.py",
    ))


def _hash_source_files(paths: Sequence[Path]) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in paths:
        checked_absolute(path)
        resolved = path.resolve(strict=True)
        if not resolved.is_file() or resolved.parent != path.parent.resolve(strict=True):
            raise ValueError(f"Source file is not a regular direct file: {path}")
        result[str(resolved)] = hashlib.sha256(resolved.read_bytes()).hexdigest()
    return result


def _source_hashes(repository_root: Path = bu.REPOSITORY_ROOT) -> dict[str, str]:
    """Hash implementation files before model construction; no data paths enter."""
    root = canonical_repository_root(repository_root)
    return _hash_source_files(_source_paths(root))


def _configuration_hash() -> str:
    value = {
        "protocol_id": PROTOCOL_ID,
        "seed_set": list(SEEDS),
        "ewma_alpha": EWMA_ALPHA,
        "training": TRAINING,
        "history_length": HISTORY_STEPS,
        "prediction_horizon": PREDICTION_HORIZON,
        "split_bounds": {name: list(bounds) for name, bounds in SPLIT_BOUNDS.items()},
    }
    return canonical_json_sha256(value)


def _validate_run_configuration(seed: int) -> None:
    if seed not in SEEDS:
        raise ValueError(f"Seed must be one of {SEEDS}")
    if PROTOCOL_ID != "geant-validity-v2-train-median" or HISTORY_STEPS != 5 or PREDICTION_HORIZON != 1:
        raise ValueError("The residual experiment differs from the frozen v2 protocol")
    if EWMA_ALPHA != 0.8 or TRAINING != {
        "max_epochs": 5,
        "early_stopping_patience": 2,
        "learning_rate": 0.005,
        "batch_size": 32,
        "optimizer": "Adam",
        "loss": "SmoothL1Loss",
        "initialization": "from_scratch",
        "mlp_input_features": 5,
        "mlp_hidden_features": 16,
        "mlp_activation": "ReLU",
        "mlp_output_features": 1,
    }:
        raise ValueError("Residual experiment configuration differs from the reviewed settings")


def _metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    target64 = np.asarray(target, dtype=np.float64)
    prediction64 = np.asarray(prediction, dtype=np.float64)
    if target64.shape != prediction64.shape or not np.isfinite(target64).all() or not np.isfinite(prediction64).all():
        raise ValueError("Metrics require aligned, fully finite target and prediction arrays")
    error = prediction64 - target64
    return {
        "mae": float(np.mean(np.abs(error), dtype=np.float64)),
        "rmse": float(np.sqrt(np.mean(np.square(error), dtype=np.float64))),
        "evaluated_scalar_targets": int(target64.size),
    }


def _validation_objective(
    model: PerLinkResidualMLP,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    utilization_scaler: bu.Normalization,
    residual_scaler: ResidualScaler,
    *,
    batch_size: int,
) -> float:
    criterion = nn.SmoothL1Loss(reduction="sum")
    total = 0.0
    count = 0
    model.eval()
    with torch.no_grad():
        for begin in range(0, len(x_validation), batch_size):
            end = min(begin + batch_size, len(x_validation))
            inputs = make_per_link_inputs(x_validation[begin:end], utilization_scaler)
            residual = residual_targets(x_validation[begin:end], y_validation[begin:end])
            targets = standardize_residuals(residual, residual_scaler)
            prediction = model(inputs)
            if prediction.shape != targets.shape or not torch.isfinite(prediction).all():
                raise ValueError("Validation residual model output is invalid")
            total += float(criterion(prediction, targets))
            count += targets.numel()
    if count == 0:
        raise ValueError("Validation objective has no scalar targets")
    return total / count


def _predict(
    model: PerLinkResidualMLP,
    history: np.ndarray,
    utilization_scaler: bu.Normalization,
    residual_scaler: ResidualScaler,
    *,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    predictions: list[np.ndarray] = []
    links = history.shape[2]
    with torch.no_grad():
        for begin in range(0, len(history), batch_size):
            end = min(begin + batch_size, len(history))
            standardized = model(make_per_link_inputs(history[begin:end], utilization_scaler))
            if not torch.isfinite(standardized).all():
                raise ValueError("Residual inference produced nonfinite values")
            raw = inverse_scale_residual(standardized, residual_scaler).reshape(end - begin, links)
            predictions.append(raw)
    if not predictions:
        raise ValueError("Cannot predict an empty history array")
    return np.concatenate(predictions, axis=0)


def output_dir_for_seed(
    seed: int,
    data_root: Path = V2_DATA_ROOT,
    *,
    repository_root: Path = bu.REPOSITORY_ROOT,
    output_path: Path | None = None,
) -> Path:
    if seed not in SEEDS:
        raise ValueError(f"Seed must be one of {SEEDS}")
    expected_name = f"neural_ewma_residual_v1_seed_{seed}"
    candidate = output_path if output_path is not None else Path(data_root) / expected_name
    _, validated = validate_fresh_child(
        data_root, candidate, repository_root, expected_name=expected_name
    )
    return validated


def load_train_validation_arrays(
    data_root: Path = V2_DATA_ROOT,
    *,
    repository_root: Path = bu.REPOSITORY_ROOT,
    array_loader=np.load,
) -> bu.WindowArrays:
    """Load only four allow-listed arrays after component-wise path checks."""
    root = validate_v2_root(data_root, repository_root)
    loaded: dict[str, np.ndarray] = {}
    for filename in INPUT_ARRAY_NAMES:
        loaded[filename] = load_permitted_split_array(
            root, filename, array_loader=array_loader
        )
    arrays = bu.WindowArrays(
        x_train=loaded["X_train.npy"],
        y_train=loaded["y_train.npy"],
        x_validation=loaded["X_validation.npy"],
        y_validation=loaded["y_validation.npy"],
    )
    bu.validate_window_arrays(arrays)
    return arrays


def load_window_provenance(data_root: Path, *, repository_root: Path = bu.REPOSITORY_ROOT) -> tuple[dict[str, Any], str]:
    root = validate_v2_root(data_root, repository_root)
    path = require_regular_file(root / PROVENANCE_FILENAME, root)
    raw = path.read_bytes()
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Window provenance sidecar must contain a JSON object")
    return value, hashlib.sha256(raw).hexdigest()


def _create_run_directory(output: Path, data_root: Path, repository_root: Path) -> None:
    output.mkdir(parents=False, exist_ok=False)
    # Never attempt path-based cleanup after creation. A concurrent process
    # could replace this pathname between mkdir and validation, and a later
    # rmdir could then remove that other process's directory. Leaving an empty
    # directory is the safe failure mode; recovery is an explicit operation.
    validate_v2_root(data_root, repository_root)
    verify_new_directory_containment(output, data_root)


def run_seed(
    seed: int,
    *,
    data_root: Path = V2_DATA_ROOT,
    repository_root: Path = bu.REPOSITORY_ROOT,
    array_loader=np.load,
) -> Path:
    _validate_run_configuration(seed)
    output = output_dir_for_seed(seed, data_root, repository_root=repository_root)
    # Validate provenance path before any data load, then load only the exact
    # four permitted arrays and train/validation-only sidecar.
    validated_root = validate_v2_root(data_root, repository_root)
    require_regular_file(validated_root / PROVENANCE_FILENAME, validated_root)
    arrays = load_train_validation_arrays(
        data_root, repository_root=repository_root, array_loader=array_loader
    )
    provenance, provenance_file_hash = load_window_provenance(
        data_root, repository_root=repository_root
    )
    provenance_hashes = validate_train_validation_provenance(
        provenance,
        {
            "train": (arrays.x_train, arrays.y_train),
            "validation": (arrays.x_validation, arrays.y_validation),
        },
    )
    source_hashes = _source_hashes(repository_root)
    config_hash = _configuration_hash()
    nodes, links = bu.parse_topology(str(TOPOLOGY_PATH))
    if len(nodes) != 22 or len(links) != 36:
        raise ValueError("Topology differs from the v2 structural contract")
    edge_order = validate_directed_edge_order(tuple(nodes), links)
    if arrays.x_train.shape[2] != len(edge_order) or arrays.x_validation.shape[2] != len(edge_order):
        raise ValueError("Array directed-link columns do not match the v2 topology contract")

    # No output is created until root, fixed inputs, sidecar, fingerprints,
    # chronology, official validity rules, topology and source/config are checked.
    _create_run_directory(output, data_root, repository_root)
    manifest_path = output / "run_metadata.json"
    started = time.perf_counter()
    initial: dict[str, Any] = {
        "status": "incomplete",
        "experiment": "Neural Residual-to-EWMA Forecasting, Version 1",
        "protocol_id": PROTOCOL_ID,
        "seed": seed,
        "mode": "full",
        "configuration": TRAINING,
        "split_bounds": {name: list(bounds) for name, bounds in SPLIT_BOUNDS.items()},
        "history_length": HISTORY_STEPS,
        "prediction_horizon": PREDICTION_HORIZON,
        "input_artifacts": [str(Path(data_root) / name) for name in INPUT_ARRAY_NAMES],
        "array_shapes": {
            "X_train": list(arrays.x_train.shape),
            "y_train": list(arrays.y_train.shape),
            "X_validation": list(arrays.x_validation.shape),
            "y_validation": list(arrays.y_validation.shape),
        },
        "sample_counts": {
            "train": int(arrays.x_train.shape[0]),
            "validation": int(arrays.x_validation.shape[0]),
        },
        "array_fingerprints": {
            "X_train.npy": array_fingerprint(arrays.x_train),
            "y_train.npy": array_fingerprint(arrays.y_train),
            "X_validation.npy": array_fingerprint(arrays.x_validation),
            "y_validation.npy": array_fingerprint(arrays.y_validation),
        },
        "preprocessing_provenance": {
            "protocol_id": PROTOCOL_ID,
            "source": "existing GEANT validity-v2 prebuilt train/validation windows",
            "window_rule": "five consecutive valid history observations plus one valid next-step target within the same official split",
            "link_order": "72 directed columns in v2 interleaved forward/reverse order, checked against the topology parser",
            "utilization_units": "dimensionless computed link-utilization ratio",
            "window_provenance_file": PROVENANCE_FILENAME,
            "window_provenance_sha256": provenance_file_hash,
            "train_window_provenance_sha256": provenance_hashes["train"],
            "validation_window_provenance_sha256": provenance_hashes["validation"],
        },
        "source_sha256_pretraining": source_hashes,
        "configuration_sha256": config_hash,
        "test_access": "no test loader or test artifact path is implemented",
    }
    _atomic_json(manifest_path, initial)
    try:
        utilization_scaler = bu.fit_training_normalization(arrays.x_train, arrays.y_train)
        training_residuals = residual_targets(arrays.x_train, arrays.y_train)
        residual_scaler = fit_residual_scaler(training_residuals)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.set_num_threads(1)
        model = PerLinkResidualMLP()
        optimizer = torch.optim.Adam(model.parameters(), lr=TRAINING["learning_rate"])
        criterion = nn.SmoothL1Loss(reduction="sum")
        best_loss = float("inf")
        best_epoch = 0
        stale_epochs = 0
        stopped_early = False
        history: list[dict[str, float | int]] = []
        checkpoint_path = output / "best_validation.pt"

        train_count, links_count = arrays.x_train.shape[0], arrays.x_train.shape[2]
        for epoch in range(1, TRAINING["max_epochs"] + 1):
            model.train()
            order = np.random.permutation(train_count)
            epoch_sum, epoch_count = 0.0, 0
            for begin in range(0, train_count, TRAINING["batch_size"]):
                indices = order[begin : begin + TRAINING["batch_size"]]
                input_tensor = make_per_link_inputs(arrays.x_train[indices], utilization_scaler)
                raw_residual = training_residuals[indices]
                target_tensor = standardize_residuals(raw_residual, residual_scaler)
                prediction = model(input_tensor)
                if prediction.shape != target_tensor.shape or not torch.isfinite(prediction).all():
                    raise ValueError("Training residual model output is invalid")
                loss = criterion(prediction, target_tensor)
                if not torch.isfinite(loss):
                    raise ValueError("Training residual loss is nonfinite")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                epoch_sum += float(loss.detach())
                epoch_count += target_tensor.numel()

            val_loss = _validation_objective(
                model,
                arrays.x_validation,
                arrays.y_validation,
                utilization_scaler,
                residual_scaler,
                batch_size=TRAINING["batch_size"],
            )
            history.append({"epoch": epoch, "train_loss": epoch_sum / epoch_count, "validation_loss": val_loss})
            if validation_loss_improved(val_loss, best_loss):
                best_loss, best_epoch, stale_epochs = val_loss, epoch, 0
                _atomic_torch_save(model.state_dict(), checkpoint_path)
            else:
                stale_epochs += 1
            if stale_epochs >= TRAINING["early_stopping_patience"]:
                stopped_early = True
                break

        if best_epoch == 0 or not checkpoint_path.is_file() or checkpoint_path.stat().st_size <= 0:
            raise RuntimeError("Training did not produce a nonempty best-validation checkpoint")
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        prediction = reconstruct_forecast(
            arrays.x_validation,
            _predict(model, arrays.x_validation, utilization_scaler, residual_scaler, batch_size=TRAINING["batch_size"]),
        )
        ewma_prediction = ewma_forecast(arrays.x_validation)
        validation_metrics = {
            "residual_to_ewma": _metrics(arrays.y_validation, prediction),
            "ewma": _metrics(arrays.y_validation, ewma_prediction),
        }
        history_path = output / "training_history.json"
        _atomic_json(history_path, {"epochs": history})
        source_hashes_after_training = _source_hashes(repository_root)
        if source_hashes_after_training != source_hashes:
            raise RuntimeError("Source files changed during the run; refusing complete status")
        completed = {
            **initial,
            "status": "complete",
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "training_seconds": time.perf_counter() - started,
            "training_history_path": str(history_path.resolve()),
            "training_history": history,
            "epochs_completed": len(history),
            "best_validation_epoch": best_epoch,
            "best_validation_smooth_l1_standardized_residual": best_loss,
            "early_stopping": {"patience": TRAINING["early_stopping_patience"], "triggered": stopped_early},
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_sha256": _sha256_file(checkpoint_path),
            "checkpoint_format": "PyTorch state_dict; strict reload verified",
            "model": {"class": "PerLinkResidualMLP", "architecture": "shared 5 -> 16 -> ReLU -> 1"},
            "ewma": {"alpha": EWMA_ALPHA, "weights_oldest_to_newest": ewma_weights().tolist()},
            "normalization": asdict(utilization_scaler),
            "residual_scaler": asdict(residual_scaler),
            "edge_order": [list(edge) for edge in edge_order],
            "validation_metrics_original_utilization_ratio": validation_metrics,
            "metric_definition": "float64 MAE/RMSE over every finite validation window-link pair; no clipping or imputation",
            "checkpoint_selection": "minimum validation SmoothL1 on standardized residual targets only",
            "source_sha256": source_hashes_after_training,
            "source_sha256_pretraining": source_hashes,
            "configuration_sha256": config_hash,
            "window_provenance_sha256": {
                "file": provenance_file_hash,
                **provenance_hashes,
            },
            "runtime": {
                "python": __import__("sys").version,
                "numpy": np.__version__,
                "torch": torch.__version__,
                "deterministic_algorithms": True,
                "device": "cpu",
            },
            "interpretation_limit": "validation was used for checkpoint selection; results are exploratory, not held-out generalization",
            "manifest_path": str(manifest_path.resolve()),
        }
        _atomic_json(manifest_path, completed)
        return output
    except BaseException as error:
        failed = {
            **initial,
            "status": "failed",
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "training_seconds": time.perf_counter() - started,
            "exception": {"type": type(error).__name__, "message": str(error)[:500]},
        }
        try:
            _atomic_json(manifest_path, failed)
        except BaseException:
            pass
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", required=True, help="explicitly start one fresh training run")
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=V2_DATA_ROOT,
        help="canonical v2 root or explicitly versioned rebuild root",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = run_seed(args.seed, data_root=args.data_root)
    print(json.dumps({"output_directory": str(output), "status": "complete", "seed": args.seed}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
