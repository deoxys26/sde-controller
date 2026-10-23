"""Isolated, explicitly gated horizon-96-only GEANT candidate workflow.

The training path accepts only the h=96 train and validation arrays. It reuses
shared scaler/metric helpers from the joint experiment module, but never calls
its joint trainer entry point or creates a test-split array path. Importing this
module performs no dataset access and starts no training.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import random
import statistics
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from phase9_geant import horizon_v2_data as data_v2
from phase9_geant.horizon_v2_experiment import (
    SEEDS,
    Standardizer,
    fit_training_scaler,
    metric_summary,
    validation_loss_improved,
)
from phase9_geant.safe_paths import canonical_repository_root, checked_absolute, require_regular_file
from src.gnn.spatiotemporal_link import (
    SHUFFLE_SEED,
    SpatioTemporalLinkForecaster,
    build_link_adjacency,
)


PROTOCOL_ID = "geant-horizon96-only-candidate-v1"
HORIZON = 96
HORIZONS = (HORIZON,)
HISTORY_LENGTH = data_v2.HISTORY_LENGTH
LINK_COUNT = data_v2.EXPECTED_DIRECTED_LINKS
SPLITS = ("train", "validation")
OUTPUT_ROOT_NAME = "geant_horizon96_only_v1"
OUTPUT_DIRECTORY = "data/processed/geant_horizon96_only_v1"
PROTOCOL_DOCUMENT = "docs/geant_horizon96_only_candidate_v1.md"
SEASONAL_FEATURES = (
    "u[t,e] - S[t,e] (same-time-yesterday source for target t+96)",
    "u[t-576,e] - S[t,e] (same-time-last-week source for target t+96)",
)

# This is a new experiment configuration.  The historical two-horizon config
# remains untouched and is not imported as an active training configuration.
H96_CONFIG: dict[str, Any] = {
    "protocol_id": PROTOCOL_ID,
    "history_length": 672,
    "horizons": [96],
    "prediction_channels": 1,
    "step_seconds": data_v2.STEP_SECONDS,
    "split_bounds": {split: list(data_v2.OFFICIAL_SPLIT_BOUNDS[split]) for split in SPLITS},
    "directed_link_count": LINK_COUNT,
    "ewma_alpha": 0.8,
    "ewma_steps": 5,
    "residual_loss": "SmoothL1(beta=0.001) in original utilization-ratio units",
    "epochs": 20,
    "patience": 4,
    "learning_rate": 0.005,
    "batch_size": 32,
    "optimizer": "Adam",
    "hidden_size": 16,
    "seeds": list(SEEDS),
    "adjacency_modes": ["topology", "identity", "shuffled"],
    "shuffle_seed": SHUFFLE_SEED,
    "scaler_fit_scope": "h=96 training histories and targets only",
    "checkpoint_selection": "minimum h=96 validation residual SmoothL1 only",
    "validation_gradient_updates": False,
}

SOURCE_FILES = (
    "phase9_geant/__init__.py",
    "phase9_geant/horizon_v2_experiment.py",
    "phase9_geant/horizon_v2_data.py",
    "phase9_geant/horizon_windows.py",
    "phase9_geant/safe_paths.py",
    "phase9_geant/validity_v2.py",
    "phase9_geant/build_link_series.py",
    "phase9_geant/parse_geant.py",
    "src/gnn/__init__.py",
    "src/gnn/dataset.py",
    "src/gnn/preprocessing.py",
    "src/gnn/model.py",
    "src/gnn/spatiotemporal_link.py",
    "phase9_geant/horizon96_experiment.py",
    PROTOCOL_DOCUMENT,
)


class H96IntegrityError(ValueError):
    """A fail-closed data, path, or provenance invariant failed."""


def _canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _array_payload_sha256(array: np.ndarray, chunk_elements: int = 1_000_000) -> str:
    """Hash the same C-order numeric payload as ndarray.tobytes without a full copy."""
    digest = hashlib.sha256()
    flat = np.asarray(array).reshape(-1)
    for start in range(0, flat.size, chunk_elements):
        chunk = np.ascontiguousarray(flat[start : start + chunk_elements])
        digest.update(memoryview(chunk).cast("B"))
    return digest.hexdigest()


def _validate_finite_in_chunks(array: np.ndarray, *, name: str, chunk_elements: int = 1_000_000) -> None:
    flat = np.asarray(array).reshape(-1)
    for start in range(0, flat.size, chunk_elements):
        if not np.isfinite(flat[start : start + chunk_elements]).all():
            raise H96IntegrityError(f"Nonfinite value in permitted h=96 array {name}")


def _expected_artifact_names() -> set[str]:
    return set(data_v2._EXPECTED_ARTIFACT_NAMES)


def _validate_dataset_metadata(root: Path, repository: Path) -> tuple[dict[str, Any], tuple[tuple[str, str], ...], str]:
    """Validate train/validation dataset identity and the local topology bytes."""
    if root.name[: len(data_v2.OUTPUT_PREFIX)] != data_v2.OUTPUT_PREFIX:
        raise H96IntegrityError("Data root is not an allow-listed versioned horizon-v2 root")
    if root.parent != repository / "data" / "processed":
        raise H96IntegrityError("Data root is not a direct child of canonical data/processed")
    names = {entry.name for entry in root.iterdir()}
    expected = _expected_artifact_names()
    if not expected <= names or names - expected - {"runs"}:
        raise H96IntegrityError("Horizon dataset artifact inventory is not exact")

    metadata_path = require_regular_file(root / "metadata.json", root)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("protocol_id") != data_v2.PROTOCOL_ID or metadata.get("status") != "complete":
        raise H96IntegrityError("Horizon-v2 dataset protocol/status mismatch")
    if metadata.get("source_scope") != {
        split: list(data_v2.OFFICIAL_SPLIT_BOUNDS[split]) for split in SPLITS
    }:
        raise H96IntegrityError("Dataset source scope is not exactly train plus validation")
    if metadata.get("source_hashes_at_build") != data_v2._current_build_source_hashes(repository):
        raise H96IntegrityError("Dataset preprocessing source identity drift")
    build_config = metadata.get("build_configuration")
    if (
        build_config != data_v2.BUILD_CONFIGURATION
        or _canonical_json_sha256(build_config) != metadata.get("build_configuration_sha256")
    ):
        raise H96IntegrityError("Dataset build configuration identity mismatch")
    if metadata.get("window") != {
        "history_length": HISTORY_LENGTH,
        "horizons": [4, 96],
        "step_seconds": data_v2.STEP_SECONDS,
    }:
        raise H96IntegrityError("Underlying horizon-v2 window contract mismatch")
    expected_artifact_list = sorted(expected)
    if sorted(metadata.get("artifact_names", [])) != expected_artifact_list:
        raise H96IntegrityError("Dataset metadata artifact allowlist mismatch")

    topology = metadata.get("topology", {})
    topology_nodes = topology.get("nodes", [])
    edge_order = tuple(tuple(edge) for edge in topology.get("directed_edges_interleaved", ()))
    if (
        len(topology_nodes) != data_v2.EXPECTED_NODES
        or len(set(topology_nodes)) != data_v2.EXPECTED_NODES
        or topology.get("undirected_links") != data_v2.EXPECTED_UNDIRECTED_LINKS
        or topology.get("directed_links") != LINK_COUNT
        or len(edge_order) != LINK_COUNT
        or len(set(edge_order)) != LINK_COUNT
        or data_v2._edge_digest(edge_order) != topology.get("ordered_edge_sha256")
    ):
        raise H96IntegrityError("Ordered directed-edge metadata is invalid")
    nodes, undirected, actual_edges, topology_hash = data_v2._topology_snapshot(
        repository / data_v2.TOPOLOGY_RELATIVE, repository
    )
    if (
        len(nodes) != data_v2.EXPECTED_NODES
        or len(undirected) != data_v2.EXPECTED_UNDIRECTED_LINKS
        or actual_edges != edge_order
        or topology_hash != topology.get("sha256_local_file_identity_only")
    ):
        raise H96IntegrityError("Local topology identity/order does not match dataset metadata")
    return metadata, edge_order, topology_hash


def _load_h96_array(root: Path, filename: str, expected_shape: Sequence[int]) -> np.ndarray:
    """Map exactly one allowed h=96 train/validation file read-only."""
    checked_name = data_v2.validate_artifact_name(filename)
    expected_names = {
        f"{kind}_{split}_h96.npy"
        for split in SPLITS
        for kind in ("history", "target")
    }
    if checked_name not in expected_names:
        raise H96IntegrityError(f"Only explicitly allow-listed h=96 arrays may be loaded: {filename}")
    path = require_regular_file(root / checked_name, root)
    try:
        value = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise H96IntegrityError(f"Could not safely load {checked_name}: {exc}") from exc
    if not isinstance(value, np.ndarray) or value.dtype.hasobject or value.dtype.kind not in "fiu":
        raise H96IntegrityError(f"{checked_name} must be a numeric non-object array")
    if tuple(value.shape) != tuple(expected_shape):
        raise H96IntegrityError(f"{checked_name} shape differs from metadata")
    _validate_finite_in_chunks(value, name=checked_name)
    return value


def load_h96_train_validation(
    data_root: str | Path,
    *,
    repository_root: str | Path,
) -> tuple[dict[str, dict[str, Any]], tuple[tuple[str, str], ...], dict[str, Any]]:
    """Load only the four h=96 train/validation arrays, memory-mapped read-only."""
    repository = canonical_repository_root(repository_root)
    root = data_v2._canonical_horizon_root(data_root, repository)
    metadata, edge_order, topology_hash = _validate_dataset_metadata(root, repository)
    result: dict[str, dict[str, Any]] = {}
    fingerprints: dict[str, Any] = {}
    for split in SPLITS:
        bounds = data_v2.OFFICIAL_SPLIT_BOUNDS[split]
        record = metadata.get("splits", {}).get(split, {})
        if record.get("bounds") != list(bounds):
            raise H96IntegrityError(f"{split} bounds differ from the official split")
        expected_source_names = [
            f"demandMatrix-geant-uhlig-15min-{(data_v2.EXPECTED_START + i * data_v2.STEP):%Y%m%d-%H%M}.txt"
            for i in range(*bounds)
        ]
        if record.get("source_file_names") != expected_source_names:
            raise H96IntegrityError(f"{split} source filename provenance mismatch")
        semantic_sha = record.get("parsed_source_semantic_sha256")
        if not isinstance(semantic_sha, str) or len(semantic_sha) != 64:
            raise H96IntegrityError(f"{split} semantic source fingerprint is malformed")

        h96_record = record.get("horizons", {}).get("96")
        if not isinstance(h96_record, dict):
            raise H96IntegrityError(f"Missing {split} h=96 shape/fingerprint metadata")
        history_shape = tuple(h96_record.get("history_shape", ()))
        target_shape = tuple(h96_record.get("target_shape", ()))
        if (
            len(history_shape) != 3
            or history_shape[1:] != (HISTORY_LENGTH, LINK_COUNT)
            or target_shape != (history_shape[0], LINK_COUNT)
            or history_shape[0] <= 0
        ):
            raise H96IntegrityError(f"{split} h=96 array dimensions are invalid")
        history_name = f"history_{split}_h96.npy"
        target_name = f"target_{split}_h96.npy"
        history = _load_h96_array(root, history_name, history_shape)
        target = _load_h96_array(root, target_name, target_shape)
        if str(history.dtype) != h96_record.get("dtype") or str(target.dtype) != h96_record.get("dtype"):
            raise H96IntegrityError(
                f"{split} h=96 array dtype differs from metadata: "
                f"history={history.dtype}, target={target.dtype}, expected={h96_record.get('dtype')}"
            )
        if _array_payload_sha256(history) != h96_record.get("history_sha256"):
            raise H96IntegrityError(f"{split} h=96 history payload fingerprint mismatch")
        if _array_payload_sha256(target) != h96_record.get("target_sha256"):
            raise H96IntegrityError(f"{split} h=96 target payload fingerprint mismatch")

        index_name = data_v2.validate_artifact_name(f"window_index_{split}.json")
        index_path = require_regular_file(root / index_name, root)
        index_bytes = index_path.read_bytes()
        index_document = json.loads(index_bytes.decode("utf-8"))
        rows = index_document.get("96")
        if not isinstance(rows, list) or len(rows) != len(history):
            raise H96IntegrityError(f"{split} h=96 window index does not align with arrays")
        lower, upper = bounds
        previous_origin = lower - 1
        for row in rows:
            try:
                history_start = int(row["history_start"])
                origin = int(row["origin"])
                target_index = int(row["target"])
            except (KeyError, TypeError, ValueError) as exc:
                raise H96IntegrityError(f"Malformed {split} h=96 window index row") from exc
            if not (
                history_start == origin - HISTORY_LENGTH + 1
                and target_index == origin + HORIZON
                and lower <= history_start <= origin < target_index < upper
                and origin > previous_origin
            ):
                raise H96IntegrityError(f"{split} h=96 window crosses an invalid split/alignment boundary")
            previous_origin = origin
        result[split] = {
            "history": history,
            "target": target,
            "windows": rows,
            "bounds": list(bounds),
            "window_index_sha256": hashlib.sha256(index_bytes).hexdigest(),
        }
        fingerprints[split] = {
            "history_payload_sha256": h96_record["history_sha256"],
            "target_payload_sha256": h96_record["target_sha256"],
            "history_shape": list(history.shape),
            "target_shape": list(target.shape),
            "dtype_history": str(history.dtype),
            "dtype_target": str(target.dtype),
            "window_index_file_sha256": result[split]["window_index_sha256"],
        }
    provenance = {
        "dataset_protocol_id": data_v2.PROTOCOL_ID,
        "experiment_protocol_id": PROTOCOL_ID,
        "data_root": str(root),
        "source_scope": {split: list(data_v2.OFFICIAL_SPLIT_BOUNDS[split]) for split in SPLITS},
        "array_fingerprints": fingerprints,
        "ordered_edge_sha256": data_v2._edge_digest(edge_order),
        "topology_sha256_local_identity_only": topology_hash,
        "upstream_topology_provenance": metadata.get("topology", {}).get("upstream_provenance", "unverified"),
        "training_windows_h96": len(result["train"]["windows"]),
        "validation_windows_h96": len(result["validation"]["windows"]),
        "loaded_array_names": [
            "history_train_h96.npy", "target_train_h96.npy",
            "history_validation_h96.npy", "target_validation_h96.npy",
        ],
        "test_array_access": "no test path, name, loader, or metric branch is present in this experiment reader",
    }
    return result, edge_order, provenance


def _source_identity(repository: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in SOURCE_FILES:
        path = checked_absolute(repository / relative)
        if not path.is_file() or path.resolve(strict=True) != path:
            raise H96IntegrityError(f"Required source/protocol file is unavailable or redirected: {relative}")
        result[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def _config_sha256(config: Mapping[str, Any] | None = None) -> str:
    return _canonical_json_sha256(dict(H96_CONFIG if config is None else config))


def _planned_output(repository: Path, seed: int, adjacency_mode: str) -> tuple[Path, Path]:
    if seed not in SEEDS:
        raise ValueError(f"Seed must be one of {SEEDS}")
    if adjacency_mode not in H96_CONFIG["adjacency_modes"]:
        raise ValueError("Adjacency mode is not in the horizon-96 protocol")
    processed = repository / "data" / "processed"
    if not processed.is_dir() or processed.resolve(strict=True) != processed:
        raise H96IntegrityError(f"Canonical processed-data directory is unavailable: {processed}")
    output_root = checked_absolute(processed / OUTPUT_ROOT_NAME, allow_missing=True)
    if output_root.parent != processed:
        raise H96IntegrityError("Horizon-96 output root escaped data/processed")
    if output_root.exists():
        if not output_root.is_dir() or output_root.resolve(strict=True) != output_root:
            raise H96IntegrityError("Horizon-96 output root is redirected or not a directory")
    run = checked_absolute(output_root / f"h96_{adjacency_mode}_seed_{seed}", allow_missing=True)
    if run.parent != output_root:
        raise H96IntegrityError("Horizon-96 run path is not a direct output-root child")
    if run.exists() or run.is_symlink():
        raise FileExistsError(f"Refusing existing horizon-96 run destination: {run}")
    return output_root, run


def _verify_source_snapshot(repository: Path, expected: Mapping[str, str]) -> None:
    if _source_identity(repository) != dict(expected):
        raise H96IntegrityError("Source/protocol bytes changed after preflight")


def _seed_h96(seed: int) -> None:
    if seed not in SEEDS:
        raise ValueError(f"Seed must be one of {SEEDS}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _atomic_json(path: Path, value: object) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n"
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _atomic_checkpoint(payload: object, destination: Path) -> str:
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    os.close(fd)
    try:
        torch.save(payload, temporary)
        with open(temporary, "r+b") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    raw = destination.read_bytes()
    if not raw:
        raise RuntimeError("Checkpoint finalization produced an empty file")
    return hashlib.sha256(raw).hexdigest()


def objective_h96(
    prediction: torch.Tensor,
    target: torch.Tensor,
    baseline: torch.Tensor,
) -> torch.Tensor:
    """The sole objective: h=96 residual SmoothL1 in utilization-ratio units."""
    if prediction.shape != target.shape or target.shape != baseline.shape or prediction.ndim != 2:
        raise ValueError("h=96 objective tensors must align as [batch, directed_link]")
    if not (torch.isfinite(prediction).all() and torch.isfinite(target).all() and torch.isfinite(baseline).all()):
        raise ValueError("h=96 objective contains nonfinite values")
    return F.smooth_l1_loss(prediction - baseline, target - baseline, beta=0.001)


def should_select_h96_checkpoint(candidate_validation_loss: float, best_validation_loss: float) -> bool:
    """Strict minimum of the sole h=96 validation objective selects a checkpoint."""
    return validation_loss_improved(candidate_validation_loss, best_validation_loss)


def fit_h96_training_scaler(train_history: np.ndarray, train_target: np.ndarray) -> Standardizer:
    """Fit only from h=96 training arrays; validation is not an argument."""
    _validate_array_pair(train_history, train_target, "training")
    fitted = fit_training_scaler({HORIZON: train_history}, {HORIZON: train_target})
    return Standardizer(fitted.mean, fitted.scale, "h=96 training histories and targets only")


def _load_checkpoint_verified(
    model: SpatioTemporalLinkForecaster,
    checkpoint: Path,
    expected_sha256: str,
    *,
    device: torch.device,
    expected_seed: int,
) -> None:
    raw = checkpoint.read_bytes()
    actual_sha = hashlib.sha256(raw).hexdigest()
    if not raw or actual_sha != expected_sha256:
        raise H96IntegrityError("Checkpoint is empty or its SHA-256 identity changed")
    payload = torch.load(io.BytesIO(raw), map_location=device, weights_only=True)
    if payload.get("protocol_id") != PROTOCOL_ID or payload.get("seed") != expected_seed:
        raise H96IntegrityError("Checkpoint protocol or seed identity mismatch")
    model.load_state_dict(payload["state_dict"], strict=True)


def metric_summary_h96(target: np.ndarray, prediction: np.ndarray) -> dict[str, float | int | str]:
    """Aggregate metrics for the one permitted horizon and exact finite pairs."""
    return metric_summary(target, prediction)


def _batch_tensor(array: np.ndarray, indices: Sequence[int], scaler: Standardizer, device: torch.device) -> torch.Tensor:
    batch = np.asarray(array[np.asarray(indices, dtype=np.int64)], dtype=np.float32)
    normalized = (batch - np.float32(scaler.mean)) / np.float32(scaler.scale)
    return torch.as_tensor(normalized, dtype=torch.float32, device=device)


def _validation_pass(
    model: SpatioTemporalLinkForecaster,
    history: np.ndarray,
    target: np.ndarray,
    scaler: Standardizer,
    batch_size: int,
    device: torch.device,
) -> tuple[float, dict[str, Any]]:
    model.eval()
    count = 0
    loss_sum = 0.0
    absolute_sum = 0.0
    square_sum = 0.0
    baseline_abs_sum = 0.0
    baseline_square_sum = 0.0
    with torch.no_grad():
        for start in range(0, len(history), batch_size):
            stop = min(start + batch_size, len(history))
            indices = range(start, stop)
            hx = _batch_tensor(history, indices, scaler, device)
            pred_norm = model(hx)
            if pred_norm.shape != (stop - start, 1, LINK_COUNT):
                raise RuntimeError("Horizon-96 model returned an unexpected output shape")
            pred = pred_norm[:, 0, :].to(torch.float64) * scaler.scale + scaler.mean
            truth = torch.as_tensor(np.asarray(target[start:stop], dtype=np.float64), device=device)
            baseline = model._ewma(hx).to(torch.float64) * scaler.scale + scaler.mean
            loss = objective_h96(pred, truth, baseline)
            n = int(truth.numel())
            error = pred - truth
            baseline_error = baseline - truth
            loss_sum += float(loss.cpu()) * n
            absolute_sum += float(error.abs().sum().cpu())
            square_sum += float(torch.square(error).sum().cpu())
            baseline_abs_sum += float(baseline_error.abs().sum().cpu())
            baseline_square_sum += float(torch.square(baseline_error).sum().cpu())
            count += n
    if count == 0:
        raise ValueError("Validation has no h=96 target-link pairs")
    candidate_metrics = {
        "evaluated_target_link_pairs": count,
        "coverage": 1.0,
        "mae": absolute_sum / count,
        "rmse": math.sqrt(square_sum / count),
        "units": "computed utilization ratio",
    }
    ewma_metrics = {
        "evaluated_target_link_pairs": count,
        "coverage": 1.0,
        "mae": baseline_abs_sum / count,
        "rmse": math.sqrt(baseline_square_sum / count),
        "units": "computed utilization ratio",
    }
    return loss_sum / count, {"h96_candidate": candidate_metrics, "ewma": ewma_metrics}


def _validate_array_pair(history: np.ndarray, target: np.ndarray, label: str) -> None:
    if history.ndim != 3 or history.shape[1:] != (HISTORY_LENGTH, LINK_COUNT):
        raise ValueError(f"{label} h=96 history must be [N,672,72]")
    if target.shape != (len(history), LINK_COUNT) or len(history) == 0:
        raise ValueError(f"{label} h=96 target must be [N,72] and align with history")
    if history.dtype.kind not in "fiu" or target.dtype.kind not in "fiu":
        raise ValueError(f"{label} h=96 arrays must be numeric")
    _validate_finite_in_chunks(history, name=f"{label} history")
    _validate_finite_in_chunks(target, name=f"{label} target")


def _train_h96_arrays(
    *,
    train_history: np.ndarray,
    train_target: np.ndarray,
    validation_history: np.ndarray,
    validation_target: np.ndarray,
    edge_order: Sequence[tuple[str, str]],
    scaler: Standardizer,
    output_directory: Path,
    seed: int,
    adjacency_mode: str,
    source_hashes: Mapping[str, str],
    config_hash: str,
    provenance: Mapping[str, Any],
    repository: Path,
    explicit_authorization: bool,
    device_name: str = "cpu",
) -> dict[str, Any]:
    """Run one single-output h=96 candidate; h=4 is absent from this API."""
    if explicit_authorization is not True:
        raise PermissionError("Low-level h=96 training also requires explicit authorization")
    if seed not in SEEDS or adjacency_mode not in H96_CONFIG["adjacency_modes"]:
        raise ValueError("seed or adjacency mode is not part of the h=96 protocol")
    if len(edge_order) != LINK_COUNT or len(set(edge_order)) != LINK_COUNT:
        raise ValueError("ordered edge contract must contain exactly 72 unique directed links")
    _validate_array_pair(train_history, train_target, "training")
    _validate_array_pair(validation_history, validation_target, "validation")
    if scaler.fit_scope != "h=96 training histories and targets only":
        raise ValueError("h=96 scaler provenance is invalid")
    if not math.isfinite(scaler.mean) or not math.isfinite(scaler.scale) or scaler.scale <= 0:
        raise ValueError("h=96 scaler parameters are invalid")
    if output_directory.exists():
        raise FileExistsError(f"Refusing existing run directory: {output_directory}")
    if not output_directory.parent.is_dir() or output_directory.parent.resolve(strict=True) != output_directory.parent:
        raise H96IntegrityError("Horizon-96 output root is not canonical")
    output_directory.mkdir(parents=False, exist_ok=False)
    if (
        checked_absolute(output_directory) != output_directory
        or output_directory.resolve(strict=True) != output_directory
        or output_directory.parent.resolve(strict=True) != output_directory.parent
    ):
        # Do not delete a directory whose filesystem identity cannot be proven.
        raise H96IntegrityError("Created horizon-96 run directory failed containment validation")

    adjacency = build_link_adjacency(edge_order, mode=adjacency_mode, shuffle_seed=SHUFFLE_SEED)
    adjacency_hash = hashlib.sha256(adjacency.cpu().contiguous().numpy().tobytes(order="C")).hexdigest()
    config = {**H96_CONFIG, "seed": seed, "adjacency_mode": adjacency_mode}
    manifest: dict[str, Any] = {
        "status": "incomplete",
        "protocol_id": PROTOCOL_ID,
        "dataset_protocol_id": data_v2.PROTOCOL_ID,
        "mode": "horizon96-only-candidate",
        "seed": seed,
        "configuration": config,
        "configuration_sha256": config_hash,
        "source_hashes_at_preflight": dict(source_hashes),
        "provenance": dict(provenance),
        "normalization": asdict(scaler),
        "ordered_edge_sha256": data_v2._edge_digest(edge_order),
        "adjacency_identity": {
            "mode": adjacency_mode,
            "normalized_adjacency_sha256": adjacency_hash,
            "shuffle_seed": SHUFFLE_SEED if adjacency_mode == "shuffled" else None,
            "directed_link_count": LINK_COUNT,
            "parameterization": "SpatioTemporalLinkForecaster; identical for topology/identity/shuffled",
        },
        "objective_and_selection": "h=96 residual SmoothL1; h=96 validation loss only",
        "checkpoint": "best.pt",
    }
    _atomic_json(output_directory / "run.json", manifest)
    started = time.perf_counter()
    try:
        # The RNG state is fixed before the only model construction in this run.
        _seed_h96(seed)
        device = torch.device(device_name)
        model = SpatioTemporalLinkForecaster(
            edge_order,
            history_length=HISTORY_LENGTH,
            horizons=HORIZONS,
            hidden_size=int(H96_CONFIG["hidden_size"]),
            alpha=0.8,
            ewma_steps=5,
            include_seasonal_features=True,
            adjacency_mode=adjacency_mode,  # type: ignore[arg-type]
            shuffle_seed=SHUFFLE_SEED,
        ).to(device)
        if model.horizons != HORIZONS or model.residual_head.out_features != 1:
            raise RuntimeError("Constructed model is not a single-output h=96 model")
        manifest["model_parameter_count"] = sum(parameter.numel() for parameter in model.parameters())
        manifest["model_output_shape"] = "[batch, 1, 72] (sole channel is horizon 96)"
        _atomic_json(output_directory / "run.json", manifest)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(H96_CONFIG["learning_rate"]))
        history_log: list[dict[str, Any]] = []
        best_loss = float("inf")
        best_epoch = 0
        stale = 0
        checkpoint = output_directory / "best.pt"
        batch_size = int(H96_CONFIG["batch_size"])
        for epoch in range(1, int(H96_CONFIG["epochs"]) + 1):
            model.train()
            order = np.random.default_rng(seed + epoch).permutation(len(train_history))
            train_loss_sum = 0.0
            train_pair_count = 0
            for start in range(0, len(order), batch_size):
                selected = order[start : start + batch_size]
                hx = _batch_tensor(train_history, selected, scaler, device)
                target = torch.as_tensor(
                    np.asarray(train_target[selected], dtype=np.float64), device=device
                )
                output = model(hx)
                if output.shape != (len(selected), 1, LINK_COUNT):
                    raise RuntimeError("Training output is not [B,1,72]")
                prediction = output[:, 0, :].to(torch.float64) * scaler.scale + scaler.mean
                baseline = model._ewma(hx).to(torch.float64) * scaler.scale + scaler.mean
                loss = objective_h96(prediction, target, baseline)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                pairs = int(target.numel())
                train_loss_sum += float(loss.detach().cpu()) * pairs
                train_pair_count += pairs
            validation_loss, _ = _validation_pass(
                model, validation_history, validation_target, scaler, batch_size, device
            )
            train_loss = train_loss_sum / train_pair_count
            history_log.append({"epoch": epoch, "train_h96_smooth_l1": train_loss, "validation_h96_smooth_l1": validation_loss})
            if should_select_h96_checkpoint(validation_loss, best_loss):
                _verify_source_snapshot(repository, source_hashes)
                if _config_sha256(config) != config_hash:
                    raise H96IntegrityError("Configuration changed before best checkpoint publication")
                best_loss, best_epoch, stale = validation_loss, epoch, 0
                _atomic_checkpoint(
                    {"state_dict": model.state_dict(), "seed": seed, "best_epoch": epoch, "protocol_id": PROTOCOL_ID},
                    checkpoint,
                )
            else:
                stale += 1
            if stale >= int(H96_CONFIG["patience"]):
                break
        if best_epoch == 0 or not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
            raise RuntimeError("No nonempty best h=96 checkpoint was finalized")
        checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        _load_checkpoint_verified(
            model, checkpoint, checkpoint_sha, device=device, expected_seed=seed
        )
        validation_loss, validation_metrics = _validation_pass(
            model, validation_history, validation_target, scaler, batch_size, device
        )
        if validation_loss != best_loss:
            # Re-evaluation can differ only by numerical nondeterminism; record the
            # selected objective separately, but do not silently select another epoch.
            if not math.isclose(validation_loss, best_loss, rel_tol=1e-6, abs_tol=1e-12):
                raise RuntimeError("Strict checkpoint reload changed h=96 validation objective")
        _verify_source_snapshot(repository, source_hashes)
        manifest.update({
            "status": "complete",
            "best_epoch": best_epoch,
            "best_validation_h96_smooth_l1": best_loss,
            "epochs_completed": len(history_log),
            "history": history_log,
            "validation_metrics_h96_only": validation_metrics,
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_reload_strict": True,
            "early_stopping": {
                "patience": int(H96_CONFIG["patience"]),
                "stopped_early": len(history_log) < int(H96_CONFIG["epochs"]),
            },
            "duration_seconds": time.perf_counter() - started,
        })
        _atomic_json(output_directory / "run.json", manifest)
    except BaseException as exc:
        manifest.update({"status": "failed", "failure": f"{type(exc).__name__}: {exc}"})
        try:
            _atomic_json(output_directory / "run.json", manifest)
        except BaseException:
            pass
        raise
    return manifest


def train_h96_from_root(
    data_root: str | Path,
    *,
    repository_root: str | Path,
    seed: int,
    adjacency_mode: str,
    run_training: bool = False,
    explicit_authorization: bool = False,
    device: str = "cpu",
) -> dict[str, Any]:
    """The only candidate training entry; doubly opt-in and disabled by default."""
    if run_training is not True or explicit_authorization is not True:
        raise PermissionError("h=96 training requires both run_training and explicit_authorization")
    if seed not in SEEDS or adjacency_mode not in H96_CONFIG["adjacency_modes"]:
        raise ValueError("seed or adjacency mode is not part of the frozen h=96 protocol")
    repository = canonical_repository_root(repository_root)
    # Destination is refused before any array path or array loader is touched.
    output_root, output = _planned_output(repository, seed, adjacency_mode)
    root = data_v2._canonical_horizon_root(data_root, repository)
    source_hashes = _source_identity(repository)
    config = {**H96_CONFIG, "seed": seed, "adjacency_mode": adjacency_mode}
    config_hash = _config_sha256(config)
    splits, edge_order, provenance = load_h96_train_validation(root, repository_root=repository)
    train = splits["train"]
    validation = splits["validation"]
    # Only the h=96 arrays reach scaler fitting and the trainer API.
    scaler = fit_h96_training_scaler(train["history"], train["target"])
    # Validate control construction before creating an output directory.
    realized_adjacency = build_link_adjacency(
        edge_order, mode=adjacency_mode, shuffle_seed=SHUFFLE_SEED  # type: ignore[arg-type]
    )
    if realized_adjacency.shape != (LINK_COUNT, LINK_COUNT):
        raise H96IntegrityError("Adjacency does not preserve the 72-link ordering")
    # Recheck source and config identity before output creation.
    _verify_source_snapshot(repository, source_hashes)
    if _config_sha256(config) != config_hash:
        raise H96IntegrityError("Configuration identity changed before output creation")
    if not output_root.exists():
        output_root.mkdir(parents=False, exist_ok=False)
    if checked_absolute(output_root) != output_root or output_root.resolve(strict=True) != output_root:
        raise H96IntegrityError("Created horizon-96 output root is redirected")
    if output.exists():
        raise FileExistsError(f"Refusing existing horizon-96 run destination: {output}")
    return _train_h96_arrays(
        train_history=train["history"],
        train_target=train["target"],
        validation_history=validation["history"],
        validation_target=validation["target"],
        edge_order=edge_order,
        scaler=scaler,
        output_directory=output,
        seed=seed,
        adjacency_mode=adjacency_mode,
        source_hashes=source_hashes,
        config_hash=config_hash,
        provenance=provenance,
        repository=repository,
        explicit_authorization=True,
        device_name=device,
    )


def _synthetic_edges_72() -> tuple[tuple[str, str], ...]:
    undirected = {(i, (i + 1) % 22) for i in range(22)}
    undirected.update((i, i + 5) for i in range(14))
    ordered = sorted((min(a, b), max(a, b)) for a, b in undirected)
    if len(ordered) != 36:
        raise RuntimeError("Synthetic benchmark topology construction failed")
    return tuple(
        edge
        for left, right in ordered
        for edge in ((f"n{left}", f"n{right}"), (f"n{right}", f"n{left}"))
    )


def _process_memory_bytes() -> dict[str, int | None]:
    try:
        import psutil

        info = psutil.Process().memory_info()
        peak = getattr(info, "peak_wset", None)
        if peak is None:
            peak = getattr(info, "peak_rss", None)
        return {"rss_bytes": int(info.rss), "peak_rss_or_working_set_bytes": int(peak) if peak is not None else None}
    except Exception:
        return {"rss_bytes": None, "peak_rss_or_working_set_bytes": None}


def run_synthetic_cpu_benchmark(*, batch_size: int = 4, measured_steps: int = 3) -> dict[str, Any]:
    """Bounded synthetic optimizer-step benchmark; never opens project data."""
    if batch_size <= 0 or measured_steps <= 0:
        raise ValueError("benchmark bounds must be positive")
    torch.set_num_threads(1)
    torch.manual_seed(20261019)
    edges = _synthetic_edges_72()
    model = SpatioTemporalLinkForecaster(
        edges,
        history_length=HISTORY_LENGTH,
        horizons=HORIZONS,
        hidden_size=int(H96_CONFIG["hidden_size"]),
        alpha=0.8,
        ewma_steps=5,
        adjacency_mode="topology",
        shuffle_seed=SHUFFLE_SEED,
    ).cpu()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(H96_CONFIG["learning_rate"]))
    rng = torch.Generator(device="cpu").manual_seed(20261019)
    history = torch.rand((batch_size, HISTORY_LENGTH, LINK_COUNT), generator=rng)
    target = torch.rand((batch_size, LINK_COUNT), generator=rng)
    before_memory = _process_memory_bytes()
    durations: list[float] = []
    for step in range(measured_steps + 1):
        started = time.perf_counter()
        model.train()
        output = model(history)
        baseline = model._ewma(history)
        loss = objective_h96(output[:, 0, :], target, baseline)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        elapsed = time.perf_counter() - started
        if step > 0:
            durations.append(elapsed)
    after_memory = _process_memory_bytes()
    train_windows = 3924  # prior fixed train-only h=96 Ridge screen accepted origins
    full_batches = math.ceil(train_windows / int(H96_CONFIG["batch_size"]))
    # Explicit linear-throughput extrapolation from batch=4 to configured batch=32.
    seconds_per_configured_batch = statistics.median(durations) * (
        int(H96_CONFIG["batch_size"]) / batch_size
    )
    return {
        "scope": "synthetic tensors only; no GEANT data, arrays, manifests, or checkpoints",
        "device": "cpu",
        "shape": [batch_size, HISTORY_LENGTH, LINK_COUNT],
        "target_shape": [batch_size, LINK_COUNT],
        "model_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "warmup_optimizer_steps": 1,
        "measured_optimizer_steps": measured_steps,
        "step_seconds": durations,
        "median_step_seconds_at_benchmark_batch": statistics.median(durations),
        "peak_memory": after_memory,
        "memory_before_model_step": before_memory,
        "peak_memory_measurement_limit": "psutil platform peak field when exposed; otherwise current RSS only",
        "epoch_time_estimate": {
            "training_windows_assumed_from_prior_train_only_screen": train_windows,
            "configured_batch_size": int(H96_CONFIG["batch_size"]),
            "optimizer_batches_per_epoch_estimate": full_batches,
            "assumption": f"linear step-time scaling from synthetic batch {batch_size} to configured batch 32; excludes validation, data scans, and checkpoint I/O",
            "seconds_per_epoch_training_batches_estimate": full_batches * seconds_per_configured_batch,
            "seconds_for_20_training_epochs_estimate": full_batches * seconds_per_configured_batch * int(H96_CONFIG["epochs"]),
            "is_observed_training_duration": False,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--benchmark-synthetic", action="store_true", help="Run only a bounded synthetic CPU benchmark")
    action.add_argument("--run-training", action="store_true", help="Select the disabled-by-default h=96 training action")
    parser.add_argument("--authorize-horizon96-candidate", action="store_true", help="Explicit training permission; required with --run-training")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--repository-root", type=Path)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--adjacency-mode", choices=H96_CONFIG["adjacency_modes"])
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.benchmark_synthetic:
        if args.authorize_horizon96_candidate or args.data_root or args.repository_root or args.seed or args.adjacency_mode:
            parser.error("synthetic benchmark accepts no data, seed, or training authorization options")
        print(json.dumps(run_synthetic_cpu_benchmark(), indent=2, sort_keys=True, allow_nan=False))
        return 0
    missing = [
        name for name, value in (
            ("--authorize-horizon96-candidate", args.authorize_horizon96_candidate),
            ("--data-root", args.data_root),
            ("--repository-root", args.repository_root),
            ("--seed", args.seed),
            ("--adjacency-mode", args.adjacency_mode),
        ) if not value
    ]
    if missing:
        parser.error("--run-training requires " + ", ".join(missing))
    result = train_h96_from_root(
        args.data_root,
        repository_root=args.repository_root,
        seed=args.seed,
        adjacency_mode=args.adjacency_mode,
        run_training=True,
        explicit_authorization=args.authorize_horizon96_candidate,
        device=args.device,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover - the CLI is not invoked by import
    raise SystemExit(main())


__all__ = [
    "H96_CONFIG",
    "H96IntegrityError",
    "HORIZON",
    "HORIZONS",
    "HISTORY_LENGTH",
    "PROTOCOL_ID",
    "_array_payload_sha256",
    "_load_checkpoint_verified",
    "_load_h96_array",
    "_planned_output",
    "_source_identity",
    "_seed_h96",
    "load_h96_train_validation",
    "main",
    "metric_summary_h96",
    "objective_h96",
    "fit_h96_training_scaler",
    "run_synthetic_cpu_benchmark",
    "should_select_h96_checkpoint",
    "train_h96_from_root",
]
