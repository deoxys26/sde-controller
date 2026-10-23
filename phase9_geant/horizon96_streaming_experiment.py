"""Full h=96 candidate runs on the compact train/validation-only stream build.

This runner deliberately does not use the older materialized horizon-array
loader in ``horizon96_experiment``.  Histories are extracted per minibatch
from the compact utilization series and validated window-index records.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from phase9_geant import horizon96_experiment as candidate
from phase9_geant import horizon96_smoke as lazy
from phase9_geant import horizon96_streaming_data as builder
from phase9_geant.safe_paths import canonical_repository_root, checked_absolute
from src.gnn.spatiotemporal_link import SpatioTemporalLinkForecaster, build_link_adjacency


PROTOCOL_ID = "geant-horizon96-only-candidate-v1"
RUNNER_ID = "geant-horizon96-streaming-experiment-v1"
OUTPUT_ROOT_NAME = "geant_horizon96_streaming_experiment_v1"
HISTORY_LENGTH = 672
HORIZON = 96
LINK_COUNT = 72
SEEDS = (42, 123, 2026)
MODES = ("topology", "identity", "shuffled")
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20261019
BOOTSTRAP_BLOCK_LENGTHS = (96, 192)

CONFIG: dict[str, Any] = {
    **candidate.H96_CONFIG,
    "protocol_id": PROTOCOL_ID,
    "runner_id": RUNNER_ID,
    "history_length": HISTORY_LENGTH,
    "horizons": [HORIZON],
    "prediction_channels": 1,
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
    "adjacency_modes": list(MODES),
    "shuffle_seed": 20261019,
    "scaler_fit_scope": "h=96 training histories and targets only; window-occurrence weighted",
    "checkpoint_selection": "minimum h=96 validation residual SmoothL1 only",
    "validation_gradient_updates": False,
    "dataset_protocol_id": builder.PROTOCOL_ID,
    "train_bounds": [0, 8020],
    "validation_bounds": [8020, 9740],
}

RUN_SOURCE_FILES = tuple(sorted(set(
    builder.SOURCE_FILES
    + lazy.SOURCE_FILES
    + candidate.SOURCE_FILES
    + ("phase9_geant/horizon96_streaming_experiment.py",)
)))


class StreamingExperimentError(ValueError):
    """Fail-closed streaming experiment provenance or integrity error."""


def _json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _source_identity(repository: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative in RUN_SOURCE_FILES:
        path = checked_absolute(repository / relative)
        if not path.is_file() or path.resolve(strict=True) != path:
            raise StreamingExperimentError(f"Missing or redirected experiment source: {relative}")
        hashes[relative] = _file_sha256(path)
    return hashes


def _check_data_source_identity(repository: Path, metadata: Mapping[str, Any]) -> None:
    expected = metadata.get("source_hashes_before_and_after")
    current = builder._source_hashes(repository)
    if not isinstance(expected, dict) or current != expected:
        raise StreamingExperimentError("Streaming dataset builder/source hashes differ from current source")


def _verified_inputs(data_root: str | Path, repository_root: str | Path):
    repository = canonical_repository_root(repository_root)
    root = checked_absolute(data_root)
    expected_root = repository / "data" / "processed" / builder.OUTPUT_NAME
    if os.path.normcase(os.fspath(root)) != os.path.normcase(os.fspath(expected_root)):
        raise StreamingExperimentError(f"Only the verified streaming train/validation root is permitted: {expected_root}")
    # This check reads only directory names and metadata, not any array.
    verified_root, metadata = lazy._verify_data_root(root, repository)
    _check_data_source_identity(repository, metadata)
    nodes, undirected, edges, topology_hash = builder.approved._topology_snapshot(
        repository / builder.TOPOLOGY_RELATIVE, repository
    )
    topology = metadata["topology"]
    recorded_edges = tuple(tuple(edge) for edge in topology.get("directed_edges_interleaved", ()))
    if (
        topology_hash != builder.EXPECTED_LOCAL_TOPOLOGY_SHA256
        or topology_hash != topology.get("sha256_local_file_identity_only")
        or len(nodes) != 22 or len(undirected) != 36 or len(edges) != LINK_COUNT
        or tuple(edges) != recorded_edges
        or builder.approved._edge_digest(edges) != topology.get("ordered_edge_sha256")
    ):
        raise StreamingExperimentError("Local topology bytes, structure, or ordered edges mismatch")
    source_hashes = _source_identity(repository)
    metadata_path = verified_root / "metadata.json"
    metadata_bytes_hash = _file_sha256(metadata_path)
    # The audited lazy reader opens only explicit train/validation series,
    # masks, timestamps, and train/validation h=96 window indexes.
    splits, loaded_metadata = lazy.load_train_validation_lazy(verified_root, repository)
    if _file_sha256(metadata_path) != metadata_bytes_hash or loaded_metadata != metadata:
        raise StreamingExperimentError("Dataset metadata changed during train/validation loading")
    _check_data_source_identity(repository, loaded_metadata)
    if _source_identity(repository) != source_hashes:
        raise StreamingExperimentError("Experiment source identity changed during data preflight")
    train, validation = splits["train"], splits["validation"]
    if (
        train["series"].shape != (8020, LINK_COUNT)
        or validation["series"].shape != (1720, LINK_COUNT)
        or len(train["windows"]) != 3924
        or len(validation["windows"]) != 953
    ):
        raise StreamingExperimentError("Streaming series/window dimensions differ from the authorized contract")
    provenance = {
        "dataset_protocol_id": builder.PROTOCOL_ID,
        "data_root": str(verified_root),
        "source_scope": {"train": [0, 8020], "validation": [8020, 9740]},
        "source_file_counts": metadata["source_file_counts"],
        "array_file_records": metadata["array_files"],
        "window_index_sha256": metadata["window_index_sha256"],
        "training_cutoff": metadata["training_cutoff"],
        "training_windows_h96": len(train["windows"]),
        "validation_windows_h96": len(validation["windows"]),
        "ordered_edge_sha256": topology["ordered_edge_sha256"],
        "topology_sha256_local_identity_only": topology_hash,
        "test_access": "no test path, array, window index, checkpoint, or metric is read by this runner",
        "builder_test_period_filename_visibility": metadata["test_period_filename_visibility"],
    }
    return repository, verified_root, splits, metadata, tuple(edges), provenance, source_hashes


def _planned_run(repository: Path, seed: int, mode: str) -> tuple[Path, Path]:
    if seed not in SEEDS or mode not in MODES:
        raise ValueError("seed or graph mode is outside the fixed candidate protocol")
    processed = repository / "data" / "processed"
    if not processed.is_dir() or processed.resolve(strict=True) != processed:
        raise StreamingExperimentError(f"Canonical processed-data root is unavailable: {processed}")
    output_root = checked_absolute(processed / OUTPUT_ROOT_NAME, allow_missing=True)
    if output_root.parent != processed:
        raise StreamingExperimentError("Experiment output root escaped data/processed")
    if output_root.exists() and (not output_root.is_dir() or output_root.resolve(strict=True) != output_root):
        raise StreamingExperimentError("Experiment output root is redirected or not a directory")
    run = checked_absolute(output_root / f"h96_{mode}_seed_{seed}", allow_missing=True)
    if run.parent != output_root or run.exists() or run.is_symlink():
        raise FileExistsError(f"Refusing invalid or existing run destination: {run}")
    allowed_root_entries = {f"h96_{item}_{value}" for item in MODES for value in SEEDS}
    allowed_root_entries |= {"results.json", "report.md", "validation_per_link.csv", "validation_per_origin.csv"}
    if output_root.exists() and ({entry.name for entry in output_root.iterdir()} - allowed_root_entries):
        raise StreamingExperimentError("Unexpected entry in experiment output namespace")
    return output_root, run


def _atomic_json(path: Path, value: object) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _atomic_npy(path: Path, array: np.ndarray) -> str:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    try:
        with open(temporary, "wb") as stream:
            np.save(stream, array, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    return _file_sha256(path)


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _metric_values(target: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    truth = np.asarray(target, dtype=np.float64)
    pred = np.asarray(prediction, dtype=np.float64)
    if truth.shape != pred.shape or truth.ndim != 2 or not np.isfinite(truth).all() or not np.isfinite(pred).all():
        raise StreamingExperimentError("Metrics require aligned, fully finite [origin,72] pairs")
    error = pred - truth
    per_link_mae = np.mean(np.abs(error), axis=0, dtype=np.float64)
    per_link_rmse = np.sqrt(np.mean(np.square(error), axis=0, dtype=np.float64))
    per_origin_mae = np.mean(np.abs(error), axis=1, dtype=np.float64)
    per_origin_rmse = np.sqrt(np.mean(np.square(error), axis=1, dtype=np.float64))
    return {
        "mae": float(np.mean(np.abs(error), dtype=np.float64)),
        "rmse": float(np.sqrt(np.mean(np.square(error), dtype=np.float64))),
        "evaluated_pairs": int(truth.size),
        "units": "computed utilization ratio",
        "per_link_mae": per_link_mae.tolist(),
        "per_link_rmse": per_link_rmse.tolist(),
        "per_origin_mae": per_origin_mae.tolist(),
        "per_origin_rmse": per_origin_rmse.tolist(),
    }


def _baseline_predictions(history: np.ndarray) -> dict[str, np.ndarray]:
    if history.ndim != 3 or history.shape[1:] != (HISTORY_LENGTH, LINK_COUNT):
        raise StreamingExperimentError("Validation history must be [batch,672,72]")
    recent = history[:, -5:, :].astype(np.float64)
    raw_weights = np.asarray([0.8 * 0.2 ** (4 - k) for k in range(5)], dtype=np.float64)
    weights = raw_weights / raw_weights.sum()
    ewma = np.einsum("bte,t->be", recent, weights)
    persistence = history[:, -1, :].astype(np.float64)
    yesterday = history[:, -1, :].astype(np.float64)
    last_week = history[:, 95, :].astype(np.float64)
    if not np.array_equal(persistence, yesterday):
        raise AssertionError("At h=96, same-time-yesterday must equal the origin observation")
    return {
        "ewma_alpha_0.8": ewma,
        "persistence": persistence,
        "same_time_yesterday": yesterday,
        "same_time_last_week": last_week,
    }


def _evaluate(model, loader, scaler) -> tuple[float, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    baselines: dict[str, list[np.ndarray]] = {name: [] for name in (
        "ewma_alpha_0.8", "persistence", "same_time_yesterday", "same_time_last_week"
    )}
    objective_total = 0.0
    pair_total = 0
    with torch.no_grad():
        for raw_history, raw_target in loader:
            hx_raw = raw_history.detach().cpu().numpy().astype(np.float64, copy=False)
            y = raw_target.detach().cpu().numpy().astype(np.float64, copy=False)
            hx = (raw_history - np.float32(scaler.mean)) / np.float32(scaler.scale)
            output = model(hx)
            if output.shape != (len(raw_history), 1, LINK_COUNT):
                raise StreamingExperimentError("Model output must be [batch,1,72]")
            pred = output[:, 0, :].detach().cpu().numpy().astype(np.float64) * scaler.scale + scaler.mean
            base = _baseline_predictions(hx_raw)
            if not np.isfinite(pred).all() or not np.isfinite(y).all():
                raise StreamingExperimentError("Nonfinite validation prediction/target")
            b_tensor = (raw_history[:, -5:, :] - np.float32(scaler.mean)) / np.float32(scaler.scale)
            b_normalized = torch.einsum("bte,t->be", b_tensor, model.ewma_weights)
            b_original = b_normalized.detach().cpu().numpy().astype(np.float64) * scaler.scale + scaler.mean
            loss = F.smooth_l1_loss(
                torch.as_tensor(pred - b_original), torch.as_tensor(y - b_original), beta=0.001
            )
            n = int(y.size)
            objective_total += float(loss) * n
            pair_total += n
            predictions.append(pred)
            targets.append(y)
            for name, values in base.items():
                baselines[name].append(values)
    if pair_total == 0:
        raise StreamingExperimentError("Validation loader emitted no targets")
    prediction = np.concatenate(predictions, axis=0)
    target = np.concatenate(targets, axis=0)
    baseline_arrays = {name: np.concatenate(values, axis=0) for name, values in baselines.items()}
    return objective_total / pair_total, prediction, target, baseline_arrays


def _training_epoch(model, dataset, scaler, optimizer, seed: int, epoch: int) -> tuple[float, dict[str, float | int]]:
    model.train()
    order = np.random.default_rng(seed + epoch).permutation(len(dataset)).tolist()
    loader = DataLoader(
        Subset(dataset, order), batch_size=int(CONFIG["batch_size"]), shuffle=False,
        num_workers=0, pin_memory=False,
    )
    total = 0.0
    pairs_total = 0
    durations: list[float] = []
    for raw_history, raw_target in loader:
        started = time.perf_counter()
        hx = (raw_history - np.float32(scaler.mean)) / np.float32(scaler.scale)
        y = raw_target.to(torch.float64)
        output = model(hx)
        if output.shape != (len(raw_history), 1, LINK_COUNT):
            raise StreamingExperimentError("Training model output must be [batch,1,72]")
        pred = output[:, 0, :].to(torch.float64) * scaler.scale + scaler.mean
        base_normalized = torch.einsum("bte,t->be", hx[:, -5:, :], model.ewma_weights)
        baseline = base_normalized.to(torch.float64) * scaler.scale + scaler.mean
        loss = F.smooth_l1_loss(pred - baseline, y - baseline, beta=0.001)
        if not torch.isfinite(loss):
            raise StreamingExperimentError("Training loss is nonfinite")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
            raise StreamingExperimentError("Training gradients are missing or nonfinite")
        optimizer.step()
        count = int(y.numel())
        total += float(loss.detach()) * count
        pairs_total += count
        durations.append(time.perf_counter() - started)
    if pairs_total == 0:
        raise StreamingExperimentError("Training epoch emitted no target pairs")
    return total / pairs_total, {
        "optimizer_steps": len(durations),
        "median_optimizer_step_seconds": float(statistics.median(durations)),
        "p95_optimizer_step_seconds": float(np.percentile(durations, 95)),
        "max_optimizer_step_seconds": float(max(durations)),
    }


def _run_path(output_root: Path, seed: int, mode: str) -> Path:
    return output_root / f"h96_{mode}_seed_{seed}"


def train_one(
    data_root: str | Path,
    *,
    repository_root: str | Path,
    seed: int,
    mode: str,
    run_training: bool = False,
    explicit_authorization: bool = False,
    device_name: str = "cpu",
) -> dict[str, Any]:
    if run_training is not True or explicit_authorization is not True:
        raise PermissionError("Full candidate training requires --run-training and explicit authorization")
    if seed not in SEEDS or mode not in MODES or device_name != "cpu":
        raise ValueError("seed, graph mode, or device is outside the reviewed protocol")
    repository = canonical_repository_root(repository_root)
    output_root, output = _planned_run(repository, seed, mode)
    resource = lazy.resource_preflight(repository / "data" / "processed")
    repository, root, splits, metadata, edge_order, provenance, source_hashes = _verified_inputs(data_root, repository)
    scaler = lazy.fit_window_scaler(splits["train"]["series"], splits["train"]["windows"], split_start=0)
    if scaler.fit_scope != "h=96 train histories plus h=96 targets; window-occurrence weighted":
        raise StreamingExperimentError("Training-only streaming scaler scope mismatch")
    config = {**CONFIG, "seed": seed, "adjacency_mode": mode, "device": device_name}
    config_hash = _json_sha256(config)
    adjacency = build_link_adjacency(edge_order, mode=mode, shuffle_seed=20261019)
    adjacency_hash = hashlib.sha256(adjacency.contiguous().numpy().tobytes(order="C")).hexdigest()
    if output_root.exists() and not output_root.is_dir():
        raise StreamingExperimentError("Output namespace is not a directory")
    if not output_root.exists():
        output_root.mkdir(parents=False, exist_ok=False)
        if checked_absolute(output_root) != output_root or output_root.resolve(strict=True) != output_root:
            raise StreamingExperimentError("Created experiment root failed containment verification")
    output.mkdir(parents=False, exist_ok=False)
    if checked_absolute(output) != output or output.resolve(strict=True) != output or output.parent.resolve(strict=True) != output_root:
        raise StreamingExperimentError("Created run directory failed containment verification")
    manifest: dict[str, Any] = {
        "status": "incomplete", "runner_id": RUNNER_ID, "protocol_id": PROTOCOL_ID,
        "dataset_protocol_id": metadata["protocol_id"], "seed": seed, "mode": mode,
        "configuration": config, "configuration_sha256": config_hash,
        "repository_root": str(repository), "data_root": str(root), "provenance": provenance,
        "source_hashes_before_run": source_hashes, "normalization": scaler.__dict__,
        "adjacency": {"mode": mode, "shuffle_seed": 20261019 if mode == "shuffled" else None, "normalized_sha256": adjacency_hash},
        "environment": {"python": __import__("sys").version, "numpy": np.__version__, "torch": torch.__version__, "device": device_name},
        "resource_preflight": resource.__dict__, "validation_selection_objective": "minimum h=96 residual SmoothL1 only",
        "test_access": {"permitted": False, "paths_or_loaders": "none"},
    }
    _atomic_json(output / "run.json", manifest)
    started = time.perf_counter()
    try:
        _seed_all(seed)
        model = SpatioTemporalLinkForecaster(
            edge_order, history_length=HISTORY_LENGTH, horizons=(HORIZON,), hidden_size=int(CONFIG["hidden_size"]),
            alpha=0.8, ewma_steps=5, include_seasonal_features=True, adjacency_mode=mode,
            shuffle_seed=20261019,
        ).to(torch.device(device_name))
        optimizer = torch.optim.Adam(model.parameters(), lr=float(CONFIG["learning_rate"]))
        train_dataset = lazy.LazyH96Dataset(splits["train"]["series"], splits["train"]["windows"], split_start=0)
        validation_dataset = lazy.LazyH96Dataset(splits["validation"]["series"], splits["validation"]["windows"], split_start=8020)
        validation_loader = DataLoader(validation_dataset, batch_size=int(CONFIG["batch_size"]), shuffle=False, num_workers=0, pin_memory=False)
        manifest["model_parameter_count"] = sum(parameter.numel() for parameter in model.parameters())
        manifest["training_window_count"] = len(train_dataset)
        manifest["validation_window_count"] = len(validation_dataset)
        _atomic_json(output / "run.json", manifest)
        history: list[dict[str, Any]] = []
        best_loss = float("inf")
        best_epoch = 0
        stale = 0
        checkpoint = output / "best.pt"
        memory_before = candidate._process_memory_bytes()
        for epoch in range(1, int(CONFIG["epochs"]) + 1):
            epoch_started = time.perf_counter()
            train_loss, step_stats = _training_epoch(model, train_dataset, scaler, optimizer, seed, epoch)
            validation_loss, _, _, _ = _evaluate(model, validation_loader, scaler)
            row = {
                "epoch": epoch, "train_h96_residual_smooth_l1": train_loss,
                "validation_h96_residual_smooth_l1": validation_loss,
                "epoch_seconds": time.perf_counter() - epoch_started, "optimizer_step_stats": step_stats,
            }
            history.append(row)
            if not math.isfinite(validation_loss):
                raise StreamingExperimentError("Validation objective is nonfinite")
            if validation_loss < best_loss:
                if _source_identity(repository) != source_hashes or _json_sha256(config) != config_hash:
                    raise StreamingExperimentError("Source/configuration identity changed before checkpoint write")
                best_loss, best_epoch, stale = validation_loss, epoch, 0
                torch_payload = {"state_dict": model.state_dict(), "seed": seed, "best_epoch": epoch, "protocol_id": PROTOCOL_ID}
                digest = candidate._atomic_checkpoint(torch_payload, checkpoint)
                manifest["checkpoint_sha256_so_far"] = digest
            else:
                stale += 1
            manifest.update({"epoch_history": history, "best_epoch_so_far": best_epoch, "best_validation_loss_so_far": best_loss})
            _atomic_json(output / "run.json", manifest)
            if stale >= int(CONFIG["patience"]):
                break
        if best_epoch == 0 or not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
            raise StreamingExperimentError("No best validation checkpoint finalized")
        # Revalidate the complete train/validation data and source identity before checkpoint loading.
        _verified_inputs(data_root, repository)
        if _source_identity(repository) != source_hashes or _json_sha256(config) != config_hash:
            raise StreamingExperimentError("Source/configuration changed before checkpoint load")
        checkpoint_hash = _file_sha256(checkpoint)
        candidate._load_checkpoint_verified(model, checkpoint, checkpoint_hash, device=torch.device(device_name), expected_seed=seed)
        final_loss, predictions, targets, baselines = _evaluate(model, validation_loader, scaler)
        if not math.isclose(final_loss, best_loss, rel_tol=1e-6, abs_tol=1e-12):
            raise StreamingExperimentError("Strict checkpoint reload changed selected validation objective")
        validation_metrics = {"h96_candidate": _metric_values(targets, predictions)}
        validation_metrics.update({key: _metric_values(targets, value) for key, value in baselines.items()})
        pred_hash = _atomic_npy(output / "validation_predictions.npy", predictions.astype(np.float64, copy=False))
        manifest.update({
            "status": "complete", "best_epoch": best_epoch, "best_validation_h96_residual_smooth_l1": best_loss,
            "epochs_completed": len(history), "early_stopping": {"patience": CONFIG["patience"], "stopped_early": len(history) < CONFIG["epochs"]},
            "epoch_history": history, "validation_metrics": validation_metrics,
            "checkpoint": "best.pt", "checkpoint_sha256": checkpoint_hash, "checkpoint_reload_strict": True,
            "validation_predictions": "validation_predictions.npy", "validation_predictions_sha256": pred_hash,
            "runtime_seconds": time.perf_counter() - started,
            "process_memory_before_training": memory_before, "process_memory_after_training": candidate._process_memory_bytes(),
            "interpretation": "validation-only candidate experiment; not held-out generalization",
        })
        if _source_identity(repository) != source_hashes or _json_sha256(config) != config_hash:
            raise StreamingExperimentError("Source/configuration changed before run completion")
        if {entry.name for entry in output.iterdir()} != {"run.json", "best.pt", "validation_predictions.npy"}:
            raise StreamingExperimentError("Run artifact inventory is incomplete or unexpected")
        _atomic_json(output / "run.json", manifest)
    except BaseException as exc:
        manifest.update({"status": "failed", "failure": f"{type(exc).__name__}: {exc}", "epoch_history": manifest.get("epoch_history", history if 'history' in locals() else [])})
        try:
            _atomic_json(output / "run.json", manifest)
        except BaseException:
            pass
        raise
    return manifest


def _block_bootstrap_difference(origin_a: np.ndarray, origin_b: np.ndarray, *, metric: str, block_length: int, seed: int) -> dict[str, float | int]:
    if origin_a.shape != origin_b.shape or origin_a.ndim != 1 or len(origin_a) < block_length:
        raise ValueError("bootstrap origin losses must align and contain at least one full block")
    difference = np.asarray(origin_a, dtype=np.float64) - np.asarray(origin_b, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = len(difference)
    blocks = math.ceil(n / block_length)
    prefix = np.concatenate(([0.0], np.cumsum(difference, dtype=np.float64)))
    starts = rng.integers(0, n - block_length + 1, size=(BOOTSTRAP_REPLICATES, blocks))
    sampled = np.zeros(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for offset in range(blocks):
        length = min(block_length, n - offset * block_length)
        left = starts[:, offset]
        sampled += prefix[left + length] - prefix[left]
    values = sampled / n
    return {
        "point_estimate": float(np.mean(difference, dtype=np.float64)),
        "ci95_low": float(np.quantile(values, 0.025)), "ci95_high": float(np.quantile(values, 0.975)),
        "replicates": BOOTSTRAP_REPLICATES, "rng_seed": seed, "block_length_origins": block_length,
        "metric": metric,
    }


def _atomic_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True)
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _atomic_text(path: Path, text: str) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def summarize_validation(data_root: str | Path, *, repository_root: str | Path) -> dict[str, Any]:
    repository = canonical_repository_root(repository_root)
    output_root = checked_absolute(repository / "data" / "processed" / OUTPUT_ROOT_NAME)
    expected_runs = {f"h96_{mode}_seed_{seed}" for mode in MODES for seed in SEEDS}
    if {entry.name for entry in output_root.iterdir()} != expected_runs:
        raise FileNotFoundError("All nine complete run directories and no prior summary files are required")
    repository, root, splits, dataset_metadata, edge_order, provenance, source_hashes = _verified_inputs(data_root, repository)
    target_rows = [lazy.LazyH96Dataset(splits["validation"]["series"], splits["validation"]["windows"], split_start=8020)[i][1].numpy() for i in range(len(splits["validation"]["windows"]))]
    target = np.stack(target_rows).astype(np.float64)
    baseline_rows: dict[str, list[np.ndarray]] = {name: [] for name in ("ewma_alpha_0.8", "persistence", "same_time_yesterday", "same_time_last_week")}
    validation_dataset = lazy.LazyH96Dataset(splits["validation"]["series"], splits["validation"]["windows"], split_start=8020)
    for index in range(len(validation_dataset)):
        history, _ = validation_dataset[index]
        one = _baseline_predictions(history.numpy()[None, ...])
        for name, values in one.items():
            baseline_rows[name].append(values[0])
    baseline_predictions = {name: np.stack(rows) for name, rows in baseline_rows.items()}
    predictions: dict[tuple[str, int], np.ndarray] = {}
    metrics: dict[str, Any] = {name: _metric_values(target, pred) for name, pred in baseline_predictions.items()}
    for mode in MODES:
        for seed in SEEDS:
            run = output_root / f"h96_{mode}_seed_{seed}"
            manifest = json.loads((run / "run.json").read_text(encoding="utf-8"))
            if manifest.get("status") != "complete" or manifest.get("protocol_id") != PROTOCOL_ID or manifest.get("seed") != seed or manifest.get("mode") != mode:
                raise StreamingExperimentError(f"Run identity/lifecycle mismatch: {run.name}")
            if manifest.get("provenance", {}).get("array_file_records") != dataset_metadata.get("array_files"):
                raise StreamingExperimentError(f"Run dataset identity differs from current data: {run.name}")
            prediction_path = run / "validation_predictions.npy"
            if _file_sha256(prediction_path) != manifest.get("validation_predictions_sha256"):
                raise StreamingExperimentError(f"Validation prediction file hash mismatch: {run.name}")
            pred = np.load(prediction_path, allow_pickle=False, mmap_mode="r")
            if pred.shape != target.shape or pred.dtype != np.float64 or not np.isfinite(pred).all():
                raise StreamingExperimentError(f"Validation prediction shape/dtype/finite mismatch: {run.name}")
            predictions[(mode, seed)] = np.asarray(pred, dtype=np.float64)
            metrics[f"{mode}_seed_{seed}"] = _metric_values(target, predictions[(mode, seed)])
    if not np.array_equal(baseline_predictions["persistence"], baseline_predictions["same_time_yesterday"]):
        raise AssertionError("Persistence and h=96 same-time-yesterday should be identical")

    link_rows: list[dict[str, Any]] = []
    origin_rows: list[dict[str, Any]] = []
    method_predictions: dict[str, np.ndarray] = dict(baseline_predictions)
    method_predictions.update({f"{mode}_seed_{seed}": pred for (mode, seed), pred in predictions.items()})
    for method, pred in method_predictions.items():
        record = metrics[method]
        seed_value = "" if "_seed_" not in method else method.rsplit("_", 1)[1]
        mode_value = method.rsplit("_seed_", 1)[0] if "_seed_" in method else "baseline"
        for edge_idx, edge in enumerate(edge_order):
            error = pred[:, edge_idx] - target[:, edge_idx]
            link_rows.append({"method": method, "mode": mode_value, "seed": seed_value, "edge_index": edge_idx, "source": edge[0], "destination": edge[1], "mae": float(np.mean(np.abs(error))), "rmse": float(np.sqrt(np.mean(error * error)))})
        for origin_idx, row in enumerate(record["per_origin_mae"]):
            origin_rows.append({"method": method, "mode": mode_value, "seed": seed_value, "origin": splits["validation"]["windows"][origin_idx]["origin"], "target_index": splits["validation"]["windows"][origin_idx]["target"], "mae": row, "rmse": record["per_origin_rmse"][origin_idx]})

    paired: dict[str, Any] = {}
    for mode in MODES:
        for seed in SEEDS:
            pred = predictions[(mode, seed)]
            record = {}
            for baseline_name, base in baseline_predictions.items():
                err = pred - target
                base_err = base - target
                # Equal 72-link weight at each chronological target origin.
                mae_a = np.mean(np.abs(err), axis=1)
                mae_b = np.mean(np.abs(base_err), axis=1)
                mse_a = np.mean(err * err, axis=1)
                mse_b = np.mean(base_err * base_err, axis=1)
                record[baseline_name] = {
                    "mae_difference_model_minus_baseline": {
                        str(block): _block_bootstrap_difference(mae_a, mae_b, metric="MAE", block_length=block, seed=BOOTSTRAP_SEED + seed + block)
                        for block in BOOTSTRAP_BLOCK_LENGTHS
                    },
                    "rmse_difference_model_minus_baseline": {
                        str(block): _block_bootstrap_difference(np.sqrt(mse_a), np.sqrt(mse_b), metric="origin-RMSE difference diagnostic", block_length=block, seed=BOOTSTRAP_SEED + seed + block)
                        for block in BOOTSTRAP_BLOCK_LENGTHS
                    },
                }
            paired[f"{mode}_seed_{seed}"] = record

    topology_controls: dict[str, Any] = {}
    for control in ("identity", "shuffled"):
        topo_origin = np.mean(np.stack([np.mean(np.abs(predictions[("topology", seed)] - target), axis=1) for seed in SEEDS]), axis=0)
        control_origin = np.mean(np.stack([np.mean(np.abs(predictions[(control, seed)] - target), axis=1) for seed in SEEDS]), axis=0)
        topo_mse_origin = np.mean(np.stack([np.mean((predictions[("topology", seed)] - target) ** 2, axis=1) for seed in SEEDS]), axis=0)
        control_mse_origin = np.mean(np.stack([np.mean((predictions[(control, seed)] - target) ** 2, axis=1) for seed in SEEDS]), axis=0)
        topology_controls[control] = {
            "topology_minus_control_paired_mean_origin_mae": {
                str(block): _block_bootstrap_difference(topo_origin, control_origin, metric="MAE", block_length=block, seed=BOOTSTRAP_SEED + block)
                for block in BOOTSTRAP_BLOCK_LENGTHS
            },
            "topology_minus_control_origin_rmse_diagnostic": {
                str(block): _block_bootstrap_difference(np.sqrt(topo_mse_origin), np.sqrt(control_mse_origin), metric="origin-RMSE difference diagnostic", block_length=block, seed=BOOTSTRAP_SEED + block)
                for block in BOOTSTRAP_BLOCK_LENGTHS
            },
        }

    seed_aggregate: dict[str, Any] = {}
    for mode in MODES:
        seed_aggregate[mode] = {}
        for metric_name in ("mae", "rmse"):
            values = [metrics[f"{mode}_seed_{seed}"][metric_name] for seed in SEEDS]
            seed_aggregate[mode][metric_name] = {
                "mean": float(np.mean(values)), "sample_std_ddof1": float(np.std(values, ddof=1)),
                "min": float(min(values)), "max": float(max(values)), "seed_values": values,
            }
    result = {
        "status": "complete", "runner_id": RUNNER_ID, "protocol_id": PROTOCOL_ID,
        "dataset_protocol_id": builder.PROTOCOL_ID, "split": "validation", "target_indices": [8020, 9740],
        "history_length": HISTORY_LENGTH, "horizon": HORIZON, "validation_windows": len(target),
        "directed_links": LINK_COUNT, "coverage_pairs_per_method": int(target.size),
        "metric_definition": "float64 mean absolute/squared error over exactly aligned 953x72 finite pairs; RMSE=sqrt(mean squared error)",
        "units": "computed utilization ratio", "baseline_metrics": {key: metrics[key] for key in baseline_predictions},
        "model_metrics": {key: metrics[key] for key in metrics if key not in baseline_predictions},
        "seed_aggregate_mean_sample_std": seed_aggregate,
        "paired_block_bootstrap": {"method": "moving blocks of contiguous validation forecast origins; all 72 links kept together", "replicates": BOOTSTRAP_REPLICATES, "base_rng_seed": BOOTSTRAP_SEED, "block_lengths_origins": list(BOOTSTRAP_BLOCK_LENGTHS), "model_vs_baselines": paired, "topology_vs_controls_seed_averaged": topology_controls},
        "coverage": {"all_methods": int(target.size), "missing_or_nonfinite_pairs": 0},
        "provenance": provenance, "source_hashes": source_hashes,
        "test_access": "none; official test arrays/observations/checkpoints are absent from allowed paths",
    }
    for name in ("results.json", "report.md", "validation_per_link.csv", "validation_per_origin.csv"):
        if (output_root / name).exists():
            raise FileExistsError(f"Refusing existing validation summary artifact: {output_root / name}")
    _atomic_json(output_root / "results.json", result)
    _atomic_csv(output_root / "validation_per_link.csv", ("method", "mode", "seed", "edge_index", "source", "destination", "mae", "rmse"), link_rows)
    _atomic_csv(output_root / "validation_per_origin.csv", ("method", "mode", "seed", "origin", "target_index", "mae", "rmse"), origin_rows)
    lines = [
        "# GEANT h=96 streaming candidate validation report", "",
        "Forecasting of computed link utilization from real GEANT traffic matrices. Validation-only evidence; no test data were accessed.", "",
        "All methods cover the same 953 validation origins × 72 directed links (68,616 pairs). Checkpoint selection used only h=96 validation residual SmoothL1.", "",
        "| Method | MAE | RMSE |", "|---|---:|---:|",
    ]
    for name, values in result["baseline_metrics"].items():
        lines.append(f"| {name} | {values['mae']:.9g} | {values['rmse']:.9g} |")
    for mode in MODES:
        agg = seed_aggregate[mode]
        lines.append(f"| {mode} mean ± sample SD | {agg['mae']['mean']:.9g} ± {agg['mae']['sample_std_ddof1']:.3g} | {agg['rmse']['mean']:.9g} ± {agg['rmse']['sample_std_ddof1']:.3g} |")
    lines.extend(["", "Seed variation is descriptive, not a confidence interval. Moving-block bootstrap resamples contiguous forecast origins with block lengths 96 and 192, 2,000 replicates, RNG seed recorded in results.json. The official test split remains sealed."])
    _atomic_text(output_root / "report.md", "\n".join(lines) + "\n")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--run-training", action="store_true")
    action.add_argument("--summarize-validation", action="store_true")
    parser.add_argument("--authorize-horizon96-candidate", action="store_true")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--adjacency-mode", choices=MODES)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.summarize_validation:
        if args.authorize_horizon96_candidate or args.seed is not None or args.adjacency_mode is not None:
            parser.error("summary mode does not accept training authorization, seed, or adjacency mode")
        result = summarize_validation(args.data_root, repository_root=args.repository_root)
        print(json.dumps({"status": result["status"], "validation_windows": result["validation_windows"], "coverage_pairs_per_method": result["coverage_pairs_per_method"]}, indent=2))
        return 0
    if not args.authorize_horizon96_candidate or args.seed is None or args.adjacency_mode is None:
        parser.error("training requires explicit authorization, seed, and adjacency mode")
    result = train_one(
        args.data_root, repository_root=args.repository_root, seed=args.seed,
        mode=args.adjacency_mode, run_training=True,
        explicit_authorization=args.authorize_horizon96_candidate, device_name=args.device,
    )
    print(json.dumps({"status": result["status"], "seed": result["seed"], "mode": result["mode"], "best_epoch": result["best_epoch"], "best_validation_h96_residual_smooth_l1": result["best_validation_h96_residual_smooth_l1"], "runtime_seconds": result["runtime_seconds"]}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - real training requires explicit authorization
    raise SystemExit(main())

