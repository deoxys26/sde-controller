"""Versioned batch-4 h=96 candidate runner with explicit memory monitoring.

This module is inert on import. It consumes the existing verified compact
train/validation stream only when the explicitly authorized CLI is invoked.
It does not access test observations or artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import time
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import psutil
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from phase9_geant import horizon96_experiment as candidate
from phase9_geant import horizon96_smoke as lazy
from phase9_geant import horizon96_streaming_data as builder
from phase9_geant import horizon96_streaming_experiment as v1
from phase9_geant.safe_paths import canonical_repository_root, checked_absolute
from src.gnn.spatiotemporal_link import SpatioTemporalLinkForecaster, build_link_adjacency


PROTOCOL_ID = "geant-horizon96-only-candidate-v2-batch4"
RUNNER_ID = "geant-horizon96-streaming-experiment-v2"
OUTPUT_ROOT_NAME = "geant_horizon96_streaming_experiment_v2"
HISTORY_LENGTH = v1.HISTORY_LENGTH
HORIZON = v1.HORIZON
LINK_COUNT = v1.LINK_COUNT
SEEDS = v1.SEEDS
MODES = v1.MODES
PROCESS_PEAK_LIMIT_BYTES = lazy.SMOKE_ESTIMATED_PEAK_RAM
SYSTEM_AVAILABLE_RESERVE_BYTES = lazy.SMOKE_RESERVE_RAM

# Only batch size and run/protocol identity differ from the reviewed v1
# candidate. Smaller batches change Adam update frequency and are not claimed
# to reproduce v1's optimization trajectory.
CONFIG: dict[str, Any] = {
    **v1.CONFIG,
    "protocol_id": PROTOCOL_ID,
    "runner_id": RUNNER_ID,
    "batch_size": 4,
    "prior_protocol_id": v1.PROTOCOL_ID,
    "prior_batch_size": int(v1.CONFIG["batch_size"]),
    "optimization_protocol_amendment": (
        "batch size changed from 32 to 4 after bounded synthetic memory benchmark; "
        "optimizer updates per epoch therefore differ; all seeds and adjacency modes use 4"
    ),
    "memory_policy": {
        "startup_available_ram_minimum_bytes": lazy.SMOKE_ESTIMATED_PEAK_RAM + lazy.SMOKE_RESERVE_RAM,
        "process_peak_limit_bytes": PROCESS_PEAK_LIMIT_BYTES,
        "in_run_os_available_reserve_bytes": SYSTEM_AVAILABLE_RESERVE_BYTES,
        "workers": 0,
    },
}

RUN_SOURCE_FILES = tuple(sorted(set(
    v1.RUN_SOURCE_FILES
    + (
        "phase9_geant/horizon96_streaming_experiment_v2.py",
        "docs/geant_horizon96_memory_v2.md",
    )
)))


class MemoryLimitError(MemoryError):
    """Process peak or system-wide available RAM crossed its fixed limit."""


class ExperimentV2Error(ValueError):
    """V2 protocol, provenance, or lifecycle invariant failed."""


def _json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _source_identity(repository: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in RUN_SOURCE_FILES:
        path = checked_absolute(repository / relative)
        if not path.is_file() or path.resolve(strict=True) != path:
            raise ExperimentV2Error(f"Missing or redirected v2 run source: {relative}")
        result[relative] = _file_sha256(path)
    return result


def _verified_inputs(data_root: str | Path, repository_root: str | Path):
    """Reuse the audited train/validation-only v1 reader, then bind v2 source."""
    values = v1._verified_inputs(data_root, repository_root)
    repository, root, splits, metadata, edge_order, provenance, old_hashes = values
    current = _source_identity(repository)
    for name, digest in old_hashes.items():
        if current.get(name) != digest:
            raise ExperimentV2Error(f"Base streaming preflight source identity changed: {name}")
    if tuple(splits) != ("train", "validation"):
        raise ExperimentV2Error("Only train and validation splits are permitted")
    return repository, root, splits, metadata, edge_order, provenance, current


def _planned_run(repository: Path, seed: int, mode: str) -> tuple[Path, Path]:
    if seed not in SEEDS or mode not in MODES:
        raise ValueError("seed or adjacency mode is outside the fixed v2 protocol")
    processed = repository / "data" / "processed"
    if not processed.is_dir() or processed.resolve(strict=True) != processed:
        raise ExperimentV2Error(f"Canonical processed-data root is unavailable: {processed}")
    root = checked_absolute(processed / OUTPUT_ROOT_NAME, allow_missing=True)
    if root.parent != processed:
        raise ExperimentV2Error("V2 output root escaped the canonical processed-data directory")
    if root.exists() and (not root.is_dir() or root.resolve(strict=True) != root):
        raise ExperimentV2Error("V2 output root is redirected or not a directory")
    run = checked_absolute(root / f"h96_{mode}_seed_{seed}", allow_missing=True)
    if run.parent != root or run.exists() or run.is_symlink():
        raise FileExistsError(f"Refusing existing or invalid v2 run destination: {run}")
    allowed = {f"h96_{m}_seed_{s}" for m in MODES for s in SEEDS}
    allowed |= {"results.json", "report.md", "validation_per_link.csv", "validation_per_origin.csv"}
    if root.exists() and {entry.name for entry in root.iterdir()} - allowed:
        raise ExperimentV2Error("Unexpected entry in v2 output namespace")
    return root, run


def _read_memory_snapshot() -> dict[str, int | str | None]:
    """Separate process working set/high-water from system-wide available RAM."""
    process = psutil.Process().memory_info()
    peak = getattr(process, "peak_wset", None)
    source = "psutil.peak_wset"
    if peak is None:
        peak = getattr(process, "peak_rss", None)
        source = "psutil.peak_rss"
    if peak is None:
        source = "sampled_rss_fallback"
    return {
        "process_rss_bytes": int(process.rss),
        "process_peak_rss_or_working_set_bytes": int(peak) if peak is not None else None,
        "process_peak_measurement_source": source,
        "os_available_ram_bytes": int(psutil.virtual_memory().available),
    }


class MemoryMonitor:
    """Sample both memory domains and abort without touching prior checkpoints."""

    def __init__(
        self,
        *,
        snapshot_fn: Callable[[], Mapping[str, int | str | None]] = _read_memory_snapshot,
        process_peak_limit_bytes: int = PROCESS_PEAK_LIMIT_BYTES,
        os_available_reserve_bytes: int = SYSTEM_AVAILABLE_RESERVE_BYTES,
    ) -> None:
        self.snapshot_fn = snapshot_fn
        self.process_peak_limit_bytes = int(process_peak_limit_bytes)
        self.os_available_reserve_bytes = int(os_available_reserve_bytes)
        self.samples = 0
        self.max_rss_bytes = 0
        self.max_process_peak_bytes = 0
        self.max_process_limit_observation_bytes = 0
        self.process_peak_measurement_sources: set[str] = set()
        self.min_os_available_bytes: int | None = None
        self.last_snapshot: dict[str, int] | None = None

    def sample(self, phase: str) -> dict[str, int | str | None]:
        snapshot = dict(self.snapshot_fn())
        required = {
            "process_rss_bytes", "process_peak_rss_or_working_set_bytes",
            "process_peak_measurement_source", "os_available_ram_bytes",
        }
        numeric = {"process_rss_bytes", "os_available_ram_bytes"}
        if snapshot.get("process_peak_rss_or_working_set_bytes") is not None:
            numeric.add("process_peak_rss_or_working_set_bytes")
        if (set(snapshot) != required or not isinstance(snapshot.get("process_peak_measurement_source"), str)
                or any(int(snapshot[key]) < 0 for key in numeric)):
            raise ExperimentV2Error("Memory snapshot is incomplete or invalid")
        current = dict(snapshot)
        current["process_rss_bytes"] = int(current["process_rss_bytes"])
        current["os_available_ram_bytes"] = int(current["os_available_ram_bytes"])
        if current["process_peak_rss_or_working_set_bytes"] is not None:
            current["process_peak_rss_or_working_set_bytes"] = int(current["process_peak_rss_or_working_set_bytes"])
        source = str(current["process_peak_measurement_source"])
        if source not in {"psutil.peak_wset", "psutil.peak_rss", "sampled_rss_fallback", "synthetic_high_water"}:
            raise ExperimentV2Error(f"Unknown process memory measurement source: {source}")
        self.samples += 1
        self.max_rss_bytes = max(self.max_rss_bytes, current["process_rss_bytes"])
        self.process_peak_measurement_sources.add(source)
        measured_for_limit = current["process_peak_rss_or_working_set_bytes"]
        if measured_for_limit is None:
            measured_for_limit = current["process_rss_bytes"]
        self.max_process_limit_observation_bytes = max(self.max_process_limit_observation_bytes, measured_for_limit)
        if source != "sampled_rss_fallback":
            self.max_process_peak_bytes = max(self.max_process_peak_bytes, measured_for_limit)
        available = current["os_available_ram_bytes"]
        self.min_os_available_bytes = available if self.min_os_available_bytes is None else min(self.min_os_available_bytes, available)
        self.last_snapshot = current
        if self.max_process_limit_observation_bytes > self.process_peak_limit_bytes:
            raise MemoryLimitError(
                f"Process memory observation {self.max_process_limit_observation_bytes} exceeds limit {self.process_peak_limit_bytes} bytes during {phase}"
            )
        if available < self.os_available_reserve_bytes:
            raise MemoryLimitError(
                f"OS-available RAM {available} is below in-run reserve {self.os_available_reserve_bytes} bytes during {phase}"
            )
        return current

    def summary(self) -> dict[str, Any]:
        return {
            "samples": self.samples,
            "max_sampled_process_rss_bytes": self.max_rss_bytes,
            "max_platform_process_high_water_bytes": self.max_process_peak_bytes if self.process_peak_measurement_sources - {"sampled_rss_fallback"} else None,
            "max_process_memory_observation_used_for_limit_bytes": self.max_process_limit_observation_bytes,
            "process_peak_measurement_sources": sorted(self.process_peak_measurement_sources),
            "min_os_available_ram_bytes": self.min_os_available_bytes,
            "process_peak_limit_bytes": self.process_peak_limit_bytes,
            "in_run_os_available_reserve_bytes": self.os_available_reserve_bytes,
            "measurement_semantics": (
                "process RSS/working-set and process high-water are process-specific; "
                "OS available RAM is system-wide and is not attributed to this process"
            ),
            "last_snapshot": self.last_snapshot,
        }


def _training_epoch(
    model: torch.nn.Module,
    dataset,
    scaler,
    optimizer: torch.optim.Optimizer,
    seed: int,
    epoch: int,
    monitor: MemoryMonitor,
) -> tuple[float, dict[str, Any]]:
    model.train()
    order = np.random.default_rng(seed + epoch).permutation(len(dataset)).tolist()
    loader = DataLoader(
        Subset(dataset, order), batch_size=int(CONFIG["batch_size"]), shuffle=False,
        num_workers=0, pin_memory=False,
    )
    total, pairs = 0.0, 0
    durations: list[float] = []
    for raw_history, raw_target in loader:
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        history = (raw_history - np.float32(scaler.mean)) / np.float32(scaler.scale)
        target = raw_target.to(torch.float64)
        output = model(history)
        if output.shape != (len(raw_history), 1, LINK_COUNT):
            raise ExperimentV2Error("Training model output must be [batch,1,72]")
        prediction = output[:, 0, :].to(torch.float64) * scaler.scale + scaler.mean
        baseline_normalized = torch.einsum("bte,t->be", history[:, -5:, :], model.ewma_weights)
        baseline = baseline_normalized.to(torch.float64) * scaler.scale + scaler.mean
        loss = F.smooth_l1_loss(prediction - baseline, target - baseline, beta=0.001)
        if not torch.isfinite(loss):
            raise ExperimentV2Error("Training loss is nonfinite")
        loss.backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
            raise ExperimentV2Error("Training gradients are missing or nonfinite")
        optimizer.step()
        size = int(target.numel())
        total += float(loss.detach()) * size
        pairs += size
        optimizer.zero_grad(set_to_none=True)
        durations.append(time.perf_counter() - started)
        # Drop references to the just-consumed autograd graph and batch tensors
        # before the monitor sample and before DataLoader advances.
        del raw_history, raw_target, history, target, output, prediction
        del baseline_normalized, baseline, loss, gradients
        monitor.sample(f"train_epoch_{epoch}_batch")
    if pairs == 0:
        raise ExperimentV2Error("Training epoch emitted no target pairs")
    return total / pairs, {
        "optimizer_steps": len(durations),
        "median_optimizer_step_seconds": float(statistics.median(durations)),
        "p95_optimizer_step_seconds": float(np.percentile(durations, 95)),
        "max_optimizer_step_seconds": float(max(durations)),
    }


def _evaluate(
    model: SpatioTemporalLinkForecaster,
    loader: DataLoader,
    scaler,
    monitor: MemoryMonitor,
    *,
    collect_arrays: bool,
) -> tuple[float, np.ndarray | None, np.ndarray | None, dict[str, np.ndarray] | None]:
    """Validation objective; only final evaluation retains prediction arrays."""
    model.eval()
    objective_sum = 0.0
    pair_count = 0
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    baseline_rows: dict[str, list[np.ndarray]] = {
        name: [] for name in ("ewma_alpha_0.8", "persistence", "same_time_yesterday", "same_time_last_week")
    } if collect_arrays else {}
    with torch.no_grad():
        for raw_history, raw_target in loader:
            history = (raw_history - np.float32(scaler.mean)) / np.float32(scaler.scale)
            output = model(history)
            if output.shape != (len(raw_history), 1, LINK_COUNT):
                raise ExperimentV2Error("Validation model output must be [batch,1,72]")
            target = raw_target.to(torch.float64)
            prediction = output[:, 0, :].to(torch.float64) * scaler.scale + scaler.mean
            baseline_normalized = torch.einsum("bte,t->be", history[:, -5:, :], model.ewma_weights)
            baseline = baseline_normalized.to(torch.float64) * scaler.scale + scaler.mean
            if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
                raise ExperimentV2Error("Validation prediction or target is nonfinite")
            loss = F.smooth_l1_loss(prediction - baseline, target - baseline, beta=0.001)
            count = int(target.numel())
            objective_sum += float(loss) * count
            pair_count += count
            if collect_arrays:
                raw = raw_history.detach().cpu().numpy().astype(np.float64, copy=False)
                predictions.append(prediction.detach().cpu().numpy().astype(np.float64))
                targets.append(target.detach().cpu().numpy().astype(np.float64))
                for name, values in v1._baseline_predictions(raw).items():
                    baseline_rows[name].append(values)
            del raw_history, raw_target, history, output, target, prediction
            del baseline_normalized, baseline, loss
            monitor.sample("validation_batch")
    if pair_count == 0:
        raise ExperimentV2Error("Validation loader emitted no target pairs")
    if not collect_arrays:
        return objective_sum / pair_count, None, None, None
    return (
        objective_sum / pair_count,
        np.concatenate(predictions, axis=0),
        np.concatenate(targets, axis=0),
        {name: np.concatenate(values, axis=0) for name, values in baseline_rows.items()},
    )


def _atomic_json(path: Path, value: object) -> None:
    candidate._atomic_json(path, value)


def _atomic_npy(path: Path, array: np.ndarray) -> str:
    return v1._atomic_npy(path, array)


def _load_checkpoint_verified(
    model: SpatioTemporalLinkForecaster,
    checkpoint: Path,
    expected_sha256: str,
    *,
    seed: int,
    configuration_sha256: str,
) -> None:
    raw = checkpoint.read_bytes()
    if not raw or hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ExperimentV2Error("Checkpoint is empty or its SHA-256 changed")
    import io

    payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    if (
        payload.get("protocol_id") != PROTOCOL_ID
        or payload.get("seed") != seed
        or payload.get("configuration_sha256") != configuration_sha256
    ):
        raise ExperimentV2Error("Checkpoint protocol, seed, or configuration identity mismatch")
    model.load_state_dict(payload["state_dict"], strict=True)


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def train_one(
    data_root: str | Path,
    *,
    repository_root: str | Path,
    seed: int,
    mode: str,
    run_training: bool = False,
    explicit_authorization: bool = False,
    device_name: str = "cpu",
    monitor: MemoryMonitor | None = None,
) -> dict[str, Any]:
    if run_training is not True or explicit_authorization is not True:
        raise PermissionError("V2 training requires --run-training and --authorize-horizon96-candidate")
    if seed not in SEEDS or mode not in MODES or device_name != "cpu":
        raise ValueError("seed, adjacency mode, or device is outside the fixed v2 protocol")
    repository = canonical_repository_root(repository_root)
    output_root, output = _planned_run(repository, seed, mode)
    resource = lazy.resource_preflight(repository / "data" / "processed")
    repository, verified_root, splits, dataset_metadata, edge_order, provenance, source_hashes = _verified_inputs(data_root, repository)
    scaler = lazy.fit_window_scaler(splits["train"]["series"], splits["train"]["windows"], split_start=0)
    if scaler.fit_scope != "h=96 train histories plus h=96 targets; window-occurrence weighted":
        raise ExperimentV2Error("Training-only scaler scope mismatch")
    config = {**CONFIG, "seed": seed, "adjacency_mode": mode, "device": device_name}
    config_sha = _json_sha256(config)
    adjacency = build_link_adjacency(edge_order, mode=mode, shuffle_seed=20261019)
    adjacency_sha = hashlib.sha256(adjacency.contiguous().numpy().tobytes(order="C")).hexdigest()
    monitor = monitor or MemoryMonitor()
    first_snapshot = monitor.sample("startup_before_output")

    if not output_root.exists():
        output_root.mkdir(parents=False, exist_ok=False)
        if checked_absolute(output_root) != output_root or output_root.resolve(strict=True) != output_root:
            raise ExperimentV2Error("Created v2 output root failed path-containment verification")
    output.mkdir(parents=False, exist_ok=False)
    if checked_absolute(output) != output or output.resolve(strict=True) != output or output.parent.resolve(strict=True) != output_root:
        raise ExperimentV2Error("Created v2 run directory failed path-containment verification")

    manifest: dict[str, Any] = {
        "status": "incomplete", "protocol_id": PROTOCOL_ID, "runner_id": RUNNER_ID,
        "seed": seed, "adjacency_mode": mode, "configuration": config,
        "configuration_sha256": config_sha, "repository_root": str(repository),
        "data_root": str(verified_root), "dataset_metadata_sha256": _json_sha256(dataset_metadata),
        "provenance": provenance, "source_hashes": source_hashes,
        "normalization": scaler.__dict__,
        "adjacency": {"mode": mode, "shuffle_seed": 20261019 if mode == "shuffled" else None,
                      "normalized_sha256": adjacency_sha},
        "environment": {"python": __import__("sys").version, "numpy": np.__version__,
                        "torch": torch.__version__, "device": device_name},
        "resource_preflight": resource.__dict__, "memory_initial_snapshot": first_snapshot,
        "memory_monitor": monitor.summary(),
        "checkpoint_selection": "minimum h=96 validation residual SmoothL1 only",
        "test_access": {"permitted": False, "paths_or_loaders": "none"},
    }
    _atomic_json(output / "run.json", manifest)
    started = time.perf_counter()
    history_rows: list[dict[str, Any]] = []
    try:
        _seed_all(seed)
        model = SpatioTemporalLinkForecaster(
            edge_order, history_length=HISTORY_LENGTH, horizons=(HORIZON,),
            hidden_size=int(CONFIG["hidden_size"]), alpha=0.8, ewma_steps=5,
            include_seasonal_features=True, adjacency_mode=mode, shuffle_seed=20261019,
        ).cpu()
        optimizer = torch.optim.Adam(model.parameters(), lr=float(CONFIG["learning_rate"]))
        train_dataset = lazy.LazyH96Dataset(splits["train"]["series"], splits["train"]["windows"], split_start=0)
        validation_dataset = lazy.LazyH96Dataset(splits["validation"]["series"], splits["validation"]["windows"], split_start=8020)
        validation_loader = DataLoader(validation_dataset, batch_size=int(CONFIG["batch_size"]), shuffle=False, num_workers=0, pin_memory=False)
        manifest.update({"model_parameter_count": sum(p.numel() for p in model.parameters()),
                         "training_window_count": len(train_dataset), "validation_window_count": len(validation_dataset)})
        _atomic_json(output / "run.json", manifest)
        best_loss, best_epoch, stale = float("inf"), 0, 0
        checkpoint = output / "best.pt"
        for epoch in range(1, int(CONFIG["epochs"]) + 1):
            epoch_started = time.perf_counter()
            train_loss, step_stats = _training_epoch(model, train_dataset, scaler, optimizer, seed, epoch, monitor)
            val_loss, _, _, _ = _evaluate(model, validation_loader, scaler, monitor, collect_arrays=False)
            if not math.isfinite(val_loss):
                raise ExperimentV2Error("Validation selection loss is nonfinite")
            if val_loss < best_loss:
                if _source_identity(repository) != source_hashes or _json_sha256(config) != config_sha:
                    raise ExperimentV2Error("Source or configuration changed before checkpoint write")
                best_loss, best_epoch, stale = val_loss, epoch, 0
                digest = candidate._atomic_checkpoint({
                    "state_dict": model.state_dict(), "seed": seed, "best_epoch": epoch,
                    "protocol_id": PROTOCOL_ID, "configuration_sha256": config_sha,
                }, checkpoint)
                manifest["checkpoint_sha256_so_far"] = digest
            else:
                stale += 1
            row = {"epoch": epoch, "train_h96_residual_smooth_l1": train_loss,
                   "validation_h96_residual_smooth_l1": val_loss,
                   "epoch_seconds": time.perf_counter() - epoch_started,
                   "optimizer_step_stats": step_stats, "memory_monitor": monitor.summary()}
            history_rows.append(row)
            manifest.update({"epoch_history": history_rows, "best_epoch_so_far": best_epoch,
                             "best_validation_loss_so_far": best_loss, "memory_monitor": monitor.summary()})
            _atomic_json(output / "run.json", manifest)
            if stale >= int(CONFIG["patience"]):
                break
        if best_epoch == 0 or not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
            raise ExperimentV2Error("No best validation checkpoint was finalized")
        repository2, root2, _, metadata2, edges2, provenance2, source_hashes2 = _verified_inputs(data_root, repository)
        if (repository2 != repository or root2 != verified_root or metadata2 != dataset_metadata
                or edges2 != edge_order or provenance2 != provenance or source_hashes2 != source_hashes
                or _source_identity(repository) != source_hashes or _json_sha256(config) != config_sha):
            raise ExperimentV2Error("Data, topology, provenance, source, or config identity changed before reload")
        checkpoint_sha = _file_sha256(checkpoint)
        _load_checkpoint_verified(model, checkpoint, checkpoint_sha, seed=seed, configuration_sha256=config_sha)
        final_loss, predictions, targets, baselines = _evaluate(model, validation_loader, scaler, monitor, collect_arrays=True)
        if not math.isclose(final_loss, best_loss, rel_tol=1e-6, abs_tol=1e-12):
            raise ExperimentV2Error("Strict checkpoint reload changed validation objective")
        assert predictions is not None and targets is not None and baselines is not None
        validation_metrics = {"h96_candidate": v1._metric_values(targets, predictions)}
        validation_metrics.update({name: v1._metric_values(targets, pred) for name, pred in baselines.items()})
        prediction_sha = _atomic_npy(output / "validation_predictions.npy", predictions.astype(np.float64, copy=False))
        if _source_identity(repository) != source_hashes or _json_sha256(config) != config_sha:
            raise ExperimentV2Error("Source/configuration changed before completion")
        if {entry.name for entry in output.iterdir()} != {"run.json", "best.pt", "validation_predictions.npy"}:
            raise ExperimentV2Error("V2 run artifact inventory mismatch")
        manifest.update({
            "status": "complete", "best_epoch": best_epoch,
            "best_validation_h96_residual_smooth_l1": best_loss,
            "epochs_completed": len(history_rows),
            "early_stopping": {"patience": CONFIG["patience"], "stopped_early": len(history_rows) < CONFIG["epochs"]},
            "epoch_history": history_rows, "validation_metrics": validation_metrics,
            "checkpoint": "best.pt", "checkpoint_sha256": checkpoint_sha,
            "checkpoint_reload_strict": True, "validation_predictions": "validation_predictions.npy",
            "validation_predictions_sha256": prediction_sha, "runtime_seconds": time.perf_counter() - started,
            "memory_monitor": monitor.summary(),
            "interpretation": "validation-only; not held-out generalization; v2 batch-4 optimization protocol",
        })
        _atomic_json(output / "run.json", manifest)
    except BaseException as exc:
        manifest.update({"status": "failed", "failure": f"{type(exc).__name__}: {exc}",
                         "epoch_history": history_rows, "memory_monitor": monitor.summary()})
        try:
            _atomic_json(output / "run.json", manifest)
        except BaseException:
            pass
        raise
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-training", action="store_true")
    parser.add_argument("--authorize-horizon96-candidate", action="store_true")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--repository-root", type=Path)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--adjacency-mode", choices=MODES)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.run_training or not args.authorize_horizon96_candidate:
        raise SystemExit("Training requires --run-training and --authorize-horizon96-candidate")
    if args.data_root is None or args.repository_root is None or args.seed is None or args.adjacency_mode is None:
        raise SystemExit("--data-root, --repository-root, --seed, and --adjacency-mode are required")
    result = train_one(
        args.data_root, repository_root=args.repository_root, seed=args.seed,
        mode=args.adjacency_mode, run_training=True, explicit_authorization=True, device_name=args.device,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
