"""Explicitly gated, isolated two-epoch h=96 topology smoke workflow.

This module is not a full candidate trainer.  It has one fixed seed (42), one
topology adjacency, at most two epochs, batch size two, and a smoke-only output
root.  It consumes only the six compact train/validation artifacts emitted by
``horizon96_streaming_data``.  There is no test split name, loader, or fallback
path in the runtime allowlist.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import random
import shutil
import statistics
import sys
import tempfile
import time
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from phase9_geant import horizon96_streaming_data as h96data
from phase9_geant.safe_paths import canonical_repository_root, checked_absolute
from src.gnn.spatiotemporal_link import SpatioTemporalLinkForecaster, build_link_adjacency


PROTOCOL_ID = "geant-horizon96-two-epoch-smoke-v1"
SEED = 42
HORIZON = 96
HISTORY_LENGTH = 672
LINK_COUNT = 72
ALPHA = 0.8
EWMA_STEPS = 5
SMOOTH_L1_BETA = 0.001
MAX_EPOCHS = 2
PATIENCE = 2
BATCH_SIZE = 2
NUM_WORKERS = 0
SMOKE_TRAIN_WINDOWS = 32
SMOKE_VALIDATION_WINDOWS = 32
LEARNING_RATE = 0.005
HIDDEN_SIZE = 16
OUTPUT_NAME = "geant_h96_smoke_stage20_v1"
RUN_NAME = "seed42_topology"
SMOKE_MIN_AVAILABLE_RAM = 1024 * 1024 * 1024
SMOKE_ESTIMATED_PEAK_RAM = 640 * 1024 * 1024
SMOKE_RESERVE_RAM = 384 * 1024 * 1024
SMOKE_MIN_FREE_DISK = 512 * 1024 * 1024
DATA_ARRAY_NAMES = frozenset(
    f"{name}_{split}.npy"
    for split in ("train", "validation")
    for name in ("utilization", "valid", "timestamps")
)
WINDOW_NAMES = frozenset(f"window_index_{split}_h96.json" for split in ("train", "validation"))
DATA_ALLOWED_NAMES = DATA_ARRAY_NAMES | WINDOW_NAMES | {"metadata.json"}
OUTPUT_ALLOWED_NAMES = {"run_metadata.json", "best.pt"}

SOURCE_FILES = tuple(dict.fromkeys((
    *h96data.SOURCE_FILES,
    "phase9_geant/horizon96_smoke.py",
    "docs/geant_horizon96_only_candidate_v1.md",
)))


class H96SmokeIntegrityError(ValueError):
    """Smoke data, source, checkpoint, or path identity failed closed."""


@dataclass(frozen=True)
class Standardizer:
    mean: float
    scale: float
    fit_scope: str
    population_count: int


@dataclass(frozen=True)
class ResourceCheck:
    available_ram_bytes: int
    free_disk_bytes: int
    estimated_peak_bytes: int
    reserve_bytes: int
    available_ram_semantics: str


class LazyH96Dataset(Dataset):
    """One h=96 sample at a time from a read-only compact series memmap."""

    def __init__(self, series: np.ndarray, windows: Sequence[Mapping[str, int]], split_start: int):
        if series.ndim != 2 or series.shape[1] != LINK_COUNT or series.dtype != np.float32:
            raise H96SmokeIntegrityError("series must be float32 [T,72]")
        self.series = series
        self.windows = tuple(dict(item) for item in windows)
        self.split_start = int(split_start)
        self.split_end = self.split_start + len(series)
        if not self.windows:
            raise H96SmokeIntegrityError("No valid h=96 windows are available")
        for item in self.windows:
            history_start, origin, target = (int(item[key]) for key in ("history_start", "origin", "target"))
            if (
                history_start != origin - HISTORY_LENGTH + 1
                or target != origin + HORIZON
                or not self.split_start <= history_start <= origin < target < self.split_end
            ):
                raise H96SmokeIntegrityError("Window has invalid alignment or crosses its split")

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        item = self.windows[index]
        history_start = item["history_start"] - self.split_start
        origin = item["origin"] - self.split_start
        target = item["target"] - self.split_start
        history = np.asarray(self.series[history_start : origin + 1], dtype=np.float32)
        label = np.asarray(self.series[target], dtype=np.float32)
        if history.shape != (HISTORY_LENGTH, LINK_COUNT) or label.shape != (LINK_COUNT,):
            raise H96SmokeIntegrityError("Lazy h=96 slice has unexpected dimensions")
        if not np.isfinite(history).all() or not np.isfinite(label).all():
            raise H96SmokeIntegrityError("A selected window contains invalid utilization; no imputation")
        return torch.from_numpy(history.copy()), torch.from_numpy(label.copy())


def bounded_smoke_records(
    windows: Sequence[Mapping[str, int]], limit: int,
) -> tuple[dict[str, int], ...]:
    """Select deterministic, evenly spaced windows for operational smoke only."""
    if limit <= 0 or not windows:
        raise ValueError("smoke window limit and source window count must be positive")
    indices = np.linspace(0, len(windows) - 1, num=min(limit, len(windows)), dtype=np.int64)
    if len(np.unique(indices)) != len(indices):
        raise RuntimeError("deterministic smoke selection produced duplicate windows")
    return tuple(dict(windows[int(index)]) for index in indices)


def _sha256_file(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
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


def _current_source_hashes(repository: Path) -> dict[str, str]:
    hashes = {}
    for relative in SOURCE_FILES:
        path = repository / Path(relative)
        if not path.is_file() or checked_absolute(path) != path:
            raise H96SmokeIntegrityError(f"Required source is missing or redirected: {relative}")
        hashes[relative] = _sha256_file(path)
    return hashes


def _verify_data_root(data_root: str | Path, repository: Path) -> tuple[Path, dict[str, Any]]:
    root = checked_absolute(data_root)
    expected = repository / "data" / "processed" / h96data.OUTPUT_NAME
    if os.path.normcase(os.fspath(root)) != os.path.normcase(os.fspath(expected)):
        raise H96SmokeIntegrityError(f"Data root must be the allow-listed h=96 root: {expected}")
    if not root.is_dir() or root.resolve(strict=True) != root:
        raise H96SmokeIntegrityError(f"Data root is missing or redirected: {root}")
    names = {entry.name for entry in root.iterdir()}
    if names != DATA_ALLOWED_NAMES:
        raise H96SmokeIntegrityError(
            f"Data artifact allowlist mismatch; extra={sorted(names - DATA_ALLOWED_NAMES)}, "
            f"missing={sorted(DATA_ALLOWED_NAMES - names)}"
        )
    if any("test" in name.casefold() or "h4" in name.casefold() for name in names):
        raise H96SmokeIntegrityError("Forbidden test or h=4 artifact name detected")
    metadata_path = root / "metadata.json"
    if checked_absolute(metadata_path) != metadata_path or not metadata_path.is_file():
        raise H96SmokeIntegrityError("Builder metadata is missing or redirected")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "complete" or metadata.get("protocol_id") != h96data.PROTOCOL_ID:
        raise H96SmokeIntegrityError("Builder metadata protocol/status mismatch")
    if metadata.get("source_scope") != {"train": [0, 8020], "validation": [8020, 9740]}:
        raise H96SmokeIntegrityError("Builder metadata split boundaries mismatch")
    if metadata.get("windows", {}).get("history_length") != HISTORY_LENGTH or metadata.get("windows", {}).get("horizon") != HORIZON:
        raise H96SmokeIntegrityError("Builder metadata is not 672-history/h=96-only")
    if metadata.get("test_period_source_content_access") is not False:
        raise H96SmokeIntegrityError("Builder metadata does not attest test-content exclusion")
    topology = metadata.get("topology", {})
    if (
        topology.get("node_count") != 22 or topology.get("undirected_link_count") != 36
        or topology.get("directed_link_count") != LINK_COUNT
        or topology.get("expected_local_sha256") != h96data.EXPECTED_LOCAL_TOPOLOGY_SHA256
    ):
        raise H96SmokeIntegrityError("Builder metadata topology contract mismatch")
    if metadata.get("artifact_names") != sorted(h96data.BASE_ARTIFACT_NAMES):
        raise H96SmokeIntegrityError("Builder metadata artifact name inventory mismatch")
    if set(metadata.get("array_files", {})) != h96data.SERIES_NAMES:
        raise H96SmokeIntegrityError("Builder array fingerprint inventory is incomplete or unexpected")
    if not isinstance(metadata.get("source_hashes_before_and_after"), dict):
        raise H96SmokeIntegrityError("Builder source provenance is missing")
    return root, metadata


def _load_json_windows(root: Path, name: str, expected_count: int, split_bounds: tuple[int, int]) -> list[dict[str, int]]:
    if name not in WINDOW_NAMES or "test" in name.casefold():
        raise H96SmokeIntegrityError(f"Window file name is not allow-listed: {name}")
    path = root / name
    if checked_absolute(path) != path or not path.is_file():
        raise H96SmokeIntegrityError(f"Window index is missing or redirected: {path}")
    items = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(items, list) or len(items) != expected_count:
        raise H96SmokeIntegrityError(f"Window index count mismatch: {name}")
    start, end = split_bounds
    previous_origin = -1
    for item in items:
        if not isinstance(item, dict) or set(item) != {"history_start", "origin", "target"}:
            raise H96SmokeIntegrityError(f"Malformed h=96 window entry in {name}")
        hstart, origin, target = (item[key] for key in ("history_start", "origin", "target"))
        if (
            not all(isinstance(value, int) and not isinstance(value, bool) for value in (hstart, origin, target))
            or hstart != origin - HISTORY_LENGTH + 1 or target != origin + HORIZON
            or not start <= hstart <= origin < target < end or origin <= previous_origin
        ):
            raise H96SmokeIntegrityError(f"Invalid, unordered, or cross-split window in {name}")
        previous_origin = origin
    return items


def _load_series(root: Path, name: str, metadata: Mapping[str, Any], split: str, expected_shape: tuple[int, ...]) -> np.memmap:
    if name not in DATA_ARRAY_NAMES or "test" in name.casefold():
        raise H96SmokeIntegrityError(f"Array filename is not allow-listed: {name}")
    path = root / name
    if checked_absolute(path) != path or not path.is_file():
        raise H96SmokeIntegrityError(f"Required train/validation array is missing or redirected: {path}")
    record = metadata.get("array_files", {}).get(name)
    if not isinstance(record, dict) or record.get("shape") != list(expected_shape):
        raise H96SmokeIntegrityError(f"Array metadata shape mismatch: {name}")
    if record.get("file_sha256") != _sha256_file(path):
        raise H96SmokeIntegrityError(f"Array file-byte SHA-256 mismatch: {name}")
    result = np.load(path, mmap_mode="r", allow_pickle=False)
    if not isinstance(result, np.memmap) or result.shape != expected_shape or str(result.dtype) != record.get("dtype"):
        raise H96SmokeIntegrityError(f"Loaded array schema mismatch: {name}")
    if split not in ("train", "validation"):
        raise H96SmokeIntegrityError("Only train and validation arrays are supported")
    if result.dtype.kind == "f":
        # Bounded blockwise finite/NaN verification; invalid utilization rows
        # are allowed only when the matching validity mask is false.
        for lower in range(0, len(result), 256):
            block = np.asarray(result[lower : lower + 256])
            if np.isinf(block).any():
                raise H96SmokeIntegrityError(f"Infinite values in {name}")
    return result


def load_train_validation_lazy(data_root: str | Path, repository: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load only explicit train/validation series, masks, timestamps and indices."""
    root, metadata = _verify_data_root(data_root, repository)
    splits: dict[str, Any] = {}
    expected = {"train": 8020, "validation": 1720}
    bounds = {"train": (0, 8020), "validation": (8020, 9740)}
    for split in ("train", "validation"):
        count = expected[split]
        series = _load_series(root, f"utilization_{split}.npy", metadata, split, (count, LINK_COUNT))
        valid = _load_series(root, f"valid_{split}.npy", metadata, split, (count,))
        timestamps = _load_series(root, f"timestamps_{split}.npy", metadata, split, (count,))
        if valid.dtype != np.bool_ or timestamps.dtype != np.int64:
            raise H96SmokeIntegrityError(f"Validity/timestamp dtypes are invalid for {split}")
        window_key = f"window_index_{split}_h96.json"
        index_record = metadata.get("window_index_sha256", {}).get(split)
        if not isinstance(index_record, str) or _sha256_file(root / window_key) != index_record:
            raise H96SmokeIntegrityError(f"Window-index file-byte hash mismatch for {split}")
        records = _load_json_windows(root, window_key, int(metadata["windows"][f"{split}_count"]), bounds[split])
        # Validity and finite values are checked only for selected samples here;
        # all invalid series rows must be NaN and valid rows finite, in chunks.
        for lower in range(0, count, 256):
            mask = np.asarray(valid[lower : lower + 256])
            block = np.asarray(series[lower : lower + 256])
            if not np.isfinite(mask).all() or not np.isfinite(np.asarray(timestamps[lower : lower + 256])).all():
                raise H96SmokeIntegrityError(f"Invalid mask/timestamps in {split}")
            if not np.isfinite(block[mask]).all() or not np.isnan(block[~mask]).all():
                raise H96SmokeIntegrityError(f"Series validity/NaN preservation mismatch in {split}")
        times = np.asarray(timestamps)
        if len(times) > 1 and not np.all(np.diff(times) == h96data.STEP_SECONDS):
            raise H96SmokeIntegrityError(f"Nonconsecutive {split} timestamps")
        for record in records:
            local_a = record["history_start"] - bounds[split][0]
            local_b = record["origin"] - bounds[split][0]
            local_y = record["target"] - bounds[split][0]
            if not np.asarray(valid[local_a : local_b + 1]).all() or not bool(valid[local_y]):
                raise H96SmokeIntegrityError(f"Window index points to invalid values in {split}")
        splits[split] = {"series": series, "valid": valid, "timestamps": timestamps, "windows": records, "bounds": bounds[split]}
    return splits, metadata


def fit_window_scaler(series: np.ndarray, windows: Sequence[Mapping[str, int]], split_start: int = 0) -> Standardizer:
    """Stream the exact legacy h=96 scaler population without stacked windows.

    The population is every utilization scalar in each training history window
    plus each h=96 target. Overlapping history occurrences are deliberately
    counted repeatedly because that is what ``fit_h96_training_scaler`` does
    on the materialized training arrays. No validation value is an argument.
    """
    if series.ndim != 2 or series.shape[1] != LINK_COUNT or not windows:
        raise ValueError("train scaler requires a nonempty [T,72] series and windows")
    history_differences = np.zeros(len(series) + 1, dtype=np.int64)
    target_weights = np.zeros(len(series), dtype=np.int64)
    for item in windows:
        left = int(item["history_start"]) - split_start
        origin = int(item["origin"]) - split_start
        target = int(item["target"]) - split_start
        if left < 0 or target >= len(series) or origin - left + 1 != HISTORY_LENGTH or target != origin + HORIZON:
            raise ValueError("training scaler received invalid h=96 alignment")
        history_differences[left] += 1
        history_differences[origin + 1] -= 1
        target_weights[target] += 1
    weights = np.cumsum(history_differences[:-1]) + target_weights
    count = 0
    mean = 0.0
    m2 = 0.0
    for lower in range(0, len(series), 256):
        block = np.asarray(series[lower : lower + 256], dtype=np.float64)
        block_weights = weights[lower : lower + len(block)]
        if not np.isfinite(block[block_weights > 0]).all():
            raise ValueError("Training scaler population contains invalid values")
        for row, weight in zip(block, block_weights):
            if weight <= 0:
                continue
            chunk_count = int(weight) * LINK_COUNT
            chunk_mean = float(np.mean(row, dtype=np.float64))
            centered = row - chunk_mean
            chunk_m2 = float(np.dot(centered, centered)) * int(weight)
            if count == 0:
                mean, m2, count = chunk_mean, chunk_m2, chunk_count
            else:
                total = count + chunk_count
                delta = chunk_mean - mean
                m2 += chunk_m2 + delta * delta * count * chunk_count / total
                mean += delta * chunk_count / total
                count = total
    if count == 0:
        raise ValueError("No training scalars available for scaler")
    scale = math.sqrt(m2 / count)
    if not math.isfinite(scale) or scale <= np.finfo(np.float64).eps:
        scale = 1.0
    return Standardizer(mean, scale, "h=96 train histories plus h=96 targets; window-occurrence weighted", count)


def resource_preflight(output_parent: Path, *, available_ram_bytes: int | None = None, free_disk_bytes: int | None = None) -> ResourceCheck:
    if available_ram_bytes is None:
        try:
            import psutil
            available_ram_bytes = int(psutil.virtual_memory().available)
            semantics = "psutil OS-available RAM; includes reclaimable pages and is not this process's peak working set"
        except Exception as exc:
            raise H96SmokeIntegrityError(f"Cannot safely query available RAM: {exc}") from exc
    else:
        semantics = "injected synthetic available-memory fixture"
    if free_disk_bytes is None:
        free_disk_bytes = int(shutil.disk_usage(output_parent).free)
    check = ResourceCheck(
        int(available_ram_bytes), int(free_disk_bytes), SMOKE_ESTIMATED_PEAK_RAM,
        SMOKE_RESERVE_RAM, semantics,
    )
    if check.available_ram_bytes < check.estimated_peak_bytes + check.reserve_bytes:
        raise MemoryError(
            f"Insufficient RAM headroom: available={check.available_ram_bytes}, "
            f"estimated_peak={check.estimated_peak_bytes}, reserve={check.reserve_bytes}"
        )
    if check.available_ram_bytes < SMOKE_MIN_AVAILABLE_RAM:
        raise MemoryError(f"Available RAM is below the minimum smoke threshold: {check.available_ram_bytes}")
    if check.free_disk_bytes < SMOKE_MIN_FREE_DISK:
        raise OSError(f"Free disk is below the minimum smoke threshold: {check.free_disk_bytes}")
    return check


def _planned_output(repository: Path) -> tuple[Path, Path]:
    processed = repository / "data" / "processed"
    output_root = checked_absolute(processed / OUTPUT_NAME, allow_missing=True)
    output = output_root / RUN_NAME
    if not processed.is_dir() or processed.resolve(strict=True) != processed:
        raise H96SmokeIntegrityError(f"Processed root is missing or redirected: {processed}")
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"Refusing existing smoke output namespace: {output_root}")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing existing smoke run directory: {output}")
    return output_root, output


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _model(edge_order: Sequence[tuple[str, str]]) -> SpatioTemporalLinkForecaster:
    return SpatioTemporalLinkForecaster(
        edge_order, history_length=HISTORY_LENGTH, horizons=(HORIZON,), hidden_size=HIDDEN_SIZE,
        alpha=ALPHA, ewma_steps=EWMA_STEPS, include_seasonal_features=True,
        adjacency_mode="topology",
    )


def _train_one_epoch(model, loader, scaler: Standardizer, optimizer, device: torch.device) -> tuple[float, dict[str, float | int]]:
    model.train()
    total = 0.0
    pairs = 0
    step_durations: list[float] = []
    for history, target in loader:
        step_started = time.perf_counter()
        history = history.to(device)
        target64 = target.to(device=device, dtype=torch.float64)
        normalized = (history - scaler.mean) / scaler.scale
        prediction_normalized = model(normalized)[:, 0, :].to(torch.float64)
        baseline_normalized = model._ewma(normalized).to(torch.float64)
        prediction = prediction_normalized * scaler.scale + scaler.mean
        baseline = baseline_normalized * scaler.scale + scaler.mean
        if prediction.shape != target64.shape or not (torch.isfinite(prediction).all() and torch.isfinite(target64).all()):
            raise H96SmokeIntegrityError("Smoke training produced invalid prediction/target values")
        loss = F.smooth_l1_loss(prediction - baseline, target64 - baseline, beta=SMOOTH_L1_BETA)
        if not torch.isfinite(loss):
            raise H96SmokeIntegrityError("Smoke training loss is nonfinite")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all() for parameter in model.parameters()):
            raise H96SmokeIntegrityError("Smoke training gradient is nonfinite")
        optimizer.step()
        step_durations.append(time.perf_counter() - step_started)
        n = int(target64.numel())
        total += float(loss.detach().cpu()) * n
        pairs += n
    if pairs == 0:
        raise H96SmokeIntegrityError("Smoke epoch had no training target pairs")
    return total / pairs, {
        "optimizer_steps": len(step_durations),
        "median_optimizer_step_seconds": float(statistics.median(step_durations)),
        "p95_optimizer_step_seconds": float(np.percentile(step_durations, 95)),
        "max_optimizer_step_seconds": float(max(step_durations)),
    }


@torch.no_grad()
def _validate(model, loader, scaler: Standardizer, device: torch.device) -> dict[str, float | int]:
    model.eval()
    objective_sum = abs_sum = square_sum = ewma_abs_sum = ewma_square_sum = 0.0
    count = 0
    for history, target in loader:
        history = history.to(device)
        target = target.to(device=device, dtype=torch.float64)
        normalized = (history - scaler.mean) / scaler.scale
        output = model(normalized)[:, 0, :].to(torch.float64) * scaler.scale + scaler.mean
        baseline = model._ewma(normalized).to(torch.float64) * scaler.scale + scaler.mean
        if output.shape != target.shape or not (torch.isfinite(output).all() and torch.isfinite(target).all()):
            raise H96SmokeIntegrityError("Validation prediction/target mismatch or nonfinite values")
        residual_loss = F.smooth_l1_loss(output - baseline, target - baseline, beta=SMOOTH_L1_BETA)
        error = output - target
        baseline_error = baseline - target
        n = int(target.numel())
        objective_sum += float(residual_loss.cpu()) * n
        abs_sum += float(error.abs().sum().cpu())
        square_sum += float(torch.square(error).sum().cpu())
        ewma_abs_sum += float(baseline_error.abs().sum().cpu())
        ewma_square_sum += float(torch.square(baseline_error).sum().cpu())
        count += n
    if count == 0:
        raise H96SmokeIntegrityError("Validation has no evaluated target-link pairs")
    return {
        "smooth_l1_residual_original_units": objective_sum / count,
        "mae": abs_sum / count,
        "rmse": math.sqrt(square_sum / count),
        "ewma_mae": ewma_abs_sum / count,
        "ewma_rmse": math.sqrt(ewma_square_sum / count),
        "evaluated_target_link_pairs": count,
        "units": "computed utilization ratio",
    }


def _atomic_checkpoint(payload: object, path: Path) -> str:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(payload, temporary)
        with open(temporary, "r+b") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    digest = _sha256_file(path)
    if not path.stat().st_size:
        raise H96SmokeIntegrityError("Checkpoint is empty")
    return digest


def _reload_checkpoint(model, path: Path, expected_hash: str, device: torch.device) -> None:
    raw = path.read_bytes()
    if not raw or hashlib.sha256(raw).hexdigest() != expected_hash:
        raise H96SmokeIntegrityError("Checkpoint hash changed before reload")
    payload = torch.load(io.BytesIO(raw), map_location=device, weights_only=True)
    if payload.get("protocol_id") != PROTOCOL_ID or payload.get("seed") != SEED or payload.get("horizon") != HORIZON:
        raise H96SmokeIntegrityError("Checkpoint identity mismatch")
    model.load_state_dict(payload["state_dict"], strict=True)


def _process_memory_snapshot() -> dict[str, int | None]:
    try:
        import psutil
        info = psutil.Process().memory_info()
        peak = getattr(info, "peak_wset", None)
        if peak is None:
            peak = getattr(info, "peak_rss", None)
        return {"current_process_rss_bytes": int(info.rss), "peak_process_working_set_or_rss_bytes": int(peak) if peak is not None else None}
    except Exception:
        return {"current_process_rss_bytes": None, "peak_process_working_set_or_rss_bytes": None}


def run_synthetic_benchmark(*, batch_size: int = BATCH_SIZE, steps: int = 3) -> dict[str, Any]:
    """Bounded optimizer benchmark at [B,672,72], using synthetic tensors only."""
    if batch_size != BATCH_SIZE or steps < 1 or steps > 5:
        raise ValueError("Benchmark requires intended batch size two and 1-5 measured steps")
    from phase9_geant.horizon96_experiment import _synthetic_edges_72

    torch.set_num_threads(1)
    _seed_all(SEED)
    model = _model(_synthetic_edges_72()).cpu()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    rng = torch.Generator(device="cpu").manual_seed(20261020)
    history = torch.rand((batch_size, HISTORY_LENGTH, LINK_COUNT), generator=rng)
    target = torch.rand((batch_size, LINK_COUNT), generator=rng)
    before = _process_memory_snapshot()
    timings = []
    for index in range(steps + 1):
        started = time.perf_counter()
        model.train()
        prediction = model(history)[:, 0, :]
        baseline = model._ewma(history)
        loss = F.smooth_l1_loss(prediction - baseline, target - baseline, beta=SMOOTH_L1_BETA)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if index:
            timings.append(time.perf_counter() - started)
    after = _process_memory_snapshot()
    train_windows_assumption = 7253  # maximum valid-origin count; actual validity can reduce this
    batches = math.ceil(train_windows_assumption / BATCH_SIZE)
    median_step = statistics.median(timings)
    smoke_train_batches = math.ceil(SMOKE_TRAIN_WINDOWS / BATCH_SIZE)
    smoke_validation_batches = math.ceil(SMOKE_VALIDATION_WINDOWS / BATCH_SIZE)
    return {
        "scope": "synthetic tensors only; no GEANT observations/arrays/manifests/checkpoints",
        "device": "cpu",
        "input_shape": [batch_size, HISTORY_LENGTH, LINK_COUNT],
        "target_shape": [batch_size, LINK_COUNT],
        "warmup_steps": 1,
        "measured_optimizer_steps": steps,
        "step_seconds": timings,
        "median_step_seconds": median_step,
        "observed_process_memory_before": before,
        "observed_process_memory_after": after,
        "memory_interpretation": "RSS/working-set is process memory, distinct from OS-available physical RAM and committed virtual memory; platform peak field may be unavailable",
        "epoch_time_estimate": {
            "assumed_train_windows_upper_bound": train_windows_assumption,
            "batch_size": BATCH_SIZE,
            "batches": batches,
            "estimated_seconds": median_step * batches,
            "assumption": "linear throughput from bounded synthetic optimizer steps; excludes data parsing, validation, serialization, and CPU contention",
        },
        "two_epoch_smoke_time_estimate": {
            "train_windows_per_epoch": SMOKE_TRAIN_WINDOWS,
            "validation_windows_per_epoch": SMOKE_VALIDATION_WINDOWS,
            "train_batches_per_epoch": smoke_train_batches,
            "validation_batches_per_epoch": smoke_validation_batches,
            "epochs_max": MAX_EPOCHS,
            "estimated_train_plus_validation_seconds": median_step * (smoke_train_batches * MAX_EPOCHS + smoke_validation_batches * MAX_EPOCHS),
            "assumption": "uses the measured synthetic training optimizer-step time as a conservative proxy for train and validation batches; fixed subset is operational only and is not a scientific validation estimate",
        },
    }


def run_authorized_smoke(
    *, data_root: str | Path, repository_root: str | Path,
    run_smoke: bool = False, explicit_authorization: bool = False,
) -> dict[str, Any]:
    if run_smoke is not True or explicit_authorization is not True:
        raise PermissionError("Smoke requires both --run-smoke and --authorize-h96-smoke")
    repository = canonical_repository_root(repository_root)
    output_root, output = _planned_output(repository)
    resource = resource_preflight(output_root.parent)
    source_hashes_before = _current_source_hashes(repository)
    root = checked_absolute(data_root)
    expected_data_root = repository / "data" / "processed" / h96data.OUTPUT_NAME
    if os.path.normcase(os.fspath(root)) != os.path.normcase(os.fspath(expected_data_root)):
        raise H96SmokeIntegrityError("--data-root must name the isolated streaming h=96 build")
    # Source/protocol/topology identity checks precede any array access.
    root, metadata = _verify_data_root(root, repository)
    recorded_builder_hashes = metadata["source_hashes_before_and_after"]
    if set(recorded_builder_hashes) != set(h96data.SOURCE_FILES):
        raise H96SmokeIntegrityError("Builder source hash inventory is incomplete or unexpected")
    for relative, digest in recorded_builder_hashes.items():
        current = source_hashes_before.get(relative)
        if current != digest:
            raise H96SmokeIntegrityError(f"Builder source identity differs from recorded build: {relative}")
    source_hashes_after_preflight = _current_source_hashes(repository)
    if source_hashes_before != source_hashes_after_preflight:
        raise H96SmokeIntegrityError("Smoke source identity changed during data preflight")
    topology_nodes, topology_undirected, topology_edges, local_topology_hash = h96data.approved._topology_snapshot(
        repository / h96data.TOPOLOGY_RELATIVE, repository
    )
    if (
        local_topology_hash != metadata["topology"]["sha256_local_file_identity_only"]
        or local_topology_hash != h96data.EXPECTED_LOCAL_TOPOLOGY_SHA256
        or len(topology_nodes) != 22 or len(topology_undirected) != 36
    ):
        raise H96SmokeIntegrityError("Current topology file differs from the approved local identity/structure")
    expected_edges = tuple(tuple(edge) for edge in metadata["topology"]["directed_edges_interleaved"])
    if tuple(topology_edges) != expected_edges:
        raise H96SmokeIntegrityError("Current topology directed-edge ordering differs from builder metadata")
    # Only now read/hash/map the exact six train/validation arrays.
    splits, metadata = load_train_validation_lazy(root, repository)
    train = splits["train"]
    validation = splits["validation"]
    scaler = fit_window_scaler(train["series"], train["windows"], split_start=0)
    if len(expected_edges) != LINK_COUNT or len(set(expected_edges)) != LINK_COUNT:
        raise H96SmokeIntegrityError("Builder edge-order identity is invalid")
    edge_digest = hashlib.sha256(json.dumps([list(edge) for edge in expected_edges], separators=(",", ":")).encode()).hexdigest()
    if edge_digest != metadata["topology"].get("ordered_edge_sha256"):
        raise H96SmokeIntegrityError("Builder ordered-edge digest mismatch")
    config = {
        "protocol_id": PROTOCOL_ID, "seed": SEED, "adjacency_mode": "topology",
        "history_length": HISTORY_LENGTH, "horizon": HORIZON, "prediction_channels": 1,
        "epochs_max": MAX_EPOCHS, "patience": PATIENCE, "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE, "num_workers": NUM_WORKERS, "optimizer": "Adam",
        "smoke_train_window_cap": SMOKE_TRAIN_WINDOWS,
        "smoke_validation_window_cap": SMOKE_VALIDATION_WINDOWS,
        "smoke_subset_rule": "evenly spaced indices over ordered valid h=96 window list; smoke-only, not full validation evidence",
        "loss": f"residual SmoothL1(beta={SMOOTH_L1_BETA}) in original utilization-ratio units",
        "alpha": ALPHA, "ewma_steps": EWMA_STEPS, "hidden_size": HIDDEN_SIZE,
        "initialization": "from scratch, deterministic seed 42", "checkpoint_selection": "minimum validation h=96 residual SmoothL1 only",
        "scaler_population": scaler.fit_scope, "scaler_population_count": scaler.population_count,
    }
    config_hash = _sha256_json(config)
    output_root.mkdir(parents=False, exist_ok=False)
    try:
        if checked_absolute(output_root) != output_root or output_root.resolve(strict=True) != output_root or output_root.parent.resolve(strict=True) != repository / "data" / "processed":
            raise H96SmokeIntegrityError("Smoke output root failed post-create containment")
        output.mkdir(parents=False, exist_ok=False)
        if checked_absolute(output) != output or output.resolve(strict=True) != output or output.parent.resolve(strict=True) != output_root:
            raise H96SmokeIntegrityError("Smoke run directory failed post-create containment")
        manifest: dict[str, Any] = {
            "status": "incomplete", "protocol_id": PROTOCOL_ID, "mode": "smoke",
            "seed": SEED, "configuration": config, "configuration_sha256": config_hash,
            "repository_root": str(repository), "data_root": str(root),
            "dataset_protocol_id": metadata["protocol_id"],
            "dataset_array_files": metadata["array_files"],
            "dataset_window_index_sha256": metadata["window_index_sha256"],
            "ordered_edges": [list(edge) for edge in expected_edges],
            "ordered_edge_sha256": edge_digest,
            "normalized_topology_adjacency_sha256": hashlib.sha256(
                build_link_adjacency(expected_edges, mode="topology").contiguous().numpy().tobytes(order="C")
            ).hexdigest(),
            "topology_sha256_local_identity_only": metadata["topology"]["sha256_local_file_identity_only"],
            "upstream_topology_provenance": "unverified",
            "source_hashes_before_run": source_hashes_before,
            "normalization": asdict(scaler),
            "resource_preflight": asdict(resource),
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
                "torch": torch.__version__,
                "device": "cpu",
            },
            "model_parameter_count": None,
            "validation_split_bounds": [8020, 9740],
            "test_access": {"permitted": False, "paths_or_loaders": "none"},
        }
        _atomic_json(output / "run_metadata.json", manifest)
        started = time.perf_counter()
        try:
            _seed_all(SEED)
            selected_train_windows = bounded_smoke_records(train["windows"], SMOKE_TRAIN_WINDOWS)
            selected_validation_windows = bounded_smoke_records(validation["windows"], SMOKE_VALIDATION_WINDOWS)
            train_dataset = LazyH96Dataset(train["series"], selected_train_windows, 0)
            validation_dataset = LazyH96Dataset(validation["series"], selected_validation_windows, 8020)
            manifest["smoke_subset"] = {
                "train_windows_selected": len(train_dataset),
                "train_windows_available": len(train["windows"]),
                "validation_windows_selected": len(validation_dataset),
                "validation_windows_available": len(validation["windows"]),
                "selection": config["smoke_subset_rule"],
            }
            _atomic_json(output / "run_metadata.json", manifest)
            train_loader = DataLoader(
                train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
                pin_memory=False, generator=torch.Generator().manual_seed(SEED),
            )
            validation_loader = DataLoader(
                validation_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False,
            )
            model = _model(expected_edges).cpu()
            manifest["model_parameter_count"] = sum(parameter.numel() for parameter in model.parameters())
            _atomic_json(output / "run_metadata.json", manifest)
            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
            history: list[dict[str, Any]] = []
            best_loss = float("inf")
            best_epoch = 0
            stale = 0
            checkpoint = output / "best.pt"
            step_times: list[float] = []
            memory_before = _process_memory_snapshot()
            for epoch in range(1, MAX_EPOCHS + 1):
                epoch_start = time.perf_counter()
                train_loss, step_metrics = _train_one_epoch(model, train_loader, scaler, optimizer, torch.device("cpu"))
                validation_result = _validate(model, validation_loader, scaler, torch.device("cpu"))
                elapsed = time.perf_counter() - epoch_start
                step_times.append(elapsed)
                row = {"epoch": epoch, "train_smooth_l1": train_loss, "validation_smooth_l1": validation_result["smooth_l1_residual_original_units"], "validation_metrics": validation_result, "elapsed_seconds": elapsed, "step_runtime": step_metrics}
                history.append(row)
                if validation_result["smooth_l1_residual_original_units"] < best_loss:
                    if _current_source_hashes(repository) != source_hashes_before or _sha256_json(config) != config_hash:
                        raise H96SmokeIntegrityError("Source or configuration changed before checkpoint publication")
                    best_loss = float(validation_result["smooth_l1_residual_original_units"])
                    best_epoch = epoch
                    stale = 0
                    _atomic_checkpoint({
                        "protocol_id": PROTOCOL_ID, "seed": SEED, "horizon": HORIZON,
                        "epoch": epoch, "state_dict": model.state_dict(),
                    }, checkpoint)
                else:
                    stale += 1
                manifest.update({"epoch_history": history, "best_epoch_so_far": best_epoch, "best_validation_loss_so_far": best_loss})
                _atomic_json(output / "run_metadata.json", manifest)
                if stale >= PATIENCE:
                    break
            if best_epoch == 0 or not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
                raise H96SmokeIntegrityError("No nonempty best validation checkpoint finalized")
            checkpoint_hash = _sha256_file(checkpoint)
            _reload_checkpoint(model, checkpoint, checkpoint_hash, torch.device("cpu"))
            final_validation = _validate(model, validation_loader, scaler, torch.device("cpu"))
            if not math.isclose(float(final_validation["smooth_l1_residual_original_units"]), best_loss, rel_tol=1e-6, abs_tol=1e-12):
                raise H96SmokeIntegrityError("Reloaded checkpoint objective differs from selected validation loss")
            source_hashes_after = _current_source_hashes(repository)
            if source_hashes_after != source_hashes_before or _sha256_json(config) != config_hash:
                raise H96SmokeIntegrityError("Source/configuration identity changed before completion")
            manifest.update({
                "status": "complete", "best_epoch": best_epoch,
                "best_validation_smooth_l1": best_loss,
                "completed_epochs": len(history), "epoch_history": history,
                "final_validation_metrics": final_validation,
                "checkpoint": "best.pt", "checkpoint_sha256": checkpoint_hash,
                "checkpoint_reload_strict": True,
                "runtime_seconds": time.perf_counter() - started,
                "epoch_durations_seconds": step_times,
                "process_memory_before_training": memory_before,
                "process_memory_after_training": _process_memory_snapshot(),
                "interpretation": "bounded two-epoch smoke only; not a predictive-quality or generalization claim",
            })
            if {entry.name for entry in output.iterdir()} != OUTPUT_ALLOWED_NAMES:
                raise H96SmokeIntegrityError("Smoke output inventory is incomplete or unexpected")
            _atomic_json(output / "run_metadata.json", manifest)
        except BaseException as exc:
            manifest.update({"status": "failed", "failure": f"{type(exc).__name__}: {exc}"})
            try:
                _atomic_json(output / "run_metadata.json", manifest)
            except BaseException:
                pass
            raise
    except BaseException:
        # Preserve incomplete/failed outputs for diagnosis; never delete or reuse.
        raise
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-synthetic", action="store_true", help="Bounded synthetic CPU benchmark only")
    parser.add_argument("--run-smoke", action="store_true", help="Run the fixed two-epoch seed-42 topology smoke")
    parser.add_argument("--authorize-h96-smoke", action="store_true", help="Explicit permission required with --run-smoke")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--repository-root", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.benchmark_synthetic:
        if args.run_smoke or args.authorize_h96_smoke or args.data_root or args.repository_root:
            parser.error("Synthetic benchmark cannot be combined with data or smoke options")
        print(json.dumps(run_synthetic_benchmark(), indent=2, sort_keys=True, allow_nan=False))
        return 0
    if not args.run_smoke or not args.authorize_h96_smoke:
        parser.error("Both --run-smoke and --authorize-h96-smoke are required; no default training action")
    if args.data_root is None or args.repository_root is None:
        parser.error("--data-root and --repository-root are required for smoke")
    result = run_authorized_smoke(
        data_root=args.data_root, repository_root=args.repository_root,
        run_smoke=True, explicit_authorization=True,
    )
    print(json.dumps({key: result.get(key) for key in ("status", "seed", "best_epoch", "best_validation_smooth_l1", "final_validation_metrics", "runtime_seconds")}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - real smoke needs separate authorization
    raise SystemExit(main())


__all__ = [
    "DATA_ALLOWED_NAMES", "H96SmokeIntegrityError", "LazyH96Dataset", "MAX_EPOCHS",
    "PROTOCOL_ID", "Standardizer", "bounded_smoke_records", "fit_window_scaler", "load_train_validation_lazy",
    "resource_preflight", "run_synthetic_benchmark", "run_authorized_smoke",
]
