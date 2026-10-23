"""Memory-bounded, train/validation-only GEANT h=96 data construction.

This is an isolated successor data format.  It never calls the legacy
all-series parser or the joint h=4/h=96 builder.  Raw source discovery is
delegated to the audited single-scan selector, which enumerates names but
performs metadata/content operations only on the explicitly selected train and
validation paths.  At most one OD matrix is resident at a time.

This module is intentionally inert on import.  Real-data construction requires
an explicit CLI action and explicit roots.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import stat
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from phase9_geant import horizon_v2_data as approved
from phase9_geant import validity_v2
from phase9_geant import build_link_series as link_series
from phase9_geant.safe_paths import canonical_repository_root, checked_absolute


PROTOCOL_ID = "geant-horizon96-streaming-trainval-v1"
HISTORY_LENGTH = 672
HORIZON = 96
STEP_SECONDS = 900
TRAIN_BOUNDS = (0, 8020)
VALIDATION_BOUNDS = (8020, 9740)
OUTPUT_NAME = "geant_horizon96_streaming_v1"
RAW_RELATIVE = approved.RAW_DIRECTORY_RELATIVE
TOPOLOGY_RELATIVE = approved.TOPOLOGY_RELATIVE
CAPACITY_MBPS = 40_000.0
EXPECTED_LOCAL_TOPOLOGY_SHA256 = "38c90629f30af70fe840e33071b12cc40498bb910e8dbeb2fa7863a7b645d121"
MAX_MATRIX_BYTES = 8 * 1024 * 1024
BUILD_MIN_AVAILABLE_RAM = 512 * 1024 * 1024
BUILD_MIN_FREE_DISK = 512 * 1024 * 1024

SPLITS = ("train", "validation")
SERIES_NAMES = frozenset(
    f"{kind}_{split}.npy"
    for split in SPLITS
    for kind in ("utilization", "valid", "timestamps")
)
WINDOW_NAMES = frozenset(f"window_index_{split}_h96.json" for split in SPLITS)
BASE_ARTIFACT_NAMES = SERIES_NAMES | WINDOW_NAMES | {"metadata.json"}

SOURCE_FILES = (
    "phase9_geant/__init__.py",
    "phase9_geant/horizon96_streaming_data.py",
    "phase9_geant/horizon_windows.py",
    "phase9_geant/horizon_v2_data.py",
    "phase9_geant/validity_v2.py",
    "phase9_geant/build_link_series.py",
    "phase9_geant/parse_geant.py",
    "phase9_geant/safe_paths.py",
    "src/gnn/__init__.py",
    "src/gnn/dataset.py",
    "src/gnn/preprocessing.py",
    "src/gnn/model.py",
    "src/gnn/spatiotemporal_link.py",
)


class H96BuildError(ValueError):
    """Fail-closed source, path, resource, or schema failure."""


@dataclass(frozen=True)
class ResourceSnapshot:
    available_ram_bytes: int
    free_disk_bytes: int
    ram_source: str


@dataclass(frozen=True)
class WindowRecord:
    history_start: int
    origin: int
    target: int


def _sha256_file(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    data = json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _canonical_output(repository_root: str | Path, output_root: str | Path) -> tuple[Path, Path]:
    repository = canonical_repository_root(repository_root)
    output = checked_absolute(output_root, allow_missing=True)
    processed = repository / "data" / "processed"
    expected = processed / OUTPUT_NAME
    if os.path.normcase(os.fspath(output)) != os.path.normcase(os.fspath(expected)):
        raise H96BuildError(f"Output must be the dedicated new root {expected}")
    if not processed.is_dir() or processed.resolve(strict=True) != processed:
        raise H96BuildError(f"Processed-data parent is missing or redirected: {processed}")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing existing h=96 streaming root: {output}")
    return repository, output


def system_resources(output_parent: Path) -> ResourceSnapshot:
    """Read available system memory and destination-volume free space only."""
    try:
        import psutil

        available = int(psutil.virtual_memory().available)
        source = "psutil.virtual_memory().available (OS-available RAM, not process working set)"
    except Exception as exc:  # fail closed instead of guessing on Windows
        raise H96BuildError(f"Cannot determine available RAM safely: {exc}") from exc
    disk_free = int(shutil.disk_usage(output_parent).free)
    return ResourceSnapshot(available, disk_free, source)


def require_build_resources(snapshot: ResourceSnapshot) -> None:
    if snapshot.available_ram_bytes < BUILD_MIN_AVAILABLE_RAM:
        raise MemoryError(
            f"Insufficient available RAM for bounded parser/build reserve: "
            f"{snapshot.available_ram_bytes} < {BUILD_MIN_AVAILABLE_RAM}"
        )
    if snapshot.free_disk_bytes < BUILD_MIN_FREE_DISK:
        raise OSError(
            f"Insufficient free disk for isolated build reserve: "
            f"{snapshot.free_disk_bytes} < {BUILD_MIN_FREE_DISK}"
        )


def _source_hashes(repository: Path) -> dict[str, str]:
    result = {}
    for relative in SOURCE_FILES:
        path = repository / Path(relative)
        if not path.is_file() or checked_absolute(path) != path:
            raise H96BuildError(f"Required source identity file is unavailable or redirected: {relative}")
        result[relative] = _sha256_file(path)
    return result


def _assert_output_inventory(root: Path, *, allow_incomplete: bool = False) -> None:
    names = {entry.name for entry in root.iterdir()}
    allowed = set(BASE_ARTIFACT_NAMES)
    if allow_incomplete:
        allowed.add("build_status.json")
    if names != allowed:
        raise H96BuildError(
            f"h=96 artifact inventory mismatch; unexpected={sorted(names - allowed)}, "
            f"missing={sorted(allowed - names)}"
        )
    for entry in root.iterdir():
        info = entry.lstat()
        if entry.is_symlink() or getattr(entry, "is_junction", lambda: False)() or not stat.S_ISREG(info.st_mode):
            raise H96BuildError(f"Output artifact is redirected or not a regular file: {entry.name}")
        if "test" in entry.name.casefold():
            raise H96BuildError(f"Forbidden test-split artifact name in output: {entry.name}")


def _record_timestamps(paths: Sequence[Path], global_start_index: int) -> np.ndarray:
    """Return cadence coordinates from the explicit anchor, not a timezone claim."""
    if global_start_index not in (TRAIN_BOUNDS[0], VALIDATION_BOUNDS[0]):
        raise ValueError("Unexpected official split start")
    result = np.arange(global_start_index, global_start_index + len(paths), dtype=np.int64) * STEP_SECONDS
    for offset, path in enumerate(paths):
        expected = validity_v2.EXPECTED_START + (global_start_index + offset) * validity_v2.STEP
        if validity_v2.timestamp_from_filename(path) != expected:
            raise H96BuildError(f"Selected filename is not at its expected chronological index: {path.name}")
    return result


def _parse_checked(path: Path, expected_timestamp: datetime, nodes: Sequence[str], node_index: Mapping[str, int], parser: Callable[..., Any], *, expected_hash: str | None = None) -> tuple[Any, str]:
    checked_absolute(path)
    before = _sha256_file(path)
    if expected_hash is not None and before != expected_hash:
        raise H96BuildError(f"Selected source changed since cutoff fitting: {path.name}")
    info = path.stat()
    if info.st_size <= 0 or info.st_size > MAX_MATRIX_BYTES:
        raise H96BuildError(
            f"Selected matrix size is outside bounded parser limits: {path.name} ({info.st_size} bytes)"
        )
    observation = parser(path, nodes, node_index)
    after = _sha256_file(path)
    if before != after:
        raise H96BuildError(f"Selected source changed while being parsed: {path.name}")
    if observation.filename != path.name or observation.timestamp != expected_timestamp:
        raise H96BuildError(f"Parsed filename/timestamp identity mismatch: {path.name}")
    matrix = np.asarray(observation.matrix_mbps)
    if matrix.shape != (len(nodes), len(nodes)) or matrix.dtype.kind not in "fiu" or not np.isfinite(matrix).all():
        raise H96BuildError(f"Parsed matrix has unexpected shape, dtype, or nonfinite values: {path.name}")
    return observation, after


def _fit_cutoff_streaming(
    train_paths: Sequence[Path], timestamps: Sequence[datetime], nodes: Sequence[str],
    node_index: Mapping[str, int], parser: Callable[..., Any],
) -> tuple[float, dict[str, str], list[float], list[str], int]:
    if len(train_paths) != TRAIN_BOUNDS[1] or len(timestamps) != len(train_paths):
        raise H96BuildError("Training selection must contain exactly [0,8020)")
    totals: list[float] = []
    contributing_names: list[str] = []
    hashes: dict[str, str] = {}
    empty_count = 0
    for path, stamp in zip(train_paths, timestamps):
        observation, digest = _parse_checked(path, stamp, nodes, node_index, parser)
        hashes[path.name] = digest
        if observation.demand_count == 0:
            empty_count += 1
            continue
        total = observation.total_demand_mbps
        if not math.isfinite(total) or total < 0:
            raise H96BuildError(f"Invalid training total demand in {path.name}")
        totals.append(total)
        contributing_names.append(path.name)
        del observation
    if not totals:
        raise H96BuildError("Training interval contains no nonempty demand matrices")
    median = float(np.median(np.asarray(totals, dtype=np.float64)))
    return validity_v2.TOTAL_DEMAND_MULTIPLIER * median, hashes, totals, contributing_names, empty_count


def _utilization_row(
    observation: Any,
    cutoff_mbps: float,
    nodes: Sequence[str],
    edges: Sequence[tuple[str, str]],
    routes: Mapping[tuple[str, str], Sequence[str]],
) -> tuple[np.ndarray, bool, tuple[str, ...]]:
    _traffic, utilization, reasons = validity_v2.build_link_series_v2(
        (observation,), edges, routes, cutoff_mbps, nodes=nodes
    )
    row = np.asarray(utilization[0], dtype=np.float32)
    valid = not reasons[0]
    if row.shape != (len(edges),) or (valid and not np.isfinite(row).all()):
        raise H96BuildError("Link utilization row failed shape/finite validation")
    if not valid and not np.isnan(row).all():
        raise H96BuildError("Invalid utilization must remain NaN; zero-fill is forbidden")
    return row, valid, reasons[0]


def _make_series(
    paths: Sequence[Path], timestamps: Sequence[datetime], cutoff: float,
    hashes: Mapping[str, str], nodes: Sequence[str], node_index: Mapping[str, int],
    edges: Sequence[tuple[str, str]], routes: Mapping[tuple[str, str], Sequence[str]],
    parser: Callable[..., Any],
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, str], dict[str, int]]:
    if len(paths) != len(timestamps):
        raise H96BuildError("Selected source path/timestamp count mismatch")
    series = np.empty((len(paths), len(edges)), dtype=np.float32)
    valid_mask = np.zeros(len(paths), dtype=np.bool_)
    reasons: list[str] = []
    observed_hashes: dict[str, str] = {}
    counts = {"valid": 0, "invalid": 0, "empty": 0, "total_demand": 0, "individual_demand": 0}
    for index, (path, stamp) in enumerate(zip(paths, timestamps)):
        expected = hashes.get(path.name)
        observation, digest = _parse_checked(
            path, stamp, nodes, node_index, parser, expected_hash=expected
        )
        observed_hashes[path.name] = digest
        row, is_valid, row_reasons = _utilization_row(observation, cutoff, nodes, edges, routes)
        series[index] = row
        valid_mask[index] = is_valid
        reason_text = "+".join(row_reasons)
        reasons.append(reason_text)
        counts["valid" if is_valid else "invalid"] += 1
        for reason in row_reasons:
            if reason in counts:
                counts[reason] += 1
        del observation, row
    return series, valid_mask, reasons, observed_hashes, counts


def enumerate_h96_records(
    split: str,
    valid_mask: Sequence[bool],
    timestamps_seconds: Sequence[int],
) -> tuple[WindowRecord, ...]:
    if split not in SPLITS:
        raise ValueError("Only train and validation windows are supported")
    start, end = TRAIN_BOUNDS if split == "train" else VALIDATION_BOUNDS
    valid = np.asarray(valid_mask, dtype=np.bool_)
    timestamps = np.asarray(timestamps_seconds, dtype=np.int64)
    if valid.ndim != 1 or timestamps.ndim != 1 or len(valid) != end - start or len(timestamps) != len(valid):
        raise ValueError(f"{split} validity/timestamp arrays must match its official boundary")
    if len(timestamps) > 1 and np.any(np.diff(timestamps) <= 0):
        raise ValueError("timestamps must be strictly increasing")
    invalid_prefix = np.concatenate(([0], np.cumsum(~valid, dtype=np.int64)))
    gap_flags = np.zeros(len(timestamps), dtype=np.bool_)
    gap_flags[1:] = np.diff(timestamps) != STEP_SECONDS
    gap_prefix = np.concatenate(([0], np.cumsum(gap_flags, dtype=np.int64)))
    origins = np.arange(HISTORY_LENGTH - 1, len(valid) - HORIZON, dtype=np.int64)
    history_starts = origins - HISTORY_LENGTH + 1
    targets = origins + HORIZON
    history_ok = (invalid_prefix[origins + 1] - invalid_prefix[history_starts]) == 0
    target_ok = valid[targets]
    continuity_ok = (gap_prefix[targets + 1] - gap_prefix[history_starts + 1]) == 0
    accepted = np.flatnonzero(history_ok & target_ok & continuity_ok)
    records = tuple(
        WindowRecord(
            int(start + history_starts[index]),
            int(start + origins[index]),
            int(start + targets[index]),
        )
        for index in accepted
    )
    for record in records:
        if not (start <= record.history_start <= record.origin < record.target < end):
            raise H96BuildError("h=96 window crosses its official split boundary")
    return records


def _record_json(records: Sequence[WindowRecord]) -> list[dict[str, int]]:
    return [
        {"history_start": item.history_start, "origin": item.origin, "target": item.target}
        for item in records
    ]


def build_h96_train_validation(
    *,
    repository_root: str | Path,
    output_root: str | Path,
    access_audit: dict[str, object] | None = None,
) -> dict[str, Any]:
    """Build the isolated compact h=96 dataset; never constructs a test path.

    The only production parser is called on selected train/validation paths.
    Training matrices are streamed once to fit the cutoff and again one-at-a-
    time to produce compact link series; validation matrices are parsed once.
    No list of MatrixObservation objects or history-window tensor is retained.
    """
    repository, output = _canonical_output(repository_root, output_root)
    resource = system_resources(output.parent)
    require_build_resources(resource)
    if validity_v2.CAPACITY_MBPS != CAPACITY_MBPS or link_series.CAPACITY_MBPS != CAPACITY_MBPS:
        raise H96BuildError("Link-series assumed capacity differs from the frozen 40,000 Mbps contract")
    source_before = _source_hashes(repository)
    raw = repository / RAW_RELATIVE
    selection_audit: dict[str, object] = {}
    selected = approved.expected_split_paths_for_splits(
        raw, SPLITS, repository_root=repository,
        expected_start=approved.EXPECTED_START,
        expected_start_source=approved.EXPECTED_START_SOURCE,
        access_audit=selection_audit,
    )
    if access_audit is not None:
        access_audit.update(selection_audit)
    train_paths, validation_paths = selected["train"], selected["validation"]
    if len(train_paths) != 8020 or len(validation_paths) != 1720:
        raise H96BuildError("Selected source counts do not match official train/validation intervals")
    topology_path = repository / TOPOLOGY_RELATIVE
    nodes, _undirected, edges, topology_hash = approved._topology_snapshot(topology_path, repository)
    if topology_hash != EXPECTED_LOCAL_TOPOLOGY_SHA256:
        raise H96BuildError("Local topology file-byte identity differs from the reviewed h=96 contract")
    nodes_ordered = tuple(sorted(nodes))
    node_index = {node: i for i, node in enumerate(nodes_ordered)}
    if len(nodes_ordered) != 22 or len(edges) != 72 or len(set(edges)) != 72:
        raise H96BuildError("Topology contract must be 22 nodes and 72 ordered directed links")
    routes = approved._routing_paths(nodes, edges)
    train_stamps = tuple(validity_v2.timestamp_from_filename(path) for path in train_paths)
    validation_stamps = tuple(validity_v2.timestamp_from_filename(path) for path in validation_paths)
    parser = validity_v2.parse_one_training_matrix

    cutoff, train_hashes, totals, cutoff_names, train_empty = _fit_cutoff_streaming(
        train_paths, train_stamps, nodes_ordered, node_index, parser
    )
    median = cutoff / validity_v2.TOTAL_DEMAND_MULTIPLIER
    train_series, train_valid, train_reasons, train_hashes_again, train_counts = _make_series(
        train_paths, train_stamps, cutoff, train_hashes, nodes_ordered, node_index,
        edges, routes, parser,
    )
    validation_series, validation_valid, validation_reasons, validation_hashes, validation_counts = _make_series(
        validation_paths, validation_stamps, cutoff, {}, nodes_ordered, node_index,
        edges, routes, parser,
    )
    if train_hashes != train_hashes_again:
        raise H96BuildError("Training source fingerprint changed between cutoff and series passes")
    train_seconds = _record_timestamps(train_paths, TRAIN_BOUNDS[0])
    validation_seconds = _record_timestamps(validation_paths, VALIDATION_BOUNDS[0])
    train_windows = enumerate_h96_records("train", train_valid, train_seconds)
    validation_windows = enumerate_h96_records("validation", validation_valid, validation_seconds)
    if not train_windows or not validation_windows:
        raise H96BuildError("No valid h=96 windows in one or both required splits")
    source_after = _source_hashes(repository)
    if source_before != source_after:
        raise H96BuildError("Builder source identity changed during data construction")

    config = {
        "protocol_id": PROTOCOL_ID,
        "history_length": HISTORY_LENGTH,
        "prediction_horizon": HORIZON,
        "step_seconds": STEP_SECONDS,
        "split_bounds": {"train": list(TRAIN_BOUNDS), "validation": list(VALIDATION_BOUNDS)},
        "horizons_emitted": [HORIZON],
        "capacity_mbps_assumed": CAPACITY_MBPS,
        "routing": "existing deterministic distance-shortest-path/haversine with lexicographic tie-break",
        "validity": {
            "cutoff_scope": "nonempty training matrices only",
            "total_demand_multiplier": validity_v2.TOTAL_DEMAND_MULTIPLIER,
            "total_demand_comparison": "strictly greater than cutoff is invalid",
            "individual_demand_limit_mbps": validity_v2.INDIVIDUAL_DEMAND_LIMIT_MBPS,
            "individual_demand_comparison": "strictly greater than limit is invalid",
            "empty_is_invalid": True,
            "invalid_link_values": "NaN; never imputed",
        },
        "seasonal_features": {
            "computed_inside_model_from_history_only": True,
            "h96_yesterday_source": "u[t,e], last history value; target is u[t+96,e]",
            "h96_last_week_source": "u[t-576,e], history local index 95; target is u[t+96,e]",
        },
    }
    cutoff_record = {
        "median_nonempty_training_total_demand_mbps": median,
        "multiplier": validity_v2.TOTAL_DEMAND_MULTIPLIER,
        "cutoff_mbps": cutoff,
        "nonempty_training_observation_count": len(totals),
        "empty_training_observation_count": train_empty,
        "contributing_training_filenames": cutoff_names,
        "scope": "only [0,8020); validation/test values are not used to fit threshold",
    }
    artifact_sources = {
        "train": train_hashes,
        "validation": validation_hashes,
    }
    metadata: dict[str, Any] = {
        "status": "incomplete",
        "protocol_id": PROTOCOL_ID,
        "anchor_timestamp": approved.EXPECTED_START.strftime("%Y%m%d-%H%M"),
        "anchor_provenance": approved.EXPECTED_START_SOURCE,
        "local_source_archive_checksum_verified": False,
        "source_directory_relative": str(RAW_RELATIVE).replace("\\", "/"),
        "source_scope": {"train": list(TRAIN_BOUNDS), "validation": list(VALIDATION_BOUNDS)},
        "source_file_counts": {"train": len(train_paths), "validation": len(validation_paths)},
        "source_file_hashes_sha256": artifact_sources,
        "test_period_source_content_access": False,
        "test_period_filename_visibility": "directory name enumeration only; no per-file stat/resolve/hash/open/parse",
        "source_selection_audit": selection_audit,
        "training_cutoff": cutoff_record,
        "topology": {
            "path_relative": str(TOPOLOGY_RELATIVE).replace("\\", "/"),
            "sha256_local_file_identity_only": topology_hash,
            "expected_local_sha256": EXPECTED_LOCAL_TOPOLOGY_SHA256,
            "upstream_provenance": "unverified; local digest is not authenticated upstream provenance",
            "node_count": len(nodes_ordered),
            "node_order": list(nodes_ordered),
            "undirected_link_count": 36,
            "directed_link_count": len(edges),
            "directed_edges_interleaved": [list(edge) for edge in edges],
            "ordered_edge_sha256": approved._edge_digest(edges),
            "capacity_mbps_assumed": CAPACITY_MBPS,
        },
        "configuration": config,
        "configuration_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        ).hexdigest(),
        "source_hashes_before_and_after": source_before,
        "source_hash_limitation": "file-byte hashes do not authenticate imported runtime code; mutable local concurrent-writer races are not eliminated",
        "resource_preflight": {
            **resource.__dict__,
            "minimum_available_ram_bytes": BUILD_MIN_AVAILABLE_RAM,
            "minimum_free_disk_bytes": BUILD_MIN_FREE_DISK,
            "per_matrix_parser_limit_bytes": MAX_MATRIX_BYTES,
            "design_peak_memory_bound": "one parser text/matrix plus compact [8020,72] and [1720,72] float32 series, masks, timestamps and window-index records; no stacked histories",
        },
        "validity_counts": {"train": {**train_counts, "empty_from_cutoff_pass": train_empty}, "validation": validation_counts},
        "invalid_reason_counts": {
            "train": {key: sum(key in reason.split("+") for reason in train_reasons) for key in ("empty", "total_demand", "individual_demand")},
            "validation": {key: sum(key in reason.split("+") for reason in validation_reasons) for key in ("empty", "total_demand", "individual_demand")},
        },
        "windows": {
            "history_length": HISTORY_LENGTH,
            "horizon": HORIZON,
            "target_alignment": "history=[origin-671,...,origin], target=origin+96",
            "validity_rule": "all 672 history values and target valid; timestamps consecutive through target; split-local",
            "train_count": len(train_windows),
            "validation_count": len(validation_windows),
        },
        "scaling": {
            "fit": "future smoke-run lazy h96 training-window histories and targets only",
            "validation_reuses_training_parameters": True,
            "overlap_policy": "historical candidate semantics intentionally count each overlapping training-window history and target occurrence; this exactly matches fit_h96_training_scaler on materialized h96 arrays",
            "parameters_fitted_at_build": False,
        },
        "artifact_names": sorted(BASE_ARTIFACT_NAMES),
    }

    processed = repository / "data" / "processed"
    # Recheck immediately before exclusive creation; no contents are overwritten.
    checked_absolute(output, allow_missing=True)
    if output.exists():
        raise FileExistsError(f"Refusing existing h=96 streaming root: {output}")
    output.mkdir(parents=False, exist_ok=False)
    try:
        if checked_absolute(output) != output or output.resolve(strict=True) != output or output.parent.resolve(strict=True) != processed:
            raise H96BuildError("New h=96 root failed post-creation containment check")
        _atomic_json(output / "build_status.json", {"status": "incomplete", "protocol_id": PROTOCOL_ID})
        payloads = {
            "utilization_train.npy": train_series,
            "valid_train.npy": train_valid,
            "timestamps_train.npy": train_seconds,
            "utilization_validation.npy": validation_series,
            "valid_validation.npy": validation_valid,
            "timestamps_validation.npy": validation_seconds,
        }
        for name, array in payloads.items():
            if name not in SERIES_NAMES:
                raise H96BuildError(f"Internal artifact allowlist rejected {name}")
            temporary = output / f".{name}.tmp"
            mapped = np.lib.format.open_memmap(temporary, mode="w+", dtype=array.dtype, shape=array.shape)
            mapped[:] = array
            mapped.flush()
            del mapped
            os.replace(temporary, output / name)
        _atomic_json(output / "window_index_train_h96.json", _record_json(train_windows))
        _atomic_json(output / "window_index_validation_h96.json", _record_json(validation_windows))
        metadata["array_files"] = {
            name: {
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "file_sha256": _sha256_file(output / name),
                "sha_semantics": "raw .npy file bytes",
            }
            for name, array in payloads.items()
        }
        metadata["window_index_sha256"] = {
            "train": _sha256_file(output / "window_index_train_h96.json"),
            "validation": _sha256_file(output / "window_index_validation_h96.json"),
        }
        metadata["status"] = "complete"
        _atomic_json(output / "metadata.json", metadata)
        (output / "build_status.json").unlink()
        _assert_output_inventory(output)
    except BaseException as exc:
        try:
            _atomic_json(output / "build_status.json", {"status": "failed", "failure": f"{type(exc).__name__}: {exc}"})
        except BaseException:
            pass
        raise
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-train-validation", action="store_true", help="Explicitly build only train/validation h=96 series")
    parser.add_argument("--repository-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.build_train_validation:
        parser.error("No default action. Pass --build-train-validation explicitly.")
    if args.repository_root is None or args.output_root is None:
        parser.error("--repository-root and --output-root are required")
    result = build_h96_train_validation(repository_root=args.repository_root, output_root=args.output_root)
    print(json.dumps({"status": result["status"], "output_root": str(args.output_root), "windows": result["windows"]}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - real build requires separate authorization
    raise SystemExit(main())


__all__ = [
    "BASE_ARTIFACT_NAMES", "H96BuildError", "HORIZON", "HISTORY_LENGTH", "PROTOCOL_ID",
    "ResourceSnapshot", "WindowRecord", "build_h96_train_validation", "enumerate_h96_records",
    "require_build_resources", "system_resources",
]
