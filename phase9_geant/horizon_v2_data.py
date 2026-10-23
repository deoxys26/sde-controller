"""Isolated, test-free data path for the GEANT horizon-v2 experiment.

This module only constructs paths for the declared train or validation
intervals. It never discovers or constructs a test-period matrix/array path.
The official test split is deliberately not a supported input.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

from phase9_geant import validity_v2 as v1_validity
from phase9_geant.build_link_series import compute_all_pairs_paths, haversine
from phase9_geant.horizon_windows import OFFICIAL_SPLIT_BOUNDS, enumerate_horizon_windows
from phase9_geant.parse_geant import parse_topology_text
from phase9_geant import safe_paths
from phase9_geant.safe_paths import canonical_repository_root, checked_absolute, require_regular_file


PROTOCOL_ID = "geant-horizons-v2-trainval"
EXPECTED_START = v1_validity.EXPECTED_START
EXPECTED_START_SOURCE = (
    "SNDlib GEANT overview and GEANT readme cited by the existing project; "
    "local archive checksum/provenance is not independently verified"
)
STEP = timedelta(minutes=15)
STEP_SECONDS = 900
HISTORY_LENGTH = 672
HORIZONS = (4, 96)
EWMA_ALPHA = 0.8
EWMA_STEPS = 5
EXPECTED_NODES = 22
EXPECTED_UNDIRECTED_LINKS = 36
EXPECTED_DIRECTED_LINKS = 72
CAPACITY_MBPS = 40_000.0
BUILD_CONFIGURATION = {
    "training_boundary": list(OFFICIAL_SPLIT_BOUNDS["train"]),
    "validation_boundary": list(OFFICIAL_SPLIT_BOUNDS["validation"]),
    "history_length": HISTORY_LENGTH,
    "horizons": list(HORIZONS),
    "step_seconds": STEP_SECONDS,
    "validity_multiplier": v1_validity.TOTAL_DEMAND_MULTIPLIER,
    "individual_demand_limit_mbps": v1_validity.INDIVIDUAL_DEMAND_LIMIT_MBPS,
    "capacity_mbps_assumed": CAPACITY_MBPS,
}
RAW_DIRECTORY_RELATIVE = Path("data/raw/geant_topology/directed-geant-uhlig-15min-over-4months-ALL-native")
TOPOLOGY_RELATIVE = Path("data/raw/geant_topology/geant.txt")
OUTPUT_PREFIX = "geant_validity_horizons_v2_trainval_"
MATRIX_PATTERN = re.compile(r"^demandMatrix-geant-uhlig-15min-(\d{8}-\d{4})\.txt$")

_ARRAY_NAMES = frozenset(
    f"{kind}_{split}_h{horizon}.npy"
    for split in ("train", "validation")
    for horizon in HORIZONS
    for kind in ("history", "target")
)
_JSON_NAMES = frozenset({"metadata.json", "window_index_train.json", "window_index_validation.json"})
_EXPECTED_ARTIFACT_NAMES = _ARRAY_NAMES | _JSON_NAMES


@dataclass(frozen=True)
class SplitArrays:
    split: str
    histories: Mapping[int, np.ndarray]
    targets: Mapping[int, np.ndarray]
    windows: Mapping[int, tuple[object, ...]]


def validate_artifact_name(name: str) -> str:
    """Accept only explicitly named train/validation artifacts; reject test/unknown."""
    if not isinstance(name, str) or Path(name).name != name or "test" in name.casefold():
        raise ValueError(f"Forbidden horizon-v2 artifact name: {name!r}")
    if name not in _ARRAY_NAMES and name not in _JSON_NAMES:
        raise ValueError(f"Unknown horizon-v2 artifact name: {name!r}")
    return name


def _canonical_horizon_root(data_root: str | Path, repository_root: str | Path) -> Path:
    repo = canonical_repository_root(repository_root)
    root = checked_absolute(data_root)
    processed = repo / "data" / "processed"
    suffix = root.name[len(OUTPUT_PREFIX) :] if root.name.startswith(OUTPUT_PREFIX) else ""
    if root.parent != processed or not suffix or not all(c.isalnum() or c in "-_" for c in suffix):
        raise ValueError(f"Horizon-v2 root must be a direct versioned child of {processed}")
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Horizon-v2 data root does not exist or is not a directory: {root}")
    if root.resolve(strict=True) != root:
        raise ValueError(f"Horizon-v2 data root is redirected: {root}")
    return root


def load_horizon_array(
    data_root: str | Path,
    filename: str,
    *,
    repository_root: str | Path,
    array_loader: Callable[..., np.ndarray] = np.load,
) -> np.ndarray:
    """Load exactly one direct-child train/validation array through the allowlist."""
    name = validate_artifact_name(filename)
    if name not in _ARRAY_NAMES:
        raise ValueError(f"Not a horizon-v2 array artifact: {name}")
    root = _canonical_horizon_root(data_root, repository_root)
    path = require_regular_file(root / name, root)
    value = array_loader(path, allow_pickle=False)
    if not isinstance(value, np.ndarray) or value.dtype.hasobject:
        raise ValueError(f"Horizon array must be a numeric ndarray: {name}")
    if value.dtype.kind not in "fiu" or not np.isfinite(value).all():
        raise ValueError(f"Horizon array must contain only finite numeric values: {name}")
    return value


def _assert_artifact_inventory(root: Path) -> None:
    """Reject unexpected names without opening any file contents.

    The sole post-build addition permitted is the dedicated ``runs`` directory
    containing only the three fixed adjacency modes crossed with fixed seeds.
    """
    names = {entry.name for entry in root.iterdir()}
    allowed_names = _EXPECTED_ARTIFACT_NAMES | {"runs"}
    if not _EXPECTED_ARTIFACT_NAMES <= names or not names <= allowed_names:
        extra = sorted(names - allowed_names)
        missing = sorted(_EXPECTED_ARTIFACT_NAMES - names)
        raise ValueError(f"Horizon-v2 artifact inventory mismatch; extra={extra}, missing={missing}")
    for entry in root.iterdir():
        if entry.name == "runs":
            if entry.is_symlink() or not entry.is_dir() or entry.resolve(strict=True) != entry:
                raise ValueError("Horizon-v2 runs directory is redirected")
            expected_runs = {
                f"horizon_v2_{mode}_seed_{seed}"
                for mode in ("topology", "identity", "shuffled")
                for seed in (42, 123, 2026)
            }
            for run in entry.iterdir():
                if run.name not in expected_runs or run.is_symlink() or not run.is_dir():
                    raise ValueError(f"Unexpected horizon-v2 run entry: {run.name}")
                if run.resolve(strict=True) != run:
                    raise ValueError(f"Horizon-v2 run directory is redirected: {run.name}")
                child_names = {child.name for child in run.iterdir()}
                if child_names not in ({"run.json"}, {"run.json", "best.pt"}):
                    raise ValueError(f"Horizon-v2 run artifact inventory mismatch: {run.name}")
                if any(child.is_symlink() or not child.is_file() for child in run.iterdir()):
                    raise ValueError(f"Horizon-v2 run contains a redirected/non-file artifact: {run.name}")
            continue
        if entry.is_symlink() or not entry.is_file():
            raise ValueError(f"Unexpected non-regular horizon-v2 artifact: {entry.name}")


def expected_split_paths(
    raw_directory: str | Path,
    split: str,
    *,
    expected_start: datetime = EXPECTED_START,
    expected_start_source: str = EXPECTED_START_SOURCE,
    repository_root: str | Path,
    access_audit: dict[str, object] | None = None,
) -> tuple[Path, ...]:
    """Select one official split through a single name-only directory scan.

    The scan necessarily sees directory-entry names (including nonselected
    GEANT matrix names), but only exact requested split entries receive a
    per-file metadata check. Nonselected matrix entries are never statted,
    resolved, hashed, opened, or parsed.
    """
    return expected_split_paths_for_splits(
        raw_directory,
        (split,),
        expected_start=expected_start,
        expected_start_source=expected_start_source,
        repository_root=repository_root,
        access_audit=access_audit,
    )[split]


def expected_split_paths_for_splits(
    raw_directory: str | Path,
    splits: Sequence[str],
    *,
    expected_start: datetime = EXPECTED_START,
    expected_start_source: str = EXPECTED_START_SOURCE,
    repository_root: str | Path,
    access_audit: dict[str, object] | None = None,
) -> dict[str, tuple[Path, ...]]:
    """Select exact train/validation path lists with one bounded scan.

    Unknown directory entries, malformed GEANT matrix names, duplicate names,
    and missing expected files fail closed. Valid matrix names outside the
    explicitly requested split(s) are counted from their names only; their
    paths are not passed to stat/resolve or any file-content operation.
    """
    if not splits or len(set(splits)) != len(splits) or any(
        split not in ("train", "validation") for split in splits
    ):
        raise ValueError("Only unique train and/or validation splits are supported; test is unsupported")
    if not isinstance(expected_start, datetime) or not isinstance(expected_start_source, str) or not expected_start_source.strip():
        raise ValueError("An explicit timestamp anchor and provenance reference are required")
    repo = canonical_repository_root(repository_root)
    raw = checked_absolute(raw_directory)
    expected_raw = repo / RAW_DIRECTORY_RELATIVE
    if os.path.normcase(str(raw)) != os.path.normcase(str(expected_raw)):
        raise ValueError(f"Source directory is not the configured GEANT matrix directory: {raw}")
    if not raw.exists() or not raw.is_dir() or raw.resolve(strict=True) != raw:
        raise FileNotFoundError(f"Canonical GEANT source directory is unavailable: {raw}")

    expected_names: dict[str, dict[str, str]] = {}
    expected_order: dict[str, tuple[str, ...]] = {}
    for split in splits:
        start, end = OFFICIAL_SPLIT_BOUNDS[split]
        names = tuple(
            f"demandMatrix-geant-uhlig-15min-{(expected_start + index * STEP):%Y%m%d-%H%M}.txt"
            for index in range(start, end)
        )
        if any(MATRIX_PATTERN.fullmatch(name) is None for name in names):
            raise RuntimeError("Internal timestamp filename construction failed")
        expected_order[split] = names
        expected_names[split] = {name.casefold(): name for name in names}

    wanted: dict[str, tuple[str, str]] = {}
    for split, names in expected_names.items():
        for folded, exact in names.items():
            if folded in wanted:
                raise ValueError("Requested official splits have overlapping filename identities")
            wanted[folded] = (split, exact)

    started = time.perf_counter()
    root_stat_before = raw.stat()
    seen_casefold: set[str] = set()
    seen_timestamps: set[datetime] = set()
    selected_entries: dict[str, dict[str, os.DirEntry[str]]] = {split: {} for split in splits}
    nonselected_matrix_names = 0
    enumerated_names = 0
    prefix_folded = "demandmatrix-geant-uhlig-15min-"
    with os.scandir(raw) as entries:
        for entry in entries:
            enumerated_names += 1
            name = entry.name
            folded = name.casefold()
            if folded in seen_casefold:
                raise ValueError(f"Duplicate/case-colliding source directory name: {name!r}")
            seen_casefold.add(folded)
            if Path(name).name != name or not folded.startswith(prefix_folded):
                raise ValueError(f"Unexpected entry in GEANT matrix directory: {name!r}")
            match = MATRIX_PATTERN.fullmatch(name)
            if match is None:
                raise ValueError(f"Malformed GEANT matrix filename: {name!r}")
            stamp = datetime.strptime(match.group(1), "%Y%m%d-%H%M")
            if stamp in seen_timestamps:
                raise ValueError(f"Duplicate GEANT source timestamp in directory listing: {stamp}")
            seen_timestamps.add(stamp)
            offset = stamp - expected_start
            if offset % STEP:
                raise ValueError(f"Unexpected off-cadence GEANT filename: {name!r}")
            selected = wanted.get(folded)
            if selected is None:
                nonselected_matrix_names += 1
                # Do not call entry.stat(), entry.is_file(), resolve(), or open
                # for any valid but nonselected (e.g. validation/test) name.
                continue
            split, exact_name = selected
            if name != exact_name:
                raise ValueError(f"Case-mismatched alias for selected GEANT file: {name!r}")
            selected_entries[split][exact_name] = entry

    missing_by_split = {
        split: [name for name in expected_order[split] if name not in selected_entries[split]]
        for split in splits
    }
    missing_by_split = {split: names for split, names in missing_by_split.items() if names}
    if missing_by_split:
        missing_count = sum(map(len, missing_by_split.values()))
        first_missing = next(iter(missing_by_split.values()))[0]
        raise FileNotFoundError(
            f"Missing {missing_count} expected source file(s); first missing filename: {first_missing}"
        )

    paths_by_split: dict[str, tuple[Path, ...]] = {}
    selected_stat_count = 0
    for split in splits:
        paths: list[Path] = []
        for name in expected_order[split]:
            entry = selected_entries[split][name]
            path = raw / name
            info = entry.stat(follow_symlinks=False)
            selected_stat_count += 1
            if (
                entry.is_symlink()
                or safe_paths._file_attributes(path, info) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
                or not stat.S_ISREG(info.st_mode)
            ):
                raise ValueError(f"Selected GEANT source is redirected or not a regular file: {name}")
            paths.append(path)
        paths_by_split[split] = tuple(paths)

    # Recheck the source directory after enumeration. This remains path-based
    # and cannot eliminate a hostile concurrent replacement race.
    checked_after = checked_absolute(raw)
    root_stat_after = raw.stat()
    if (
        checked_after != raw
        or raw.resolve(strict=True) != raw
        or (root_stat_before.st_dev, root_stat_before.st_ino)
        != (root_stat_after.st_dev, root_stat_after.st_ino)
    ):
        raise ValueError("GEANT source directory changed or redirected during filename preflight")
    if access_audit is not None:
        access_audit.update({
            "directory_scans": 1,
            "directory_entry_names_enumerated": enumerated_names,
            "nonselected_geant_matrix_names_seen_without_per_file_stat": nonselected_matrix_names,
            "selected_file_metadata_checks": selected_stat_count,
            "filename_preflight_seconds": time.perf_counter() - started,
            "content_reads_during_preflight": 0,
        })
    return paths_by_split


def _directed_edge_order(undirected_links: Sequence[tuple[str, str]]) -> tuple[tuple[str, str], ...]:
    edges = tuple(edge for left, right in undirected_links for edge in ((left, right), (right, left)))
    if len(edges) != EXPECTED_DIRECTED_LINKS or len(set(edges)) != EXPECTED_DIRECTED_LINKS:
        raise ValueError("Topology does not satisfy the ordered 72-directed-link contract")
    return edges


def _edge_digest(edges: Sequence[tuple[str, str]]) -> str:
    return hashlib.sha256(json.dumps([list(edge) for edge in edges], separators=(",", ":")).encode("utf-8")).hexdigest()


def _topology_snapshot(topology_path: Path, repository_root: Path) -> tuple[dict, list, tuple, str]:
    repo = canonical_repository_root(repository_root)
    expected_path = repo / TOPOLOGY_RELATIVE
    topology_path = checked_absolute(topology_path)
    if os.path.normcase(str(topology_path)) != os.path.normcase(str(expected_path)):
        raise ValueError(f"Topology path is not the configured GEANT topology: {topology_path}")
    path = require_regular_file(topology_path, topology_path.parent)
    raw_bytes = path.read_bytes()
    digest = hashlib.sha256(raw_bytes).hexdigest()
    nodes, undirected = parse_topology_text(raw_bytes.decode("utf-8"), path=str(path))
    if len(nodes) != EXPECTED_NODES or len(undirected) != EXPECTED_UNDIRECTED_LINKS:
        raise ValueError("Unexpected GEANT topology structure")
    edges = _directed_edge_order(undirected)
    return nodes, undirected, edges, digest


def _routing_paths(nodes: dict, edges: Sequence[tuple[str, str]]) -> dict[tuple[str, str], list[str]]:
    def distance(left: str, right: str) -> float:
        lon1, lat1 = nodes[left]
        lon2, lat2 = nodes[right]
        return haversine(lon1, lat1, lon2, lat2)

    paths = compute_all_pairs_paths(nodes, list(edges), distance)
    if len(paths) != len(nodes) * (len(nodes) - 1):
        raise ValueError("Topology routing did not produce all ordered OD paths")
    return paths


def _build_split_arrays(
    split: str,
    observations: Sequence[v1_validity.MatrixObservation],
    cutoff_mbps: float,
    nodes: Sequence[str],
    edges: Sequence[tuple[str, str]],
    paths: Mapping[tuple[str, str], Sequence[str]],
) -> SplitArrays:
    start, end = OFFICIAL_SPLIT_BOUNDS[split]
    if len(observations) != end - start:
        raise ValueError(f"{split} reader returned the wrong source count")
    traffic, utilization, reasons = v1_validity.build_link_series_v2(
        observations, edges, paths, cutoff_mbps, nodes=nodes
    )
    del traffic
    valid = np.asarray([not reason for reason in reasons], dtype=np.bool_)
    timestamps = tuple(observation.timestamp for observation in observations)
    seconds = tuple(int((stamp - timestamps[0]).total_seconds()) for stamp in timestamps)
    windows_by_horizon = enumerate_horizon_windows(
        split, valid, seconds, history_length=HISTORY_LENGTH, horizons=HORIZONS, step_seconds=STEP_SECONDS
    )
    histories: dict[int, np.ndarray] = {}
    targets: dict[int, np.ndarray] = {}
    for horizon, windows in windows_by_horizon.items():
        if not windows:
            raise ValueError(f"No valid {split} windows for horizon {horizon}")
        histories[horizon] = np.stack(
            [utilization[item.history_start - start : item.origin - start + 1] for item in windows]
        ).astype(np.float32, copy=False)
        targets[horizon] = np.stack(
            [utilization[item.target - start] for item in windows]
        ).astype(np.float32, copy=False)
        if not np.isfinite(histories[horizon]).all() or not np.isfinite(targets[horizon]).all():
            raise ValueError(f"Invalid/NaN value entered a {split} h={horizon} window")
    return SplitArrays(split, histories, targets, windows_by_horizon)


def _observation_fingerprint(observations: Sequence[v1_validity.MatrixObservation]) -> str:
    """Hash parsed, ordered source semantics without persisting raw matrices."""
    digest = hashlib.sha256()
    for observation in observations:
        digest.update(observation.filename.encode("utf-8"))
        digest.update(b"\0")
        digest.update(observation.timestamp.isoformat().encode("ascii"))
        digest.update(int(observation.demand_count).to_bytes(8, "big", signed=False))
        digest.update(np.asarray(observation.matrix_mbps, dtype="<f8").tobytes(order="C"))
    return digest.hexdigest()


def _current_build_source_hashes(repository_root: Path) -> dict[str, str]:
    source_names = (
        "phase9_geant/horizon_v2_data.py",
        "phase9_geant/horizon_windows.py",
        "phase9_geant/validity_v2.py",
        "phase9_geant/parse_geant.py",
        "phase9_geant/build_link_series.py",
    )
    return {
        name: hashlib.sha256((repository_root / name).read_bytes()).hexdigest()
        for name in source_names
    }


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


def build_train_validation(
    *,
    repository_root: str | Path,
    output_root: str | Path,
    raw_directory: str | Path | None = None,
    topology_path: str | Path | None = None,
) -> dict[str, object]:
    """Build train/validation horizon arrays; test paths are never constructed.

    This entry point is intentionally not invoked by this module. The output
    path must be new. Training cutoff fitting is delegated to the reader-bound
    v2 API with an injected list containing only the 8,020 training paths.
    """
    repo = canonical_repository_root(repository_root)
    output = checked_absolute(output_root, allow_missing=True)
    processed = repo / "data" / "processed"
    suffix = output.name[len(OUTPUT_PREFIX) :] if output.name.startswith(OUTPUT_PREFIX) else ""
    if output.parent != processed or not suffix or not all(c.isalnum() or c in "-_" for c in suffix):
        raise ValueError(f"Output must be a direct versioned child of {processed}")
    if not processed.is_dir() or processed.resolve(strict=True) != processed:
        raise ValueError(f"Processed-data parent is missing or redirected: {processed}")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite horizon-v2 output: {output}")
    raw = repo / RAW_DIRECTORY_RELATIVE if raw_directory is None else Path(raw_directory)
    topology = repo / TOPOLOGY_RELATIVE if topology_path is None else Path(topology_path)
    selected_paths = expected_split_paths_for_splits(
        raw, ("train", "validation"), repository_root=repo
    )
    train_paths = selected_paths["train"]
    validation_paths = selected_paths["validation"]
    nodes, _undirected, edges, topology_hash = _topology_snapshot(topology, repo)
    ordered_nodes = tuple(sorted(nodes))
    node_index = {node: index for index, node in enumerate(ordered_nodes)}
    routes = _routing_paths(nodes, edges)

    parser = v1_validity.parse_one_training_matrix
    training_read = v1_validity.read_training_period(
        raw,
        ordered_nodes,
        node_index,
        expected_start_timestamp=EXPECTED_START,
        expected_start_timestamp_source=EXPECTED_START_SOURCE,
        discovery=lambda _directory: train_paths,
        parser=parser,
    )
    validation_observations = tuple(parser(path, ordered_nodes, node_index) for path in validation_paths)
    for observation, path in zip(validation_observations, validation_paths):
        if observation.filename != path.name or observation.timestamp != v1_validity.timestamp_from_filename(path):
            raise ValueError(f"Parsed validation record did not match selected filename: {path.name}")

    split_data = {
        "train": _build_split_arrays("train", training_read.observations, training_read.cutoff_mbps,
                                     ordered_nodes, edges, routes),
        "validation": _build_split_arrays("validation", validation_observations, training_read.cutoff_mbps,
                                          ordered_nodes, edges, routes),
    }
    metadata: dict[str, object] = {
        "protocol_id": PROTOCOL_ID,
        "status": "incomplete",
        "source_directory": str(raw.relative_to(repo)),
        "source_scope": {split: list(OFFICIAL_SPLIT_BOUNDS[split]) for split in ("train", "validation")},
        "anchor_timestamp": EXPECTED_START.strftime("%Y%m%d-%H%M"),
        "anchor_provenance": EXPECTED_START_SOURCE,
        "local_source_archive_checksum_verified": False,
        "source_hashes_at_build": _current_build_source_hashes(repo),
        "source_hash_limitation": "hashes identify file bytes at build time but do not prove runtime import identity",
        "build_configuration": BUILD_CONFIGURATION,
        "validity_protocol_id": training_read.metadata["protocol_id"],
        "training_cutoff_mbps": training_read.cutoff_mbps,
        "training_cutoff_nonempty_count": training_read.metadata["cutoff_fit_nonempty_count"],
        "training_cutoff_filenames": training_read.cutoff_fit_filenames,
        "topology": {
            "path": str(topology.relative_to(repo)),
            "sha256_local_file_identity_only": topology_hash,
            "provenance": "upstream identity not independently authenticated",
            "nodes": list(ordered_nodes),
            "undirected_links": EXPECTED_UNDIRECTED_LINKS,
            "directed_edges_interleaved": [list(edge) for edge in edges],
            "ordered_edge_sha256": _edge_digest(edges),
            "directed_links": len(edges),
            "capacity_mbps_assumed": CAPACITY_MBPS,
            "routing": "distance-shortest-path using haversine distance; existing lexicographic tie-break",
            "utilization_semantics": "computed link load divided by assumed capacity; not measured telemetry",
        },
        "window": {"history_length": HISTORY_LENGTH, "horizons": list(HORIZONS), "step_seconds": STEP_SECONDS},
        "ewma": {"alpha": EWMA_ALPHA, "steps": EWMA_STEPS, "horizon_forecast": "flat carry-forward"},
        "splits": {},
        "artifact_names": [],
    }
    metadata["build_configuration_sha256"] = hashlib.sha256(
        json.dumps(metadata["build_configuration"], sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    split_observations = {"train": training_read.observations, "validation": validation_observations}
    for split, data in split_data.items():
        split_record: dict[str, object] = {
            "bounds": list(OFFICIAL_SPLIT_BOUNDS[split]),
            "source_file_names": [item.filename for item in split_observations[split]],
            "parsed_source_semantic_sha256": _observation_fingerprint(split_observations[split]),
            "validity_counts": {
                "total": len(split_observations[split]),
                "valid": sum(not v1_validity.invalidity_reason(item, training_read.cutoff_mbps) for item in split_observations[split]),
                "invalid": sum(bool(v1_validity.invalidity_reason(item, training_read.cutoff_mbps)) for item in split_observations[split]),
                "empty": sum(v1_validity.is_empty_matrix_v2(item) for item in split_observations[split]),
            },
            "horizons": {},
        }
        for horizon in HORIZONS:
            xname, yname = f"history_{split}_h{horizon}.npy", f"target_{split}_h{horizon}.npy"
            validate_artifact_name(xname)
            validate_artifact_name(yname)
            x, y = data.histories[horizon], data.targets[horizon]
            split_record["horizons"][str(horizon)] = {
                "window_count": int(x.shape[0]), "history_shape": list(x.shape), "target_shape": list(y.shape),
                "dtype": str(x.dtype), "history_sha256": hashlib.sha256(x.tobytes()).hexdigest(),
                "target_sha256": hashlib.sha256(y.tobytes()).hexdigest(),
                "window_index_file": f"window_index_{split}.json",
                "alignment": "history=[origin-671,...,origin], target=origin+horizon",
                "window_validity": "all history steps and target valid; timestamps consecutive through target; split-local",
            }
            metadata["artifact_names"].extend((xname, yname))
        metadata["splits"][split] = split_record
        metadata["artifact_names"].append(f"window_index_{split}.json")

    metadata["artifact_names"].append("metadata.json")
    # All parsing and validation completes before the output directory is made.
    output.mkdir(parents=False, exist_ok=False)
    try:
        checked_output = checked_absolute(output)
        if checked_output != output or checked_output.resolve(strict=True) != output or output.parent.resolve(strict=True) != processed:
            raise ValueError("New horizon-v2 output directory failed post-creation containment")
        _atomic_json(output / "metadata.json", metadata)
        for split, data in split_data.items():
            starts = {
                str(horizon): [
                    {"history_start": window.history_start, "origin": window.origin, "target": window.target}
                    for window in data.windows[horizon]
                ]
                for horizon in HORIZONS
            }
            _atomic_json(output / f"window_index_{split}.json", starts)
            for horizon in HORIZONS:
                np.save(output / f"history_{split}_h{horizon}.npy", data.histories[horizon], allow_pickle=False)
                np.save(output / f"target_{split}_h{horizon}.npy", data.targets[horizon], allow_pickle=False)
        for split, data in split_data.items():
            for horizon in HORIZONS:
                for kind, expected in (("history", data.histories[horizon]), ("target", data.targets[horizon])):
                    path = output / f"{kind}_{split}_h{horizon}.npy"
                    saved = np.load(path, allow_pickle=False)
                    if saved.shape != expected.shape or saved.dtype != expected.dtype:
                        raise ValueError(f"Written {split} h={horizon} {kind} schema mismatch")
                    if not np.array_equal(saved, expected):
                        raise ValueError(f"Written {split} h={horizon} {kind} data mismatch")
        metadata["status"] = "complete"
        _atomic_json(output / "metadata.json", metadata)
    except BaseException as exc:
        metadata["status"] = "failed"
        metadata["failure"] = f"{type(exc).__name__}: {exc}"
        try:
            _atomic_json(output / "metadata.json", metadata)
        except BaseException:
            pass
        raise
    return metadata


__all__ = [
    "BUILD_CONFIGURATION", "CAPACITY_MBPS", "EXPECTED_START", "HORIZONS", "HISTORY_LENGTH", "OFFICIAL_SPLIT_BOUNDS",
    "PROTOCOL_ID", "SplitArrays", "build_train_validation", "expected_split_paths",
    "expected_split_paths_for_splits",
    "load_horizon_array", "validate_artifact_name",
]
