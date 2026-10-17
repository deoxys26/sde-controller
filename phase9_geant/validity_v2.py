"""Isolated, training-only GEANT validity protocol v2 primitives.

This module intentionally does not import the legacy all-series parser or any
processed-data loader. The directory reader sorts filenames by their embedded
15-minute timestamps, validates/selects the training prefix, and only then
opens the selected files.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
from phase9_geant.build_link_series import compute_link_series

PROTOCOL_ID = "geant-validity-v2-train-median"
EXPECTED_START = datetime(2005, 5, 4, 15, 0)
TRAIN_START, TRAIN_END = 0, 8020
VALIDATION_START, VALIDATION_END = 8020, 9740
TEST_START, TEST_END = 9740, 11460
SPLIT_BOUNDS = {
    "train": (TRAIN_START, TRAIN_END),
    "validation": (VALIDATION_START, VALIDATION_END),
    "test": (TEST_START, TEST_END),
}
MATRIX_GLOB = "demandMatrix-geant-uhlig-15min-*.txt"
TIMESTAMP_PATTERN = re.compile(
    r"^demandMatrix-geant-uhlig-15min-(\d{8}-\d{4})\.txt$"
)
STEP = timedelta(minutes=15)
TOTAL_DEMAND_MULTIPLIER = 5.0
INDIVIDUAL_DEMAND_LIMIT_MBPS = 20_000.0
CAPACITY_MBPS = 40_000.0
HISTORY_STEPS = 5
PREDICTION_HORIZON = 1


@dataclass(frozen=True)
class MatrixObservation:
    """One parsed matrix; absent OD pairs are zero, empty means count is zero."""

    timestamp: datetime
    matrix_mbps: np.ndarray
    demand_count: int
    filename: str

    @property
    def total_demand_mbps(self) -> float:
        return float(np.sum(self.matrix_mbps, dtype=np.float64))


@dataclass(frozen=True)
class TrainingRead:
    observations: tuple[MatrixObservation, ...]
    cutoff_mbps: float
    cutoff_fit_filenames: tuple[str, ...]
    metadata: Mapping[str, object]


@dataclass(frozen=True)
class TrainingFileSelection:
    """Validated filename-only provenance for the fixed training prefix."""

    paths: tuple[Path, ...]
    timestamps: tuple[datetime, ...]
    expected_start_timestamp: datetime
    expected_start_timestamp_source: str


def timestamp_from_filename(path: str | Path) -> datetime:
    """Parse only the documented filename timestamp; never opens the file."""
    name = Path(path).name
    match = TIMESTAMP_PATTERN.fullmatch(name)
    if match is None:
        raise ValueError(f"Unexpected GEANT matrix filename: {name!r}")
    return datetime.strptime(match.group(1), "%Y%m%d-%H%M")


def select_training_files(
    paths: Iterable[str | Path],
    *,
    expected_start_timestamp: datetime,
    expected_start_timestamp_source: str,
    training_end: int = TRAIN_END,
) -> TrainingFileSelection:
    """Order filenames, enforce a sourced start anchor, then select the prefix.

    ``training_end`` exists to make focused synthetic unit tests small; the
    production reader fixes the protocol boundary at 8,020 files. The expected
    start and its provenance are always explicit; neither is inferred from the
    files present in the directory.
    """
    if not isinstance(training_end, int) or training_end <= 0:
        raise ValueError("training_end must be a positive integer")
    if not isinstance(expected_start_timestamp, datetime):
        raise ValueError("expected_start_timestamp must be explicitly supplied")
    if not isinstance(expected_start_timestamp_source, str) or not expected_start_timestamp_source.strip():
        raise ValueError("A provenance source for the expected start is required")
    dated = [(timestamp_from_filename(path), Path(path)) for path in paths]
    dated.sort(key=lambda item: item[0])
    for previous, current in zip(dated, dated[1:]):
        if current[0] == previous[0]:
            raise ValueError(f"Duplicate matrix timestamp: {current[0]}")
        if current[0] - previous[0] != STEP:
            raise ValueError(
                f"Matrix chronology gap between {previous[1].name} and "
                f"{current[1].name}: expected {STEP}, "
                f"observed {current[0] - previous[0]}"
            )
    if len(dated) < training_end:
        raise ValueError(
            f"Insufficient training files: need {training_end}, found {len(dated)}"
        )
    if dated[0][0] != expected_start_timestamp:
        raise ValueError(
            "Training chronology start mismatch: "
            f"expected {expected_start_timestamp}, found {dated[0][0]}"
        )
    selected = dated[:training_end]
    previous: datetime | None = None
    for timestamp, path in selected:
        if previous is not None:
            if timestamp == previous:
                raise ValueError(f"Duplicate training timestamp: {timestamp}")
            if timestamp - previous != STEP:
                raise ValueError(
                    f"Training chronology gap before {path.name}: "
                    f"expected {STEP}, observed {timestamp - previous}"
                )
        previous = timestamp
    return TrainingFileSelection(
        paths=tuple(path for _, path in selected),
        timestamps=tuple(timestamp for timestamp, _ in selected),
        expected_start_timestamp=expected_start_timestamp,
        expected_start_timestamp_source=expected_start_timestamp_source.strip(),
    )


def parse_one_training_matrix(
    path: str | Path,
    nodes: Sequence[str],
    node_to_index: Mapping[str, int],
) -> MatrixObservation:
    """Parse exactly one already-selected training matrix file."""
    path = Path(path)
    filename_timestamp = timestamp_from_filename(path)
    try:
        content = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Unreadable training matrix {path}: {exc}") from exc

    meta_match = re.search(r"time\s*=\s*([\w-]+)", content)
    if meta_match is None:
        raise ValueError(f"No META time= in {path.name!r}")
    timestamp_text = meta_match.group(1)
    timestamp = datetime.strptime(timestamp_text, "%Y%m%d-%H%M")
    if timestamp != filename_timestamp:
        raise ValueError(
            f"Filename/META timestamp mismatch in {path.name!r}: "
            f"{filename_timestamp} != {timestamp}"
        )

    matrix = np.zeros((len(nodes), len(nodes)), dtype=np.float64)
    count = 0
    inside_demands = False
    demand_section_found = False
    demand_section_closed = False
    seen_od_pairs: set[tuple[str, str]] = set()
    demand_pattern = re.compile(
        r"^\S+\s+\(\s*(\S+)\s+(\S+)\s*\)\s+(\S+)\s+([0-9.eE+-]+)\s+(\S+)\s*$"
    )
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("DEMANDS ("):
            if demand_section_found:
                raise ValueError(f"Duplicate DEMANDS section in {path.name!r}")
            demand_section_found = True
            inside_demands = True
            continue
        if line.startswith("ADMISSIBLE_PATHS"):
            if inside_demands:
                raise ValueError(f"Unclosed DEMANDS section in {path.name!r}")
            break
        if line == ")":
            if inside_demands:
                inside_demands = False
                demand_section_closed = True
            continue
        if not inside_demands:
            continue
        match = demand_pattern.search(line)
        if match is None:
            raise ValueError(f"Malformed demand row in {path.name!r}: {line!r}")
        source, target, routing_unit, value_text, max_hops = match.groups()
        if routing_unit != "1":
            raise ValueError(f"Unexpected routing unit in {path.name!r}")
        if max_hops != "UNLIMITED" and not max_hops.isdigit():
            raise ValueError(f"Unexpected max_hops field in {path.name!r}")
        if source not in node_to_index or target not in node_to_index:
            raise ValueError(f"Unknown node in demand entry in {path.name!r}")
        if (source, target) in seen_od_pairs:
            raise ValueError(f"Duplicate OD pair {(source, target)!r} in {path.name!r}")
        seen_od_pairs.add((source, target))
        value = float(value_text)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid demand value in {path.name!r}")
        matrix[node_to_index[source], node_to_index[target]] = value
        count += 1
    if not demand_section_found or not demand_section_closed:
        raise ValueError(f"Missing or incomplete DEMANDS section in {path.name!r}")
    return MatrixObservation(timestamp, matrix, count, path.name)


def read_training_period(
    raw_directory: str | Path,
    nodes: Sequence[str],
    node_to_index: Mapping[str, int],
    *,
    expected_start_timestamp: datetime,
    expected_start_timestamp_source: str,
    discovery: Callable[[Path], Iterable[str | Path]] | None = None,
    parser: Callable[[str | Path, Sequence[str], Mapping[str, int]], MatrixObservation]
    = parse_one_training_matrix,
) -> TrainingRead:
    """Select, parse, validate, and fit from the anchored training files only.

    ``discovery`` and ``parser`` are dependency-injection seams for synthetic
    tests. Production callers should leave them unset. Cutoff construction is
    deliberately kept behind this entry point so callers cannot pass an
    arbitrary observation list to a public fitting function.
    """
    raw_directory = Path(raw_directory)
    # Discovery examines names only. No file content is opened until the
    # selected training prefix has passed timestamp, duplicate, and gap checks.
    discover = _discover_matrix_paths if discovery is None else discovery
    discovered = discover(raw_directory)
    selection = select_training_files(
        discovered,
        expected_start_timestamp=expected_start_timestamp,
        expected_start_timestamp_source=expected_start_timestamp_source,
        training_end=TRAIN_END,
    )
    observations = tuple(parser(path, nodes, node_to_index) for path in selection.paths)
    _validate_parsed_observations(observations, selection)
    return _fit_training_cutoff(observations, selection)


def _discover_matrix_paths(raw_directory: Path) -> tuple[Path, ...]:
    """List candidate matrix paths by filename only; never reads file contents."""
    prefix = MATRIX_GLOB.removesuffix("*.txt")
    return tuple(
        entry for entry in raw_directory.iterdir()
        if entry.is_file() and entry.name.startswith(prefix)
    )


def _validate_parsed_observations(
    observations: Sequence[MatrixObservation], selection: TrainingFileSelection
) -> None:
    """Require parser outputs to correspond positionally to selected files."""
    if len(observations) != len(selection.paths):
        raise ValueError("Parser did not return exactly one observation per selected file")
    for index, (observation, path, timestamp) in enumerate(
        zip(observations, selection.paths, selection.timestamps)
    ):
        if observation.filename != path.name:
            raise ValueError(
                f"Parsed filename mismatch at training index {index}: "
                f"{observation.filename!r} != {path.name!r}"
            )
        if observation.timestamp != timestamp:
            raise ValueError(
                f"Parsed timestamp mismatch at training index {index}: "
                f"{observation.timestamp!r} != {timestamp!r}"
            )


def _fit_training_cutoff(
    observations: Sequence[MatrixObservation],
    selection: TrainingFileSelection,
) -> TrainingRead:
    """Private arithmetic helper; call only after the validated reader."""
    if not isinstance(selection, TrainingFileSelection):
        raise ValueError("Validated TrainingFileSelection provenance is required")
    if len(selection.paths) != TRAIN_END or len(observations) != TRAIN_END:
        raise ValueError(
            f"Production cutoff fitting requires exactly {TRAIN_END} training files"
        )
    validated_selection = select_training_files(
        selection.paths,
        expected_start_timestamp=selection.expected_start_timestamp,
        expected_start_timestamp_source=selection.expected_start_timestamp_source,
    )
    if validated_selection != selection:
        raise ValueError("TrainingFileSelection provenance failed revalidation")
    _validate_parsed_observations(observations, selection)
    totals: list[float] = []
    fit_names: list[str] = []
    for observation in observations:
        if observation.demand_count <= 0:
            continue
        total = observation.total_demand_mbps
        if not math.isfinite(total) or total < 0:
            raise ValueError(f"Invalid total demand in {observation.filename!r}")
        totals.append(total)
        fit_names.append(observation.filename)
    if not totals:
        raise ValueError("Training period has no nonempty matrices")
    median_total = float(np.median(np.asarray(totals, dtype=np.float64)))
    cutoff = TOTAL_DEMAND_MULTIPLIER * median_total
    metadata: dict[str, object] = {
        "protocol_id": PROTOCOL_ID,
        "training_boundary": [TRAIN_START, TRAIN_END],
        "training_indices": list(range(TRAIN_START, TRAIN_END)),
        "expected_start_timestamp": selection.expected_start_timestamp.strftime("%Y%m%d-%H%M"),
        "expected_start_timestamp_source": selection.expected_start_timestamp_source,
        "threshold_scope": "nonempty training matrices only",
        "total_demand_median_mbps": median_total,
        "total_demand_multiplier": TOTAL_DEMAND_MULTIPLIER,
        "total_demand_cutoff_mbps": cutoff,
        "total_demand_comparison": "strictly greater than cutoff is invalid",
        "individual_demand_limit_mbps": INDIVIDUAL_DEMAND_LIMIT_MBPS,
        "individual_demand_comparison": "strictly greater than limit is invalid",
        "cutoff_fit_filenames": list(fit_names),
        "cutoff_fit_nonempty_count": len(fit_names),
        "training_file_count": len(observations),
        "empty_matrices_invalid": True,
        "invalid_link_utilization": "NaN",
        "history_steps": HISTORY_STEPS,
        "prediction_horizon": PREDICTION_HORIZON,
        "split_bounds": {name: list(bounds) for name, bounds in SPLIT_BOUNDS.items()},
    }
    return TrainingRead(tuple(observations), cutoff, tuple(fit_names), metadata)


def is_empty_matrix_v2(observation: MatrixObservation) -> bool:
    """Versioned empty-observation check based on present DEMANDS entries."""
    return observation.demand_count == 0


def exceeds_total_demand_cutoff_v2(
    observation: MatrixObservation, total_demand_cutoff_mbps: float
) -> bool:
    """Use the protocol's strict greater-than comparison for total demand."""
    return observation.total_demand_mbps > total_demand_cutoff_mbps


def exceeds_individual_demand_limit_v2(observation: MatrixObservation) -> bool:
    """Apply the fixed, strict per-OD demand limit."""
    return bool(
        observation.matrix_mbps.size
        and float(np.max(observation.matrix_mbps)) > INDIVIDUAL_DEMAND_LIMIT_MBPS
    )


def invalidity_reason(
    observation: MatrixObservation, total_demand_cutoff_mbps: float
) -> tuple[str, ...]:
    """Return invalidity reasons; all comparisons preserve the legacy strict >."""
    reasons: list[str] = []
    if is_empty_matrix_v2(observation):
        reasons.append("empty")
    if exceeds_total_demand_cutoff_v2(observation, total_demand_cutoff_mbps):
        reasons.append("total_demand")
    if exceeds_individual_demand_limit_v2(observation):
        reasons.append("individual_demand")
    return tuple(reasons)


def build_link_series_v2(
    observations: Sequence[MatrixObservation],
    directed_edges: Sequence[tuple[str, str]],
    paths: Mapping[tuple[str, str], Sequence[str]],
    total_demand_cutoff_mbps: float,
    *,
    nodes: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, tuple[tuple[str, ...], ...]]:
    """Compute directed-link Mbps/utilization, retaining invalid rows as NaN."""
    reasons: list[tuple[str, ...]] = []
    flagged_indices: set[int] = set()
    parsed_rows = []
    for row, observation in enumerate(observations):
        reason = invalidity_reason(observation, total_demand_cutoff_mbps)
        reasons.append(reason)
        if reason and observation.demand_count > 0:
            flagged_indices.add(row)
        parsed_rows.append(
            (observation.timestamp, observation.matrix_mbps, observation.demand_count, observation.filename)
        )
    # Reuse the existing pure link-load accumulator with only the caller's
    # explicitly supplied records. It does no file I/O or split loading.
    traffic, utilization, _, _, _ = compute_link_series(
        parsed_rows, list(nodes), list(directed_edges), dict(paths), flagged_indices
    )
    return traffic, utilization, tuple(reasons)


def valid_window_starts(
    timestamps: Sequence[datetime],
    valid_mask: Sequence[bool],
    *,
    split_start: int,
    split_end: int,
) -> tuple[int, ...]:
    """Return start indices for valid consecutive six-step windows inside a split."""
    if len(timestamps) != len(valid_mask):
        raise ValueError("timestamps and valid_mask must have equal lengths")
    if (split_start, split_end) not in SPLIT_BOUNDS.values():
        raise ValueError(
            "Window bounds must exactly match one declared official split"
        )
    if not (0 <= split_start <= split_end <= len(timestamps)):
        raise ValueError("Invalid split bounds")
    window_size = HISTORY_STEPS + PREDICTION_HORIZON
    valid_starts: list[int] = []
    for start in range(split_start, split_end - window_size + 1):
        end = start + window_size
        if not all(bool(value) for value in valid_mask[start:end]):
            continue
        if any(
            timestamps[index] - timestamps[index - 1] != STEP
            for index in range(start + 1, end)
        ):
            continue
        valid_starts.append(start)
    return tuple(valid_starts)


def validate_output_root(
    output_root: str | Path,
    *,
    project_root: str | Path,
) -> Path:
    """Validate the original v2 root or a direct, versioned rebuild sibling.

    Component checks happen before resolution, so an existing redirected
    ancestor cannot be concealed by ``Path.resolve``. Callers remain
    responsible for refusing an existing destination before writing.
    """
    from phase9_geant.safe_paths import canonical_repository_root, checked_absolute

    root = canonical_repository_root(project_root)
    candidate = Path(output_root)
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = checked_absolute(candidate, allow_missing=True)
    processed = root / "data" / "processed"
    original = processed / "geant_validity_v2"
    version_prefix = "geant_validity_v2_rebuild_"
    valid_name = (
        candidate.name.startswith(version_prefix)
        and len(candidate.name) > len(version_prefix)
        and all(character.isalnum() or character in "_-" for character in candidate.name)
    )
    if candidate != original and not (candidate.parent == processed and valid_name):
        raise ValueError(
            "Output root must be the original v2 root or a direct, explicitly versioned rebuild child"
        )
    if processed.resolve(strict=True) != processed:
        raise ValueError("Processed-data parent is not canonical")
    legacy = processed / "geant"
    try:
        candidate.relative_to(legacy)
    except ValueError:
        pass
    else:
        raise ValueError(f"Output root must not be the legacy tree or its descendant: {candidate}")
    return candidate


def metadata_json(training_read: TrainingRead) -> str:
    """Stable JSON serialization helper for protocol metadata tests/callers."""
    return json.dumps(training_read.metadata, sort_keys=True, separators=(",", ":"))
