"""Build isolated GEANT validity-v2 preprocessing artifacts.

This command performs preprocessing only. It fits the validity cutoff from
the anchored training prefix, then applies that fixed cutoff to the complete
source series to create computed directed-link utilization and split-local
forecasting windows. It never writes to the legacy processed-data directory.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from phase9_geant import validity_v2 as v2
from phase9_geant.build_link_series import (
    CAPACITY_MBPS,
    CAPACITY_SOURCE,
    compute_all_pairs_paths,
    haversine,
)
from phase9_geant.parse_geant import parse_topology
from phase9_geant.window_provenance_v2 import (
    PROVENANCE_FILENAME,
    build_train_validation_provenance,
)
from phase9_geant.split_access import load_permitted_split_array

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data/raw/geant_topology/directed-geant-uhlig-15min-over-4months-ALL-native"
TOPOLOGY_FILE = ROOT / "data/raw/geant_topology/geant.txt"
OUTPUT_REQUESTED = ROOT / "data/processed/geant_validity_v2"
EXPECTED_START = v2.EXPECTED_START
EXPECTED_START_SOURCE = (
    "Official SNDlib GEANT overview: https://sndlib.put.poznan.pl/geant.overview.action; "
    "official GEANT readme: https://sndlib.put.poznan.pl/download/geant-rm.txt; "
    "intended archive: directed-geant-uhlig-15min-over-4months-ALL-native.tgz; "
    "local archive not checksum-verified against upstream"
)
EXPECTED_SOURCE_FILES = 11_460
ALL_SPLITS = ("train", "validation", "test")


def parse_requested_splits(value: str | None) -> tuple[str, ...]:
    """Parse the optional output-split restriction; default retains legacy layout."""
    if value is None:
        return ALL_SPLITS
    raw = tuple(part.strip().casefold() for part in value.split(",") if part.strip())
    selected = tuple("validation" if part == "val" else part for part in raw)
    if selected not in (("train", "validation"), ALL_SPLITS):
        raise ValueError("splits must be exactly 'train,validation' or 'train,validation,test'")
    return selected


def _json_write(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _windows_for_split(
    utilization: np.ndarray, starts: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray]:
    history = np.stack(
        [utilization[start : start + v2.HISTORY_STEPS] for start in starts], axis=0
    ).astype(np.float32, copy=False)
    target = np.stack(
        [utilization[start + v2.HISTORY_STEPS] for start in starts], axis=0
    ).astype(np.float32, copy=False)
    return history, target


def _write_and_verify_window_arrays(
    output_root: Path,
    window_arrays: dict[str, tuple[np.ndarray, np.ndarray]],
) -> None:
    """Write all split arrays, then read back only the permitted train/val set."""
    for split_name, (history, target) in window_arrays.items():
        np.save(output_root / f"X_{split_name}.npy", history, allow_pickle=False)
        np.save(output_root / f"y_{split_name}.npy", target, allow_pickle=False)

    # Test arrays are part of the pre-existing dataset layout and are written
    # above, but intentionally are never reopened for storage verification.
    for split_name in ("train", "validation"):
        history, target = window_arrays[split_name]
        for prefix, expected in (("X", history), ("y", target)):
            filename = f"{prefix}_{split_name}.npy"
            saved = load_permitted_split_array(output_root, filename)
            if saved.shape != expected.shape or saved.dtype != expected.dtype:
                raise ValueError(f"Saved {split_name} {prefix} schema mismatch")
            if not np.array_equal(saved, expected, equal_nan=True):
                raise ValueError(f"Saved {split_name} {prefix} values mismatch")


def build(
    *,
    output_root_requested: Path = OUTPUT_REQUESTED,
    splits: tuple[str, ...] = ALL_SPLITS,
) -> dict[str, object]:
    """Validate, preprocess, save, and verify the isolated v2 artifact set."""
    splits = parse_requested_splits(",".join(splits))
    if splits not in (("train", "validation"), ALL_SPLITS):
        raise ValueError("Builder requires train and validation; optional test output is all-or-none")
    if not RAW_DIR.is_dir():
        raise FileNotFoundError(f"GEANT source directory does not exist: {RAW_DIR}")
    if not TOPOLOGY_FILE.is_file():
        raise FileNotFoundError(f"GEANT topology file does not exist: {TOPOLOGY_FILE}")

    output_root = v2.validate_output_root(output_root_requested, project_root=ROOT)
    if output_root.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing v2 output directory: {output_root}"
        )

    # Name-only preflight: no matrix content is opened before the exact count,
    # upstream anchor, duplicate/gap checks, and training prefix are verified.
    source_paths = v2._discover_matrix_paths(RAW_DIR)
    if len(source_paths) != EXPECTED_SOURCE_FILES:
        raise ValueError(
            f"Expected exactly {EXPECTED_SOURCE_FILES} source filenames, "
            f"found {len(source_paths)}"
        )
    selection = v2.select_training_files(
        source_paths,
        expected_start_timestamp=EXPECTED_START,
        expected_start_timestamp_source=EXPECTED_START_SOURCE,
    )
    if len(selection.paths) != v2.TRAIN_END:
        raise ValueError("Validated training selection has an unexpected length")

    # Topology-only preflight and distance-shortest-path construction.
    nodes, undirected_links = parse_topology(str(TOPOLOGY_FILE))
    if len(nodes) != 22 or len(undirected_links) != 36:
        raise ValueError(
            f"Unexpected topology dimensions: {len(nodes)} nodes, "
            f"{len(undirected_links)} undirected links"
        )
    nodes_list = sorted(nodes)
    node_to_index = {name: index for index, name in enumerate(nodes_list)}
    directed_edges = [edge for u, w in undirected_links for edge in ((u, w), (w, u))]
    if len(directed_edges) != 72:
        raise ValueError(f"Expected 72 directed links, found {len(directed_edges)}")

    def distance_weight(source: str, target: str) -> float:
        lon1, lat1 = nodes[source]
        lon2, lat2 = nodes[target]
        return haversine(lon1, lat1, lon2, lat2)

    paths = compute_all_pairs_paths(nodes, directed_edges, distance_weight)
    if len(paths) != len(nodes) * (len(nodes) - 1):
        raise ValueError("Topology routing did not produce all ordered node pairs")

    # Fit cutoff through the reader-controlled API. This call parses only the
    # selected 8,020 training files after selection/chronology validation.
    training_read = v2.read_training_period(
        RAW_DIR,
        nodes_list,
        node_to_index,
        expected_start_timestamp=EXPECTED_START,
        expected_start_timestamp_source=EXPECTED_START_SOURCE,
        discovery=lambda _directory: source_paths,
    )
    if len(training_read.observations) != v2.TRAIN_END:
        raise ValueError("Training reader returned an unexpected observation count")
    if tuple(item.filename for item in training_read.observations) != tuple(
        path.name for path in selection.paths
    ):
        raise ValueError("Training parser output does not match the preflight selection")

    # Apply the frozen training cutoff to the whole series. Holdout files are
    # parsed only after the training cutoff has been fitted and are never used
    # to refit or alter that cutoff.
    observations_by_name = {
        item.filename: item for item in training_read.observations
    }
    ordered_paths = sorted(source_paths, key=v2.timestamp_from_filename)
    for path in ordered_paths[v2.TRAIN_END :]:
        observation = v2.parse_one_training_matrix(path, nodes_list, node_to_index)
        expected_timestamp = v2.timestamp_from_filename(path)
        if observation.filename != path.name or observation.timestamp != expected_timestamp:
            raise ValueError(f"Parsed source record does not match filename: {path.name}")
        observations_by_name[path.name] = observation
    if len(observations_by_name) != EXPECTED_SOURCE_FILES:
        raise ValueError("Parsed observation set does not match the source file set")
    observations = tuple(observations_by_name[path.name] for path in ordered_paths)
    timestamps = tuple(item.timestamp for item in observations)
    if timestamps[0] != EXPECTED_START:
        raise ValueError("Parsed series start does not match the documented anchor")
    if any(
        current - previous != v2.STEP
        for previous, current in zip(timestamps, timestamps[1:])
    ):
        raise ValueError("Parsed META timestamps are not consecutive 15-minute steps")

    traffic, utilization, reason_rows = v2.build_link_series_v2(
        observations,
        directed_edges,
        paths,
        training_read.cutoff_mbps,
        nodes=nodes_list,
    )
    valid_mask = np.asarray([not reasons for reasons in reason_rows], dtype=np.bool_)
    expected_shape = (EXPECTED_SOURCE_FILES, len(directed_edges))
    if traffic.shape != expected_shape or utilization.shape != expected_shape:
        raise ValueError(
            f"Unexpected link-series shapes: traffic={traffic.shape}, "
            f"utilization={utilization.shape}, expected={expected_shape}"
        )
    if valid_mask.shape != (EXPECTED_SOURCE_FILES,):
        raise ValueError(f"Unexpected validity-mask shape: {valid_mask.shape}")
    if not np.isnan(utilization[~valid_mask]).all() or not np.isnan(traffic[~valid_mask]).all():
        raise ValueError("Invalid observations were not preserved as NaN")
    if not np.isfinite(utilization[valid_mask]).all() or not np.isfinite(traffic[valid_mask]).all():
        raise ValueError("A valid observation contains non-finite link values")

    split_windows: dict[str, tuple[int, ...]] = {}
    split_bounds = {name: v2.SPLIT_BOUNDS[name] for name in splits}
    for split_name, (start, end) in split_bounds.items():
        split_windows[split_name] = v2.valid_window_starts(
            timestamps, valid_mask, split_start=start, split_end=end
        )
    window_arrays = {
        name: _windows_for_split(utilization, starts)
        for name, starts in split_windows.items()
    }
    # This separate sidecar is deliberately restricted to train/validation.
    # It binds their arrays to exact start/target indices and validity metadata;
    # it does not serialize the sealed test starts or test array fingerprints.
    train_validation_provenance = build_train_validation_provenance(
        timestamps, valid_mask, split_windows, window_arrays
    )
    for split_name, (history, target) in window_arrays.items():
        if history.shape[1:] != (v2.HISTORY_STEPS, len(directed_edges)):
            raise ValueError(f"Unexpected history shape for {split_name}: {history.shape}")
        if target.shape != (len(split_windows[split_name]), len(directed_edges)):
            raise ValueError(f"Unexpected target shape for {split_name}: {target.shape}")
        if not np.isfinite(history).all() or not np.isfinite(target).all():
            raise ValueError(f"A {split_name} window contains an invalid observation")

    included_indices = range(0, split_bounds[splits[-1]][1])
    included_reasons = [reason_rows[index] for index in included_indices]
    included_valid = valid_mask[: split_bounds[splits[-1]][1]]
    invalid_counts = {
        "scope": "all observations" if splits == ALL_SPLITS else "train and validation observations only",
        "total": int((~included_valid).sum()),
        "empty": sum("empty" in reasons for reasons in included_reasons),
        "total_demand": sum("total_demand" in reasons for reasons in included_reasons),
        "individual_demand": sum("individual_demand" in reasons for reasons in included_reasons),
        "empty_and_threshold_flagged": sum(
            "empty" in reasons and len(reasons) > 1 for reasons in included_reasons
        ),
        "valid": int(included_valid.sum()),
    }
    split_counts = {
        name: {
            "bounds": list(split_bounds[name]),
            "invalid_observations": int((~valid_mask[slice(*split_bounds[name])]).sum()),
            "empty_observations": sum(
                "empty" in reasons
                for reasons in reason_rows[slice(*split_bounds[name])]
            ),
            "window_count": len(split_windows[name]),
        }
        for name in split_bounds
    }
    artifact_names: list[str] = []
    if splits == ALL_SPLITS:
        artifact_names.extend([
            "link_series_traffic.npy", "link_series_utilization.npy", "valid_mask.npy",
            "timestamps.json", "invalid_reasons.json",
        ])
    artifact_names.extend([
        "path_table_distance.json",
        "window_starts.json",
        PROVENANCE_FILENAME,
        "X_train.npy",
        "y_train.npy",
        "X_validation.npy",
        "y_validation.npy",
    ])
    if splits == ALL_SPLITS:
        artifact_names.extend(["X_test.npy", "y_test.npy"])
    artifact_names.append("metadata.json")
    metadata = dict(training_read.metadata)
    metadata.update(
        {
            "source_directory": str(RAW_DIR.relative_to(ROOT)),
            "source_file_count": len(source_paths),
            "source_archive": "directed-geant-uhlig-15min-over-4months-ALL-native.tgz",
            "local_archive_checksum_verified": False,
            "anchor_sources": [
                "https://sndlib.put.poznan.pl/geant.overview.action",
                "https://sndlib.put.poznan.pl/download/geant-rm.txt",
            ],
            "topology_file": str(TOPOLOGY_FILE.relative_to(ROOT)),
            "topology_assumptions": {
                "format": "SNDlib native format 1.0",
                "node_count": len(nodes_list),
                "undirected_link_count": len(undirected_links),
                "directed_link_count": len(directed_edges),
                "routing": "distance-shortest-path using haversine distance in km",
                "tie_breaking": "lexicographically smallest node sequence",
                "capacity_mbps": CAPACITY_MBPS,
                "capacity_source": CAPACITY_SOURCE,
                "utilization": "computed link Mbps divided by assumed capacity; not measured telemetry",
            },
            "selected_training_filenames": [path.name for path in selection.paths],
            "validity_counts": invalid_counts,
            "split_counts": split_counts,
            "window_definition": {
                "history_steps": v2.HISTORY_STEPS,
                "prediction_horizon": v2.PREDICTION_HORIZON,
                "requirements": [
                    "all six observations valid",
                    "consecutive 15-minute timestamps",
                    "all observations within one official split",
                ],
            },
            "array_metadata": {
                "link_series_traffic.npy": {"shape": list(traffic.shape), "dtype": str(traffic.dtype)},
                "link_series_utilization.npy": {"shape": list(utilization.shape), "dtype": str(utilization.dtype)},
                "valid_mask.npy": {"shape": list(valid_mask.shape), "dtype": str(valid_mask.dtype)},
                **{
                    filename: {"shape": list(array.shape), "dtype": str(array.dtype)}
                    for split_name, (history, target) in window_arrays.items()
                    for filename, array in (
                        (f"X_{split_name}.npy", history),
                        (f"y_{split_name}.npy", target),
                    )
                },
            },
            "generated_artifacts": artifact_names,
        }
    )

    # Last safe-path check before the first write. Refuse overwrite/race rather
    # than merging with or replacing an existing artifact tree.
    output_root = v2.validate_output_root(output_root_requested, project_root=ROOT)
    output_root.mkdir(parents=True, exist_ok=False)
    if splits == ALL_SPLITS:
        np.save(output_root / "link_series_traffic.npy", traffic, allow_pickle=False)
        np.save(output_root / "link_series_utilization.npy", utilization, allow_pickle=False)
        np.save(output_root / "valid_mask.npy", valid_mask, allow_pickle=False)
        _json_write(output_root / "timestamps.json", [stamp.isoformat() for stamp in timestamps])
        _json_write(output_root / "invalid_reasons.json", [list(row) for row in reason_rows])
    _json_write(
        output_root / "path_table_distance.json",
        {f"{src}→{dst}": path for (src, dst), path in paths.items()},
    )
    _json_write(
        output_root / "window_starts.json",
        {name: list(starts) for name, starts in split_windows.items()},
    )
    _json_write(output_root / PROVENANCE_FILENAME, train_validation_provenance)
    _write_and_verify_window_arrays(output_root, window_arrays)
    _json_write(output_root / "metadata.json", metadata)

    # Read back newly generated full-series arrays only in the default mode.
    # Train/validation-only mode deliberately writes none of those arrays.
    if splits == ALL_SPLITS:
        for filename, expected in (
            ("link_series_traffic.npy", traffic),
            ("link_series_utilization.npy", utilization),
            ("valid_mask.npy", valid_mask),
        ):
            saved = np.load(output_root / filename, allow_pickle=False)
            if saved.shape != expected.shape or saved.dtype != expected.dtype:
                raise ValueError(f"Saved artifact schema mismatch: {filename}")
            if not np.array_equal(saved, expected, equal_nan=True):
                raise ValueError(f"Saved artifact values mismatch: {filename}")
    with (output_root / "metadata.json").open(encoding="utf-8") as handle:
        saved_metadata = json.load(handle)
    if saved_metadata != metadata:
        raise ValueError("Saved metadata did not round-trip consistently")
    # Only train/validation arrays are reopened for storage verification.
    # Test arrays are written above for the dataset format, but never read back.
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_REQUESTED)
    parser.add_argument(
        "--splits",
        help="optional restriction; use train,val to omit test-window and full-series artifacts",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        splits = parse_requested_splits(args.splits)
    except ValueError as exc:
        parser.error(str(exc))
    metadata = build(output_root_requested=args.output_root, splits=splits)
    print(json.dumps({
        "output_directory": str(args.output_root),
        "source_file_count": metadata["source_file_count"],
        "training_file_count": metadata["training_file_count"],
        "cutoff_mbps": metadata["total_demand_cutoff_mbps"],
        "validity_counts": metadata["validity_counts"],
        "split_counts": metadata["split_counts"],
        "generated_artifacts": metadata["generated_artifacts"],
    }, indent=2))


if __name__ == "__main__":
    main()
