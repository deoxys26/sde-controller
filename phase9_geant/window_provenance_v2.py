"""Train/validation-only window provenance and array fingerprints for v2."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Mapping, Sequence

import numpy as np

from phase9_geant import validity_v2 as v2


PROVENANCE_FILENAME = "window_provenance_train_validation.json"
PROVENANCE_SCHEMA = 1
PROVENANCE_SPLITS = ("train", "validation")


def canonical_json_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def array_fingerprint(array: np.ndarray) -> str:
    """Bind array shape, dtype, and C-order bytes without writing data."""
    values = np.asarray(array)
    header = json.dumps(
        {"dtype": values.dtype.str, "shape": list(values.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\0")
    digest.update(np.ascontiguousarray(values).tobytes(order="C"))
    return digest.hexdigest()


def build_train_validation_provenance(
    timestamps: Sequence[datetime],
    valid_mask: Sequence[bool],
    split_windows: Mapping[str, Sequence[int]],
    window_arrays: Mapping[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, object]:
    """Build a sidecar containing no test-split keys or test-derived values."""
    splits: dict[str, object] = {}
    for name in PROVENANCE_SPLITS:
        start, end = v2.SPLIT_BOUNDS[name]
        if len(timestamps) < end or len(valid_mask) < end:
            raise ValueError(f"Insufficient timestamp/validity metadata for {name}")
        raw_starts = list(split_windows[name])
        if any(type(item) is not int for item in raw_starts):
            raise ValueError(f"{name} window starts must be integer indices")
        starts = list(raw_starts)
        official = v2.valid_window_starts(
            timestamps, valid_mask, split_start=start, split_end=end
        )
        if tuple(starts) != official:
            raise ValueError(f"{name} window starts differ from the official validity rule")
        history, target = window_arrays[name]
        if len(history) != len(starts) or len(target) != len(starts):
            raise ValueError(f"{name} arrays do not align with window starts")
        splits[name] = {
            "bounds": [start, end],
            "timestamps": [stamp.isoformat() for stamp in timestamps[start:end]],
            "valid_mask": [bool(item) for item in valid_mask[start:end]],
            "window_starts": starts,
            "target_indices": [item + v2.HISTORY_STEPS for item in starts],
            "array_rows": len(starts),
            "array_fingerprints": {
                f"X_{name}.npy": array_fingerprint(history),
                f"y_{name}.npy": array_fingerprint(target),
            },
        }
    return {
        "schema_version": PROVENANCE_SCHEMA,
        "protocol_id": v2.PROTOCOL_ID,
        "anchor_timestamp": v2.EXPECTED_START.isoformat(),
        "step_seconds": int(v2.STEP.total_seconds()),
        "history_length": v2.HISTORY_STEPS,
        "prediction_horizon": v2.PREDICTION_HORIZON,
        "splits": splits,
    }


def validate_train_validation_provenance(
    provenance: object,
    arrays_by_split: Mapping[str, tuple[np.ndarray, np.ndarray]],
    *,
    required_splits: Sequence[str] = PROVENANCE_SPLITS,
) -> dict[str, str]:
    """Validate sidecar rules through official v2 start selection.

    Returns canonical per-split hashes. The accepted sidecar schema is strictly
    train/validation-only; it has no channel for test starts or test arrays.
    """
    if not isinstance(provenance, dict):
        raise ValueError("Window provenance must be a JSON object")
    expected_top_level = {
        "schema_version", "protocol_id", "anchor_timestamp", "step_seconds",
        "history_length", "prediction_horizon", "splits",
    }
    if set(provenance) != expected_top_level:
        raise ValueError("Window provenance schema fields are missing or unrecognized")
    if type(provenance.get("schema_version")) is not int:
        raise ValueError("Window provenance schema version must be an integer")
    if provenance.get("schema_version") != PROVENANCE_SCHEMA:
        raise ValueError("Unsupported window provenance schema")
    if provenance.get("protocol_id") != v2.PROTOCOL_ID:
        raise ValueError("Window provenance protocol mismatch")
    if provenance.get("anchor_timestamp") != v2.EXPECTED_START.isoformat():
        raise ValueError("Window provenance anchor mismatch")
    if provenance.get("step_seconds") != int(v2.STEP.total_seconds()):
        raise ValueError("Window provenance timestep mismatch")
    if provenance.get("history_length") != v2.HISTORY_STEPS or provenance.get("prediction_horizon") != v2.PREDICTION_HORIZON:
        raise ValueError("Window provenance history/horizon mismatch")
    splits = provenance.get("splits")
    if not isinstance(splits, dict) or set(splits) != set(PROVENANCE_SPLITS):
        raise ValueError("Provenance must contain exactly train and validation; test metadata is forbidden")
    if set(required_splits) - set(PROVENANCE_SPLITS):
        raise ValueError("Only train and validation provenance can be requested")
    if not isinstance(arrays_by_split, Mapping):
        raise ValueError("Expected a mapping of permitted split arrays")
    if set(arrays_by_split) != set(required_splits):
        raise ValueError("Loaded array splits differ from the explicitly required provenance splits")

    timestamps_by_index: dict[int, datetime] = {}
    validity_by_index: dict[int, bool] = {}
    checked: dict[str, dict[str, object]] = {}
    for name in PROVENANCE_SPLITS:
        block = splits[name]
        if not isinstance(block, dict):
            raise ValueError(f"Malformed {name} provenance block")
        expected_block_fields = {
            "bounds", "timestamps", "valid_mask", "window_starts",
            "target_indices", "array_rows", "array_fingerprints",
        }
        if set(block) != expected_block_fields:
            raise ValueError(f"{name} provenance contains missing or unrecognized fields")
        start, end = v2.SPLIT_BOUNDS[name]
        if block.get("bounds") != [start, end]:
            raise ValueError(f"{name} provenance bounds mismatch")
        stamp_values = block.get("timestamps")
        valid_values = block.get("valid_mask")
        starts_value = block.get("window_starts")
        targets_value = block.get("target_indices")
        if not isinstance(stamp_values, list) or len(stamp_values) != end - start:
            raise ValueError(f"{name} timestamp provenance length mismatch")
        if not isinstance(valid_values, list) or len(valid_values) != end - start:
            raise ValueError(f"{name} validity provenance length mismatch")
        if any(type(value) is not bool for value in valid_values):
            raise ValueError(f"{name} validity provenance must contain booleans")
        for offset, text in enumerate(stamp_values):
            if not isinstance(text, str):
                raise ValueError(f"Malformed {name} timestamp")
            stamp = datetime.fromisoformat(text)
            index = start + offset
            expected = v2.EXPECTED_START + index * v2.STEP
            if stamp != expected:
                raise ValueError(f"{name} timestamp does not match its chronological index")
            timestamps_by_index[index] = stamp
            validity_by_index[index] = valid_values[offset]
        if not isinstance(starts_value, list) or not isinstance(targets_value, list):
            raise ValueError(f"{name} start/target index lists are malformed")
        if any(type(item) is not int for item in starts_value + targets_value):
            raise ValueError(f"{name} indices must be integers")
        if starts_value != sorted(set(starts_value)):
            raise ValueError(f"{name} starts must be sorted and unique")
        if targets_value != [item + v2.HISTORY_STEPS for item in starts_value]:
            raise ValueError(f"{name} target indices do not follow the five-plus-one convention")
        if type(block.get("array_rows")) is not int or block.get("array_rows") != len(starts_value):
            raise ValueError(f"{name} provenance length does not match array rows")
        fingerprints = block.get("array_fingerprints")
        expected_fingerprints = {f"X_{name}.npy", f"y_{name}.npy"}
        if not isinstance(fingerprints, dict) or set(fingerprints) != expected_fingerprints:
            raise ValueError(f"Missing {name} array fingerprints")
        if name in required_splits:
            window_arrays = arrays_by_split[name]
            if len(window_arrays) != 2 or any(len(array) != len(starts_value) for array in window_arrays):
                raise ValueError(f"{name} provenance length does not match loaded array rows")
            for filename, array in zip((f"X_{name}.npy", f"y_{name}.npy"), window_arrays):
                if fingerprints.get(filename) != array_fingerprint(array):
                    raise ValueError(f"{name} provenance fingerprint mismatch for {filename}")
        checked[name] = block

    # Reuse the authoritative six-observation validity/window rule. The
    # timeline contains only train and validation provenance, never test data.
    timeline_length = v2.SPLIT_BOUNDS["validation"][1]
    stamps = [v2.EXPECTED_START + index * v2.STEP for index in range(timeline_length)]
    valid = [False] * timeline_length
    for index, stamp in timestamps_by_index.items():
        stamps[index] = stamp
        valid[index] = validity_by_index[index]
    digests: dict[str, str] = {}
    for name in required_splits:
        start, end = v2.SPLIT_BOUNDS[name]
        official_starts = list(v2.valid_window_starts(stamps, valid, split_start=start, split_end=end))
        if checked[name]["window_starts"] != official_starts:
            raise ValueError(f"{name} starts differ from the official validity/window rules")
        digests[name] = canonical_json_sha256(checked[name])
    return digests

