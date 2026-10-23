from __future__ import annotations

import inspect

import pytest

from phase9_geant.horizon_windows import (
    OFFICIAL_SPLIT_BOUNDS,
    enumerate_horizon_windows,
)


STEP_SECONDS = 900
HISTORY = 672


def _timestamps(start_index: int, count: int) -> list[int]:
    return [(start_index + offset) * STEP_SECONDS for offset in range(count)]


def test_horizon_targets_are_exactly_origin_plus_four_and_ninety_six() -> None:
    valid = [True] * 8020
    windows = enumerate_horizon_windows(
        "train", valid, _timestamps(0, len(valid)), history_length=HISTORY
    )

    for horizon in (4, 96):
        first = windows[horizon][0]
        assert first.history_start == 0
        assert first.origin == HISTORY - 1
        assert first.target == first.origin + horizon
        assert first.horizon == horizon
        # Synthetic index-valued observations make target alignment explicit.
        synthetic_values = list(range(8020))
        assert synthetic_values[first.target] - synthetic_values[first.origin] == horizon


def test_windows_remain_inside_their_official_train_or_validation_block() -> None:
    assert OFFICIAL_SPLIT_BOUNDS == {"train": (0, 8020), "validation": (8020, 9740)}
    train_valid = [True] * 8020
    train = enumerate_horizon_windows(
        "train", train_valid, _timestamps(0, len(train_valid)), history_length=HISTORY
    )
    validation_valid = [True] * 1720
    validation = enumerate_horizon_windows(
        "validation",
        validation_valid,
        _timestamps(8020, len(validation_valid)),
        history_length=HISTORY,
    )

    for horizons in (train, validation):
        for windows in horizons.values():
            start, end = OFFICIAL_SPLIT_BOUNDS[windows[0].split]
            assert all(start <= item.history_start <= item.origin < item.target < end for item in windows)
    assert max(item.target for item in train[96]) == 8019
    assert min(item.history_start for item in validation[96]) == 8020
    assert max(item.target for item in validation[96]) == 9739
    with pytest.raises(ValueError, match="test is unsupported"):
        enumerate_horizon_windows("test", [], [], history_length=HISTORY)


def test_invalid_history_target_and_chronology_gap_are_excluded() -> None:
    valid = [True] * 8020
    # A bad history observation excludes origins whose 672-step history contains it.
    valid[100] = False
    # A bad target excludes exactly that target for the associated horizon.
    valid[900] = False
    timestamps = _timestamps(0, len(valid))
    # A chronology gap excludes windows spanning the discontinuity, without
    # shifting target indices or filling the missing step.
    timestamps[1500:] = [value + STEP_SECONDS for value in timestamps[1500:]]
    windows = enumerate_horizon_windows(
        "train", valid, timestamps, history_length=HISTORY
    )

    assert all(100 not in range(item.history_start, item.origin + 1) for item in windows[4])
    assert all(item.target != 900 for item in windows[4])
    assert all(
        not (item.history_start < 1500 <= item.target)
        for horizon_windows in windows.values()
        for item in horizon_windows
    )


def test_index_builder_accepts_metadata_only_not_observations_or_targets() -> None:
    parameters = inspect.signature(enumerate_horizon_windows).parameters
    assert "valid_mask" in parameters
    assert "timestamps_seconds" in parameters
    assert "observations" not in parameters
    assert "targets" not in parameters


def test_split_local_metadata_must_have_exact_official_length() -> None:
    with pytest.raises(ValueError, match="exactly 8020"):
        enumerate_horizon_windows("train", [True] * 10, _timestamps(0, 10))
