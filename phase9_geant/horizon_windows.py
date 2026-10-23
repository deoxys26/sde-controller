"""Pure index construction for the proposed GEANT multi-horizon protocol.

This module performs no file I/O and accepts no utilization observations. Callers
must provide metadata for one official split only; the test split is deliberately
not a supported input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


OFFICIAL_SPLIT_BOUNDS = {
    "train": (0, 8020),
    "validation": (8020, 9740),
}


@dataclass(frozen=True)
class HorizonWindow:
    split: str
    history_start: int
    origin: int
    target: int
    horizon: int


def enumerate_horizon_windows(
    split: str,
    valid_mask: Sequence[bool],
    timestamps_seconds: Sequence[int],
    *,
    history_length: int = 672,
    horizons: Sequence[int] = (4, 96),
    step_seconds: int = 900,
) -> Mapping[int, tuple[HorizonWindow, ...]]:
    """Return valid windows for one official split, using split-local metadata.

    A window requires a valid, consecutive history and a valid target at exactly
    ``origin + horizon``. Timestamps must remain consecutive from the beginning
    of history through the target. Intermediate observation validity is not
    required, since only the origin history and target are model input/label.
    Indices in returned records are global series indices.
    """

    if split not in OFFICIAL_SPLIT_BOUNDS:
        raise ValueError("split must be 'train' or 'validation'; test is unsupported")
    if history_length <= 0 or step_seconds <= 0:
        raise ValueError("history_length and step_seconds must be positive")
    if not horizons or any(not isinstance(h, int) or h <= 0 for h in horizons):
        raise ValueError("horizons must contain positive integer steps")

    split_start, split_end = OFFICIAL_SPLIT_BOUNDS[split]
    expected_length = split_end - split_start
    if len(valid_mask) != expected_length or len(timestamps_seconds) != expected_length:
        raise ValueError(
            f"{split} metadata must contain exactly {expected_length} split-local steps"
        )

    result: dict[int, tuple[HorizonWindow, ...]] = {}
    for horizon in horizons:
        windows: list[HorizonWindow] = []
        # Local indices ensure neither history nor target can cross the boundary.
        for origin_local in range(history_length - 1, expected_length - horizon):
            history_start_local = origin_local - history_length + 1
            target_local = origin_local + horizon
            if not all(valid_mask[history_start_local : origin_local + 1]):
                continue
            if not valid_mask[target_local]:
                continue
            if any(
                timestamps_seconds[index] - timestamps_seconds[index - 1]
                != step_seconds
                for index in range(history_start_local + 1, target_local + 1)
            ):
                continue

            windows.append(
                HorizonWindow(
                    split=split,
                    history_start=split_start + history_start_local,
                    origin=split_start + origin_local,
                    target=split_start + target_local,
                    horizon=horizon,
                )
            )
        result[horizon] = tuple(windows)
    return result
