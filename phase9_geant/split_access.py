"""Fail-closed access to the approved train/validation window arrays.

This module intentionally has no test-array name in its allowlist.  It is a
small policy choke point for readers, not an OS security boundary against
arbitrary Python code in the same process.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from phase9_geant.safe_paths import require_regular_file


PERMITTED_SPLIT_ARRAY_NAMES = frozenset(
    {"X_train.npy", "y_train.npy", "X_validation.npy", "y_validation.npy"}
)


def validate_split_array_name(filename: str) -> str:
    """Return an explicitly permitted name; reject test and unknown names."""
    if not isinstance(filename, str):
        raise TypeError("Split array name must be a string")
    if "test" in filename.casefold():
        raise ValueError("Test-named arrays are forbidden by the split-access policy")
    if filename not in PERMITTED_SPLIT_ARRAY_NAMES:
        raise ValueError(f"Split array is not explicitly allow-listed: {filename!r}")
    return filename


def load_permitted_split_array(
    data_root: str | Path,
    filename: str,
    *,
    array_loader: Callable | None = None,
) -> np.ndarray:
    """Load one direct-child train/validation array after exact-name checking."""
    allowed_name = validate_split_array_name(filename)
    root = Path(data_root)
    path = require_regular_file(root / allowed_name, root)
    loader = np.load if array_loader is None else array_loader
    return loader(path, allow_pickle=False)
