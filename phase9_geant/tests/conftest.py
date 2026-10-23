"""Synthetic-test read guard for GEANT artifacts.

Only the repository's approved array/checkpoint/text read APIs are wrapped.
Temporary pytest paths may contain ``test`` in their directory or test-file
names; split detection is therefore based on an exact filename token.
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from phase9_geant.split_access import validate_split_array_name


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
assert (REPOSITORY_ROOT / "phase9_geant").is_dir(), (
    f"Incorrect repository-root calculation from conftest.py: {REPOSITORY_ROOT}"
)
_PROCESSED_ROOT = REPOSITORY_ROOT / "data" / "processed"
_RAW_ROOT = REPOSITORY_ROOT / "data" / "raw"
REAL_DATA_ROOTS = (
    # Protect the processed-data tree as one canonical root so every current
    # and future geant_validity_v2* sibling is covered without filesystem
    # enumeration or suffix-based path guessing.
    _PROCESSED_ROOT,
    _RAW_ROOT / "geant_topology",
)
_ARRAY_SUFFIXES = {".npy", ".npz"}
_TEST_SPLIT_TOKEN = re.compile(r"(?:^|[_-])test(?=[-_.]|$)", re.IGNORECASE)


def _path_from_argument(value: object) -> Path | None:
    if isinstance(value, (str, bytes, os.PathLike)):
        try:
            return Path(os.path.normcase(os.path.abspath(os.fsdecode(value))))
        except (OSError, RuntimeError, TypeError, ValueError):
            return None
    name = getattr(value, "name", None)
    if isinstance(name, (str, bytes, os.PathLike)):
        try:
            return Path(os.path.normcase(os.path.abspath(os.fsdecode(name))))
        except (OSError, RuntimeError, TypeError, ValueError):
            return None
    return None


def _is_contained(path: Path, root: Path) -> bool:
    normalized_path = os.path.normcase(os.path.abspath(os.fspath(path)))
    normalized_root = os.path.normcase(os.path.abspath(os.fspath(root)))
    try:
        return os.path.commonpath((normalized_path, normalized_root)) == normalized_root
    except ValueError:
        return False


def _is_real_geant_data_path(path: Path) -> bool:
    return any(_is_contained(path, root) for root in REAL_DATA_ROOTS)


def _has_test_array_split_token(path: Path) -> bool:
    return path.suffix.casefold() in _ARRAY_SUFFIXES and bool(
        _TEST_SPLIT_TOKEN.search(path.name)
    )


def _check_read(api: str, value: object, seen: list[tuple[str, str]]) -> None:
    path = _path_from_argument(value)
    if path is None:
        return
    seen.append((api, str(path)))
    if _is_real_geant_data_path(path):
        raise AssertionError(f"synthetic test attempted {api} under real GEANT data root: {path}")
    if _has_test_array_split_token(path):
        # Apply the shared policy to canonical test-array names; the token
        # check above also catches decorated array filenames such as X_test_v2.npy.
        if path.name.casefold() in {"x_test.npy", "y_test.npy"}:
            try:
                validate_split_array_name(path.name)
            except ValueError as exc:
                raise AssertionError(f"synthetic test attempted {api} of test-split array: {path}") from exc
        raise AssertionError(f"synthetic test attempted {api} of test-split array: {path}")


@pytest.fixture(autouse=True)
def guard_real_geant_and_test_split_reads(monkeypatch):
    """Track reads through np/torch and pathlib, restoring every patch at teardown."""
    seen: list[tuple[str, str]] = []
    original_numpy_load = np.load
    original_torch_load = torch.load
    original_read_bytes = Path.read_bytes
    original_read_text = Path.read_text

    def guarded_numpy_load(file, *args, **kwargs):
        _check_read("np.load", file, seen)
        return original_numpy_load(file, *args, **kwargs)

    def guarded_torch_load(file, *args, **kwargs):
        _check_read("torch.load", file, seen)
        return original_torch_load(file, *args, **kwargs)

    def guarded_read_bytes(path: Path):
        _check_read("Path.read_bytes", path, seen)
        return original_read_bytes(path)

    def guarded_read_text(path: Path, *args, **kwargs):
        _check_read("Path.read_text", path, seen)
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", guarded_numpy_load)
    monkeypatch.setattr(torch, "load", guarded_torch_load)
    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)
    monkeypatch.setattr(Path, "read_text", guarded_read_text)
    yield seen


@pytest.fixture(scope="session", autouse=True)
def assert_pytest_basetemp_is_outside_repository(tmp_path_factory):
    """Fail closed if synthetic test files could land inside the real checkout."""
    basetemp = Path(os.path.normcase(os.path.abspath(os.fspath(tmp_path_factory.getbasetemp()))))
    if _is_contained(basetemp, REPOSITORY_ROOT):
        raise AssertionError(f"pytest basetemp must be outside repository root: {basetemp}")


@pytest.fixture
def tmp_path(request, tmp_path_factory):
    """Create an external synthetic directory without pytest's ``current`` symlink.

    In this Windows sandbox pytest's built-in fixture fails while creating its
    numbered-directory symlink under the virtualized Temp mount.  A unique
    direct child of pytest's already-validated external basetemp is sufficient
    for these tests and does not depend on symlink/junction privileges.
    Directories are intentionally left in that per-command basetemp; no
    pre-existing scratch content is cleaned or replaced.
    """
    base = Path(os.path.normcase(os.path.abspath(os.fspath(tmp_path_factory.getbasetemp()))))
    if _is_contained(base, REPOSITORY_ROOT):
        raise AssertionError(f"pytest basetemp must be outside repository root: {base}")
    # Keep the path short enough for Windows MAX_PATH even when basetemp is
    # supplied by a deeply nested virtualized external scratch root.
    del request
    return Path(tempfile.mkdtemp(prefix="t-", dir=base))
