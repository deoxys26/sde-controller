from __future__ import annotations

import builtins
from pathlib import Path

import numpy as np
import pytest

from phase9_geant import build_validity_v2 as builder
from phase9_geant import validity_v2


def test_synthetic_window_artifact_build_writes_but_never_reads_test_arrays(
    tmp_path, monkeypatch
):
    root = tmp_path / "fresh-synthetic-output"
    root.mkdir()
    arrays = {
        "train": (
            np.arange(2 * 5 * 3, dtype=np.float32).reshape(2, 5, 3),
            np.arange(2 * 3, dtype=np.float32).reshape(2, 3),
        ),
        "validation": (
            np.arange(1 * 5 * 3, dtype=np.float32).reshape(1, 5, 3),
            np.arange(1 * 3, dtype=np.float32).reshape(1, 3),
        ),
        "test": (
            np.arange(1 * 5 * 3, dtype=np.float32).reshape(1, 5, 3),
            np.arange(1 * 3, dtype=np.float32).reshape(1, 3),
        ),
    }
    reads: list[str] = []
    original_numpy_load = np.load
    original_open = builtins.open
    original_read_bytes = Path.read_bytes

    def tracking_numpy_load(file, *args, **kwargs):
        reads.append(Path(file).name)
        return original_numpy_load(file, *args, **kwargs)

    def tracking_open(file, mode="r", *args, **kwargs):
        if "r" in mode and isinstance(file, (str, Path)):
            reads.append(Path(file).name)
        return original_open(file, mode, *args, **kwargs)

    def tracking_read_bytes(path):
        reads.append(path.name)
        return original_read_bytes(path)

    monkeypatch.setattr(np, "load", tracking_numpy_load)
    monkeypatch.setattr(builtins, "open", tracking_open)
    monkeypatch.setattr(Path, "read_bytes", tracking_read_bytes)

    builder._write_and_verify_window_arrays(root, arrays)

    assert (root / "X_test.npy").is_file()
    assert (root / "y_test.npy").is_file()
    assert set(reads) == {
        "X_train.npy", "y_train.npy", "X_validation.npy", "y_validation.npy"
    }
    assert not any("test" in name.casefold() for name in reads)


def test_train_validation_only_option_excludes_test_and_full_series_artifacts(tmp_path):
    assert builder.parse_requested_splits("train,val") == ("train", "validation")
    assert builder.parse_requested_splits("train,validation") == ("train", "validation")
    assert builder.parse_requested_splits(None) == ("train", "validation", "test")
    with pytest.raises(ValueError, match="exactly"):
        builder.parse_requested_splits("train,test")

    root = tmp_path / "trainval-only"
    root.mkdir()
    arrays = {
        "train": (
            np.ones((1, 5, 2), dtype=np.float32),
            np.ones((1, 2), dtype=np.float32),
        ),
        "validation": (
            np.ones((1, 5, 2), dtype=np.float32),
            np.ones((1, 2), dtype=np.float32),
        ),
    }
    builder._write_and_verify_window_arrays(root, arrays)
    assert {path.name for path in root.iterdir()} == {
        "X_train.npy", "y_train.npy", "X_validation.npy", "y_validation.npy"
    }


def test_new_versioned_rebuild_root_is_direct_child_and_refuses_legacy_tree(tmp_path):
    repo = tmp_path / "repo"
    processed = repo / "data" / "processed"
    processed.mkdir(parents=True)
    new_root = processed / "geant_validity_v2_rebuild_synthetic_v1"
    assert validity_v2.validate_output_root(new_root, project_root=repo) == new_root
    with pytest.raises(ValueError, match="versioned rebuild"):
        validity_v2.validate_output_root(processed / "geant", project_root=repo)
