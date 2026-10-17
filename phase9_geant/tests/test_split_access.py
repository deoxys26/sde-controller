from __future__ import annotations

import numpy as np
import pytest

from phase9_geant.split_access import (
    PERMITTED_SPLIT_ARRAY_NAMES,
    load_permitted_split_array,
    validate_split_array_name,
)


def test_split_access_has_only_four_explicit_train_validation_names():
    assert PERMITTED_SPLIT_ARRAY_NAMES == {
        "X_train.npy", "y_train.npy", "X_validation.npy", "y_validation.npy"
    }
    for filename in PERMITTED_SPLIT_ARRAY_NAMES:
        assert validate_split_array_name(filename) == filename


@pytest.mark.parametrize("filename", [
    "X_test.npy", "test.npy", "X_TEST.npy", "contest.npy",
    "X_train_extra.npy", "window_starts.json", "../X_train.npy",
])
def test_split_access_rejects_test_substrings_and_unknown_names(filename):
    with pytest.raises(ValueError):
        validate_split_array_name(filename)


def test_split_loader_only_passes_allowlisted_file_to_injected_loader(tmp_path):
    values = np.arange(10, dtype=np.float32)
    np.save(tmp_path / "X_validation.npy", values, allow_pickle=False)
    calls = []

    def loader(path, *, allow_pickle):
        calls.append((path.name, allow_pickle))
        return np.load(path, allow_pickle=allow_pickle)

    actual = load_permitted_split_array(
        tmp_path, "X_validation.npy", array_loader=loader
    )
    np.testing.assert_array_equal(actual, values)
    assert calls == [("X_validation.npy", False)]
    with pytest.raises(ValueError, match="Test-named"):
        load_permitted_split_array(tmp_path, "X_test.npy", array_loader=loader)
    assert len(calls) == 1
