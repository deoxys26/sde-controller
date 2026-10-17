from __future__ import annotations

import pytest

from phase9_geant.tests.conftest import REPOSITORY_ROOT, _check_read


def test_read_guard_accepts_test_named_tmp_directory_and_python_test_file(tmp_path):
    seen: list[tuple[str, str]] = []
    array_path = (
        tmp_path / "data" / "processed" / "geant_validity_v2"
        / "neural_bu_v2_full_seed_123" / "run_metadata.json"
    )
    train_array_path = tmp_path / "test_something" / "X_train.npy"
    test_source = REPOSITORY_ROOT / "phase9_geant" / "tests" / "test_foo.py"
    _check_read("synthetic", array_path, seen)
    _check_read("synthetic", train_array_path, seen)
    _check_read("synthetic", test_source, seen)
    assert len(seen) == 3


@pytest.mark.parametrize("filename", ["X_test.npy", "y_test_future.npz", "values-test.npy"])
def test_read_guard_rejects_test_split_array_token(tmp_path, filename):
    with pytest.raises(AssertionError, match="test-split array"):
        _check_read("synthetic", tmp_path / filename, [])


def test_read_guard_rejects_real_versioned_geant_root_by_path_containment():
    protected = REPOSITORY_ROOT / "data" / "processed" / "geant_validity_v2" / "anything.json"
    with pytest.raises(AssertionError, match="real GEANT data root"):
        _check_read("synthetic", protected, [])


def test_read_guard_rejects_approved_rebuild_root_by_path_containment():
    protected = (
        REPOSITORY_ROOT / "data" / "processed"
        / "geant_validity_v2_rebuild_trainval_20261017_v1" / "X_train.npy"
    )
    with pytest.raises(AssertionError, match="real GEANT data root"):
        _check_read("synthetic", protected, [])


def test_read_guard_rejects_path_with_different_casing():
    protected = str(
        REPOSITORY_ROOT / "data" / "processed" / "geant_validity_v2" / "anything.json"
    ).swapcase()
    with pytest.raises(AssertionError, match="real GEANT data root"):
        _check_read("synthetic", protected, [])


def test_read_guard_rejects_parent_segments_resolving_into_real_root():
    # This is a path-string-only check; no path under the real repository is
    # statted or opened.
    protected = REPOSITORY_ROOT / "data" / "processed" / "sandbox" / ".." / "geant_validity_v2" / "anything.json"
    with pytest.raises(AssertionError, match="real GEANT data root"):
        _check_read("synthetic", protected, [])


def test_read_guard_uses_the_shared_allowlist_for_canonical_test_array_names(tmp_path):
    with pytest.raises(AssertionError, match="test-split array"):
        _check_read("synthetic", tmp_path / "X_TEST.npy", [])
