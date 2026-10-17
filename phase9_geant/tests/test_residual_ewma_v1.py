"""Synthetic-only tests for residual-to-EWMA forecasting helpers."""

from __future__ import annotations

from datetime import datetime, timedelta
import inspect
import json
import os
import subprocess
import sys
import stat
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from phase9_geant import train_bu_v2 as bu
from phase9_geant import train_residual_ewma_v1 as residual
from phase9_geant import validity_v2 as validity
from phase9_geant import safe_paths
from phase9_geant.window_provenance_v2 import (
    PROVENANCE_FILENAME,
    build_train_validation_provenance,
    validate_train_validation_provenance,
)


def _toy_history() -> np.ndarray:
    return np.asarray(
        [
            [[0.01, 0.20], [0.02, 0.18], [0.03, 0.16], [0.04, 0.14], [0.05, 0.12]],
            [[0.10, 0.02], [0.09, 0.03], [0.08, 0.04], [0.07, 0.05], [0.06, 0.06]],
        ],
        dtype=np.float32,
    )


def _temp_v2_root(tmp_path):
    repo = tmp_path / "repo"
    data_root = repo / "data" / "processed" / "geant_validity_v2"
    data_root.mkdir(parents=True)
    return repo, data_root


def _synthetic_arrays(train_rows=2, validation_rows=2):
    x_train = np.arange(train_rows * 5 * 72, dtype=np.float32).reshape(train_rows, 5, 72) + 1
    y_train = np.arange(train_rows * 72, dtype=np.float32).reshape(train_rows, 72) + 2
    x_validation = np.arange(validation_rows * 5 * 72, dtype=np.float32).reshape(validation_rows, 5, 72) + 3
    y_validation = np.arange(validation_rows * 72, dtype=np.float32).reshape(validation_rows, 72) + 4
    return bu.WindowArrays(x_train, y_train, x_validation, y_validation)


def _synthetic_provenance(arrays=None, train_starts=(0, 10), validation_starts=(8020, 8030)):
    arrays = arrays or _synthetic_arrays(len(train_starts), len(validation_starts))
    timestamps = [validity.EXPECTED_START + validity.STEP * index for index in range(9740)]
    valid = [False] * 9740
    for start in (*train_starts, *validation_starts):
        for index in range(start, start + 6):
            valid[index] = True
    windows = {"train": tuple(train_starts), "validation": tuple(validation_starts), "test": ()}
    arrays_by_split = {
        "train": (arrays.x_train, arrays.y_train),
        "validation": (arrays.x_validation, arrays.y_validation),
    }
    provenance = build_train_validation_provenance(timestamps, valid, windows, arrays_by_split)
    return provenance


def test_ewma_weights_and_prediction_match_frozen_baseline_formula():
    history = _toy_history()
    weights = 0.8 * (1.0 - 0.8) ** (4 - np.arange(5, dtype=np.float64))
    weights /= weights.sum()
    expected = np.tensordot(history, weights, axes=([1], [0]))
    np.testing.assert_allclose(residual.ewma_weights(), weights, rtol=0, atol=1e-15)
    np.testing.assert_allclose(residual.ewma_forecast(history), expected, rtol=0, atol=1e-15)
    assert weights[-1] > weights[:-1].max()


def test_residual_target_alignment_uses_next_step_target_minus_history_only_ewma():
    history = _toy_history()
    target = np.asarray([[0.08, 0.10], [0.02, 0.07]], dtype=np.float32)
    expected = target.astype(np.float64) - residual.ewma_forecast(history)
    np.testing.assert_allclose(residual.residual_targets(history, target), expected, rtol=0, atol=1e-15)
    changed_future_target = target + 100.0
    np.testing.assert_array_equal(residual.ewma_forecast(history), residual.ewma_forecast(history.copy()))
    assert not np.array_equal(residual.residual_targets(history, target), residual.residual_targets(history, changed_future_target))


def test_model_inputs_are_only_each_links_five_history_values():
    history = _toy_history()
    scaler = bu.Normalization(mean=0.0, scale=1.0)
    inputs = residual.make_per_link_inputs(history, scaler)
    assert inputs.shape == (history.shape[0] * history.shape[2], 5)
    torch.testing.assert_close(inputs[0], torch.tensor(history[0, :, 0]))
    torch.testing.assert_close(inputs[1], torch.tensor(history[0, :, 1]))
    torch.testing.assert_close(inputs[2], torch.tensor(history[1, :, 0]))
    torch.testing.assert_close(inputs, residual.make_per_link_inputs(history, scaler))
    assert "target" not in inspect.signature(residual.make_per_link_inputs).parameters
    model = residual.PerLinkResidualMLP()
    assert model(inputs).shape == (4, 1)


def test_zero_residual_reconstructs_ewma_and_inverse_scaler_is_correct():
    history = _toy_history()
    baseline = residual.ewma_forecast(history)
    reconstructed = residual.reconstruct_forecast(history, np.zeros_like(baseline))
    np.testing.assert_array_equal(reconstructed, baseline)
    scaler = residual.ResidualScaler(mean=0.25, scale=0.5)
    normalized = np.asarray([[-2.0, 0.0], [3.0, 1.0]])
    expected = normalized * scaler.scale + scaler.mean
    inverse_scaled = residual.inverse_scale_residual(normalized, scaler)
    np.testing.assert_array_equal(inverse_scaled, expected)
    np.testing.assert_array_equal(
        residual.reconstruct_forecast(history, inverse_scaled.reshape(history.shape[0], history.shape[2])),
        baseline + expected.reshape(history.shape[0], history.shape[2]),
    )


def test_residual_scaler_uses_only_values_passed_from_training_residuals():
    train_residuals = np.asarray([[-0.2, 0.0], [0.1, 0.3]], dtype=np.float64)
    scaler = residual.fit_residual_scaler(train_residuals)
    assert scaler.mean == pytest.approx(float(train_residuals.mean()))
    assert scaler.scale == pytest.approx(float(train_residuals.std()))
    # Validation values are not an argument to the fit API and cannot affect it.
    validation_residuals = np.full((3, 2), 999.0)
    repeated = residual.fit_residual_scaler(train_residuals.copy())
    assert repeated == scaler
    assert validation_residuals.mean() != scaler.mean


def test_checkpoint_selection_uses_only_finite_validation_objective():
    assert residual.validation_loss_improved(0.2, float("inf"))
    assert residual.validation_loss_improved(0.1, 0.2)
    assert not residual.validation_loss_improved(0.2, 0.1)
    with pytest.raises(ValueError, match="finite"):
        residual.validation_loss_improved(float("nan"), 0.1)


def test_validation_objective_is_standardized_residual_smooth_l1_without_parameter_updates():
    history = _toy_history()
    target = np.asarray([[0.08, 0.10], [0.02, 0.07]], dtype=np.float32)
    residual_scaler = residual.fit_residual_scaler(residual.residual_targets(history, target))
    utilization_scaler = bu.Normalization(mean=0.0, scale=1.0)
    model = residual.PerLinkResidualMLP()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    before = [parameter.detach().clone() for parameter in model.parameters()]
    observed = residual._validation_objective(
        model, history, target, utilization_scaler, residual_scaler, batch_size=1
    )
    standardized_target = residual.standardize_residuals(
        residual.residual_targets(history, target), residual_scaler
    )
    expected = float(torch.nn.functional.smooth_l1_loss(torch.zeros_like(standardized_target), standardized_target))
    assert observed == pytest.approx(expected)
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(torch.equal(old, current) for old, current in zip(before, model.parameters()))


def test_shape_order_and_finite_value_guards_fail_closed():
    history = _toy_history()
    with pytest.raises(ValueError, match="shape"):
        residual.residual_targets(history, np.zeros((2, 3)))
    bad_history = history.copy()
    bad_history[0, 0, 0] = np.inf
    with pytest.raises(ValueError, match="nonfinite"):
        residual.ewma_forecast(bad_history)
    with pytest.raises(ValueError, match="nonfinite"):
        residual.residual_targets(history, np.asarray([[np.nan, 0.0], [0.0, 0.0]]))


def test_directed_edge_order_helper_rejects_wrong_column_order():
    nodes = ("a", "b", "c", "d")
    links = (("a", "b"), ("b", "c"), ("c", "d"), ("d", "a"))
    expected = residual.validate_directed_edge_order(nodes, links, expected_links=8)
    assert expected == bu.directed_edges_v2(links)
    with pytest.raises(ValueError, match="ordering"):
        bu.assert_edge_order(expected, tuple(reversed(expected)))


def test_window_start_validation_rejects_invalid_gap_and_split_crossings():
    count = 9746
    timestamps = [datetime(2005, 1, 1) + timedelta(minutes=15 * i) for i in range(count)]
    valid = [True] * count
    residual.validate_window_starts([100], timestamps, valid, split_name="train")
    with pytest.raises(ValueError, match="boundary"):
        residual.validate_window_starts([8015], timestamps, valid, split_name="train")
    with pytest.raises(ValueError, match="boundary"):
        residual.validate_window_starts([9735], timestamps, valid, split_name="validation")
    invalid = valid.copy()
    invalid[102] = False
    with pytest.raises(ValueError, match="invalid"):
        residual.validate_window_starts([100], timestamps, invalid, split_name="train")
    gap = timestamps.copy()
    gap[103] += timedelta(minutes=15)
    with pytest.raises(ValueError, match="chronology gap"):
        residual.validate_window_starts([100], gap, valid, split_name="train")


@pytest.mark.parametrize("seed", residual.SEEDS)
def test_existing_seed_directory_is_rejected_before_array_loader(tmp_path, monkeypatch, seed):
    repo, data_root = _temp_v2_root(tmp_path)
    output = data_root / f"neural_ewma_residual_v1_seed_{seed}"
    output.mkdir()

    def forbidden_loader(_root):
        raise AssertionError("array loader must not run for an existing output")

    monkeypatch.setattr(residual, "load_train_validation_arrays", forbidden_loader)
    with pytest.raises(FileExistsError, match="Refusing to reuse"):
        residual.run_seed(seed, data_root=data_root, repository_root=repo)


def test_runner_declares_only_the_four_train_validation_arrays():
    assert residual.INPUT_ARRAY_NAMES == (
        "X_train.npy",
        "y_train.npy",
        "X_validation.npy",
        "y_validation.npy",
    )
    assert residual.INPUT_ARRAY_NAMES == bu.ARRAY_NAMES
    assert residual.TRAINING["max_epochs"] == 5
    assert residual.TRAINING["early_stopping_patience"] == 2
    assert residual.EWMA_ALPHA == 0.8


def test_cli_requires_explicit_run_and_seed_and_paths_are_seed_specific(tmp_path):
    parser = residual.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
    args = parser.parse_args(["--run", "--seed", "123"])
    assert args.run is True and args.seed == 123
    repo, root = _temp_v2_root(tmp_path)
    paths = [residual.output_dir_for_seed(seed, root, repository_root=repo) for seed in residual.SEEDS]
    assert len(set(paths)) == 3


def test_train_validation_provenance_validates_official_split_windows_and_fingerprints():
    arrays = _synthetic_arrays()
    provenance = _synthetic_provenance(arrays)
    hashes = validate_train_validation_provenance(
        provenance,
        {"train": (arrays.x_train, arrays.y_train), "validation": (arrays.x_validation, arrays.y_validation)},
    )
    assert set(hashes) == {"train", "validation"}
    assert provenance["splits"]["train"]["target_indices"] == [5, 15]
    assert provenance["splits"]["validation"]["target_indices"] == [8025, 8035]
    assert "test" not in provenance["splits"]


def test_provenance_rejects_unrecognized_fields_that_could_carry_test_metadata():
    arrays = _synthetic_arrays()
    provenance = _synthetic_provenance(arrays)
    provenance["splits"]["validation"]["test_window_starts"] = [9740]
    with pytest.raises(ValueError, match="unrecognized fields"):
        validate_train_validation_provenance(
            provenance,
            {"train": (arrays.x_train, arrays.y_train), "validation": (arrays.x_validation, arrays.y_validation)},
        )


def test_provenance_rejects_boolean_array_row_count_even_when_equal_to_one():
    arrays = _synthetic_arrays(train_rows=1, validation_rows=1)
    provenance = _synthetic_provenance(
        arrays, train_starts=(0,), validation_starts=(8020,)
    )
    provenance["splits"]["train"]["array_rows"] = True
    with pytest.raises(ValueError, match="array rows"):
        validate_train_validation_provenance(
            provenance,
            {"train": (arrays.x_train, arrays.y_train), "validation": (arrays.x_validation, arrays.y_validation)},
        )


def test_provenance_sidecar_is_independent_of_synthetic_test_tail():
    arrays = _synthetic_arrays()
    timestamps = [validity.EXPECTED_START + validity.STEP * i for i in range(11460)]
    valid = [False] * len(timestamps)
    train_starts, validation_starts = (0, 10), (8020, 8030)
    for start in (*train_starts, *validation_starts):
        for index in range(start, start + 6):
            valid[index] = True
    windows = {"train": train_starts, "validation": validation_starts, "test": ()}
    arrays_by_split = {
        "train": (arrays.x_train, arrays.y_train),
        "validation": (arrays.x_validation, arrays.y_validation),
    }
    first = build_train_validation_provenance(timestamps, valid, windows, arrays_by_split)
    changed_valid = valid.copy()
    changed_valid[9740:] = [not value for value in changed_valid[9740:]]
    changed_timestamps = timestamps.copy()
    changed_timestamps[9740:] = [stamp + validity.STEP for stamp in changed_timestamps[9740:]]
    second = build_train_validation_provenance(changed_timestamps, changed_valid, windows, arrays_by_split)
    assert first == second


def test_fixed_loader_accepts_only_synthetic_arrays_under_normal_root(tmp_path):
    repo, data_root = _temp_v2_root(tmp_path)
    arrays = _synthetic_arrays()
    for filename, value in zip(
        residual.INPUT_ARRAY_NAMES,
        (arrays.x_train, arrays.y_train, arrays.x_validation, arrays.y_validation),
    ):
        np.save(data_root / filename, value, allow_pickle=False)
    loaded = residual.load_train_validation_arrays(data_root, repository_root=repo)
    np.testing.assert_array_equal(loaded.x_train, arrays.x_train)
    np.testing.assert_array_equal(loaded.y_validation, arrays.y_validation)


def test_array_loader_checks_ancestor_redirect_before_numpy_load(tmp_path):
    repo = tmp_path / "repo"
    data = repo / "data"
    data.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        os.symlink(outside, data / "processed", target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"Directory symlink unavailable in this environment: {exc}")
    called = False

    def forbidden_numpy_load(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("NumPy load must not run for a redirected ancestor")

    with pytest.raises(ValueError, match="Symlink|Junction|Reparse"):
        residual.load_train_validation_arrays(
            data / "processed" / "geant_validity_v2",
            repository_root=repo,
            array_loader=forbidden_numpy_load,
        )
    assert not called


@pytest.mark.parametrize("mutation, message", [
    ("missing", "JSON object"),
    ("length", "length does not match"),
    ("unsorted", "sorted and unique"),
    ("duplicate", "sorted and unique"),
    ("train_boundary", "official validity/window rules"),
    ("validation_boundary", "official validity/window rules"),
    ("invalid_history", "official validity/window rules"),
    ("nonconsecutive", "chronological index"),
    ("target", "target indices"),
    ("fingerprint", "fingerprint mismatch"),
    ("test_entry", "exactly train and validation"),
])
def test_provenance_rejects_missing_malformed_or_mismatched_values(mutation, message):
    arrays = _synthetic_arrays()
    provenance = _synthetic_provenance(arrays)
    if mutation == "missing":
        provenance = None
    elif mutation == "length":
        provenance["splits"]["train"]["array_rows"] = 99
    elif mutation == "unsorted":
        provenance["splits"]["train"]["window_starts"] = [10, 0]
    elif mutation == "duplicate":
        provenance["splits"]["train"]["window_starts"] = [0, 0]
    elif mutation == "train_boundary":
        provenance["splits"]["train"]["window_starts"] = [8015, 8016]
        provenance["splits"]["train"]["target_indices"] = [8020, 8021]
    elif mutation == "validation_boundary":
        provenance["splits"]["validation"]["window_starts"] = [8020, 9735]
        provenance["splits"]["validation"]["target_indices"] = [8025, 9740]
    elif mutation == "invalid_history":
        provenance["splits"]["train"]["valid_mask"][2] = False
    elif mutation == "nonconsecutive":
        provenance["splits"]["validation"]["timestamps"][3] = (
            datetime.fromisoformat(provenance["splits"]["validation"]["timestamps"][3])
            + timedelta(minutes=15)
        ).isoformat()
    elif mutation == "target":
        provenance["splits"]["train"]["target_indices"][0] += 1
    elif mutation == "fingerprint":
        provenance["splits"]["validation"]["array_fingerprints"]["X_validation.npy"] = "0" * 64
    elif mutation == "test_entry":
        provenance["splits"]["test"] = {"window_starts": [9740]}
    with pytest.raises(ValueError, match=message):
        validate_train_validation_provenance(
            provenance,
            {"train": (arrays.x_train, arrays.y_train), "validation": (arrays.x_validation, arrays.y_validation)},
        )


def test_invalid_provenance_prevents_model_and_output_creation(tmp_path, monkeypatch):
    repo, data_root = _temp_v2_root(tmp_path)
    (data_root / PROVENANCE_FILENAME).write_text("{}", encoding="utf-8")
    arrays = _synthetic_arrays()
    monkeypatch.setattr(residual, "load_train_validation_arrays", lambda *_a, **_k: arrays)
    monkeypatch.setattr(residual, "load_window_provenance", lambda *_a, **_k: ({}, "synthetic-hash"))
    model_called = False

    def forbidden_model():
        nonlocal model_called
        model_called = True
        raise AssertionError("model construction must not be reached")

    monkeypatch.setattr(residual, "PerLinkResidualMLP", forbidden_model)
    output = data_root / "neural_ewma_residual_v1_seed_42"
    with pytest.raises(ValueError, match="schema"):
        residual.run_seed(42, data_root=data_root, repository_root=repo)
    assert not model_called
    assert not output.exists()


def test_safe_paths_reject_symlinked_root_or_ancestor_before_array_loading(tmp_path, monkeypatch):
    repo, data_root = _temp_v2_root(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    link = repo / "data" / "processed" / "geant_validity_v2"
    link.rmdir()
    try:
        os.symlink(outside, link, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"Directory symlink unavailable in this environment: {exc}")

    def forbidden_loader(*_args, **_kwargs):
        raise AssertionError("arrays must not be loaded through redirected paths")

    monkeypatch.setattr(residual, "load_train_validation_arrays", forbidden_loader)
    with pytest.raises(ValueError, match="Symlink|Junction|Reparse"):
        residual.run_seed(42, data_root=link, repository_root=repo)
    assert not (outside / "neural_ewma_residual_v1_seed_42").exists()


def test_safe_paths_reject_ancestor_symlink_and_escaped_output(tmp_path):
    repo = tmp_path / "repo"
    outside = tmp_path / "outside"
    outside.mkdir()
    (repo / "data").mkdir(parents=True)
    try:
        os.symlink(outside, repo / "data" / "processed", target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"Directory symlink unavailable in this environment: {exc}")
    redirected_root = repo / "data" / "processed" / "geant_validity_v2"
    with pytest.raises(ValueError, match="Symlink|Junction|Reparse"):
        residual.output_dir_for_seed(42, redirected_root, repository_root=repo)

    repo2, data_root2 = _temp_v2_root(tmp_path / "second")
    escaped = tmp_path / "escaped-output"
    with pytest.raises(ValueError, match="dedicated child"):
        residual.output_dir_for_seed(42, data_root2, repository_root=repo2, output_path=escaped)
    assert not escaped.exists()


def test_post_creation_containment_failure_leaves_new_empty_directory(tmp_path, monkeypatch):
    repo, data_root = _temp_v2_root(tmp_path)
    output = data_root / "neural_ewma_residual_v1_seed_42"

    def fail_containment(*_args, **_kwargs):
        raise ValueError("synthetic post-create escape")

    monkeypatch.setattr(residual, "verify_new_directory_containment", fail_containment)
    with pytest.raises(ValueError, match="post-create escape"):
        residual._create_run_directory(output, data_root, repo)
    assert output.is_dir()
    assert list(output.iterdir()) == []


def test_cleanup_never_removes_replacement_directory(tmp_path, monkeypatch):
    repo, data_root = _temp_v2_root(tmp_path)
    output = data_root / "neural_ewma_residual_v1_seed_42"
    replacement = tmp_path / "replacement"

    def replace_then_fail(_output, _root):
        output.rmdir()
        replacement.mkdir()
        (replacement / "owner.txt").write_text("other process", encoding="utf-8")
        replacement.rename(output)
        raise ValueError("synthetic replacement detected")

    monkeypatch.setattr(residual, "verify_new_directory_containment", replace_then_fail)
    with pytest.raises(ValueError, match="replacement detected"):
        residual._create_run_directory(output, data_root, repo)
    assert (output / "owner.txt").read_text(encoding="utf-8") == "other process"


def test_post_creation_failure_preserves_nonempty_directory(tmp_path, monkeypatch):
    repo, data_root = _temp_v2_root(tmp_path)
    output = data_root / "neural_ewma_residual_v1_seed_42"

    def add_file_then_fail(path, _root):
        (path / "diagnostic.txt").write_text("keep", encoding="utf-8")
        raise ValueError("synthetic validation failure")

    monkeypatch.setattr(residual, "verify_new_directory_containment", add_file_then_fail)
    with pytest.raises(ValueError, match="validation failure"):
        residual._create_run_directory(output, data_root, repo)
    assert (output / "diagnostic.txt").read_text(encoding="utf-8") == "keep"


def test_post_creation_failure_preserves_redirected_reparse_replacement(tmp_path, monkeypatch):
    repo, data_root = _temp_v2_root(tmp_path)
    output = data_root / "neural_ewma_residual_v1_seed_42"
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "owner.txt").write_text("redirect target", encoding="utf-8")

    def redirect_then_fail(path, _root):
        path.rmdir()
        try:
            os.symlink(outside, path, target_is_directory=True)
        except (OSError, NotImplementedError) as exc:
            pytest.skip(f"Directory symlink/reparse replacement unavailable: {exc}")
        raise ValueError("synthetic redirected replacement")

    monkeypatch.setattr(residual, "verify_new_directory_containment", redirect_then_fail)
    with pytest.raises(ValueError, match="redirected replacement"):
        residual._create_run_directory(output, data_root, repo)
    assert output.is_symlink()
    assert (outside / "owner.txt").read_text(encoding="utf-8") == "redirect target"


def test_preexisting_directory_is_preserved_when_creation_is_refused(tmp_path):
    repo, data_root = _temp_v2_root(tmp_path)
    output = data_root / "neural_ewma_residual_v1_seed_42"
    output.mkdir()
    marker = output / "owner.txt"
    marker.write_text("preexisting", encoding="utf-8")
    with pytest.raises(FileExistsError):
        residual._create_run_directory(output, data_root, repo)
    assert marker.read_text(encoding="utf-8") == "preexisting"


def test_safe_paths_rejects_synthetic_reparse_component(tmp_path, monkeypatch):
    repo, _ = _temp_v2_root(tmp_path)
    component = repo / "data" / "processed"
    original_lstat = Path.lstat

    def reparse_lstat(path):
        if path == component:
            return SimpleNamespace(st_mode=stat.S_IFDIR, st_file_attributes=0x400)
        return original_lstat(path)

    monkeypatch.setattr(Path, "lstat", reparse_lstat)
    with pytest.raises(ValueError, match="Reparse-point"):
        residual.checked_absolute(component)


def test_checked_absolute_returns_nonexistent_trailing_output_path(tmp_path):
    requested = tmp_path / "new-root" / "new-run" / "manifest.json"
    assert safe_paths.checked_absolute(requested) == requested.absolute()
    assert not requested.exists()


def test_existing_reparse_flag_is_rejected_before_missing_suffix(tmp_path, monkeypatch):
    repo, _ = _temp_v2_root(tmp_path)
    existing = repo / "data" / "processed"
    requested = existing / "not-created" / "output"
    original = safe_paths._file_attributes

    def report_reparse(path, info):
        if path == existing:
            return 0x400
        return original(path, info)

    monkeypatch.setattr(safe_paths, "_file_attributes", report_reparse)
    with pytest.raises(ValueError, match="Reparse-point"):
        safe_paths.checked_absolute(requested)
    assert not (existing / "not-created").exists()


def test_missing_loader_data_root_has_clear_error_and_no_side_effects(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    data_root = repo / "data" / "processed" / "geant_validity_v2"
    calls = []

    def forbidden_loader(path, **_kwargs):
        calls.append(path)
        raise AssertionError("array loader must not run")

    with pytest.raises(FileNotFoundError, match="GEANT v2 data root does not exist"):
        residual.load_train_validation_arrays(
            data_root,
            repository_root=repo,
            array_loader=forbidden_loader,
        )
    assert not calls
    assert not data_root.exists()


@pytest.mark.parametrize("component_kind", ["root", "ancestor"])
def test_reparse_root_or_ancestor_rejected_before_array_load_or_output_creation(
    tmp_path, monkeypatch, component_kind
):
    repo, data_root = _temp_v2_root(tmp_path)
    component = data_root if component_kind == "root" else repo / "data" / "processed"
    original = safe_paths._file_attributes

    def report_reparse(path, info):
        if path == component:
            return 0x400
        return original(path, info)

    loaded = False

    def forbidden_loader(*_args, **_kwargs):
        nonlocal loaded
        loaded = True
        raise AssertionError("array loader must not run for reparse component")

    monkeypatch.setattr(safe_paths, "_file_attributes", report_reparse)
    with pytest.raises(ValueError, match="Reparse-point"):
        residual.run_seed(
            42,
            data_root=data_root,
            repository_root=repo,
            array_loader=forbidden_loader,
        )
    assert not loaded
    assert not (data_root / "neural_ewma_residual_v1_seed_42").exists()


def test_source_hashes_are_deterministic_over_synthetic_source_files_only(tmp_path):
    source = tmp_path / "synthetic.py"
    source.write_text("VALUE = 3\n", encoding="utf-8")
    first = residual._hash_source_files([source])
    second = residual._hash_source_files([source])
    assert first == second
    assert list(first) == [str(source.resolve())]


def _create_junction_or_skip(link: Path, target: Path, request) -> str:
    if sys.platform != "win32":
        pytest.skip(f"Windows junction creation is unavailable on platform {sys.platform!r}")
    failures: list[str] = []
    try:
        import _winapi

        _winapi.CreateJunction(str(target), str(link))
        if link.exists() and link.is_dir():
            method = "winapi"
            request.node.user_properties.append(("junction_method", method))
            print(f"junction_method={method}", flush=True)
            return method
        failures.append("_winapi.CreateJunction returned without a usable directory")
    except (ImportError, OSError, AttributeError) as exc:
        failures.append(f"_winapi.CreateJunction: {type(exc).__name__}: {exc}")

    if link.exists() or link.is_symlink():
        pytest.skip(
            "junction behavior unverified; native creation left an unusable path, "
            + "; ".join(failures)
        )
    command = f'mklink /J "{link}" "{target}"'
    try:
        result = subprocess.run(
            ["cmd.exe", "/d", "/c", command],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0 and link.exists() and link.is_dir():
            method = "mklink"
            request.node.user_properties.append(("junction_method", method))
            print(f"junction_method={method}", flush=True)
            return method
        failures.append(
            "mklink /J: "
            f"exit={result.returncode}; stdout={result.stdout!r}; stderr={result.stderr!r}"
        )
    except OSError as exc:
        failures.append(f"mklink /J launch: {type(exc).__name__}: {exc}")
    pytest.skip(
        "junction behavior unverified; both creation methods failed: " + " | ".join(failures)
    )


def _assert_real_junction_detection_and_validation(link: Path, repo: Path, target: Path) -> None:
    marker = target / "junction-target-marker.txt"
    before = marker.read_bytes()
    info = link.lstat()
    attributes = safe_paths._file_attributes(link, info)
    assert attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)

    # Exercise the real filesystem detector (no injected attribute reader).
    with pytest.raises(ValueError, match="Junction|Reparse"):
        safe_paths.checked_absolute(link)
    child = link / "must-not-be-created"
    with pytest.raises(ValueError, match="Junction|Reparse"):
        safe_paths.validate_v2_root(link, repo)
    with pytest.raises(ValueError, match="Junction|Reparse"):
        safe_paths.validate_v2_root(child, repo)
    assert not child.exists()
    assert marker.read_bytes() == before


def test_safe_paths_reject_real_directory_junction_created_with_mklink(tmp_path, request):
    repo, data_root = _temp_v2_root(tmp_path)
    data_root.rmdir()
    target = tmp_path / "junction-target"
    target.mkdir()
    marker = target / "junction-target-marker.txt"
    marker.write_bytes(b"target-preserved")
    method = _create_junction_or_skip(data_root, target, request)
    assert method in {"winapi", "mklink"}
    _assert_real_junction_detection_and_validation(data_root, repo, target)


def test_post_creation_junction_replacement_is_detected_and_not_removed(tmp_path, monkeypatch, request):
    repo, data_root = _temp_v2_root(tmp_path)
    output = data_root / "neural_ewma_residual_v1_seed_42"
    target = tmp_path / "junction-replacement-target"
    target.mkdir()
    marker = target / "owner.txt"
    marker.write_text("replacement owner", encoding="utf-8")
    marker.write_bytes(marker.read_bytes() + b"\n")
    original_verify = residual.verify_new_directory_containment
    junction_method = {}

    def replace_then_verify(path, root):
        path.rmdir()
        junction_method["method"] = _create_junction_or_skip(path, target, request)
        return original_verify(path, root)

    monkeypatch.setattr(residual, "verify_new_directory_containment", replace_then_verify)
    with pytest.raises(ValueError, match="Junction|Reparse"):
        residual._create_run_directory(output, data_root, repo)
    assert junction_method["method"] in {"winapi", "mklink"}
    assert safe_paths._file_attributes(output, output.lstat()) & getattr(
        stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400
    )
    with pytest.raises(ValueError, match="Junction|Reparse"):
        safe_paths.validate_v2_root(output, repo)
    with pytest.raises(ValueError, match="Junction|Reparse"):
        safe_paths.validate_v2_root(output / "child", repo)
    assert not (output / "child").exists()
    assert output.exists()
    assert (output / "owner.txt").read_bytes() == marker.read_bytes()
    assert marker.read_bytes() == b"replacement owner\n"
