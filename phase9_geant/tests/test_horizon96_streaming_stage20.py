from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
import torch

from phase9_geant import horizon96_streaming_data as build
from phase9_geant import horizon96_smoke as smoke
from phase9_geant.horizon_v2_experiment import fit_training_scaler
from phase9_geant.validity_v2 import MatrixObservation


def _times(count: int, start: datetime = datetime(2005, 5, 4, 15, 0)):
    return tuple(start + timedelta(minutes=15 * index) for index in range(count))


def test_h96_origin_counts_and_exact_target_alignment_are_split_local():
    train_valid = np.ones(8020, dtype=np.bool_)
    train_times = np.arange(8020, dtype=np.int64) * 900
    train = build.enumerate_h96_records("train", train_valid, train_times)
    assert len(train) == 7253
    assert (train[0].history_start, train[0].origin, train[0].target) == (0, 671, 767)
    assert (train[-1].history_start, train[-1].origin, train[-1].target) == (7252, 7923, 8019)

    validation_valid = np.ones(1720, dtype=np.bool_)
    validation_times = 8020 * 900 + np.arange(1720, dtype=np.int64) * 900
    validation = build.enumerate_h96_records("validation", validation_valid, validation_times)
    assert len(validation) == 953
    assert (validation[0].history_start, validation[0].origin, validation[0].target) == (8020, 8691, 8787)
    assert validation[-1].target == 9739
    with pytest.raises(ValueError, match="Only train and validation"):
        build.enumerate_h96_records("test", np.ones(2), np.arange(2))


def test_h96_window_rejects_invalid_history_target_and_timestamp_gap():
    valid = np.ones(8020, dtype=np.bool_)
    timestamps = np.arange(8020, dtype=np.int64) * 900
    valid[17] = False
    records = build.enumerate_h96_records("train", valid, timestamps)
    assert all(not (item.history_start <= 17 <= item.origin or item.target == 17) for item in records)
    valid[:] = True
    timestamps[700] += 1
    records_gap = build.enumerate_h96_records("train", valid, timestamps)
    assert all(not (item.history_start < 700 <= item.target) for item in records_gap)


def test_lazy_dataset_returns_only_exact_history_and_h96_target():
    series = np.arange(8020 * 72, dtype=np.float32).reshape(8020, 72)
    window = {"history_start": 10, "origin": 681, "target": 777}
    dataset = smoke.LazyH96Dataset(series, [window], split_start=0)
    history, target = dataset[0]
    assert history.shape == (672, 72)
    assert target.shape == (72,)
    assert torch.equal(history[0], torch.from_numpy(series[10]))
    assert torch.equal(history[-1], torch.from_numpy(series[681]))
    assert torch.equal(target, torch.from_numpy(series[777]))


def test_smoke_subset_is_fixed_bounded_and_deterministic():
    windows = [
        {"history_start": index, "origin": index + 671, "target": index + 767}
        for index in range(500)
    ]
    selected_a = smoke.bounded_smoke_records(windows, smoke.SMOKE_TRAIN_WINDOWS)
    selected_b = smoke.bounded_smoke_records(windows, smoke.SMOKE_TRAIN_WINDOWS)
    assert len(selected_a) == 32
    assert selected_a == selected_b
    assert selected_a[0] == windows[0] and selected_a[-1] == windows[-1]
    assert len({row["origin"] for row in selected_a}) == len(selected_a)


def test_lazy_dataset_rejects_cross_boundary_and_wrong_alignment():
    series = np.ones((1720, 72), dtype=np.float32)
    with pytest.raises(smoke.H96SmokeIntegrityError, match="crosses its split"):
        smoke.LazyH96Dataset(series, [{"history_start": 8973, "origin": 9644, "target": 9740}], 8020)
    with pytest.raises(smoke.H96SmokeIntegrityError, match="alignment"):
        smoke.LazyH96Dataset(series, [{"history_start": 8020, "origin": 8691, "target": 8787}], 0)


def test_lazy_scaler_matches_materialized_h96_scaler_population():
    rng = np.random.default_rng(91)
    series = rng.normal(size=(802, 72)).astype(np.float32)
    windows = [
        {"history_start": 0, "origin": 671, "target": 767},
        {"history_start": 20, "origin": 691, "target": 787},
    ]
    lazy = smoke.fit_window_scaler(series, windows)
    histories = np.stack([series[item["history_start"] : item["origin"] + 1] for item in windows])
    targets = np.stack([series[item["target"]] for item in windows])
    reference = fit_training_scaler({96: histories}, {96: targets})
    assert lazy.mean == pytest.approx(reference.mean, rel=1e-12, abs=1e-12)
    assert lazy.scale == pytest.approx(reference.scale, rel=1e-12, abs=1e-12)
    assert lazy.population_count == histories.size + targets.size
    assert "overlapping" not in lazy.fit_scope


def test_scaler_api_has_no_validation_or_test_input_and_is_train_only():
    train = np.full((800, 72), 0.25, dtype=np.float32)
    windows = [{"history_start": 0, "origin": 671, "target": 767}]
    first = smoke.fit_window_scaler(train, windows)
    changed_unrelated_holdout = np.full((800, 72), 9000.0, dtype=np.float32)
    second = smoke.fit_window_scaler(train.copy(), windows)
    assert first == second
    assert changed_unrelated_holdout.shape == train.shape  # not accepted by the scaler API


def test_cutoff_streamer_reads_one_synthetic_observation_per_training_filename(monkeypatch):
    count = 8020
    paths = tuple(Path(f"synthetic-{index:05d}.txt") for index in range(count))
    stamps = _times(count)
    seen: list[str] = []

    def fake_parse(path, _stamp, _nodes, _index, _parser, *, expected_hash=None):
        assert expected_hash is None
        seen.append(path.name)
        index = len(seen) - 1
        matrix = np.array([[float(index + 1)]], dtype=np.float64)
        observation = MatrixObservation(stamps[index], matrix, 0 if index == 0 else 1, path.name)
        return observation, f"{index:064x}"

    monkeypatch.setattr(build, "_parse_checked", fake_parse)
    cutoff, hashes, totals, contributing, empty = build._fit_cutoff_streaming(
        paths, stamps, ("n0",), {"n0": 0}, lambda *_args: None
    )
    assert len(seen) == count and tuple(seen) == tuple(path.name for path in paths)
    assert len(hashes) == count
    assert len(totals) == count - 1 and len(contributing) == count - 1
    assert empty == 1
    assert cutoff == pytest.approx(5 * np.median(np.arange(2, count + 1, dtype=np.float64)))


def test_builder_artifact_allowlist_is_h96_train_validation_only():
    assert build.BASE_ARTIFACT_NAMES == {
        "utilization_train.npy", "valid_train.npy", "timestamps_train.npy",
        "utilization_validation.npy", "valid_validation.npy", "timestamps_validation.npy",
        "window_index_train_h96.json", "window_index_validation_h96.json", "metadata.json",
    }
    assert not any("test" in name.casefold() or "h4" in name.casefold() for name in build.BASE_ARTIFACT_NAMES)
    assert smoke.DATA_ALLOWED_NAMES == build.BASE_ARTIFACT_NAMES


def test_invalid_utilization_row_remains_nan_without_zero_filling():
    observation = MatrixObservation(_times(1)[0], np.zeros((2, 2), dtype=np.float64), 0, "synthetic-empty")
    nodes = ("a", "b")
    edges = (("a", "b"), ("b", "a"))
    paths = {("a", "b"): ["a", "b"], ("b", "a"): ["b", "a"]}
    row, valid, reasons = build._utilization_row(observation, 10.0, nodes, edges, paths)
    assert not valid
    assert "empty" in reasons
    assert np.isnan(row).all()


def test_resource_preflight_enforces_ram_and_disk_without_allocations(tmp_path):
    result = smoke.resource_preflight(
        tmp_path, available_ram_bytes=2 * 1024**3, free_disk_bytes=2 * 1024**3
    )
    assert result.estimated_peak_bytes == smoke.SMOKE_ESTIMATED_PEAK_RAM
    with pytest.raises(MemoryError, match="Insufficient RAM"):
        smoke.resource_preflight(tmp_path, available_ram_bytes=100, free_disk_bytes=2 * 1024**3)
    with pytest.raises(OSError, match="Free disk"):
        smoke.resource_preflight(tmp_path, available_ram_bytes=2 * 1024**3, free_disk_bytes=100)
    with pytest.raises(MemoryError, match="available RAM"):
        build.require_build_resources(build.ResourceSnapshot(10, 2 * 1024**3, "synthetic"))


def test_smoke_requires_both_explicit_switches_before_any_path_access(monkeypatch, tmp_path):
    touched = False

    def forbidden(*_args, **_kwargs):
        nonlocal touched
        touched = True
        raise AssertionError("path access before explicit smoke authorization")

    monkeypatch.setattr(smoke, "canonical_repository_root", forbidden)
    with pytest.raises(PermissionError, match="both --run-smoke"):
        smoke.run_authorized_smoke(
            data_root=tmp_path / "data", repository_root=tmp_path,
            run_smoke=True, explicit_authorization=False,
        )
    assert not touched


def test_existing_smoke_namespace_is_refused_before_dataset_loader(tmp_path, monkeypatch):
    repository = tmp_path / "repo"
    processed = repository / "data" / "processed"
    processed.mkdir(parents=True)
    (repository / "phase9_geant").mkdir()
    smoke_root = processed / smoke.OUTPUT_NAME
    smoke_root.mkdir()
    called = False

    def forbidden(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("existing destination must be rejected before data loading")

    monkeypatch.setattr(smoke, "load_train_validation_lazy", forbidden)
    with pytest.raises(FileExistsError, match="existing smoke output namespace"):
        smoke.run_authorized_smoke(
            data_root=repository / "data" / "processed" / build.OUTPUT_NAME,
            repository_root=repository, run_smoke=True, explicit_authorization=True,
        )
    assert not called


def test_smoke_cli_has_no_default_training_action():
    with pytest.raises(SystemExit) as error:
        smoke.main([])
    assert error.value.code == 2


def test_builder_refuses_existing_versioned_root_before_source_access(tmp_path, monkeypatch):
    repository = tmp_path / "repo"
    (repository / "phase9_geant").mkdir(parents=True)
    processed = repository / "data" / "processed"
    processed.mkdir(parents=True)
    output = processed / build.OUTPUT_NAME
    output.mkdir()
    touched = False

    def forbidden(*_args, **_kwargs):
        nonlocal touched
        touched = True
        raise AssertionError("existing build destination must be rejected before source selection")

    monkeypatch.setattr(build, "system_resources", forbidden)
    with pytest.raises(FileExistsError, match="Refusing existing h=96 streaming root"):
        build.build_h96_train_validation(repository_root=repository, output_root=output)
    assert not touched


def test_synthetic_cpu_benchmark_uses_intended_batch_and_is_bounded():
    result = smoke.run_synthetic_benchmark(batch_size=2, steps=1)
    assert result["input_shape"] == [2, 672, 72]
    assert result["target_shape"] == [2, 72]
    assert len(result["step_seconds"]) == 1 and result["step_seconds"][0] > 0
    assert "synthetic tensors only" in result["scope"]
    with pytest.raises(ValueError, match="batch size two"):
        smoke.run_synthetic_benchmark(batch_size=32, steps=1)
