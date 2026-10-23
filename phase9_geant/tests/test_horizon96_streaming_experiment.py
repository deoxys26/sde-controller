from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from phase9_geant import horizon96_experiment as candidate
from phase9_geant import horizon96_smoke as lazy
from phase9_geant import horizon96_streaming_data as builder
from phase9_geant import horizon96_streaming_experiment as experiment
from src.gnn.spatiotemporal_link import SpatioTemporalLinkForecaster


def _edges():
    return candidate._synthetic_edges_72()


def _windows(start: int, count: int):
    return [
        {"history_start": start + i, "origin": start + i + 671, "target": start + i + 767}
        for i in range(count)
    ]


def test_fixed_streaming_configuration_matches_approved_candidate_protocol():
    assert experiment.CONFIG["history_length"] == 672
    assert experiment.CONFIG["horizons"] == [96]
    assert experiment.CONFIG["prediction_channels"] == 1
    assert experiment.CONFIG["seeds"] == [42, 123, 2026]
    assert experiment.CONFIG["adjacency_modes"] == ["topology", "identity", "shuffled"]
    assert experiment.CONFIG["epochs"] == 20 and experiment.CONFIG["patience"] == 4
    assert experiment.CONFIG["optimizer"] == "Adam" and experiment.CONFIG["learning_rate"] == 0.005
    assert experiment.CONFIG["batch_size"] == 32 and experiment.CONFIG["hidden_size"] == 16
    assert experiment.CONFIG["ewma_alpha"] == 0.8 and experiment.CONFIG["ewma_steps"] == 5
    assert experiment.CONFIG["residual_loss"] == "SmoothL1(beta=0.001) in original utilization-ratio units"
    assert experiment.OUTPUT_ROOT_NAME == "geant_horizon96_streaming_experiment_v1"


def test_compact_lazy_dataset_alignment_and_h96_baselines_are_causal():
    series = (np.arange(769 * 72, dtype=np.float32).reshape(769, 72) / 100_000).astype(np.float32)
    windows = [{"history_start": 0, "origin": 671, "target": 767}]
    dataset = lazy.LazyH96Dataset(series, windows, split_start=0)
    history, target = dataset[0]
    assert history.shape == (672, 72) and target.shape == (72,)
    baseline = experiment._baseline_predictions(history.numpy()[None])
    assert np.array_equal(baseline["persistence"], baseline["same_time_yesterday"])
    assert np.array_equal(baseline["same_time_last_week"][0], history[95].numpy())
    weights = np.array([0.8 * 0.2 ** (4 - k) for k in range(5)], dtype=np.float64)
    weights /= weights.sum()
    expected = np.einsum("te,t->e", history[-5:].numpy().astype(np.float64), weights)
    assert np.allclose(baseline["ewma_alpha_0.8"][0], expected)


def test_metrics_require_complete_finite_aligned_pairs():
    truth = np.zeros((3, 72), dtype=np.float32)
    pred = np.ones((3, 72), dtype=np.float32)
    result = experiment._metric_values(truth, pred)
    assert result["mae"] == 1.0 and result["rmse"] == 1.0
    assert result["evaluated_pairs"] == 216
    assert len(result["per_link_mae"]) == 72 and len(result["per_origin_rmse"]) == 3
    with pytest.raises(experiment.StreamingExperimentError, match="aligned, fully finite"):
        experiment._metric_values(truth, pred[:2])
    pred[0, 0] = np.nan
    with pytest.raises(experiment.StreamingExperimentError, match="aligned, fully finite"):
        experiment._metric_values(truth, pred)


def test_source_preflight_uses_only_the_compact_train_validation_reader(tmp_path, monkeypatch):
    repository = tmp_path / "repo"
    (repository / "phase9_geant").mkdir(parents=True)
    root = repository / "data" / "processed" / builder.OUTPUT_NAME
    root.mkdir(parents=True)
    metadata_path = root / "metadata.json"
    source_hashes = {name: hashlib.sha256(name.encode()).hexdigest() for name in builder.SOURCE_FILES}
    edges = _edges()
    undirected = tuple((edges[i][0], edges[i][1]) for i in range(0, len(edges), 2))
    nodes = {f"n{i}": (float(i), float(i)) for i in range(22)}
    topology_hash = builder.EXPECTED_LOCAL_TOPOLOGY_SHA256
    edge_hash = builder.approved._edge_digest(edges)
    metadata = {
        "status": "complete", "protocol_id": builder.PROTOCOL_ID,
        "source_hashes_before_and_after": source_hashes,
        "source_scope": {"train": [0, 8020], "validation": [8020, 9740]},
        "source_file_counts": {"train": 8020, "validation": 1720},
        "array_files": {name: {"shape": [], "dtype": "synthetic", "file_sha256": "a" * 64} for name in builder.SERIES_NAMES},
        "window_index_sha256": {"train": "b" * 64, "validation": "c" * 64},
        "training_cutoff": {"scope": "only [0,8020)"},
        "test_period_filename_visibility": "names only",
        "topology": {
            "node_count": 22, "undirected_link_count": 36, "directed_link_count": 72,
            "sha256_local_file_identity_only": topology_hash,
            "ordered_edge_sha256": edge_hash,
            "directed_edges_interleaved": [list(edge) for edge in edges],
        },
    }
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    train_series = np.lib.format.open_memmap(root / "utilization_train.npy", mode="w+", dtype=np.float32, shape=(8020, 72))
    val_series = np.lib.format.open_memmap(root / "utilization_validation.npy", mode="w+", dtype=np.float32, shape=(1720, 72))
    train_valid = np.lib.format.open_memmap(root / "valid_train.npy", mode="w+", dtype=np.bool_, shape=(8020,))
    val_valid = np.lib.format.open_memmap(root / "valid_validation.npy", mode="w+", dtype=np.bool_, shape=(1720,))
    train_times = np.lib.format.open_memmap(root / "timestamps_train.npy", mode="w+", dtype=np.int64, shape=(8020,))
    val_times = np.lib.format.open_memmap(root / "timestamps_validation.npy", mode="w+", dtype=np.int64, shape=(1720,))
    for arr in (train_series, val_series):
        arr[:] = 0.1
    for arr in (train_valid, val_valid):
        arr[:] = True
    train_times[:] = np.arange(8020, dtype=np.int64) * 900
    val_times[:] = (8020 + np.arange(1720, dtype=np.int64)) * 900
    del train_series, val_series, train_valid, val_valid, train_times, val_times
    train = {"series": np.load(root / "utilization_train.npy", mmap_mode="r"), "windows": _windows(0, 3924), "valid": None, "timestamps": None, "bounds": (0, 8020)}
    validation = {"series": np.load(root / "utilization_validation.npy", mmap_mode="r"), "windows": _windows(8020, 953), "valid": None, "timestamps": None, "bounds": (8020, 9740)}
    loaded_names = []

    monkeypatch.setattr(experiment, "canonical_repository_root", lambda _path: repository)
    monkeypatch.setattr(experiment, "_source_identity", lambda _repo: {"runner": "synthetic"})
    monkeypatch.setattr(builder, "_source_hashes", lambda _repo: source_hashes.copy())
    monkeypatch.setattr(builder.approved, "_topology_snapshot", lambda *_args: (nodes, undirected, edges, topology_hash))
    monkeypatch.setattr(lazy, "_verify_data_root", lambda *_args: (root, metadata))

    def compact_reader(_root, _repository):
        loaded_names.extend(("utilization_train.npy", "valid_train.npy", "timestamps_train.npy", "window_index_train_h96.json", "utilization_validation.npy", "valid_validation.npy", "timestamps_validation.npy", "window_index_validation_h96.json"))
        return {"train": train, "validation": validation}, metadata

    monkeypatch.setattr(lazy, "load_train_validation_lazy", compact_reader)
    monkeypatch.setattr(candidate, "load_h96_train_validation", lambda *_args, **_kwargs: pytest.fail("legacy materialized loader invoked"))
    result = experiment._verified_inputs(root, repository)
    assert result[2]["train"]["series"].shape == (8020, 72)
    assert isinstance(result[2]["train"]["series"], np.memmap)
    assert len(result[2]["train"]["windows"]) == 3924
    assert len(result[2]["validation"]["windows"]) == 953
    assert all("test" not in name.casefold() for name in loaded_names)
    assert loaded_names == [
        "utilization_train.npy", "valid_train.npy", "timestamps_train.npy", "window_index_train_h96.json",
        "utilization_validation.npy", "valid_validation.npy", "timestamps_validation.npy", "window_index_validation_h96.json",
    ]


def test_synthetic_full_run_uses_compact_windows_and_finalizes_checkpoint(tmp_path, monkeypatch):
    repository = tmp_path / "repo"
    (repository / "phase9_geant").mkdir(parents=True)
    processed = repository / "data" / "processed"
    processed.mkdir(parents=True)
    data_root = processed / builder.OUTPUT_NAME
    data_root.mkdir()
    train_series = np.linspace(0.01, 0.2, 8020 * 72, dtype=np.float32).reshape(8020, 72)
    val_series = np.linspace(0.02, 0.25, 1720 * 72, dtype=np.float32).reshape(1720, 72)
    train_windows = _windows(0, 1)
    val_windows = _windows(8020, 1)
    splits = {
        "train": {"series": train_series, "windows": train_windows, "bounds": (0, 8020)},
        "validation": {"series": val_series, "windows": val_windows, "bounds": (8020, 9740)},
    }
    metadata = {
        "protocol_id": builder.PROTOCOL_ID,
        "source_file_counts": {"train": 8020, "validation": 1720},
        "array_files": {name: {"shape": [], "dtype": "synthetic", "file_sha256": "d" * 64} for name in builder.SERIES_NAMES},
        "window_index_sha256": {"train": "e" * 64, "validation": "f" * 64},
    }
    provenance = {"dataset_protocol_id": builder.PROTOCOL_ID, "array_file_records": metadata["array_files"], "source_scope": {"train": [0, 8020], "validation": [8020, 9740]}}
    source_hashes = {"runner": "synthetic-hash"}
    monkeypatch.setattr(experiment, "_verified_inputs", lambda *_args: (repository, data_root, splits, metadata, _edges(), provenance, source_hashes))
    monkeypatch.setattr(experiment, "_source_identity", lambda _repo: source_hashes.copy())
    monkeypatch.setattr(
        experiment.lazy, "resource_preflight",
        lambda *_args: lazy.ResourceCheck(
            2**31, 2**31, lazy.SMOKE_ESTIMATED_PEAK_RAM, lazy.SMOKE_RESERVE_RAM,
            "synthetic resource fixture",
        ),
    )
    monkeypatch.setattr(experiment, "CONFIG", {**experiment.CONFIG, "epochs": 1, "patience": 1, "batch_size": 1})
    result = experiment.train_one(
        data_root, repository_root=repository, seed=42, mode="identity",
        run_training=True, explicit_authorization=True,
    )
    assert result["status"] == "complete"
    assert result["training_window_count"] == 1 and result["validation_window_count"] == 1
    assert result["checkpoint_reload_strict"] is True
    output = processed / experiment.OUTPUT_ROOT_NAME / "h96_identity_seed_42"
    checkpoint = output / "best.pt"
    assert checkpoint.is_file() and checkpoint.stat().st_size > 0
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == result["checkpoint_sha256"]
    assert (output / "validation_predictions.npy").is_file()
    saved = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert saved["status"] == "complete" and saved["seed"] == 42 and saved["mode"] == "identity"


def test_training_requires_explicit_authorization_before_paths(tmp_path, monkeypatch):
    touched = False

    def forbidden(*_args, **_kwargs):
        nonlocal touched
        touched = True
        raise AssertionError("preflight must not run without explicit authorization")

    monkeypatch.setattr(experiment, "canonical_repository_root", forbidden)
    with pytest.raises(PermissionError, match="explicit authorization"):
        experiment.train_one(tmp_path, repository_root=tmp_path, seed=42, mode="topology")
    assert not touched


def test_bootstrap_is_deterministic_and_root_refuses_existing_run(tmp_path):
    x = np.linspace(-1, 1, 953)
    y = np.zeros(953)
    first = experiment._block_bootstrap_difference(x, y, metric="MAE", block_length=96, seed=11)
    second = experiment._block_bootstrap_difference(x, y, metric="MAE", block_length=96, seed=11)
    assert first == second
    repository = tmp_path / "repo"
    (repository / "phase9_geant").mkdir(parents=True)
    (repository / "data" / "processed").mkdir(parents=True)
    root, run = experiment._planned_run(repository, 42, "topology")
    root.mkdir()
    run.mkdir()
    with pytest.raises(FileExistsError):
        experiment._planned_run(repository, 42, "topology")

