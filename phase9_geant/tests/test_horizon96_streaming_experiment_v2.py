"""Synthetic-only safety tests for the versioned h=96 batch-4 runner."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from phase9_geant import horizon96_experiment as candidate
from phase9_geant import horizon96_smoke as lazy
from phase9_geant import horizon96_streaming_experiment_v2 as experiment
from phase9_geant.colab_h96_v1 import linux_resources


def _snapshot(rss=300 * 1024**2, peak=420 * 1024**2, available=1024**3):
    return {
        "process_rss_bytes": rss,
        "process_peak_rss_or_working_set_bytes": peak,
        "process_peak_measurement_source": "synthetic_high_water",
        "os_available_ram_bytes": available,
    }


def _edges():
    return candidate._synthetic_edges_72()


class _CountingDataset(Dataset):
    def __init__(self, count: int):
        self.count = count
        self.requests: list[int] = []

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        self.requests.append(int(index))
        history = torch.full((672, 72), 0.1 + index / 1000, dtype=torch.float32)
        target = torch.full((72,), 0.2 + index / 1000, dtype=torch.float32)
        return history, target


class _TinyResidualModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.offset = torch.nn.Parameter(torch.tensor(0.01))
        raw = torch.tensor([0.8 * 0.2**(4 - i) for i in range(5)], dtype=torch.float64)
        self.register_buffer("ewma_weights", (raw / raw.sum()).float())

    def forward(self, history):
        baseline = torch.einsum("bte,t->be", history[:, -5:, :], self.ewma_weights)
        return (baseline + self.offset).unsqueeze(1)


def test_v2_isolated_protocol_uses_one_batch_size_and_preserves_scientific_settings():
    assert experiment.PROTOCOL_ID == "geant-horizon96-only-candidate-v2-batch4"
    assert experiment.OUTPUT_ROOT_NAME == "geant_horizon96_streaming_experiment_v2"
    assert experiment.CONFIG["batch_size"] == 4
    assert experiment.CONFIG["history_length"] == 672
    assert experiment.CONFIG["horizons"] == [96]
    assert experiment.CONFIG["ewma_alpha"] == 0.8
    assert experiment.CONFIG["residual_loss"].startswith("SmoothL1(beta=0.001)")
    assert experiment.CONFIG["learning_rate"] == 0.005
    assert experiment.CONFIG["epochs"] == 20 and experiment.CONFIG["patience"] == 4
    assert experiment.CONFIG["seeds"] == [42, 123, 2026]
    assert experiment.CONFIG["adjacency_modes"] == ["topology", "identity", "shuffled"]
    assert experiment.CONFIG["prior_batch_size"] == 32
    assert "optimizer updates per epoch therefore differ" in experiment.CONFIG["optimization_protocol_amendment"]
    counts = []
    for mode in experiment.MODES:
        model = experiment.SpatioTemporalLinkForecaster(
            _edges(), history_length=672, horizons=(96,), hidden_size=16,
            alpha=0.8, ewma_steps=5, adjacency_mode=mode, shuffle_seed=20261019,
        )
        counts.append(sum(parameter.numel() for parameter in model.parameters()))
    assert counts == [counts[0]] * len(experiment.MODES)


def test_memory_monitor_separates_process_peak_from_system_available_ram():
    # 500 MiB is below the 1 GiB startup gate but above the in-run 384 MiB reserve.
    monitor = experiment.MemoryMonitor(snapshot_fn=lambda: _snapshot(available=500 * 1024**2))
    assert monitor.sample("synthetic") ["os_available_ram_bytes"] == 500 * 1024**2
    summary = monitor.summary()
    assert summary["max_platform_process_high_water_bytes"] == 420 * 1024**2
    assert summary["min_os_available_ram_bytes"] == 500 * 1024**2
    assert summary["process_peak_limit_bytes"] == 640 * 1024**2
    assert summary["in_run_os_available_reserve_bytes"] == 384 * 1024**2

    fallback = experiment.MemoryMonitor(snapshot_fn=lambda: {
        "process_rss_bytes": 310 * 1024**2,
        "process_peak_rss_or_working_set_bytes": None,
        "process_peak_measurement_source": "sampled_rss_fallback",
        "os_available_ram_bytes": 500 * 1024**2,
    })
    fallback.sample("fallback")
    assert fallback.summary()["max_platform_process_high_water_bytes"] is None
    assert fallback.summary()["max_process_memory_observation_used_for_limit_bytes"] == 310 * 1024**2

    with pytest.raises(experiment.MemoryLimitError, match="Process memory observation"):
        experiment.MemoryMonitor(
            snapshot_fn=lambda: _snapshot(peak=641 * 1024**2)
        ).sample("process-over-limit")
    with pytest.raises(experiment.MemoryLimitError, match="OS-available RAM"):
        experiment.MemoryMonitor(
            snapshot_fn=lambda: _snapshot(available=383 * 1024**2)
        ).sample("system-under-reserve")


def test_training_epoch_uses_bounded_batches_and_clears_gradients(monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setitem(experiment.CONFIG, "batch_size", 4)
    dataset = _CountingDataset(9)
    model = _TinyResidualModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    monitor = experiment.MemoryMonitor(snapshot_fn=_snapshot)
    loss, stats = experiment._training_epoch(
        model, dataset, type("Scaler", (), {"mean": 0.0, "scale": 1.0})(),
        optimizer, seed=42, epoch=1, monitor=monitor,
    )
    assert np.isfinite(loss)
    assert stats["optimizer_steps"] == 3  # ceil(9/4), with no gradient accumulation
    assert len(dataset.requests) == 9
    assert sorted(dataset.requests) == list(range(9))
    assert all(parameter.grad is None for parameter in model.parameters())
    assert monitor.samples == 3


def test_validation_selection_pass_does_not_retain_prediction_arrays():
    torch.set_num_threads(1)
    model = experiment.SpatioTemporalLinkForecaster(
        _edges(), history_length=672, horizons=(96,), hidden_size=16,
        alpha=0.8, ewma_steps=5, adjacency_mode="identity", shuffle_seed=20261019,
    )
    history = torch.full((2, 672, 72), 0.15, dtype=torch.float32)
    target = torch.full((2, 72), 0.2, dtype=torch.float32)
    loader = [(history, target)]
    scaler = type("Scaler", (), {"mean": 0.0, "scale": 1.0})()
    monitor = experiment.MemoryMonitor(snapshot_fn=_snapshot)
    objective, predictions, targets, baselines = experiment._evaluate(
        model, loader, scaler, monitor, collect_arrays=False,
    )
    assert np.isfinite(objective)
    assert predictions is targets is baselines is None
    assert monitor.samples == 1
    objective2, predictions2, targets2, baselines2 = experiment._evaluate(
        model, loader, scaler, experiment.MemoryMonitor(snapshot_fn=_snapshot), collect_arrays=True,
    )
    assert objective2 == objective
    assert predictions2.shape == targets2.shape == (2, 72)
    assert set(baselines2) == {"ewma_alpha_0.8", "persistence", "same_time_yesterday", "same_time_last_week"}


def test_checkpoint_hash_is_verified_before_strict_load(tmp_path):
    torch.set_num_threads(1)
    model = experiment.SpatioTemporalLinkForecaster(
        _edges(), history_length=672, horizons=(96,), hidden_size=16,
        alpha=0.8, ewma_steps=5, adjacency_mode="identity", shuffle_seed=20261019,
    )
    path = tmp_path / "synthetic.pt"
    config_hash = "f" * 64
    digest = candidate._atomic_checkpoint({
        "state_dict": model.state_dict(), "seed": 42,
        "protocol_id": experiment.PROTOCOL_ID,
        "configuration_sha256": config_hash,
    }, path)
    experiment._load_checkpoint_verified(
        model, path, digest, seed=42, configuration_sha256=config_hash,
    )
    raw = path.read_bytes()
    path.write_bytes(raw + b"synthetic-corruption")
    with pytest.raises(experiment.ExperimentV2Error, match="SHA-256 changed"):
        experiment._load_checkpoint_verified(
            model, path, digest, seed=42, configuration_sha256=config_hash,
        )


def test_existing_v2_run_directory_is_refused_before_input_loader(tmp_path, monkeypatch):
    repository = tmp_path / "repo"
    processed = repository / "data" / "processed"
    processed.mkdir(parents=True)
    output_root = processed / experiment.OUTPUT_ROOT_NAME
    run = output_root / "h96_topology_seed_42"
    run.mkdir(parents=True)
    monkeypatch.setattr(experiment.lazy, "resource_preflight", lambda *_: pytest.fail("resource/data step ran"))
    monkeypatch.setattr(experiment, "_verified_inputs", lambda *_: pytest.fail("input loader ran"))
    with pytest.raises(FileExistsError, match="Refusing existing"):
        experiment.train_one(
            processed / "geant_horizon96_streaming_v1", repository_root=repository,
            seed=42, mode="topology", run_training=True, explicit_authorization=True,
            monitor=experiment.MemoryMonitor(snapshot_fn=_snapshot),
        )


@pytest.mark.parametrize("stop_kind", ["keyboard-interrupt", "memory-limit", "colab-cgroup-limit"])
def test_interrupted_run_records_failure_and_preserves_last_good_checkpoint(tmp_path, monkeypatch, stop_kind):
    torch.set_num_threads(1)
    repository = tmp_path / "repo"
    processed = repository / "data" / "processed"
    processed.mkdir(parents=True)
    data_root = processed / "synthetic_train_validation_only"
    train_series = np.linspace(0.01, 0.2, 8020 * 72, dtype=np.float32).reshape(8020, 72)
    validation_series = np.linspace(0.02, 0.25, 1720 * 72, dtype=np.float32).reshape(1720, 72)
    train_windows = [{"history_start": 0, "origin": 671, "target": 767}]
    validation_windows = [{"history_start": 8020, "origin": 8691, "target": 8787}]
    splits = {
        "train": {"series": train_series, "windows": train_windows},
        "validation": {"series": validation_series, "windows": validation_windows},
    }
    metadata = {"synthetic": True, "split": "train_validation_only"}
    provenance = {"synthetic_fixture": True}
    source_hashes = {"synthetic-source": hashlib.sha256(b"fixture").hexdigest()}
    monkeypatch.setattr(
        experiment, "_verified_inputs",
        lambda *_: (repository, data_root, splits, metadata, _edges(), provenance, source_hashes),
    )
    monkeypatch.setattr(experiment, "_source_identity", lambda _repo: source_hashes.copy())
    monkeypatch.setattr(
        experiment.lazy, "resource_preflight",
        lambda *_: lazy.ResourceCheck(2**31, 2**31, 640 * 1024**2, 384 * 1024**2, "synthetic"),
    )
    monkeypatch.setattr(experiment, "CONFIG", {**experiment.CONFIG, "batch_size": 1, "epochs": 2, "patience": 2})

    real_epoch = experiment._training_epoch
    calls = 0

    def interrupt_second_epoch(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            if stop_kind == "keyboard-interrupt":
                raise KeyboardInterrupt("synthetic interruption")
            if stop_kind == "colab-cgroup-limit":
                active_monitor = kwargs.get("monitor", args[-1])
                active_monitor.snapshot_fn = lambda: {
                    "process_rss_bytes": 500 * 1024**2,
                    "process_peak_bytes": 600 * 1024**2,
                    "process_peak_source": "synthetic ru_maxrss",
                    "os_available_ram_bytes": 3 * 1024**3,
                    "disk_free_bytes": 2 * 1024**3,
                    "cgroup_source": "synthetic cgroup_v2",
                    "cgroup_status": "finite",
                    "cgroup_memory_usage_bytes": 900 * 1024**2,
                    "cgroup_memory_limit_bytes": 1000 * 1024**2,
                    "cgroup_memory_headroom_bytes": 100 * 1024**2,
                    "cgroup_visibility_uncertainty": ["synthetic hidden ancestor"],
                    "cgroup_visible_levels": [{
                        "path": "synthetic-cgroup", "usage_bytes": 900 * 1024**2,
                        "limits": [{"name": "memory.max", "limit_bytes": 1000 * 1024**2, "kind": "finite"}],
                    }],
                    "cgroup_memory_events": None,
                }
                return real_epoch(*args, **kwargs)
            active_monitor = kwargs.get("monitor", args[-1])
            active_monitor.snapshot_fn = lambda: _snapshot(available=1)
            return real_epoch(*args, **kwargs)
        return real_epoch(*args, **kwargs)

    monkeypatch.setattr(experiment, "_training_epoch", interrupt_second_epoch)
    if stop_kind == "keyboard-interrupt":
        expected_exception = KeyboardInterrupt
        expected_failure = "KeyboardInterrupt: synthetic interruption"
        active_monitor = experiment.MemoryMonitor(snapshot_fn=_snapshot)
    elif stop_kind == "colab-cgroup-limit":
        expected_exception = linux_resources.LinuxResourceError
        expected_failure = "LinuxResourceError: Cgroup memory headroom is below reserve 402653184 bytes"
        calls_to_monitor = 0

        def colab_snapshot():
            nonlocal calls_to_monitor
            calls_to_monitor += 1
            headroom = 4 * 1024**3 if calls_to_monitor <= 3 else 100 * 1024**2
            return {
                "process_rss_bytes": 500 * 1024**2,
                "process_peak_bytes": 600 * 1024**2,
                "process_peak_source": "synthetic ru_maxrss",
                "os_available_ram_bytes": 3 * 1024**3,
                "disk_free_bytes": 2 * 1024**3,
                "cgroup_source": "synthetic cgroup_v2",
                "cgroup_status": "finite",
                "cgroup_memory_usage_bytes": 4 * 1024**3 - headroom,
                "cgroup_memory_limit_bytes": 4 * 1024**3,
                "cgroup_memory_headroom_bytes": headroom,
                "cgroup_visibility_uncertainty": ["synthetic hidden ancestor"],
                "cgroup_visible_levels": [{
                    "path": "synthetic-cgroup", "usage_bytes": 4 * 1024**3 - headroom,
                    "limits": [{"name": "memory.max", "limit_bytes": 4 * 1024**3, "kind": "finite"}],
                }],
                "cgroup_memory_events": None,
            }
        active_monitor = linux_resources.LinuxMemoryMonitor(snapshot_fn=colab_snapshot)
    else:
        expected_exception = experiment.MemoryLimitError
        expected_failure = "MemoryLimitError: OS-available RAM 1 is below in-run reserve"
        active_monitor = experiment.MemoryMonitor(snapshot_fn=_snapshot)
    with pytest.raises(expected_exception) as captured:
        experiment.train_one(
            data_root, repository_root=repository, seed=42, mode="topology",
            run_training=True, explicit_authorization=True,
            monitor=active_monitor,
        )
    output = processed / experiment.OUTPUT_ROOT_NAME / "h96_topology_seed_42"
    manifest = json.loads((output / "run.json").read_text(encoding="utf-8"))
    checkpoint = output / "best.pt"
    assert isinstance(captured.value, expected_exception)
    assert manifest["status"] == "failed"
    assert manifest["failure"].startswith(expected_failure)
    assert manifest["best_epoch_so_far"] == 1
    assert checkpoint.is_file() and checkpoint.stat().st_size > 0
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == manifest["checkpoint_sha256_so_far"]
