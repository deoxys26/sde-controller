from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from phase9_geant import diagnose_train_only_stage9 as audit


def _synthetic_topology():
    nodes = {f"n{i}": (0.0, 0.0) for i in range(22)}
    links = [(f"n{i}", f"n{(i + 1) % 22}") for i in range(22)]
    links.extend((f"n{i}", f"n{i + 3}") for i in range(14))
    assert len(links) == 36
    return nodes, links


def test_diagnostic_opens_only_synthetic_train_arrays_and_provenance(tmp_path, monkeypatch):
    """The full diagnostic path is constrained to its explicit train allowlist."""
    data_root = tmp_path / "synthetic_data"
    data_root.mkdir()
    output_path = tmp_path / "diagnostic.json"
    observations_count = 1200
    index = np.arange(observations_count, dtype=np.float64)
    series = np.stack(
        [
            0.15
            + 0.04 * np.sin(index * 2 * np.pi / (96 + link % 11))
            + 0.01 * np.sin(index / (13 + link % 7))
            for link in range(72)
        ],
        axis=1,
    ).astype(np.float32)
    x_train = np.stack([series[start : start + 5] for start in range(observations_count - 5)])
    y_train = series[5:].copy()
    np.save(data_root / "X_train.npy", x_train, allow_pickle=False)
    np.save(data_root / "y_train.npy", y_train, allow_pickle=False)
    provenance_path = data_root / "window_provenance_train_validation.json"
    provenance_path.write_text(
        json.dumps(
            {
                "anchor_timestamp": "2005-05-04T15:00:00",
                "step_seconds": 900,
                "splits": {"train": {"window_starts": list(range(observations_count - 5))}},
                # The diagnostic must not use or inspect these holdout fields.
                "validation": {"window_starts": [999999]},
            }
        ),
        encoding="utf-8",
    )

    opened_arrays: list[str] = []
    opened_text: list[str] = []
    topology_calls: list[str] = []
    real_numpy_load = np.load
    real_read_text = Path.read_text

    def tracked_numpy_load(file, *args, **kwargs):
        path = Path(file)
        opened_arrays.append(path.name)
        assert path.name in audit.ALLOWED_ARRAY_NAMES
        return real_numpy_load(file, *args, **kwargs)

    def tracked_read_text(path: Path, *args, **kwargs):
        opened_text.append(path.name)
        assert path.name == "window_provenance_train_validation.json"
        return real_read_text(path, *args, **kwargs)

    def synthetic_topology_parser(path: str):
        topology_calls.append(path)
        return _synthetic_topology()

    monkeypatch.setattr(np, "load", tracked_numpy_load)
    monkeypatch.setattr(Path, "read_text", tracked_read_text)

    topology_path = tmp_path / "synthetic_geant_topology.txt"
    result = audit.run_diagnostics(
        data_root,
        provenance_path,
        output_path,
        topology_path,
        topology_parser=synthetic_topology_parser,
    )

    assert opened_arrays == ["X_train.npy", "y_train.npy"]
    assert opened_text == ["window_provenance_train_validation.json"]
    assert topology_calls == [str(topology_path)]
    assert result["split"] == "train only"
    assert result["target_distribution"]["target_count"] == y_train.size
    assert set(result["training_baselines"]["metrics"]) == {
        "last_value_persistence",
        "five_step_simple_mean",
        "ewma_alpha_0.2",
        "ewma_alpha_0.4",
        "ewma_alpha_0.6",
        "ewma_alpha_0.8",
        "ewma_alpha_0.95",
    }
    assert all(
        row["evaluated_window_link_pairs"] == y_train.size
        for row in result["training_baselines"]["metrics"].values()
    )
    assert result["chronological_ridge_signal_check"]["embargo_steps"] == 96
    assert result["topology_node_count"] == 22
    assert result["topology_undirected_link_count"] == 36
    assert result["directed_link_count"] == 72
    assert result["spatial_signal"]["shuffle_replicates"] == 20
    assert output_path.is_file()


def test_diagnostic_array_loader_rejects_nontrain_names_before_numpy_load(tmp_path, monkeypatch):
    calls: list[str] = []

    def forbidden_load(path, *args, **kwargs):
        calls.append(Path(path).name)
        raise AssertionError("np.load must not be called for an unapproved name")

    monkeypatch.setattr(np, "load", forbidden_load)
    for name in ("X_validation.npy", "y_test.npy", "X_test_future.npy"):
        with pytest.raises(ValueError, match="explicit train allowlist"):
            audit._load_train_array(tmp_path, name)
    assert calls == []


def test_directed_neighbors_use_interleaved_edges_and_shuffle_is_reproducible():
    nodes, links = _synthetic_topology()
    edges, neighbors = audit._directed_neighbors(links, 72)
    assert edges[:4] == [
        ("n0", "n1"),
        ("n1", "n0"),
        ("n1", "n2"),
        ("n2", "n1"),
    ]
    for index, (src, dst) in enumerate(edges):
        reverse_index = edges.index((dst, src))
        assert reverse_index in neighbors[index]
        assert index not in neighbors[index]

    observations = {q: np.full(72, q / 100.0) + np.arange(72) / 1000.0 for q in range(1, 12)}
    residuals = {q: np.sin(np.arange(72) + q) for q in range(2, 12)}
    first = audit._spatial_signal_summary(observations, residuals, neighbors)
    second = audit._spatial_signal_summary(observations, residuals, neighbors)
    assert first == second
    assert first["shuffle_replicates"] == 20
    assert first["target_indices"]["count"] == 10
