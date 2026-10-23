from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from phase9_geant import horizon_v2_data
from phase9_geant import horizon96_experiment as h96
from src.gnn.spatiotemporal_link import SpatioTemporalLinkForecaster


def _edges_72():
    return h96._synthetic_edges_72()


def _model(mode="topology"):
    return SpatioTemporalLinkForecaster(
        _edges_72(), history_length=672, horizons=(96,), hidden_size=16,
        alpha=0.8, ewma_steps=5, adjacency_mode=mode, shuffle_seed=20261019,
    )


def _make_h96_dataset(tmp_path, monkeypatch):
    repo = tmp_path / "synthetic_repo"
    (repo / "phase9_geant").mkdir(parents=True)
    processed = repo / "data" / "processed"
    processed.mkdir(parents=True)
    root = processed / f"{horizon_v2_data.OUTPUT_PREFIX}synthetic-fixture"
    root.mkdir()
    edges = _edges_72()
    nodes = {f"n{i}": (float(i), float(i)) for i in range(22)}
    undirected = [(f"n{i}", f"n{(i + 1) % 22}") for i in range(22)]
    undirected.extend((f"n{i}", f"n{i + 5}") for i in range(14))
    topology_hash = "b" * 64
    monkeypatch.setattr(
        h96.data_v2, "_topology_snapshot",
        lambda _path, _repo: (nodes, undirected, edges, topology_hash),
    )
    source_hashes = {
        name: hashlib.sha256(name.encode()).hexdigest()
        for name in (
            "phase9_geant/horizon_v2_data.py", "phase9_geant/horizon_windows.py",
            "phase9_geant/validity_v2.py", "phase9_geant/parse_geant.py",
            "phase9_geant/build_link_series.py",
        )
    }
    monkeypatch.setattr(h96.data_v2, "_current_build_source_hashes", lambda _repo: source_hashes.copy())

    artifact_names = sorted(horizon_v2_data._EXPECTED_ARTIFACT_NAMES)
    for name in artifact_names:
        if name.endswith("h4.npy"):
            (root / name).write_bytes(b"synthetic h4 placeholder; loader must not open")

    split_records = {}
    indexes = {}
    written_h96 = {}
    for split, (lower, upper) in horizon_v2_data.OFFICIAL_SPLIT_BOUNDS.items():
        source_names = [
            f"demandMatrix-geant-uhlig-15min-{(horizon_v2_data.EXPECTED_START + i * horizon_v2_data.STEP):%Y%m%d-%H%M}.txt"
            for i in range(lower, upper)
        ]
        history = (
            np.arange(2 * 672 * 72, dtype=np.float32).reshape(2, 672, 72) / 100_000
        ).astype(np.float32)
        target = np.full((2, 72), 0.25, dtype=np.float32)
        history_name, target_name = f"history_{split}_h96.npy", f"target_{split}_h96.npy"
        np.save(root / history_name, history, allow_pickle=False)
        np.save(root / target_name, target, allow_pickle=False)
        written_h96[(split, "history")] = history
        written_h96[(split, "target")] = target
        origins = [lower + 671, lower + 672]
        rows96 = [
            {"history_start": origin - 671, "origin": origin, "target": origin + 96}
            for origin in origins
        ]
        # Deliberately malformed h=4 row: it must not enter h=96 validation.
        indexes[split] = {"4": [{"not": "used"}], "96": rows96}
        (root / f"window_index_{split}.json").write_text(json.dumps(indexes[split]), encoding="utf-8")
        split_records[split] = {
            "bounds": [lower, upper],
            "source_file_names": source_names,
            "parsed_source_semantic_sha256": "d" * 64,
            "horizons": {
                "96": {
                    "history_shape": list(history.shape),
                    "target_shape": list(target.shape),
                    "dtype": str(history.dtype),
                    "history_sha256": h96._array_payload_sha256(history),
                    "target_sha256": h96._array_payload_sha256(target),
                }
            },
        }
    metadata = {
        "protocol_id": horizon_v2_data.PROTOCOL_ID,
        "status": "complete",
        "source_scope": {
            split: list(bounds) for split, bounds in horizon_v2_data.OFFICIAL_SPLIT_BOUNDS.items()
        },
        "source_hashes_at_build": source_hashes,
        "build_configuration": horizon_v2_data.BUILD_CONFIGURATION,
        "build_configuration_sha256": h96._canonical_json_sha256(horizon_v2_data.BUILD_CONFIGURATION),
        "window": {"history_length": 672, "horizons": [4, 96], "step_seconds": 900},
        "topology": {
            "nodes": list(nodes), "undirected_links": 36, "directed_links": 72,
            "sha256_local_file_identity_only": topology_hash,
            "directed_edges_interleaved": [list(edge) for edge in edges],
            "ordered_edge_sha256": horizon_v2_data._edge_digest(edges),
            "upstream_provenance": "synthetic fixture",
        },
        "splits": split_records,
        "artifact_names": artifact_names,
    }
    (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return repo, root, written_h96


def test_h96_configuration_and_model_have_one_prediction_channel():
    assert h96.PROTOCOL_ID == "geant-horizon96-only-candidate-v1"
    assert h96.HORIZONS == (96,)
    assert h96.H96_CONFIG["horizons"] == [96]
    assert h96.H96_CONFIG["prediction_channels"] == 1
    model = _model()
    assert model.horizons == (96,)
    assert model.residual_head.out_features == 1
    assert "target" not in inspect.signature(model.forward).parameters


def test_chunked_array_hash_matches_numpy_payload_hash():
    fixture = np.arange(3 * 5 * 7, dtype=np.float32).reshape(3, 5, 7)
    assert h96._array_payload_sha256(fixture, chunk_elements=11) == hashlib.sha256(fixture.tobytes()).hexdigest()


def test_h4_cannot_enter_objective_metrics_or_checkpoint_choice():
    prediction = torch.zeros((2, 72))
    baseline = torch.ones((2, 72))
    target = torch.full((2, 72), 1.5)
    assert torch.isfinite(h96.objective_h96(prediction, target, baseline))
    with pytest.raises(ValueError, match=r"align as \[batch, directed_link\]"):
        h96.objective_h96(torch.zeros((2, 2, 72)), torch.zeros((2, 2, 72)), torch.zeros((2, 2, 72)))
    with pytest.raises(ValueError):
        h96.metric_summary_h96(np.zeros((2, 2, 72)), np.zeros((2, 72)))
    assert h96.should_select_h96_checkpoint(0.2, 0.3)
    assert not h96.should_select_h96_checkpoint(0.3, 0.2)
    assert not h96.should_select_h96_checkpoint(float("nan"), 0.2)


def test_output_shape_link_order_and_zero_head_is_fixed_ewma_for_each_control():
    edges = _edges_72()
    history = torch.linspace(0.0, 1.0, steps=2 * 672 * 72).reshape(2, 672, 72)
    outputs = []
    parameter_counts = []
    for mode in ("topology", "identity", "shuffled"):
        model = _model(mode).eval()
        assert model.edge_order == edges
        with torch.no_grad():
            output = model(history)
            baseline = model._ewma(history).unsqueeze(1)
        assert output.shape == (2, 1, 72)
        assert torch.isfinite(output).all()
        assert torch.equal(output, baseline)
        outputs.append(output)
        parameter_counts.append(sum(parameter.numel() for parameter in model.parameters()))
    assert parameter_counts == [parameter_counts[0]] * 3
    assert all(torch.equal(outputs[0], value) for value in outputs[1:])


def test_control_modes_have_identical_seeded_learned_parameters():
    parameter_sets = []
    for mode in ("topology", "identity", "shuffled"):
        h96._seed_h96(42)
        model = _model(mode)
        parameter_sets.append({name: value.detach().clone() for name, value in model.named_parameters()})
    assert parameter_sets[0].keys() == parameter_sets[1].keys() == parameter_sets[2].keys()
    assert all(torch.equal(parameter_sets[0][key], parameter_sets[1][key]) for key in parameter_sets[0])
    assert all(torch.equal(parameter_sets[0][key], parameter_sets[2][key]) for key in parameter_sets[0])


def test_h96_seasonal_features_are_exactly_causal_history_sources():
    model = _model()
    history = torch.arange(672 * 72, dtype=torch.float32).reshape(1, 672, 72) / 1000
    baseline = model._ewma(history)
    features = model.seasonal_residual_features(history)
    assert features.shape == (1, 1, 72, 2)
    expected = torch.stack((history[:, 671, :] - baseline, history[:, 95, :] - baseline), dim=-1)
    assert torch.equal(features[:, 0, :, :], expected)
    changed = history.clone()
    changed[:, 670, :] += 1.0
    assert not torch.equal(model.seasonal_residual_features(changed), features)


def test_graph_message_passing_never_mixes_batch_samples():
    model = _model("topology").eval()
    with torch.no_grad():
        model.residual_head.weight.fill_(0.02)
        model.residual_head.bias.fill_(0.01)
        history = torch.rand((2, 672, 72), generator=torch.Generator().manual_seed(22))
        first = model(history)[0].clone()
        history[1] += 0.5
        second_run = model(history)[0]
    assert torch.equal(first, second_run)


def test_validation_pass_is_gradient_free_and_does_not_update_parameters():
    model = _model("identity")
    history = np.full((1, 672, 72), 0.25, dtype=np.float32)
    target = np.full((1, 72), 0.3, dtype=np.float32)
    scaler = h96.Standardizer(0.2, 0.5, "h=96 training histories and targets only")
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    loss, metrics = h96._validation_pass(
        model, history, target, scaler, 1, torch.device("cpu")
    )
    assert np.isfinite(loss)
    assert set(metrics) == {"h96_candidate", "ewma"}
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(torch.equal(value, before[key]) for key, value in model.state_dict().items())


def test_gradients_reach_temporal_and_graph_modules_and_loss_is_finite():
    model = _model("topology")
    torch.nn.init.constant_(model.residual_head.weight, 0.02)
    history = torch.rand((2, 672, 72), generator=torch.Generator().manual_seed(23))
    output = model(history)
    loss = h96.objective_h96(output[:, 0], torch.full((2, 72), 0.4), model._ewma(history))
    loss.backward()
    assert torch.isfinite(loss)
    for module in (model.temporal_encoder, model.message_projection, model.residual_head):
        grads = [parameter.grad for parameter in module.parameters()]
        assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in grads)
        assert sum(float(gradient.abs().sum()) for gradient in grads) > 0


def test_h96_scaler_accepts_training_only_arrays_and_ignores_validation_values():
    train_x = np.arange(2 * 672 * 72, dtype=np.float32).reshape(2, 672, 72) / 1000
    train_y = np.full((2, 72), 0.3, dtype=np.float32)
    val_a = np.full((2, 672, 72), -1000, dtype=np.float32)
    val_b = np.full((2, 672, 72), 1000, dtype=np.float32)
    del val_a, val_b  # neither is accepted by the scaler API
    scaler_a = h96.fit_h96_training_scaler(train_x, train_y)
    scaler_b = h96.fit_h96_training_scaler(train_x.copy(), train_y.copy())
    assert scaler_a == scaler_b
    assert scaler_a.fit_scope == "h=96 training histories and targets only"


def test_loader_maps_only_h96_train_and_validation_arrays(tmp_path, monkeypatch):
    repo, root, _ = _make_h96_dataset(tmp_path, monkeypatch)
    real_load = np.load
    calls = []

    def tracked_load(path, *args, **kwargs):
        calls.append(Path(path).name)
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(h96.np, "load", tracked_load)
    splits, edges, provenance = h96.load_h96_train_validation(root, repository_root=repo)
    assert calls == [
        "history_train_h96.npy", "target_train_h96.npy",
        "history_validation_h96.npy", "target_validation_h96.npy",
    ]
    assert set(splits) == {"train", "validation"}
    assert all(set(splits[split]) >= {"history", "target", "windows"} for split in splits)
    assert len(edges) == 72 and provenance["loaded_array_names"] == calls
    assert all(isinstance(splits[split]["history"], np.memmap) for split in splits)


def test_test_split_and_h4_array_names_are_rejected_before_numpy_load(tmp_path, monkeypatch):
    repo, root, _ = _make_h96_dataset(tmp_path, monkeypatch)
    called = False

    def forbidden(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("unexpected array read")

    monkeypatch.setattr(h96.np, "load", forbidden)
    for name in ("history_test_h96.npy", "history_train_h4.npy", "anything.npy"):
        with pytest.raises(ValueError):
            h96._load_h96_array(root, name, (1,))
    assert not called


def test_loader_rejects_invalid_provenance_before_loading_arrays(tmp_path, monkeypatch):
    repo, root, _ = _make_h96_dataset(tmp_path, monkeypatch)
    metadata_path = root / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["protocol_id"] = "tampered"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    called = False

    def forbidden(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("array loader called before provenance rejection")

    monkeypatch.setattr(h96.np, "load", forbidden)
    with pytest.raises(h96.H96IntegrityError, match="protocol/status"):
        h96.load_h96_train_validation(root, repository_root=repo)
    assert not called


def test_loader_rejects_unexpected_test_artifact_by_name_only(tmp_path, monkeypatch):
    repo, root, _ = _make_h96_dataset(tmp_path, monkeypatch)
    (root / "X_test.npy").write_bytes(b"synthetic name-only forbidden marker")
    called = False

    def forbidden(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("array loading must not occur after inventory failure")

    monkeypatch.setattr(h96.np, "load", forbidden)
    with pytest.raises(h96.H96IntegrityError, match="inventory"):
        h96.load_h96_train_validation(root, repository_root=repo)
    assert not called


def test_h96_window_target_cannot_cross_official_split_boundary(tmp_path, monkeypatch):
    repo, root, _ = _make_h96_dataset(tmp_path, monkeypatch)
    index_path = root / "window_index_validation.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["96"][0]["target"] = horizon_v2_data.OFFICIAL_SPLIT_BOUNDS["validation"][1]
    index_path.write_text(json.dumps(index), encoding="utf-8")
    with pytest.raises(h96.H96IntegrityError, match="crosses an invalid split/alignment boundary"):
        h96.load_h96_train_validation(root, repository_root=repo)


def test_output_destination_refusal_and_training_permission_precede_array_access(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / "phase9_geant").mkdir(parents=True)
    processed = repo / "data" / "processed"
    processed.mkdir(parents=True)
    output_root = processed / h96.OUTPUT_ROOT_NAME
    output_root.mkdir()
    destination = output_root / "h96_topology_seed_42"
    destination.mkdir()
    calls = []
    monkeypatch.setattr(h96, "load_h96_train_validation", lambda *_a, **_k: calls.append("load"))
    with pytest.raises(PermissionError, match="explicit_authorization"):
        h96.train_h96_from_root(
            "unused", repository_root=repo, seed=42, adjacency_mode="topology",
        )
    with pytest.raises(FileExistsError, match="Refusing existing"):
        h96.train_h96_from_root(
            "unused", repository_root=repo, seed=42, adjacency_mode="topology",
            run_training=True, explicit_authorization=True,
        )
    assert calls == []


def test_fresh_output_paths_are_unique_by_mode_and_seed(tmp_path):
    repo = tmp_path / "repo"
    (repo / "phase9_geant").mkdir(parents=True)
    (repo / "data" / "processed").mkdir(parents=True)
    root = repo / "data" / "processed"
    _, topology_42 = h96._planned_output(repo, 42, "topology")
    _, identity_42 = h96._planned_output(repo, 42, "identity")
    _, topology_123 = h96._planned_output(repo, 123, "topology")
    assert len({topology_42, identity_42, topology_123}) == 3
    assert all(path.parent == root / h96.OUTPUT_ROOT_NAME for path in (topology_42, identity_42, topology_123))


def test_checkpoint_hash_roundtrip_and_mismatch_fail_closed(tmp_path, monkeypatch):
    model = _model("identity")
    path = tmp_path / "synthetic.pt"
    expected_sha = h96._atomic_checkpoint(
        {"state_dict": model.state_dict(), "seed": 42, "protocol_id": h96.PROTOCOL_ID}, path
    )
    target = _model("identity")
    h96._load_checkpoint_verified(target, path, expected_sha, device=torch.device("cpu"), expected_seed=42)
    assert all(torch.equal(model.state_dict()[key], target.state_dict()[key]) for key in model.state_dict())
    called = False
    real_load = torch.load

    def tracked_load(*args, **kwargs):
        nonlocal called
        called = True
        return real_load(*args, **kwargs)

    monkeypatch.setattr(h96.torch, "load", tracked_load)
    with pytest.raises(h96.H96IntegrityError, match="SHA-256"):
        h96._load_checkpoint_verified(
            target, path, "0" * 64, device=torch.device("cpu"), expected_seed=42
        )
    assert not called


def test_seeded_initialization_is_reproducible():
    h96._seed_h96(123)
    first = _model("identity").state_dict()
    h96._seed_h96(123)
    second = _model("identity").state_dict()
    assert all(torch.equal(first[key], second[key]) for key in first)


def test_training_entry_requires_both_explicit_switches_without_path_access(tmp_path, monkeypatch):
    touched = False

    def forbidden(*_args, **_kwargs):
        nonlocal touched
        touched = True
        raise AssertionError("data path accessed without both authorization switches")

    monkeypatch.setattr(h96, "canonical_repository_root", forbidden)
    with pytest.raises(PermissionError):
        h96.train_h96_from_root(
            tmp_path / "not-a-data-root", repository_root=tmp_path, seed=42,
            adjacency_mode="topology", run_training=True, explicit_authorization=False,
        )
    assert not touched


def test_fixed_shuffle_seed_and_control_adjacency_hashes_are_reproducible():
    edges = _edges_72()
    first = h96._canonical_json_sha256(
        h96.build_link_adjacency(edges, mode="shuffled", shuffle_seed=20261019).tolist()
    )
    second = h96._canonical_json_sha256(
        h96.build_link_adjacency(edges, mode="shuffled", shuffle_seed=20261019).tolist()
    )
    assert first == second
    assert h96.H96_CONFIG["shuffle_seed"] == 20261019
