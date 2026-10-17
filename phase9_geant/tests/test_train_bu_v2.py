from __future__ import annotations

import json
import stat
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from phase9_geant import train_bu_v2 as bu
from phase9_geant import safe_paths
from phase9_geant import validity_v2
from src.gnn.model import EdgeQoSForecaster
from phase9_geant.window_provenance_v2 import (
    build_train_validation_provenance,
    canonical_json_sha256,
)


def _synthetic_arrays(*, train_n=3, validation_n=2, edges=72):
    rng = np.random.default_rng(17)
    return bu.WindowArrays(
        x_train=rng.normal(size=(train_n, 5, edges)).astype(np.float32),
        y_train=rng.normal(size=(train_n, edges)).astype(np.float32),
        x_validation=rng.normal(size=(validation_n, 5, edges)).astype(np.float32),
        y_validation=rng.normal(size=(validation_n, edges)).astype(np.float32),
    )


def _write_synthetic_arrays(root: Path, arrays: bu.WindowArrays):
    root.mkdir(parents=True)
    values = {
        "X_train.npy": arrays.x_train,
        "y_train.npy": arrays.y_train,
        "X_validation.npy": arrays.x_validation,
        "y_validation.npy": arrays.y_validation,
    }
    for filename, value in values.items():
        np.save(root / filename, value, allow_pickle=False)


def _synthetic_edges():
    nodes = tuple(f"n{i:02d}" for i in range(22))
    links = [(nodes[i], nodes[(i + 1) % 22]) for i in range(22)]
    links.extend((nodes[i], nodes[i + 2]) for i in range(14))
    return bu.directed_edges_v2(links)


def _synthetic_provenance(arrays):
    timestamps = [
        validity_v2.EXPECTED_START + index * validity_v2.STEP
        for index in range(validity_v2.SPLIT_BOUNDS["validation"][1])
    ]
    valid = [False] * len(timestamps)
    train_starts = [index * 7 for index in range(len(arrays.x_train))]
    validation_starts = [8020 + index * 7 for index in range(len(arrays.x_validation))]
    for start in (*train_starts, *validation_starts):
        for index in range(start, start + 6):
            valid[index] = True
    return build_train_validation_provenance(
        timestamps,
        valid,
        {"train": train_starts, "validation": validation_starts},
        {
            "train": (arrays.x_train, arrays.y_train),
            "validation": (arrays.x_validation, arrays.y_validation),
        },
    )


def _install_synthetic_run_identity(monkeypatch, arrays):
    provenance = _synthetic_provenance(arrays)
    provenance_hash = canonical_json_sha256(provenance)
    split_hashes = {
        name: canonical_json_sha256(provenance["splits"][name])
        for name in ("train", "validation")
    }
    monkeypatch.setattr(
        bu, "load_window_provenance",
        lambda _root: (provenance, provenance_hash, split_hashes),
    )
    monkeypatch.setattr(bu, "_topology_edge_order", _synthetic_edges)
    monkeypatch.setattr(
        bu, "_source_sha256", lambda _root: {"synthetic/source.py": "a" * 64}
    )
    return provenance


def _synthetic_identity(arrays, edges=None):
    edges = edges or _synthetic_edges()
    provenance = _synthetic_provenance(arrays)
    file_hash = canonical_json_sha256(provenance)
    split_hashes = {
        name: canonical_json_sha256(provenance["splits"][name])
        for name in ("train", "validation")
    }
    preflight = bu._preflight_run_identity(
        mode="full", seed=123, run_id=None,
        provenance_file_sha256=file_hash, provenance_hashes=split_hashes,
        edges=edges, source_sha256={"synthetic/source.py": "a" * 64},
    )
    identity = bu.build_array_provenance(
        arrays, provenance, provenance_file_sha256=file_hash,
        edges=edges, preflight_identity=preflight,
    )
    return identity, provenance, file_hash


def test_v2_arrays_load_only_the_four_fixed_train_validation_files(tmp_path):
    repo = tmp_path / "synthetic-repo"
    root = repo / "data" / "processed" / "geant_validity_v2"
    root.mkdir(parents=True)
    arrays = _synthetic_arrays()
    for filename, value in zip(
        bu.ARRAY_NAMES,
        (arrays.x_train, arrays.y_train, arrays.x_validation, arrays.y_validation),
    ):
        np.save(root / filename, value, allow_pickle=False)
    calls = []

    def tracking_loader(path, *, allow_pickle):
        calls.append(Path(path).name)
        return np.load(path, allow_pickle=allow_pickle)

    loaded = bu.load_v2_windows(root, array_loader=tracking_loader, repository_root=repo)
    assert loaded.x_train.shape == (3, 5, 72)
    assert loaded.y_train.shape == (3, 72)
    assert loaded.x_validation.shape == (2, 5, 72)
    assert loaded.y_validation.shape == (2, 72)
    assert tuple(calls) == bu.ARRAY_NAMES
    assert set(calls) == {
        "X_train.npy", "y_train.npy", "X_validation.npy", "y_validation.npy"
    }


def test_v2_window_dimensions_dtypes_counts_and_finiteness_fail_closed():
    arrays = _synthetic_arrays()
    bu.validate_window_arrays(arrays)

    with pytest.raises(ValueError, match="history"):
        bu.validate_window_arrays(
            bu.WindowArrays(
                arrays.x_train[:, :4], arrays.y_train,
                arrays.x_validation, arrays.y_validation,
            )
        )
    with pytest.raises(ValueError, match="targets must have shape"):
        bu.validate_window_arrays(
            bu.WindowArrays(
                arrays.x_train, arrays.y_train[:, :-1],
                arrays.x_validation, arrays.y_validation,
            )
        )
    with pytest.raises(ValueError, match="directed links"):
        bu.validate_window_arrays(arrays, expected_links=71)
    with pytest.raises(ValueError, match="float32"):
        bu.validate_window_arrays(
            bu.WindowArrays(
                arrays.x_train.astype(np.float64), arrays.y_train,
                arrays.x_validation, arrays.y_validation,
            )
        )
    for bad_value in (np.nan, np.inf, -np.inf):
        bad = arrays.x_validation.copy()
        bad[0, 0, 0] = bad_value
        with pytest.raises(ValueError, match="NaN or infinite"):
            bu.validate_window_arrays(
                bu.WindowArrays(arrays.x_train, arrays.y_train, bad, arrays.y_validation)
            )


def test_v2_interleaved_edge_order_and_model_edge_index_alignment():
    links = (("z", "a"), ("a", "b"))
    expected = (("z", "a"), ("a", "z"), ("a", "b"), ("b", "a"))
    edges_from_v2 = bu.directed_edges_v2(links)
    edge_index, node_order, model_edges = bu.build_edge_index(("b", "z", "a"), links)
    assert edges_from_v2 == expected
    assert model_edges == expected
    assert node_order == ("a", "b", "z")
    actual_index_edges = tuple(
        (node_order[int(source)], node_order[int(destination)])
        for source, destination in edge_index.T.tolist()
    )
    assert actual_index_edges == expected
    bu.assert_edge_order(edges_from_v2, actual_index_edges)

    with pytest.raises(ValueError, match="ordering"):
        bu.assert_edge_order(edges_from_v2, tuple(reversed(actual_index_edges)))


def test_real_bu_forward_preserves_disconnected_batch_graphs():
    links = (("a", "b"), ("b", "c"))
    edge_index, node_order, edges = bu.build_edge_index(("c", "b", "a"), links)
    assert node_order == ("a", "b", "c")
    assert edges == (("a", "b"), ("b", "a"), ("b", "c"), ("c", "b"))

    # Distinct constant histories make accidental cross-sample edge mixing visible.
    histories = np.stack(
        (
            np.full((5, len(edges)), -1.0, dtype=np.float32),
            np.full((5, len(edges)), 2.0, dtype=np.float32),
        ),
        axis=0,
    )
    normalization = bu.Normalization(mean=0.0, scale=1.0)
    sequence = bu.make_bu_sequence(histories, edge_index, len(node_order), normalization)

    assert len(sequence) == 5
    assert sequence[0].x.shape == (2 * len(node_order), 1)
    assert sequence[0].edge_index.shape == (2, 2 * len(edges))
    assert sequence[0].edge_attr.shape == (2 * len(edges), 1)
    assert torch.equal(sequence[0].edge_index[:, : len(edges)], edge_index)
    assert torch.equal(
        sequence[0].edge_index[:, len(edges) :], edge_index + len(node_order)
    )
    assert torch.equal(sequence[0].edge_attr[: len(edges)], torch.full((len(edges), 1), -1.0))
    assert torch.equal(sequence[0].edge_attr[len(edges) :], torch.full((len(edges), 1), 2.0))

    model = EdgeQoSForecaster(edge_feature_count=1, target_count=1)
    output = model(sequence)
    assert output.shape == (2 * len(edges), 1)
    assert torch.isfinite(output).all()
    batched_output = output.reshape(2, len(edges), 1)
    assert batched_output.shape == (2, len(edges), 1)
    assert torch.isfinite(batched_output).all()


def test_training_normalization_is_independent_of_validation_values():
    arrays = _synthetic_arrays()
    before = bu.fit_training_normalization(arrays.x_train, arrays.y_train)
    arrays.x_validation.fill(1e20)
    arrays.y_validation.fill(-1e20)
    after = bu.fit_training_normalization(arrays.x_train, arrays.y_train)
    assert after == before
    assert after.fit_scope == "all scalar values in X_train.npy and y_train.npy only"


def test_validation_loss_disables_gradients_and_does_not_update_parameters():
    class ProbeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(0.75))

        def forward(self, sequence):
            assert not torch.is_grad_enabled()
            return sequence[-1].edge_attr * self.weight

    arrays = _synthetic_arrays(validation_n=2)
    normalization = bu.fit_training_normalization(arrays.x_train, arrays.y_train)
    edge_index = torch.zeros((2, 72), dtype=torch.long)
    model = ProbeModel()
    before = model.weight.detach().clone()
    loss = bu.validation_loss(
        model, arrays.x_validation, arrays.y_validation,
        edge_index, 1, normalization, batch_size=1,
    )
    assert np.isfinite(loss)
    assert torch.equal(model.weight.detach(), before)
    assert not model.training


def test_checkpoint_choice_depends_only_on_validation_loss():
    assert bu.validation_loss_improved(0.4, float("inf"))
    assert bu.validation_loss_improved(0.3, 0.4)
    assert not bu.validation_loss_improved(0.4, 0.4)
    assert not bu.validation_loss_improved(0.5, 0.4)
    with pytest.raises(ValueError, match="finite"):
        bu.validation_loss_improved(float("nan"), 0.4)


def test_output_directory_is_confined_to_dedicated_v2_child(tmp_path):
    v2_root = tmp_path / "data" / "processed" / "geant_validity_v2"
    accepted = v2_root / "neural_bu_v2"
    assert bu.validate_output_dir(accepted, v2_root) == accepted.resolve()
    with pytest.raises(ValueError, match="direct child"):
        bu.validate_output_dir(v2_root.parent / "geant" / "neural_bu_v2", v2_root)
    with pytest.raises(ValueError, match="Parent traversal"):
        bu.validate_output_dir(v2_root / ".." / "geant" / "outputs", v2_root)
    with pytest.raises(ValueError, match="dedicated"):
        bu.validate_output_dir(v2_root, v2_root)


def test_checked_absolute_returns_nonexistent_trailing_output_path(tmp_path):
    requested = tmp_path / "new-root" / "new-run"
    assert safe_paths.checked_absolute(requested) == requested.absolute()
    assert not requested.exists()


def test_bu_output_path_rejects_reparse_ancestor_before_resolution(tmp_path, monkeypatch):
    repository = tmp_path / "repo"
    root = repository / "data" / "processed" / "geant_validity_v2"
    root.mkdir(parents=True)
    redirected_component = repository / "data"
    real_lstat = Path.lstat

    def synthetic_reparse_lstat(path):
        info = real_lstat(path)
        if path == redirected_component:
            return SimpleNamespace(
                st_mode=stat.S_IFDIR | 0o755,
                st_file_attributes=safe_paths._REPARSE_POINT_ATTRIBUTE,
            )
        return info

    monkeypatch.setattr(Path, "lstat", synthetic_reparse_lstat)
    with pytest.raises(ValueError, match="Reparse-point"):
        bu.output_dir_for_run("full", 42, root)


def test_bu_post_creation_containment_failure_leaves_directory_and_skips_training(
    tmp_path, monkeypatch
):
    root = tmp_path / "data" / "processed" / "geant_validity_v2"
    root.mkdir(parents=True)
    arrays = _synthetic_arrays(train_n=1, validation_n=1)
    _install_synthetic_run_identity(monkeypatch, arrays)
    trained = []

    def fail_containment(_output, _root):
        raise ValueError("synthetic post-create containment failure")

    monkeypatch.setattr(bu, "verify_new_directory_containment", fail_containment)
    with pytest.raises(ValueError, match="post-create containment failure"):
        bu._run_validation_training(
            mode="full", seed=42, data_root=root,
            array_loader=lambda _root: arrays,
            training_fn=lambda *_args, **_kwargs: trained.append(True),
            repository_root=tmp_path,
        )
    output = root / "neural_bu_v2_full_seed_42"
    assert output.is_dir()
    assert list(output.iterdir()) == []
    assert trained == []


def test_bu_post_creation_replacement_is_never_removed(tmp_path, monkeypatch):
    root = tmp_path / "data" / "processed" / "geant_validity_v2"
    root.mkdir(parents=True)
    arrays = _synthetic_arrays(train_n=1, validation_n=1)
    _install_synthetic_run_identity(monkeypatch, arrays)
    output = root / "neural_bu_v2_full_seed_42"
    trained = []

    def replace_then_fail(path, _root):
        path.rmdir()
        path.mkdir()
        (path / "replacement.marker").write_text("preserve", encoding="utf-8")
        raise ValueError("synthetic directory replacement detected")

    monkeypatch.setattr(bu, "verify_new_directory_containment", replace_then_fail)
    with pytest.raises(ValueError, match="replacement detected"):
        bu._run_validation_training(
            mode="full", seed=42, data_root=root,
            array_loader=lambda _root: arrays,
            training_fn=lambda *_args, **_kwargs: trained.append(True),
            repository_root=tmp_path,
        )
    assert (output / "replacement.marker").read_text(encoding="utf-8") == "preserve"
    assert trained == []


def test_training_path_is_explicitly_opt_in_without_loading_data_by_default():
    with pytest.raises(SystemExit):
        bu._parse_args([])


def test_cli_requires_explicit_mode_and_full_seed():
    smoke = bu._parse_args(["--run-smoke"])
    assert smoke.run_smoke and not smoke.run_full and smoke.seed is None

    full = bu._parse_args(["--run-full", "--seed", "123"])
    assert full.run_full and not full.run_smoke and full.seed == 123

    isolated = bu._parse_args(
        ["--run-full", "--seed", "42", "--run-id", "fresh_20261016"]
    )
    assert isolated.run_full and isolated.seed == 42
    assert isolated.run_id == "fresh_20261016"

    for invalid_args in (
        [],
        ["--run-full"],
        ["--run-smoke", "--seed", "42"],
        ["--run-smoke", "--run-id", "fresh"],
        ["--run-full", "--seed", "42", "--run-id", "../escape"],
    ):
        with pytest.raises(SystemExit):
            bu._parse_args(invalid_args)


def test_run_id_creates_distinct_full_paths_without_changing_defaults(tmp_path):
    root = tmp_path / "data" / "processed" / "geant_validity_v2"
    assert bu.output_dir_for_run("smoke", 42, root) == (root / "neural_bu_v2").resolve()
    assert bu.output_dir_for_run("full", 42, root) == (
        root / "neural_bu_v2_full_seed_42"
    ).resolve()

    paths = [
        bu.output_dir_for_run("full", seed, root, run_id="fresh_20261016")
        for seed in (42, 123, 2026)
    ]
    assert len(set(paths)) == 3
    assert paths[0] == (root / "neural_bu_v2_full_fresh_20261016_seed_42").resolve()
    with pytest.raises(ValueError, match="only for full runs"):
        bu.output_dir_for_run("smoke", 42, root, run_id="fresh")


def test_run_id_lifecycle_records_identity_and_refuses_existing_destination(tmp_path, monkeypatch):
    root = tmp_path / "data" / "processed" / "geant_validity_v2"
    run_id = "fresh_20261016"
    output = bu.output_dir_for_run("full", 123, root, run_id=run_id)
    output.mkdir(parents=True)
    loader_calls = []

    def loader(_data_root):
        loader_calls.append(True)
        raise AssertionError("existing custom run destination must fail before loading")

    with pytest.raises(FileExistsError, match="Refusing to reuse existing"):
        bu._run_validation_training(
            mode="full", seed=123, data_root=root,
            array_loader=loader, run_id=run_id, repository_root=tmp_path,
        )
    assert loader_calls == []

    # A fresh seed path uses the same lifecycle and records the label in both
    # the durable initial manifest and the completed manifest.
    arrays = _synthetic_arrays(train_n=1, validation_n=1)
    _install_synthetic_run_identity(monkeypatch, arrays)
    other_output = bu.output_dir_for_run("full", 2026, root, run_id=run_id)

    def synthetic_trainer(_arrays, *, mode, seed, output, data_root):
        checkpoint = output / "best_validation.pt"
        checkpoint.write_bytes(b"synthetic checkpoint")
        return {"checkpoint": str(checkpoint), "mode": mode, "seed": seed}

    def synthetic_loader(_data_root):
        assert not other_output.exists()
        return arrays

    completed = bu._run_validation_training(
        mode="full", seed=2026, data_root=root, array_loader=synthetic_loader,
        training_fn=synthetic_trainer, run_id=run_id, repository_root=tmp_path,
    )
    manifest = _read_run_manifest(completed)
    assert manifest["status"] == "complete"
    assert manifest["run_id"] == run_id


def test_run_output_paths_are_distinct_and_preserve_smoke_location(tmp_path):
    root = tmp_path / "data" / "processed" / "geant_validity_v2"
    smoke = bu.output_dir_for_run("smoke", 42, root)
    full_42 = bu.output_dir_for_run("full", 42, root)
    full_123 = bu.output_dir_for_run("full", 123, root)
    full_2026 = bu.output_dir_for_run("full", 2026, root)

    assert smoke == (root / "neural_bu_v2").resolve()
    assert len({smoke, full_42, full_123, full_2026}) == 4
    assert full_123 == (root / "neural_bu_v2_full_seed_123").resolve()
    with pytest.raises(ValueError, match="retains its original seed"):
        bu.output_dir_for_run("smoke", 123, root)

    assert bu.SEED == 42
    assert bu.TRAINING == {
        "epochs": 5,
        "early_stopping_patience": 2,
        "learning_rate": 0.005,
        "batch_size": 32,
        "optimizer": "Adam",
        "loss": "SmoothL1Loss",
        "hidden_channels": 16,
        "gru_hidden_size": 16,
        "initialization": "from_scratch",
    }


def test_existing_run_output_is_never_overwritten(tmp_path):
    output = tmp_path / "already-created-run"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing to reuse existing"):
        bu.assert_output_available(output)
    assert marker.read_text(encoding="utf-8") == "preserve"


def test_manifest_records_mode_seed_protocol_and_validation_selection(tmp_path):
    arrays = _synthetic_arrays()
    normalization = bu.fit_training_normalization(arrays.x_train, arrays.y_train)
    edges = _synthetic_edges()
    provenance = _synthetic_provenance(arrays)
    file_hash = canonical_json_sha256(provenance)
    split_hashes = {
        name: canonical_json_sha256(provenance["splits"][name])
        for name in ("train", "validation")
    }
    preflight = bu._preflight_run_identity(
        mode="full", seed=123, run_id=None,
        provenance_file_sha256=file_hash, provenance_hashes=split_hashes,
        edges=edges, source_sha256={"synthetic/source.py": "a" * 64},
    )
    run_identity = bu.build_array_provenance(
        arrays, provenance, provenance_file_sha256=file_hash,
        edges=edges, preflight_identity=preflight,
    )
    checkpoint = tmp_path / "best_validation.pt"
    manifest = tmp_path / "run_metadata.json"
    history = [
        {"epoch": 1, "train_loss": 0.5, "validation_loss": 0.4},
        {"epoch": 2, "train_loss": 0.3, "validation_loss": 0.2},
    ]
    result = bu.build_run_manifest(
        mode="full",
        seed=123,
        arrays=arrays,
        normalization=normalization,
        edges=edges,
        best_epoch=2,
        best_validation_loss=0.2,
        epochs_completed=2,
        history=history,
        early_stopping_triggered=True,
        training_seconds=12.5,
        checkpoint_path=checkpoint,
        manifest_path=manifest,
        run_identity=run_identity,
    )
    assert result["protocol_id"] == bu.PROTOCOL_ID
    assert result["mode"] == "full"
    assert result["seed"] == 123
    assert result["model_settings"] == bu.TRAINING
    assert result["best_validation_epoch"] == 2
    assert result["best_validation_loss"] == 0.2
    assert result["history"] == history
    assert result["early_stopping"]["triggered"] is True
    assert result["data_split"]["test"] == "not loaded or evaluated"
    assert result["normalization"]["fit_scope"] == normalization.fit_scope
    assert result["checkpoint"] == str(checkpoint.resolve())
    assert result["manifest_path"] == str(manifest.resolve())


def test_bu_metadata_requires_provenance_fingerprints_edges_and_checkpoint_hash():
    with pytest.raises(ValueError, match="lacks required provenance"):
        bu.validate_required_run_metadata({})

    arrays = _synthetic_arrays()
    identity, provenance, file_hash = _synthetic_identity(arrays)
    metadata = {**identity, "checkpoint_sha256": "b" * 64}
    bu.validate_required_run_metadata(metadata, expected_edges=_synthetic_edges())

    changed = _synthetic_arrays()
    changed.x_validation[0, 0, 0] += 1.0
    preflight = {key: value for key, value in identity.items() if key != "array_fingerprints"}
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        bu.build_array_provenance(
            changed, provenance, provenance_file_sha256=file_hash,
            edges=_synthetic_edges(), preflight_identity=preflight,
        )

    altered_order = dict(metadata)
    reordered_pairs = [
        [item["source"], item["destination"]]
        for item in reversed(metadata["directed_edge_order"])
    ]
    altered_order["directed_edge_order"] = [
        {"index": index, "source": pair[0], "destination": pair[1]}
        for index, pair in enumerate(reordered_pairs)
    ]
    altered_order["edge_order_sha256"] = canonical_json_sha256(reordered_pairs)
    with pytest.raises(ValueError, match="edge order"):
        bu.validate_required_run_metadata(
            altered_order, expected_edges=_synthetic_edges()
        )


def test_checkpoint_digest_mismatch_fails_closed(tmp_path):
    checkpoint = tmp_path / "synthetic-checkpoint.bin"
    checkpoint.write_bytes(b"synthetic weights")
    expected = bu._sha256_file(checkpoint)
    assert bu.verify_checkpoint_sha256(checkpoint, expected) == expected
    checkpoint.write_bytes(b"changed synthetic weights")
    with pytest.raises(ValueError, match="Checkpoint SHA-256 mismatch"):
        bu.verify_checkpoint_sha256(checkpoint, expected)


def test_missing_provenance_fails_before_output_creation(tmp_path, monkeypatch):
    root = tmp_path / "data" / "processed" / "geant_validity_v2"
    monkeypatch.setattr(
        bu, "load_window_provenance",
        lambda _root: (_ for _ in ()).throw(FileNotFoundError("missing provenance")),
    )
    with pytest.raises(FileNotFoundError, match="missing provenance"):
        bu._run_validation_training(mode="full", seed=42, data_root=root, repository_root=tmp_path)
    assert not (root / "neural_bu_v2_full_seed_42").exists()


def test_array_fingerprint_mismatch_fails_before_output_or_training(tmp_path, monkeypatch):
    root = tmp_path / "data" / "processed" / "geant_validity_v2"
    provenance_arrays = _synthetic_arrays()
    _install_synthetic_run_identity(monkeypatch, provenance_arrays)
    loaded_arrays = _synthetic_arrays()
    loaded_arrays.y_train[0, 0] += 0.5
    training_calls = []

    def trainer(*_args, **_kwargs):
        training_calls.append(True)
        raise AssertionError("training must not run on mismatched provenance")

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        bu._run_validation_training(
            mode="full", seed=42, data_root=root,
            array_loader=lambda _root: loaded_arrays, training_fn=trainer,
            repository_root=tmp_path,
        )
    assert training_calls == []
    assert not (root / "neural_bu_v2_full_seed_42").exists()


def test_checkpoint_hash_mismatch_prevents_completion(tmp_path, monkeypatch):
    root = tmp_path / "data" / "processed" / "geant_validity_v2"
    root.mkdir(parents=True)
    arrays = _synthetic_arrays()
    _install_synthetic_run_identity(monkeypatch, arrays)

    def trainer(_arrays, *, output, **_kwargs):
        checkpoint = output / "best_validation.pt"
        checkpoint.write_bytes(b"synthetic checkpoint")
        return {"checkpoint": str(checkpoint), "checkpoint_sha256": "0" * 64}

    with pytest.raises(ValueError, match="Checkpoint SHA-256 mismatch"):
        bu._run_validation_training(
            mode="full", seed=123, data_root=root,
            array_loader=lambda _root: arrays, training_fn=trainer,
            repository_root=tmp_path,
        )
    assert _read_run_manifest(root / "neural_bu_v2_full_seed_123")["status"] == "failed"


def _read_run_manifest(output: Path):
    return json.loads((output / "run_metadata.json").read_text(encoding="utf-8"))


def test_provenance_precedes_output_and_incomplete_precedes_training(tmp_path, monkeypatch):
    root = tmp_path / "data" / "processed" / "geant_validity_v2"
    root.mkdir(parents=True)
    arrays = _synthetic_arrays()
    _install_synthetic_run_identity(monkeypatch, arrays)
    observed = {}
    real_atomic_write = bu.atomic_write_manifest

    def track_atomic_write(path, manifest):
        if manifest.get("status") == "complete":
            checkpoint = Path(manifest["checkpoint"])
            observed["checkpoint_existed_before_complete"] = checkpoint.is_file() and checkpoint.stat().st_size > 0
        return real_atomic_write(path, manifest)

    monkeypatch.setattr(bu, "atomic_write_manifest", track_atomic_write)

    def loader(data_root):
        output = root / "neural_bu_v2_full_seed_123"
        observed["output_absent_before_load"] = not output.exists()
        return arrays

    def trainer(loaded, *, mode, seed, output, data_root):
        assert loaded is arrays
        assert _read_run_manifest(output)["status"] == "incomplete"
        checkpoint = output / "best_validation.pt"
        checkpoint.write_bytes(b"synthetic checkpoint")
        return {"checkpoint": str(checkpoint)}

    output = bu._run_validation_training(
        mode="full", seed=123, data_root=root,
        array_loader=loader, training_fn=trainer, repository_root=tmp_path,
    )
    manifest = _read_run_manifest(output)
    assert observed["output_absent_before_load"] is True
    assert observed["checkpoint_existed_before_complete"] is True
    assert manifest["status"] == "complete"
    assert Path(manifest["checkpoint"]).is_file()


def test_array_loading_exception_does_not_create_run_output(tmp_path, monkeypatch):
    root = tmp_path / "data" / "processed" / "geant_validity_v2"
    _install_synthetic_run_identity(monkeypatch, _synthetic_arrays())

    def loader(_data_root):
        raise ValueError("synthetic loader failure")

    with pytest.raises(ValueError, match="synthetic loader failure"):
        bu._run_validation_training(
            mode="full", seed=123, data_root=root, array_loader=loader,
            repository_root=tmp_path,
        )
    assert not (root / "neural_bu_v2_full_seed_123").exists()


@pytest.mark.parametrize("failure_kind", ["training", "checkpoint_save"])
def test_training_or_checkpoint_failure_never_marks_run_complete(tmp_path, monkeypatch, failure_kind):
    root = tmp_path / "data" / "processed" / "geant_validity_v2"
    root.mkdir(parents=True)
    arrays = _synthetic_arrays()
    _install_synthetic_run_identity(monkeypatch, arrays)

    def trainer(_arrays, *, mode, seed, output, data_root):
        if failure_kind == "checkpoint_save":
            raise OSError("synthetic checkpoint write failure")
        (output / "best_validation.pt").write_bytes(b"partial checkpoint")
        raise RuntimeError("synthetic training failure")

    with pytest.raises((RuntimeError, OSError)):
        bu._run_validation_training(
            mode="full", seed=2026, data_root=root,
            array_loader=lambda _root: arrays, training_fn=trainer,
            repository_root=tmp_path,
        )
    manifest = _read_run_manifest(root / "neural_bu_v2_full_seed_2026")
    assert manifest["status"] == "failed"
    assert manifest["status"] != "complete"


def test_failure_manifest_write_keeps_original_exception_and_incomplete_state(
    tmp_path, monkeypatch
):
    root = tmp_path / "data" / "processed" / "geant_validity_v2"
    root.mkdir(parents=True)
    arrays = _synthetic_arrays()
    _install_synthetic_run_identity(monkeypatch, arrays)
    original_error = RuntimeError("original training error")
    real_atomic_write = bu.atomic_write_manifest

    def fail_failed_status(path, manifest):
        if manifest.get("status") == "failed":
            raise OSError("cannot record failure")
        return real_atomic_write(path, manifest)

    monkeypatch.setattr(bu, "atomic_write_manifest", fail_failed_status)

    def trainer(_arrays, **_kwargs):
        raise original_error

    with pytest.raises(RuntimeError) as caught:
        bu._run_validation_training(
            mode="full", seed=123, data_root=root,
            array_loader=lambda _root: arrays, training_fn=trainer,
            repository_root=tmp_path,
        )
    assert caught.value is original_error
    manifest = _read_run_manifest(root / "neural_bu_v2_full_seed_123")
    assert manifest["status"] == "incomplete"


def test_atomic_manifest_replace_failure_preserves_previous_manifest(tmp_path, monkeypatch):
    manifest_path = tmp_path / "run_metadata.json"
    original = {"status": "incomplete", "seed": 123}
    manifest_path.write_text(json.dumps(original), encoding="utf-8")

    def replace_failure(_source, _destination):
        raise OSError("synthetic replace failure")

    monkeypatch.setattr(bu.os, "replace", replace_failure)
    with pytest.raises(OSError, match="synthetic replace failure"):
        bu.atomic_write_manifest(manifest_path, {"status": "complete", "seed": 123})
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == original
    assert not [path for path in tmp_path.iterdir() if path.suffix == ".tmp"]


@pytest.mark.parametrize(
    ("mode", "seed", "existing_status"),
    [
        ("smoke", 42, None),
        ("full", 42, "complete"),
        ("full", 123, "incomplete"),
        ("full", 2026, "failed"),
    ],
)
def test_existing_smoke_full_incomplete_or_failed_outputs_reject_before_loading(
    tmp_path, mode, seed, existing_status
):
    root = tmp_path / "data" / "processed" / "geant_validity_v2"
    output = bu.output_dir_for_run(mode, seed, root)
    output.mkdir(parents=True)
    if existing_status is not None:
        (output / "run_metadata.json").write_text(
            json.dumps({"status": existing_status}), encoding="utf-8"
        )
    else:
        # Represents the pre-lifecycle seed-42 smoke manifest.
        (output / "run_metadata.json").write_text(
            json.dumps({"seed": 42}), encoding="utf-8"
        )
    marker = output / "preserve.txt"
    marker.write_text("keep", encoding="utf-8")
    loader_calls = []

    def loader(_data_root):
        loader_calls.append(True)
        raise AssertionError("arrays must not be loaded for an existing output")

    with pytest.raises(FileExistsError, match="Refusing to reuse existing"):
        bu._run_validation_training(
            mode=mode, seed=seed, data_root=root, array_loader=loader,
            repository_root=tmp_path,
        )
    assert loader_calls == []
    assert marker.read_text(encoding="utf-8") == "keep"


def test_real_training_function_reuses_wrapper_created_output_directory(
    tmp_path, monkeypatch
):
    root = tmp_path / "data" / "processed" / "geant_validity_v2"
    root.mkdir(parents=True)
    arrays = _synthetic_arrays(train_n=1, validation_n=1, edges=72)
    _install_synthetic_run_identity(monkeypatch, arrays)
    nodes = {f"n{index:02d}": (0.0, 0.0) for index in range(22)}
    links = [(f"n{index:02d}", f"n{(index + 1) % 22:02d}") for index in range(22)]
    links.extend((f"n{index:02d}", f"n{index + 2:02d}") for index in range(14))
    monkeypatch.setattr(bu, "parse_topology", lambda _path: (nodes, links))

    class NoOpOptimizer:
        def __init__(self, *_args, **_kwargs):
            pass

        def zero_grad(self, **_kwargs):
            pass

        def step(self):
            pass

    class SyntheticForecaster(nn.Module):
        def __init__(self, **_kwargs):
            super().__init__()
            self.bias = nn.Parameter(torch.zeros(1))

        def forward(self, sequence):
            return self.bias.expand(sequence[-1].edge_attr.shape[0], 1)

    monkeypatch.setattr(bu.torch.optim, "Adam", NoOpOptimizer)
    monkeypatch.setattr(bu, "EdgeQoSForecaster", SyntheticForecaster)
    monkeypatch.setattr(bu, "validation_loss", lambda *_args, **_kwargs: 0.25)
    monkeypatch.setattr(
        bu.torch,
        "save",
        lambda _state, path: Path(path).write_bytes(b"synthetic checkpoint"),
    )

    observed = {}

    def loader(data_root):
        output = root / "neural_bu_v2_full_seed_123"
        observed["output_absent_before_load"] = not output.exists()
        return arrays

    output = bu._run_validation_training(
        mode="full", seed=123, data_root=root, array_loader=loader,
        repository_root=tmp_path,
    )
    manifest = _read_run_manifest(output)
    assert observed["output_absent_before_load"] is True
    assert manifest["status"] == "complete"
    assert manifest["seed"] == 123
    assert manifest["epochs_completed"] == 3
    assert manifest["early_stopping"]["triggered"] is True
    assert Path(manifest["checkpoint"]).read_bytes() == b"synthetic checkpoint"

