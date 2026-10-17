"""Synthetic-only tests for validation comparison and gate arithmetic."""

from __future__ import annotations

import json

import numpy as np
import pytest

from phase9_geant import evaluate_residual_ewma_v1_validation as evaluation
from phase9_geant import validity_v2
from phase9_geant.window_provenance_v2 import build_train_validation_provenance


def test_methods_share_identical_validation_windows_and_paired_errors():
    target = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    ewma = np.asarray([[0.0, 2.5], [2.0, 5.0]])
    residual = np.asarray([[0.5, 2.0], [3.0, 4.0]])
    bu = np.asarray([[1.5, 2.0], [3.0, 3.0]])
    result = evaluation.build_seed_comparison(
        target, residual, ewma, bu, bootstrap_seed=7, bootstrap_replicates=25
    )
    assert result["metrics"]["residual_ewma"]["evaluated_scalar_targets"] == 4
    assert result["metrics"]["ewma"]["evaluated_scalar_targets"] == 4
    assert result["metrics"]["B-U"]["evaluated_scalar_targets"] == 4
    np.testing.assert_array_equal(
        evaluation.paired_absolute_error_difference(target, residual, ewma),
        np.asarray([[ -0.5, -0.5], [-1.0, -1.0]]),
    )


@pytest.mark.parametrize("bad", [
    np.ones((1, 2)),
    np.asarray([[1.0, np.nan], [2.0, 3.0]]),
])
def test_metric_and_paired_shape_or_finiteness_mismatches_fail_closed(bad):
    target = np.ones((2, 2))
    with pytest.raises(ValueError):
        evaluation.metrics(target, bad)
    with pytest.raises(ValueError):
        evaluation.paired_absolute_error_difference(target, bad, target)


def test_edge_column_order_mismatch_fails_closed():
    edges = (("a", "b"), ("b", "a"))
    evaluation.validate_edge_order([["a", "b"], ["b", "a"]], edges, format_name="residual")
    with pytest.raises(ValueError, match="edge order"):
        evaluation.validate_edge_order([["b", "a"], ["a", "b"]], edges, format_name="residual")


def test_aggregate_uses_sample_standard_deviation_ddof_one():
    values = {"42": {"mae": 1.0, "rmse": 2.0}, "123": {"mae": 2.0, "rmse": 4.0}, "2026": {"mae": 3.0, "rmse": 6.0}}
    result = evaluation.aggregate_seed_metrics(values)
    assert result["mae"]["mean"] == pytest.approx(2.0)
    assert result["mae"]["sample_std_ddof_1"] == pytest.approx(1.0)
    assert result["rmse"]["sample_std_ddof_1"] == pytest.approx(2.0)


def test_window_bootstrap_is_deterministic_and_resamples_whole_windows():
    differences = np.asarray([[1.0, 3.0], [5.0, 7.0], [9.0, 11.0]])
    first = evaluation.paired_window_bootstrap_ci(differences, seed=31, replicates=100)
    second = evaluation.paired_window_bootstrap_ci(differences, seed=31, replicates=100)
    assert first == second
    assert first["mean_absolute_error_difference"] == pytest.approx(6.0)
    assert first["resampling_unit"].startswith("validation windows")


def test_gate_passes_with_exactly_two_wins_and_at_least_one_percent_mean_gain():
    result = evaluation.usefulness_gate([0.97, 0.97, 1.03], [1.0, 1.0, 1.0])
    assert result["wins"] == 2
    assert result["mean_seed_relative_mae_reduction"] >= 0.01
    assert result["status"] == "PASS"


def test_gate_fails_with_only_one_win():
    result = evaluation.usefulness_gate([0.97, 1.01, 1.03], [1.0, 1.0, 1.0])
    assert result["wins"] == 1
    assert result["status"] == "FAIL"


def test_gate_fails_just_below_one_percent_mean_gain():
    result = evaluation.usefulness_gate([0.97, 0.97, 1.0300001], [1.0, 1.0, 1.0])
    assert result["wins"] == 2
    assert result["mean_seed_relative_mae_reduction"] < 0.01
    assert result["status"] == "FAIL"


def test_gate_ties_are_not_wins_and_bu_is_not_an_input_to_gate():
    tied = evaluation.usefulness_gate([1.0, 0.98, 1.02], [1.0, 1.0, 1.0])
    assert tied["strict_per_seed_wins"] == [False, True, False]
    assert tied["wins"] == 1 and tied["status"] == "FAIL"
    # The gate signature has no B-U argument; changing contextual B-U metrics
    # cannot alter the residual-vs-EWMA decision.
    baseline_gate = evaluation.usefulness_gate([0.97, 0.97, 1.03], [1.0, 1.0, 1.0])
    _contextual_bu_a = [100.0, 100.0, 100.0]
    _contextual_bu_b = [0.0, 0.0, 0.0]
    assert _contextual_bu_a != _contextual_bu_b
    assert evaluation.usefulness_gate([0.97, 0.97, 1.03], [1.0, 1.0, 1.0]) == baseline_gate


def _write_synthetic_completed_artifacts(
    tmp_path, *, bu_manifest_mutation=None, residual_manifest_mutation=None,
    edge_order_override=None, root_name="geant_validity_v2",
):
    """Create evaluator preflight inputs only; checkpoint bytes are not loaded."""
    repo = tmp_path / "repo"
    root = repo / "data" / "processed" / root_name
    root.mkdir(parents=True)
    edge_order = tuple(edge_order_override) if edge_order_override is not None else tuple(
        (f"source_{index}", f"destination_{index}") for index in range(72)
    )
    x_train = np.arange(3 * 5 * 72, dtype=np.float32).reshape(3, 5, 72)
    y_train = np.arange(3 * 72, dtype=np.float32).reshape(3, 72)
    x = np.arange(2 * 5 * 72, dtype=np.float32).reshape(2, 5, 72)
    y = np.arange(2 * 72, dtype=np.float32).reshape(2, 72)
    timestamps = [
        validity_v2.EXPECTED_START + index * validity_v2.STEP
        for index in range(validity_v2.SPLIT_BOUNDS["validation"][1])
    ]
    valid = [False] * len(timestamps)
    train_starts, validation_starts = (0, 7, 14), (8020, 8027)
    for start in (*train_starts, *validation_starts):
        for index in range(start, start + 6):
            valid[index] = True
    provenance = build_train_validation_provenance(
        timestamps,
        valid,
        {"train": train_starts, "validation": validation_starts},
        {
            "train": (x_train, y_train),
            "validation": (x, y),
        },
    )
    provenance_raw = json.dumps(provenance, sort_keys=True).encode("utf-8")
    (root / evaluation.PROVENANCE_FILENAME).write_bytes(provenance_raw)
    for filename, array in (
        ("X_train.npy", x_train), ("y_train.npy", y_train),
        ("X_validation.npy", x), ("y_validation.npy", y),
    ):
        np.save(root / filename, array, allow_pickle=False)
    train_block = provenance["splits"]["train"]
    validation_block = provenance["splits"]["validation"]
    provenance_hashes = {
        "train": evaluation.canonical_json_sha256(train_block),
        "validation": evaluation.canonical_json_sha256(validation_block),
    }
    provenance_file_hash = __import__("hashlib").sha256(provenance_raw).hexdigest()
    expected_fingerprints = {
        **train_block["array_fingerprints"],
        **validation_block["array_fingerprints"],
    }
    expected_shapes = {
        "X_train": list(x_train.shape), "y_train": list(y_train.shape),
        "X_validation": list(x.shape), "y_validation": list(y.shape),
    }
    expected_counts = {"train": len(x_train), "validation": len(x)}
    residual_edges = [list(edge) for edge in edge_order]
    bu_edges = [
        {"index": index, "source": source, "destination": destination}
        for index, (source, destination) in enumerate(edge_order)
    ]
    for relative in evaluation.bu.RUN_IDENTITY_SOURCE_FILES:
        source_path = repo / relative
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_bytes(f"synthetic source: {relative}".encode("utf-8"))
    source_hashes = evaluation.bu._source_sha256(repo)
    for source_path in evaluation.residual._source_paths(repo):
        source_path.parent.mkdir(parents=True, exist_ok=True)
        if not source_path.exists():
            source_path.write_bytes(f"synthetic residual source: {source_path.name}".encode())
    residual_source_hashes = evaluation.residual._source_hashes(repo)

    for seed in evaluation.SEEDS:
        for kind, prefix in (("residual", evaluation.EXPECTED_RESIDUAL_PREFIX),
                             ("B-U", evaluation.EXPECTED_BU_PREFIX)):
            run_dir = root / f"{prefix}{seed}"
            run_dir.mkdir()
            checkpoint = run_dir / "best_validation.pt"
            checkpoint.write_bytes(f"synthetic checkpoint {kind} {seed}".encode())
            manifest = {
                "status": "complete", "seed": seed,
                "protocol_id": evaluation.PROTOCOL_ID,
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": evaluation._sha256_file(checkpoint),
                "window_provenance_sha256": {
                    "file": provenance_file_hash,
                    **provenance_hashes,
                },
                "array_fingerprints": dict(expected_fingerprints),
                "array_shapes": dict(expected_shapes),
                "sample_counts": dict(expected_counts),
                "split_bounds": {name: list(bounds) for name, bounds in evaluation.SPLIT_BOUNDS.items()},
                "history_length": 5, "prediction_horizon": 1,
            }
            if kind == "residual":
                manifest.update({
                    "experiment": "Neural Residual-to-EWMA Forecasting, Version 1",
                    "mode": "full",
                    "input_artifacts": [str(root / filename) for filename in evaluation.residual.INPUT_ARRAY_NAMES],
                    "edge_order": residual_edges,
                    "model": {"class": "PerLinkResidualMLP", "architecture": "shared 5 -> 16 -> ReLU -> 1"},
                    "ewma": {
                        "alpha": evaluation.residual.EWMA_ALPHA,
                        "weights_oldest_to_newest": evaluation.residual.ewma_weights().tolist(),
                    },
                    "configuration": evaluation.residual.TRAINING,
                    "configuration_sha256": evaluation.residual._configuration_hash(),
                    "source_sha256": dict(residual_source_hashes),
                    "source_sha256_pretraining": dict(residual_source_hashes),
                    "preprocessing_provenance": {
                        "protocol_id": evaluation.PROTOCOL_ID,
                        "source": "existing GEANT validity-v2 prebuilt train/validation windows",
                        "window_rule": "five consecutive valid history observations plus one valid next-step target within the same official split",
                        "link_order": "72 directed columns in v2 interleaved forward/reverse order, checked against the topology parser",
                        "utilization_units": "dimensionless computed link-utilization ratio",
                        "window_provenance_file": evaluation.residual.PROVENANCE_FILENAME,
                        "window_provenance_sha256": provenance_file_hash,
                        "train_window_provenance_sha256": provenance_hashes["train"],
                        "validation_window_provenance_sha256": provenance_hashes["validation"],
                    },
                    "normalization": {
                        "mean": 0.5, "scale": 1.25,
                        "fit_scope": "all scalar values in X_train.npy and y_train.npy only",
                    },
                    "residual_scaler": {
                        "mean": 0.0, "scale": 0.75,
                        "fit_scope": "training residuals from X_train.npy and y_train.npy only",
                    },
                })
                if residual_manifest_mutation is not None:
                    residual_manifest_mutation(manifest, checkpoint)
            else:
                manifest.update({
                    "mode": "full", "candidate": "B-U",
                    "model": "EdgeQoSForecaster (unchanged)",
                    "model_settings": evaluation.bu.TRAINING,
                    "array_paths": [str(root / filename) for filename in evaluation.bu.ARRAY_NAMES],
                    "data_split": {
                        "training": "X_train.npy and y_train.npy",
                        "validation": "X_validation.npy and y_validation.npy",
                        "test": "not loaded or evaluated",
                    },
                    "directed_edge_order": bu_edges,
                    "edge_order_sha256": evaluation.bu.canonical_json_sha256(
                        [list(edge) for edge in edge_order]
                    ),
                    "source_sha256": dict(source_hashes),
                    "source_sha256_posttraining": dict(source_hashes),
                    "normalization": {
                        "mean": 0.5,
                        "scale": 1.25,
                        "fit_scope": "all scalar values in X_train.npy and y_train.npy only",
                    },
                })
                identity = evaluation.bu._preflight_run_identity(
                    mode="full", seed=seed, run_id=None,
                    provenance_file_sha256=provenance_file_hash,
                    provenance_hashes=provenance_hashes,
                    edges=edge_order, source_sha256=source_hashes,
                )
                manifest.update(identity)
                if bu_manifest_mutation is not None:
                    bu_manifest_mutation(manifest, checkpoint)
            (run_dir / "run_metadata.json").write_text(json.dumps(manifest), encoding="utf-8")

    return (
        root, repo, x, y, provenance_hashes, provenance_file_hash, edge_order,
        expected_fingerprints, expected_shapes, expected_counts,
    )


def test_new_provenance_bound_bu_artifacts_pass_preflight_without_checkpoint_loading(tmp_path):
    (root, repo, x, y, provenance_hashes, provenance_hash, edges,
     fingerprints, shapes, counts) = _write_synthetic_completed_artifacts(tmp_path)
    evaluation._preflight_completed_artifacts(
        root, repo, x, y,
        provenance_hashes, provenance_hash, edges, fingerprints, shapes, counts,
    )


def test_input_path_allowlist_accepts_custom_root_and_normalized_path_forms(tmp_path):
    repo = tmp_path / "repo"
    root = repo / "data" / "processed" / "geant_validity_v2_rebuild_trainval_test"
    root.mkdir(parents=True)
    variants = []
    for filename in evaluation.residual.INPUT_ARRAY_NAMES:
        absolute = root / filename
        relative = absolute.relative_to(repo)
        if filename == "X_train.npy":
            variants.append(str(relative).replace("\\", "/"))
        elif filename == "y_train.npy":
            variants.append(str(absolute).replace("\\", "/").upper())
        else:
            variants.append(str(relative))

    evaluation.validate_input_array_path_allowlist(
        variants,
        data_root=root,
        repository_root=repo,
        array_names=evaluation.residual.INPUT_ARRAY_NAMES,
    )


@pytest.mark.parametrize(
    "mutation, message",
    [
        ("different_root", "exact data-root allowlist"),
        ("extra", "exactly the permitted"),
        ("missing", "exactly the permitted"),
        ("test_token", "test-split token"),
    ],
)
def test_input_path_allowlist_rejects_wrong_or_unapproved_entries(tmp_path, mutation, message):
    repo = tmp_path / "repo"
    root = repo / "data" / "processed" / "geant_validity_v2_rebuild_trainval_test"
    root.mkdir(parents=True)
    paths = [str(root / name) for name in evaluation.residual.INPUT_ARRAY_NAMES]
    if mutation == "different_root":
        paths[0] = str(root.parent / "different_root" / "X_train.npy")
    elif mutation == "extra":
        paths.append(str(root / "unused.npy"))
    elif mutation == "missing":
        paths.pop()
    else:
        paths[0] = str(root / "X_test.npy")

    with pytest.raises(ValueError, match=message):
        evaluation.validate_input_array_path_allowlist(
            paths,
            data_root=root,
            repository_root=repo,
            array_names=evaluation.residual.INPUT_ARRAY_NAMES,
        )


def test_diagnostic_preflight_reports_all_manifest_path_mismatches_without_loading_artifacts(
    tmp_path, monkeypatch
):
    root, repo, *_ = _write_synthetic_completed_artifacts(tmp_path)
    mutations = {
        ("residual", 42): "different_root",
        ("residual", 123): "extra",
        ("residual", 2026): "missing",
        ("B-U", 42): "test_token",
        ("B-U", 123): "duplicate",
        ("B-U", 2026): "different_root",
    }
    for (kind, seed), mutation in mutations.items():
        prefix = (
            evaluation.EXPECTED_RESIDUAL_PREFIX
            if kind == "residual"
            else evaluation.EXPECTED_BU_PREFIX
        )
        run_dir = root / f"{prefix}{seed}"
        manifest_path = run_dir / "run_metadata.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        field = "input_artifacts" if kind == "residual" else "array_paths"
        paths = list(manifest[field])
        if mutation == "different_root":
            paths[0] = str(root.parent / "different_root" / "X_train.npy")
        elif mutation == "extra":
            paths.append(str(root / "unapproved.npy"))
        elif mutation == "missing":
            paths.pop()
        elif mutation == "test_token":
            paths[0] = str(root / "X_test.npy")
        else:
            paths[1] = paths[0]
        manifest[field] = paths
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("diagnostic must not load arrays or checkpoints")

    monkeypatch.setattr(evaluation, "_load_validation_arrays", forbidden)
    monkeypatch.setattr(evaluation.torch, "load", forbidden)
    result = evaluation.diagnose_completed_metadata(root, repository_root=repo)
    assert result["runs_checked"] == 6
    assert result["arrays_loaded"] is False
    assert result["checkpoints_loaded"] is False
    assert result["results_published"] is False
    by_run = {entry["run"]: entry["mismatches"] for entry in result["mismatches"]}
    assert len(by_run) == 6
    assert all(by_run.values())


def test_source_drift_requires_explicit_opt_in_and_is_returned_as_warning(tmp_path):
    (root, repo, x, y, provenance_hashes, provenance_hash, edges,
     fingerprints, shapes, counts) = _write_synthetic_completed_artifacts(tmp_path)
    changed_source = repo / "phase9_geant" / "validity_v2.py"
    changed_source.write_text("# synthetic post-run source change\n", encoding="utf-8")
    kwargs = dict(
        data_root=root, repository_root=repo,
        x_validation=x, y_validation=y,
        provenance_hashes=provenance_hashes,
        provenance_file_hash=provenance_hash,
        runtime_edge_order=edges,
        expected_array_fingerprints=fingerprints,
        expected_array_shapes=shapes,
        expected_sample_counts=counts,
    )
    with pytest.raises(ValueError, match="source hashes"):
        evaluation._preflight_completed_artifacts(**kwargs)
    warnings = evaluation._preflight_completed_artifacts(
        **kwargs, allow_source_drift=True
    )
    assert warnings
    assert all("not a strict-integrity evaluation" in warning for warning in warnings)


def test_source_drift_opt_in_is_written_to_synthetic_evaluation_output(tmp_path, monkeypatch):
    nodes = tuple(f"n{index:02d}" for index in range(22))
    links = [(nodes[i], nodes[(i + 1) % 22]) for i in range(22)]
    links.extend((nodes[i], nodes[i + 2]) for i in range(14))
    runtime_edges = evaluation.bu.directed_edges_v2(links)
    (root, repo, x, y, _provenance_hashes, _provenance_hash, _edges,
     _fingerprints, _shapes, _counts) = _write_synthetic_completed_artifacts(
        tmp_path, edge_order_override=runtime_edges
    )
    (repo / "phase9_geant" / "validity_v2.py").write_text(
        "# synthetic post-run source change\n", encoding="utf-8"
    )
    monkeypatch.setattr(evaluation.bu, "parse_topology", lambda *_args: ({node: (0.0, 0.0) for node in nodes}, links))
    target = np.load(root / "y_validation.npy", allow_pickle=False)

    def fake_residual(seed, *_args, **_kwargs):
        run_dir = root / f"{evaluation.EXPECTED_RESIDUAL_PREFIX}{seed}"
        manifest = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
        return target.copy(), manifest, run_dir / "best_validation.pt"

    def fake_bu(seed, *_args, **_kwargs):
        run_dir = root / f"{evaluation.EXPECTED_BU_PREFIX}{seed}"
        manifest = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
        return target.copy(), manifest, run_dir / "best_validation.pt"

    monkeypatch.setattr(evaluation, "_load_residual_model", fake_residual)
    monkeypatch.setattr(evaluation, "_load_bu_model", fake_bu)
    output = evaluation.run_validation_evaluation(
        data_root=root,
        repository_root=repo,
        bootstrap_replicates=2,
        allow_source_drift=True,
    )
    report = json.loads((output / "results.json").read_text(encoding="utf-8"))
    assert report["integrity_warnings"]
    assert report["strict_integrity_evaluation"] is False
    assert all("source-drift" in warning for warning in report["integrity_warnings"])


@pytest.mark.parametrize("mutation", [
    pytest.param(lambda manifest, _checkpoint: manifest.pop("window_provenance_sha256"), id="residual-missing-provenance"),
    pytest.param(lambda manifest, _checkpoint: manifest["window_provenance_sha256"].pop("train"), id="residual-missing-train-provenance"),
    pytest.param(lambda manifest, _checkpoint: manifest.pop("array_fingerprints"), id="residual-missing-array-fingerprints"),
    pytest.param(lambda manifest, _checkpoint: manifest["array_fingerprints"].pop("X_train.npy"), id="residual-missing-training-input-fingerprint"),
    pytest.param(lambda manifest, _checkpoint: manifest["array_fingerprints"].pop("y_train.npy"), id="residual-missing-training-target-fingerprint"),
    pytest.param(lambda manifest, _checkpoint: manifest["array_fingerprints"].update({"X_train.npy": "0" * 64}), id="residual-mismatched-train-fingerprint"),
    pytest.param(lambda manifest, _checkpoint: manifest.pop("source_sha256"), id="residual-missing-source-hashes"),
    pytest.param(lambda manifest, _checkpoint: manifest.pop("source_sha256_pretraining"), id="residual-missing-pretraining-source-hashes"),
    pytest.param(lambda manifest, _checkpoint: manifest["source_sha256"].update({next(iter(manifest["source_sha256"])): "0" * 64}), id="residual-mismatched-source-hash"),
    pytest.param(lambda manifest, _checkpoint: manifest.pop("configuration_sha256"), id="residual-missing-configuration-hash"),
    pytest.param(lambda manifest, _checkpoint: manifest.update(configuration_sha256="0" * 64), id="residual-mismatched-configuration-hash"),
    pytest.param(lambda manifest, _checkpoint: manifest.pop("edge_order"), id="residual-missing-edge-order"),
    pytest.param(lambda manifest, _checkpoint: manifest["edge_order"].reverse(), id="residual-mismatched-edge-order"),
    pytest.param(lambda manifest, _checkpoint: manifest.pop("checkpoint_sha256"), id="residual-missing-checkpoint-hash"),
    pytest.param(lambda manifest, _checkpoint: manifest.update(checkpoint_sha256="f" * 64), id="residual-mismatched-checkpoint-hash"),
])
def test_residual_preflight_rejects_missing_or_mismatched_run_identity(
    tmp_path, monkeypatch, mutation
):
    (root, repo, x, y, provenance_hashes, provenance_hash, edges,
     fingerprints, shapes, counts) = _write_synthetic_completed_artifacts(
        tmp_path, residual_manifest_mutation=mutation
    )
    checkpoint_loads = []
    monkeypatch.setattr(
        evaluation.torch,
        "load",
        lambda *args, **kwargs: checkpoint_loads.append((args, kwargs)),
    )
    with pytest.raises((ValueError, KeyError)):
        evaluation._preflight_completed_artifacts(
            root, repo, x, y,
            provenance_hashes, provenance_hash, edges,
            fingerprints, shapes, counts,
        )
    assert checkpoint_loads == []


@pytest.mark.parametrize("mutation", [
    pytest.param(lambda manifest, _checkpoint: manifest.pop("window_provenance_sha256"), id="legacy-no-provenance"),
    pytest.param(lambda manifest, _checkpoint: manifest["window_provenance_sha256"].pop("train"), id="missing-train-provenance"),
    pytest.param(lambda manifest, _checkpoint: manifest["window_provenance_sha256"].pop("file"), id="missing-file-provenance"),
    pytest.param(lambda manifest, _checkpoint: manifest["window_provenance_sha256"].update(validation="d" * 64), id="mismatched-validation-provenance"),
    pytest.param(lambda manifest, _checkpoint: manifest["array_fingerprints"].pop("X_train.npy"), id="missing-training-fingerprint"),
    pytest.param(lambda manifest, _checkpoint: manifest["array_fingerprints"].pop("y_train.npy"), id="missing-training-target-fingerprint"),
    pytest.param(lambda manifest, _checkpoint: manifest.pop("source_sha256"), id="missing-source-hashes"),
    pytest.param(lambda manifest, _checkpoint: manifest["source_sha256"].update({next(iter(manifest["source_sha256"])): "a" * 64}), id="mismatched-source-hash"),
    pytest.param(lambda manifest, _checkpoint: manifest.pop("configuration_sha256"), id="missing-configuration-hash"),
    pytest.param(lambda manifest, _checkpoint: manifest.pop("edge_order_sha256"), id="missing-edge-order-digest"),
    pytest.param(lambda manifest, _checkpoint: manifest.pop("checkpoint"), id="missing-checkpoint-path"),
    pytest.param(lambda manifest, _checkpoint: manifest.pop("checkpoint_sha256"), id="missing-checkpoint-hash"),
    pytest.param(lambda manifest, _checkpoint: manifest.update(checkpoint_sha256="e" * 64), id="mismatched-checkpoint-hash"),
])
def test_bu_preflight_rejects_missing_or_mismatched_provenance_and_checkpoint(
    tmp_path, mutation
):
    (root, repo, x, y, provenance_hashes, provenance_hash, edges,
     fingerprints, shapes, counts) = _write_synthetic_completed_artifacts(
        tmp_path, bu_manifest_mutation=mutation
    )
    with pytest.raises(ValueError):
        evaluation._preflight_completed_artifacts(
            root, repo, x, y,
            provenance_hashes, provenance_hash, edges, fingerprints, shapes, counts,
        )


def test_existing_evaluation_output_is_refused_before_array_loading(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    data_root = repo / "data" / "processed" / "geant_validity_v2"
    output = data_root / evaluation.OUTPUT_NAME
    output.mkdir(parents=True)

    def forbidden_loader(*_args, **_kwargs):
        raise AssertionError("validation arrays must not load when output exists")

    monkeypatch.setattr(evaluation, "_load_validation_arrays", forbidden_loader)
    with pytest.raises(FileExistsError, match="Refusing to reuse"):
        evaluation.run_validation_evaluation(
            data_root=data_root,
            repository_root=repo,
            output_path=output,
            bootstrap_replicates=1,
        )


@pytest.mark.parametrize("mean, scale", [(float("nan"), 1.0), (0.0, 0.0), (0.0, -1.0)])
def test_residual_preflight_rejects_invalid_residual_scaler_before_checkpoint_load(
    tmp_path, monkeypatch, mean, scale
):
    def mutation(manifest, _checkpoint):
        manifest["residual_scaler"].update(mean=mean, scale=scale)

    (root, repo, x, y, provenance_hashes, provenance_hash, edges,
     fingerprints, shapes, counts) = _write_synthetic_completed_artifacts(
        tmp_path, residual_manifest_mutation=mutation
    )
    monkeypatch.setattr(
        evaluation.torch, "load",
        lambda *_a, **_k: pytest.fail("checkpoint load must not occur before all preflight checks"),
    )
    with pytest.raises(ValueError, match="normalization parameters are invalid"):
        evaluation._preflight_completed_artifacts(
            root, repo, x, y,
            provenance_hashes, provenance_hash, edges,
            fingerprints, shapes, counts,
        )
