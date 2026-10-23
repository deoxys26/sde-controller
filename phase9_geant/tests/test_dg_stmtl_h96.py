from __future__ import annotations

import hashlib
import inspect
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from phase9_geant import compare_dg_stmtl_h96 as compare
from phase9_geant import dg_stmtl_h96_experiment as experiment
from phase9_geant import dg_stmtl_h96_workflow as workflow
from src.gnn.dg_stmtl_link import DGSTMTLLinkForecaster, build_hybrid_prior

# These tests are CPU-only and tiny; a single BLAS thread avoids thread-pool
# startup dominating their deliberately small synthetic tensor operations.
torch.set_num_threads(1)


@pytest.fixture
def synthetic_root(tmp_path_factory):
    # Use pytest's external basetemp allocator. The repository-level tmp_path
    # override avoids a Windows symlink, while tmp_path_factory creates a
    # writable isolated directory under the same validated external root.
    return tmp_path_factory.mktemp("dg-stmtl")


def _edges():
    return tuple((str(i), str(j)) for i in range(9) for j in range(9) if i != j)


def _small_model():
    return DGSTMTLLinkForecaster(_edges(), hidden_size=4, dynamic_dim=2)


def test_hybrid_prior_preserves_caller_edge_order_and_three_slot_contract():
    edges = _edges()
    prior = build_hybrid_prior(edges)
    permutation = tuple(reversed(range(len(edges))))
    reordered = tuple(edges[index] for index in permutation)
    reordered_prior = build_hybrid_prior(reordered)
    assert prior.shape == (216, 216)
    assert torch.equal(prior, prior.T)
    indices = torch.tensor([slot * 72 + index for slot in range(3) for index in permutation])
    assert torch.equal(reordered_prior, prior.index_select(0, indices).index_select(1, indices))


def test_single_h96_output_is_finite_and_zero_head_equals_ewma():
    model = _small_model().eval()
    history = torch.rand(2, 672, 72)
    with torch.no_grad():
        forecast = model(history)
        baseline = model.ewma(history)
    assert forecast.shape == (2, 72)
    assert torch.isfinite(forecast).all()
    assert torch.equal(forecast, baseline)
    assert model.residual_head.out_features == 1


def test_batch_samples_do_not_exchange_messages():
    torch.manual_seed(19)
    model = _small_model().eval()
    a, b = torch.rand(1, 672, 72), torch.rand(1, 672, 72)
    with torch.no_grad():
        batched = model(torch.cat((a, b), dim=0))
        separate = torch.cat((model(a), model(b)), dim=0)
    assert torch.allclose(batched, separate, atol=1e-6, rtol=1e-6)


def test_dynamic_adjacency_is_input_conditioned_and_forward_has_no_target_argument():
    model = _small_model().eval()
    x1 = torch.zeros(1, 672, 72)
    x2 = x1.clone()
    x2[:, :, 0] = torch.linspace(0, 1, 672)
    with torch.no_grad():
        a1 = model.hybrid_adjacency(model._input_tokens(x1))
        a2 = model.hybrid_adjacency(model._input_tokens(x2))
    assert a1.shape == (1, 216, 216)
    assert not torch.allclose(a1, a2)
    assert "target" not in inspect.signature(model.forward).parameters


def test_gradient_propagates_through_dynamic_gate_and_group_graphs():
    torch.manual_seed(21)
    model = _small_model()
    torch.nn.init.normal_(model.residual_head.weight, std=0.1)
    history = torch.rand(2, 672, 72)
    loss = model(history).square().mean()
    loss.backward()
    for name in ("task_gate_logits", "dynamic_projection.weight", "temporal_group_block.self_layers.0.weight",
                 "feature_group_block.self_layers.0.weight", "residual_head.weight"):
        gradient = dict(model.named_parameters())[name].grad
        assert gradient is not None and torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient) > 0


def test_explicit_authorization_precedes_any_preflight_or_data_loader(monkeypatch, synthetic_root):
    def forbidden(*args, **kwargs):
        raise AssertionError("preflight/data loader must not run without authorization")
    monkeypatch.setattr(experiment, "canonical_repository_root", forbidden)
    with pytest.raises(PermissionError):
        experiment.train_one(synthetic_root, repository_root=synthetic_root, seed=42)


def test_run_authorization_phrase_is_per_seed_and_exact():
    phrase = workflow.authorization_phrase(123)
    assert phrase == "AUTHORIZE DG-STMTL H96 SEED 123 TOPOLOGY"
    workflow.require_authorization(123, phrase)
    with pytest.raises(PermissionError):
        workflow.require_authorization(123, "AUTHORIZE DG-STMTL H96 SEED 42 TOPOLOGY")


def _completed_synthetic_run(root: Path, seed: int = 42) -> tuple[Path, dict]:
    run = root / f"dg_stmtl_topology_seed_{seed}"
    run.mkdir(parents=True)
    checkpoint = b"synthetic-opaque-checkpoint"
    predictions = b"synthetic-opaque-validation-predictions"
    (run / "best.pt").write_bytes(checkpoint)
    (run / "validation_predictions.npy").write_bytes(predictions)
    metric = {"mae": 0.01, "rmse": 0.02, "evaluated_pairs": 953 * 72}
    manifest = {
        "status": "complete", "protocol_id": workflow.PROTOCOL_ID, "seed": seed,
        "adjacency_mode": "hybrid_topology_dynamic", "checkpoint": "best.pt",
        "validation_predictions": "validation_predictions.npy", "checkpoint_reload_strict": True,
        "test_access": {"permitted": False}, "epochs_completed": 1, "best_epoch": 1,
        "validation_metrics": {"dg_stmtl_based": metric, "ewma_alpha_0.8": metric},
        "checkpoint_sha256": hashlib.sha256(checkpoint).hexdigest(),
        "validation_predictions_sha256": hashlib.sha256(predictions).hexdigest(),
    }
    (run / "run.json").write_text(json.dumps(manifest), encoding="utf-8")
    return run, manifest


def test_persistent_backup_verifies_readback_and_refuses_overwrite(synthetic_root):
    run, _ = _completed_synthetic_run(synthetic_root / "local")
    drive = synthetic_root / "drive"
    drive.mkdir()
    result = workflow.backup_completed_run(run, drive, expected_seed=42)
    assert result["verified"] is True
    backup = Path(result["directory"])
    assert {p.name for p in backup.iterdir()} == set(workflow.ARTIFACTS) | {"backup_manifest.json"}
    assert hashlib.sha256((backup / "best.pt").read_bytes()).hexdigest() == result["artifacts"]["best.pt"]["sha256"]
    with pytest.raises(FileExistsError):
        workflow.backup_completed_run(run, drive, expected_seed=42)


def test_persistent_backup_rejects_changed_checkpoint_before_copy(synthetic_root):
    run, _ = _completed_synthetic_run(synthetic_root / "local")
    (run / "best.pt").write_bytes(b"tampered")
    drive = synthetic_root / "drive"
    drive.mkdir()
    with pytest.raises(workflow.DGWorkflowError, match="SHA-256"):
        workflow.backup_completed_run(run, drive, expected_seed=42)
    assert not (drive / "runs").exists()


def test_training_existing_destination_refuses_before_input_loader(monkeypatch, synthetic_root):
    repo = synthetic_root / "repo"
    processed = repo / "data" / "processed"
    output = processed / experiment.OUTPUT_ROOT_NAME / "dg_stmtl_topology_seed_42"
    output.mkdir(parents=True)
    called = []
    monkeypatch.setattr(experiment, "canonical_repository_root", lambda path: Path(path))
    monkeypatch.setattr(experiment, "_verified_inputs", lambda *args: called.append(True))
    with pytest.raises(FileExistsError):
        experiment.train_one(synthetic_root / "data", repository_root=repo, seed=42,
                             run_training=True, explicit_authorization=True)
    assert called == []


def test_synthetic_training_lifecycle_finalizes_checkpoint_before_complete(monkeypatch, synthetic_root):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        _run_synthetic_training_lifecycle(monkeypatch, synthetic_root)
    finally:
        torch.set_num_threads(previous_threads)


def _run_synthetic_training_lifecycle(monkeypatch, synthetic_root):
    repo = synthetic_root / "repo"
    processed = repo / "data" / "processed"
    processed.mkdir(parents=True)
    data_root = processed / "synthetic_train_validation"
    data_root.mkdir()
    train_series = np.linspace(0.01, 0.2, 800 * 72, dtype=np.float32).reshape(800, 72)
    validation_series = np.linspace(0.02, 0.18, 800 * 72, dtype=np.float32).reshape(800, 72)
    splits = {
        "train": {"series": train_series, "windows": [{"history_start": 0, "origin": 671, "target": 767}]},
        "validation": {"series": validation_series, "windows": [{"history_start": 8020, "origin": 8691, "target": 8787}]},
    }
    hashes = {name: f"{index:064x}" for index, name in enumerate(sorted(experiment.base.lazy.DATA_ARRAY_NAMES), 1)}
    metadata = {"array_files": {name: {"file_sha256": digest} for name, digest in hashes.items()},
                "synthetic": True}
    edges = _edges()
    source_hashes = {"synthetic_source.py": "a" * 64}

    def verified(_data_root, repository_root):
        return (Path(repository_root), data_root, splits, metadata, edges,
                {"synthetic_train_validation_only": True}, source_hashes)

    class Monitor:
        def __init__(self):
            self.phases = []
        def sample(self, phase):
            self.phases.append(phase)
            return {"synthetic": True}
        def summary(self):
            return {"synthetic": True, "samples": len(self.phases)}

    test_config = {**experiment.CONFIG, "max_epochs": 1, "early_stopping_patience": 1, "batch_size": 1}
    monkeypatch.setattr(experiment, "CONFIG", test_config)
    monkeypatch.setattr(experiment, "canonical_repository_root", lambda path: Path(path))
    monkeypatch.setattr(experiment, "_verified_inputs", verified)
    monkeypatch.setattr(experiment, "source_identity", lambda _repo: source_hashes)
    monkeypatch.setattr(experiment.linux_resources, "preflight", lambda **kwargs: {"status": "pass", "synthetic": True})
    monitor = Monitor()
    result = experiment.train_one(data_root, repository_root=repo, seed=42,
                                  run_training=True, explicit_authorization=True, monitor=monitor)
    run = processed / experiment.OUTPUT_ROOT_NAME / "dg_stmtl_topology_seed_42"
    assert result["status"] == "complete"
    assert (run / "best.pt").is_file() and (run / "best.pt").stat().st_size > 0
    assert result["checkpoint_sha256"] == hashlib.sha256((run / "best.pt").read_bytes()).hexdigest()
    assert result["checkpoint_reload_strict"] is True
    assert result["validation_metrics"]["dg_stmtl_based"]["evaluated_pairs"] == 72
    assert monitor.phases[0] == "startup_before_output"
    assert {item.name for item in run.iterdir()} == {"run.json", "best.pt", "validation_predictions.npy"}


def test_comparison_refuses_existing_output_directory(synthetic_root):
    existing = synthetic_root / "report"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        compare.write_comparison({}, existing)


def test_comparison_validates_exact_manifest_counts_and_test_denial(synthetic_root):
    # Manifests are synthetic JSON only; no arrays or checkpoints are opened.
    paths = []
    for mode in ("topology", "identity", "shuffled"):
        for seed in (42, 123, 2026):
            path = synthetic_root / f"{mode}_{seed}.json"
            path.write_text(json.dumps({"status": "failed", "seed": seed, "adjacency_mode": mode}), encoding="utf-8")
            paths.append(path)
    dg_paths = []
    for seed in (42, 123, 2026):
        path = synthetic_root / f"dg_{seed}.json"
        path.write_text(json.dumps({"status": "failed", "seed": seed}), encoding="utf-8")
        dg_paths.append(path)
    with pytest.raises(compare.ComparisonError, match="incomplete"):
        compare.collect_comparison(paths, dg_paths)


def _synthetic_comparison_inputs(root: Path, *, alter_one_ewma: bool = False,
                                ewma_delta: float = 0.0, ewma_delta_field: str = "mae",
                                config_mismatch: bool = False, provenance_mismatch: bool = False,
                                edge_mismatch: bool = False, normalization_mismatch: bool = False,
                                coverage_mismatch: bool = False):
    gnn_paths, dg_paths = [], []
    provenance = {
        "dataset_protocol_id": "geant-h96-streaming-trainval-v1",
        "source_scope": {"train": [0, 8020], "validation": [8020, 9740]},
        "source_file_counts": {"train": 8020, "validation": 1720},
        "window_index_sha256": dict(compare.APPROVED_WINDOW_INDEX_SHA256),
        "array_file_records": {
            name: {"file_sha256": digest} for name, digest in compare.APPROVED_ARRAY_SHA256.items()
        },
        "training_windows_h96": 3924, "validation_windows_h96": 953,
        "ordered_edge_sha256": compare.APPROVED_ORDERED_EDGE_SHA256,
    }
    config_common = {
        "history_length": 672, "ewma_alpha": 0.8, "ewma_steps": 5,
        "train_bounds": [0, 8020], "validation_bounds": [8020, 9740],
    }
    metric_source_hash = "f" * 64
    scaler_source_hash = "e" * 64
    for mode in ("topology", "identity", "shuffled"):
        for seed in (42, 123, 2026):
            run_provenance = dict(provenance)
            if provenance_mismatch and mode == "identity" and seed == 123:
                run_provenance["window_index_sha256"] = {
                    "train": "3" * 64, "validation": compare.APPROVED_WINDOW_INDEX_SHA256["validation"]
                }
            gnn_config = {**config_common, "horizons": [96], "seed": seed,
                          "adjacency_mode": mode, "device": "cpu",
                          "dataset_protocol_id": "geant-h96-streaming-trainval-v1",
                          "scaler_fit_scope": next(iter(compare.GNN_SCALER_SCOPE_DESCRIPTIONS)),
                          "colab_bundle_identity": {
                              "dataset_bundle_sha256": compare.DATASET_BUNDLE_SHA256,
                              "dataset_manifest_sha256": compare.DATASET_BUNDLE_MANIFEST_SHA256,
                          }}
            payload = {
                "status": "complete", "protocol_id": compare.GNN_PROTOCOL,
                "seed": seed, "adjacency_mode": mode,
                "configuration": gnn_config,
                "configuration_sha256": compare._canonical_json_sha256(gnn_config),
                "provenance": run_provenance,
                "source_hashes": {compare.SHARED_METRIC_SOURCE: metric_source_hash,
                                  compare.SHARED_SCALER_SOURCE: scaler_source_hash},
                "test_access": {"permitted": False}, "checkpoint_reload_strict": True,
                "checkpoint": "best.pt", "checkpoint_sha256": "c" * 64,
                "validation_windows": 953, "dataset_metadata_sha256": "a" * 64,
                "topology": {"ordered_edge_sha256": compare.APPROVED_ORDERED_EDGE_SHA256},
                "normalization": {"mean": 0.1, "scale": 0.2, "fit_scope": compare.SCALER_FIT_SCOPE,
                                  "population_count": 100},
                "model_parameter_count": 123, "runtime_seconds": 10.0,
                "validation_metrics": {
                    "h96_candidate": {"mae": 0.01, "rmse": 0.02, "evaluated_pairs": 953 * 72,
                                       "units": "computed utilization ratio"},
                    "ewma_alpha_0.8": {"mae": 0.03, "rmse": 0.04, "evaluated_pairs": 953 * 72,
                                        "units": "computed utilization ratio"},
                },
            }
            if config_mismatch and mode == "identity" and seed == 123:
                payload["configuration"]["ewma_steps"] = 4
                payload["configuration_sha256"] = compare._canonical_json_sha256(payload["configuration"])
            if edge_mismatch and mode == "identity" and seed == 123:
                payload["topology"]["ordered_edge_sha256"] = "9" * 64
            if normalization_mismatch and mode == "identity" and seed == 123:
                payload["normalization"]["mean"] = 0.101
            if coverage_mismatch and mode == "identity" and seed == 123:
                payload["validation_metrics"]["h96_candidate"]["evaluated_pairs"] -= 1
            path = root / f"gnn_{mode}_{seed}.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            gnn_paths.append(path)
    for seed in (42, 123, 2026):
        ewma_mae = 0.031 if alter_one_ewma and seed == 123 else 0.03
        ewma_rmse = 0.04 + (ewma_delta if ewma_delta_field == "rmse" else 0.0)
        if ewma_delta_field == "mae":
            ewma_mae += ewma_delta
        run_provenance = dict(provenance)
        if provenance_mismatch and seed == 123:
            run_provenance["window_index_sha256"] = {
                "train": "3" * 64, "validation": compare.APPROVED_WINDOW_INDEX_SHA256["validation"]
            }
        dg_config = {**config_common, "horizon": 96, "seed": seed,
                     "adjacency_mode": "hybrid_topology_dynamic", "device": "cpu",
                     "scaler_fit_scope": next(iter(compare.DG_SCALER_SCOPE_DESCRIPTIONS))}
        payload = {
            "status": "complete", "protocol_id": compare.DG_PROTOCOL,
            "seed": seed, "adjacency_mode": "hybrid_topology_dynamic",
            "configuration": dg_config,
            "configuration_sha256": compare._canonical_json_sha256(dg_config),
            "provenance": run_provenance,
            "source_hashes": {compare.SHARED_METRIC_SOURCE: metric_source_hash,
                              compare.SHARED_SCALER_SOURCE: scaler_source_hash},
            "test_access": {"permitted": False}, "checkpoint_reload_strict": True,
            "checkpoint": "best.pt", "checkpoint_sha256": "d" * 64,
            "validation_windows": 953, "dataset_metadata_sha256": "a" * 64,
            "topology": {"ordered_edge_sha256": compare.APPROVED_ORDERED_EDGE_SHA256},
            "permitted_array_file_hashes": dict(compare.APPROVED_ARRAY_SHA256),
            "normalization": {"mean": 0.1, "scale": 0.2, "fit_scope": compare.SCALER_FIT_SCOPE,
                              "population_count": 100},
            "model_parameter_count": 52513, "runtime_seconds": 11.0,
            "validation_metrics": {
                "dg_stmtl_based": {"mae": 0.009, "rmse": 0.019, "evaluated_pairs": 953 * 72,
                                    "units": "computed utilization ratio"},
                "ewma_alpha_0.8": {"mae": ewma_mae, "rmse": ewma_rmse, "evaluated_pairs": 953 * 72,
                                     "units": "computed utilization ratio"},
            },
        }
        path = root / f"dg_{seed}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        dg_paths.append(path)
    return gnn_paths, dg_paths


def test_comparison_accepts_identical_per_seed_ewma_baselines(synthetic_root):
    gnn_paths, dg_paths = _synthetic_comparison_inputs(synthetic_root)
    result = compare.collect_comparison(gnn_paths, dg_paths)
    assert result["status"] == "complete"
    assert len(result["per_seed"]) == 3
    assert result["test_access"] is False
    assert "relative_mae_improvement_vs_ewma_percent" in result["per_seed"][0]["dg_stmtl_based"]
    assert "relative_rmse_improvement_vs_ewma_percent" in result["per_seed"][0]["gnn"]["topology"]


def test_comparison_accepts_equivalent_provenance_schema_layouts(synthetic_root):
    gnn_paths, dg_paths = _synthetic_comparison_inputs(synthetic_root)
    # Equivalent layouts: GNN declares the dataset protocol in config and carries
    # per-file records in provenance; DG declares protocol in provenance and
    # carries the same approved hashes in its explicit permitted-file map.
    for path in gnn_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["provenance"].pop("dataset_protocol_id")
        payload["training_window_count"] = payload["provenance"].pop("training_windows_h96")
        payload["validation_window_count"] = payload["provenance"].pop("validation_windows_h96")
        path.write_text(json.dumps(payload), encoding="utf-8")
    for path in dg_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["configuration"].pop("dataset_protocol_id", None)
        path.write_text(json.dumps(payload), encoding="utf-8")
    result = compare.collect_comparison(gnn_paths, dg_paths)
    assert result["status"] == "complete"


def test_manifest_diagnostic_is_read_only_and_accepts_equivalent_layouts(synthetic_root, capsys):
    original_gnn, original_dg = _synthetic_comparison_inputs(synthetic_root)
    gnn_paths, dg_paths = [], []
    for kind, sources, destination in (("gnn", original_gnn, gnn_paths), ("dg", original_dg, dg_paths)):
        for source in sources:
            payload = json.loads(source.read_text(encoding="utf-8"))
            if kind == "gnn" and payload["adjacency_mode"] == "topology" and payload["seed"] == 42:
                continue
            folder = synthetic_root / f"diagnostic_{kind}_{payload.get('adjacency_mode', payload['seed'])}_{payload['seed']}"
            folder.mkdir()
            run_path = folder / "run.json"
            run_path.write_text(json.dumps(payload), encoding="utf-8")
            destination.append(run_path)
    arguments = []
    for path in gnn_paths:
        arguments.extend(("--gnn-run-json", str(path)))
    for path in dg_paths:
        arguments.extend(("--dg-run-json", str(path)))
    assert compare.main([*arguments, "--allow-incomplete-gnn-matrix", "--diagnostic-only"]) == 0
    diagnostic = json.loads(capsys.readouterr().out)
    assert diagnostic["status"] == "compatible_manifest_identity"
    assert diagnostic["artifacts_written"] is False
    assert diagnostic["common_semantic_data_identity"]["validation_windows"] == 953
    assert diagnostic["test_access"] is False


def test_comparison_rejects_validation_window_hash_mismatch(synthetic_root):
    gnn_paths, dg_paths = _synthetic_comparison_inputs(synthetic_root)
    payload = json.loads(dg_paths[0].read_text(encoding="utf-8"))
    payload["provenance"]["window_index_sha256"]["validation"] = "9" * 64
    dg_paths[0].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(compare.ComparisonError, match="window identities"):
        compare.collect_comparison(gnn_paths, dg_paths)


def test_comparison_rejects_validation_mask_fingerprint_mismatch(synthetic_root):
    gnn_paths, dg_paths = _synthetic_comparison_inputs(synthetic_root)
    payload = json.loads(dg_paths[0].read_text(encoding="utf-8"))
    payload["permitted_array_file_hashes"]["valid_validation.npy"] = "9" * 64
    dg_paths[0].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(compare.ComparisonError, match="array fingerprints"):
        compare.collect_comparison(gnn_paths, dg_paths)


def test_comparison_rejects_target_unit_mismatch(synthetic_root):
    gnn_paths, dg_paths = _synthetic_comparison_inputs(synthetic_root)
    payload = json.loads(dg_paths[0].read_text(encoding="utf-8"))
    payload["validation_metrics"]["dg_stmtl_based"]["units"] = "percent"
    dg_paths[0].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(compare.ComparisonError, match="metric units"):
        compare.collect_comparison(gnn_paths, dg_paths)


def test_comparison_rejects_scaler_algorithm_description_mismatch(synthetic_root):
    gnn_paths, dg_paths = _synthetic_comparison_inputs(synthetic_root)
    payload = json.loads(dg_paths[0].read_text(encoding="utf-8"))
    payload["configuration"]["scaler_fit_scope"] = "validation-fitted scaler"
    payload["configuration_sha256"] = compare._canonical_json_sha256(payload["configuration"])
    dg_paths[0].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(compare.ComparisonError, match="scaler configuration description"):
        compare.collect_comparison(gnn_paths, dg_paths)


def test_comparison_rejects_metric_source_implementation_mismatch(synthetic_root):
    gnn_paths, dg_paths = _synthetic_comparison_inputs(synthetic_root)
    payload = json.loads(dg_paths[0].read_text(encoding="utf-8"))
    payload["source_hashes"][compare.SHARED_METRIC_SOURCE] = "9" * 64
    dg_paths[0].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(compare.ComparisonError, match="metric and scaler implementation"):
        compare.collect_comparison(gnn_paths, dg_paths)


def test_comparison_accepts_observed_scale_ewma_rounding_differences(synthetic_root):
    gnn_paths, dg_paths = _synthetic_comparison_inputs(
        synthetic_root, ewma_delta=9.1523301e-11,
    )
    # Pin the reported MAE pair from the user's verified manifests, without
    # loading those real manifests or changing either recorded value.
    gnn_payload = json.loads(gnn_paths[0].read_text(encoding="utf-8"))
    gnn_payload["validation_metrics"]["ewma_alpha_0.8"]["mae"] = 0.006502658485291489
    for path in gnn_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["validation_metrics"]["ewma_alpha_0.8"]["mae"] = 0.006502658485291489
        path.write_text(json.dumps(payload), encoding="utf-8")
    for path in dg_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["validation_metrics"]["ewma_alpha_0.8"]["mae"] = 0.006502658393768188
        path.write_text(json.dumps(payload), encoding="utf-8")
    result = compare.collect_comparison(gnn_paths, dg_paths)
    assert result["per_seed"][0]["ewma"]["recorded_by_method"]["original_gnn"]["mae"] == 0.006502658485291489
    assert result["per_seed"][0]["ewma"]["recorded_by_method"]["dg_stmtl_based"]["mae"] == 0.006502658393768188


def test_comparison_accepts_small_ewma_rmse_rounding_difference(synthetic_root):
    gnn_paths, dg_paths = _synthetic_comparison_inputs(
        synthetic_root, ewma_delta=9.15e-11, ewma_delta_field="rmse",
    )
    result = compare.collect_comparison(gnn_paths, dg_paths)
    assert result["per_seed"][0]["ewma"]["recorded_by_method"]["original_gnn"]["rmse"] == 0.04
    assert result["per_seed"][0]["ewma"]["recorded_by_method"]["dg_stmtl_based"]["rmse"] == 0.04 + 9.15e-11


def test_comparison_rejects_ewma_rounding_difference_above_fixed_tolerance(synthetic_root):
    gnn_paths, dg_paths = _synthetic_comparison_inputs(synthetic_root, ewma_delta=1.01e-9)
    with pytest.raises(compare.ComparisonError, match="absolute difference"):
        compare.collect_comparison(gnn_paths, dg_paths)


def test_comparison_rejects_rmse_difference_above_fixed_tolerance(synthetic_root):
    gnn_paths, dg_paths = _synthetic_comparison_inputs(
        synthetic_root, ewma_delta=1.01e-9, ewma_delta_field="rmse",
    )
    with pytest.raises(compare.ComparisonError, match="EWMA rmse differs"):
        compare.collect_comparison(gnn_paths, dg_paths)


def test_comparison_rejects_incompatible_ewma_configuration(synthetic_root):
    gnn_paths, dg_paths = _synthetic_comparison_inputs(synthetic_root, config_mismatch=True)
    with pytest.raises(compare.ComparisonError, match="ewma_steps"):
        compare.collect_comparison(gnn_paths, dg_paths)


def test_comparison_rejects_configuration_hash_mismatch(synthetic_root):
    gnn_paths, dg_paths = _synthetic_comparison_inputs(synthetic_root)
    payload = json.loads(gnn_paths[0].read_text(encoding="utf-8"))
    payload["configuration"]["ewma_alpha"] = 0.81
    gnn_paths[0].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(compare.ComparisonError, match="Configuration identity hash mismatch"):
        compare.collect_comparison(gnn_paths, dg_paths)


def test_comparison_rejects_validation_window_provenance_mismatch(synthetic_root):
    gnn_paths, dg_paths = _synthetic_comparison_inputs(synthetic_root, provenance_mismatch=True)
    with pytest.raises(compare.ComparisonError, match="window identities"):
        compare.collect_comparison(gnn_paths, dg_paths)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_comparison_rejects_nonfinite_ewma_metrics(synthetic_root, bad_value):
    gnn_paths, dg_paths = _synthetic_comparison_inputs(synthetic_root)
    payload = json.loads(dg_paths[0].read_text(encoding="utf-8"))
    payload["validation_metrics"]["ewma_alpha_0.8"]["mae"] = bad_value
    dg_paths[0].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(compare.ComparisonError, match="Invalid .*\\.mae"):
        compare.collect_comparison(gnn_paths, dg_paths)


def test_comparison_rejects_edge_order_mismatch(synthetic_root):
    gnn_paths, dg_paths = _synthetic_comparison_inputs(synthetic_root, edge_mismatch=True)
    with pytest.raises(compare.ComparisonError, match="ordering"):
        compare.collect_comparison(gnn_paths, dg_paths)


def test_comparison_rejects_normalization_mismatch(synthetic_root):
    gnn_paths, dg_paths = _synthetic_comparison_inputs(synthetic_root, normalization_mismatch=True)
    with pytest.raises(compare.ComparisonError, match="normalization differs"):
        compare.collect_comparison(gnn_paths, dg_paths)


def test_comparison_rejects_validation_coverage_mismatch(synthetic_root):
    gnn_paths, dg_paths = _synthetic_comparison_inputs(synthetic_root, coverage_mismatch=True)
    with pytest.raises(compare.ComparisonError, match="pair coverage"):
        compare.collect_comparison(gnn_paths, dg_paths)


def test_partial_comparison_labels_only_missing_original_topology_seed_42(synthetic_root):
    gnn_paths, dg_paths = _synthetic_comparison_inputs(synthetic_root)
    partial_paths = [
        path for path in gnn_paths
        if not (json.loads(path.read_text(encoding="utf-8")).get("adjacency_mode") == "topology"
                and json.loads(path.read_text(encoding="utf-8")).get("seed") == 42)
    ]
    with pytest.raises(compare.ComparisonError, match="exactly nine GNN"):
        compare.collect_comparison(partial_paths, dg_paths)

    result = compare.collect_comparison(partial_paths, dg_paths, allow_incomplete_gnn_matrix=True)
    assert result["status"] == "partial"
    assert result["gnn_matrix_complete"] is False
    assert result["architectural_comparison_claim"] is False
    assert result["missing_gnn_combinations"] == [
        {"model": "original_gnn", "adjacency_mode": "topology", "seed": 42}
    ]
    assert "topology" not in result["per_seed"][0]["gnn"]
    assert result["aggregate"]["gnn_topology"]["n"] == 2
    assert result["aggregate"]["gnn_identity"]["n"] == 3
    assert result["aggregate"]["gnn_shuffled"]["n"] == 3
    assert result["aggregate"]["dg_stmtl_based"]["n"] == 3
    assert result["aggregate"]["ewma"]["n"] == 3

    output = compare.write_comparison(result, synthetic_root / "partial-report")
    saved = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
    report = (output / "report.md").read_text(encoding="utf-8")
    assert saved["missing_gnn_combinations"] == result["missing_gnn_combinations"]
    assert saved["per_seed"][0]["ewma"]["mae"] == 0.03
    assert saved["per_seed"][0]["gnn"]["identity"]["checkpoint_sha256"] == "c" * 64
    assert "Not a full three-seed architectural comparison" in report
    assert "original GNN Topology seed 42 run is missing" in report
    csv_rows = (output / "comparison.csv").read_text(encoding="utf-8").splitlines()
    assert any(row.startswith("gnn_topology,2,") for row in csv_rows)


def test_partial_comparison_cli_requires_opt_in_and_reports_partial_status(synthetic_root, capsys):
    gnn_paths, dg_paths = _synthetic_comparison_inputs(synthetic_root)
    partial_paths = [
        path for path in gnn_paths
        if not (json.loads(path.read_text(encoding="utf-8")).get("adjacency_mode") == "topology"
                and json.loads(path.read_text(encoding="utf-8")).get("seed") == 42)
    ]
    common = []
    for path in partial_paths:
        common.extend(("--gnn-run-json", str(path)))
    for path in dg_paths:
        common.extend(("--dg-run-json", str(path)))
    output = synthetic_root / "cli-partial-report"
    with pytest.raises(compare.ComparisonError, match="exactly nine GNN"):
        compare.main([*common, "--output-directory", str(synthetic_root / "strict-out")])
    assert compare.main([*common, "--output-directory", str(output), "--allow-incomplete-gnn-matrix"]) == 0
    cli_result = json.loads(capsys.readouterr().out)
    assert cli_result["status"] == "partial"
    assert cli_result["missing_gnn_combinations"] == [
        {"model": "original_gnn", "adjacency_mode": "topology", "seed": 42}
    ]


def test_partial_comparison_rejects_duplicate_mode_seed(synthetic_root):
    gnn_paths, dg_paths = _synthetic_comparison_inputs(synthetic_root)
    partial = [
        path for path in gnn_paths
        if not (json.loads(path.read_text(encoding="utf-8")).get("adjacency_mode") == "topology"
                and json.loads(path.read_text(encoding="utf-8")).get("seed") == 42)
    ]
    # Keep eight arguments while replacing topology/123 with a duplicate identity/123 manifest.
    duplicate_identity = next(
        path for path in partial
        if json.loads(path.read_text(encoding="utf-8")).get("adjacency_mode") == "identity"
        and json.loads(path.read_text(encoding="utf-8")).get("seed") == 123
    )
    topology_123 = next(
        path for path in partial
        if json.loads(path.read_text(encoding="utf-8")).get("adjacency_mode") == "topology"
        and json.loads(path.read_text(encoding="utf-8")).get("seed") == 123
    )
    partial.remove(topology_123)
    partial.append(duplicate_identity)
    with pytest.raises(compare.ComparisonError, match="duplicate"):
        compare.collect_comparison(partial, dg_paths, allow_incomplete_gnn_matrix=True)


def test_partial_comparison_rejects_dataset_identity_mismatch(synthetic_root):
    gnn_paths, dg_paths = _synthetic_comparison_inputs(synthetic_root)
    partial = [
        path for path in gnn_paths
        if not (json.loads(path.read_text(encoding="utf-8")).get("adjacency_mode") == "topology"
                and json.loads(path.read_text(encoding="utf-8")).get("seed") == 42)
    ]
    payload = json.loads(partial[0].read_text(encoding="utf-8"))
    payload["dataset_metadata_sha256"] = "e" * 64
    partial[0].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(compare.ComparisonError, match="Dataset metadata"):
        compare.collect_comparison(partial, dg_paths, allow_incomplete_gnn_matrix=True)


def test_partial_comparison_rejects_a_different_missing_gnn_slot(synthetic_root):
    gnn_paths, dg_paths = _synthetic_comparison_inputs(synthetic_root)
    partial = list(gnn_paths)
    identity_123 = next(
        path for path in partial
        if json.loads(path.read_text(encoding="utf-8")).get("adjacency_mode") == "identity"
        and json.loads(path.read_text(encoding="utf-8")).get("seed") == 123
    )
    partial.remove(identity_123)
    with pytest.raises(compare.ComparisonError, match="only the missing original GNN Topology seed 42"):
        compare.collect_comparison(partial, dg_paths, allow_incomplete_gnn_matrix=True)


def test_comparison_rejects_missing_checkpoint_integrity(synthetic_root):
    gnn_paths, dg_paths = _synthetic_comparison_inputs(synthetic_root)
    payload = json.loads(gnn_paths[0].read_text(encoding="utf-8"))
    payload.pop("checkpoint_sha256")
    gnn_paths[0].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(compare.ComparisonError, match="checkpoint SHA-256"):
        compare.collect_comparison(gnn_paths, dg_paths)


def test_comparison_rejects_missing_dg_checkpoint_integrity(synthetic_root):
    gnn_paths, dg_paths = _synthetic_comparison_inputs(synthetic_root)
    payload = json.loads(dg_paths[0].read_text(encoding="utf-8"))
    payload.pop("checkpoint_sha256")
    dg_paths[0].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(compare.ComparisonError, match="checkpoint SHA-256"):
        compare.collect_comparison(gnn_paths, dg_paths)


def test_comparison_rejects_ewma_baseline_mismatch_across_methods(synthetic_root):
    gnn_paths, dg_paths = _synthetic_comparison_inputs(synthetic_root, alter_one_ewma=True)
    with pytest.raises(compare.ComparisonError, match="EWMA mae differs"):
        compare.collect_comparison(gnn_paths, dg_paths)


def test_synthetic_cli_check_does_not_authorize_training(capsys):
    result = experiment.main(["--self-check"])
    payload = json.loads(capsys.readouterr().out)
    assert result == 0
    assert payload["status"] == "pass" and payload["training"] is False
    with pytest.raises(SystemExit):
        experiment.main([])

