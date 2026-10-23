"""Synthetic-only recovery validation/lifecycle tests; no GEANT files are read."""

from __future__ import annotations

import hashlib
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest

from phase9_geant.colab_h96_v1 import matrix_recovery as recovery
from phase9_geant.colab_h96_v1 import launch_matrix_recovery as launcher
from phase9_geant.colab_h96_v1 import launch_matrix as matrix
from phase9_geant.colab_h96_v1 import launch as bootstrap
from phase9_geant.colab_h96_v1 import linux_resources


SCIENCE = {
    "protocol_id": "synthetic-h96-v2", "runner_id": "synthetic-runner",
    "history_length": 672, "horizons": [96], "prediction_channels": 1,
    "ewma_alpha": 0.8, "ewma_steps": 5,
    "residual_loss": "SmoothL1(beta=0.001) in original utilization-ratio units",
    "epochs": 20, "patience": 4, "learning_rate": 0.005, "batch_size": 4,
    "optimizer": "Adam", "hidden_size": 16, "seeds": [42, 123, 2026],
    "adjacency_modes": ["topology", "identity", "shuffled"], "shuffle_seed": 20261019,
    "scaler_fit_scope": "train only", "checkpoint_selection": "validation h96 loss",
    "validation_gradient_updates": False, "dataset_protocol_id": "synthetic-data",
    "train_bounds": [0, 8020], "validation_bounds": [8020, 9740],
    "memory_policy": {"policy_id": "synthetic-v3"},
}
SOURCES = {"src/gnn/model.py": "a" * 64, "phase9_geant/horizon96_streaming_experiment_v2.py": "b" * 64}
ADJACENCY_HASHES = {"topology": "d" * 64, "identity": "e" * 64, "shuffled": "f" * 64}


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_six_backups(root: Path):
    root.mkdir(parents=True)
    for seed, mode in recovery.COMPLETED_RUNS:
        run_dir = root / f"h96_{mode}_seed_{seed}"
        run_dir.mkdir()
        checkpoint = b"synthetic opaque checkpoint" + str(seed).encode()
        predictions = b"synthetic opaque validation prediction" + mode.encode()
        (run_dir / "best.pt").write_bytes(checkpoint)
        (run_dir / "validation_predictions.npy").write_bytes(predictions)
        run = {
            "status": "complete", "protocol_id": "synthetic-h96-v2", "seed": seed,
            "adjacency_mode": mode, "configuration": {
                **SCIENCE, "seed": seed, "adjacency_mode": mode, "device": "cpu",
                "matrix_experiment_id": "geant-h96-reproducibility-matrix-v3-drive-backed",
                "matrix_run_order": recovery.ORIGINAL_MATRIX_ORDER,
                "colab_bundle_identity": {
                    "source_snapshot_manifest_sha256": "1" * 64,
                    "source_bundle_sha256": "2" * 64,
                    "dataset_manifest_sha256": "3" * 64,
                    "dataset_bundle_sha256": "4" * 64,
                }, "colab_environment": {"python": "3.12"},
            }, "checkpoint": "best.pt", "checkpoint_sha256": _digest(run_dir / "best.pt"),
            "checkpoint_reload_strict": True,
            "validation_predictions": "validation_predictions.npy",
            "validation_predictions_sha256": _digest(run_dir / "validation_predictions.npy"),
            "test_access": {"permitted": False},
            "validation_metrics": {
                "h96_candidate": {"mae": 0.01, "rmse": 0.02, "evaluated_pairs": 953 * 72},
                "ewma_alpha_0.8": {"mae": 0.011, "rmse": 0.021, "evaluated_pairs": 953 * 72},
            }, "adjacency": {"mode": mode, "shuffle_seed": 20261019 if mode == "shuffled" else None,
                              "normalized_sha256": ADJACENCY_HASHES[mode]},
            "provenance": {"data_root": "/old/session/data", "source_scope": {"train": [0, 8020]}},
            "dataset_metadata_sha256": "c" * 64,
            "normalization": {"fit_scope": "h=96 train histories and targets only"},
            "source_hashes": SOURCES,
        }
        run_raw = json.dumps(run, sort_keys=True).encode()
        (run_dir / "run.json").write_bytes(run_raw)
        artifacts = {}
        for filename in ("run.json", "best.pt", "validation_predictions.npy"):
            file_path = run_dir / filename
            artifacts[filename] = {"sha256": _digest(file_path), "size_bytes": file_path.stat().st_size}
        backup = {
            "schema": "geant-h96-matrix3-run-backup-v1", "status": "complete",
            "seed": seed, "adjacency_mode": mode,
            "source_run_manifest_sha256": _digest(run_dir / "run.json"),
            "artifacts": artifacts, "readback_verified": True, "test_access": False,
        }
        (run_dir / "backup_manifest.json").write_text(json.dumps(backup), encoding="utf-8")


def _verify(root: Path, adjacency_hashes=ADJACENCY_HASHES):
    return recovery.verify_completed_backup_set(
        root, protocol_id="synthetic-h96-v2", expected_config=SCIENCE,
        current_critical_source_hashes=SOURCES,
        expected_dataset_metadata_sha256="c" * 64,
        expected_adjacency_hashes=adjacency_hashes,
        expected_provenance={"data_root": "/new/session/data", "source_scope": {"train": [0, 8020]}},
    )


def test_exact_six_backups_verify_protocol_hashes_and_ephemeral_root(tmp_path):
    root = tmp_path / "drive" / "runs"
    _make_six_backups(root)
    result = _verify(root)
    assert result["status"] == "six_completed_backups_verified"
    assert len(result["completed_runs"]) == 6
    assert result["remaining_runs"] == [
        {"seed": 123, "adjacency_mode": "shuffled"},
        {"seed": 2026, "adjacency_mode": "shuffled"},
    ]
    assert result["test_artifacts_opened"] is False


@pytest.mark.parametrize("mutation", ["checkpoint", "prediction", "backup_manifest", "protocol", "source", "test_access"])
def test_backup_integrity_and_protocol_mismatches_fail_closed(tmp_path, mutation):
    root = tmp_path / "runs"
    _make_six_backups(root)
    target = root / "h96_topology_seed_123"
    if mutation == "checkpoint":
        (target / "best.pt").write_bytes(b"tampered")
    elif mutation == "prediction":
        (target / "validation_predictions.npy").write_bytes(b"tampered")
    elif mutation == "backup_manifest":
        backup_path = target / "backup_manifest.json"
        backup = json.loads(backup_path.read_text(encoding="utf-8"))
        backup["readback_verified"] = False
        backup_path.write_text(json.dumps(backup), encoding="utf-8")
    else:
        run_path = target / "run.json"
        run = json.loads(run_path.read_text(encoding="utf-8"))
        if mutation == "protocol":
            run["configuration"]["batch_size"] = 32
        elif mutation == "source":
            run["source_hashes"]["src/gnn/model.py"] = "f" * 64
        else:
            run["test_access"]["permitted"] = True
        run_path.write_text(json.dumps(run), encoding="utf-8")
    with pytest.raises(recovery.RecoveryError):
        _verify(root)


def test_missing_or_extra_backup_refuses_preflight(tmp_path):
    root = tmp_path / "runs"
    _make_six_backups(root)
    (root / "h96_identity_seed_42").rename(root / "missing")
    with pytest.raises(recovery.RecoveryError, match="exactly the six"):
        _verify(root)


def test_reconstructed_ordered_adjacency_must_match_each_recorded_run(tmp_path):
    root = tmp_path / "runs"
    _make_six_backups(root)
    wrong = {**ADJACENCY_HASHES, "topology": "9" * 64}
    with pytest.raises(recovery.RecoveryError, match="ordered-link graph"):
        _verify(root, wrong)


def _bootstrap_fixture(tmp_path: Path, *, source=b"BOOTSTRAP_VALUE = 7\n"):
    repository = tmp_path / "snapshot"
    bootstrap_path = repository / "phase9_geant" / "colab_h96_v1" / "launch.py"
    bootstrap_path.parent.mkdir(parents=True)
    bootstrap_path.write_bytes(source)
    archive_path = tmp_path / "source.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("phase9_geant/colab_h96_v1/launch.py", source)
    archive_digest = _digest(archive_path)
    manifest = {
        "archive_sha256": archive_digest,
        "entries": [{"path": "phase9_geant/colab_h96_v1/launch.py",
                     "sha256": hashlib.sha256(source).hexdigest(), "size_bytes": len(source)}],
    }
    manifest_path = tmp_path / "source.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    args = argparse.Namespace(
        source_bundle=archive_path, source_manifest=manifest_path,
        expected_source_bundle_sha256=archive_digest,
        expected_source_manifest_sha256=_digest(manifest_path),
    )
    return repository, bootstrap_path, args


def test_recovery_bootstrap_executes_only_manifest_bound_source_bytes(tmp_path):
    repository, _bootstrap_path, args = _bootstrap_fixture(tmp_path)
    module = launcher._load_verified_bootstrap(repository, args)
    assert module.BOOTSTRAP_VALUE == 7


def test_recovery_bootstrap_rejects_source_change_before_exec(tmp_path):
    repository, bootstrap_path, args = _bootstrap_fixture(tmp_path)
    bootstrap_path.write_text("raise AssertionError('must not execute')\n", encoding="utf-8")
    with pytest.raises(launcher.RecoveryBootstrapError, match="differs from approved manifest"):
        launcher._load_verified_bootstrap(repository, args)


def test_recovery_preflight_reads_nested_policy_then_binds_before_data_access():
    # Reproduces the observed shape: preflight's outer result has no policy_id;
    # the approved identity and fixed limits are in its nested "policy" record.
    policy = {
        "policy_id": linux_resources.RESOURCE_POLICY_ID,
        "process_peak_limit_bytes": 2 * 1024**3,
        "startup_headroom_bytes": 2 * 1024**3 + 384 * 1024**2,
        "in_run_os_available_reserve_bytes": 384 * 1024**2,
        "in_run_cgroup_headroom_reserve_bytes": 384 * 1024**2,
        "minimum_free_disk_bytes": 512 * 1024**2,
        "process_peak_measurement": "synthetic-test-measurement",
        "visible_unlimited_cgroup_policy": "allow literal unlimited visible limits only; host and process guards remain mandatory",
        "cgroup_visibility_limitation": "synthetic fixture",
        "invalid_or_missing_cgroup_policy": "fail closed before output creation or training",
    }
    trainer = SimpleNamespace(
        CONFIG={"memory_policy": {"workers": 0}},
        lazy=SimpleNamespace(SMOKE_ESTIMATED_PEAK_RAM=671088640),
        PROCESS_PEAK_LIMIT_BYTES=671088640,
        SYSTEM_AVAILABLE_RESERVE_BYTES=384 * 1024**2,
    )

    class SyntheticResources:
        def preflight(self, *, disk_path):
            assert Path(disk_path).name == "synthetic-repository"
            return {"status": "pass", "snapshot": {"synthetic": True}, "policy": policy}

    # The trainer does not have RESOURCE_POLICY_ID before the binder runs. The
    # authoritative postcondition is its actual limit/configuration binding.
    assert not hasattr(trainer, "RESOURCE_POLICY_ID")
    result = launcher._activate_resource_policy(
        bootstrap, SyntheticResources(), trainer, Path("synthetic-repository")
    )
    assert "policy_id" not in result
    assert trainer.PROCESS_PEAK_LIMIT_BYTES == 2 * 1024**3
    assert trainer.lazy.SMOKE_ESTIMATED_PEAK_RAM == 2 * 1024**3
    assert trainer.SYSTEM_AVAILABLE_RESERVE_BYTES == 384 * 1024**2
    assert trainer.CONFIG["linux_resource_policy"] == policy


def test_recovery_preflight_rejects_wrong_nested_policy_before_binding():
    trainer = SimpleNamespace(
        CONFIG={}, lazy=SimpleNamespace(SMOKE_ESTIMATED_PEAK_RAM=671088640),
        PROCESS_PEAK_LIMIT_BYTES=671088640,
        SYSTEM_AVAILABLE_RESERVE_BYTES=384 * 1024**2,
    )

    class WrongPolicyResources:
        def preflight(self, *, disk_path):
            return {"status": "pass", "policy": {"policy_id": "wrong"}}

    with pytest.raises(launcher.RecoveryBootstrapError, match="policy is not active"):
        launcher._activate_resource_policy(
            bootstrap, WrongPolicyResources(), trainer, Path("synthetic-repository")
        )
    assert trainer.PROCESS_PEAK_LIMIT_BYTES == 671088640
    assert "linux_resource_policy" not in trainer.CONFIG


def test_recovery_sources_are_bound_before_first_dataset_verification():
    expected_sources = {
        "phase9_geant/colab_h96_v1/launch_matrix_recovery.py",
        "phase9_geant/colab_h96_v1/matrix_recovery.py",
        "phase9_geant/colab_h96_v1/matrix_workflow.py",
    }
    trainer = SimpleNamespace(RUN_SOURCE_FILES=("base_runner.py",))
    package = SimpleNamespace(MIGRATION_RUNTIME_FILES=(
        "phase9_geant/colab_h96_v1/matrix_workflow.py",
    ))
    marker = ("synthetic verified train/validation inputs",)

    def verify_inputs(data_root, repository):
        assert Path(data_root).name == "synthetic-data"
        assert Path(repository).name == "synthetic-repo"
        assert expected_sources.issubset(set(trainer.RUN_SOURCE_FILES))
        return marker

    trainer._verified_inputs = verify_inputs
    assert launcher._verified_recovery_inputs(
        trainer, package, Path("synthetic-data"), Path("synthetic-repo")
    ) == marker


def test_changed_train_validation_fingerprint_blocks_run_before_training(monkeypatch, tmp_path):
    # Synthetic identities model a genuine byte-level change in one permitted
    # train array. No GEANT files or arrays are opened by this test.
    names = ("timestamps_train.npy", "utilization_train.npy", "valid_train.npy",
             "timestamps_validation.npy", "utilization_validation.npy", "valid_validation.npy",
             "window_index_train_h96.json", "window_index_validation_h96.json")

    original_bytes = b"synthetic utilization train payload, approved identity"
    changed_bytes = b"synthetic utilization train payload, changed identity"
    original_hash = hashlib.sha256(original_bytes).hexdigest()
    changed_hash = hashlib.sha256(changed_bytes).hexdigest()
    assert original_hash != changed_hash

    def inputs(utilization_train_hash):
        records = {
            name: {"sha256": hashlib.sha256(("synthetic:" + name).encode()).hexdigest()}
            for name in names
        }
        records["utilization_train.npy"] = {"sha256": utilization_train_hash}
        metadata = {"array_files": records, "protocol_id": "synthetic-h96"}
        splits = {"train": {"windows": ["train-window"]},
                  "validation": {"windows": ["validation-window"]}}
        provenance = {"array_file_records": records, "source_scope": {"train": [0, 8020]}}
        return ("repo", "data-root", splits, metadata, ((0, 1),), provenance, {"source.py": "b" * 64})

    expected_inputs = inputs(original_hash)
    changed_inputs = inputs(changed_hash)
    assert not matrix._same_input_identity(changed_inputs, expected_inputs)

    policy = {"policy_id": "synthetic-policy"}
    events = []

    class Resources:
        def preflight(self, *, disk_path):
            return {"status": "pass", "policy": policy}

    class Workflow:
        def persist_lifecycle_snapshot(self, *args, **kwargs):
            events.append((kwargs.get("state"), kwargs.get("record", {}).get("status")))
            return {"path": "synthetic", "sha256": "0" * 64}

    trainer = SimpleNamespace(
        PROTOCOL_ID="synthetic-h96", CONFIG={},
        _verified_inputs=lambda *_: changed_inputs,
        _source_identity=lambda *_: pytest.fail("source checks follow data-identity validation"),
        _planned_run=lambda *_: pytest.fail("run planning follows data-identity validation"),
        train_one=lambda *_args, **_kwargs: pytest.fail("changed dataset must never train"),
    )
    prepared = {
        "matrix": matrix,
        "recovery": recovery,
        "trainer": trainer,
        "repository": tmp_path / "repository",
        "output_root": tmp_path / "local" / "recovery",
        "persistent_root": tmp_path / "drive" / "recovery",
        "source_hashes": {"source.py": "b" * 64},
        "policy": policy,
        "resources": Resources(),
        "verified_inputs": expected_inputs,
        "data_root": tmp_path / "synthetic-data",
        "prior": {"status": "six_completed_backups_verified", "completed_runs": []},
        "workflow": Workflow(),
    }
    monkeypatch.setattr(launcher, "_authorization", lambda *_: True)
    with pytest.raises(recovery.RecoveryError, match="data identity changed before this run"):
        launcher._execute(prepared)
    assert events[-1] == ("failed", "failed")
    assert not (prepared["output_root"] / "h96_shuffled_seed_123").exists()


def _synthetic_run(seed: int, mode: str, destination: Path):
    destination.mkdir()
    for name, body in (("run.json", b"run"), ("best.pt", b"checkpoint"),
                       ("validation_predictions.npy", b"predictions")):
        (destination / name).write_bytes(body)
    return {"seed": seed, "adjacency_mode": mode, "status": "complete",
            "epochs_completed": 2, "best_epoch": 1,
            "validation_metrics": {"h96_candidate": {"mae": 0.1, "rmse": 0.2}},
            "runtime_seconds": 3.0}


def _synthetic_verify(result, seed, mode, destination):
    assert result["seed"] == seed and result["adjacency_mode"] == mode
    return {"run_manifest_sha256": _digest(destination / "run.json"),
            "checkpoint_sha256": _digest(destination / "best.pt"),
            "validation_predictions_sha256": _digest(destination / "validation_predictions.npy")}


def _synthetic_backup(result, seed, mode, destination, backup_root):
    ids = _synthetic_verify(result, seed, mode, destination)
    return {"verified": True, "artifacts": {
        "run.json": {"sha256": ids["run_manifest_sha256"]},
        "best.pt": {"sha256": ids["checkpoint_sha256"]},
        "validation_predictions.npy": {"sha256": ids["validation_predictions_sha256"]},
    }, "directory": str(backup_root / f"h96_{mode}_seed_{seed}")}


def test_authorization_backup_order_and_combined_summary(tmp_path):
    prior = {"status": "six_completed_backups_verified", "completed_runs": []}
    output = tmp_path / "o"
    persistent = tmp_path / "drive" / "fresh_recovery"
    persistent.parent.mkdir()
    events = []

    def authorize(index, seed, mode):
        if index == 1:
            assert events[-1] == ("snapshot", 0, "incomplete")
        else:
            assert "backup" in events and events.index("backup") < len(events) - 1
        events.append(("auth", seed, mode))
        return True

    def train(seed, mode, destination):
        assert events[-2][0] == "auth" and events[-1] == ("snapshot", 1 if seed == 123 else 2, "in_progress")
        events.append(("train", seed, mode))
        return _synthetic_run(seed, mode, destination)

    def verify(result, seed, mode, destination):
        events.append(("verify", seed, mode))
        return _synthetic_verify(result, seed, mode, destination)

    def backup(result, seed, mode, destination):
        ids = _synthetic_verify(result, seed, mode, destination)
        events.append("backup")
        return _synthetic_backup(result, seed, mode, destination, persistent / "runs")

    def persist(sequence, state, record):
        events.append(("snapshot", sequence, state))
        return {"path": state, "sha256": "0" * 64}

    events.append("ready")
    result = recovery.run_remaining_experiments(
        output_root=output, persistent_root=persistent, prior_verified=prior,
        authorize=authorize, train_one=train, verify_result=verify, backup_run=backup,
        persist_snapshot=persist,
    )
    assert result["status"] == "complete"
    assert [x for x in events if isinstance(x, tuple) and x[0] == "auth"] == [
        ("auth", 123, "shuffled"), ("auth", 2026, "shuffled")]
    summary = json.loads((persistent / "combined_results.json").read_text(encoding="utf-8"))
    assert len(summary["runs"]) == 2
    assert summary["test_access"] is False


def test_backup_failure_blocks_second_run_and_existing_destinations_are_refused(tmp_path):
    output = tmp_path / "out"
    persistent = tmp_path / "drive" / "new"
    persistent.parent.mkdir()
    calls = []
    with pytest.raises(recovery.RecoveryError, match="backup failed"):
        recovery.run_remaining_experiments(
            output_root=output, persistent_root=persistent,
            prior_verified={"status": "six_completed_backups_verified", "completed_runs": []},
            authorize=lambda *_: True,
            train_one=lambda seed, mode, dest: _synthetic_run(seed, mode, dest),
            verify_result=_synthetic_verify,
            backup_run=lambda *_: calls.append("backup") or (_ for _ in ()).throw(recovery.RecoveryError("backup failed")),
            persist_snapshot=lambda *_: {"path": "synthetic", "sha256": "0" * 64},
        )
    assert calls == ["backup"]
    assert {p.name for p in output.iterdir()} == {"h96_shuffled_seed_123"}
    with pytest.raises(FileExistsError, match="existing recovery output"):
        recovery.run_remaining_experiments(
            output_root=output, persistent_root=tmp_path / "drive" / "other",
            prior_verified={"status": "six_completed_backups_verified", "completed_runs": []},
            authorize=lambda *_: pytest.fail("must refuse before authorization"),
            train_one=lambda *_: pytest.fail("must not train"), verify_result=lambda *_: {},
            backup_run=lambda *_: {}, persist_snapshot=lambda *_: {},
        )


def test_interrupted_run_is_failed_and_never_resumed(tmp_path):
    output = tmp_path / "out"
    persistent = tmp_path / "drive" / "fresh"
    persistent.parent.mkdir()
    states = []

    def persist(sequence, state, record):
        states.append((sequence, state, record["status"]))
        return {"path": state, "sha256": "0" * 64}

    def interrupt(*_args):
        raise KeyboardInterrupt("synthetic interruption")

    with pytest.raises(KeyboardInterrupt, match="synthetic interruption"):
        recovery.run_remaining_experiments(
            output_root=output, persistent_root=persistent,
            prior_verified={"status": "six_completed_backups_verified", "completed_runs": []},
            authorize=lambda *_: True, train_one=interrupt,
            verify_result=lambda *_: {}, backup_run=lambda *_: {}, persist_snapshot=persist,
        )
    assert states[-1] == (1, "failed", "failed")
    with pytest.raises(FileExistsError, match="existing recovery output"):
        recovery.run_remaining_experiments(
            output_root=output, persistent_root=tmp_path / "drive" / "other",
            prior_verified={"status": "six_completed_backups_verified", "completed_runs": []},
            authorize=lambda *_: pytest.fail("interrupted attempt must not be resumed"),
            train_one=lambda *_: {}, verify_result=lambda *_: {}, backup_run=lambda *_: {},
            persist_snapshot=lambda *_: {},
        )

