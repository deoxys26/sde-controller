"""Synthetic-only tests for the separate h96 reproducibility launcher."""

from __future__ import annotations

import importlib.util
import hashlib
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from phase9_geant.colab_h96_v1 import matrix_workflow


MODULE_PATH = Path(__file__).parents[1] / "launch_matrix.py"
SPEC = importlib.util.spec_from_file_location("launch_matrix_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
matrix = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(matrix)
matrix.matrix_workflow = matrix_workflow
SYNTHETIC_CONFIG = {"fixed_synthetic_setting": True}


def _allow_run(_index, _seed, _mode):
    return True


def _fake_artifact_verifier(*_args):
    return {"run_manifest_sha256": "d" * 64, "checkpoint_sha256": "c" * 64,
            "validation_predictions_sha256": "e" * 64}


def _fake_backup(*, artifact_identity, **_kwargs):
    return {
        "verified": True,
        "directory": "/synthetic-drive/run",
        "backup_manifest_sha256": "f" * 64,
        "artifacts": {
            "run.json": {"sha256": artifact_identity["run_manifest_sha256"]},
            "best.pt": {"sha256": artifact_identity["checkpoint_sha256"]},
            "validation_predictions.npy": {"sha256": artifact_identity["validation_predictions_sha256"]},
        },
    }


def _fake_inputs(repository: Path, data_root: Path):
    return (
        repository,
        data_root,
        {"train": {"windows": [1, 2]}, "validation": {"windows": [3]}},
        {"array_fingerprints": {"X_train.npy": "a" * 64}},
        ((0, 1), (1, 0)),
        {"provenance_id": "synthetic"},
        {"source.py": "b" * 64},
    )


def test_run_order_is_the_eight_missing_seed_mode_pairs():
    assert matrix.RUN_ORDER == (
        (123, "topology"), (2026, "topology"),
        (42, "identity"), (123, "identity"), (2026, "identity"),
        (42, "shuffled"), (123, "shuffled"), (2026, "shuffled"),
    )
    assert len(set(matrix.RUN_ORDER)) == 8


def _valid_result(seed=123, mode="topology"):
    return {
        "status": "complete", "seed": seed, "adjacency_mode": mode,
        "protocol_id": "synthetic-protocol", "checkpoint": "best.pt",
        "checkpoint_reload_strict": True, "checkpoint_sha256": "c" * 64,
        "validation_predictions_sha256": "e" * 64,
        "configuration": {**SYNTHETIC_CONFIG, "seed": seed, "adjacency_mode": mode, "device": "cpu"},
        "test_access": {"permitted": False},
        "validation_metrics": {
            "h96_candidate": {"mae": 0.1, "rmse": 0.2, "evaluated_pairs": 953 * 72},
            "ewma_alpha_0.8": {"mae": 0.11, "rmse": 0.21, "evaluated_pairs": 953 * 72},
        },
    }


def test_completed_run_identity_and_checkpoint_integrity_are_mandatory():
    matrix._validate_run_result(_valid_result(), 123, "topology", "synthetic-protocol")
    with pytest.raises(RuntimeError, match="identity/finalization"):
        matrix._validate_run_result(_valid_result(seed=42), 123, "topology", "synthetic-protocol")
    bad = _valid_result()
    bad["checkpoint_sha256"] = "not-a-digest"
    with pytest.raises(RuntimeError, match="checkpoint SHA-256"):
        matrix._validate_run_result(bad, 123, "topology", "synthetic-protocol")


def test_run_result_rejects_test_access_and_unpaired_metrics():
    bad = _valid_result()
    bad["test_access"] = {"permitted": True}
    with pytest.raises(RuntimeError, match="test-split denial"):
        matrix._validate_run_result(bad, 123, "topology", "synthetic-protocol")
    bad = _valid_result()
    bad["validation_metrics"].pop("ewma_alpha_0.8")
    with pytest.raises(RuntimeError, match="paired candidate/EWMA"):
        matrix._validate_run_result(bad, 123, "topology", "synthetic-protocol")


def test_namespace_refuses_preexisting_root_and_each_run_destination(tmp_path):
    repository = tmp_path / "repo"
    processed = repository / "data" / "processed"
    processed.mkdir(parents=True)
    bootstrap = SimpleNamespace(_canonical_no_redirect=lambda p, require_exists=True: p)
    root, destinations, control_path = matrix._check_namespace(repository, bootstrap)
    assert root.parent == processed and len(destinations) == 8
    assert control_path == processed / matrix.MATRIX_CONTROL_MANIFEST
    root.mkdir()
    with pytest.raises(FileExistsError, match="existing matrix namespace"):
        matrix._check_namespace(repository, bootstrap)

    fresh_repository = tmp_path / "fresh_repo"
    fresh_processed = fresh_repository / "data" / "processed"
    fresh_processed.mkdir(parents=True)
    fresh_control = fresh_processed / matrix.MATRIX_CONTROL_MANIFEST
    fresh_control.write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="existing matrix control manifest"):
        matrix._check_namespace(fresh_repository, bootstrap)


def test_matrix_record_rejects_wrong_type_and_namespace_paths(tmp_path):
    processed = tmp_path / "repo" / "data" / "processed"
    processed.mkdir(parents=True)
    output_root = processed / matrix.MATRIX_OUTPUT_ROOT
    control_path = processed / matrix.MATRIX_CONTROL_MANIFEST
    record = {
        "matrix_id": matrix.MATRIX_ID,
        "matrix_output_root": matrix.MATRIX_OUTPUT_ROOT,
        "output_root": str(output_root),
        "matrix_manifest": str(control_path),
        "run_order": matrix._expected_run_order(),
        "status": "incomplete",
        "failure": None,
        "completed_runs": [],
    }
    with pytest.raises(RuntimeError, match="must be a JSON object"):
        matrix._validate_matrix_record([], output_root=output_root, control_path=control_path)
    altered = {**record, "output_root": str(processed / "elsewhere")}
    with pytest.raises(RuntimeError, match="identity/path mismatch"):
        matrix._validate_matrix_record(altered, output_root=output_root, control_path=control_path)
    with pytest.raises(RuntimeError, match="canonical namespace-bound sidecar"):
        matrix._persist_matrix_record(
            processed / "wrong.matrix_run.json", output_root=output_root, record=record,
        )


def test_run_sequence_is_sequential_and_uses_each_destination_once(tmp_path):
    repository = tmp_path / "repo"
    repository.mkdir()
    data_root = tmp_path / "data"
    data_root.mkdir()
    output_root = tmp_path / "processed" / matrix.MATRIX_OUTPUT_ROOT
    expected = _fake_inputs(repository, data_root)
    source_hashes = {"source.py": "b" * 64}
    policy = {"policy_id": "synthetic-v3"}
    calls = []
    authorizations = []
    registrations = []
    verified_artifacts = []

    class Trainer:
        @staticmethod
        def _source_identity(_repository):
            return source_hashes

        CONFIG = SYNTHETIC_CONFIG

        @staticmethod
        def _planned_run(_repository, seed, mode):
            return output_root, output_root / f"h96_{mode}_seed_{seed}"

        @staticmethod
        def train_one(_data_root, **kwargs):
            calls.append((kwargs["seed"], kwargs["mode"], kwargs["run_training"], kwargs["explicit_authorization"]))
            return {**_valid_result(kwargs["seed"], kwargs["mode"]), "best_epoch": 2}

    class Resource:
        @staticmethod
        def preflight(*, disk_path):
            assert disk_path == repository
            return {"policy": policy}

        @staticmethod
        def LinuxMemoryMonitor():
            return object()

    results = matrix._run_sequentially(
        trainer=Trainer(), resource_module=Resource(), data_root=data_root,
        repository=repository, output_root=output_root, expected_inputs=expected,
        expected_source_hashes=source_hashes, expected_resource_policy=policy,
        protocol_id="synthetic-protocol",
        preflight=lambda _data, _repo: expected,
        authorize_run=lambda index, seed, mode: authorizations.append((index, seed, mode)) or True,
        record_attempt=lambda *_args: None,
        artifact_verifier=lambda result, destination, seed, mode, config, sources:
            verified_artifacts.append((seed, mode, destination)) or _fake_artifact_verifier(result, destination, seed, mode, config, sources),
        backup_run=_fake_backup,
        register_result=registrations.append,
    )
    assert len(results) == len(calls) == 8
    assert [(x[0], x[1]) for x in calls] == list(matrix.RUN_ORDER)
    assert all(x[2:] == (True, True) for x in calls)
    assert [(x[1], x[2]) for x in authorizations] == list(matrix.RUN_ORDER)
    assert len(verified_artifacts) == len(registrations) == 8
    assert registrations == results


def test_first_failed_run_stops_matrix_without_retry_or_later_runs(tmp_path):
    repository = tmp_path / "repo"
    repository.mkdir()
    data_root = tmp_path / "data"
    data_root.mkdir()
    output_root = tmp_path / "processed" / matrix.MATRIX_OUTPUT_ROOT
    expected = _fake_inputs(repository, data_root)
    source_hashes = {"source.py": "b" * 64}
    calls = []

    class Trainer:
        _source_identity = staticmethod(lambda _repo: source_hashes)
        CONFIG = SYNTHETIC_CONFIG
        _planned_run = staticmethod(lambda _repo, seed, mode: (output_root, output_root / f"h96_{mode}_seed_{seed}"))

    class Resource:
        preflight = staticmethod(lambda **_: {"policy": {"id": 1}})
        LinuxMemoryMonitor = staticmethod(object)

    def fail_once(_data, **kwargs):
        calls.append((kwargs["seed"], kwargs["mode"]))
        return {"status": "failed"}

    with pytest.raises(RuntimeError, match="preserve it and do not retry"):
        matrix._run_sequentially(
            trainer=Trainer(), resource_module=Resource(), data_root=data_root,
            repository=repository, output_root=output_root, expected_inputs=expected,
            expected_source_hashes=source_hashes, expected_resource_policy={"id": 1},
            protocol_id="synthetic-protocol",
            preflight=lambda _data, _repo: expected, run_one=fail_once,
            authorize_run=_allow_run, artifact_verifier=_fake_artifact_verifier,
            record_attempt=lambda *_args: None,
            backup_run=_fake_backup,
        )
    assert calls == [matrix.RUN_ORDER[0]]


def test_persistent_backup_failure_stops_before_registration_and_next_run(tmp_path):
    repository = tmp_path / "repo"
    repository.mkdir()
    data_root = tmp_path / "data"
    data_root.mkdir()
    output_root = tmp_path / "processed" / matrix.MATRIX_OUTPUT_ROOT
    expected = _fake_inputs(repository, data_root)
    source_hashes = {"source.py": "b" * 64}
    calls, registrations = [], []

    class Trainer:
        _source_identity = staticmethod(lambda _repo: source_hashes)
        CONFIG = SYNTHETIC_CONFIG
        _planned_run = staticmethod(lambda _repo, seed, mode: (
            output_root, output_root / f"h96_{mode}_seed_{seed}",
        ))

    class Resource:
        preflight = staticmethod(lambda **_: {"policy": {"p": 1}})
        LinuxMemoryMonitor = staticmethod(object)

    def train(_data, **kwargs):
        calls.append((kwargs["seed"], kwargs["mode"]))
        return {**_valid_result(kwargs["seed"], kwargs["mode"]), "best_epoch": 1}

    with pytest.raises(RuntimeError, match="Drive read-back failed"):
        matrix._run_sequentially(
            trainer=Trainer(), resource_module=Resource(), data_root=data_root,
            repository=repository, output_root=output_root, expected_inputs=expected,
            expected_source_hashes=source_hashes, expected_resource_policy={"p": 1},
            protocol_id="synthetic-protocol", preflight=lambda *_: expected,
            run_one=train, authorize_run=_allow_run,
            record_attempt=lambda *_args: None,
            artifact_verifier=_fake_artifact_verifier,
            backup_run=lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("Drive read-back failed")),
            register_result=registrations.append,
        )
    assert calls == [matrix.RUN_ORDER[0]]
    assert registrations == []


def test_source_or_train_validation_provenance_drift_stops_before_run(tmp_path):
    repository = tmp_path / "repo"
    repository.mkdir()
    data_root = tmp_path / "data"
    data_root.mkdir()
    output_root = tmp_path / "processed" / matrix.MATRIX_OUTPUT_ROOT
    expected = _fake_inputs(repository, data_root)
    runs = []

    class Trainer:
        _source_identity = staticmethod(lambda _repo: {"source.py": "changed"})
        _planned_run = staticmethod(lambda *_: pytest.fail("destination should not be planned after source drift"))

    class Resource:
        preflight = staticmethod(lambda **_: {"policy": {"id": 1}})
        LinuxMemoryMonitor = staticmethod(object)

    with pytest.raises(RuntimeError, match="Source identity changed"):
        matrix._run_sequentially(
            trainer=Trainer(), resource_module=Resource(), data_root=data_root,
            repository=repository, output_root=output_root, expected_inputs=expected,
            expected_source_hashes={"source.py": "expected"}, expected_resource_policy={"id": 1},
            protocol_id="synthetic-protocol",
            preflight=lambda _data, _repo: expected,
            run_one=lambda *_a, **_k: runs.append(True),
            authorize_run=_allow_run, artifact_verifier=_fake_artifact_verifier,
            record_attempt=lambda *_args: None,
            backup_run=_fake_backup,
        )
    assert runs == []

    changed = list(expected)
    changed[5] = {"provenance_id": "different"}
    with pytest.raises(RuntimeError, match="provenance or array identity changed"):
        matrix._run_sequentially(
            trainer=SimpleNamespace(
                _source_identity=lambda _repo: {"source.py": "expected"},
                _planned_run=lambda *_: pytest.fail("destination should not be planned after provenance drift"),
            ),
            resource_module=Resource(), data_root=data_root, repository=repository,
            output_root=output_root, expected_inputs=expected,
            expected_source_hashes={"source.py": "expected"}, expected_resource_policy={"id": 1},
            protocol_id="synthetic-protocol",
            preflight=lambda _data, _repo: tuple(changed),
            run_one=lambda *_a, **_k: runs.append(True),
            authorize_run=_allow_run, artifact_verifier=_fake_artifact_verifier,
            record_attempt=lambda *_args: None,
            backup_run=_fake_backup,
        )
    assert runs == []


def test_changed_policy_stops_before_any_training_call(tmp_path):
    repository = tmp_path / "repo"
    repository.mkdir()
    data_root = tmp_path / "data"
    data_root.mkdir()
    expected = _fake_inputs(repository, data_root)
    calls = []

    class Trainer:
        _source_identity = staticmethod(lambda _repo: {"source.py": "ok"})

    class Resource:
        preflight = staticmethod(lambda **_: {"policy": {"policy_id": "changed"}})

    with pytest.raises(RuntimeError, match="resource policy changed"):
        matrix._run_sequentially(
            trainer=Trainer(), resource_module=Resource(), data_root=data_root,
            repository=repository, output_root=tmp_path / "root",
            expected_inputs=expected, expected_source_hashes={"source.py": "ok"},
            expected_resource_policy={"policy_id": "v3"},
            protocol_id="synthetic-protocol",
            preflight=lambda _data, _repo: expected,
            run_one=lambda *_a, **_k: calls.append(True),
            authorize_run=_allow_run, artifact_verifier=_fake_artifact_verifier,
            record_attempt=lambda *_args: None,
            backup_run=_fake_backup,
        )
    assert calls == []


def test_each_run_requires_its_exact_separate_authorization(monkeypatch):
    answers = iter([matrix._authorization_phrase(i, seed, mode) for i, (seed, mode) in enumerate(matrix.RUN_ORDER, 1)])
    monkeypatch.setattr(matrix.sys, "stdin", SimpleNamespace(readline=lambda: next(answers) + "\n"))
    for index, (seed, mode) in enumerate(matrix.RUN_ORDER, 1):
        assert matrix._request_run_authorization(index, seed, mode) is True
    monkeypatch.setattr(matrix.sys, "stdin", SimpleNamespace(readline=lambda: "wrong\n"))
    with pytest.raises(PermissionError, match="Authorization declined"):
        matrix._request_run_authorization(1, *matrix.RUN_ORDER[0])


def test_missing_authorization_callback_fails_before_any_run(tmp_path):
    repository = tmp_path / "repo"
    repository.mkdir()
    data_root = tmp_path / "data"
    data_root.mkdir()
    calls = []
    trainer = SimpleNamespace(
        CONFIG=SYNTHETIC_CONFIG,
        _source_identity=lambda _repo: {"source.py": "ok"},
        _planned_run=lambda *_args: calls.append(True),
    )
    resource = SimpleNamespace(preflight=lambda **_: {"policy": {"p": 1}})
    with pytest.raises(PermissionError, match="authorization callback"):
        matrix._run_sequentially(
            trainer=trainer, resource_module=resource, data_root=data_root,
            repository=repository, output_root=tmp_path / "root",
            expected_inputs=_fake_inputs(repository, data_root), expected_source_hashes={"source.py": "ok"},
            expected_resource_policy={"p": 1}, protocol_id="synthetic-protocol",
            preflight=lambda *_: _fake_inputs(repository, data_root),
        )
    assert calls == []


def test_launcher_has_no_test_array_or_split_loader_path():
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "X_test.npy" not in source and "y_test.npy" not in source
    assert "np.load" not in source and "torch.load" not in source
    assert "AUTH_REQUIRED:" in source and "_request_run_authorization" in source


def test_run_artifact_validation_hashes_opaque_files_and_rejects_mismatch(tmp_path):
    destination = tmp_path / "h96_topology_seed_123"
    destination.mkdir()
    checkpoint_bytes = b"synthetic checkpoint bytes"
    prediction_bytes = b"synthetic prediction bytes; never deserialized"
    (destination / "best.pt").write_bytes(checkpoint_bytes)
    (destination / "validation_predictions.npy").write_bytes(prediction_bytes)
    configuration = {**SYNTHETIC_CONFIG, "seed": 123, "adjacency_mode": "topology", "device": "cpu"}
    source_hashes = {"trainer.py": "b" * 64}
    result = {
        **_valid_result(123, "topology"),
        "source_hashes": source_hashes,
        "validation_predictions": "validation_predictions.npy",
        "checkpoint_sha256": hashlib.sha256(checkpoint_bytes).hexdigest(),
        "validation_predictions_sha256": hashlib.sha256(prediction_bytes).hexdigest(),
        "configuration": configuration,
    }
    manifest_path = destination / "run.json"
    manifest_path.write_text(json.dumps(result), encoding="utf-8")
    identities = matrix._validate_run_artifacts(
        result, destination, 123, "topology", configuration, source_hashes,
    )
    assert identities["run_manifest_sha256"] == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert identities["checkpoint_sha256"] == hashlib.sha256(checkpoint_bytes).hexdigest()
    (destination / "best.pt").write_bytes(b"modified checkpoint")
    with pytest.raises(RuntimeError, match="Checkpoint file is empty or its SHA-256"):
        matrix._validate_run_artifacts(result, destination, 123, "topology", configuration, source_hashes)


def test_preimport_guard_rejects_changed_bootstrap_before_import(tmp_path, monkeypatch):
    repository = tmp_path / "repo"
    bootstrap = repository / "phase9_geant/colab_h96_v1/launch.py"
    bootstrap.parent.mkdir(parents=True)
    bootstrap.write_text("# changed bootstrap\n", encoding="utf-8")
    archive_path = tmp_path / "source.zip"
    manifest_path = tmp_path / "source.json"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("phase9_geant/colab_h96_v1/launch.py", "# approved bootstrap\n")
    entry_digest = hashlib.sha256(b"# approved bootstrap\n").hexdigest()
    archive_digest = matrix._sha256_file(archive_path)
    manifest_payload = {
        "archive_sha256": archive_digest,
        "entries": [{"path": "phase9_geant/colab_h96_v1/launch.py", "sha256": entry_digest}],
    }
    manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")
    args = SimpleNamespace(
        source_bundle=archive_path, source_manifest=manifest_path,
        expected_source_bundle_sha256=archive_digest,
        expected_source_manifest_sha256=matrix._sha256_file(manifest_path),
    )
    with pytest.raises(RuntimeError, match="does not match the approved source manifest"):
        matrix._preverify_bootstrap_import(repository, args)


def test_namespace_preflight_preserves_completed_seed42_and_refuses_matrix_reuse(tmp_path):
    repository = tmp_path / "repo"
    processed = repository / "data" / "processed"
    processed.mkdir(parents=True)
    successful_seed42 = processed / "geant_horizon96_streaming_experiment_v2_colab_memv3" / "h96_topology_seed_42"
    successful_seed42.mkdir(parents=True)
    bootstrap = SimpleNamespace(_canonical_no_redirect=lambda p, require_exists=True: p)
    root, destinations, control_path = matrix._check_namespace(repository, bootstrap)
    assert all(path.parent == root for path in destinations)
    assert control_path.parent == processed and control_path.name == matrix.MATRIX_CONTROL_MANIFEST
    assert successful_seed42.is_dir()
    root.mkdir()
    with pytest.raises(FileExistsError, match="existing matrix namespace"):
        matrix._check_namespace(repository, bootstrap)


def test_real_trainer_reproduces_old_manifest_collision_and_accepts_sibling_sidecar(tmp_path, monkeypatch):
    """Exercise the actual trainer namespace check, not a stubbed equivalent."""
    from phase9_geant import horizon96_streaming_experiment_v2 as trainer

    monkeypatch.setattr(trainer, "OUTPUT_ROOT_NAME", matrix.MATRIX_OUTPUT_ROOT)

    old_repository = tmp_path / "old" / "repo"
    old_processed = old_repository / "data" / "processed"
    old_root = old_processed / matrix.MATRIX_OUTPUT_ROOT
    old_root.mkdir(parents=True)
    (old_root / matrix.MATRIX_MANIFEST).write_text('{"status":"incomplete"}', encoding="utf-8")
    with pytest.raises(trainer.ExperimentV2Error, match="Unexpected entry in v2 output namespace"):
        trainer._planned_run(old_repository, 123, "topology")

    repository = tmp_path / "fixed" / "repo"
    processed = repository / "data" / "processed"
    processed.mkdir(parents=True)
    output_root = processed / matrix.MATRIX_OUTPUT_ROOT
    control_path = processed / matrix.MATRIX_CONTROL_MANIFEST
    persistent_root = tmp_path / "drive" / "matrix3"
    persistent_root.parent.mkdir()
    matrix_workflow.create_persistent_matrix_root(persistent_root)
    record = {
        "matrix_id": matrix.MATRIX_ID,
        "matrix_output_root": matrix.MATRIX_OUTPUT_ROOT,
        "output_root": str(output_root),
        "matrix_manifest": str(control_path),
        "run_order": [{"seed": 123, "adjacency_mode": "topology"}],
        "status": "incomplete",
        "failure": None,
        "completed_runs": [],
    }
    monkeypatch.setattr(matrix, "RUN_ORDER", ((123, "topology"),))
    authorization_calls = []

    def execute(register_result, authorize_run, record_attempt):
        assert authorize_run(1, 123, "topology") is True
        record_attempt(1, 123, "topology")
        planned_root, destination = trainer._planned_run(repository, 123, "topology")
        assert planned_root == output_root
        destination.mkdir()
        backup_dir = persistent_root / "runs" / "h96_topology_seed_123"
        backup_dir.parent.mkdir(parents=True, exist_ok=True)
        backup_dir.mkdir()
        item = {
            "seed": 123, "adjacency_mode": "topology", "run_directory": str(destination),
            "run_manifest_sha256": "a" * 64, "checkpoint_sha256": "b" * 64,
            "validation_predictions_sha256": "c" * 64,
            "persistent_backup": {
                "verified": True, "directory": str(backup_dir),
                "artifacts": {
                    "run.json": {"sha256": "a" * 64}, "best.pt": {"sha256": "b" * 64},
                    "validation_predictions.npy": {"sha256": "c" * 64},
                },
            },
        }
        register_result(item)
        return [item]

    results = matrix._execute_matrix_lifecycle(
        output_root=output_root,
        control_path=control_path,
        record=record,
        persistent_root=persistent_root,
        execute_matrix=execute,
        authorize_run=lambda *args: authorization_calls.append(args) or True,
    )
    assert results == record["run_results"]
    assert authorization_calls == [(1, 123, "topology")]
    assert output_root.is_dir() and control_path.is_file()
    assert control_path.parent == output_root.parent and control_path not in output_root.iterdir()
    saved = json.loads(control_path.read_text(encoding="utf-8"))
    assert saved["status"] == "complete"
    assert saved["completed_runs"] == results
    assert {p.name for p in (persistent_root / "matrix_lifecycle").iterdir()} == {
        "matrix_run_000_incomplete.json", "matrix_run_001_in_progress.json",
        "matrix_run_001_registered.json", "matrix_run_002_complete.json",
    }


def test_matrix_lifecycle_preserves_failed_status_and_refuses_reuse(tmp_path, monkeypatch):
    processed = tmp_path / "repo" / "data" / "processed"
    processed.mkdir(parents=True)
    output_root = processed / matrix.MATRIX_OUTPUT_ROOT
    control_path = processed / matrix.MATRIX_CONTROL_MANIFEST
    persistent_root = tmp_path / "drive" / "matrix3"
    persistent_root.parent.mkdir()
    matrix_workflow.create_persistent_matrix_root(persistent_root)
    monkeypatch.setattr(matrix, "RUN_ORDER", ((123, "topology"),))
    record = {
        "matrix_id": matrix.MATRIX_ID,
        "matrix_output_root": matrix.MATRIX_OUTPUT_ROOT,
        "output_root": str(output_root),
        "matrix_manifest": str(control_path),
        "run_order": matrix._expected_run_order(),
        "status": "incomplete",
        "failure": None,
        "completed_runs": [],
    }

    def fail_after_authorization(_register, authorize, record_attempt):
        assert authorize(1, 123, "topology") is True
        record_attempt(1, 123, "topology")
        raise RuntimeError("synthetic injected failure")

    with pytest.raises(RuntimeError, match="synthetic injected failure"):
        matrix._execute_matrix_lifecycle(
            output_root=output_root, control_path=control_path, record=record,
            persistent_root=persistent_root,
            execute_matrix=fail_after_authorization,
            authorize_run=lambda *_args: True,
        )
    saved = json.loads(control_path.read_text(encoding="utf-8"))
    assert saved["status"] == "failed"
    assert saved["failure"] == "RuntimeError: synthetic injected failure"
    assert not saved.get("run_results")
    assert saved["current_attempt"]["status"] == "authorized_in_progress"
    assert (persistent_root / "matrix_lifecycle" / "matrix_run_001_failed.json").is_file()
    with pytest.raises(FileExistsError, match="existing matrix namespace"):
        matrix._execute_matrix_lifecycle(
            output_root=output_root, control_path=control_path, record=record,
            persistent_root=persistent_root,
            execute_matrix=lambda *_args: pytest.fail("must not resume"),
            authorize_run=lambda *_args: True,
        )
