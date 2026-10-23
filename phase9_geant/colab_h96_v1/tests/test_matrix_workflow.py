"""Synthetic-only integration checks for upload staging and matrix notebook helpers."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import pytest

from phase9_geant.colab_h96_v1 import matrix_workflow as workflow, package


def _expected(payloads: dict[str, bytes]) -> dict[str, str]:
    return {name: hashlib.sha256(value).hexdigest() for name, value in payloads.items()}


def test_uploads_are_resolved_by_hash_and_filename_suffix_is_irrelevant(tmp_path):
    payloads = {
        "geant_h96_colab_source_v1.zip": b"source synthetic",
        "source_snapshot_manifest.json": b"source manifest synthetic",
        "geant_h96_colab_trainval_v1.zip": b"dataset synthetic",
        "dataset_bundle_manifest.json": b"dataset manifest synthetic",
    }
    expected = _expected(payloads)
    uploads = {f"{name.rsplit('.', 1)[0]} (1).{name.rsplit('.', 1)[1]}": body for name, body in payloads.items()}
    staged = workflow.stage_verified_uploads(uploads, expected, tmp_path / "incoming")
    assert {p.name for p in staged.iterdir()} == set(payloads)
    assert all(hashlib.sha256((staged / name).read_bytes()).hexdigest() == expected[name] for name in payloads)
    with pytest.raises(FileExistsError, match="Refusing existing incoming"):
        workflow.stage_verified_uploads(uploads, expected, staged)


def test_upload_identity_rejects_duplicate_digest_and_extra_files():
    payloads = {
        "geant_h96_colab_source_v1.zip": b"a",
        "source_snapshot_manifest.json": b"b",
        "geant_h96_colab_trainval_v1.zip": b"c",
        "dataset_bundle_manifest.json": b"d",
    }
    expected = _expected(payloads)
    duplicate = {**payloads, "renamed (1).zip": payloads["geant_h96_colab_source_v1.zip"]}
    with pytest.raises(workflow.WorkflowError, match="found 2"):
        workflow.identify_uploaded_files(duplicate, expected)
    with pytest.raises(workflow.WorkflowError, match="Exactly four"):
        workflow.identify_uploaded_files({**payloads, "extra.bin": b"extra"}, expected)


def _bundle(kind: str, members: dict[str, bytes], *, metadata: dict | None = None):
    entries = [{"path": name, "size_bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}
               for name, body in members.items()]
    archive_io = io.BytesIO()
    with zipfile.ZipFile(archive_io, "w") as archive:
        for name, body in members.items():
            archive.writestr(name, body)
        if kind == "source":
            internal = {"entries": entries, "first_party_import_inventory": {}}
            archive.writestr("PORTABLE_SOURCE_MANIFEST.json", json.dumps(internal).encode())
    payload = archive_io.getvalue()
    if kind == "source":
        internal = {"entries": entries, "first_party_import_inventory": {}}
        manifest = {**internal, "archive_sha256": hashlib.sha256(payload).hexdigest(),
                    "internal_manifest_member": "PORTABLE_SOURCE_MANIFEST.json"}
    else:
        exact = list(members)
        manifest = {
            "artifacts": entries, "archive_sha256": hashlib.sha256(payload).hexdigest(),
            "exact_archive_members": exact,
            "source_scope": {"train": [0, 8020], "validation": [8020, 9740]},
            "topology": {
                "node_count": 22, "undirected_link_count": 36, "directed_link_count": 72,
                "local_topology_sha256_identity_only": "38c90629f30af70fe840e33071b12cc40498bb910e8dbeb2fa7863a7b645d121",
                "ordered_edge_sha256": "e" * 64,
            },
        }
    return payload, json.dumps(manifest, sort_keys=True).encode()


def test_source_and_dataset_bundles_verify_exact_members_and_extract_fresh(tmp_path):
    source_payload, source_manifest = _bundle("source", {"phase9_geant/x.py": b"synthetic source"})
    source_digest = hashlib.sha256(source_payload).hexdigest()
    source_manifest_digest = hashlib.sha256(source_manifest).hexdigest()
    source_data, _ = workflow.verify_archive_payload(
        source_payload, source_manifest, expected_archive_sha256=source_digest,
        expected_manifest_sha256=source_manifest_digest, kind="source",
    )
    source_root = workflow.safe_extract_members(
        {"phase9_geant/x.py": b"synthetic source", "PORTABLE_SOURCE_MANIFEST.json": json.dumps({"entries": source_data["entries"], "first_party_import_inventory": {}}).encode()},
        tmp_path / "source_root",
    )
    assert (source_root / "phase9_geant/x.py").read_bytes() == b"synthetic source"

    dataset_members = {name: f"synthetic:{name}".encode() for name in package.DATA_ARTIFACTS}
    dataset_payload, dataset_manifest = _bundle("data", dataset_members)
    manifest, verified = workflow.verify_archive_payload(
        dataset_payload, dataset_manifest,
        expected_archive_sha256=hashlib.sha256(dataset_payload).hexdigest(),
        expected_manifest_sha256=hashlib.sha256(dataset_manifest).hexdigest(), kind="data",
    )
    assert len(verified) == 9 and manifest["topology"]["directed_link_count"] == 72
    output = workflow.safe_extract_members(verified, tmp_path / "data_root")
    assert {path.name for path in output.iterdir()} == set(package.DATA_ARTIFACTS)


@pytest.mark.parametrize("bad_name", ["../escape", "X_test.npy", "nested/test/thing", "C:/outside"])
def test_dataset_bundle_verifier_rejects_test_and_traversal_members(bad_name):
    members = {name: b"synthetic" for name in package.DATA_ARTIFACTS}
    members[bad_name] = b"forbidden"
    payload, manifest = _bundle("data", members)
    with pytest.raises(workflow.WorkflowError):
        workflow.verify_archive_payload(
            payload, manifest, expected_archive_sha256=hashlib.sha256(payload).hexdigest(),
            expected_manifest_sha256=hashlib.sha256(manifest).hexdigest(), kind="data",
        )


def test_matrix_authorization_bridge_prompts_individually_and_exports_without_overwrite(tmp_path):
    class FakePipe:
        def __init__(self, lines):
            self.stdout = iter(lines)
            self.stdin = io.StringIO()
            self.code = 0
        def wait(self):
            return self.code

    fake = FakePipe(["before\n", "AUTH_REQUIRED: type exactly AUTHORIZE H96 RUN 1/8 SEED 123 TOPOLOGY\n", "after\n"])
    prompts, output = [], []
    result = workflow.run_matrix_with_prompt_bridge(
        ["synthetic-child"], prompt_fn=lambda phrase: prompts.append(phrase) or phrase,
        popen_factory=lambda *a, **k: fake, output_fn=output.append,
    )
    assert result == 0
    assert prompts == ["AUTHORIZE H96 RUN 1/8 SEED 123 TOPOLOGY"]
    assert fake.stdin.getvalue().strip() == prompts[0]

    root = tmp_path / "geant_horizon96_streaming_experiment_v3_repro_matrix2"
    root.mkdir()
    control = root.parent / f"{root.name}.matrix_run.json"
    control.write_text('{"status":"incomplete"}', encoding="utf-8")
    run = root / "h96_topology_seed_123"
    run.mkdir()
    for name in ("run.json", "best.pt", "validation_predictions.npy", "unrelated.bin"):
        (run / name).write_bytes(f"opaque:{name}".encode())
    export = workflow.export_matrix_results(
        root, tmp_path / "result.zip", matrix_manifest_path=control,
    )
    with zipfile.ZipFile(export) as archive:
        assert set(archive.namelist()) == {
            "matrix_run.json", "h96_topology_seed_123/run.json",
            "h96_topology_seed_123/best.pt", "h96_topology_seed_123/validation_predictions.npy",
        }
    with pytest.raises(FileExistsError, match="Refusing existing matrix export"):
        workflow.export_matrix_results(root, export, matrix_manifest_path=control)


def _synthetic_completed_run(root: Path, *, seed=123, mode="topology"):
    run = root / f"h96_{mode}_seed_{seed}"
    run.mkdir()
    checkpoint = b"opaque synthetic checkpoint"
    predictions = b"opaque synthetic validation predictions"
    (run / "best.pt").write_bytes(checkpoint)
    (run / "validation_predictions.npy").write_bytes(predictions)
    checkpoint_hash = workflow.sha256_bytes(checkpoint)
    prediction_hash = workflow.sha256_bytes(predictions)
    manifest = {
        "status": "complete", "seed": seed, "adjacency_mode": mode,
        "checkpoint_sha256": checkpoint_hash,
        "validation_predictions_sha256": prediction_hash,
    }
    manifest_bytes = json.dumps(manifest, sort_keys=True).encode()
    (run / "run.json").write_bytes(manifest_bytes)
    return run, manifest, workflow.sha256_bytes(manifest_bytes)


def test_persistent_backup_copies_and_read_verifies_all_run_artifacts(tmp_path):
    local = tmp_path / "local"
    local.mkdir()
    drive_parent = tmp_path / "drive"
    drive_parent.mkdir()
    persistent = workflow.create_persistent_matrix_root(drive_parent / "matrix3")
    run, manifest, manifest_hash = _synthetic_completed_run(local)
    backup = workflow.backup_verified_run(
        run, persistent, run_name="h96_topology_seed_123", expected_seed=123,
        expected_adjacency_mode="topology",
        expected_checkpoint_sha256=manifest["checkpoint_sha256"],
        expected_predictions_sha256=manifest["validation_predictions_sha256"],
        expected_run_manifest_sha256=manifest_hash,
    )
    assert backup["verified"] is True
    target = Path(backup["directory"])
    assert {p.name for p in target.iterdir()} == {
        "run.json", "best.pt", "validation_predictions.npy", "backup_manifest.json",
    }
    for filename, expected in (("run.json", manifest_hash),
                               ("best.pt", manifest["checkpoint_sha256"]),
                               ("validation_predictions.npy", manifest["validation_predictions_sha256"])):
        assert workflow._sha256_path(target / filename) == expected
    assert json.loads((target / "backup_manifest.json").read_text())["readback_verified"] is True


def test_backup_fails_closed_on_source_integrity_mismatch(tmp_path):
    local = tmp_path / "local"
    local.mkdir()
    drive_parent = tmp_path / "drive"
    drive_parent.mkdir()
    persistent = workflow.create_persistent_matrix_root(drive_parent / "matrix3")
    run, manifest, manifest_hash = _synthetic_completed_run(local)
    (run / "best.pt").write_bytes(b"changed after validation")
    with pytest.raises(workflow.WorkflowError, match="Source artifact changed"):
        workflow.backup_verified_run(
            run, persistent, run_name="h96_topology_seed_123", expected_seed=123,
            expected_adjacency_mode="topology",
            expected_checkpoint_sha256=manifest["checkpoint_sha256"],
            expected_predictions_sha256=manifest["validation_predictions_sha256"],
            expected_run_manifest_sha256=manifest_hash,
        )
    assert not (persistent / "runs").exists()


def test_interrupted_partial_backup_is_preserved_and_not_resumed(tmp_path):
    local = tmp_path / "local"
    local.mkdir()
    drive_parent = tmp_path / "drive"
    drive_parent.mkdir()
    persistent = workflow.create_persistent_matrix_root(drive_parent / "matrix3")
    (persistent / "runs").mkdir()
    partial = persistent / "runs" / "h96_topology_seed_123"
    partial.mkdir()
    (partial / "best.pt").write_bytes(b"partial bytes")
    run, manifest, manifest_hash = _synthetic_completed_run(local)
    with pytest.raises(FileExistsError, match="Refusing existing persistent run backup"):
        workflow.backup_verified_run(
            run, persistent, run_name="h96_topology_seed_123", expected_seed=123,
            expected_adjacency_mode="topology",
            expected_checkpoint_sha256=manifest["checkpoint_sha256"],
            expected_predictions_sha256=manifest["validation_predictions_sha256"],
            expected_run_manifest_sha256=manifest_hash,
        )
    assert (partial / "best.pt").read_bytes() == b"partial bytes"


def test_existing_persistent_matrix_root_is_never_reused(tmp_path):
    parent = tmp_path / "drive"
    parent.mkdir()
    existing = parent / "matrix3"
    existing.mkdir()
    with pytest.raises(workflow.WorkflowError, match="already exists"):
        workflow.create_persistent_matrix_root(existing)
    with pytest.raises(workflow.WorkflowError, match="already exists"):
        workflow.validate_persistent_matrix_target(existing)


def test_persistent_lifecycle_snapshots_are_immutable_and_read_verified(tmp_path):
    parent = tmp_path / "drive"
    parent.mkdir()
    root = workflow.create_persistent_matrix_root(parent / "matrix3")
    record = {"status": "incomplete", "completed_runs": []}
    first = workflow.persist_lifecycle_snapshot(
        root, sequence=0, state="incomplete", record=record,
    )
    record["completed_runs"].append({"seed": 123, "backup_verified": True})
    second = workflow.persist_lifecycle_snapshot(
        root, sequence=1, state="registered", record=record,
    )
    assert first["sha256"] != second["sha256"]
    assert '"completed_runs": []' in Path(first["path"]).read_text()
    with pytest.raises(FileExistsError, match="Refusing existing persistent lifecycle snapshot"):
        workflow.persist_lifecycle_snapshot(
            root, sequence=1, state="registered", record=record,
        )

