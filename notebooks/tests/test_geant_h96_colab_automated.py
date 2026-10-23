"""Synthetic functional tests for the embedded Colab preparation logic.

These tests extract and execute selected notebook functions. They create only
temporary synthetic archives/files and never import project code or read data.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import shutil
import stat
import subprocess
import sys
import types
import zipfile
from pathlib import Path, PurePosixPath

import pytest


NOTEBOOK = Path(__file__).parents[1] / "geant_h96_colab_automated.ipynb"
CELLS = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]


def _source(cell_index: int) -> str:
    return "".join(CELLS[cell_index]["source"])


def _load_notebook_definitions(cell_index: int, names: set[str], namespace: dict) -> dict:
    tree = ast.parse(_source(cell_index), filename=f"{NOTEBOOK}:cell{cell_index}")
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names]
    found = {node.name for node in nodes}
    if found != names:
        raise AssertionError(f"Notebook functions missing: {sorted(names - found)}")
    code = compile(ast.Module(body=nodes, type_ignores=[]), str(NOTEBOOK), "exec")
    exec(code, namespace)
    return namespace


def _load_verifier(namespace: dict) -> dict:
    namespace.setdefault("Path", Path)
    _load_notebook_definitions(4, {"_sha256_file"}, namespace)
    tree = ast.parse(_source(5), filename=f"{NOTEBOOK}:cell5")
    names = {"EXPECTED_DATA_ARTIFACTS", "SOURCE_INTERNAL_MANIFEST", "REQUIRED_SOURCE_PATHS"}
    selected = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = {target.id for target in node.targets if isinstance(target, ast.Name)}
            if targets & names:
                selected.append(node)
                names -= targets
    if names:
        raise AssertionError(f"Notebook verifier constants missing: {sorted(names)}")
    funcs = {"_safe_member", "_zip_inventory", "_hash_zip_member", "_verify_bundle"}
    selected.extend(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in funcs)
    if {node.name for node in selected if isinstance(node, ast.FunctionDef)} != funcs:
        raise AssertionError("Notebook archive-verifier function set is incomplete")
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(NOTEBOOK), "exec"), namespace)
    return namespace


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _make_synthetic_bundles(tmp_path: Path, ns: dict) -> tuple[dict, dict, dict]:
    source_entries = []
    source_payloads = {}
    for name in sorted(ns["REQUIRED_SOURCE_PATHS"]):
        payload = ("synthetic-source:" + name).encode()
        source_payloads[name] = payload
        entry = {"path": name, "size_bytes": len(payload), "sha256": _sha(payload)}
        if name.endswith(".py"):
            entry["first_party_imports"] = []
        source_entries.append(entry)
    imports = {entry["path"]: entry.get("first_party_imports", []) for entry in source_entries if entry["path"].endswith(".py")}
    internal = {
        "schema": "geant-h96-colab-source-snapshot-v1",
        "entries": source_entries,
        "first_party_import_inventory": imports,
    }
    source_zip = tmp_path / "geant_h96_colab_source_v1.zip"
    with zipfile.ZipFile(source_zip, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, payload in source_payloads.items():
            archive.writestr(name, payload)
        archive.writestr(ns["SOURCE_INTERNAL_MANIFEST"], json.dumps(internal, sort_keys=True).encode())
    source_manifest = {
        **internal,
        "archive_name": "synthetic_source.zip",
        "archive_sha256": _sha(source_zip.read_bytes()),
        "internal_manifest_member": ns["SOURCE_INTERNAL_MANIFEST"],
    }
    source_manifest_path = tmp_path / "source_snapshot_manifest.json"
    source_manifest_path.write_text(json.dumps(source_manifest, sort_keys=True), encoding="utf-8")

    data_payloads = {name: ("synthetic-data:" + name).encode() for name in ns["EXPECTED_DATA_ARTIFACTS"]}
    artifact_entries = [
        {"path": name, "size_bytes": len(data_payloads[name]), "sha256": _sha(data_payloads[name])}
        for name in ns["EXPECTED_DATA_ARTIFACTS"]
    ]
    dataset_zip = tmp_path / "geant_h96_colab_trainval_v1.zip"
    with zipfile.ZipFile(dataset_zip, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, payload in data_payloads.items():
            archive.writestr(name, payload)
    dataset_manifest = {
        "schema": "geant-h96-colab-trainval-bundle-v1",
        "archive_name": "synthetic_dataset.zip",
        "archive_sha256": _sha(dataset_zip.read_bytes()),
        "artifacts": artifact_entries,
        "exact_archive_members": list(ns["EXPECTED_DATA_ARTIFACTS"]),
        "source_scope": {"train": [0, 8020], "validation": [8020, 9740]},
        "history_length": 672,
        "horizon": 96,
        "window_counts": {"train": 3924, "validation": 953},
        "topology": {
            "node_count": 22,
            "undirected_link_count": 36,
            "directed_link_count": 72,
            "ordered_edge_sha256": "a" * 64,
        },
    }
    dataset_manifest_path = tmp_path / "dataset_bundle_manifest.json"
    dataset_manifest_path.write_text(json.dumps(dataset_manifest, sort_keys=True), encoding="utf-8")
    paths = {
        "geant_h96_colab_source_v1.zip": source_zip,
        "source_snapshot_manifest.json": source_manifest_path,
        "geant_h96_colab_trainval_v1.zip": dataset_zip,
        "dataset_bundle_manifest.json": dataset_manifest_path,
    }
    return source_manifest, dataset_manifest, paths


def _load_acquisition(monkeypatch, tmp_path: Path):
    payloads = {
        "geant_h96_colab_source_v1.zip": b"synthetic-source-zip",
        "source_snapshot_manifest.json": b"synthetic-source-manifest",
        "geant_h96_colab_trainval_v1.zip": b"synthetic-dataset-zip",
        "dataset_bundle_manifest.json": b"synthetic-dataset-manifest",
    }
    ns = {
        "hashlib": hashlib,
        "os": os,
        "re": __import__("re"),
        "stat": stat,
        "shutil": shutil,
        "Path": Path,
        "INCOMING": tmp_path / "incoming",
        "EXPECTED_FILES": {name: _sha(data) for name, data in payloads.items()},
        "ACQUISITION_MODE": "upload",
        "LOCAL_PACKAGE_DIRECTORY": "",
    }
    ns["INCOMING"].mkdir()
    _load_notebook_definitions(4, {
        "_sha256_bytes", "_sha256_file", "_canonical_candidate", "_write_new_bytes",
        "_safe_components", "_acquire_candidates", "_identify_and_stage",
    }, ns)
    uploaded = {
        "geant_h96_colab_source_v1 (1).zip": payloads["geant_h96_colab_source_v1.zip"],
        "source_snapshot_manifest.json": payloads["source_snapshot_manifest.json"],
        "geant_h96_colab_trainval_v1.zip": payloads["geant_h96_colab_trainval_v1.zip"],
        "dataset_bundle_manifest.json": payloads["dataset_bundle_manifest.json"],
    }
    google = types.ModuleType("google")
    colab = types.ModuleType("google.colab")
    files = types.SimpleNamespace(upload=lambda: uploaded)
    colab.files = files
    google.colab = colab
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    return ns, payloads


def test_local_folder_acquisition_stages_by_hash_and_reuses_exact_matches(monkeypatch, tmp_path):
    ns, payloads = _load_acquisition(monkeypatch, tmp_path)
    local = tmp_path / "provided_packages"
    local.mkdir()
    for canonical, payload in payloads.items():
        stem, suffix = os.path.splitext(canonical)
        (local / f"{stem} (1){suffix}").write_bytes(payload)
    ns["ACQUISITION_MODE"] = "local"
    ns["LOCAL_PACKAGE_DIRECTORY"] = str(local)
    first = ns["_identify_and_stage"]()
    staged_bytes = {name: path.read_bytes() for name, path in first.items()}
    second = ns["_identify_and_stage"]()
    assert first == second
    assert {name: path.read_bytes() for name, path in second.items()} == staged_bytes


def test_existing_staged_digest_mismatch_is_never_overwritten(monkeypatch, tmp_path):
    ns, _ = _load_acquisition(monkeypatch, tmp_path)
    staged_path = ns["INCOMING"] / "geant_h96_colab_source_v1.zip"
    staged_path.write_bytes(b"old-or-corrupt")
    uploaded = {
        "geant_h96_colab_source_v1.zip": b"synthetic-source-zip",
        "source_snapshot_manifest.json": b"synthetic-source-manifest",
        "geant_h96_colab_trainval_v1.zip": b"synthetic-dataset-zip",
        "dataset_bundle_manifest.json": b"synthetic-dataset-manifest",
    }
    colab = sys.modules["google.colab"]
    colab.files = types.SimpleNamespace(upload=lambda: uploaded)
    with pytest.raises(RuntimeError, match="Refusing existing mismatched staged file"):
        ns["_identify_and_stage"]()
    assert staged_path.read_bytes() == b"old-or-corrupt"


def test_upload_acquisition_uses_digest_and_accepts_colab_suffix(monkeypatch, tmp_path):
    ns, payloads = _load_acquisition(monkeypatch, tmp_path)
    staged = ns["_identify_and_stage"]()
    assert set(staged) == set(payloads)
    assert staged["geant_h96_colab_source_v1.zip"].name == "geant_h96_colab_source_v1.zip"
    assert all(path.read_bytes() == payloads[name] for name, path in staged.items())


def test_synthetic_manifests_and_exact_archives_verify(tmp_path):
    ns = _load_verifier({
        "hashlib": hashlib,
        "json": json,
        "re": __import__("re"),
        "stat": stat,
        "zipfile": zipfile,
        "PurePosixPath": PurePosixPath,
    })
    source_manifest, dataset_manifest, paths = _make_synthetic_bundles(tmp_path, ns)
    ns["STAGED_FILES"] = paths
    ns["EXPECTED_FILES"] = {
        "geant_h96_colab_source_v1.zip": source_manifest["archive_sha256"],
        "source_snapshot_manifest.json": _sha(paths["source_snapshot_manifest.json"].read_bytes()),
        "geant_h96_colab_trainval_v1.zip": dataset_manifest["archive_sha256"],
        "dataset_bundle_manifest.json": _sha(paths["dataset_bundle_manifest.json"].read_bytes()),
    }
    _load_notebook_definitions(5, {"_verify_packages"}, ns)
    source_result, dataset_result = ns["_verify_packages"]()
    assert source_result["entries"] == source_manifest["entries"]
    assert dataset_result["artifacts"] == dataset_manifest["artifacts"]


@pytest.mark.parametrize("member", ["../escape.txt", "metadata.json"])
def test_zip_verifier_rejects_traversal_and_duplicate_names(tmp_path, member):
    ns = _load_verifier({
        "hashlib": hashlib,
        "json": json,
        "re": __import__("re"),
        "stat": stat,
        "zipfile": zipfile,
        "PurePosixPath": PurePosixPath,
    })
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(member, b"one")
        if member == "metadata.json":
            zf.writestr(member, b"two")
    manifest_path = tmp_path / "manifest.json"
    manifest = {"schema": "geant-h96-colab-trainval-bundle-v1", "archive_sha256": _sha(archive.read_bytes())}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises((ValueError, RuntimeError)):
        ns["_verify_bundle"]("dataset", archive, manifest_path, manifest["archive_sha256"], _sha(manifest_path.read_bytes()))


def test_extraction_and_deep_verification_use_verified_dataset_manifest(tmp_path):
    verifier_ns = _load_verifier({
        "hashlib": hashlib,
        "json": json,
        "re": __import__("re"),
        "stat": stat,
        "zipfile": zipfile,
        "PurePosixPath": PurePosixPath,
    })
    source_manifest, dataset_manifest, paths = _make_synthetic_bundles(tmp_path, verifier_ns)
    verifier_ns["STAGED_FILES"] = paths
    verifier_ns["EXPECTED_FILES"] = {
        "geant_h96_colab_source_v1.zip": source_manifest["archive_sha256"],
        "source_snapshot_manifest.json": _sha(paths["source_snapshot_manifest.json"].read_bytes()),
        "geant_h96_colab_trainval_v1.zip": dataset_manifest["archive_sha256"],
        "dataset_bundle_manifest.json": _sha(paths["dataset_bundle_manifest.json"].read_bytes()),
    }
    _load_notebook_definitions(5, {"_verify_packages"}, verifier_ns)
    source_manifest, dataset_manifest = verifier_ns["_verify_packages"]()
    source_root = tmp_path / "synthetic_source_root"
    dataset_root = source_root / "data/processed/geant_horizon96_streaming_v1"

    extraction_ns = {
        "hashlib": hashlib,
        "os": os,
        "stat": stat,
        "shutil": shutil,
        "zipfile": zipfile,
        "Path": Path,
        "_sha256_file": verifier_ns["_sha256_file"],
        "_safe_member": verifier_ns["_safe_member"],
        "SOURCE_MANIFEST": source_manifest,
        "DATASET_MANIFEST": dataset_manifest,
        "SOURCE_INTERNAL_MANIFEST": verifier_ns["SOURCE_INTERNAL_MANIFEST"],
        "EXPECTED_DATA_ARTIFACTS": verifier_ns["EXPECTED_DATA_ARTIFACTS"],
        "STAGED_FILES": paths,
        "REPOSITORY_ROOT": source_root,
    }
    _load_notebook_definitions(4, {"_safe_components"}, extraction_ns)
    _load_notebook_definitions(6, {"_verify_existing_extraction", "_extract_verified"}, extraction_ns)
    extracted = extraction_ns["_extract_verified"]()
    assert extracted == dataset_root
    assert {path.name for path in extracted.iterdir()} == set(verifier_ns["EXPECTED_DATA_ARTIFACTS"])
    assert extraction_ns["_verify_existing_extraction"](source_manifest, dataset_manifest) == dataset_root
    (source_root / "stale_debug_module.py").write_text("# synthetic stale extraction residue\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="unexpected/missing files"):
        extraction_ns["_verify_existing_extraction"](source_manifest, dataset_manifest)
    (source_root / "stale_debug_module.py").unlink()

    calls = []
    fake_package = types.SimpleNamespace(
        verify_source_bundle=lambda *args: calls.append(("source", args)),
        verify_dataset_bundle=lambda *args: calls.append(("dataset", args)),
        verify_extracted_source=lambda *args: calls.append(("extracted_source", args)),
        _read_dataset_metadata=lambda root: {
            "protocol_id": "geant-horizon96-streaming-trainval-v1",
            "source_scope": {"train": [0, 8020], "validation": [8020, 9740]},
            "windows": {"train_count": 3924, "validation_count": 953, "history_length": 672, "horizon": 96},
            "topology": {
                "node_count": 22,
                "undirected_link_count": 36,
                "directed_link_count": 72,
                "directed_edges_interleaved": list(range(72)),
                "ordered_edge_sha256": dataset_manifest["topology"]["ordered_edge_sha256"],
            },
        },
    )
    deep_ns = {
        "portable_package": fake_package,
        "STAGED_FILES": paths,
        "EXPECTED_FILES": {
            "geant_h96_colab_source_v1.zip": source_manifest["archive_sha256"],
            "source_snapshot_manifest.json": _sha(paths["source_snapshot_manifest.json"].read_bytes()),
            "geant_h96_colab_trainval_v1.zip": dataset_manifest["archive_sha256"],
            "dataset_bundle_manifest.json": _sha(paths["dataset_bundle_manifest.json"].read_bytes()),
        },
        "SOURCE_MANIFEST": source_manifest,
        "DATASET_MANIFEST": dataset_manifest,
        "REPOSITORY_ROOT": source_root,
        "DATA_ROOT": extracted,
        "VERIFIED_MANIFESTS": (source_manifest, dataset_manifest),
    }
    _load_notebook_definitions(6, {"_package_reverification"}, deep_ns)
    metadata = deep_ns["_package_reverification"]()
    assert metadata["protocol_id"] == "geant-horizon96-streaming-trainval-v1"
    assert any(name == "dataset" and args[1] is paths["dataset_bundle_manifest.json"] for name, args in calls)


def test_deep_verification_rejects_manifest_topology_mismatch(tmp_path):
    ns = _load_verifier({
        "hashlib": hashlib,
        "json": json,
        "re": __import__("re"),
        "stat": stat,
        "zipfile": zipfile,
        "PurePosixPath": PurePosixPath,
    })
    source_manifest, dataset_manifest, paths = _make_synthetic_bundles(tmp_path, ns)
    fake_package = types.SimpleNamespace(
        verify_source_bundle=lambda *args: None,
        verify_dataset_bundle=lambda *args: None,
        verify_extracted_source=lambda *args: None,
        _read_dataset_metadata=lambda root: {
            "protocol_id": "geant-horizon96-streaming-trainval-v1",
            "source_scope": {"train": [0, 8020], "validation": [8020, 9740]},
            "windows": {"train_count": 3924, "validation_count": 953, "history_length": 672, "horizon": 96},
            "topology": {
                "node_count": 22,
                "undirected_link_count": 36,
                "directed_link_count": 72,
                "directed_edges_interleaved": list(range(72)),
                "ordered_edge_sha256": "b" * 64,
            },
        },
    )
    deep_ns = {
        "portable_package": fake_package,
        "STAGED_FILES": paths,
        "EXPECTED_FILES": {name: "0" * 64 for name in paths},
        "SOURCE_MANIFEST": source_manifest,
        "DATASET_MANIFEST": dataset_manifest,
        "REPOSITORY_ROOT": tmp_path,
        "DATA_ROOT": tmp_path,
        "VERIFIED_MANIFESTS": (source_manifest, dataset_manifest),
    }
    _load_notebook_definitions(6, {"_package_reverification"}, deep_ns)
    with pytest.raises(RuntimeError, match="Ordered directed-edge identity"):
        deep_ns["_package_reverification"]()


def test_portable_verifier_loads_only_from_verified_snapshot_not_cached_decoy(tmp_path, monkeypatch):
    root = tmp_path / "approved_source_snapshot"
    package_path = root / "phase9_geant/colab_h96_v1/package.py"
    package_path.parent.mkdir(parents=True)
    package_path.write_text("VALUE = 'approved-snapshot'\n", encoding="utf-8")
    digest = _sha(package_path.read_bytes())
    decoy = types.ModuleType("phase9_geant.colab_h96_v1.package")
    decoy.VALUE = "stale-decoy"
    monkeypatch.setitem(sys.modules, decoy.__name__, decoy)
    ns = {
        "_safe_components": _load_notebook_definitions(
            4, {"_safe_components"}, {"os": os, "Path": Path, "stat": stat}
        )["_safe_components"],
        "REPOSITORY_ROOT": root,
        "SOURCE_MANIFEST": {"entries": [{"path": "phase9_geant/colab_h96_v1/package.py", "sha256": digest}]},
        "_sha256_file": _load_notebook_definitions(
            4, {"_sha256_file"}, {"hashlib": hashlib, "Path": Path}
        )["_sha256_file"],
        "importlib": importlib,
        "sys": sys,
    }
    _load_notebook_definitions(6, {"_load_verified_portable_package"}, ns)
    loaded = ns["_load_verified_portable_package"]()
    assert loaded.VALUE == "approved-snapshot"
    assert loaded.__file__ == str(package_path)
    assert sys.modules[decoy.__name__].VALUE == "stale-decoy"


def test_synthetic_suite_orchestrator_requires_exact_68_passes(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    test_names = [
        "phase9_geant/colab_h96_v1/tests/test_portability.py",
        "phase9_geant/tests/test_horizon96_streaming_experiment.py",
        "phase9_geant/tests/test_horizon96_streaming_experiment_v2.py",
        "phase9_geant/tests/test_spatiotemporal_link.py",
    ]
    for name in test_names:
        target = repo / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("# synthetic placeholder\n", encoding="utf-8")
    basetemp = tmp_path / "pytest-synthetic"
    basetemp.mkdir()

    class FakeSubprocess:
        PIPE = object()
        STDOUT = object()

        @staticmethod
        def run(command, **kwargs):
            assert "--basetemp" in command
            assert str(basetemp) in command
            assert command[-4:] == test_names
            return types.SimpleNamespace(returncode=0, stdout="68 passed in 1.2s\n")

    ns = {
        "SYNTHETIC_TESTS": test_names,
        "REPOSITORY_ROOT": repo,
        "TEST_BASETEMP": basetemp,
        "sys": types.SimpleNamespace(executable="python"),
        "subprocess": FakeSubprocess,
        "print": lambda *args, **kwargs: None,
        "flush": True,
        "re": __import__("re"),
    }
    _load_notebook_definitions(7, {"_run_synthetic_suite"}, ns)
    assert ns["_run_synthetic_suite"]()["exit_code"] == 0
    FakeSubprocess.run = staticmethod(lambda *args, **kwargs: types.SimpleNamespace(returncode=0, stdout="67 passed in 1.2s\n"))
    with pytest.raises(RuntimeError, match="Expected exactly 68"):
        ns["_run_synthetic_suite"]()


def _dry_run_namespace(tmp_path, *, policy_override=None):
    repo = tmp_path / "approved_source_snapshot"
    repo.mkdir()
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    names = (
        "geant_h96_colab_source_v1.zip", "source_snapshot_manifest.json",
        "geant_h96_colab_trainval_v1.zip", "dataset_bundle_manifest.json",
    )
    staged = {}
    expected = {}
    for name in names:
        path = incoming / name
        path.write_bytes(b"synthetic:" + name.encode())
        staged[name] = path
        expected[name] = _sha(path.read_bytes())
    output_root = repo / "data/processed/geant_horizon96_streaming_experiment_v2_colab_memv3"
    policy = {
        "policy_id": "colab-linux-process-memory-v3-visible-cgroup",
        "process_peak_limit_bytes": 2 * 1024**3,
        "startup_headroom_bytes": 2 * 1024**3 + 384 * 1024**2,
        "minimum_free_disk_bytes": 512 * 1024**2,
        "in_run_os_available_reserve_bytes": 384 * 1024**2,
        "in_run_cgroup_headroom_reserve_bytes": 384 * 1024**2,
        "visible_unlimited_cgroup_policy": "allow literal unlimited visible limits only; host and process guards remain mandatory",
    }
    if policy_override:
        policy.update(policy_override)
    summary = {
        "status": "preflight_passed_no_training", "seed": 42, "adjacency_mode": "topology",
        "horizon": 96, "history_length": 672,
        "run_destination": str(output_root / "h96_topology_seed_42"),
        "output_root": str(output_root),
        "data_protocol_id": "geant-horizon96-streaming-trainval-v1",
        "environment": {"device_policy": "CPU only for this migration protocol"},
        "linux_resource_preflight": {"status": "pass", "policy": policy},
        "train_windows": 3924, "validation_windows": 953,
        "test_access": "no test files included or loaded",
        "source_bundle_sha256": expected["geant_h96_colab_source_v1.zip"],
        "source_snapshot_manifest_sha256": expected["source_snapshot_manifest.json"],
        "dataset_bundle_sha256": expected["geant_h96_colab_trainval_v1.zip"],
        "dataset_manifest_sha256": expected["dataset_bundle_manifest.json"],
    }

    class FakeSubprocess:
        PIPE = object()
        STDOUT = object()

        @staticmethod
        def run(command, **kwargs):
            assert "--dry-run" in command
            assert "--run-training" not in command
            assert "--authorize-training" not in command
            assert command[1:3] == ["-I", "-S"]
            return types.SimpleNamespace(returncode=0, stdout=json.dumps(summary), stderr="")

    return {
        "REPOSITORY_ROOT": repo, "STAGED_FILES": staged, "EXPECTED_FILES": expected,
        "Path": Path, "sys": types.SimpleNamespace(executable="python"),
        "subprocess": FakeSubprocess, "json": json, "re": __import__("re"),
        "print": lambda *args, **kwargs: None,
    }, summary, output_root


def test_launcher_dry_run_orchestrator_checks_identity_and_v3_policy(tmp_path):
    ns, _, output_root = _dry_run_namespace(tmp_path)
    _load_notebook_definitions(8, {"_run_launcher_dry_run"}, ns)
    result = ns["_run_launcher_dry_run"]()
    assert result["status"] == "preflight_passed_no_training"
    assert not output_root.exists()


def test_launcher_dry_run_rejects_resource_policy_drift(tmp_path):
    ns, _, _ = _dry_run_namespace(tmp_path, policy_override={"process_peak_limit_bytes": 640 * 1024**2})
    _load_notebook_definitions(8, {"_run_launcher_dry_run"}, ns)
    with pytest.raises(RuntimeError, match="memory-policy identity/limit mismatch"):
        ns["_run_launcher_dry_run"]()


def _training_namespace(tmp_path):
    # The integration test derives the fixed nested run path below this root.
    # Use a short synthetic repo basename so Windows MAX_PATH does not make
    # test execution depend on the machine's pytest temp-root length.
    repo = tmp_path / "r"
    launcher = repo / "phase9_geant/colab_h96_v1/launch.py"
    launcher.parent.mkdir(parents=True)
    launcher.write_text("# synthetic launcher placeholder", encoding="utf-8")
    session = tmp_path / "session"
    session.mkdir()
    incoming = session / "incoming"
    incoming.mkdir()
    staged_payloads = {
        "geant_h96_colab_source_v1.zip": b"src",
        "source_snapshot_manifest.json": b"src-manifest",
        "geant_h96_colab_trainval_v1.zip": b"data",
        "dataset_bundle_manifest.json": b"data-manifest",
    }
    staged = {}
    for name, payload in staged_payloads.items():
        path = incoming / name
        path.write_bytes(payload)
        staged[name] = path
    expected = {name: _sha(payload) for name, payload in staged_payloads.items()}
    run_root = repo / "data/processed/geant_horizon96_streaming_experiment_v2_colab_memv3"
    run_dir = run_root / "h96_topology_seed_42"
    ns = {
        "uuid": __import__("uuid"),
        "zipfile": zipfile,
        "Path": Path,
        "PurePosixPath": PurePosixPath,
        "re": __import__("re"),
        "math": __import__("math"),
        "json": json,
        "os": os,
        "hashlib": hashlib,
        "subprocess": subprocess,
        "signal": __import__("signal"),
        "traceback": __import__("traceback"),
        "sys": sys,
        "print": print,
        "input": input,
        "_safe_components": _load_notebook_definitions(
            4,
            {"_safe_components"},
            {"os": os, "Path": Path, "stat": stat},
        )["_safe_components"],
        "_sha256_file": _load_notebook_definitions(
            4, {"_sha256_file"}, {"hashlib": hashlib, "Path": Path}
        )["_sha256_file"],
        "WORKFLOW_ID": "fresh-workflow",
        "PACKAGE_WORKFLOW_ID": "fresh-workflow",
        "SYNTHETIC_SUITE_WORKFLOW_ID": "fresh-workflow",
        "DRY_RUN_WORKFLOW_ID": "fresh-workflow",
        "DRY_RUN_RESULT": {"status": "preflight_passed_no_training"},
        "SYNTHETIC_SUITE_RESULT": {"exit_code": 0},
        "PACKAGE_VERSIONS": {"numpy": "1.26.4", "torch": "2.12.0", "psutil": "7.2.2", "pytest": "9.1.1"},
        "PINNED": {"numpy": "1.26.4", "torch": "2.12.0", "psutil": "7.2.2", "pytest": "9.1.1"},
        "OUTPUT_ROOT": run_root,
        "RUN_DIR": run_dir,
        "REPOSITORY_ROOT": repo,
        "SESSION_ROOT": session,
        "STAGED_FILES": staged,
        "EXPECTED_FILES": expected,
        "AUTHORIZATION_PHRASE": "AUTHORIZE SEED 42 TOPOLOGY TRAINING",
    }
    return ns


def _load_training_functions(ns):
    tree = ast.parse(_source(10), filename=f"{NOTEBOOK}:cell10")
    wanted = {"_archive_attempt", "_authorized_training_workflow"}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    if {node.name for node in nodes} != wanted:
        raise AssertionError("Notebook authorization/launch/export function set is incomplete")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(NOTEBOOK), "exec"), ns)
    return ns


class _FakeProcess:
    def __init__(self, lines, exit_code=0):
        self.stdout = iter(lines)
        self._exit_code = exit_code

    def wait(self, **kwargs):
        return self._exit_code

    def poll(self):
        return self._exit_code


def _successful_fake_popen_factory(ns, *, status="complete", exit_code=0):
    def fake_popen(command, **kwargs):
        assert "--run-training" in command
        assert "--authorize-training" in command
        assert command[-4:] == ["--seed", "42", "--adjacency-mode", "topology"]
        run_dir = ns["RUN_DIR"]
        run_dir.mkdir(parents=True, exist_ok=False)
        checkpoint = b"synthetic-checkpoint"
        (run_dir / "best.pt").write_bytes(checkpoint)
        config = {
            "protocol_id": "geant-horizon96-only-candidate-v2-batch4",
            "history_length": 672, "horizons": [96], "batch_size": 4,
            "epochs": 20, "patience": 4, "learning_rate": 0.005,
            "optimizer": "Adam", "hidden_size": 16, "ewma_alpha": 0.8,
            "ewma_steps": 5,
            "residual_loss": "SmoothL1(beta=0.001) in original utilization-ratio units",
        }
        manifest = {
            "status": status, "configuration": config, "seed": 42,
            "adjacency_mode": "topology", "checkpoint": "best.pt",
            "checkpoint_reload_strict": True, "test_access": {"permitted": False},
            "checkpoint_sha256": _sha(checkpoint),
            "validation_predictions": "validation_predictions.npy",
            "validation_predictions_sha256": _sha(b"synthetic-validation-predictions"),
            "configuration": {
                **config,
                "colab_bundle_identity": {
                    "source_snapshot_manifest_sha256": ns["EXPECTED_FILES"]["source_snapshot_manifest.json"],
                    "source_bundle_sha256": ns["EXPECTED_FILES"]["geant_h96_colab_source_v1.zip"],
                    "dataset_manifest_sha256": ns["EXPECTED_FILES"]["dataset_bundle_manifest.json"],
                    "dataset_bundle_sha256": ns["EXPECTED_FILES"]["geant_h96_colab_trainval_v1.zip"],
                },
                "linux_resource_policy": {
                    "policy_id": "colab-linux-process-memory-v3-visible-cgroup",
                    "process_peak_limit_bytes": 2 * 1024**3,
                    "in_run_os_available_reserve_bytes": 384 * 1024**2,
                    "in_run_cgroup_headroom_reserve_bytes": 384 * 1024**2,
                    "minimum_free_disk_bytes": 512 * 1024**2,
                },
            },
            "best_epoch": 1, "epochs_completed": 1,
            "best_validation_h96_residual_smooth_l1": 0.01,
            "epoch_history": [{"epoch": 1}], "runtime_seconds": 1.0,
            "memory_monitor": {}, "validation_metrics": {
                "h96_candidate": {"mae": 0.08, "rmse": 0.1},
                "ewma_alpha_0.8": {"mae": 0.09, "rmse": 0.11},
            },
        }
        (run_dir / "run.json").write_text(json.dumps(manifest), encoding="utf-8")
        (run_dir / "validation_predictions.npy").write_bytes(b"synthetic-validation-predictions")
        return _FakeProcess(["synthetic progress\n"], exit_code=exit_code)
    return fake_popen


def test_training_launch_requires_current_preflight_and_exact_confirmation(tmp_path):
    ns = _load_training_functions(_training_namespace(tmp_path))
    called = []
    denied = ns["_authorized_training_workflow"](
        prompt_fn=lambda _: "no",
        popen_factory=lambda *args, **kwargs: called.append(args),
        download_fn=lambda _: None,
    )
    assert denied["status"] == "not_authorized"
    assert called == []
    ns["DRY_RUN_WORKFLOW_ID"] = "stale"
    with pytest.raises(RuntimeError, match="absent or stale"):
        ns["_authorized_training_workflow"](prompt_fn=lambda _: ns["AUTHORIZATION_PHRASE"])
    assert called == []


def test_authorized_training_attempt_verifies_and_exports_synthetic_run(tmp_path):
    ns = _load_training_functions(_training_namespace(tmp_path))
    downloads = []
    ns["subprocess"] = subprocess
    result = ns["_authorized_training_workflow"](
        prompt_fn=lambda _: ns["AUTHORIZATION_PHRASE"],
        popen_factory=_successful_fake_popen_factory(ns),
        download_fn=downloads.append,
    )
    assert result["status"] == "complete"
    assert result["exit_code"] == 0
    assert len(downloads) == 1
    export_path = Path(downloads[0])
    assert export_path.is_file()
    with zipfile.ZipFile(export_path) as archive:
        names = set(archive.namelist())
    assert names == {
        "launcher_output.log", "run/best.pt", "run/run.json",
        "run/validation_predictions.npy",
    }
    with zipfile.ZipFile(export_path) as archive:
        launcher_logs = [name for name in archive.namelist() if name == "launcher_output.log"]
        assert launcher_logs and archive.read("launcher_output.log") == b"synthetic progress\n"
    assert ns["RUN_DIR"].exists()


def test_training_attempt_exports_failed_run_diagnostics_without_retry(tmp_path):
    ns = _load_training_functions(_training_namespace(tmp_path))
    downloads = []
    result = ns["_authorized_training_workflow"](
        prompt_fn=lambda _: ns["AUTHORIZATION_PHRASE"],
        popen_factory=_successful_fake_popen_factory(ns, status="failed", exit_code=1),
        download_fn=downloads.append,
    )
    assert result["status"] == "failed"
    assert result["exit_code"] == 1
    assert len(downloads) == 1
    with zipfile.ZipFile(downloads[0]) as archive:
        assert "run/run.json" in archive.namelist()
        assert "launcher_output.log" in archive.namelist()


def test_existing_training_namespace_is_rejected_before_prompt_or_launch(tmp_path):
    ns = _load_training_functions(_training_namespace(tmp_path))
    ns["OUTPUT_ROOT"].mkdir(parents=True)
    called = []
    with pytest.raises(FileExistsError, match="Refusing existing run namespace"):
        ns["_authorized_training_workflow"](
            prompt_fn=lambda _: pytest.fail("authorization prompt must not be reached"),
            popen_factory=lambda *args, **kwargs: called.append(args),
        )
    assert called == []


def test_consumed_authorization_marker_prevents_second_attempt(tmp_path):
    ns = _load_training_functions(_training_namespace(tmp_path))
    marker = ns["SESSION_ROOT"] / "seed42_topology_attempt_consumed.json"
    marker.write_text("already consumed", encoding="utf-8")
    called = []
    with pytest.raises(FileExistsError, match="already recorded"):
        ns["_authorized_training_workflow"](
            prompt_fn=lambda _: ns["AUTHORIZATION_PHRASE"],
            popen_factory=lambda *args, **kwargs: called.append(args),
        )
    assert called == []


def test_completed_but_unverifiable_run_is_not_reported_as_complete(tmp_path):
    ns = _load_training_functions(_training_namespace(tmp_path))
    downloads = []

    def fake_popen(command, **kwargs):
        fake = _successful_fake_popen_factory(ns)(command, **kwargs)
        manifest_path = ns["RUN_DIR"] / "run.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["checkpoint_sha256"] = "0" * 64
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return fake

    result = ns["_authorized_training_workflow"](
        prompt_fn=lambda _: ns["AUTHORIZATION_PHRASE"],
        popen_factory=fake_popen,
        download_fn=downloads.append,
    )
    assert result["manifest_status"] == "complete"
    assert result["status"] == "complete_unverified"
    assert "Checkpoint is missing" in result["verification_error"]
    assert len(downloads) == 1


def test_interrupted_subprocess_keeps_incremental_log_and_exports_it(tmp_path):
    ns = _load_training_functions(_training_namespace(tmp_path))
    downloads = []

    class InterruptedLines:
        def __iter__(self):
            yield "batch progress persisted first\n"
            raise KeyboardInterrupt

    class InterruptedProcess:
        stdout = InterruptedLines()

        @staticmethod
        def poll():
            return 0

    result = ns["_authorized_training_workflow"](
        prompt_fn=lambda _: ns["AUTHORIZATION_PHRASE"],
        popen_factory=lambda *args, **kwargs: InterruptedProcess(),
        download_fn=downloads.append,
    )
    assert result["interrupted"] is True
    assert len(downloads) == 1
    with zipfile.ZipFile(downloads[0]) as archive:
        assert archive.read("launcher_output.log") == b"batch progress persisted first\n"
