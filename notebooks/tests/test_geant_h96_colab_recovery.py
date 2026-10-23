"""Synthetic-only tests for the export-only Colab recovery workflow."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import stat
import zipfile
from pathlib import Path

import pytest


NOTEBOOK = Path(__file__).parents[1] / "geant_h96_colab_recovery.ipynb"
CELLS = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]


def _load_recovery_functions(namespace):
    tree = ast.parse("".join(CELLS[1]["source"]), filename=f"{NOTEBOOK}:cell1")
    names = {
        "_checked_existing_path",
        "_validate_interrupted_attempt",
        "_append_progress_durably",
        "_stream_child_output",
        "_assert_no_active_training_process",
        "_sha256_path",
        "_export_incomplete_attempt",
    }
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in nodes} != names:
        raise AssertionError("Recovery workflow function set is incomplete")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(NOTEBOOK), "exec"), namespace)
    return namespace


def _fixture(tmp_path):
    # Keep synthetic paths compact: Windows MAX_PATH applies in some local
    # test environments, while the Colab notebook itself uses its fixed path.
    session = tmp_path / "s"
    repository = session / "p"
    run_dir = repository / "data/processed/geant_horizon96_streaming_experiment_v2_colab_memv3/h96_topology_seed_42"
    run_dir.mkdir(parents=True)
    policy = {
        "policy_id": "colab-linux-process-memory-v3-visible-cgroup",
        "process_peak_limit_bytes": 2 * 1024**3,
        "in_run_os_available_reserve_bytes": 384 * 1024**2,
        "in_run_cgroup_headroom_reserve_bytes": 384 * 1024**2,
        "minimum_free_disk_bytes": 512 * 1024**2,
    }
    manifest = {
        "status": "incomplete",
        "failure": None,
        "epoch_history": [],
        "seed": 42,
        "adjacency_mode": "topology",
        "protocol_id": "geant-horizon96-only-candidate-v2-batch4",
        "test_access": {"permitted": False},
        "configuration": {
            "history_length": 672, "horizons": [96], "batch_size": 4,
            "epochs": 20, "patience": 4, "learning_rate": 0.005,
            "optimizer": "Adam", "hidden_size": 16, "ewma_alpha": 0.8,
            "ewma_steps": 5,
            "residual_loss": "SmoothL1(beta=0.001) in original utilization-ratio units",
            "linux_resource_policy": policy,
        },
    }
    (run_dir / "run.json").write_text(json.dumps(manifest), encoding="utf-8")
    export_dir = tmp_path / "persistent_export"
    export_dir.mkdir()
    ns = {
        "Path": Path,
        "os": os,
        "stat": stat,
        "json": json,
        "hashlib": hashlib,
        "zipfile": zipfile,
        "uuid": __import__("uuid"),
        "SESSION_ROOT": session,
        "REPOSITORY_ROOT": repository,
        "OUTPUT_ROOT": run_dir.parent,
        "RUN_DIR": run_dir,
        "ALLOWED_RUN_FILES": {"run.json"},
    }
    return _load_recovery_functions(ns), session, run_dir, export_dir


def test_incomplete_attempt_is_validated_without_checkpoint_access(tmp_path):
    ns, _, run_dir, _ = _fixture(tmp_path)
    manifest_path, manifest = ns["_validate_interrupted_attempt"](run_dir)
    assert manifest_path == run_dir / "run.json"
    assert manifest["status"] == "incomplete"
    assert not (run_dir / "best.pt").exists()


def test_initial_trainer_manifest_may_omit_epoch_history_and_failure_keys(tmp_path):
    ns, _, run_dir, _ = _fixture(tmp_path)
    manifest_path = run_dir / "run.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("epoch_history")
    manifest.pop("failure")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _, verified = ns["_validate_interrupted_attempt"](run_dir)
    assert verified.get("epoch_history", []) == []


@pytest.mark.parametrize(
    "mutation",
    [
        {"status": "complete"},
        {"failure": "exception"},
        {"epoch_history": [{"epoch": 1}]},
        {"seed": 123},
        {"adjacency_mode": "identity"},
        {"test_access": {"permitted": True}},
    ],
)
def test_recovery_rejects_manifest_that_is_not_the_reported_interruption(tmp_path, mutation):
    ns, _, run_dir, _ = _fixture(tmp_path)
    manifest_path = run_dir / "run.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(mutation)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError):
        ns["_validate_interrupted_attempt"](run_dir)


def test_export_is_unique_preserves_original_and_includes_available_log(tmp_path):
    ns, session, run_dir, export_dir = _fixture(tmp_path)
    original = (run_dir / "run.json").read_bytes()
    log = session / "seed42_topology_launcher_abc.log"
    log.write_text("epoch output that was durably persisted\n", encoding="utf-8")
    result = ns["_export_incomplete_attempt"](run_dir, session, export_dir, process_check=lambda _: None)
    archive_path = Path(result["archive"])
    assert archive_path.is_file()
    with zipfile.ZipFile(archive_path) as archive:
        assert set(archive.namelist()) == {
            "run/run.json", "recovery_report.json", "recovery_progress.jsonl",
            "launcher_logs/1_seed42_topology_launcher_abc.log"
        }
        report = json.loads(archive.read("recovery_report.json"))
        assert report["status"] == "recovered_incomplete_export_only"
        assert report["retry"].startswith("prohibited")
    assert (run_dir / "run.json").read_bytes() == original
    second = ns["_export_incomplete_attempt"](run_dir, session, export_dir, process_check=lambda _: None)
    assert second["archive"] != result["archive"]


def test_export_refuses_extra_run_artifacts_instead_of_reading_them(tmp_path):
    ns, session, run_dir, export_dir = _fixture(tmp_path)
    (run_dir / "best.pt").write_bytes(b"synthetic checkpoint")
    with pytest.raises(RuntimeError, match="contents differ"):
        ns["_export_incomplete_attempt"](run_dir, session, export_dir, process_check=lambda _: None)
    assert len(list(export_dir.glob("h96_seed42_recovery_progress_*.jsonl"))) == 1
    assert not list(export_dir.glob("*.zip"))


def test_validation_failure_leaves_durable_recovery_journal_without_touching_run(tmp_path):
    ns, session, run_dir, export_dir = _fixture(tmp_path)
    manifest_path = run_dir / "run.json"
    original = manifest_path.read_bytes()
    manifest = json.loads(original)
    manifest["seed"] = 123
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="does not match"):
        ns["_export_incomplete_attempt"](run_dir, session, export_dir, process_check=lambda _: None)
    journals = list(export_dir.glob("h96_seed42_recovery_progress_*.jsonl"))
    assert len(journals) == 1
    events = [json.loads(line)["event"] for line in journals[0].read_text(encoding="utf-8").splitlines()]
    assert events == ["recovery_started", "recovery_validation_failed"]
    # Recovery must not modify the source attempt while recording its failure.
    assert manifest_path.read_text(encoding="utf-8") == json.dumps(manifest)
    assert not list(export_dir.glob("*.zip"))


def test_active_trainer_detection_fails_closed_and_does_not_export(tmp_path):
    ns, session, run_dir, export_dir = _fixture(tmp_path)
    proc_root = tmp_path / "proc"
    proc = proc_root / "123"
    proc.mkdir(parents=True)
    command = b"python\x00" + str(ns["REPOSITORY_ROOT"]).encode() + b"\x00phase9_geant/colab_h96_v1/launch.py\x00"
    (proc / "cmdline").write_bytes(command)
    with pytest.raises(RuntimeError, match="still active"):
        ns["_export_incomplete_attempt"](
            run_dir, session, export_dir,
            process_check=lambda root: ns["_assert_no_active_training_process"](root, proc_root),
        )
    assert list(export_dir.iterdir()) == []


def test_process_visibility_error_fails_closed(tmp_path, monkeypatch):
    ns, _, _, _ = _fixture(tmp_path)
    proc_root = tmp_path / "proc"
    proc = proc_root / "321"
    proc.mkdir(parents=True)
    cmdline = proc / "cmdline"
    cmdline.write_bytes(b"synthetic process")
    original_read_bytes = Path.read_bytes

    def permission_limited_read(path):
        if path == cmdline:
            raise PermissionError("synthetic /proc visibility restriction")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", permission_limited_read)
    with pytest.raises(RuntimeError, match="Cannot verify process 321"):
        ns["_assert_no_active_training_process"](ns["REPOSITORY_ROOT"], proc_root)


def test_export_archive_is_verifiable_and_journal_is_preserved(tmp_path):
    ns, session, run_dir, export_dir = _fixture(tmp_path)
    result = ns["_export_incomplete_attempt"](run_dir, session, export_dir, process_check=lambda _: None)
    archive_path = Path(result["archive"])
    assert archive_path.is_file()
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.testzip() is None
        journal_bytes = archive.read("recovery_progress.jsonl")
        manifest_bytes = archive.read("run/run.json")
        report = json.loads(archive.read("recovery_report.json"))
    entries = {item["archive_path"]: item for item in report["included_files"]}
    assert entries["run/run.json"]["sha256"] == hashlib.sha256(manifest_bytes).hexdigest()
    assert entries["recovery_progress.jsonl"]["sha256"] == hashlib.sha256(journal_bytes).hexdigest()
    assert (export_dir / Path(result["progress_journal"]).name).is_file()


def test_archive_interruption_leaves_journal_and_partial_not_a_false_complete_export(tmp_path, monkeypatch):
    ns, session, run_dir, export_dir = _fixture(tmp_path)

    class FailingZip:
        def __init__(self, *args, **kwargs):
            raise OSError("synthetic interrupted archive write")

    monkeypatch.setattr(ns["zipfile"], "ZipFile", FailingZip)
    with pytest.raises(OSError, match="synthetic interrupted"):
        ns["_export_incomplete_attempt"](run_dir, session, export_dir, process_check=lambda _: None)
    journals = list(export_dir.glob("h96_seed42_recovery_progress_*.jsonl"))
    assert len(journals) == 1
    events = [json.loads(line)["event"] for line in journals[0].read_text(encoding="utf-8").splitlines()]
    assert events[-1] == "archive_write_failed"
    assert len(list(export_dir.glob("*.partial"))) == 1
    assert not list(export_dir.glob("*.zip"))


def test_export_refuses_to_replace_an_existing_archive(tmp_path, monkeypatch):
    ns, session, run_dir, export_dir = _fixture(tmp_path)
    archive_path = export_dir / "h96_seed42_incomplete_recovery_fixed.zip"
    sentinel = b"pre-existing user artifact"
    archive_path.write_bytes(sentinel)
    monkeypatch.setattr(ns["uuid"], "uuid4", lambda: type("UUID", (), {"hex": "fixed"})())
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        ns["_export_incomplete_attempt"](run_dir, session, export_dir, process_check=lambda _: None)
    assert archive_path.read_bytes() == sentinel


def test_progress_log_is_flushed_per_line_and_survives_interruption(tmp_path):
    ns, session, _, _ = _fixture(tmp_path)
    local_log = session / "local.log"
    persistent_log = tmp_path / "persistent_export" / "progress.log"

    class InterruptingOutput:
        def __iter__(self):
            yield "epoch 1 started\n"
            raise KeyboardInterrupt

    class Process:
        stdout = InterruptingOutput()

        def wait(self):
            return 0

    with pytest.raises(KeyboardInterrupt):
        ns["_stream_child_output"](Process(), [local_log, persistent_log], emit=lambda *args, **kwargs: None)
    assert local_log.read_text(encoding="utf-8") == "epoch 1 started\n"
    assert persistent_log.read_text(encoding="utf-8") == "epoch 1 started\n"


def test_recovery_notebook_contains_no_training_or_dataset_loader_path():
    code = "\n".join("".join(cell["source"]) for cell in CELLS if cell["cell_type"] == "code")
    ast.parse(code)
    for forbidden in ("--run-training", "numpy.load", "np.load", "torch.load", "X_test.npy", "y_test.npy"):
        assert forbidden not in code
