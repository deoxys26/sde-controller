"""Versioned eight-run Colab matrix launcher for h=96 reproducibility.

The existing launch.py remains the seed-42/topology first-run launcher. This
separate entry point uses the same verified source/data package, v3 resource
policy, trainer, and scientific configuration, but a fresh output namespace.
It never selects or loads the official test split.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import stat
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Callable, Sequence



MATRIX_ID = "geant-h96-reproducibility-matrix-v3-drive-backed"
MATRIX_OUTPUT_ROOT = "geant_horizon96_streaming_experiment_v3_repro_matrix3"
RUN_ORDER: tuple[tuple[int, str], ...] = (
    (123, "topology"), (2026, "topology"),
    (42, "identity"), (123, "identity"), (2026, "identity"),
    (42, "shuffled"), (123, "shuffled"), (2026, "shuffled"),
)
MATRIX_MANIFEST = "matrix_run.json"
MATRIX_CONTROL_MANIFEST = f"{MATRIX_OUTPUT_ROOT}.matrix_run.json"
AUTHORIZATION_PREFIX = "AUTHORIZE H96 RUN"
matrix_workflow: Any | None = None


class MatrixLaunchError(RuntimeError):
    """The approved bundle, resource policy, or output namespace is invalid."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _preverify_bootstrap_import(repository: Path, args: argparse.Namespace) -> None:
    """Verify the exact bundle and bootstrap source before importing launch.py."""
    repository = Path(os.path.abspath(repository))
    if not repository.is_dir() or repository.resolve(strict=True) != repository:
        raise MatrixLaunchError("Repository root is missing or redirected")
    source_archive = Path(os.path.abspath(args.source_bundle))
    source_manifest = Path(os.path.abspath(args.source_manifest))
    for path in (source_archive, source_manifest):
        if path.resolve(strict=True) != path or not path.is_file():
            raise MatrixLaunchError(f"Source identity file is missing or redirected: {path}")
    expected_archive = args.expected_source_bundle_sha256.lower()
    expected_manifest = args.expected_source_manifest_sha256.lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected_archive) or not re.fullmatch(r"[0-9a-f]{64}", expected_manifest):
        raise MatrixLaunchError("Expected source digests must be lowercase or uppercase SHA-256 hex")
    if _sha256_file(source_archive) != expected_archive or _sha256_file(source_manifest) != expected_manifest:
        raise MatrixLaunchError("Source bundle or manifest differs from its externally reviewed digest")
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    if manifest.get("archive_sha256") != expected_archive:
        raise MatrixLaunchError("Source manifest does not bind the supplied source archive")
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise MatrixLaunchError("Source manifest has no entries")
    entry = next((item for item in entries if item.get("path") == "phase9_geant/colab_h96_v1/launch.py"), None)
    if not isinstance(entry, dict) or not isinstance(entry.get("sha256"), str):
        raise MatrixLaunchError("Verified bundle does not identify the bootstrap launcher")
    bootstrap_path = repository / "phase9_geant" / "colab_h96_v1" / "launch.py"
    current = Path(bootstrap_path.anchor)
    for component in bootstrap_path.parts[1:]:
        current = current / component
        info = current.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise MatrixLaunchError(f"Refusing redirected bootstrap component: {current}")
    if not bootstrap_path.is_file() or _sha256_file(bootstrap_path) != entry["sha256"]:
        raise MatrixLaunchError("Extracted bootstrap launcher does not match the approved source manifest")
    with zipfile.ZipFile(source_archive) as archive:
        info = archive.getinfo(entry["path"])
        mode = (info.external_attr >> 16) & 0xFFFF
        if stat.S_ISLNK(mode):
            raise MatrixLaunchError("Bootstrap launcher is a symlink in the source archive")
        digest = hashlib.sha256()
        with archive.open(info) as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != entry["sha256"]:
            raise MatrixLaunchError("Bootstrap member differs from the reviewed source manifest")


def _load_bootstrap(repository: Path):
    launch_path = repository / "phase9_geant" / "colab_h96_v1" / "launch.py"
    spec = importlib.util.spec_from_file_location("_geant_h96_verified_launch", launch_path)
    if spec is None or spec.loader is None:
        raise MatrixLaunchError("Could not load the sibling bootstrap launcher")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _matrix_control_path(processed: Path, output_root: Path, bootstrap: Any) -> Path:
    """Place matrix lifecycle metadata beside—not inside—the trainer namespace.

    The trainer's `_planned_run()` owns the output-root inventory and accepts
    only run directories and its documented aggregate artifacts. Keeping this
    exact control manifest as a sibling lets that strict check remain intact.
    """
    if output_root.name != MATRIX_OUTPUT_ROOT or output_root.parent != processed:
        raise MatrixLaunchError("Matrix output root does not match its fixed namespace identity")
    control_path = processed / MATRIX_CONTROL_MANIFEST
    bootstrap._canonical_no_redirect(control_path, require_exists=False)
    if control_path.parent.resolve(strict=True) != processed or control_path.name != MATRIX_CONTROL_MANIFEST:
        raise MatrixLaunchError("Matrix control manifest is outside its canonical sibling location")
    if control_path.exists() or control_path.is_symlink():
        raise FileExistsError(f"Refusing existing matrix control manifest: {control_path}")
    return control_path


def _check_namespace(
    repository: Path, bootstrap: Any,
) -> tuple[Path, tuple[Path, ...], Path]:
    processed = repository / "data" / "processed"
    bootstrap._canonical_no_redirect(processed, require_exists=True)
    if not processed.is_dir():
        raise MatrixLaunchError(f"Canonical processed-data root is unavailable: {processed}")
    output_root = repository / "data" / "processed" / MATRIX_OUTPUT_ROOT
    bootstrap._canonical_no_redirect(output_root, require_exists=False)
    if output_root.parent.resolve(strict=True) != processed:
        raise MatrixLaunchError("Matrix output root is not a direct child of the canonical processed-data root")
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"Refusing existing matrix namespace; no resume or overwrite: {output_root}")
    control_path = _matrix_control_path(processed, output_root, bootstrap)
    destinations = tuple(
        output_root / f"h96_{mode}_seed_{seed}" for seed, mode in RUN_ORDER
    )
    if len(set(destinations)) != len(RUN_ORDER):
        raise MatrixLaunchError("Matrix output plan contains duplicate destinations")
    if any(path.exists() or path.is_symlink() for path in destinations):
        raise FileExistsError("Refusing one or more pre-existing matrix run destinations")
    return output_root, destinations, control_path


def _same_input_identity(current: tuple[Any, ...], expected: tuple[Any, ...]) -> bool:
    """Compare provenance/schema identities without Python array equality."""
    return (
        current[0] == expected[0]
        and current[1] == expected[1]
        and current[3] == expected[3]
        and current[4] == expected[4]
        and current[5] == expected[5]
        and current[6] == expected[6]
        and len(current[2]["train"]["windows"]) == len(expected[2]["train"]["windows"])
        and len(current[2]["validation"]["windows"]) == len(expected[2]["validation"]["windows"])
    )


def _validate_run_result(result: dict[str, Any], seed: int, mode: str, protocol_id: str) -> None:
    if result.get("status") != "complete":
        raise MatrixLaunchError(
            f"Run did not complete ({seed}, {mode}); preserve it and do not retry this matrix"
        )
    expected = {
        "status": "complete", "seed": seed, "adjacency_mode": mode,
        "protocol_id": protocol_id, "checkpoint": "best.pt",
        "checkpoint_reload_strict": True,
    }
    if any(result.get(key) != value for key, value in expected.items()):
        raise MatrixLaunchError(f"Run result identity/finalization mismatch for seed={seed}, mode={mode}")
    digest = result.get("checkpoint_sha256")
    if not isinstance(digest, str) or len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise MatrixLaunchError("Run result has no valid checkpoint SHA-256")
    if result.get("test_access", {}).get("permitted") is not False:
        raise MatrixLaunchError("Run result does not affirm test-split denial")
    metrics = result.get("validation_metrics", {})
    if not {"h96_candidate", "ewma_alpha_0.8"}.issubset(metrics):
        raise MatrixLaunchError("Run result lacks paired candidate/EWMA validation metrics")
    for name in ("h96_candidate", "ewma_alpha_0.8"):
        values = metrics[name]
        if (not isinstance(values, dict)
                or any(not isinstance(values.get(key), (int, float)) or not __import__("math").isfinite(values[key])
                       for key in ("mae", "rmse"))):
            raise MatrixLaunchError(f"Run result has invalid {name} validation metrics")
        if values.get("evaluated_pairs") != 953 * 72:
            raise MatrixLaunchError(f"Run result does not cover the fixed 953x72 validation target pairs: {name}")


def _validate_run_artifacts(
    result: dict[str, Any], destination: Path, seed: int, mode: str,
    expected_configuration: dict[str, Any], expected_source_hashes: dict[str, str],
) -> dict[str, Any]:
    """Verify only the exact run files, treating arrays/checkpoints as opaque bytes."""
    if not destination.is_dir() or destination.is_symlink():
        raise MatrixLaunchError(f"Completed run directory is missing or redirected: {destination}")
    expected_files = {"run.json", "best.pt", "validation_predictions.npy"}
    if {entry.name for entry in destination.iterdir()} != expected_files:
        raise MatrixLaunchError(f"Completed run has an unexpected artifact inventory: {destination}")
    manifest_path = destination / "run.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or manifest.get("seed") != seed or manifest.get("adjacency_mode") != mode:
        raise MatrixLaunchError("On-disk run manifest identity differs from the completed run result")
    if manifest.get("protocol_id") != result.get("protocol_id") or manifest.get("configuration") != expected_configuration:
        raise MatrixLaunchError("On-disk run protocol/configuration differs from the approved matrix configuration")
    if (manifest.get("checkpoint") != "best.pt"
            or manifest.get("validation_predictions") != "validation_predictions.npy"
            or manifest.get("checkpoint_reload_strict") is not True):
        raise MatrixLaunchError("Run manifest does not identify strictly reloaded expected artifacts")
    if manifest.get("source_hashes") != expected_source_hashes:
        raise MatrixLaunchError("On-disk run source identity differs from preflight")
    if manifest.get("test_access", {}).get("permitted") is not False:
        raise MatrixLaunchError("On-disk run does not affirm test-split denial")
    checkpoint = destination / "best.pt"
    predictions = destination / "validation_predictions.npy"
    if checkpoint.stat().st_size <= 0 or _sha256_file(checkpoint) != manifest.get("checkpoint_sha256"):
        raise MatrixLaunchError("Checkpoint file is empty or its SHA-256 differs from the manifest")
    if predictions.stat().st_size <= 0 or _sha256_file(predictions) != manifest.get("validation_predictions_sha256"):
        raise MatrixLaunchError("Validation prediction file is empty or its SHA-256 differs from the manifest")
    if result.get("checkpoint_sha256") != manifest.get("checkpoint_sha256"):
        raise MatrixLaunchError("Returned checkpoint identity differs from the on-disk artifact")
    if result.get("validation_predictions_sha256") != manifest.get("validation_predictions_sha256"):
        raise MatrixLaunchError("Returned prediction identity differs from the on-disk artifact")
    return {
        "run_manifest_sha256": _sha256_file(manifest_path),
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "validation_predictions_sha256": manifest["validation_predictions_sha256"],
    }


def _authorization_phrase(index: int, seed: int, mode: str) -> str:
    return f"{AUTHORIZATION_PREFIX} {index}/8 SEED {seed} {mode.upper()}"


def _request_run_authorization(index: int, seed: int, mode: str) -> bool:
    phrase = _authorization_phrase(index, seed, mode)
    print(f"AUTH_REQUIRED: type exactly {phrase}", flush=True)
    response = sys.stdin.readline()
    if response == "":
        raise PermissionError(f"No authorization response received for {phrase}")
    if response.strip() != phrase:
        raise PermissionError(f"Authorization declined for matrix run {index}/8; no retry")
    return True


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=False, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            os.replace(temporary, path)
        else:
            os.rename(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _expected_run_order() -> list[dict[str, Any]]:
    return [{"seed": seed, "adjacency_mode": mode} for seed, mode in RUN_ORDER]


def _validate_matrix_record(
    record: object, *, output_root: Path, control_path: Path,
) -> dict[str, Any]:
    """Require the exact sidecar type, paths, and matrix namespace identity."""
    if not isinstance(record, dict):
        raise MatrixLaunchError("Matrix lifecycle metadata must be a JSON object")
    if (record.get("matrix_id") != MATRIX_ID
            or record.get("matrix_output_root") != MATRIX_OUTPUT_ROOT
            or record.get("output_root") != str(output_root)
            or record.get("matrix_manifest") != str(control_path)
            or record.get("run_order") != _expected_run_order()):
        raise MatrixLaunchError("Matrix lifecycle metadata identity/path mismatch")
    if record.get("status") not in {"incomplete", "failed", "complete"}:
        raise MatrixLaunchError("Matrix lifecycle status is invalid")
    if not isinstance(record.get("completed_runs"), list):
        raise MatrixLaunchError("Matrix lifecycle completed_runs must be a list")
    if record.get("failure") is not None and not isinstance(record.get("failure"), str):
        raise MatrixLaunchError("Matrix lifecycle failure field must be null or a string")
    return record


def _persist_matrix_record(
    record_path: Path, *, output_root: Path, record: dict[str, Any],
) -> None:
    expected_path = output_root.parent / MATRIX_CONTROL_MANIFEST
    if (record_path != expected_path or output_root.name != MATRIX_OUTPUT_ROOT
            or record_path.parent != output_root.parent or record_path.name != MATRIX_CONTROL_MANIFEST
            or record_path.resolve(strict=False) != record_path):
        raise MatrixLaunchError("Refusing matrix metadata outside its canonical namespace-bound sidecar")
    _validate_matrix_record(record, output_root=output_root, control_path=record_path)
    if record_path.exists() or record_path.is_symlink():
        if record_path.is_symlink() or not record_path.is_file() or record_path.resolve(strict=True) != record_path:
            raise MatrixLaunchError("Existing matrix metadata is redirected or not a regular file")
        existing = _validate_matrix_record(
            json.loads(record_path.read_text(encoding="utf-8")),
            output_root=output_root, control_path=record_path,
        )
        if existing.get("status") in {"failed", "complete"}:
            raise MatrixLaunchError("Terminal matrix metadata is immutable; refusing reuse")
    elif record.get("status") != "incomplete" or record.get("completed_runs"):
        raise MatrixLaunchError("A new matrix sidecar must begin with an empty incomplete record")
    _atomic_json(record_path, record)


def _execute_matrix_lifecycle(
    *, output_root: Path, control_path: Path, record: dict[str, Any],
    persistent_root: Path,
    execute_matrix: Callable[[Callable[[dict[str, Any]], None], Callable[[int, int, str], bool], Callable[[int, int, str], None]], list[dict[str, Any]]],
    authorize_run: Callable[[int, int, str], bool],
) -> list[dict[str, Any]]:
    """Create a fresh trainer namespace and persist incomplete/failed/complete."""
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"Refusing existing matrix namespace; no resume or overwrite: {output_root}")
    if control_path.exists() or control_path.is_symlink():
        raise FileExistsError(f"Refusing existing matrix control manifest: {control_path}")
    output_root.mkdir(parents=False, exist_ok=False)
    if output_root.resolve(strict=True) != output_root or not output_root.is_dir():
        raise MatrixLaunchError("Created matrix trainer namespace is redirected or not a directory")
    try:
        _persist_matrix_record(control_path, output_root=output_root, record=record)
        matrix_workflow.persist_lifecycle_snapshot(
            persistent_root, sequence=0, state="incomplete", record=record,
        )
    except BaseException:
        # Leave the newly created namespace as a durable refusal marker; never
        # remove a path whose filesystem identity could have changed.
        raise

    def record_attempt(index: int, seed: int, mode: str) -> None:
        if index != len(record["completed_runs"]) + 1 or index > len(RUN_ORDER):
            raise MatrixLaunchError("Attempt record does not match the next fixed matrix run")
        if RUN_ORDER[index - 1] != (seed, mode):
            raise MatrixLaunchError("Attempt seed/mode does not match the fixed matrix order")
        record["current_attempt"] = {
            "index": index, "seed": seed, "adjacency_mode": mode,
            "status": "authorized_in_progress",
        }
        _persist_matrix_record(control_path, output_root=output_root, record=record)
        matrix_workflow.persist_lifecycle_snapshot(
            persistent_root, sequence=index, state="in_progress", record=record,
        )

    def register_result(item: dict[str, Any]) -> None:
        index = len(record["completed_runs"])
        if index >= len(RUN_ORDER):
            raise MatrixLaunchError("Matrix attempted to register more runs than the fixed plan")
        seed, mode = RUN_ORDER[index]
        expected_dir = output_root / f"h96_{mode}_seed_{seed}"
        if (not isinstance(item, dict) or item.get("seed") != seed
                or item.get("adjacency_mode") != mode
                or item.get("run_directory") != str(expected_dir)):
            raise MatrixLaunchError("Registered run does not match the next fixed matrix destination")
        backup = item.get("persistent_backup")
        if not isinstance(backup, dict) or backup.get("verified") is not True:
            raise MatrixLaunchError("Run cannot be registered before persistent backup verification")
        backup_dir = Path(backup.get("directory", ""))
        if (backup_dir.resolve(strict=True) != backup_dir or persistent_root not in backup_dir.parents
                or backup_dir.name != f"h96_{mode}_seed_{seed}"):
            raise MatrixLaunchError("Persistent backup path is outside this fresh matrix namespace")
        for filename, key in (("run.json", "run_manifest_sha256"), ("best.pt", "checkpoint_sha256"),
                              ("validation_predictions.npy", "validation_predictions_sha256")):
            if backup.get("artifacts", {}).get(filename, {}).get("sha256") != item.get(key):
                raise MatrixLaunchError(f"Persistent backup identity mismatch for {filename}")
        record["completed_runs"].append(item)
        record["last_completed_run"] = item
        record["current_attempt"] = None
        _persist_matrix_record(control_path, output_root=output_root, record=record)
        matrix_workflow.persist_lifecycle_snapshot(
            persistent_root, sequence=len(record["completed_runs"]),
            state="registered", record=record,
        )

    try:
        results = execute_matrix(register_result, authorize_run, record_attempt)
        if len(record["completed_runs"]) != len(RUN_ORDER) or len(results) != len(RUN_ORDER):
            raise MatrixLaunchError("Matrix cannot complete before every planned run is registered")
        record.update({"status": "complete", "failure": None, "run_results": results})
        _persist_matrix_record(control_path, output_root=output_root, record=record)
        matrix_workflow.persist_lifecycle_snapshot(
            persistent_root, sequence=len(record["completed_runs"]) + 1,
            state="complete", record=record,
        )
        return results
    except BaseException as exc:
        record.update({"status": "failed", "failure": f"{type(exc).__name__}: {exc}"})
        try:
            _persist_matrix_record(control_path, output_root=output_root, record=record)
        except BaseException:
            pass
        try:
            matrix_workflow.persist_lifecycle_snapshot(
                persistent_root, sequence=len(record["completed_runs"]) + 1,
                state="failed", record=record,
            )
        except BaseException:
            pass
        raise


def _run_sequentially(
    *,
    trainer: Any,
    resource_module: Any,
    data_root: Path,
    repository: Path,
    output_root: Path,
    expected_inputs: tuple[Any, ...],
    expected_source_hashes: dict[str, str],
    expected_resource_policy: dict[str, Any],
    protocol_id: str,
    preflight: Callable[[Path], Any] | None = None,
    run_one: Callable[..., dict[str, Any]] | None = None,
    authorize_run: Callable[[int, int, str], bool] | None = None,
    register_result: Callable[[dict[str, Any]], None] | None = None,
    record_attempt: Callable[[int, int, str], None] | None = None,
    artifact_verifier: Callable[..., dict[str, Any]] | None = None,
    backup_run: Callable[..., dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Run the fixed order; stop at the first failure and never retry."""
    if authorize_run is None:
        raise PermissionError("A separate explicit authorization callback is required before every run")
    if backup_run is None:
        raise MatrixLaunchError("A persistent Drive backup callback is mandatory before matrix registration")
    if record_attempt is None:
        raise MatrixLaunchError("Persistent in-progress lifecycle recording is mandatory before training")
    preflight = preflight or trainer._verified_inputs
    run_one = run_one or trainer.train_one
    artifact_verifier = artifact_verifier or _validate_run_artifacts
    results: list[dict[str, Any]] = []
    for index, (seed, mode) in enumerate(RUN_ORDER, start=1):
        resource = resource_module.preflight(disk_path=repository)
        if resource.get("policy") != expected_resource_policy:
            raise MatrixLaunchError("Colab resource policy changed between matrix runs")
        inputs = preflight(data_root, repository)
        if not _same_input_identity(inputs, expected_inputs):
            raise MatrixLaunchError("Train/validation provenance or array identity changed between runs")
        if trainer._source_identity(repository) != expected_source_hashes:
            raise MatrixLaunchError("Source identity changed between matrix runs")
        planned_root, destination = trainer._planned_run(repository, seed, mode)
        if planned_root != output_root or destination.exists() or destination.is_symlink():
            raise FileExistsError(f"Refusing existing or redirected run destination: {destination}")
        if authorize_run(index, seed, mode) is not True:
            raise PermissionError(f"Run {index}/8 was not explicitly authorized; stop without retry")
        record_attempt(index, seed, mode)
        result = run_one(
            data_root, repository_root=repository, seed=seed, mode=mode,
            run_training=True, explicit_authorization=True, device_name="cpu",
            monitor=resource_module.LinuxMemoryMonitor(),
        )
        _validate_run_result(result, seed, mode, protocol_id)
        expected_configuration = {
            **dict(trainer.CONFIG), "seed": seed, "adjacency_mode": mode, "device": "cpu",
        }
        if result.get("configuration") != expected_configuration:
            raise MatrixLaunchError("Returned run configuration differs from the fixed matrix protocol")
        artifact_identity = artifact_verifier(
            result, destination, seed, mode, expected_configuration, expected_source_hashes,
        )
        backup = backup_run(
            result=result, destination=destination, seed=seed, mode=mode,
            artifact_identity=artifact_identity,
        )
        if (not isinstance(backup, dict) or backup.get("verified") is not True
                or backup.get("artifacts", {}).get("best.pt", {}).get("sha256") != artifact_identity.get("checkpoint_sha256")
                or backup.get("artifacts", {}).get("validation_predictions.npy", {}).get("sha256") != artifact_identity.get("validation_predictions_sha256")
                or backup.get("artifacts", {}).get("run.json", {}).get("sha256") != artifact_identity.get("run_manifest_sha256")):
            raise MatrixLaunchError("Persistent backup did not verify exact run artifacts; stopping before registration")
        registered = {
            "seed": seed, "adjacency_mode": mode,
            "run_directory": str(destination),
            "status": result.get("status"),
            "epochs_completed": result.get("epochs_completed"),
            "best_epoch": result.get("best_epoch"),
            "best_validation_h96_residual_smooth_l1": result.get("best_validation_h96_residual_smooth_l1"),
            "validation_metrics": result.get("validation_metrics"),
            "runtime_seconds": result.get("runtime_seconds"),
            "checkpoint_sha256": result.get("checkpoint_sha256"),
            "validation_predictions_sha256": result.get("validation_predictions_sha256"),
            "persistent_backup": backup,
            **artifact_identity,
        }
        results.append(registered)
        if register_result is not None:
            register_result(registered)
        print(
            f"RUN_COMPLETE seed={seed} adjacency={mode} epochs={result.get('epochs_completed')} "
            f"mae={result['validation_metrics']['h96_candidate']['mae']:.10g} "
            f"rmse={result['validation_metrics']['h96_candidate']['rmse']:.10g} "
            f"runtime_s={result.get('runtime_seconds')} "
            f"checkpoint_sha256={artifact_identity['checkpoint_sha256']} backup=verified",
            flush=True,
        )
    return results


def _prepare(args: argparse.Namespace):
    global matrix_workflow
    repository = Path(os.path.abspath(args.repository_root))
    if not repository.is_dir() or repository.resolve(strict=True) != repository:
        raise MatrixLaunchError(f"Repository root is missing or redirected: {repository}")
    _preverify_bootstrap_import(repository, args)
    bootstrap = _load_bootstrap(repository)
    identity = bootstrap._verify_source_before_import(
        repository, args.source_bundle, args.source_manifest,
        expected_archive_sha256=args.expected_source_bundle_sha256,
        expected_manifest_sha256=args.expected_source_manifest_sha256,
    )
    bootstrap._activate_system_site_packages()
    approved_paths = {
        (repository / entry["path"]).resolve(strict=True)
        for entry in identity["manifest"]["entries"] if entry["path"].endswith(".py")
    }
    package, environment_module, resource_module, trainer = bootstrap._import_verified_runtime(
        repository, approved_paths
    )
    workflow_module = __import__("importlib").import_module("phase9_geant.colab_h96_v1.matrix_workflow")
    workflow_path = Path(workflow_module.__file__).resolve(strict=True)
    if workflow_path not in approved_paths:
        raise MatrixLaunchError("Matrix persistence helper did not originate in the approved source snapshot")
    matrix_workflow = workflow_module
    environment = environment_module.verify_environment(require_pytest=True)
    persistent_target = matrix_workflow.validate_persistent_matrix_target(args.persistent_root)
    # Refuse the entire matrix namespace before bundle verification or any
    # train/validation array read. No attempt resumes a partially used root.
    output_root, destinations, control_path = _check_namespace(repository, bootstrap)
    dataset_result = package.verify_dataset_bundle(
        args.dataset_bundle, args.dataset_manifest,
        args.expected_dataset_bundle_sha256, args.expected_dataset_manifest_sha256,
    )
    data_root = repository / package.DATA_ROOT_RELATIVE
    package._read_dataset_metadata(data_root)
    if {entry.name for entry in data_root.iterdir()} != set(package.DATA_ARTIFACTS):
        raise MatrixLaunchError("Extracted dataset inventory differs from the exact train/validation allowlist")
    policy = resource_module.preflight(disk_path=repository)
    trainer.OUTPUT_ROOT_NAME = MATRIX_OUTPUT_ROOT
    trainer.RUN_SOURCE_FILES = tuple(sorted(set(trainer.RUN_SOURCE_FILES) | set(package.MIGRATION_RUNTIME_FILES)))
    trainer.CONFIG = {
        **trainer.CONFIG,
        "matrix_experiment_id": MATRIX_ID,
        "matrix_run_order": [{"seed": seed, "adjacency_mode": mode} for seed, mode in RUN_ORDER],
        "colab_bundle_identity": {
            "source_snapshot_manifest_sha256": identity["manifest_sha256"],
            "source_bundle_sha256": identity["archive_sha256"],
            "dataset_manifest_sha256": args.expected_dataset_manifest_sha256,
            "dataset_bundle_sha256": dataset_result["archive_sha256"],
        },
        "colab_environment": environment,
    }
    bootstrap._bind_resource_policy(trainer, policy["policy"])
    verified_inputs = trainer._verified_inputs(data_root, repository)
    if len(verified_inputs[2]["train"]["windows"]) != 3924 or len(verified_inputs[2]["validation"]["windows"]) != 953:
        raise MatrixLaunchError("Approved train/validation window counts do not match the h96 protocol")
    source_hashes = trainer._source_identity(repository)
    if "phase9_geant/colab_h96_v1/launch_matrix.py" not in source_hashes:
        raise MatrixLaunchError("Matrix launcher is not included in run source provenance")
    return (
        bootstrap, package, resource_module, trainer, data_root, output_root, control_path,
        destinations, verified_inputs, source_hashes, policy, identity, dataset_result,
        persistent_target,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--dry-run", action="store_true")
    actions.add_argument("--run-matrix", action="store_true")
    parser.add_argument("--authorize-matrix3", action="store_true", help="explicitly authorize beginning the new matrix3 lifecycle")
    parser.add_argument("--persistent-root", type=Path, required=True, help="fresh Drive-backed matrix destination; must not exist")
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--source-bundle", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--expected-source-bundle-sha256", required=True)
    parser.add_argument("--expected-source-manifest-sha256", required=True)
    parser.add_argument("--dataset-bundle", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--expected-dataset-bundle-sha256", required=True)
    parser.add_argument("--expected-dataset-manifest-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    prepared = _prepare(args)
    (bootstrap, package, resource_module, trainer, data_root, output_root, control_path,
     destinations, verified_inputs, source_hashes, policy, identity, dataset_result,
     persistent_target) = prepared
    if args.run_matrix and not args.authorize_matrix3:
        raise PermissionError("--run-matrix requires the separate --authorize-matrix3 flag")
    summary = {
        "status": "preflight_passed_no_training" if args.dry_run else "matrix_training_started",
        "matrix_id": MATRIX_ID, "protocol_id": trainer.PROTOCOL_ID,
        "output_root": str(output_root),
        "matrix_output_root": MATRIX_OUTPUT_ROOT,
        "matrix_manifest": str(control_path),
        "planned_runs": [
            {"seed": seed, "adjacency_mode": mode, "destination": str(path)}
            for (seed, mode), path in zip(RUN_ORDER, destinations, strict=True)
        ],
        "train_windows": len(verified_inputs[2]["train"]["windows"]),
        "validation_windows": len(verified_inputs[2]["validation"]["windows"]),
        "history_length": trainer.HISTORY_LENGTH, "horizon": trainer.HORIZON,
        "resource_policy": policy["policy"],
        "source_hashes": source_hashes,
        "source_bundle_sha256": identity["archive_sha256"],
        "source_manifest_sha256": identity["manifest_sha256"],
        "dataset_bundle_sha256": dataset_result["archive_sha256"],
        "dataset_artifacts": list(package.DATA_ARTIFACTS),
        "test_access": "no test files included or loaded",
        "persistent_root": str(persistent_target),
        "backup_policy": "read-back SHA-256 verification before the next run; no resume or overwrite",
    }
    if args.dry_run:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    repository = Path(os.path.abspath(args.repository_root))
    record = {
        **summary,
        "status": "incomplete", "failure": None, "completed_runs": [],
        "run_order": _expected_run_order(),
        "created_at_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
    }
    record_path = control_path
    persistent_root = matrix_workflow.create_persistent_matrix_root(persistent_target)

    def execute_matrix(register_result, authorize_run, record_attempt):
        return _run_sequentially(
            trainer=trainer, resource_module=resource_module, data_root=data_root,
            repository=repository, output_root=output_root,
            expected_inputs=verified_inputs, expected_source_hashes=source_hashes,
            expected_resource_policy=policy["policy"],
            protocol_id=trainer.PROTOCOL_ID,
            authorize_run=authorize_run,
            register_result=register_result,
            record_attempt=record_attempt,
            backup_run=lambda *, result, destination, seed, mode, artifact_identity: matrix_workflow.backup_verified_run(
                destination, persistent_root,
                run_name=f"h96_{mode}_seed_{seed}", expected_seed=seed,
                expected_adjacency_mode=mode,
                expected_checkpoint_sha256=artifact_identity["checkpoint_sha256"],
                expected_predictions_sha256=artifact_identity["validation_predictions_sha256"],
                expected_run_manifest_sha256=artifact_identity["run_manifest_sha256"],
            ),
        )

    results = _execute_matrix_lifecycle(
        output_root=output_root, control_path=record_path, record=record,
        persistent_root=persistent_root,
        execute_matrix=execute_matrix, authorize_run=_request_run_authorization,
    )
    print(_format_compact_results(record["completed_runs"]))
    print(f"MATRIX_COMPLETE persistent_root={persistent_root} lifecycle_snapshots={persistent_root / 'matrix_lifecycle'}")
    return 0


def _format_compact_results(results: Sequence[dict[str, Any]]) -> str:
    headers = ("seed", "adjacency", "epochs", "MAE", "RMSE", "runtime_s", "checkpoint_sha256", "backup")
    rows = [headers]
    for item in results:
        metrics = item.get("validation_metrics", {}).get("h96_candidate", {})
        rows.append((
            str(item.get("seed")), str(item.get("adjacency_mode")), str(item.get("epochs_completed", "")),
            f"{metrics.get('mae', float('nan')):.8g}", f"{metrics.get('rmse', float('nan')):.8g}",
            str(item.get("runtime_seconds", "")), str(item.get("checkpoint_sha256", "")),
            "verified" if item.get("persistent_backup", {}).get("verified") is True else "MISSING",
        ))
    widths = [max(len(row[i]) for row in rows) for i in range(len(headers))]
    return "\n".join(" | ".join(value.ljust(widths[i]) for i, value in enumerate(row)) for row in rows)


if __name__ == "__main__":
    raise SystemExit(main())
