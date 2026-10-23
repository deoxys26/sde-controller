"""Recovery-only validation and orchestration for the two missing Matrix3 runs.

No training starts on import. Existing Drive backups are read-only inputs and
are validated as opaque checkpoint/prediction bytes plus their JSON manifests.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


RECOVERY_ID = "geant-h96-matrix3-recovery-v1"
RECOVERY_OUTPUT_ROOT = "geant_horizon96_streaming_experiment_v3_repro_matrix3_recovery1"
ORIGINAL_BACKUP_ROOT = "geant_h96_matrix3_5fe371f8aab74f42983569579477c328"
COMPLETED_RUNS: tuple[tuple[int, str], ...] = (
    (123, "topology"), (2026, "topology"),
    (42, "identity"), (123, "identity"), (2026, "identity"),
    (42, "shuffled"),
)
REMAINING_RUNS: tuple[tuple[int, str], ...] = ((123, "shuffled"), (2026, "shuffled"))
RUN_FILES = {"run.json", "best.pt", "validation_predictions.npy", "backup_manifest.json"}
SCIENCE_CONFIG_KEYS = (
    "protocol_id", "runner_id", "history_length", "horizons", "prediction_channels",
    "ewma_alpha", "ewma_steps", "residual_loss", "epochs", "patience",
    "learning_rate", "batch_size", "optimizer", "hidden_size", "seeds",
    "adjacency_modes", "shuffle_seed", "scaler_fit_scope", "checkpoint_selection",
    "validation_gradient_updates", "dataset_protocol_id", "train_bounds",
    "validation_bounds", "memory_policy",
)
CONFIG_METADATA_KEYS = {
    "seed", "adjacency_mode", "device", "matrix_experiment_id", "matrix_run_order",
    "colab_bundle_identity", "colab_environment",
}
IDENTITY_KEYS = ("provenance", "dataset_metadata_sha256", "normalization", "source_hashes")
ORIGINAL_MATRIX_ORDER = [
    {"seed": 123, "adjacency_mode": "topology"},
    {"seed": 2026, "adjacency_mode": "topology"},
    {"seed": 42, "adjacency_mode": "identity"},
    {"seed": 123, "adjacency_mode": "identity"},
    {"seed": 2026, "adjacency_mode": "identity"},
    {"seed": 42, "adjacency_mode": "shuffled"},
    {"seed": 123, "adjacency_mode": "shuffled"},
    {"seed": 2026, "adjacency_mode": "shuffled"},
]


class RecoveryError(RuntimeError):
    """A completed backup or recovery lifecycle failed closed."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_directory(path: str | Path, label: str) -> Path:
    target = Path(os.path.abspath(path))
    if (target.is_symlink() or not target.is_dir()
            or target.resolve(strict=True) != target):
        raise RecoveryError(f"{label} must be a canonical existing directory: {target}")
    return target


def _finite_metric_pair(value: object, label: str) -> None:
    if not isinstance(value, dict):
        raise RecoveryError(f"{label} metrics are missing")
    for key in ("mae", "rmse"):
        number = value.get(key)
        if not isinstance(number, (int, float)) or not math.isfinite(number):
            raise RecoveryError(f"{label} {key} is not finite")
    if value.get("evaluated_pairs") != 953 * 72:
        raise RecoveryError(f"{label} does not cover the fixed validation windows and links")


def _validate_scientific_run(
    manifest: Mapping[str, Any], *, seed: int, mode: str,
    protocol_id: str, expected_config: Mapping[str, Any],
    expected_dataset_bundle_identity: Mapping[str, str] | None = None,
    expected_adjacency_sha256: str | None = None,
) -> None:
    if (manifest.get("status") != "complete" or manifest.get("seed") != seed
            or manifest.get("adjacency_mode") != mode or manifest.get("protocol_id") != protocol_id):
        raise RecoveryError(f"Completed-run identity mismatch for seed={seed}, mode={mode}")
    if manifest.get("checkpoint") != "best.pt" or manifest.get("validation_predictions") != "validation_predictions.npy":
        raise RecoveryError("Run manifest does not bind the expected checkpoint and validation predictions")
    if manifest.get("checkpoint_reload_strict") is not True:
        raise RecoveryError("Run checkpoint was not strictly reloaded")
    if manifest.get("test_access", {}).get("permitted") is not False:
        raise RecoveryError("Run manifest does not explicitly deny test access")
    normalization = manifest.get("normalization")
    if not isinstance(normalization, dict):
        raise RecoveryError("Run normalization provenance is missing")
    fit_scope = normalization.get("fit_scope")
    if (not isinstance(fit_scope, str) or "train" not in fit_scope.casefold()
            or "validation" in fit_scope.casefold() or "test" in fit_scope.casefold()):
        raise RecoveryError("Run scaler does not prove train-only fitting")
    config = manifest.get("configuration")
    if not isinstance(config, dict):
        raise RecoveryError("Run configuration is missing")
    if set(config) - (set(expected_config) | CONFIG_METADATA_KEYS):
        raise RecoveryError("Run configuration contains unexpected fields")
    for key in SCIENCE_CONFIG_KEYS:
        if key not in expected_config or config.get(key) != expected_config[key]:
            raise RecoveryError(f"Scientific configuration mismatch: {key}")
    if config.get("seed") != seed or config.get("adjacency_mode") != mode or config.get("device") != "cpu":
        raise RecoveryError("Run-specific seed, adjacency, or device configuration mismatch")
    if (config.get("matrix_experiment_id") != "geant-h96-reproducibility-matrix-v3-drive-backed"
            or config.get("matrix_run_order") != ORIGINAL_MATRIX_ORDER):
        raise RecoveryError("Backup was not produced by the fixed original Matrix3 run plan")
    bundle_identity = config.get("colab_bundle_identity")
    if not isinstance(bundle_identity, dict) or set(bundle_identity) != {
            "source_snapshot_manifest_sha256", "source_bundle_sha256",
            "dataset_manifest_sha256", "dataset_bundle_sha256"}:
        raise RecoveryError("Backup lacks the original Matrix3 source/dataset bundle identity")
    if any(not isinstance(value, str) or len(value) != 64
           or any(ch not in "0123456789abcdef" for ch in value)
           for value in bundle_identity.values()):
        raise RecoveryError("Backup bundle identity contains an invalid digest")
    if expected_dataset_bundle_identity is not None:
        if any(bundle_identity.get(key) != expected_dataset_bundle_identity.get(key) for key in (
                "dataset_manifest_sha256", "dataset_bundle_sha256")):
            raise RecoveryError("Completed run was built from a different approved dataset bundle")
    metrics = manifest.get("validation_metrics", {})
    _finite_metric_pair(metrics.get("h96_candidate"), "h96 candidate")
    _finite_metric_pair(metrics.get("ewma_alpha_0.8"), "EWMA alpha 0.8")
    adjacency = manifest.get("adjacency", {})
    if adjacency.get("mode") != mode:
        raise RecoveryError("Recorded realized adjacency mode mismatch")
    adjacency_digest = adjacency.get("normalized_sha256")
    if (not isinstance(adjacency_digest, str) or len(adjacency_digest) != 64
            or any(ch not in "0123456789abcdef" for ch in adjacency_digest)):
        raise RecoveryError("Run manifest has no valid realized adjacency digest")
    if expected_adjacency_sha256 is not None and adjacency_digest != expected_adjacency_sha256:
        raise RecoveryError("Realized adjacency differs from the approved ordered-link graph")
    if mode == "shuffled" and adjacency.get("shuffle_seed") != 20261019:
        raise RecoveryError("Shuffled adjacency seed differs from the approved protocol")
    if mode != "shuffled" and adjacency.get("shuffle_seed") is not None:
        raise RecoveryError("Unexpected shuffle seed for non-shuffled topology")


def verify_completed_backup_set(
    runs_root: str | Path, *, protocol_id: str, expected_config: Mapping[str, Any],
    current_critical_source_hashes: Mapping[str, str],
    expected_dataset_metadata_sha256: str | None = None,
    expected_dataset_bundle_identity: Mapping[str, str] | None = None,
    expected_adjacency_hashes: Mapping[str, str] | None = None,
    expected_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify exactly the six declared completed backups without deserializing artifacts."""
    root = _canonical_directory(runs_root, "Original completed-backup runs root")
    expected_names = {f"h96_{mode}_seed_{seed}" for seed, mode in COMPLETED_RUNS}
    if {item.name for item in root.iterdir()} != expected_names:
        raise RecoveryError("Original Drive backup inventory is not exactly the six completed Matrix3 runs")
    reference_identity: dict[str, Any] | None = None
    reference_source_hashes: dict[str, str] | None = None
    verified: list[dict[str, Any]] = []
    for seed, mode in COMPLETED_RUNS:
        run_dir = root / f"h96_{mode}_seed_{seed}"
        if run_dir.is_symlink() or not run_dir.is_dir() or run_dir.resolve(strict=True) != run_dir:
            raise RecoveryError(f"Backup run path is missing or redirected: {run_dir.name}")
        if {entry.name for entry in run_dir.iterdir()} != RUN_FILES:
            raise RecoveryError(f"Backup artifact inventory mismatch: {run_dir.name}")
        backup_path = run_dir / "backup_manifest.json"
        run_path = run_dir / "run.json"
        backup = json.loads(backup_path.read_text(encoding="utf-8"))
        run = json.loads(run_path.read_text(encoding="utf-8"))
        if (backup.get("schema") != "geant-h96-matrix3-run-backup-v1"
                or backup.get("status") != "complete" or backup.get("readback_verified") is not True
                or backup.get("test_access") is not False or backup.get("seed") != seed
                or backup.get("adjacency_mode") != mode):
            raise RecoveryError(f"Drive backup manifest identity/readback mismatch: {run_dir.name}")
        run_digest = sha256_file(run_path)
        if backup.get("source_run_manifest_sha256") != run_digest:
            raise RecoveryError(f"Backup manifest does not bind run.json: {run_dir.name}")
        entries = backup.get("artifacts")
        if not isinstance(entries, dict) or set(entries) != {"run.json", "best.pt", "validation_predictions.npy"}:
            raise RecoveryError(f"Backup artifact hash inventory mismatch: {run_dir.name}")
        expected_hashes = {
            "run.json": run_digest,
            "best.pt": run.get("checkpoint_sha256"),
            "validation_predictions.npy": run.get("validation_predictions_sha256"),
        }
        for name, expected_hash in expected_hashes.items():
            entry = entries.get(name, {})
            path = run_dir / name
            if (not isinstance(expected_hash, str) or len(expected_hash) != 64
                    or entry.get("sha256") != expected_hash
                    or not path.is_file() or path.stat().st_size <= 0
                    or entry.get("size_bytes") != path.stat().st_size
                    or sha256_file(path) != expected_hash):
                raise RecoveryError(f"Backup byte hash/size mismatch: {run_dir.name}/{name}")
        _validate_scientific_run(run, seed=seed, mode=mode,
                                 protocol_id=protocol_id, expected_config=expected_config,
                                 expected_dataset_bundle_identity=expected_dataset_bundle_identity,
                                 expected_adjacency_sha256=(expected_adjacency_hashes or {}).get(mode))
        source_hashes = run.get("source_hashes")
        if not isinstance(source_hashes, dict):
            raise RecoveryError(f"Run source-hash inventory missing: {run_dir.name}")
        for source_name, digest in current_critical_source_hashes.items():
            if source_hashes.get(source_name) != digest:
                raise RecoveryError(f"Critical training source differs from the completed run: {source_name}")
        identity = {key: run.get(key) for key in IDENTITY_KEYS if key != "source_hashes"}
        if reference_identity is None:
            reference_identity, reference_source_hashes = identity, source_hashes
        elif identity != reference_identity or source_hashes != reference_source_hashes:
            raise RecoveryError("Completed runs do not share identical data/provenance/source identities")
        if expected_dataset_metadata_sha256 is not None and run.get("dataset_metadata_sha256") != expected_dataset_metadata_sha256:
            raise RecoveryError("Completed runs do not match the currently approved dataset metadata")
        if expected_provenance is not None:
            prior_provenance = dict(run.get("provenance", {}))
            current_provenance = dict(expected_provenance)
            # A Colab session path is ephemeral; all content/provenance identity
            # fields remain exact, while this single location field is compared
            # only for consistency across the six original backups above.
            prior_provenance.pop("data_root", None)
            current_provenance.pop("data_root", None)
            if prior_provenance != current_provenance:
                raise RecoveryError("Completed runs do not match the currently approved dataset provenance")
        candidate = run["validation_metrics"]["h96_candidate"]
        verified.append({
            "seed": seed, "adjacency_mode": mode, "status": "complete",
            "epochs_completed": run.get("epochs_completed"), "best_epoch": run.get("best_epoch"),
            "validation_mae": candidate["mae"], "validation_rmse": candidate["rmse"],
            "run_manifest_sha256": run_digest,
            "checkpoint_sha256": run["checkpoint_sha256"],
            "validation_predictions_sha256": run["validation_predictions_sha256"],
            "backup_manifest_sha256": sha256_file(backup_path),
        })
    return {
        "status": "six_completed_backups_verified", "runs_root": str(root),
        "completed_runs": verified, "remaining_runs": [
            {"seed": seed, "adjacency_mode": mode} for seed, mode in REMAINING_RUNS
        ], "test_artifacts_opened": False,
    }


def _write_exclusive_json(path: Path, payload: Mapping[str, Any]) -> str:
    raw = json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n"
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    if path.read_bytes() != raw:
        raise RecoveryError(f"JSON artifact read-back mismatch: {path.name}")
    return hashlib.sha256(raw).hexdigest()


def run_remaining_experiments(
    *, output_root: str | Path, persistent_root: str | Path,
    prior_verified: Mapping[str, Any], authorize: Callable[[int, int, str], bool],
    train_one: Callable[[int, str, Path], Mapping[str, Any]],
    verify_result: Callable[[Mapping[str, Any], int, str, Path], Mapping[str, Any]],
    backup_run: Callable[[Mapping[str, Any], int, str, Path], Mapping[str, Any]],
    persist_snapshot: Callable[[int, str, Mapping[str, Any]], Mapping[str, str]],
) -> dict[str, Any]:
    """Run only shuffled seeds 123/2026; no resume and backup-before-next semantics."""
    if prior_verified.get("status") != "six_completed_backups_verified":
        raise RecoveryError("The six original Matrix3 backups must verify before recovery")
    output = Path(os.path.abspath(output_root))
    persistent = Path(os.path.abspath(persistent_root))
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing existing recovery output; no resume/overwrite: {output}")
    if persistent.exists() or persistent.is_symlink():
        raise FileExistsError(f"Refusing existing Drive recovery destination; no resume/overwrite: {persistent}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.parent.resolve(strict=True) != output.parent:
        raise RecoveryError("Recovery output parent is redirected")
    output.mkdir(exist_ok=False)
    if output.resolve(strict=True) != output:
        raise RecoveryError("Recovery output identity changed after creation")
    persistent.parent.mkdir(parents=True, exist_ok=True)
    persistent.mkdir(exist_ok=False)
    if persistent.resolve(strict=True) != persistent:
        raise RecoveryError("Drive recovery destination identity changed after creation")
    record: dict[str, Any] = {
        "recovery_id": RECOVERY_ID, "status": "incomplete", "failure": None,
        "output_root": str(output), "persistent_root": str(persistent),
        "run_order": [{"seed": s, "adjacency_mode": m} for s, m in REMAINING_RUNS],
        "completed_prior_runs": list(prior_verified["completed_runs"]),
        "completed_runs": [], "test_access": False,
    }
    persist_snapshot(0, "incomplete", record)
    try:
        for index, (seed, mode) in enumerate(REMAINING_RUNS, start=1):
            destination = output / f"h96_{mode}_seed_{seed}"
            if destination.exists() or destination.is_symlink():
                raise FileExistsError(f"Refusing existing remaining run destination: {destination}")
            if authorize(index, seed, mode) is not True:
                raise PermissionError(f"Run {index}/2 was not explicitly authorized")
            record["current_attempt"] = {"index": index, "seed": seed,
                                         "adjacency_mode": mode, "status": "authorized_in_progress"}
            persist_snapshot(index, "in_progress", record)
            result = train_one(seed, mode, destination)
            artifact_ids = verify_result(result, seed, mode, destination)
            backup = backup_run(result, seed, mode, destination)
            if backup.get("verified") is not True:
                raise RecoveryError("Persistent backup failed read-back verification; next run prohibited")
            for filename, key in (("run.json", "run_manifest_sha256"),
                                  ("best.pt", "checkpoint_sha256"),
                                  ("validation_predictions.npy", "validation_predictions_sha256")):
                if backup.get("artifacts", {}).get(filename, {}).get("sha256") != artifact_ids.get(key):
                    raise RecoveryError(f"Persistent copy does not match verified {filename}")
            row = {
                "seed": seed, "adjacency_mode": mode, "status": "complete",
                "epochs_completed": result.get("epochs_completed"),
                "best_epoch": result.get("best_epoch"),
                "validation_metrics": result.get("validation_metrics"),
                "runtime_seconds": result.get("runtime_seconds"),
                **dict(artifact_ids), "persistent_backup": dict(backup),
            }
            record["completed_runs"].append(row)
            record["current_attempt"] = None
            persist_snapshot(index, "registered", record)
        if len(record["completed_runs"]) != 2:
            raise RecoveryError("Recovery cannot finalize before both remaining runs are registered")
        summary = {
            "schema": "geant-h96-matrix3-recovery-results-v1", "status": "complete",
            "recovery_id": RECOVERY_ID,
            "runs": list(record["completed_prior_runs"]) + list(record["completed_runs"]),
            "test_access": False,
        }
        summary_path = persistent / "combined_results.json"
        summary_hash = _write_exclusive_json(summary_path, summary)
        if sha256_file(summary_path) != summary_hash:
            raise RecoveryError("Combined summary failed final hash verification")
        record.update({"status": "complete", "combined_results": str(summary_path),
                       "combined_results_sha256": summary_hash})
        persist_snapshot(3, "complete", record)
        return record
    except BaseException as exc:
        record.update({"status": "failed", "failure": f"{type(exc).__name__}: {exc}"})
        try:
            persist_snapshot(len(record["completed_runs"]) + 1, "failed", record)
        except BaseException:
            pass
        raise

