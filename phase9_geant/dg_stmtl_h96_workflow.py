"""Integrity checks and persistent-copy helper for DG-STMTL h=96 Colab runs.

This module never reads arrays, predictions, or checkpoints into a model. It
validates completed run files as opaque bytes and refuses overwrite/resume.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any


PROTOCOL_ID = "geant-dg-stmtl-based-h96-v1"
SEEDS = (42, 123, 2026)
PAIR_COUNT = 953 * 72
ARTIFACTS = ("run.json", "best.pt", "validation_predictions.npy")


class DGWorkflowError(RuntimeError):
    """A run or persistent backup failed integrity validation."""


def authorization_phrase(seed: int) -> str:
    if seed not in SEEDS:
        raise ValueError("seed is outside the fixed DG-STMTL seed list")
    return f"AUTHORIZE DG-STMTL H96 SEED {seed} TOPOLOGY"


def require_authorization(seed: int, response: str) -> None:
    if not isinstance(response, str) or response.strip() != authorization_phrase(seed):
        raise PermissionError(f"Explicit per-run authorization required: {authorization_phrase(seed)}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_directory(path: str | Path, label: str) -> Path:
    result = Path(os.path.abspath(path))
    if result.is_symlink() or not result.is_dir() or result.resolve(strict=True) != result:
        raise DGWorkflowError(f"{label} must be an existing canonical directory: {result}")
    return result


def validate_completed_run(run_directory: str | Path, *, expected_seed: int) -> dict[str, Any]:
    run = _canonical_directory(run_directory, "Run directory")
    if expected_seed not in SEEDS or run.name != f"dg_stmtl_topology_seed_{expected_seed}":
        raise DGWorkflowError("Run directory name does not match the expected seed")
    if {item.name for item in run.iterdir()} != set(ARTIFACTS):
        raise DGWorkflowError("Completed run artifact inventory is not exact")
    manifest_path = run / "run.json"
    if manifest_path.is_symlink() or not manifest_path.is_file() or manifest_path.resolve(strict=True) != manifest_path:
        raise DGWorkflowError("Run manifest is missing or redirected")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_fields = {
        "status": "complete", "protocol_id": PROTOCOL_ID, "seed": expected_seed,
        "adjacency_mode": "hybrid_topology_dynamic", "checkpoint": "best.pt",
        "validation_predictions": "validation_predictions.npy", "checkpoint_reload_strict": True,
    }
    if any(manifest.get(key) != value for key, value in expected_fields.items()):
        raise DGWorkflowError("Run manifest does not match the completed DG-STMTL protocol")
    if manifest.get("test_access", {}).get("permitted") is not False:
        raise DGWorkflowError("Run manifest does not affirm test-split denial")
    metrics = manifest.get("validation_metrics", {})
    for name in ("dg_stmtl_based", "ewma_alpha_0.8"):
        values = metrics.get(name)
        if not isinstance(values, dict) or values.get("evaluated_pairs") != PAIR_COUNT:
            raise DGWorkflowError(f"Missing or misaligned validation metrics: {name}")
        for field in ("mae", "rmse"):
            value = values.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not (value >= 0 and value < float("inf")):
                raise DGWorkflowError(f"Invalid metric {name}.{field}")
    if manifest.get("epochs_completed", 0) <= 0 or manifest.get("best_epoch", 0) <= 0:
        raise DGWorkflowError("Completed run has no completed epoch or selected checkpoint")
    for filename, key in (("best.pt", "checkpoint_sha256"),
                          ("validation_predictions.npy", "validation_predictions_sha256")):
        artifact = run / filename
        if artifact.is_symlink() or not artifact.is_file() or artifact.stat().st_size <= 0:
            raise DGWorkflowError(f"Required artifact is missing, empty, or redirected: {filename}")
        digest = manifest.get(key)
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest) or _sha256(artifact) != digest:
            raise DGWorkflowError(f"Artifact SHA-256 mismatch: {filename}")
    manifest_digest = _sha256(manifest_path)
    return {"manifest": manifest, "manifest_sha256": manifest_digest,
            "checkpoint_sha256": manifest["checkpoint_sha256"],
            "validation_predictions_sha256": manifest["validation_predictions_sha256"]}


def backup_completed_run(
    run_directory: str | Path,
    persistent_root: str | Path,
    *,
    expected_seed: int,
) -> dict[str, Any]:
    """Copy one verified run into a new persistent child and verify read-back.

    On a copy/verification failure, preserve the partial destination. Retrying
    into it is forbidden and requires a separately reviewed recovery decision.
    """
    checked = validate_completed_run(run_directory, expected_seed=expected_seed)
    source = Path(os.path.abspath(run_directory))
    store = _canonical_directory(persistent_root, "Persistent Drive root")
    runs = store / "runs"
    if runs.exists() or runs.is_symlink():
        runs = _canonical_directory(runs, "Persistent runs directory")
    else:
        runs.mkdir(exist_ok=False)
        if runs.is_symlink() or runs.resolve(strict=True) != runs:
            raise DGWorkflowError("Created persistent runs directory failed identity check")
    name = f"dg_stmtl_topology_seed_{expected_seed}"
    destination = runs / name
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing existing persistent run backup: {destination}")
    destination.mkdir(exist_ok=False)
    if destination.is_symlink() or destination.resolve(strict=True) != destination:
        raise DGWorkflowError("Persistent destination failed post-creation check")
    copied: dict[str, dict[str, Any]] = {}
    expected_hashes = {
        "run.json": checked["manifest_sha256"],
        "best.pt": checked["checkpoint_sha256"],
        "validation_predictions.npy": checked["validation_predictions_sha256"],
    }
    for filename in ARTIFACTS:
        source_path, target_path = source / filename, destination / filename
        digest = hashlib.sha256()
        count = 0
        with source_path.open("rb") as input_stream, target_path.open("xb") as output_stream:
            for block in iter(lambda: input_stream.read(1024 * 1024), b""):
                output_stream.write(block)
                digest.update(block)
                count += len(block)
            output_stream.flush()
            os.fsync(output_stream.fileno())
        actual = digest.hexdigest()
        if actual != expected_hashes[filename] or _sha256(target_path) != actual:
            raise DGWorkflowError(f"Persistent read-back verification failed: {filename}")
        copied[filename] = {"sha256": actual, "size_bytes": count}
    backup = {
        "schema": "geant-dg-stmtl-h96-backup-v1", "status": "complete",
        "protocol_id": PROTOCOL_ID, "seed": expected_seed,
        "adjacency_mode": "hybrid_topology_dynamic", "artifacts": copied,
        "readback_verified": True, "test_access": False,
    }
    backup_path = destination / "backup_manifest.json"
    raw = json.dumps(backup, sort_keys=True, indent=2).encode("utf-8") + b"\n"
    with backup_path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    if backup_path.read_bytes() != raw or {item.name for item in destination.iterdir()} != set(ARTIFACTS) | {"backup_manifest.json"}:
        raise DGWorkflowError("Persistent backup manifest/inventory verification failed")
    return {"verified": True, "directory": str(destination),
            "backup_manifest_sha256": hashlib.sha256(raw).hexdigest(), "artifacts": copied}


__all__ = ["ARTIFACTS", "DGWorkflowError", "authorization_phrase", "backup_completed_run",
           "require_authorization", "validate_completed_run"]
