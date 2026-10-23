"""Small synthetic-testable helpers for the versioned h96 Colab matrix notebook.

This module deliberately does not import the trainer or open dataset arrays.
The notebook verifies the bundle as opaque bytes before importing this module.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


class WorkflowError(RuntimeError):
    """The staged Colab inputs or exported run do not match the frozen plan."""


def _require_fresh_directory(path: str | Path, *, label: str) -> Path:
    """Require a canonical existing directory that has not been used before."""
    target = Path(os.path.abspath(path))
    if target.is_symlink() or not target.is_dir() or target.resolve(strict=True) != target:
        raise WorkflowError(f"{label} must be an existing canonical directory: {target}")
    return target


def create_persistent_matrix_root(path: str | Path) -> Path:
    """Reserve a new Drive-backed matrix namespace; never reuse partial state."""
    target = Path(os.path.abspath(path))
    parent = target.parent
    if (not parent.is_dir() or parent.resolve(strict=True) != parent
            or target.exists() or target.is_symlink()):
        raise WorkflowError(f"Persistent matrix root is unavailable, redirected, or already exists: {target}")
    target.mkdir(exist_ok=False)
    if target.is_symlink() or not target.is_dir() or target.resolve(strict=True) != target:
        raise WorkflowError(f"Persistent matrix root failed post-creation identity check: {target}")
    return target


def validate_persistent_matrix_target(path: str | Path) -> Path:
    """Preflight a Drive-backed destination without creating it."""
    target = Path(os.path.abspath(path))
    parent = target.parent
    if (not parent.is_dir() or parent.resolve(strict=True) != parent
            or target.exists() or target.is_symlink()):
        raise WorkflowError(f"Persistent matrix target is unavailable, redirected, or already exists: {target}")
    return target


def backup_verified_run(
    source_run: str | Path,
    persistent_root: str | Path,
    *,
    run_name: str,
    expected_seed: int,
    expected_adjacency_mode: str,
    expected_checkpoint_sha256: str,
    expected_predictions_sha256: str,
    expected_run_manifest_sha256: str,
) -> dict[str, Any]:
    """Copy a completed run to a fresh persistent directory and read-verify it.

    The source checkpoint and prediction are treated as opaque bytes. A partial
    destination is deliberately retained and blocks reuse; callers must stop.
    """
    if not re.fullmatch(r"h96_(?:topology|identity|shuffled)_seed_(?:42|123|2026)", run_name):
        raise WorkflowError("Persistent run name is outside the fixed matrix allowlist")
    source = _require_fresh_directory(source_run, label="Completed local run")
    store = _require_fresh_directory(persistent_root, label="Persistent Drive root")
    expected = {"run.json", "best.pt", "validation_predictions.npy"}
    if {item.name for item in source.iterdir()} != expected:
        raise WorkflowError("Completed local run inventory changed before backup")
    source_manifest = source / "run.json"
    if _sha256_path(source_manifest) != expected_run_manifest_sha256:
        raise WorkflowError("Run manifest changed after local artifact verification")
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    if (manifest.get("status") != "complete" or manifest.get("seed") != expected_seed
            or manifest.get("adjacency_mode") != expected_adjacency_mode
            or manifest.get("checkpoint_sha256") != expected_checkpoint_sha256
            or manifest.get("validation_predictions_sha256") != expected_predictions_sha256):
        raise WorkflowError("Run manifest identity or artifact digest changed before backup")
    for filename, digest in (("best.pt", expected_checkpoint_sha256),
                             ("validation_predictions.npy", expected_predictions_sha256)):
        if not digest or _sha256_path(source / filename) != digest:
            raise WorkflowError(f"Source artifact changed before persistent backup: {filename}")
    runs_root = store / "runs"
    if runs_root.exists() or runs_root.is_symlink():
        if runs_root.is_symlink() or not runs_root.is_dir() or runs_root.resolve(strict=True) != runs_root:
            raise WorkflowError("Persistent runs directory is redirected")
    else:
        runs_root.mkdir(exist_ok=False)
    destination = runs_root / run_name
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing existing persistent run backup; no overwrite/resume: {destination}")
    destination.mkdir(exist_ok=False)
    if destination.resolve(strict=True) != destination or destination.is_symlink():
        raise WorkflowError("Persistent run backup destination failed containment check")
    copied: dict[str, dict[str, Any]] = {}
    # Exclusive creation leaves a diagnosable partial backup on interruption.
    for filename in ("run.json", "best.pt", "validation_predictions.npy"):
        source_path = source / filename
        target_path = destination / filename
        digest = hashlib.sha256()
        size = 0
        with source_path.open("rb") as input_stream, target_path.open("xb") as output_stream:
            for block in iter(lambda: input_stream.read(1024 * 1024), b""):
                output_stream.write(block)
                digest.update(block)
                size += len(block)
            output_stream.flush()
            os.fsync(output_stream.fileno())
        actual = digest.hexdigest()
        if actual != _sha256_path(target_path):
            raise WorkflowError(f"Persistent read-back hash mismatch: {filename}")
        expected_digest = (expected_run_manifest_sha256 if filename == "run.json" else
                           expected_checkpoint_sha256 if filename == "best.pt" else
                           expected_predictions_sha256)
        if actual != expected_digest:
            raise WorkflowError(f"Persistent artifact does not match verified source: {filename}")
        copied[filename] = {"sha256": actual, "size_bytes": size}
    backup_manifest = {
        "schema": "geant-h96-matrix3-run-backup-v1",
        "status": "complete",
        "seed": expected_seed,
        "adjacency_mode": expected_adjacency_mode,
        "source_run_manifest_sha256": expected_run_manifest_sha256,
        "artifacts": copied,
        "readback_verified": True,
        "test_access": False,
    }
    manifest_path = destination / "backup_manifest.json"
    raw = json.dumps(backup_manifest, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    with manifest_path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    if manifest_path.read_bytes() != raw:
        raise WorkflowError("Persistent backup manifest failed read-back verification")
    if {item.name for item in destination.iterdir()} != expected | {"backup_manifest.json"}:
        raise WorkflowError("Persistent backup inventory is not exact")
    for filename, entry in copied.items():
        if _sha256_path(destination / filename) != entry["sha256"]:
            raise WorkflowError(f"Persistent final verification failed: {filename}")
    return {
        "verified": True,
        "directory": str(destination),
        "backup_manifest_sha256": sha256_bytes(raw),
        "artifacts": copied,
    }


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def persist_lifecycle_snapshot(
    persistent_root: str | Path, *, sequence: int, state: str,
    record: Mapping[str, Any],
) -> dict[str, str]:
    """Persist an immutable lifecycle snapshot and verify exact read-back bytes."""
    if not isinstance(sequence, int) or sequence < 0 or state not in {
            "incomplete", "in_progress", "registered", "failed", "complete"}:
        raise WorkflowError("Lifecycle snapshot sequence must be a nonnegative integer")
    store = _require_fresh_directory(persistent_root, label="Persistent Drive root")
    snapshots = store / "matrix_lifecycle"
    if snapshots.exists() or snapshots.is_symlink():
        if snapshots.is_symlink() or not snapshots.is_dir() or snapshots.resolve(strict=True) != snapshots:
            raise WorkflowError("Persistent lifecycle directory is redirected")
    else:
        snapshots.mkdir(exist_ok=False)
    target = snapshots / f"matrix_run_{sequence:03d}_{state}.json"
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Refusing existing persistent lifecycle snapshot: {target}")
    raw = json.dumps(dict(record), indent=2, sort_keys=True).encode("utf-8") + b"\n"
    with target.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    digest = sha256_bytes(raw)
    if not target.is_file() or target.read_bytes() != raw or _sha256_path(target) != digest:
        raise WorkflowError("Persistent lifecycle snapshot read-back verification failed")
    return {"path": str(target), "sha256": digest}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def identify_uploaded_files(
    uploaded: Mapping[str, bytes], expected_hashes: Mapping[str, str],
) -> dict[str, bytes]:
    """Identify uploaded files by exact digest, not mutable browser filenames."""
    if set(expected_hashes) != {
        "geant_h96_colab_source_v1.zip", "source_snapshot_manifest.json",
        "geant_h96_colab_trainval_v1.zip", "dataset_bundle_manifest.json",
    }:
        raise WorkflowError("Expected-file hash allowlist must contain exactly the four bundle files")
    by_hash: dict[str, list[bytes]] = {}
    for _display_name, payload in uploaded.items():
        if not isinstance(payload, bytes):
            raise WorkflowError("Upload payload is not bytes")
        by_hash.setdefault(sha256_bytes(payload), []).append(payload)
    resolved: dict[str, bytes] = {}
    for name, digest in expected_hashes.items():
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise WorkflowError(f"Expected digest must be lowercase SHA-256: {name}")
        matches = by_hash.get(digest, [])
        if len(matches) != 1:
            raise WorkflowError(f"Expected exactly one upload with approved digest for {name}; found {len(matches)}")
        resolved[name] = matches[0]
    if sum(len(items) for items in by_hash.values()) != 4:
        raise WorkflowError("Exactly four uploaded files are required; extras/duplicates are refused")
    return resolved


def stage_verified_uploads(
    uploaded: Mapping[str, bytes], expected_hashes: Mapping[str, str], destination: str | Path,
) -> Path:
    """Create a new incoming folder without replacing any pre-existing file."""
    target = Path(os.path.abspath(destination))
    resolved = identify_uploaded_files(uploaded, expected_hashes)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Refusing existing incoming bundle directory: {target}")
    target.mkdir(parents=True, exist_ok=False)
    try:
        for name, payload in resolved.items():
            output = target / name
            with output.open("xb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            if sha256_bytes(output.read_bytes()) != expected_hashes[name]:
                raise WorkflowError(f"Staged upload digest changed during write: {name}")
    except BaseException:
        # Preserve the partial folder for diagnosis; never overwrite on retry.
        raise
    return target


def verify_archive_payload(
    payload: bytes, manifest_payload: bytes, *, expected_archive_sha256: str,
    expected_manifest_sha256: str, kind: str,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    """Verify archive/member bytes, traversal, duplicate names and symlinks."""
    if sha256_bytes(payload) != expected_archive_sha256:
        raise WorkflowError(f"{kind} archive SHA-256 mismatch")
    if sha256_bytes(manifest_payload) != expected_manifest_sha256:
        raise WorkflowError(f"{kind} manifest SHA-256 mismatch")
    manifest = json.loads(manifest_payload.decode("utf-8"))
    if manifest.get("archive_sha256") != expected_archive_sha256:
        raise WorkflowError(f"{kind} manifest does not bind the expected archive")
    entries_key = "entries" if kind == "source" else "artifacts" if kind == "data" else None
    if entries_key is None or not isinstance(manifest.get(entries_key), list):
        raise WorkflowError(f"Invalid bundle kind/schema: {kind}")
    entries = manifest[entries_key]
    by_name: dict[str, dict[str, Any]] = {}
    folded: set[str] = set()
    for entry in entries:
        name = entry.get("path")
        if not isinstance(name, str) or "\\" in name:
            raise WorkflowError(f"Unsafe archive member path: {name!r}")
        path = PurePosixPath(name)
        if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts) or ":" in path.parts[0]:
            raise WorkflowError(f"Unsafe archive member path: {name!r}")
        if name.casefold() in folded:
            raise WorkflowError(f"Duplicate/case-conflicting manifest path: {name}")
        folded.add(name.casefold())
        digest, size = entry.get("sha256"), entry.get("size_bytes")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest) or not isinstance(size, int) or size < 0:
            raise WorkflowError(f"Malformed archive entry: {name}")
        by_name[name] = entry
    if kind == "data":
        approved = {
            "metadata.json", "timestamps_train.npy", "timestamps_validation.npy",
            "utilization_train.npy", "utilization_validation.npy", "valid_train.npy",
            "valid_validation.npy", "window_index_train_h96.json", "window_index_validation_h96.json",
        }
        if set(by_name) != approved:
            raise WorkflowError("Dataset archive manifest is not the exact nine-file train/validation allowlist")
        if manifest.get("exact_archive_members") != list(manifest.get("exact_archive_members", [])) or set(manifest.get("exact_archive_members", [])) != approved:
            raise WorkflowError("Dataset manifest exact-member list is malformed")
        if manifest.get("source_scope") != {"train": [0, 8020], "validation": [8020, 9740]}:
            raise WorkflowError("Dataset source split identity mismatch")
        topology = manifest.get("topology", {})
        if (topology.get("node_count"), topology.get("undirected_link_count"), topology.get("directed_link_count")) != (22, 36, 72):
            raise WorkflowError("Dataset topology structure mismatch")
        if (topology.get("local_topology_sha256_identity_only")
                != "38c90629f30af70fe840e33071b12cc40498bb910e8dbeb2fa7863a7b645d121"
                or not re.fullmatch(r"[0-9a-f]{64}", str(topology.get("ordered_edge_sha256", "")))):
            raise WorkflowError("Dataset topology local identity or ordered-edge digest is missing/mismatched")
    else:
        internal_name = manifest.get("internal_manifest_member")
        if not isinstance(internal_name, str) or internal_name != "PORTABLE_SOURCE_MANIFEST.json":
            raise WorkflowError("Source manifest internal-member identity mismatch")
    found: dict[str, bytes] = {}
    folded_zip: set[str] = set()
    import io
    try:
        with zipfile.ZipFile(io.BytesIO(payload), "r") as archive:
            for info in archive.infolist():
                name = info.filename
                path = PurePosixPath(name)
                if (path.is_absolute() or "\\" in name or not path.parts
                        or any(part in {"", ".", ".."} for part in path.parts) or ":" in path.parts[0]):
                    raise WorkflowError(f"Unsafe ZIP member: {name!r}")
                mode = (info.external_attr >> 16) & 0xFFFF
                import stat
                if stat.S_ISLNK(mode) or info.is_dir():
                    raise WorkflowError(f"Directory/symlink ZIP member refused: {name}")
                if name.casefold() in folded_zip:
                    raise WorkflowError(f"Duplicate/case-conflicting ZIP member: {name}")
                folded_zip.add(name.casefold())
                raw = archive.read(info)
                found[name] = raw
    except zipfile.BadZipFile as exc:
        raise WorkflowError(f"Invalid {kind} ZIP: {exc}") from exc
    expected_names = set(by_name)
    if kind == "source":
        expected_names.add("PORTABLE_SOURCE_MANIFEST.json")
    if set(found) != expected_names:
        raise WorkflowError(f"{kind} ZIP inventory mismatch: missing={sorted(expected_names-set(found))}, extra={sorted(set(found)-expected_names)}")
    for name, entry in by_name.items():
        raw = found[name]
        if len(raw) != entry["size_bytes"] or sha256_bytes(raw) != entry["sha256"]:
            raise WorkflowError(f"{kind} member hash/size mismatch: {name}")
    if kind == "source":
        internal = json.loads(found["PORTABLE_SOURCE_MANIFEST.json"].decode("utf-8"))
        if (internal.get("entries") != manifest.get("entries")
                or internal.get("first_party_import_inventory") != manifest.get("first_party_import_inventory")):
            raise WorkflowError("Source internal/external manifests disagree")
    return manifest, found


def safe_extract_members(members: Mapping[str, bytes], destination: str | Path) -> Path:
    """Extract already-verified member bytes into a fresh, non-overwriting root."""
    root = Path(os.path.abspath(destination))
    if root.exists() or root.is_symlink():
        raise FileExistsError(f"Refusing existing extraction root: {root}")
    root.mkdir(parents=True, exist_ok=False)
    for name, payload in sorted(members.items()):
        relative = PurePosixPath(name)
        if relative.is_absolute() or "\\" in name or any(part in {"", ".", ".."} for part in relative.parts):
            raise WorkflowError(f"Unsafe extraction path: {name}")
        output = root.joinpath(*relative.parts)
        output.parent.mkdir(parents=True, exist_ok=True)
        if root.resolve(strict=True) not in output.parent.resolve(strict=True).parents and output.parent.resolve(strict=True) != root.resolve(strict=True):
            raise WorkflowError(f"Extraction escaped approved root: {name}")
        with output.open("xb") as stream:
            stream.write(payload)
    return root


def run_matrix_with_prompt_bridge(
    command: list[str], *, prompt_fn: Any, popen_factory: Any = subprocess.Popen,
    output_fn: Any = print,
) -> int:
    """Run the matrix process sequentially while asking separately per run."""
    process = popen_factory(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert process.stdout is not None and process.stdin is not None
    for line in process.stdout:
        output_fn(line.rstrip("\n"))
        if line.startswith("AUTH_REQUIRED: type exactly "):
            phrase = line.strip().removeprefix("AUTH_REQUIRED: type exactly ")
            answer = prompt_fn(phrase)
            process.stdin.write(answer + "\n")
            process.stdin.flush()
    return process.wait()


def export_matrix_results(
    output_root: str | Path, export_zip: str | Path, *, matrix_manifest_path: str | Path,
) -> Path:
    """Export run artifacts plus the explicitly named sibling matrix sidecar."""
    root = Path(output_root).resolve(strict=True)
    control = Path(matrix_manifest_path)
    expected_control = root.parent / f"{root.name}.matrix_run.json"
    if (control != expected_control or control.is_symlink() or not control.is_file()
            or control.resolve(strict=True) != control):
        raise WorkflowError("Matrix lifecycle sidecar is not the canonical sibling regular file")
    target = Path(os.path.abspath(export_zip))
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Refusing existing matrix export: {target}")
    allowed_names = {"matrix_run.json", "run.json", "best.pt", "validation_predictions.npy"}
    files = [path for path in root.rglob("*") if path.is_file() and path.name in allowed_names]
    if not files:
        raise WorkflowError("Matrix manifest is missing; refusing to export an unidentified run")
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(control, "matrix_run.json")
        for path in sorted(files):
            archive.write(path, path.relative_to(root).as_posix())
    return target
