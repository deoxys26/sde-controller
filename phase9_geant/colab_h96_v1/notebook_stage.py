"""Synthetic-testable, non-overwriting staging for the Matrix3 recovery notebook."""

from __future__ import annotations

import hashlib
import os
import re
import uuid
from pathlib import Path
from typing import Mapping, MutableMapping, MutableSequence


UPLOAD_NAMES = frozenset({
    "geant_h96_colab_source_v1.zip",
    "source_snapshot_manifest.json",
    "geant_h96_colab_trainval_v1.zip",
    "dataset_bundle_manifest.json",
})
SESSION_DIR_PREFIX = "geant_h96_matrix3_recovery_review18_"


class StagingError(RuntimeError):
    """An upload or fresh staging session failed closed."""


def create_fresh_preparation_session(
    base_directory: str | Path,
    payloads: Mapping[str, bytes],
    expected_hashes: Mapping[str, str],
) -> tuple[str, Path, Path]:
    """Create a new staging session and write exactly the four approved inputs.

    This function never reuses a prior session. On any post-creation error it
    intentionally leaves the partial session in place for diagnosis; a rerun
    obtains a different UUID and a new path.
    """
    base = Path(os.path.abspath(base_directory))
    if (not base.is_dir() or base.is_symlink()
            or base.resolve(strict=True) != base):
        raise StagingError(f"Staging base must be an existing canonical directory: {base}")
    if set(payloads) != UPLOAD_NAMES or set(expected_hashes) != UPLOAD_NAMES:
        raise StagingError("Exactly the four approved source/dataset ZIP and manifest files are required")
    for name in sorted(UPLOAD_NAMES):
        payload, digest = payloads[name], expected_hashes[name]
        if not isinstance(payload, bytes):
            raise StagingError(f"Upload payload is not bytes: {name}")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise StagingError(f"Expected digest is not lowercase SHA-256: {name}")
        if hashlib.sha256(payload).hexdigest() != digest:
            raise StagingError(f"Upload digest mismatch: {name}")

    session_id = uuid.uuid4().hex
    if not re.fullmatch(r"[0-9a-f]{32}", session_id):
        raise StagingError("UUID provider returned an invalid session identifier")
    session = base / f"{SESSION_DIR_PREFIX}{session_id}"
    session.mkdir(exist_ok=False)
    if session.is_symlink() or not session.is_dir() or session.resolve(strict=True) != session:
        raise StagingError(f"New staging session failed identity check: {session}")
    incoming = session / "incoming"
    incoming.mkdir(exist_ok=False)
    if incoming.is_symlink() or incoming.resolve(strict=True) != incoming:
        raise StagingError(f"New incoming directory failed identity check: {incoming}")
    for name in sorted(UPLOAD_NAMES):
        target = incoming / name
        with target.open("xb") as stream:
            stream.write(payloads[name])
            stream.flush()
            os.fsync(stream.fileno())
    if {path.name for path in incoming.iterdir()} != UPLOAD_NAMES:
        raise StagingError("Staged upload inventory is not exact")
    for name in UPLOAD_NAMES:
        if hashlib.sha256((incoming / name).read_bytes()).hexdigest() != expected_hashes[name]:
            raise StagingError(f"Staged upload read-back digest mismatch: {name}")
    return session_id, session, incoming


def reset_prior_staged_project_imports(
    search_path: MutableSequence[str],
    modules: MutableMapping[str, object],
    base_directory: str | Path,
) -> tuple[str, ...]:
    """Remove only first-party imports from prior Matrix3 staging sessions.

    If a first-party project module came from anywhere else, refuse rather than
    silently shadowing or deleting an unrelated import. This is a notebook
    retry hygiene check, not a security boundary against arbitrary kernel code.
    """
    base = Path(os.path.abspath(base_directory))
    prefix = base / "geant_h96_matrix3_recovery_"

    def session_path(path_value: object) -> bool:
        try:
            path = Path(os.path.abspath(os.fspath(path_value)))
            common = Path(os.path.commonpath((str(base), str(path))))
            if common != base or path == base:
                return False
            relative = path.relative_to(base)
            return bool(relative.parts and relative.parts[0].startswith(prefix.name))
        except (TypeError, ValueError, OSError):
            return False

    project_names = [name for name in tuple(modules)
                     if name in {"phase9_geant", "src"}
                     or name.startswith("phase9_geant.") or name.startswith("src.")]
    stale: list[str] = []
    for name in project_names:
        module = modules[name]
        locations: list[object] = []
        file_path = getattr(module, "__file__", None)
        if file_path:
            locations.append(file_path)
        locations.extend(getattr(module, "__path__", ()) or ())
        if locations and all(session_path(location) for location in locations):
            stale.append(name)
        else:
            raise StagingError(f"Refusing unexpected preloaded project module: {name}")
    for name in stale:
        del modules[name]
    retained = [entry for entry in search_path if not session_path(entry)]
    search_path[:] = retained
    return tuple(stale)
