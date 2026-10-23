"""Build and verify exact train/validation-only Colab transfer bundles.

No project modules are imported at module import time. ``prepare_bundles`` is
an explicit packaging action and reads only the approved compact train/validation
artifacts plus the local topology file and source files in its allowlist.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import shutil
import stat
import subprocess
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping


DATA_ROOT_RELATIVE = Path("data/processed/geant_horizon96_streaming_v1")
DATA_ARTIFACTS = (
    "metadata.json",
    "timestamps_train.npy",
    "timestamps_validation.npy",
    "utilization_train.npy",
    "utilization_validation.npy",
    "valid_train.npy",
    "valid_validation.npy",
    "window_index_train_h96.json",
    "window_index_validation_h96.json",
)
SOURCE_MANIFEST_MEMBER = "PORTABLE_SOURCE_MANIFEST.json"
SOURCE_BUNDLE_NAME = "geant_h96_colab_source_v1.zip"
DATA_BUNDLE_NAME = "geant_h96_colab_trainval_v1.zip"
SOURCE_MANIFEST_NAME = "source_snapshot_manifest.json"
DATA_MANIFEST_NAME = "dataset_bundle_manifest.json"
OUTPUT_RELATIVE = Path("data/transfer_bundles/geant_h96_colab_v1_review6_final3")

# This is the trainer's first-party run inventory as inspected in the source.
# package.py itself compares this inventory to the live runner before packaging.
TRAINER_SOURCE_FILES = (
    "docs/geant_horizon96_memory_v2.md",
    "docs/geant_horizon96_only_candidate_v1.md",
    "phase9_geant/__init__.py",
    "phase9_geant/build_link_series.py",
    "phase9_geant/horizon96_experiment.py",
    "phase9_geant/horizon96_smoke.py",
    "phase9_geant/horizon96_streaming_data.py",
    "phase9_geant/horizon96_streaming_experiment.py",
    "phase9_geant/horizon96_streaming_experiment_v2.py",
    "phase9_geant/horizon_v2_data.py",
    "phase9_geant/horizon_v2_experiment.py",
    "phase9_geant/horizon_windows.py",
    "phase9_geant/parse_geant.py",
    "phase9_geant/safe_paths.py",
    "phase9_geant/validity_v2.py",
    "src/gnn/__init__.py",
    "src/gnn/dataset.py",
    "src/gnn/model.py",
    "src/gnn/preprocessing.py",
    "src/gnn/spatiotemporal_link.py",
)
PORTABLE_SUPPORT_FILES = (
    "phase9_geant/colab_h96_v1/__init__.py",
    "phase9_geant/colab_h96_v1/package.py",
    "phase9_geant/colab_h96_v1/environment.py",
    "phase9_geant/colab_h96_v1/linux_resources.py",
    "phase9_geant/colab_h96_v1/launch.py",
    "phase9_geant/colab_h96_v1/launch_matrix.py",
    "phase9_geant/colab_h96_v1/matrix_workflow.py",
    "phase9_geant/colab_h96_v1/matrix_recovery.py",
    "phase9_geant/colab_h96_v1/launch_matrix_recovery.py",
    "phase9_geant/colab_h96_v1/notebook_stage.py",
    "phase9_geant/colab_h96_v1/tests/test_portability.py",
    "phase9_geant/colab_h96_v1/tests/test_launch_matrix.py",
    "phase9_geant/colab_h96_v1/tests/test_matrix_workflow.py",
    "phase9_geant/colab_h96_v1/tests/test_matrix_recovery.py",
    "phase9_geant/colab_h96_v1/tests/test_notebook_stage.py",
    "phase9_geant/tests/conftest.py",
    # conftest imports the shared train/validation-only filename validator.
    "phase9_geant/split_access.py",
    "phase9_geant/tests/test_horizon96_streaming_experiment.py",
    "phase9_geant/tests/test_horizon96_streaming_experiment_v2.py",
    "phase9_geant/tests/test_spatiotemporal_link.py",
    "notebooks/geant_h96_colab_v1.ipynb",
    "requirements-colab-h96.txt",
    "docs/geant_horizon96_colab_migration.md",
    "data/raw/geant_topology/geant.txt",
)
MIGRATION_RUNTIME_FILES = (
    "phase9_geant/colab_h96_v1/__init__.py",
    "phase9_geant/colab_h96_v1/package.py",
    "phase9_geant/colab_h96_v1/environment.py",
    "phase9_geant/colab_h96_v1/linux_resources.py",
    "phase9_geant/colab_h96_v1/launch.py",
    "phase9_geant/colab_h96_v1/launch_matrix.py",
    "phase9_geant/colab_h96_v1/matrix_workflow.py",
    "phase9_geant/colab_h96_v1/matrix_recovery.py",
    "phase9_geant/colab_h96_v1/launch_matrix_recovery.py",
    "phase9_geant/colab_h96_v1/notebook_stage.py",
)


class BundleError(ValueError):
    """A bundle path, member, manifest, or digest failed closed."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _safe_existing_file(path: Path, root: Path) -> Path:
    absolute = Path(os.path.abspath(path))
    root_absolute = Path(os.path.abspath(root))
    try:
        absolute.relative_to(root_absolute)
    except ValueError as exc:
        raise BundleError(f"Path escapes approved root: {path}") from exc
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current = current / component
        info = current.lstat()
        if current.is_symlink() or getattr(current, "is_junction", lambda: False)():
            raise BundleError(f"Symlink or junction component is not permitted: {current}")
        if int(getattr(info, "st_file_attributes", 0)) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400):
            raise BundleError(f"Reparse-point component is not permitted: {current}")
    if not absolute.is_file() or absolute.resolve(strict=True) != absolute:
        raise BundleError(f"Expected a canonical regular file: {absolute}")
    return absolute


def _relative_path(value: str) -> PurePosixPath:
    if not isinstance(value, str) or "\\" in value:
        raise BundleError(f"Unsafe archive path: {value!r}")
    path = PurePosixPath(value)
    if (path.is_absolute() or not path.parts or ":" in path.parts[0]
            or any(part in ("", ".", "..") for part in path.parts)):
        raise BundleError(f"Unsafe archive path: {value!r}")
    return path


def _validate_entries(entries: Iterable[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    by_name: dict[str, Mapping[str, Any]] = {}
    folded: set[str] = set()
    for entry in entries:
        name = str(entry.get("path", ""))
        _relative_path(name)
        key = name.casefold()
        if key in folded:
            raise BundleError(f"Duplicate or case-conflicting archive path: {name}")
        folded.add(key)
        digest = entry.get("sha256")
        size = entry.get("size_bytes")
        if (not isinstance(digest, str) or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest.casefold())
                or not isinstance(size, int) or size < 0):
            raise BundleError(f"Malformed bundle entry: {name}")
        by_name[name] = entry
    return by_name


def _zip_member_names(archive: zipfile.ZipFile) -> tuple[str, ...]:
    names: list[str] = []
    folded: set[str] = set()
    for info in archive.infolist():
        name = info.filename
        _relative_path(name)
        if info.is_dir():
            raise BundleError(f"Directory entries are not permitted in bundles: {name}")
        mode = (info.external_attr >> 16) & 0xFFFF
        if stat.S_ISLNK(mode):
            raise BundleError(f"Symlink archive member is not permitted: {name}")
        key = name.casefold()
        if key in folded:
            raise BundleError(f"Duplicate or case-conflicting archive member: {name}")
        folded.add(key)
        names.append(name)
    return tuple(names)


def _make_zip(entries: Iterable[tuple[str, Path]], manifest_member: tuple[str, bytes] | None = None) -> bytes:
    import io

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, path in sorted(entries, key=lambda item: item[0]):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            archive.writestr(info, path.read_bytes())
        if manifest_member is not None:
            name, payload = manifest_member
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            archive.writestr(info, payload)
    return buffer.getvalue()


def _write_new_file(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _git_value(repository: Path, *args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *args], cwd=repository, check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _source_paths(repository: Path) -> tuple[Path, ...]:
    # Import only the inert trainer module to compare the static source inventory.
    from phase9_geant import horizon96_streaming_experiment_v2 as trainer

    if tuple(trainer.RUN_SOURCE_FILES) != tuple(sorted(TRAINER_SOURCE_FILES)):
        raise BundleError("Colab source inventory differs from the actual v2 trainer RUN_SOURCE_FILES")
    relative = set(TRAINER_SOURCE_FILES) | set(PORTABLE_SUPPORT_FILES)
    return tuple(_safe_existing_file(repository / item, repository) for item in sorted(relative))


def _source_module_name(path: Path, repository: Path) -> str:
    relative = path.relative_to(repository).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _existing_module_file(repository: Path, module_name: str) -> Path | None:
    if not module_name or not (module_name == "phase9_geant" or module_name.startswith("phase9_geant.")
                               or module_name == "src" or module_name.startswith("src.")):
        return None
    base = repository.joinpath(*module_name.split("."))
    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


def _module_file(repository: Path, module_name: str) -> Path | None:
    found = _existing_module_file(repository, module_name)
    if found is not None:
        return found
    # src is an intentional namespace package in this repository.
    if module_name == "src" and (repository / "src").is_dir():
        return None
    raise BundleError(f"Unresolved first-party import in source snapshot: {module_name}")


def _first_party_import_inventory(repository: Path, paths: Iterable[Path]) -> dict[str, list[str]]:
    """Return and validate direct first-party import edges for the snapshot."""
    paths = tuple(paths)
    selected = {path.resolve(strict=True) for path in paths}
    inventory: dict[str, list[str]] = {}

    def add_dependency(dependencies: set[str], module_name: str) -> None:
        if not (module_name == "phase9_geant" or module_name.startswith("phase9_geant.")
                or module_name == "src" or module_name.startswith("src.")):
            return
        module_path = _module_file(repository, module_name)
        if module_path is not None:
            resolved = module_path.resolve(strict=True)
            if resolved not in selected:
                raise BundleError(f"First-party import is absent from source allowlist: {module_name} ({module_path})")
            dependencies.add(module_name)
        # Require each concrete parent package initializer as part of the closure.
        pieces = module_name.split(".")
        for index in range(1, len(pieces)):
            parent = ".".join(pieces[:index])
            parent_path = _module_file(repository, parent)
            if parent_path is not None and parent_path.resolve(strict=True) not in selected:
                raise BundleError(f"First-party parent package is absent from source allowlist: {parent}")
            if parent_path is not None:
                dependencies.add(parent)

    for path in sorted((item for item in paths if item.suffix == ".py"), key=lambda item: item.as_posix()):
        module_name = _source_module_name(path, repository)
        package_name = module_name if path.name == "__init__.py" else module_name.rpartition(".")[0]
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        dependencies: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    add_dependency(dependencies, alias.name)
            elif isinstance(node, ast.ImportFrom):
                try:
                    base = (importlib.util.resolve_name("." * node.level + (node.module or ""), package_name)
                            if node.level else (node.module or ""))
                except (ImportError, ValueError) as exc:
                    raise BundleError(f"Cannot resolve relative import in {path}: {exc}") from exc
                if not base:
                    continue
                candidates = [base]
                # `from package import module` loads that submodule when it exists.
                for alias in node.names:
                    candidate = f"{base}.{alias.name}"
                    if _existing_module_file(repository, candidate) is not None:
                        candidates.append(candidate)
                for candidate in candidates:
                    if candidate == "phase9_geant" or candidate.startswith("phase9_geant.") or candidate == "src" or candidate.startswith("src."):
                        add_dependency(dependencies, candidate)
        inventory[path.relative_to(repository).as_posix()] = sorted(dependencies)
    return inventory


def _entries(repository: Path, paths: Iterable[Path]) -> list[dict[str, Any]]:
    result = []
    imports = _first_party_import_inventory(repository, paths)
    for path in paths:
        relative = path.relative_to(repository).as_posix()
        entry = {"path": relative, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        if relative in imports:
            entry["first_party_imports"] = imports[relative]
        result.append(entry)
    return result


def _source_manifest_base(repository: Path, entries: list[dict[str, Any]]) -> dict[str, Any]:
    head = _git_value(repository, "rev-parse", "HEAD")
    tag = _git_value(repository, "rev-parse", "v1-frozen")
    dirty = bool(_git_value(repository, "status", "--porcelain"))
    return {
        "schema": "geant-h96-colab-source-snapshot-v1",
        "source_snapshot_status": "generated-local-snapshot; NOT independently reviewed or approved",
        "git_head_at_packaging": head,
        "v1_frozen_commit": tag,
        "working_tree_was_dirty": dirty,
        "source_root_relative": ".",
        "entries": entries,
        "first_party_import_inventory": {
            entry["path"]: entry.get("first_party_imports", []) for entry in entries if entry["path"].endswith(".py")
        },
        "dependency_notes": {
            "phase9_geant/split_access.py": (
                "Included because phase9_geant/tests/conftest.py imports its shared train/validation array-name validator; "
                "the module is tracked in v1-frozen and its source is preserved verbatim."
            )
        },
        "excluded": [
            "all raw GEANT demand matrices", "all test-split artifacts", "checkpoints",
            "predictions", "run directories", "credentials and personal files",
        ],
        "topology_provenance": "local file-byte identity only; upstream provenance remains unverified",
    }


def _read_dataset_metadata(data_root: Path) -> dict[str, Any]:
    path = _safe_existing_file(data_root / "metadata.json", data_root)
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("protocol_id") != "geant-horizon96-streaming-trainval-v1" or value.get("status") != "complete":
        raise BundleError("Approved compact dataset metadata protocol/status mismatch")
    if value.get("source_scope") != {"train": [0, 8020], "validation": [8020, 9740]}:
        raise BundleError("Dataset boundaries do not match the frozen train/validation protocol")
    if value.get("windows", {}).get("history_length") != 672 or value.get("windows", {}).get("horizon") != 96:
        raise BundleError("Dataset is not the approved h96-only/672-history dataset")
    if value.get("artifact_names") != sorted(DATA_ARTIFACTS):
        raise BundleError("Dataset metadata artifact allowlist differs from the nine approved files")
    if set(value.get("array_files", {})) != set(DATA_ARTIFACTS) - {"metadata.json", "window_index_train_h96.json", "window_index_validation_h96.json"}:
        raise BundleError("Dataset metadata array fingerprint allowlist mismatch")
    topology = value.get("topology", {})
    if (topology.get("node_count") != 22 or topology.get("undirected_link_count") != 36
            or topology.get("directed_link_count") != 72
            or topology.get("sha256_local_file_identity_only") != "38c90629f30af70fe840e33071b12cc40498bb910e8dbeb2fa7863a7b645d121"
            or len(topology.get("directed_edges_interleaved", ())) != 72):
        raise BundleError("Dataset topology contract or local identity mismatch")
    return value


def make_dataset_bundle(data_root: Path) -> tuple[bytes, dict[str, Any]]:
    root = Path(os.path.abspath(data_root))
    if not root.is_dir() or root.resolve(strict=True) != root:
        raise BundleError(f"Dataset root is missing or redirected: {root}")
    found = {item.name for item in root.iterdir()}
    if found != set(DATA_ARTIFACTS):
        raise BundleError(
            f"Dataset artifact inventory mismatch: extra={sorted(found - set(DATA_ARTIFACTS))}, "
            f"missing={sorted(set(DATA_ARTIFACTS) - found)}"
        )
    if any("test" in name.casefold() or "h4" in name.casefold() for name in found):
        raise BundleError("Test/h4-named artifact is not permitted in the train/validation bundle")
    metadata = _read_dataset_metadata(root)
    artifact_paths = [_safe_existing_file(root / name, root) for name in DATA_ARTIFACTS]
    artifact_entries = [
        {"path": name, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for name, path in zip(DATA_ARTIFACTS, artifact_paths, strict=True)
    ]
    # Verify hashes against the builder's metadata and exact array/window schemas
    # before reading any array through NumPy. Names are explicit and allowlisted.
    for entry in artifact_entries:
        name = entry["path"]
        if name in metadata["array_files"] and metadata["array_files"][name].get("file_sha256") != entry["sha256"]:
            raise BundleError(f"Dataset array byte hash mismatch: {name}")
    if metadata.get("window_index_sha256", {}).get("train") != next(e["sha256"] for e in artifact_entries if e["path"] == "window_index_train_h96.json"):
        raise BundleError("Training window-index hash mismatch")
    if metadata.get("window_index_sha256", {}).get("validation") != next(e["sha256"] for e in artifact_entries if e["path"] == "window_index_validation_h96.json"):
        raise BundleError("Validation window-index hash mismatch")
    payload = _make_zip((name, root / name) for name in DATA_ARTIFACTS)
    bundle_hash = sha256_bytes(payload)
    manifest = {
        "schema": "geant-h96-colab-trainval-bundle-v1",
        "bundle_status": "generated-local; user-controlled transfer only; NOT independently approved",
        "archive_name": DATA_BUNDLE_NAME,
        "archive_sha256": bundle_hash,
        "dataset_protocol_id": metadata["protocol_id"],
        "source_scope": metadata["source_scope"],
        "history_length": 672,
        "horizon": 96,
        "window_counts": {"train": metadata["windows"]["train_count"], "validation": metadata["windows"]["validation_count"]},
        "topology": {
            "node_count": 22,
            "undirected_link_count": 36,
            "directed_link_count": 72,
            "ordered_edge_sha256": metadata["topology"]["ordered_edge_sha256"],
            "local_topology_sha256_identity_only": metadata["topology"]["sha256_local_file_identity_only"],
        },
        "artifacts": artifact_entries,
        "exact_archive_members": list(DATA_ARTIFACTS),
        "excluded": ["test arrays/masks/timestamps/windows", "raw GEANT demand matrices", "checkpoints", "predictions", "run outputs"],
    }
    return payload, manifest


def make_source_bundle(repository: Path) -> tuple[bytes, dict[str, Any]]:
    repo = Path(os.path.abspath(repository))
    if not repo.is_dir() or repo.resolve(strict=True) != repo:
        raise BundleError(f"Source root is missing or redirected: {repo}")
    paths = _source_paths(repo)
    entries = _entries(repo, paths)
    internal = _source_manifest_base(repo, entries)
    internal_bytes = canonical_json_bytes(internal)
    payload = _make_zip(
        ((path.relative_to(repo).as_posix(), path) for path in paths),
        (SOURCE_MANIFEST_MEMBER, internal_bytes),
    )
    external = {**internal, "archive_name": SOURCE_BUNDLE_NAME, "archive_sha256": sha256_bytes(payload),
                "internal_manifest_member": SOURCE_MANIFEST_MEMBER}
    return payload, external


def _verify_archive_payload(
    payload: bytes,
    manifest: Mapping[str, Any],
    *,
    expected_archive_sha256: str,
    expected_manifest_sha256: str,
    manifest_payload: bytes,
    kind: str,
) -> dict[str, Any]:
    if sha256_bytes(manifest_payload) != expected_manifest_sha256:
        raise BundleError("External manifest does not match the independently supplied expected SHA-256")
    if sha256_bytes(payload) != expected_archive_sha256 or manifest.get("archive_sha256") != expected_archive_sha256:
        raise BundleError("Archive SHA-256 mismatch")
    try:
        import io
        with zipfile.ZipFile(io.BytesIO(payload), "r") as archive:
            names = _zip_member_names(archive)
            if kind == "data":
                entries = _validate_entries(manifest.get("artifacts", ()))
                expected = set(DATA_ARTIFACTS)
            elif kind == "source":
                entries = _validate_entries(manifest.get("entries", ()))
                expected = set(entries) | {SOURCE_MANIFEST_MEMBER}
                import_inventory = manifest.get("first_party_import_inventory")
                expected_import_inventory = {
                    name: entry.get("first_party_imports", [])
                    for name, entry in entries.items() if name.endswith(".py")
                }
                if import_inventory != expected_import_inventory:
                    raise BundleError("Source first-party import inventory is missing or inconsistent")
                if any(not isinstance(deps, list) or deps != sorted(set(deps))
                       or any(not isinstance(dep, str) for dep in deps)
                       for deps in import_inventory.values()):
                    raise BundleError("Malformed first-party import dependency list")
                internal = json.loads(archive.read(SOURCE_MANIFEST_MEMBER).decode("utf-8"))
                if (internal.get("entries") != manifest.get("entries")
                        or internal.get("first_party_import_inventory") != manifest.get("first_party_import_inventory")):
                    raise BundleError("Internal and external source manifests disagree")
                if internal.get("schema") != "geant-h96-colab-source-snapshot-v1":
                    raise BundleError("Unsupported source manifest schema")
            else:
                raise BundleError(f"Unknown bundle kind: {kind}")
            if set(names) != expected or len(names) != len(expected):
                raise BundleError(f"Unexpected archive members: missing={sorted(expected-set(names))}, extra={sorted(set(names)-expected)}")
            for name, entry in entries.items():
                raw = archive.read(name)
                if len(raw) != entry["size_bytes"] or sha256_bytes(raw) != entry["sha256"]:
                    raise BundleError(f"Archive member hash/size mismatch: {name}")
    except (zipfile.BadZipFile, KeyError, json.JSONDecodeError) as exc:
        raise BundleError(f"Malformed or incomplete {kind} archive: {exc}") from exc
    return {"kind": kind, "archive_sha256": expected_archive_sha256, "manifest_sha256": expected_manifest_sha256, "members": sorted(expected)}


def verify_source_bundle(archive_path: Path, manifest_path: Path, expected_archive_sha256: str, expected_manifest_sha256: str) -> dict[str, Any]:
    manifest_payload = manifest_path.read_bytes()
    manifest = json.loads(manifest_payload.decode("utf-8"))
    payload = archive_path.read_bytes()
    return _verify_archive_payload(payload, manifest, expected_archive_sha256=expected_archive_sha256,
                                   expected_manifest_sha256=expected_manifest_sha256,
                                   manifest_payload=manifest_payload, kind="source")


def verify_dataset_bundle(archive_path: Path, manifest_path: Path, expected_archive_sha256: str, expected_manifest_sha256: str) -> dict[str, Any]:
    manifest_payload = manifest_path.read_bytes()
    manifest = json.loads(manifest_payload.decode("utf-8"))
    payload = archive_path.read_bytes()
    return _verify_archive_payload(payload, manifest, expected_archive_sha256=expected_archive_sha256,
                                   expected_manifest_sha256=expected_manifest_sha256,
                                   manifest_payload=manifest_payload, kind="data")


def _safe_extract(payload: bytes, destination: Path, expected_members: Iterable[str]) -> Path:
    target = Path(os.path.abspath(destination))
    if target.exists():
        raise FileExistsError(f"Refusing existing extraction destination: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.parent.resolve(strict=True) != target.parent:
        raise BundleError("Extraction parent resolves through an unexpected alias")
    target.mkdir(exist_ok=False)
    allowed = set(expected_members)
    import io
    with zipfile.ZipFile(io.BytesIO(payload), "r") as archive:
        infos = {info.filename: info for info in archive.infolist()}
        if set(infos) != allowed:
            raise BundleError("Archive changed after verification")
        for name in sorted(allowed):
            relative = _relative_path(name)
            output = target.joinpath(*relative.parts)
            output.parent.mkdir(parents=True, exist_ok=True)
            if output.parent.resolve(strict=True) != output.parent or target not in output.parents:
                raise BundleError(f"Extraction path escaped target: {name}")
            with output.open("xb") as stream:
                stream.write(archive.read(name))
    return target


def extract_source_bundle(archive_path: Path, repository_destination: Path, manifest: Mapping[str, Any]) -> Path:
    entries = _validate_entries(manifest.get("entries", ()))
    names = set(entries) | {SOURCE_MANIFEST_MEMBER}
    return _safe_extract(archive_path.read_bytes(), repository_destination, names)


def extract_dataset_bundle(archive_path: Path, repository_root: Path) -> Path:
    repo = Path(os.path.abspath(repository_root))
    data_root = repo / DATA_ROOT_RELATIVE
    data_root.parent.mkdir(parents=True, exist_ok=True)
    if data_root.parent.resolve(strict=True) != data_root.parent:
        raise BundleError("Dataset extraction parent is redirected")
    payload = archive_path.read_bytes()
    _safe_extract(payload, data_root, DATA_ARTIFACTS)
    return data_root


def verify_extracted_source(repository: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    entries = _validate_entries(manifest.get("entries", ()))
    for name, entry in entries.items():
        path = _safe_existing_file(repository / Path(*PurePosixPath(name).parts), repository)
        if path.stat().st_size != entry["size_bytes"] or sha256_file(path) != entry["sha256"]:
            raise BundleError(f"Extracted source differs from snapshot manifest: {name}")
    return {"verified_source_files": len(entries), "source_manifest_sha256": sha256_bytes(canonical_json_bytes(dict(manifest)))}


def prepare_bundles(repository_root: str | Path, *, output_root: str | Path | None = None) -> dict[str, Any]:
    repository = Path(os.path.abspath(repository_root))
    if repository.resolve(strict=True) != repository:
        raise BundleError("Repository root is not canonical")
    data_root = repository / DATA_ROOT_RELATIVE
    # Run the experiment's exact train/validation preflight before packaging.
    from phase9_geant import horizon96_streaming_experiment_v2 as trainer

    checked = trainer._verified_inputs(data_root, repository)
    splits, metadata = checked[2], checked[3]
    if (len(splits["train"]["windows"]) != 3924 or len(splits["validation"]["windows"]) != 953
            or tuple(checked[4]) != tuple(tuple(edge) for edge in metadata["topology"]["directed_edges_interleaved"])):
        raise BundleError("Experiment preflight window or edge-order verification failed")
    data_payload, data_manifest = make_dataset_bundle(data_root)
    source_payload, source_manifest = make_source_bundle(repository)
    output = Path(os.path.abspath(output_root)) if output_root is not None else repository / OUTPUT_RELATIVE
    if output.exists():
        raise FileExistsError(f"Refusing existing transfer bundle directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.parent.resolve(strict=True) != output.parent:
        raise BundleError("Transfer output parent is redirected")
    output.mkdir(exist_ok=False)
    _write_new_file(output / SOURCE_BUNDLE_NAME, source_payload)
    _write_new_file(output / DATA_BUNDLE_NAME, data_payload)
    _write_new_file(output / SOURCE_MANIFEST_NAME, canonical_json_bytes(source_manifest))
    _write_new_file(output / DATA_MANIFEST_NAME, canonical_json_bytes(data_manifest))
    source_result = verify_source_bundle(output / SOURCE_BUNDLE_NAME, output / SOURCE_MANIFEST_NAME,
                                        source_manifest["archive_sha256"], sha256_file(output / SOURCE_MANIFEST_NAME))
    data_result = verify_dataset_bundle(output / DATA_BUNDLE_NAME, output / DATA_MANIFEST_NAME,
                                        data_manifest["archive_sha256"], sha256_file(output / DATA_MANIFEST_NAME))
    if set(item.name for item in output.iterdir()) != {SOURCE_BUNDLE_NAME, DATA_BUNDLE_NAME, SOURCE_MANIFEST_NAME, DATA_MANIFEST_NAME}:
        raise BundleError("Transfer output directory has unexpected entries")
    return {
        "output_root": str(output),
        "source": {**source_result, "archive_size_bytes": (output / SOURCE_BUNDLE_NAME).stat().st_size,
                   "manifest_path": str(output / SOURCE_MANIFEST_NAME),
                   "manifest_sha256": sha256_file(output / SOURCE_MANIFEST_NAME)},
        "dataset": {**data_result, "archive_size_bytes": (output / DATA_BUNDLE_NAME).stat().st_size,
                    "manifest_path": str(output / DATA_MANIFEST_NAME),
                    "manifest_sha256": sha256_file(output / DATA_MANIFEST_NAME)},
        "dataset_window_counts": {"train": len(splits["train"]["windows"]), "validation": len(splits["validation"]["windows"])},
        "dataset_members": list(DATA_ARTIFACTS),
        "upload_performed": False,
        "independent_review_status": "NOT APPROVED; human review required before transfer/execution",
    }


def prepare_source_only(repository_root: str | Path, *, output_root: str | Path | None = None) -> dict[str, Any]:
    """Create a fresh source-only review bundle without opening dataset artifacts."""
    repository = Path(os.path.abspath(repository_root))
    if not repository.is_dir() or repository.resolve(strict=True) != repository:
        raise BundleError("Repository root is missing or not canonical")
    output = Path(os.path.abspath(output_root)) if output_root is not None else repository / OUTPUT_RELATIVE
    if output.exists():
        raise FileExistsError(f"Refusing existing source-only bundle directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.parent.resolve(strict=True) != output.parent:
        raise BundleError("Source-only transfer output parent is redirected")
    payload, manifest = make_source_bundle(repository)
    manifest_bytes = canonical_json_bytes(manifest)
    output.mkdir(exist_ok=False)
    archive_path = output / SOURCE_BUNDLE_NAME
    manifest_path = output / SOURCE_MANIFEST_NAME
    _write_new_file(archive_path, payload)
    _write_new_file(manifest_path, manifest_bytes)
    archive_hash = sha256_file(archive_path)
    manifest_hash = sha256_file(manifest_path)
    verified = verify_source_bundle(archive_path, manifest_path, archive_hash, manifest_hash)
    # This companion is deliberately outside the source archive to avoid a
    # self-referential digest. The operator still approves these hashes separately.
    hashes_path = output / "SOURCE_HASHES.txt"
    record = (
        "Independent review required; these generated values are not approval tokens.\n"
        f"{SOURCE_BUNDLE_NAME} SHA-256: {archive_hash}\n"
        f"{SOURCE_MANIFEST_NAME} SHA-256: {manifest_hash}\n"
        "This record is external to the source archive and is not self-hashed.\n"
    ).encode("ascii")
    _write_new_file(hashes_path, record)
    expected_names = {SOURCE_BUNDLE_NAME, SOURCE_MANIFEST_NAME, hashes_path.name}
    if {item.name for item in output.iterdir()} != expected_names:
        raise BundleError("Source-only transfer directory has unexpected entries")
    return {
        "output_root": str(output),
        "source_archive": str(archive_path),
        "source_manifest": str(manifest_path),
        "source_hash_record": str(hashes_path),
        "source_archive_sha256": archive_hash,
        "source_manifest_sha256": manifest_hash,
        "source_member_count": len(manifest["entries"]),
        "verified_members": len(verified["members"]),
        "dataset_artifacts_opened": False,
        "upload_performed": False,
        "independent_review_status": "NOT APPROVED; human review required before transfer/execution",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-bundles", action="store_true", help="explicitly build fresh source and train/validation archives")
    parser.add_argument("--prepare-source-only", action="store_true", help="package approved source files without opening dataset artifacts")
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.prepare_bundles == args.prepare_source_only:
        raise SystemExit("Choose exactly one of --prepare-bundles or --prepare-source-only")
    if args.prepare_source_only:
        result = prepare_source_only(args.repository_root, output_root=args.output_root)
    else:
        result = prepare_bundles(args.repository_root, output_root=args.output_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
