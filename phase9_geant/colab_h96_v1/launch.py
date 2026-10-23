"""Isolated CPU-only Colab launcher for the first authorized h96 v2 run.

The bootstrap verifies the external snapshot manifest and archive before it
adds the repository snapshot to ``sys.path`` or imports any project module.
Training requires both ``--run-training`` and ``--authorize-training``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.machinery
import json
import os
import stat
import sys
import types
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


SOURCE_MEMBER = "PORTABLE_SOURCE_MANIFEST.json"
FIRST_RUN_SEED = 42
FIRST_RUN_MODE = "topology"
COLAB_OUTPUT_ROOT = "geant_horizon96_streaming_experiment_v2_colab_memv3"


class LaunchIntegrityError(RuntimeError):
    """The uploaded snapshot, package, dataset, or launch request is invalid."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_no_redirect(path: Path, *, require_exists: bool = True) -> Path:
    absolute = Path(os.path.abspath(path))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        try:
            info = current.lstat()
        except FileNotFoundError:
            if require_exists:
                raise LaunchIntegrityError(f"Required path component is missing: {current}")
            break
        if current.is_symlink() or getattr(current, "is_junction", lambda: False)():
            raise LaunchIntegrityError(f"Path redirection is prohibited: {current}")
        if int(getattr(info, "st_file_attributes", 0)) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400):
            raise LaunchIntegrityError(f"Reparse-point path is prohibited: {current}")
    if require_exists and absolute.resolve(strict=True) != absolute:
        raise LaunchIntegrityError(f"Path is not canonical: {absolute}")
    return absolute


def _safe_member(name: str) -> PurePosixPath:
    if not name or "\\" in name:
        raise LaunchIntegrityError(f"Unsafe archive path: {name!r}")
    relative = PurePosixPath(name)
    if (relative.is_absolute() or not relative.parts or ":" in relative.parts[0]
            or any(part in ("", ".", "..") for part in relative.parts)):
        raise LaunchIntegrityError(f"Unsafe archive path: {name!r}")
    return relative


def _verify_source_before_import(
    repository: Path,
    source_archive: Path,
    source_manifest: Path,
    *,
    expected_archive_sha256: str,
    expected_manifest_sha256: str,
) -> dict[str, Any]:
    """Stdlib-only verification; must run before any first-party import."""
    repo = _canonical_no_redirect(repository)
    archive_path = _canonical_no_redirect(source_archive)
    manifest_path = _canonical_no_redirect(source_manifest)
    manifest_bytes = manifest_path.read_bytes()
    if _sha256(manifest_bytes) != expected_manifest_sha256:
        raise LaunchIntegrityError("Source manifest differs from the externally reviewed digest")
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    if manifest.get("archive_sha256") != expected_archive_sha256:
        raise LaunchIntegrityError("Source manifest archive identity differs from approved digest")
    archive_bytes = archive_path.read_bytes()
    if _sha256(archive_bytes) != expected_archive_sha256:
        raise LaunchIntegrityError("Source snapshot archive SHA-256 mismatch")
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise LaunchIntegrityError("Source manifest has no file-entry list")
    expected: dict[str, Mapping[str, Any]] = {}
    folded: set[str] = set()
    for entry in entries:
        name = entry.get("path")
        _safe_member(name)
        if name.casefold() in folded:
            raise LaunchIntegrityError(f"Duplicate/case-conflicting source member: {name}")
        folded.add(name.casefold())
        if not isinstance(entry.get("size_bytes"), int) or entry["size_bytes"] < 0:
            raise LaunchIntegrityError(f"Invalid source file size: {name}")
        if not isinstance(entry.get("sha256"), str) or len(entry["sha256"]) != 64:
            raise LaunchIntegrityError(f"Invalid source file digest: {name}")
        expected[name] = entry
    if SOURCE_MEMBER in expected:
        raise LaunchIntegrityError("Reserved source-manifest member appears in file inventory")
    try:
        import io
        with zipfile.ZipFile(io.BytesIO(archive_bytes), "r") as archive:
            names: list[str] = []
            seen: set[str] = set()
            for info in archive.infolist():
                _safe_member(info.filename)
                if info.is_dir():
                    raise LaunchIntegrityError(f"Directory archive member prohibited: {info.filename}")
                mode = (info.external_attr >> 16) & 0xFFFF
                if stat.S_ISLNK(mode):
                    raise LaunchIntegrityError(f"Symlink archive member prohibited: {info.filename}")
                folded_name = info.filename.casefold()
                if folded_name in seen:
                    raise LaunchIntegrityError(f"Duplicate/case-conflicting archive member: {info.filename}")
                seen.add(folded_name)
                names.append(info.filename)
            wanted = set(expected) | {SOURCE_MEMBER}
            if set(names) != wanted or len(names) != len(wanted):
                raise LaunchIntegrityError("Source archive has missing or unexpected members")
            internal = json.loads(archive.read(SOURCE_MEMBER).decode("utf-8"))
            if internal.get("entries") != entries or internal.get("schema") != "geant-h96-colab-source-snapshot-v1":
                raise LaunchIntegrityError("Internal and external source manifests disagree")
            for name, entry in expected.items():
                content = archive.read(name)
                if len(content) != entry["size_bytes"] or _sha256(content) != entry["sha256"]:
                    raise LaunchIntegrityError(f"Source archive member hash/size mismatch: {name}")
    except (zipfile.BadZipFile, KeyError, json.JSONDecodeError) as exc:
        raise LaunchIntegrityError(f"Malformed source archive: {exc}") from exc

    # Verify the already-extracted snapshot before importing project code.
    for name, entry in expected.items():
        file_path = _canonical_no_redirect(repo.joinpath(*PurePosixPath(name).parts))
        if file_path.stat().st_size != entry["size_bytes"] or _sha256(file_path.read_bytes()) != entry["sha256"]:
            raise LaunchIntegrityError(f"Extracted source differs from approved snapshot: {name}")
    launch_entry = expected.get(Path(__file__).resolve().relative_to(repo).as_posix())
    if launch_entry is None:
        raise LaunchIntegrityError("Executed launcher is absent from the source snapshot manifest")
    return {"manifest": manifest, "manifest_sha256": _sha256(manifest_bytes), "archive_sha256": expected_archive_sha256}


def _assert_clean_project_import_state(*, modules=None) -> None:
    loaded = sys.modules if modules is None else modules
    conflicts = [name for name in loaded if name == "phase9_geant" or name.startswith("phase9_geant.") or name == "src" or name.startswith("src.")]
    if conflicts:
        raise LaunchIntegrityError(f"Project modules were imported before source verification: {conflicts[:8]}")


def _install_controlled_src_namespace(repository: Path, *, modules=None):
    """Bind the namespace package to this snapshot, excluding site-package shadows."""
    loaded = sys.modules if modules is None else modules
    if "src" in loaded or any(name.startswith("src.") for name in loaded):
        raise LaunchIntegrityError("Cannot bind src namespace after a project src module was loaded")
    src_path = _canonical_no_redirect(repository / "src")
    if not src_path.is_dir():
        raise LaunchIntegrityError(f"Approved source snapshot has no src directory: {src_path}")
    module = types.ModuleType("src")
    module.__package__ = "src"
    module.__path__ = [str(src_path)]
    spec = importlib.machinery.ModuleSpec("src", loader=None, is_package=True)
    spec.submodule_search_locations = [str(src_path)]
    module.__spec__ = spec
    loaded["src"] = module
    return module


def _verify_loaded_project_modules(repository: Path, allowed_source_paths: set[Path], *, modules=None) -> None:
    """Reject any loaded first-party module not backed by the approved bundle."""
    expected_root = repository.resolve(strict=True)
    for name, module in tuple((sys.modules if modules is None else modules).items()):
        if not (name == "phase9_geant" or name.startswith("phase9_geant.")
                or name == "src" or name.startswith("src.")):
            continue
        module_path = getattr(module, "__file__", None)
        if module_path is not None:
            resolved = Path(module_path).resolve(strict=True)
            try:
                resolved.relative_to(expected_root)
            except ValueError as exc:
                raise LaunchIntegrityError(f"Unexpected loaded project module {name}: {resolved}") from exc
            if resolved not in allowed_source_paths:
                raise LaunchIntegrityError(f"Loaded project source is not in the approved manifest: {name}: {resolved}")
        if name == "src":
            locations = [Path(item).resolve(strict=True) for item in getattr(module, "__path__", ())]
            if locations != [expected_root / "src"]:
                raise LaunchIntegrityError(f"Unexpected namespace package locations for src: {locations}")


def _import_verified_runtime(repository: Path, allowed_source_paths: set[Path]):
    _assert_clean_project_import_state()
    sys.path.insert(0, str(repository))
    _install_controlled_src_namespace(repository)
    from phase9_geant.colab_h96_v1 import environment, linux_resources, package
    from phase9_geant import horizon96_streaming_experiment_v2 as trainer

    expected_root = repository.resolve(strict=True)
    _verify_loaded_project_modules(expected_root, allowed_source_paths)
    return package, environment, linux_resources, trainer


def _activate_system_site_packages() -> None:
    """Add installed dependency directories without executing .pth/sitecustomize."""
    import sysconfig
    for key in ("purelib", "platlib"):
        location = Path(sysconfig.get_paths()[key]).resolve(strict=True)
        if str(location) not in sys.path:
            sys.path.append(str(location))


def _bind_resource_policy(trainer: Any, policy: Mapping[str, Any]) -> None:
    """Bind the launcher's enforced policy into the hashed run configuration."""
    if policy.get("policy_id") != "colab-linux-process-memory-v3-visible-cgroup":
        raise LaunchIntegrityError("Unexpected Colab resource policy identity")
    if (policy.get("process_peak_limit_bytes") != 2 * 1024**3
            or policy.get("in_run_os_available_reserve_bytes") != 384 * 1024**2
            or policy.get("in_run_cgroup_headroom_reserve_bytes") != 384 * 1024**2
            or policy.get("minimum_free_disk_bytes") != 512 * 1024**2):
        raise LaunchIntegrityError("Colab resource policy values differ from the reviewed v3 limits")
    # The trainer's shared startup ResourceCheck reports an estimated process
    # allowance. Bind it to the same Colab cap so its manifest cannot retain
    # the old Windows-derived 640 MiB estimate beside the Linux monitor.
    trainer.lazy.SMOKE_ESTIMATED_PEAK_RAM = int(policy["process_peak_limit_bytes"])
    trainer.PROCESS_PEAK_LIMIT_BYTES = int(policy["process_peak_limit_bytes"])
    trainer.SYSTEM_AVAILABLE_RESERVE_BYTES = int(policy["in_run_os_available_reserve_bytes"])
    trainer.CONFIG = {
        **trainer.CONFIG,
        "memory_policy": {**trainer.CONFIG.get("memory_policy", {}), **dict(policy), "workers": 0},
        "linux_resource_policy": dict(policy),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--dry-run", action="store_true")
    action.add_argument("--run-training", action="store_true")
    parser.add_argument("--authorize-training", action="store_true")
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--source-bundle", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--expected-source-bundle-sha256", required=True)
    parser.add_argument("--expected-source-manifest-sha256", required=True)
    parser.add_argument("--dataset-bundle", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--expected-dataset-bundle-sha256", required=True)
    parser.add_argument("--expected-dataset-manifest-sha256", required=True)
    parser.add_argument("--seed", type=int, choices=(FIRST_RUN_SEED,), default=FIRST_RUN_SEED)
    parser.add_argument("--adjacency-mode", choices=(FIRST_RUN_MODE,), default=FIRST_RUN_MODE)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.run_training and not args.authorize_training:
        raise SystemExit("Training requires a separately approved --authorize-training flag")
    if args.dry_run and args.authorize_training:
        raise SystemExit("Do not combine dry-run with training authorization")
    if args.seed != FIRST_RUN_SEED or args.adjacency_mode != FIRST_RUN_MODE:
        raise SystemExit("This reviewed migration launcher only permits first-run seed 42/topology")
    repository = Path(os.path.abspath(args.repository_root))
    source_identity = _verify_source_before_import(
        repository, args.source_bundle, args.source_manifest,
        expected_archive_sha256=args.expected_source_bundle_sha256,
        expected_manifest_sha256=args.expected_source_manifest_sha256,
    )
    _activate_system_site_packages()
    approved_paths = {
        (repository / entry["path"]).resolve(strict=True)
        for entry in source_identity["manifest"]["entries"]
        if entry["path"].endswith(".py")
    }
    package, environment_module, resource_module, trainer = _import_verified_runtime(repository, approved_paths)
    environment = environment_module.verify_environment(require_pytest=args.dry_run)
    data_result = package.verify_dataset_bundle(
        args.dataset_bundle, args.dataset_manifest,
        args.expected_dataset_bundle_sha256, args.expected_dataset_manifest_sha256,
    )
    data_root = repository / package.DATA_ROOT_RELATIVE
    # This only validates the already-extracted exact train/validation bundle.
    package._read_dataset_metadata(data_root)
    data_files = {item.name for item in data_root.iterdir()}
    if data_files != set(package.DATA_ARTIFACTS):
        raise LaunchIntegrityError("Extracted dataset contains unexpected or missing names")
    resource_preflight = resource_module.preflight(disk_path=repository)
    # A resource-policy amendment gets a separate run namespace; never reuse
    # an output from the earlier 640 MiB policy, including a failed attempt.
    output_root_name = COLAB_OUTPUT_ROOT
    output_root = repository / "data" / "processed" / output_root_name
    run_destination = output_root / f"h96_{FIRST_RUN_MODE}_seed_{FIRST_RUN_SEED}"
    if output_root.exists() or run_destination.exists():
        raise LaunchIntegrityError(f"Refusing existing Colab run namespace: {run_destination}")

    # The full trainer performs its own canonical path, data, source, scaler,
    # edge-order, resource, and lifecycle checks before model construction.
    trainer.OUTPUT_ROOT_NAME = output_root_name
    trainer.RUN_SOURCE_FILES = tuple(sorted(set(trainer.RUN_SOURCE_FILES) | set(package.MIGRATION_RUNTIME_FILES)))
    trainer.CONFIG = {
        **trainer.CONFIG,
        "colab_bundle_identity": {
            "source_snapshot_manifest_sha256": source_identity["manifest_sha256"],
            "source_bundle_sha256": source_identity["archive_sha256"],
            "dataset_manifest_sha256": args.expected_dataset_manifest_sha256,
            "dataset_bundle_sha256": data_result["archive_sha256"],
        },
        "colab_environment": environment,
    }
    _bind_resource_policy(trainer, resource_preflight["policy"])
    verified = trainer._verified_inputs(data_root, repository)
    summary = {
        "status": "preflight_passed_no_training" if args.dry_run else "training_authorized",
        "protocol_id": trainer.PROTOCOL_ID,
        "seed": FIRST_RUN_SEED,
        "adjacency_mode": FIRST_RUN_MODE,
        "output_root": str(output_root),
        "run_destination": str(run_destination),
        "source_snapshot_manifest_sha256": source_identity["manifest_sha256"],
        "source_bundle_sha256": source_identity["archive_sha256"],
        "dataset_manifest_sha256": args.expected_dataset_manifest_sha256,
        "dataset_bundle_sha256": data_result["archive_sha256"],
        "environment": environment,
        "linux_resource_preflight": resource_preflight,
        "data_protocol_id": verified[3]["protocol_id"],
        "train_windows": len(verified[2]["train"]["windows"]),
        "validation_windows": len(verified[2]["validation"]["windows"]),
        "history_length": 672,
        "horizon": 96,
        "test_access": "no test files included or loaded",
    }
    if args.dry_run:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if not args.run_training or not args.authorize_training:
        raise SystemExit("No training action was authorized")
    result = trainer.train_one(
        data_root, repository_root=repository, seed=FIRST_RUN_SEED, mode=FIRST_RUN_MODE,
        run_training=True, explicit_authorization=True, device_name="cpu",
        monitor=resource_module.LinuxMemoryMonitor(),
    )
    summary["training_result"] = {key: result.get(key) for key in (
        "status", "epoch_history", "best_epoch", "best_validation_h96_residual_smooth_l1",
        "validation_metrics", "runtime_seconds", "memory_monitor", "checkpoint_sha256",
    )}
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
