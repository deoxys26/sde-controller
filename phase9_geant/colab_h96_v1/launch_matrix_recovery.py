
"""Drive-backup-verified recovery launcher for only Matrix3 shuffled seeds 123/2026.

Importing this module is inert. The CLI has a verify-only mode and a separate
training mode requiring an explicit top-level flag plus per-run typed approval.
It never opens test-split files and never resumes an existing namespace.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import re
import stat
import sys
import types
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence


CRITICAL_TRAINING_SOURCES = (
    "phase9_geant/horizon96_streaming_experiment.py",
    "phase9_geant/horizon96_streaming_experiment_v2.py",
    "phase9_geant/horizon96_streaming_data.py",
    "phase9_geant/horizon96_experiment.py",
    "src/gnn/spatiotemporal_link.py",
    "src/gnn/model.py",
)


class RecoveryBootstrapError(RuntimeError):
    """Failure before the verified recovery helpers have been imported."""


RESOURCE_POLICY_ID = "colab-linux-process-memory-v3-visible-cgroup"


def _activate_resource_policy(
    bootstrap: Any,
    resources: Any,
    trainer: Any,
    disk_path: str | Path,
) -> dict[str, Any]:
    """Validate the preflight envelope, then bind its nested policy to the trainer.

    ``linux_resources.preflight`` returns ``{status, snapshot, policy}``;
    the policy identity is intentionally nested.
    """
    result = resources.preflight(disk_path=disk_path)

    if not isinstance(result, Mapping) or result.get("status") != "pass":
        raise RecoveryBootstrapError("Colab resource preflight did not pass")

    policy = result.get("policy")

    if not isinstance(policy, Mapping) or policy.get("policy_id") != RESOURCE_POLICY_ID:
        raise RecoveryBootstrapError("Approved Colab v3 resource policy is not active")

    bootstrap._bind_resource_policy(trainer, policy)

    bound_config = getattr(trainer, "CONFIG", {})

    if (
        getattr(trainer, "PROCESS_PEAK_LIMIT_BYTES", None) != 2 * 1024**3
        or getattr(
            getattr(trainer, "lazy", None),
            "SMOKE_ESTIMATED_PEAK_RAM",
            None,
        ) != 2 * 1024**3
        or getattr(
            trainer,
            "SYSTEM_AVAILABLE_RESERVE_BYTES",
            None,
        ) != 384 * 1024**2
        or bound_config.get("linux_resource_policy") != dict(policy)
    ):
        raise RecoveryBootstrapError(
            "Colab v3 resource policy binding did not take effect"
        )

    return dict(result)


def _verified_recovery_inputs(
    trainer: Any,
    package: Any,
    data_root: Path,
    repository: Path,
) -> tuple[Any, ...]:
    """Bind recovery source provenance before the first dataset identity read."""
    trainer.RUN_SOURCE_FILES = tuple(
        sorted(
            set(trainer.RUN_SOURCE_FILES)
            | set(package.MIGRATION_RUNTIME_FILES)
            | {
                "phase9_geant/colab_h96_v1/matrix_recovery.py",
                "phase9_geant/colab_h96_v1/launch_matrix_recovery.py",
            }
        )
    )
    return trainer._verified_inputs(data_root, repository)


def _load_verified_bootstrap(
    repository: Path,
    args: argparse.Namespace,
):
    """Execute the exact hash-verified stdlib bootstrap source bytes."""

    archive_path = Path(os.path.abspath(args.source_bundle))
    manifest_path = Path(os.path.abspath(args.source_manifest))

    for path in (archive_path, manifest_path):
        if (
            path.is_symlink()
            or not path.is_file()
            or path.resolve(strict=True) != path
        ):
            raise RecoveryBootstrapError(
                f"Source identity file is missing or redirected: {path}"
            )

    manifest_bytes = manifest_path.read_bytes()

    if hashlib.sha256(manifest_bytes).hexdigest() != args.expected_source_manifest_sha256:
        raise RecoveryBootstrapError(
            "Source manifest differs from the externally reviewed digest"
        )

    manifest = json.loads(manifest_bytes.decode("utf-8"))

    if manifest.get("archive_sha256") != args.expected_source_bundle_sha256:
        raise RecoveryBootstrapError(
            "Source manifest does not bind the approved source archive"
        )

    archive_bytes = archive_path.read_bytes()

    if hashlib.sha256(archive_bytes).hexdigest() != args.expected_source_bundle_sha256:
        raise RecoveryBootstrapError(
            "Source archive differs from the externally reviewed digest"
        )

    entries = manifest.get("entries")

    if not isinstance(entries, list):
        raise RecoveryBootstrapError("Source manifest has no entries")

    entry = next(
        (
            item
            for item in entries
            if isinstance(item, dict)
            and item.get("path") == "phase9_geant/colab_h96_v1/launch.py"
        ),
        None,
    )

    if not isinstance(entry, dict):
        raise RecoveryBootstrapError(
            "Approved source manifest omits the stdlib bootstrap"
        )

    bootstrap_path = (
        repository / "phase9_geant" / "colab_h96_v1" / "launch.py"
    )

    current = Path(bootstrap_path.anchor)

    for component in bootstrap_path.parts[1:]:
        current = current / component
        info = current.lstat()

        if current.is_symlink() or getattr(
            current, "is_junction", lambda: False
        )():
            raise RecoveryBootstrapError(
                f"Redirected bootstrap source path: {current}"
            )

        if int(getattr(info, "st_file_attributes", 0)) & getattr(
            stat,
            "FILE_ATTRIBUTE_REPARSE_POINT",
            0x400,
        ):
            raise RecoveryBootstrapError(
                f"Reparse-point bootstrap source path: {current}"
            )

    bootstrap_bytes = bootstrap_path.read_bytes()

    if (
        len(bootstrap_bytes) != entry.get("size_bytes")
        or hashlib.sha256(bootstrap_bytes).hexdigest() != entry.get("sha256")
    ):
        raise RecoveryBootstrapError(
            "Extracted bootstrap source differs from approved manifest"
        )

    try:
        with zipfile.ZipFile(archive_path, "r") as archive:
            info = archive.getinfo(entry["path"])
            mode = (info.external_attr >> 16) & 0xFFFF

            if stat.S_ISLNK(mode) or archive.read(info) != bootstrap_bytes:
                raise RecoveryBootstrapError(
                    "Bootstrap archive member differs from approved source"
                )

    except (KeyError, zipfile.BadZipFile) as exc:
        raise RecoveryBootstrapError(
            f"Cannot verify bootstrap archive member: {exc}"
        ) from exc

    spec = importlib.util.spec_from_file_location(
        "_geant_h96_recovery_verified_bootstrap",
        bootstrap_path,
    )

    if spec is None:
        raise RecoveryBootstrapError(
            "Cannot create verified bootstrap module spec"
        )

    module = types.ModuleType(spec.name)
    module.__file__ = str(bootstrap_path)
    module.__package__ = "_geant_h96_recovery_verified_bootstrap"

    exec(
        compile(bootstrap_bytes, str(bootstrap_path), "exec"),
        module.__dict__,
    )

    return module


def _load_and_preflight(
    args: argparse.Namespace,
) -> dict[str, Any]:

    repository = Path(os.path.abspath(args.repository_root))

    if not repository.is_dir() or repository.resolve(strict=True) != repository:
        raise RecoveryBootstrapError(
            "Approved source snapshot root is missing or redirected"
        )

    bootstrap = _load_verified_bootstrap(repository, args)

    source_identity = bootstrap._verify_source_before_import(
        repository,
        args.source_bundle,
        args.source_manifest,
        expected_archive_sha256=args.expected_source_bundle_sha256,
        expected_manifest_sha256=args.expected_source_manifest_sha256,
    )

    try:
        running_relative = (
            Path(__file__)
            .resolve(strict=True)
            .relative_to(repository)
            .as_posix()
        )

    except ValueError as exc:
        raise RecoveryBootstrapError(
            "Recovery launcher did not execute from the approved snapshot"
        ) from exc

    running_entry = next(
        (
            item
            for item in source_identity["manifest"]["entries"]
            if item.get("path") == running_relative
        ),
        None,
    )

    if (
        not isinstance(running_entry, dict)
        or hashlib.sha256(
            Path(__file__).resolve(strict=True).read_bytes()
        ).hexdigest() != running_entry.get("sha256")
    ):
        raise RecoveryBootstrapError(
            "Executing recovery launcher differs from the approved source snapshot"
        )

    bootstrap._activate_system_site_packages()

    approved_paths = {
        (repository / entry["path"]).resolve(strict=True)
        for entry in source_identity["manifest"]["entries"]
        if entry["path"].endswith(".py")
    }

    package, environment, resources, trainer = (
        bootstrap._import_verified_runtime(repository, approved_paths)
    )

    matrix = importlib.import_module(
        "phase9_geant.colab_h96_v1.launch_matrix"
    )

    recovery = importlib.import_module(
        "phase9_geant.colab_h96_v1.matrix_recovery"
    )

    workflow = importlib.import_module(
        "phase9_geant.colab_h96_v1.matrix_workflow"
    )

    recovery_module = importlib.import_module(
        "phase9_geant.colab_h96_v1.matrix_recovery"
    )

    launcher_module = importlib.import_module(
        "phase9_geant.colab_h96_v1.launch_matrix_recovery"
    )

    for module in (workflow, recovery_module, launcher_module):
        if Path(module.__file__).resolve(strict=True) not in approved_paths:
            raise recovery.RecoveryError(
                f"Runtime helper is outside the verified source bundle: {module.__name__}"
            )

    environment_result = environment.verify_environment(
        require_pytest=True
    )

    policy = _activate_resource_policy(
        bootstrap,
        resources,
        trainer,
        repository,
    )

    dataset_result = package.verify_dataset_bundle(
        args.dataset_bundle,
        args.dataset_manifest,
        args.expected_dataset_bundle_sha256,
        args.expected_dataset_manifest_sha256,
    )

    data_root = repository / package.DATA_ROOT_RELATIVE
    metadata = package._read_dataset_metadata(data_root)

    if {
        item.name for item in data_root.iterdir()
    } != set(package.DATA_ARTIFACTS):
        raise recovery.RecoveryError(
            "Dataset exact train/validation allowlist mismatch"
        )

    # ------------------------------------------------------------
    # FIX:
    # Configure recovery source provenance BEFORE the initial
    # dataset identity verification.
    #
    # Previously, this was configured after verified_inputs,
    # producing different source identities at training time.
    # ------------------------------------------------------------

    # Read only approved train/validation files.
    verified_inputs = _verified_recovery_inputs(
        trainer,
        package,
        data_root,
        repository,
    )

    if (
        len(verified_inputs[2]["train"]["windows"]) != 3924
        or len(verified_inputs[2]["validation"]["windows"]) != 953
        or tuple(verified_inputs[2]) != ("train", "validation")
    ):
        raise recovery.RecoveryError(
            "Expected train/validation h96 windows or split allowlist mismatch"
        )

    current_sources = trainer._source_identity(repository)

    critical = {
        name: current_sources[name]
        for name in CRITICAL_TRAINING_SOURCES
    }

    expected_adjacency_hashes = {}

    for mode in ("topology", "identity", "shuffled"):
        graph = trainer.build_link_adjacency(
            verified_inputs[4],
            mode=mode,
            shuffle_seed=20261019,
        )

        expected_adjacency_hashes[mode] = hashlib.sha256(
            graph.contiguous().numpy().tobytes(order="C")
        ).hexdigest()

    prior = recovery.verify_completed_backup_set(
        args.completed_backup_root,
        protocol_id=trainer.PROTOCOL_ID,
        expected_config=trainer.CONFIG,
        current_critical_source_hashes=critical,
        expected_dataset_metadata_sha256=trainer._json_sha256(
            verified_inputs[3]
        ),
        expected_dataset_bundle_identity={
            "dataset_manifest_sha256": args.expected_dataset_manifest_sha256,
            "dataset_bundle_sha256": dataset_result["archive_sha256"],
        },
        expected_adjacency_hashes=expected_adjacency_hashes,
        expected_provenance=verified_inputs[5],
    )

    output_root = (
        repository
        / "data"
        / "processed"
        / recovery.RECOVERY_OUTPUT_ROOT
    )

    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(
            f"Refusing existing recovery namespace; never resume or overwrite: {output_root}"
        )

    if output_root.parent.resolve(strict=True) != output_root.parent:
        raise recovery.RecoveryError(
            "Recovery output parent is redirected"
        )

    persistent = Path(os.path.abspath(args.persistent_root))

    workflow.validate_persistent_matrix_target(persistent)

    # Recovery destination and metadata configuration.
    # RUN_SOURCE_FILES is already configured above.
    trainer.OUTPUT_ROOT_NAME = recovery.RECOVERY_OUTPUT_ROOT

    trainer.CONFIG = {
        **trainer.CONFIG,
        "matrix_experiment_id": recovery.RECOVERY_ID,
        "matrix_run_order": [
            {
                "seed": seed,
                "adjacency_mode": mode,
            }
            for seed, mode in recovery.REMAINING_RUNS
        ],
        "colab_bundle_identity": {
            "source_bundle_sha256": source_identity["archive_sha256"],
            "source_manifest_sha256": source_identity["manifest_sha256"],
            "dataset_bundle_sha256": dataset_result["archive_sha256"],
            "dataset_manifest_sha256": args.expected_dataset_manifest_sha256,
        },
        "colab_environment": environment_result,
    }

    source_hashes = trainer._source_identity(repository)

    plan = [
        {
            "seed": seed,
            "adjacency_mode": mode,
            "destination": str(
                output_root / f"h96_{mode}_seed_{seed}"
            ),
        }
        for seed, mode in recovery.REMAINING_RUNS
    ]

    return {
        "repository": repository,
        "data_root": data_root,
        "trainer": trainer,
        "matrix": matrix,
        "recovery": recovery,
        "resources": resources,
        "workflow": workflow,
        "prior": prior,
        "source_hashes": source_hashes,
        "policy": policy["policy"],
        "verified_inputs": verified_inputs,
        "output_root": output_root,
        "persistent_root": persistent,
        "source_identity": source_identity,
        "dataset_result": dataset_result,
        "metadata": metadata,
        "plan": plan,
    }


def _authorization(
    index: int,
    seed: int,
    mode: str,
) -> bool:

    phrase = (
        f"AUTHORIZE MATRIX3 RECOVERY "
        f"{index}/2 SHUFFLED SEED {seed}"
    )

    print(f"AUTH_REQUIRED: type exactly {phrase}", flush=True)

    return sys.stdin.readline().strip() == phrase


def _execute(
    prepared: dict[str, Any],
) -> dict[str, Any]:

    matrix = prepared["matrix"]
    recovery = prepared["recovery"]
    trainer = prepared["trainer"]
    repository = prepared["repository"]

    output_root = prepared["output_root"]
    persistent_root = prepared["persistent_root"]

    source_hashes = prepared["source_hashes"]
    policy = prepared["policy"]

    protocol_id = trainer.PROTOCOL_ID

    def train_one(
        seed: int,
        mode: str,
        destination: Path,
    ):
        current_policy = prepared["resources"].preflight(
            disk_path=repository
        )

        if current_policy.get("policy") != policy:
            raise recovery.RecoveryError(
                "Resource policy changed before this run"
            )

        current_inputs = trainer._verified_inputs(
            prepared["data_root"],
            repository,
        )

        if not matrix._same_input_identity(
            current_inputs,
            prepared["verified_inputs"],
        ):
            raise recovery.RecoveryError(
                "Train/validation data identity changed before this run"
            )

        if trainer._source_identity(repository) != source_hashes:
            raise recovery.RecoveryError(
                "Approved source identity changed before this run"
            )

        planned_root, planned = trainer._planned_run(
            repository,
            seed,
            mode,
        )

        if (
            planned_root != output_root
            or planned != destination
            or destination.exists()
            or destination.is_symlink()
        ):
            raise FileExistsError(
                f"Refusing existing or mismatched run destination: {destination}"
            )

        return trainer.train_one(
            prepared["data_root"],
            repository_root=repository,
            seed=seed,
            mode=mode,
            run_training=True,
            explicit_authorization=True,
            device_name="cpu",
            monitor=prepared["resources"].LinuxMemoryMonitor(),
        )

    def verify_result(
        result,
        seed,
        mode,
        destination,
    ):
        matrix._validate_run_result(
            result,
            seed,
            mode,
            protocol_id,
        )

        expected_provenance = prepared["verified_inputs"][5]

        if (
            result.get("provenance") != expected_provenance
            or result.get("dataset_metadata_sha256")
            != trainer._json_sha256(
                prepared["verified_inputs"][3]
            )
        ):
            raise recovery.RecoveryError(
                "Completed run data/provenance identity differs from recovery preflight"
            )

        configuration = {
            **dict(trainer.CONFIG),
            "seed": seed,
            "adjacency_mode": mode,
            "device": "cpu",
        }

        if result.get("configuration") != configuration:
            raise recovery.RecoveryError(
                "New run scientific/recovery configuration mismatch"
            )

        return matrix._validate_run_artifacts(
            result,
            destination,
            seed,
            mode,
            configuration,
            source_hashes,
        )

    def backup(
        result,
        seed,
        mode,
        destination,
    ):
        ids = verify_result(
            result,
            seed,
            mode,
            destination,
        )

        return prepared["workflow"].backup_verified_run(
            destination,
            persistent_root,
            run_name=f"h96_{mode}_seed_{seed}",
            expected_seed=seed,
            expected_adjacency_mode=mode,
            expected_checkpoint_sha256=ids["checkpoint_sha256"],
            expected_predictions_sha256=ids[
                "validation_predictions_sha256"
            ],
            expected_run_manifest_sha256=ids[
                "run_manifest_sha256"
            ],
        )

    def snapshot(
        sequence,
        state,
        record,
    ):
        return prepared["workflow"].persist_lifecycle_snapshot(
            persistent_root,
            sequence=sequence,
            state=state,
            record=record,
        )

    return recovery.run_remaining_experiments(
        output_root=output_root,
        persistent_root=persistent_root,
        prior_verified=prepared["prior"],
        authorize=_authorization,
        train_one=train_one,
        verify_result=verify_result,
        backup_run=backup,
        persist_snapshot=snapshot,
    )


def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    action = parser.add_mutually_exclusive_group(
        required=True
    )

    action.add_argument(
        "--preflight-only",
        action="store_true",
    )

    action.add_argument(
        "--run-remaining",
        action="store_true",
    )

    parser.add_argument(
        "--authorize-recovery",
        action="store_true",
    )

    parser.add_argument(
        "--repository-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--source-bundle",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--source-manifest",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--expected-source-bundle-sha256",
        required=True,
    )

    parser.add_argument(
        "--expected-source-manifest-sha256",
        required=True,
    )

    parser.add_argument(
        "--dataset-bundle",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--expected-dataset-bundle-sha256",
        required=True,
    )

    parser.add_argument(
        "--expected-dataset-manifest-sha256",
        required=True,
    )

    parser.add_argument(
        "--completed-backup-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--persistent-root",
        type=Path,
        required=True,
    )

    return parser


def main(
    argv: Sequence[str] | None = None,
) -> int:

    args = build_parser().parse_args(argv)

    if args.run_remaining and args.authorize_recovery is not True:
        raise PermissionError(
            "--run-remaining requires --authorize-recovery"
        )

    for name in (
        "expected_source_bundle_sha256",
        "expected_source_manifest_sha256",
        "expected_dataset_bundle_sha256",
        "expected_dataset_manifest_sha256",
    ):
        if not re.fullmatch(
            r"[0-9a-f]{64}",
            getattr(args, name),
        ):
            raise ValueError(
                f"{name} must be lowercase SHA-256"
            )

    prepared = _load_and_preflight(args)

    recovery = prepared["recovery"]

    report = {
        "status": "preflight_passed_no_training",
        "recovery_id": recovery.RECOVERY_ID,
        "verified_existing_runs": prepared["prior"]["completed_runs"],
        "remaining_runs": prepared["plan"],
        "output_root": str(prepared["output_root"]),
        "persistent_root": str(prepared["persistent_root"]),
        "resource_policy": prepared["policy"],
        "protocol_id": prepared["trainer"].PROTOCOL_ID,
        "test_access": False,
    }

    if args.preflight_only:
        print(
            json.dumps(
                report,
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    result = _execute(prepared)

    print(
        json.dumps(
            {
                "status": result["status"],
                "completed_runs": [
                    {
                        "seed": item["seed"],
                        "adjacency_mode": item["adjacency_mode"],
                        "epochs_completed": item.get(
                            "epochs_completed"
                        ),
                        "mae": item.get(
                            "validation_metrics",
                            {},
                        ).get(
                            "h96_candidate",
                            {},
                        ).get("mae"),
                        "rmse": item.get(
                            "validation_metrics",
                            {},
                        ).get(
                            "h96_candidate",
                            {},
                        ).get("rmse"),
                        "runtime_seconds": item.get(
                            "runtime_seconds"
                        ),
                        "checkpoint_sha256": item.get(
                            "checkpoint_sha256"
                        ),
                        "backup": "verified",
                    }
                    for item in result["completed_runs"]
                ],
                "combined_results_sha256": result.get(
                    "combined_results_sha256"
                ),
                "test_access": False,
            },
            indent=2,
            sort_keys=True,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
