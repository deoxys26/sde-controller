"""Authorized, train/validation-only DG-STMTL-based GEANT h=96 experiment.

Import is inert. Real training requires both explicit CLI flags and is intended
to be launched from the separately reviewed Colab workflow. The only data
loader imported is the existing fail-closed h=96 train/validation loader.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from phase9_geant import horizon96_streaming_experiment_v2 as base
from phase9_geant.colab_h96_v1 import linux_resources
from phase9_geant.safe_paths import canonical_repository_root, checked_absolute
from src.gnn.dg_stmtl_link import DGSTMTLLinkForecaster


PROTOCOL_ID = "geant-dg-stmtl-based-h96-v1"
RUNNER_ID = "geant-dg-stmtl-h96-v1"
OUTPUT_ROOT_NAME = "geant_dg_stmtl_based_h96_v1"
SEEDS = (42, 123, 2026)
HISTORY_LENGTH = 672
HORIZON = 96
LINK_COUNT = 72
SHUFFLE_SEED = 20261019  # recorded for complete protocol traceability; DG model uses true topology only
CONFIG: dict[str, Any] = {
    "protocol_id": PROTOCOL_ID,
    "runner_id": RUNNER_ID,
    "target": "single h=96 computed directed-link utilization ratio",
    "history_length": HISTORY_LENGTH,
    "horizon": HORIZON,
    "directed_link_count": LINK_COUNT,
    "ewma_alpha": 0.8,
    "ewma_steps": 5,
    "loss": "SmoothL1(beta=0.001) on original-unit residuals (prediction-EWMA vs target-EWMA)",
    "optimizer": "Adam",
    "learning_rate": 0.005,
    "batch_size": 4,
    "num_workers": 0,
    "max_epochs": 20,
    "early_stopping_patience": 4,
    "checkpoint_selection": "minimum validation h96 residual SmoothL1 only",
    "scaler_fit_scope": "existing window-occurrence-weighted h96 training histories plus h96 targets only",
    "train_bounds": [0, 8020],
    "validation_bounds": [8020, 9740],
    "expected_train_windows": 3924,
    "expected_validation_windows": 953,
    "validation_gradient_updates": False,
    "test_access": False,
    "adjacency": "input-conditioned dynamic line-graph adjacency gated with static three-slot topology prior",
    "model_adaptation": {
        "temporal_tokens": 12,
        "history_patch_length": 56,
        "group_width": 3,
        "gcn_agg_layers_per_group_block": 3,
        "graph_blocks": 2,
        "dynamic_embedding_dim": 8,
        "hidden_size": 16,
    },
    "comparison_budget_reference": base.CONFIG["protocol_id"],
}
RUN_SOURCE_FILES = tuple(sorted(set(
    base.RUN_SOURCE_FILES
    + (
        "phase9_geant/dg_stmtl_h96_experiment.py",
        "phase9_geant/dg_stmtl_h96_workflow.py",
        "phase9_geant/colab_h96_v1/__init__.py",
        "phase9_geant/colab_h96_v1/linux_resources.py",
        "src/gnn/dg_stmtl_link.py",
        "docs/geant_dg_stmtl_h96_v1.md",
    )
)))


class DGSTMTLError(ValueError):
    """Protocol, provenance, path, or lifecycle invariant failed closed."""


def _canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_identity(repository: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative in RUN_SOURCE_FILES:
        path = checked_absolute(repository / relative)
        if not path.is_file() or path.resolve(strict=True) != path:
            raise DGSTMTLError(f"Missing or redirected DG-STMTL source: {relative}")
        hashes[relative] = _sha256_file(path)
    return hashes


def _verified_inputs(data_root: str | Path, repository_root: str | Path):
    """Use the existing v2 verifier and bind the additional DG-STMTL sources."""
    values = base._verified_inputs(data_root, repository_root)
    repository, root, splits, metadata, edges, provenance, base_hashes = values
    current = source_identity(repository)
    for name, digest in base_hashes.items():
        if current.get(name) != digest:
            raise DGSTMTLError(f"Base h96 source identity changed: {name}")
    if tuple(splits) != ("train", "validation"):
        raise DGSTMTLError("Only train and validation splits are permitted")
    if len(edges) != LINK_COUNT or len(set(edges)) != LINK_COUNT:
        raise DGSTMTLError("Verified topology must have 72 unique ordered directed links")
    if len(splits["train"]["windows"]) != 3924 or len(splits["validation"]["windows"]) != 953:
        raise DGSTMTLError("Verified window counts differ from the frozen h96 protocol")
    return repository, root, splits, metadata, edges, provenance, current


def _planned_run(repository: Path, seed: int) -> tuple[Path, Path]:
    if seed not in SEEDS:
        raise ValueError("seed is outside the fixed DG-STMTL seed list")
    processed = repository / "data" / "processed"
    if not processed.is_dir() or processed.resolve(strict=True) != processed:
        raise DGSTMTLError(f"Canonical processed-data root is unavailable: {processed}")
    root = checked_absolute(processed / OUTPUT_ROOT_NAME, allow_missing=True)
    if root.parent != processed or (root.exists() and (not root.is_dir() or root.resolve(strict=True) != root)):
        raise DGSTMTLError("DG-STMTL output root is redirected or outside data/processed")
    run = checked_absolute(root / f"dg_stmtl_topology_seed_{seed}", allow_missing=True)
    if run.parent != root or run.exists() or run.is_symlink():
        raise FileExistsError(f"Refusing existing DG-STMTL run destination: {run}")
    allowed = {f"dg_stmtl_topology_seed_{value}" for value in SEEDS}
    if root.exists() and {entry.name for entry in root.iterdir()} - allowed:
        raise DGSTMTLError("Unexpected entry in DG-STMTL output namespace")
    return root, run


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=False, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    except BaseException:
        try:
            os.unlink(name)
        except OSError:
            pass
        raise


def _atomic_npy(path: Path, array: np.ndarray) -> str:
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        with open(name, "wb") as stream:
            np.save(stream, array, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    except BaseException:
        try:
            os.unlink(name)
        except OSError:
            pass
        raise
    return _sha256_file(path)


def _atomic_checkpoint(payload: object, path: Path) -> str:
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    except BaseException:
        try:
            os.unlink(name)
        except OSError:
            pass
        raise
    return _sha256_file(path)


def _load_checkpoint_verified(model: DGSTMTLLinkForecaster, path: Path, digest: str, *, seed: int, config_hash: str) -> None:
    raw = path.read_bytes()
    if not raw or hashlib.sha256(raw).hexdigest() != digest:
        raise DGSTMTLError("Checkpoint is empty or its SHA-256 changed")
    import io

    payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    if payload.get("protocol_id") != PROTOCOL_ID or payload.get("seed") != seed or payload.get("configuration_sha256") != config_hash:
        raise DGSTMTLError("Checkpoint identity does not match this DG-STMTL run")
    model.load_state_dict(payload["state_dict"], strict=True)


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _train_epoch(model, dataset, scaler, optimizer, seed: int, epoch: int, monitor) -> tuple[float, dict[str, float | int]]:
    model.train()
    order = np.random.default_rng(seed + epoch).permutation(len(dataset)).tolist()
    loader = DataLoader(
        Subset(dataset, order), batch_size=int(CONFIG["batch_size"]), shuffle=False,
        num_workers=0, pin_memory=False,
    )
    loss_sum = 0.0
    pair_count = 0
    durations: list[float] = []
    for raw_history, raw_target in loader:
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        history = (raw_history - np.float32(scaler.mean)) / np.float32(scaler.scale)
        target = raw_target.to(torch.float64)
        forecast = model(history).to(torch.float64) * scaler.scale + scaler.mean
        if forecast.shape != target.shape or not torch.isfinite(forecast).all():
            raise DGSTMTLError("Training forecast shape or values are invalid")
        baseline_norm = model.ewma(history)
        baseline = baseline_norm.to(torch.float64) * scaler.scale + scaler.mean
        loss = F.smooth_l1_loss(forecast - baseline, target - baseline, beta=0.001)
        if not torch.isfinite(loss):
            raise DGSTMTLError("Training residual loss is nonfinite")
        loss.backward()
        grads = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        if not grads or not all(torch.isfinite(grad).all() for grad in grads):
            raise DGSTMTLError("Training gradients are missing or nonfinite")
        optimizer.step()
        count = int(target.numel())
        loss_sum += float(loss.detach()) * count
        pair_count += count
        optimizer.zero_grad(set_to_none=True)
        durations.append(time.perf_counter() - started)
        del raw_history, raw_target, history, target, forecast, baseline_norm, baseline, loss, grads
        monitor.sample(f"train_epoch_{epoch}_batch")
    if pair_count == 0:
        raise DGSTMTLError("Training loader emitted no pairs")
    return loss_sum / pair_count, {
        "optimizer_steps": len(durations),
        "median_optimizer_step_seconds": float(statistics.median(durations)),
        "p95_optimizer_step_seconds": float(np.percentile(durations, 95)),
    }


def _evaluate(model, loader, scaler, monitor, *, collect: bool) -> tuple[float, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    model.eval()
    total = 0.0
    pairs = 0
    prediction_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    ewma_rows: list[np.ndarray] = []
    with torch.no_grad():
        for raw_history, raw_target in loader:
            history = (raw_history - np.float32(scaler.mean)) / np.float32(scaler.scale)
            target = raw_target.to(torch.float64)
            forecast = model(history).to(torch.float64) * scaler.scale + scaler.mean
            baseline = model.ewma(history).to(torch.float64) * scaler.scale + scaler.mean
            if forecast.shape != target.shape or not torch.isfinite(forecast).all() or not torch.isfinite(target).all():
                raise DGSTMTLError("Validation prediction/target is nonfinite or misaligned")
            loss = F.smooth_l1_loss(forecast - baseline, target - baseline, beta=0.001)
            count = int(target.numel())
            total += float(loss) * count
            pairs += count
            if collect:
                prediction_rows.append(forecast.cpu().numpy().astype(np.float64))
                target_rows.append(target.cpu().numpy().astype(np.float64))
                ewma_rows.append(baseline.cpu().numpy().astype(np.float64))
            del raw_history, raw_target, history, target, forecast, baseline, loss
            monitor.sample("validation_batch")
    if pairs == 0:
        raise DGSTMTLError("Validation loader emitted no target pairs")
    if not collect:
        return total / pairs, None, None, None
    return (
        total / pairs,
        np.concatenate(prediction_rows, axis=0),
        np.concatenate(target_rows, axis=0),
        np.concatenate(ewma_rows, axis=0),
    )


def _synthetic_edges() -> tuple[tuple[str, str], ...]:
    return tuple((str(i), str(j)) for i in range(9) for j in range(9) if i != j)


def self_check() -> dict[str, Any]:
    """Bounded synthetic-only protocol/model check; never touches project data."""
    torch.manual_seed(42)
    model = DGSTMTLLinkForecaster(_synthetic_edges())
    x = torch.linspace(0.0, 1.0, 672 * 72, dtype=torch.float32).reshape(1, 672, 72)
    with torch.no_grad():
        output = model(x)
        baseline = model.ewma(x)
    if output.shape != (1, 72) or not torch.isfinite(output).all() or not torch.equal(output, baseline):
        raise DGSTMTLError("Synthetic DG-STMTL shape/finite/EWMA initialization check failed")
    return {"status": "pass", "scope": "synthetic tensors only", "output_shape": list(output.shape),
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "test_access": False, "training": False}


def train_one(
    data_root: str | Path,
    *,
    repository_root: str | Path,
    seed: int,
    run_training: bool = False,
    explicit_authorization: bool = False,
    monitor=None,
) -> dict[str, Any]:
    if run_training is not True or explicit_authorization is not True:
        raise PermissionError("DG-STMTL training requires --run-training and --authorize-dg-stmtl")
    if seed not in SEEDS:
        raise ValueError("Only the fixed seeds 42, 123, and 2026 are permitted")
    repository = canonical_repository_root(repository_root)
    output_root, output = _planned_run(repository, seed)
    resource_preflight = linux_resources.preflight(disk_path=repository)
    monitor = monitor or linux_resources.LinuxMemoryMonitor()
    startup_snapshot = monitor.sample("startup_before_output")
    (repository, verified_root, splits, dataset_metadata, edge_order,
     provenance, source_hashes) = _verified_inputs(data_root, repository)
    scaler = base.lazy.fit_window_scaler(splits["train"]["series"], splits["train"]["windows"], split_start=0)
    if scaler.fit_scope != "h=96 train histories plus h=96 targets; window-occurrence weighted":
        raise DGSTMTLError("Training-only scaler scope mismatch")
    config = {**CONFIG, "seed": seed, "adjacency_mode": "hybrid_topology_dynamic", "device": "cpu"}
    config_hash = _canonical_json_sha256(config)
    current_sources = source_identity(repository)
    if current_sources != source_hashes:
        raise DGSTMTLError("Source identity changed during input preflight")

    if not output_root.exists():
        output_root.mkdir(parents=False, exist_ok=False)
        if checked_absolute(output_root) != output_root or output_root.resolve(strict=True) != output_root:
            raise DGSTMTLError("New DG-STMTL output root failed containment verification")
    output.mkdir(parents=False, exist_ok=False)
    if checked_absolute(output) != output or output.resolve(strict=True) != output or output.parent.resolve(strict=True) != output_root:
        raise DGSTMTLError("DG-STMTL run directory failed post-creation containment verification")

    manifest: dict[str, Any] = {
        "status": "incomplete", "protocol_id": PROTOCOL_ID, "runner_id": RUNNER_ID,
        "seed": seed, "adjacency_mode": "hybrid_topology_dynamic", "configuration": config,
        "configuration_sha256": config_hash, "repository_root": str(repository),
        "data_root": str(verified_root), "dataset_metadata_sha256": _canonical_json_sha256(dataset_metadata),
        "provenance": provenance,
        "permitted_array_file_hashes": {
            name: item.get("file_sha256") for name, item in dataset_metadata.get("array_files", {}).items()
            if name in base.lazy.DATA_ARRAY_NAMES
        },
        "source_hashes": source_hashes,
        "normalization": scaler.__dict__,
        "topology": {
            "node_count": 22, "undirected_link_count": 36, "directed_link_count": LINK_COUNT,
            "ordered_edge_sha256": base.builder.approved._edge_digest(edge_order),
            "directed_edges_interleaved": [list(edge) for edge in edge_order],
            "adjacency": "adapted link line graph; caller edge index is preserved",
        },
        "training_windows": len(splits["train"]["windows"]),
        "validation_windows": len(splits["validation"]["windows"]),
        "resource_preflight": resource_preflight,
        "memory_initial_snapshot": startup_snapshot,
        "memory_monitor": monitor.summary(),
        "checkpoint_selection": CONFIG["checkpoint_selection"],
        "test_access": {"permitted": False, "paths_or_loaders": "none"},
    }
    if set(manifest["permitted_array_file_hashes"]) != set(base.lazy.DATA_ARRAY_NAMES):
        raise DGSTMTLError("Train/validation array fingerprint inventory is incomplete")
    _atomic_json(output / "run.json", manifest)
    started = time.perf_counter()
    history_rows: list[dict[str, Any]] = []
    try:
        base._seed_all(seed)
        model = DGSTMTLLinkForecaster(edge_order, history_length=HISTORY_LENGTH, hidden_size=16, alpha=0.8).cpu()
        optimizer = torch.optim.Adam(model.parameters(), lr=float(CONFIG["learning_rate"]))
        train_data = base.lazy.LazyH96Dataset(splits["train"]["series"], splits["train"]["windows"], split_start=0)
        validation_data = base.lazy.LazyH96Dataset(splits["validation"]["series"], splits["validation"]["windows"], split_start=8020)
        validation_loader = DataLoader(validation_data, batch_size=int(CONFIG["batch_size"]), shuffle=False, num_workers=0, pin_memory=False)
        manifest["model_parameter_count"] = sum(parameter.numel() for parameter in model.parameters())
        _atomic_json(output / "run.json", manifest)
        best_loss, best_epoch, stale = float("inf"), 0, 0
        checkpoint = output / "best.pt"
        for epoch in range(1, int(CONFIG["max_epochs"]) + 1):
            epoch_start = time.perf_counter()
            train_loss, step_stats = _train_epoch(model, train_data, scaler, optimizer, seed, epoch, monitor)
            val_loss, _, _, _ = _evaluate(model, validation_loader, scaler, monitor, collect=False)
            if not math.isfinite(val_loss):
                raise DGSTMTLError("Validation selection objective is nonfinite")
            if val_loss < best_loss:
                if source_identity(repository) != source_hashes or _canonical_json_sha256(config) != config_hash:
                    raise DGSTMTLError("Source/configuration changed before checkpoint write")
                best_loss, best_epoch, stale = val_loss, epoch, 0
                digest = _atomic_checkpoint({
                    "state_dict": model.state_dict(), "seed": seed, "best_epoch": epoch,
                    "protocol_id": PROTOCOL_ID, "configuration_sha256": config_hash,
                    "ordered_edge_sha256": manifest["topology"]["ordered_edge_sha256"],
                }, checkpoint)
                manifest["checkpoint_sha256_so_far"] = digest
            else:
                stale += 1
            history_rows.append({
                "epoch": epoch, "train_residual_smooth_l1": train_loss,
                "validation_residual_smooth_l1": val_loss,
                "epoch_seconds": time.perf_counter() - epoch_start,
                "optimizer_step_stats": step_stats,
            })
            manifest.update({"epoch_history": history_rows, "best_epoch_so_far": best_epoch,
                             "best_validation_loss_so_far": best_loss, "memory_monitor": monitor.summary()})
            _atomic_json(output / "run.json", manifest)
            if stale >= int(CONFIG["early_stopping_patience"]):
                break
        if best_epoch == 0 or not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
            raise DGSTMTLError("No nonempty best-validation checkpoint was finalized")
        current = _verified_inputs(data_root, repository)
        if (current[1] != verified_root or current[3] != dataset_metadata or current[4] != edge_order
                or current[5] != provenance or current[6] != source_hashes):
            raise DGSTMTLError("Dataset, provenance, topology, or source identity drifted before reload")
        checkpoint_hash = _sha256_file(checkpoint)
        _load_checkpoint_verified(model, checkpoint, checkpoint_hash, seed=seed, config_hash=config_hash)
        val_objective, predictions, targets, ewma = _evaluate(model, validation_loader, scaler, monitor, collect=True)
        if not math.isclose(val_objective, best_loss, rel_tol=1e-6, abs_tol=1e-12):
            raise DGSTMTLError("Strict checkpoint reload changed selected validation loss")
        assert predictions is not None and targets is not None and ewma is not None
        metrics = {"dg_stmtl_based": base.v1._metric_values(targets, predictions),
                   "ewma_alpha_0.8": base.v1._metric_values(targets, ewma)}
        prediction_hash = _atomic_npy(output / "validation_predictions.npy", predictions.astype(np.float64, copy=False))
        if source_identity(repository) != source_hashes or _canonical_json_sha256(config) != config_hash:
            raise DGSTMTLError("Source/configuration changed before completion")
        if {entry.name for entry in output.iterdir()} != {"run.json", "best.pt", "validation_predictions.npy"}:
            raise DGSTMTLError("DG-STMTL run artifact allowlist mismatch")
        manifest.update({
            "status": "complete", "best_epoch": best_epoch,
            "best_validation_residual_smooth_l1": best_loss,
            "epochs_completed": len(history_rows),
            "early_stopping": {"patience": CONFIG["early_stopping_patience"],
                               "stopped_early": len(history_rows) < CONFIG["max_epochs"]},
            "epoch_history": history_rows, "validation_metrics": metrics,
            "checkpoint": "best.pt", "checkpoint_sha256": checkpoint_hash,
            "checkpoint_reload_strict": True,
            "validation_predictions": "validation_predictions.npy",
            "validation_predictions_sha256": prediction_hash,
            "runtime_seconds": time.perf_counter() - started,
            "memory_monitor": monitor.summary(),
            "interpretation": "validation-only; DG-STMTL-based single-target adaptation, not paper reproduction; not held-out generalization",
        })
        _atomic_json(output / "run.json", manifest)
    except BaseException as exc:
        manifest.update({"status": "failed", "failure": f"{type(exc).__name__}: {exc}",
                         "epoch_history": history_rows, "memory_monitor": monitor.summary()})
        try:
            _atomic_json(output / "run.json", manifest)
        except BaseException:
            pass
        raise
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true", help="Run only a synthetic model check; no data access")
    parser.add_argument("--run-training", action="store_true")
    parser.add_argument("--authorize-dg-stmtl", action="store_true")
    parser.add_argument("--repository-root", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_check:
        if args.run_training or args.authorize_dg_stmtl:
            raise SystemExit("--self-check cannot be combined with training flags")
        print(json.dumps(self_check(), indent=2))
        return 0
    if not args.run_training or not args.authorize_dg_stmtl:
        raise SystemExit("No default action. Real training requires --run-training --authorize-dg-stmtl")
    if args.repository_root is None or args.data_root is None or args.seed is None:
        raise SystemExit("--repository-root, --data-root, and --seed are required")
    result = train_one(args.data_root, repository_root=args.repository_root, seed=args.seed,
                       run_training=True, explicit_authorization=True)
    print(json.dumps({key: result.get(key) for key in (
        "status", "seed", "best_epoch", "best_validation_residual_smooth_l1",
        "validation_metrics", "runtime_seconds", "checkpoint_sha256",
    )}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - real run requires separate review/authorization
    raise SystemExit(main())


__all__ = ["CONFIG", "DGSTMTLError", "OUTPUT_ROOT_NAME", "PROTOCOL_ID", "RUN_SOURCE_FILES",
           "SEEDS", "source_identity", "train_one", "self_check"]
