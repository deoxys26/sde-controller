"""Validation-only audit of completed GEANT v2 B-U checkpoints.

This evaluator reads only X_validation.npy, y_validation.npy, the three
explicit full-run manifests/checkpoints, and validation/protocol metadata. It
does not load training arrays or any test-split array or label.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from phase9_geant.parse_geant import parse_topology
from phase9_geant.split_access import load_permitted_split_array
from phase9_geant.train_bu_v2 import (
    PROTOCOL_ID,
    TRAINING,
    V2_DATA_ROOT,
    Normalization,
    build_edge_index,
    make_bu_sequence,
    normalize_targets,
)
from src.gnn.model import EdgeQoSForecaster


SCRIPT_VERSION = "1.0.0"
SEEDS = (42, 123, 2026)
HISTORY_LENGTH = 5
VALIDATION_BOUNDS = (8020, 9740)
EWMA_ALPHA = 0.8
AR5_COEFFICIENTS = (-0.01057, 0.07411, 0.13645, 0.13489, 0.65767)
AR5_INTERCEPT = 0.000212
OUTPUT_DIR = V2_DATA_ROOT / "neural_bu_v2_validation_audit"
BASELINE_REPORT = V2_DATA_ROOT / "baselines_validation.json"
X_VALIDATION = V2_DATA_ROOT / "X_validation.npy"
Y_VALIDATION = V2_DATA_ROOT / "y_validation.npy"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    if target.shape != prediction.shape:
        raise ValueError("Prediction and target shapes differ")
    if not np.isfinite(target).all():
        raise ValueError("Validation target contains nonfinite values; refusing to alter coverage")
    if not np.isfinite(prediction).all():
        raise ValueError("Validation prediction contains nonfinite values")
    errors = prediction.astype(np.float64) - target.astype(np.float64)
    return {
        "mae": float(np.mean(np.abs(errors), dtype=np.float64)),
        "rmse": float(np.sqrt(np.mean(np.square(errors), dtype=np.float64))),
        "evaluated_scalar_targets": int(target.size),
        "total_scalar_targets": int(target.size),
        "nonfinite_target_count": 0,
        "nonfinite_prediction_count": 0,
    }


def _write_atomic(path: Path, content: str) -> None:
    temporary: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        temporary = Path(name)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
        raise


def _manifest_for_seed(seed: int) -> tuple[dict[str, Any], Path]:
    run_dir = V2_DATA_ROOT / f"neural_bu_v2_full_seed_{seed}"
    manifest_path = run_dir / "run_metadata.json"
    checkpoint_path = run_dir / "best_validation.pt"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing run manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or manifest.get("mode") != "full":
        raise ValueError(f"Seed {seed} is not a completed full run")
    if manifest.get("seed") != seed or manifest.get("protocol_id") != PROTOCOL_ID:
        raise ValueError(f"Seed {seed} manifest identity or protocol mismatch")
    if manifest.get("candidate") != "B-U" or manifest.get("model") != "EdgeQoSForecaster (unchanged)":
        raise ValueError(f"Seed {seed} manifest does not identify the expected B-U model")
    if manifest.get("model_settings") != TRAINING:
        raise ValueError(f"Seed {seed} model settings differ from the fixed v2 protocol")
    if manifest.get("history_length") != HISTORY_LENGTH or manifest.get("prediction_horizon") != 1:
        raise ValueError(f"Seed {seed} window configuration mismatch")
    expected_paths = [str(V2_DATA_ROOT / name) for name in (
        "X_train.npy", "y_train.npy", "X_validation.npy", "y_validation.npy"
    )]
    if manifest.get("array_paths") != expected_paths:
        raise ValueError(f"Seed {seed} manifest array paths do not match the fixed v2 inputs")
    normalization = manifest.get("normalization", {})
    if normalization.get("fit_scope") != "all scalar values in X_train.npy and y_train.npy only":
        raise ValueError(f"Seed {seed} normalization provenance is not training-only")
    if manifest.get("data_split", {}).get("test") != "not loaded or evaluated":
        raise ValueError(f"Seed {seed} manifest does not preserve the sealed test split")
    saved_checkpoint = Path(str(manifest.get("checkpoint", ""))).resolve()
    if saved_checkpoint != checkpoint_path.resolve():
        raise ValueError(f"Seed {seed} checkpoint path differs from its expected run artifact")
    if not checkpoint_path.is_file() or checkpoint_path.stat().st_size <= 0:
        raise FileNotFoundError(f"Missing or empty checkpoint: {checkpoint_path}")
    return manifest, checkpoint_path


def _evaluate_checkpoint(
    checkpoint_path: Path,
    manifest: dict[str, Any],
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    edge_index: torch.Tensor,
    node_count: int,
) -> tuple[np.ndarray, float]:
    norm_record = manifest["normalization"]
    normalization = Normalization(mean=float(norm_record["mean"]), scale=float(norm_record["scale"]))
    if not np.isfinite(normalization.mean) or not np.isfinite(normalization.scale) or normalization.scale <= 0:
        raise ValueError("Invalid recorded train-only normalization parameters")

    model = EdgeQoSForecaster(
        edge_feature_count=1,
        target_count=1,
        hidden_channels=TRAINING["hidden_channels"],
        gru_hidden_size=TRAINING["gru_hidden_size"],
    )
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError("Checkpoint does not contain a model state dictionary")
    model.load_state_dict(state, strict=True)
    model.eval()

    criterion = nn.SmoothL1Loss(reduction="sum")
    predicted_batches: list[np.ndarray] = []
    loss_sum = 0.0
    scalar_count = 0
    batch_size = int(TRAINING["batch_size"])
    with torch.no_grad():
        for begin in range(0, len(x_validation), batch_size):
            end = min(begin + batch_size, len(x_validation))
            sequence = make_bu_sequence(
                x_validation[begin:end], edge_index, node_count, normalization
            )
            standardized_prediction = model(sequence).reshape(end - begin, edge_index.shape[1], 1)
            standardized_target = normalize_targets(y_validation[begin:end], normalization)
            if not torch.isfinite(standardized_prediction).all():
                raise ValueError("Model produced nonfinite validation predictions")
            loss_sum += float(criterion(standardized_prediction, standardized_target))
            scalar_count += standardized_target.numel()
            predicted_batches.append(standardized_prediction.squeeze(-1).cpu().numpy())

    if scalar_count != y_validation.size:
        raise ValueError("Validation loss coverage does not equal the target scalar count")
    validation_smooth_l1 = loss_sum / scalar_count
    recomputed_raw_prediction = np.concatenate(predicted_batches, axis=0).astype(np.float64)
    recomputed_raw_prediction = recomputed_raw_prediction * normalization.scale + normalization.mean
    recorded_loss = float(manifest["best_validation_loss"])
    if not np.isclose(validation_smooth_l1, recorded_loss, rtol=0.0, atol=1e-7):
        raise ValueError(
            f"Checkpoint validation loss {validation_smooth_l1} does not match manifest {recorded_loss}"
        )
    history = manifest.get("history", [])
    if not history:
        raise ValueError("Run manifest has no validation-loss history")
    best_history = min(history, key=lambda row: float(row["validation_loss"]))
    if int(best_history["epoch"]) != int(manifest["best_validation_epoch"]):
        raise ValueError("Manifest best epoch is inconsistent with its validation history")
    if not np.isclose(float(best_history["validation_loss"]), recorded_loss, rtol=0.0, atol=1e-12):
        raise ValueError("Manifest best validation loss is inconsistent with its history")
    return recomputed_raw_prediction, validation_smooth_l1


def run() -> Path:
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite validation audit: {OUTPUT_DIR}")
    if not X_VALIDATION.is_file() or not Y_VALIDATION.is_file():
        raise FileNotFoundError("Required validation arrays are missing")
    if not BASELINE_REPORT.is_file():
        raise FileNotFoundError(f"Missing existing validation baseline report: {BASELINE_REPORT}")

    v2_metadata = json.loads((V2_DATA_ROOT / "metadata.json").read_text(encoding="utf-8"))
    if v2_metadata.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("v2 metadata protocol ID mismatch")
    if v2_metadata.get("history_steps") != HISTORY_LENGTH or v2_metadata.get("prediction_horizon") != 1:
        raise ValueError("v2 metadata history or horizon mismatch")
    if v2_metadata.get("split_bounds", {}).get("validation") != list(VALIDATION_BOUNDS):
        raise ValueError("v2 metadata validation split bounds mismatch")

    start_metadata = json.loads((V2_DATA_ROOT / "window_starts.json").read_text(encoding="utf-8"))
    validation_starts = start_metadata.get("validation")
    if not isinstance(validation_starts, list) or not validation_starts:
        raise ValueError("Validation window-start metadata is missing")
    expected_starts = list(range(VALIDATION_BOUNDS[0], VALIDATION_BOUNDS[1] - HISTORY_LENGTH))
    if validation_starts != expected_starts:
        raise ValueError("Validation window starts are not the expected consecutive split-local sequence")
    target_indices = [int(start) + HISTORY_LENGTH for start in validation_starts]

    baseline = json.loads(BASELINE_REPORT.read_text(encoding="utf-8"))
    if baseline.get("protocol_id") != PROTOCOL_ID or baseline.get("split") != "validation":
        raise ValueError("Existing baseline report protocol/split mismatch")
    expected_baseline_inputs = [
        "data/processed/geant_validity_v2/X_validation.npy",
        "data/processed/geant_validity_v2/y_validation.npy",
    ]
    if baseline.get("input_artifacts") != expected_baseline_inputs:
        raise ValueError("Existing baseline report does not use the same validation arrays")

    manifests: dict[int, dict[str, Any]] = {}
    checkpoints: dict[int, Path] = {}
    common_edge_order: list[dict[str, Any]] | None = None
    common_normalization: dict[str, Any] | None = None
    for seed in SEEDS:
        manifest, checkpoint = _manifest_for_seed(seed)
        manifests[seed] = manifest
        checkpoints[seed] = checkpoint
        if common_edge_order is None:
            common_edge_order = manifest.get("directed_edge_order")
            common_normalization = manifest.get("normalization")
        elif manifest.get("directed_edge_order") != common_edge_order:
            raise ValueError("Seed manifests use different directed-link ordering")
        if manifest.get("normalization") != common_normalization:
            raise ValueError("Seed manifests use different normalization parameters/provenance")

    nodes, undirected_links = parse_topology(
        str(Path(__file__).resolve().parents[1] / "data" / "raw" / "geant_topology" / "geant.txt")
    )
    edge_index, node_order, edges = build_edge_index(nodes, undirected_links)
    runtime_edge_order = [
        {"index": index, "source": source, "destination": destination}
        for index, (source, destination) in enumerate(edges)
    ]
    if runtime_edge_order != common_edge_order:
        raise ValueError("Runtime topology edge order differs from the saved validation arrays' provenance")

    # The only array loads in this evaluator are the two validation files.
    x_validation = load_permitted_split_array(ARRAY_ROOT, "X_validation.npy")
    y_validation = load_permitted_split_array(ARRAY_ROOT, "y_validation.npy")
    if x_validation.dtype != np.float32 or y_validation.dtype != np.float32:
        raise ValueError("Validation arrays must retain the v2 float32 dtype")
    if x_validation.shape != (len(validation_starts), HISTORY_LENGTH, len(edges)):
        raise ValueError(f"Unexpected validation-history shape: {x_validation.shape}")
    if y_validation.shape != (len(validation_starts), len(edges)):
        raise ValueError(f"Unexpected validation-target shape: {y_validation.shape}")
    if not np.isfinite(x_validation).all() or not np.isfinite(y_validation).all():
        raise ValueError("Validation arrays contain nonfinite values; refusing silent filtering/imputation")
    expected_shapes = {"X_validation": list(x_validation.shape), "y_validation": list(y_validation.shape)}
    for seed, manifest in manifests.items():
        if any(manifest.get("array_shapes", {}).get(k) != v for k, v in expected_shapes.items()):
            raise ValueError(f"Seed {seed} manifest validation shapes differ from loaded arrays")
        if manifest.get("sample_counts", {}).get("validation") != len(validation_starts):
            raise ValueError(f"Seed {seed} manifest validation window count mismatch")

    baseline_metrics = baseline["metrics"]
    ewma_weights = EWMA_ALPHA * (1.0 - EWMA_ALPHA) ** (HISTORY_LENGTH - 1 - np.arange(HISTORY_LENGTH, dtype=np.float64))
    ewma_weights /= ewma_weights.sum()
    baseline_predictions = {
        "persistence": x_validation[:, -1, :].astype(np.float64),
        "ewma": np.tensordot(x_validation.astype(np.float64), ewma_weights, axes=([1], [0])),
        "ar5": np.tensordot(
            x_validation.astype(np.float64),
            np.asarray(AR5_COEFFICIENTS, dtype=np.float64),
            axes=([1], [0]),
        ) + AR5_INTERCEPT,
    }
    computed_baselines = {name: _metrics(y_validation, prediction) for name, prediction in baseline_predictions.items()}
    for name, metrics in computed_baselines.items():
        recorded = baseline_metrics[name]
        if metrics["evaluated_scalar_targets"] != recorded["evaluated_scalar_targets"]:
            raise ValueError(f"Baseline report coverage mismatch for {name}")
        if not np.isclose(metrics["mae"], float(recorded["mae"]), rtol=0.0, atol=1e-12):
            raise ValueError(f"Baseline report MAE mismatch for {name}")
        if not np.isclose(metrics["rmse"], float(recorded["rmse"]), rtol=0.0, atol=1e-12):
            raise ValueError(f"Baseline report RMSE mismatch for {name}")

    torch.set_num_threads(1)
    neural_results: dict[str, Any] = {}
    for seed in SEEDS:
        manifest = manifests[seed]
        prediction, recomputed_loss = _evaluate_checkpoint(
            checkpoints[seed], manifest, x_validation, y_validation, edge_index, len(node_order)
        )
        neural_results[str(seed)] = {
            "checkpoint": str(checkpoints[seed].resolve()),
            "checkpoint_sha256": _sha256(checkpoints[seed]),
            "best_validation_epoch": int(manifest["best_validation_epoch"]),
            "recorded_best_validation_smooth_l1_standardized": float(manifest["best_validation_loss"]),
            "recomputed_checkpoint_validation_smooth_l1_standardized": float(recomputed_loss),
            "epochs_completed": int(manifest["epochs_completed"]),
            "early_stopping_triggered": bool(manifest["early_stopping"]["triggered"]),
            "training_seconds": float(manifest["training_seconds"]),
            "metrics": _metrics(y_validation, prediction),
        }

    mae_values = [neural_results[str(seed)]["metrics"]["mae"] for seed in SEEDS]
    rmse_values = [neural_results[str(seed)]["metrics"]["rmse"] for seed in SEEDS]
    aggregate = {
        "mae_mean_across_seeds": float(np.mean(mae_values, dtype=np.float64)),
        "mae_sample_std_across_seeds": float(np.std(mae_values, ddof=1, dtype=np.float64)),
        "rmse_mean_across_seeds": float(np.mean(rmse_values, dtype=np.float64)),
        "rmse_sample_std_across_seeds": float(np.std(rmse_values, ddof=1, dtype=np.float64)),
    }
    common = {
        "protocol_id": PROTOCOL_ID,
        "split": "validation",
        "validation_observation_block_half_open": list(VALIDATION_BOUNDS),
        "history_start_indices": [validation_starts[0], validation_starts[-1]],
        "target_indices_inclusive": [target_indices[0], target_indices[-1]],
        "history_length": HISTORY_LENGTH,
        "prediction_horizon": 1,
        "validation_windows": len(validation_starts),
        "directed_links": int(x_validation.shape[2]),
        "total_scalar_targets": int(y_validation.size),
        "finite_target_count": int(np.isfinite(y_validation).sum()),
        "finite_input_count": int(np.isfinite(x_validation).sum()),
        "mask": "all values must be finite; no invalid values were filtered or imputed",
        "metric_units": "dimensionless computed link-utilization ratio",
        "metric_formulas": {
            "MAE": "mean(abs(prediction - target)) in float64 over all finite scalar target pairs",
            "RMSE": "sqrt(mean((prediction - target)^2)) in float64 over all finite scalar target pairs",
            "smooth_l1": "validation checkpoint objective in standardized utilization space; not interchangeable with MAE/RMSE",
        },
        "normalization": {
            "mean": float(common_normalization["mean"]),
            "scale": float(common_normalization["scale"]),
            "fit_scope": common_normalization["fit_scope"],
            "application": "model inputs and targets standardized using each run's recorded training-only scaler; predictions inverse-transformed before MAE/RMSE",
        },
        "model": "EdgeQoSForecaster (unchanged), edge_feature_count=1, target_count=1, hidden_channels=16, GRU hidden=16",
        "checkpoint_selection": "retained best checkpoint selected by validation SmoothL1 loss only; recomputed loss checked against manifest",
        "array_artifacts_loaded": [str(X_VALIDATION), str(Y_VALIDATION)],
        "test_split_accessed": False,
    }
    results = {
        "script": {
            "name": "phase9_geant/evaluate_bu_v2_validation_audit.py",
            "version": SCRIPT_VERSION,
            "successful_command": "python -m phase9_geant.evaluate_bu_v2_validation_audit",
        },
        "protocol": common,
        "baseline_parameters": {
            "persistence": "last of five observed validation history steps",
            "ewma": {"alpha": EWMA_ALPHA, "weights_oldest_to_newest": ewma_weights.tolist(), "selection": "fixed existing protocol; not tuned on validation"},
            "ar5": {"coefficients_oldest_to_newest": list(AR5_COEFFICIENTS), "intercept": AR5_INTERCEPT, "provenance": "fixed rounded legacy distance-series coefficients; not refit for v2"},
        },
        "baselines": computed_baselines,
        "neural_by_seed": neural_results,
        "neural_aggregate_across_seeds": aggregate,
        "baseline_source_report": str(BASELINE_REPORT),
        "limitations": [
            "Validation data was used for checkpoint selection, so these metrics are descriptive validation results, not an unbiased final generalization estimate.",
            "The fixed AR(5) coefficients were originally fit on legacy training windows and were not refit for the v2 protocol.",
            "Seed standard deviations describe variability across three training seeds, not uncertainty across independent traffic periods.",
            "Results concern forecasting of computed link utilization from real GEANT traffic matrices; they are not measured live network telemetry.",
        ],
    }
    report_path = OUTPUT_DIR / "report.md"
    results_path = OUTPUT_DIR / "results.json"
    rows = []
    for seed in SEEDS:
        result = neural_results[str(seed)]
        rows.append(
            f"| B-U seed {seed} | {result['metrics']['mae']:.10f} | {result['metrics']['rmse']:.10f} | "
            f"{result['recorded_best_validation_smooth_l1_standardized']:.10f} | {result['metrics']['evaluated_scalar_targets']} |"
        )
    for name in ("persistence", "ewma", "ar5"):
        result = computed_baselines[name]
        rows.append(f"| {name} | {result['mae']:.10f} | {result['rmse']:.10f} | — | {result['evaluated_scalar_targets']} |")
    mean_row = aggregate
    rows.append(
        f"| B-U mean ± sample SD across seeds | {mean_row['mae_mean_across_seeds']:.10f} ± {mean_row['mae_sample_std_across_seeds']:.10f} | "
        f"{mean_row['rmse_mean_across_seeds']:.10f} ± {mean_row['rmse_sample_std_across_seeds']:.10f} | — | 3 seeds |"
    )
    manifest_lines = [
        f"- Seed {seed}: checkpoint `{neural_results[str(seed)]['checkpoint']}`; SHA-256 `{neural_results[str(seed)]['checkpoint_sha256']}`; "
        f"best epoch {neural_results[str(seed)]['best_validation_epoch']}; recorded standardized validation SmoothL1 "
        f"{neural_results[str(seed)]['recorded_best_validation_smooth_l1_standardized']:.12g}."
        for seed in SEEDS
    ]
    report = f"""# GEANT v2 B-U validation-only checkpoint audit

## Decision and scope

**Comparison status: GO.** The three retained B-U checkpoints and persistence/EWMA/AR(5) were evaluated against the same validation target array, directed-link order, finite-value mask, and utilization units. This is validation evidence only; validation was also used to select each checkpoint.

Protocol: `{PROTOCOL_ID}`. Target series: forecasting of computed link utilization from real GEANT traffic matrices; utilization is a dimensionless ratio, not measured live telemetry.

## Coverage and metrics

- Official validation observation block: `[8020, 9740)`.
- Five-history starts: `{validation_starts[0]}–{validation_starts[-1]}`; one-step target indices: `{target_indices[0]}–{target_indices[-1]}` inclusive.
- Windows: {len(validation_starts)}; directed links per target: {x_validation.shape[2]}; scalar targets per method: {y_validation.size}.
- All validation inputs and targets were finite; finite target coverage was {int(np.isfinite(y_validation).sum())}/{y_validation.size}. Predictions were required to be finite. No target was dropped, imputed, or clipped.
- MAE and RMSE are computed in float64 over the same complete set of target/link pairs, in dimensionless utilization-ratio units. Validation SmoothL1 is the recorded checkpoint objective in standardized utilization space and is not interchangeable with MAE/RMSE.
- Neural inputs and targets use the recorded scaler fitted on training arrays only. Predictions were inverse-transformed before MAE/RMSE. Recomputed checkpoint SmoothL1 was checked against each manifest.

## Results

| Method | Validation MAE | Validation RMSE | Recorded standardized validation SmoothL1 | Evaluated scalar targets |
|---|---:|---:|---:|---:|
{chr(10).join(rows)}

The B-U mean ± standard deviation is across the three training seeds (sample SD, `ddof=1`), not across time points. Baseline methods are single fixed forecasts and have no seed variability. EWMA uses fixed alpha {EWMA_ALPHA}; AR(5) uses the fixed rounded legacy coefficients and was not refit on v2 validation.

## Checkpoint identities and protocol

{chr(10).join(manifest_lines)}

All manifests recorded full mode, the expected seed and protocol, identical architecture/settings, validation split, and the same training-only normalization. The topology-derived directed edge ordering was checked against all three manifests. The retained checkpoints are loaded without retraining; selection remains the recorded minimum validation SmoothL1 epoch.

Settings: `{json.dumps(TRAINING, sort_keys=True)}`; history length {HISTORY_LENGTH}, horizon 1. Existing baseline definitions were reproduced on these same arrays: persistence is the last history step, EWMA alpha {EWMA_ALPHA}, and AR(5) is the fixed reported coefficient vector plus intercept. Recomputed baseline MAE/RMSE matched the existing validation baseline JSON within 1e-12.

## Limitations

- These are validation metrics on data used for checkpoint selection, not final held-out generalization estimates.
- The AR(5) coefficients were fit on legacy training windows and are not a v2-fitted AR(5) baseline.
- Three-seed standard deviation measures initialization/training variability only; it is not uncertainty over independent traffic periods.
- No test split was loaded, inspected, or evaluated. This audit does not establish forecasting quality beyond the reported validation comparison.

## Artifacts and command

- Machine-readable results: `results.json`.
- Report: `report.md`.
- Validation inputs loaded: `{X_VALIDATION}`, `{Y_VALIDATION}`.
- Successful command: `python -m phase9_geant.evaluate_bu_v2_validation_audit`.
- A direct-file invocation (`python phase9_geant/evaluate_bu_v2_validation_audit.py`) exited 1 at package import before loading arrays or creating the audit directory; the package-module invocation then completed successfully.
- No training, test evaluation, or model-weight modification was performed.
"""

    # Final no-overwrite guard; publish only after all audits and computations pass.
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite validation audit: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=False, exist_ok=False)
    _write_atomic(results_path, json.dumps(results, indent=2, sort_keys=True, allow_nan=False) + "\n")
    _write_atomic(report_path, report)
    print(json.dumps({"output_dir": str(OUTPUT_DIR), "results": str(results_path), "report": str(report_path)}, indent=2))
    return OUTPUT_DIR


if __name__ == "__main__":
    run()
