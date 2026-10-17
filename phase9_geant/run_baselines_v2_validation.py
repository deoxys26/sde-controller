"""Validation-only classical baselines for GEANT validity protocol v2.

This script intentionally loads only the v2 validation input and target arrays.
It does not import the legacy all-splits runner and does not save predictions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from phase9_geant.split_access import load_permitted_split_array


SCRIPT_VERSION = "1.0.0"
PROTOCOL_ID = "geant-validity-v2-train-median"
EWMA_ALPHA = 0.8
# Frozen values documented for the distance-shortest-path series in
# docs/geant_baselines.md. Coefficient order is oldest to newest history step.
AR5_COEFFICIENTS = (-0.01057, 0.07411, 0.13645, 0.13489, 0.65767)
AR5_INTERCEPT = 0.000212


def _metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    if prediction.shape != target.shape:
        raise ValueError("Prediction and target shapes differ")
    if not np.isfinite(prediction).all():
        raise ValueError("A baseline produced nonfinite predictions for validation windows")

    finite_target = np.isfinite(target)
    evaluated_count = int(finite_target.sum())
    if evaluated_count == 0:
        raise ValueError("No finite validation target/prediction pairs are available")
    errors = prediction[finite_target].astype(np.float64) - target[finite_target].astype(np.float64)
    return {
        "evaluated_scalar_targets": evaluated_count,
        "total_scalar_targets": int(target.size),
        "nonfinite_target_count": int(target.size - evaluated_count),
        "mae": float(np.mean(np.abs(errors), dtype=np.float64)),
        "rmse": float(np.sqrt(np.mean(np.square(errors), dtype=np.float64))),
    }


def run() -> dict[str, Any]:
    repository_root = Path(__file__).resolve().parents[1]
    data_root = repository_root / "data" / "processed" / "geant_validity_v2"
    x_path = data_root / "X_validation.npy"
    y_path = data_root / "y_validation.npy"
    report_path = data_root / "baselines_validation.json"

    if not x_path.is_file() or not y_path.is_file():
        raise FileNotFoundError("Required v2 validation arrays are missing")
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing report: {report_path}")

    # These are the only NumPy artifacts read by this program.
    history = load_permitted_split_array(data_root, "X_validation.npy")
    target = load_permitted_split_array(data_root, "y_validation.npy")
    if history.ndim != 3 or history.shape[1] != 5:
        raise ValueError(f"Expected validation history shape (N, 5, E), got {history.shape}")
    if target.ndim != 2 or target.shape != (history.shape[0], history.shape[2]):
        raise ValueError(
            f"Expected validation target shape {(history.shape[0], history.shape[2])}, "
            f"got {target.shape}"
        )
    if history.shape[0] == 0 or history.shape[2] == 0:
        raise ValueError("Validation arrays contain no windows or directed links")

    persistence = history[:, -1, :]
    weights = EWMA_ALPHA * (1.0 - EWMA_ALPHA) ** (4 - np.arange(5, dtype=np.float64))
    weights /= weights.sum()
    ewma = np.tensordot(history, weights, axes=([1], [0]))
    ar5 = np.tensordot(
        history,
        np.asarray(AR5_COEFFICIENTS, dtype=np.float64),
        axes=([1], [0]),
    ) + AR5_INTERCEPT

    metrics = {
        "persistence": _metrics(target, persistence),
        "ewma": _metrics(target, ewma),
        "ar5": _metrics(target, ar5),
    }
    report = {
        "protocol_id": PROTOCOL_ID,
        "split": "validation",
        "input_artifacts": [
            "data/processed/geant_validity_v2/X_validation.npy",
            "data/processed/geant_validity_v2/y_validation.npy",
        ],
        "history_shape": list(history.shape),
        "target_shape": list(target.shape),
        "metric_definitions": {
            "mae": "mean absolute error over finite target/prediction scalar pairs",
            "rmse": "square root of mean squared error over finite target/prediction scalar pairs",
            "nonfinite_predictions": "fatal error; no predictions are silently excluded",
        },
        "baseline_parameters": {
            "persistence": "prediction is the fifth (most recent) history step",
            "ewma": {
                "alpha": EWMA_ALPHA,
                "formula": "normalized weights alpha*(1-alpha)^(4-k), k=0..4 oldest to newest",
                "provenance": "fixed from the existing distance-series protocol; not retuned on v2 validation",
            },
            "ar5": {
                "coefficients_oldest_to_newest": list(AR5_COEFFICIENTS),
                "intercept": AR5_INTERCEPT,
                "formula": "dot(history, coefficients) + intercept",
                "provenance": (
                    "fixed rounded coefficients documented for the existing distance-series baseline; "
                    "originally fit by pooled OLS on legacy training windows, not refit or tuned here"
                ),
            },
        },
        "metrics": metrics,
        "script": {
            "name": "phase9_geant/run_baselines_v2_validation.py",
            "version": SCRIPT_VERSION,
            "prediction_archives_written": False,
            "test_split_accessed": False,
        },
    }

    # Output is created only after all input checks and metric calculations succeed.
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report_path), "metrics": metrics}, indent=2))
    return report


if __name__ == "__main__":
    run()
