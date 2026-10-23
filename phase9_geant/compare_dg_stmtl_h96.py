"""Compare completed DG-STMTL-based runs with frozen h96 GNN/EWMA manifests.

Only explicitly supplied run.json files are opened. Checkpoint identity is
validated from each manifest's declared filename and SHA-256; checkpoint bytes,
arrays, predictions, and test-split artifacts are not opened.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any


DG_PROTOCOL = "geant-dg-stmtl-based-h96-v1"
GNN_PROTOCOL = "geant-horizon96-only-candidate-v2-batch4"
SEEDS = (42, 123, 2026)
MODES = ("topology", "identity", "shuffled")
PAIR_COUNT = 953 * 72
EXPECTED_MISSING_GNN = ("topology", 42)
# The two runners use the same EWMA definition and metric routine but reach the
# original utilization units through different tensor paths. A 1e-9 absolute
# tolerance admits only sub-nanounit floating-point accumulation differences;
# relative tolerance is deliberately disabled so the allowance is fixed.
EWMA_METRIC_ABS_TOL = 1e-9
SHARED_METRIC_SOURCE = "phase9_geant/horizon96_streaming_experiment.py"
SHARED_SCALER_SOURCE = "phase9_geant/horizon96_smoke.py"
DATASET_BUNDLE_SHA256 = "df2bac044c9b9a577d5117310a99022cce402b9370e2ba483ea18b8d7045521e"
DATASET_BUNDLE_MANIFEST_SHA256 = "70f60f09b71c0cb89003ac2987af4433df0eeeb64b696845faa25d0fbadb8a95"
APPROVED_ARRAY_SHA256 = {
    "timestamps_train.npy": "4442cbafad28af487bab8da11fcd0f5508fd681c929df952715e2dfc30e037c6",
    "timestamps_validation.npy": "d8797769a2719154733088f17082896f334c98a83dd1f47d39b69d7049c211ba",
    "utilization_train.npy": "f3e3d3b3246688804985cc776411765755ba241ff1c833683f8a07c45ffc3a74",
    "utilization_validation.npy": "0c26c0dd171274c12eddd3215992d1c92e79eb3a56e7807ae075637c3b07c78d",
    "valid_train.npy": "9bb4947ee0043c66240496fb86bae4734d7880f220b7e005618bfc703d349117",
    "valid_validation.npy": "98b8cfc07a2aa53bb47835abcb68f2e9ed1de2300ba4f2fe2c92c9fa766a1bb5",
}
APPROVED_WINDOW_INDEX_SHA256 = {
    "train": "2f1e172e3684ad7f9b5eaed0e8d6fd3574a5fce496408e2326b93e2068903703",
    "validation": "ee46cc4462ddf04ad9c20557b5a901597299eeb1b24308c0c5bc23fb540ae9d8",
}
APPROVED_ORDERED_EDGE_SHA256 = "ce0073367329c2530302fffec3c76672d3b293160a4ec84f8f2f04db145c545c"
SCALER_FIT_SCOPE = "h=96 train histories plus h=96 targets; window-occurrence weighted"
GNN_SCALER_SCOPE_DESCRIPTIONS = {
    "h=96 training histories and targets only; window-occurrence weighted",
}
DG_SCALER_SCOPE_DESCRIPTIONS = {
    "existing window-occurrence-weighted h96 training histories plus h96 targets only",
}


class ComparisonError(ValueError):
    """Run manifests do not form a scientifically comparable set."""


def _read_run(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(os.path.abspath(path))
    if manifest_path.is_symlink() or not manifest_path.is_file() or manifest_path.resolve(strict=True) != manifest_path:
        raise ComparisonError(f"Run manifest is missing or redirected: {manifest_path}")
    try:
        result = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ComparisonError(f"Cannot read run manifest {manifest_path}: {exc}") from exc
    if not isinstance(result, dict):
        raise ComparisonError(f"Run manifest is not an object: {manifest_path}")
    return result


def _number(metric: dict[str, Any], key: str, label: str) -> float:
    value = metric.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ComparisonError(f"Invalid {label}.{key}")
    return float(value)


def _validate_metric(run: dict[str, Any], name: str, path_label: str) -> dict[str, Any]:
    metrics = run.get("validation_metrics")
    if not isinstance(metrics, dict) or not isinstance(metrics.get(name), dict):
        raise ComparisonError(f"Missing validation metric {name} in {path_label}")
    metric = metrics[name]
    if metric.get("evaluated_pairs") != PAIR_COUNT:
        raise ComparisonError(f"Validation pair coverage mismatch in {path_label}: {name}")
    _number(metric, "mae", path_label)
    _number(metric, "rmse", path_label)
    return metric


def _validate_checkpoint_identity(run: dict[str, Any], path_label: str) -> None:
    """Require the frozen run's declared checkpoint identity without opening it."""
    checkpoint = run.get("checkpoint")
    digest = run.get("checkpoint_sha256")
    if checkpoint != "best.pt":
        raise ComparisonError(f"Missing or unexpected checkpoint filename in {path_label}")
    if (not isinstance(digest, str) or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest.lower())):
        raise ComparisonError(f"Missing or malformed checkpoint SHA-256 in {path_label}")


def _validate_evaluation_contract(run: dict[str, Any], kind: str, path_label: str) -> None:
    """Check manifest-declared horizon, EWMA, split, and metric-unit contract."""
    config = run.get("configuration")
    if not isinstance(config, dict):
        raise ComparisonError(f"Missing evaluation configuration in {path_label}")
    declared_config_hash = _require_sha256(run.get("configuration_sha256"), f"configuration in {path_label}")
    if _canonical_json_sha256(config) != declared_config_hash:
        raise ComparisonError(f"Configuration identity hash mismatch in {path_label}")
    common = {
        "history_length": 672,
        "ewma_alpha": 0.8,
        "ewma_steps": 5,
        "train_bounds": [0, 8020],
        "validation_bounds": [8020, 9740],
    }
    for name, expected in common.items():
        if config.get(name) != expected:
            raise ComparisonError(f"Incompatible evaluation configuration field {name} in {path_label}")
    if kind == "gnn":
        if config.get("horizons") != [96]:
            raise ComparisonError(f"GNN evaluation is not h=96-only in {path_label}")
    elif config.get("horizon") != 96:
        raise ComparisonError(f"DG-STMTL evaluation is not h=96-only in {path_label}")
    provenance = run.get("provenance")
    if not isinstance(provenance, dict) or provenance.get("source_scope") != {
        "train": [0, 8020], "validation": [8020, 9740]
    }:
        raise ComparisonError(f"Missing or incompatible train/validation window provenance in {path_label}")
    validation_count = provenance.get("validation_windows_h96", run.get("validation_window_count", run.get("validation_windows")))
    if validation_count != 953:
        raise ComparisonError(f"Validation window identity/count mismatch in {path_label}")
    source_hashes = run.get("source_hashes")
    if not isinstance(source_hashes, dict):
        raise ComparisonError(f"Missing source identity for metric implementation in {path_label}")
    _require_sha256(source_hashes.get(SHARED_METRIC_SOURCE), f"shared metric source in {path_label}")


def _require_sha256(value: Any, label: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value.lower())):
        raise ComparisonError(f"Missing or malformed SHA-256 identity: {label}")
    return value.lower()


def _canonical_json_sha256(value: Any) -> str:
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ComparisonError("Malformed configuration cannot be canonically hashed") from exc
    return hashlib.sha256(encoded).hexdigest()


def _array_hashes(run: dict[str, Any], kind: str, label: str) -> dict[str, str]:
    """Normalize the two manifest layouts to the approved six train/val files."""
    if kind == "gnn":
        records = run.get("provenance", {}).get("array_file_records")
        if not isinstance(records, dict):
            raise ComparisonError(f"Missing GNN array-file provenance in {label}")
        normalized = {
            name: _require_sha256(record.get("file_sha256") if isinstance(record, dict) else None,
                                  f"array {name} in {label}")
            for name, record in records.items()
        }
    else:
        records = run.get("permitted_array_file_hashes")
        if not isinstance(records, dict):
            raise ComparisonError(f"Missing DG permitted-array fingerprints in {label}")
        normalized = {name: _require_sha256(value, f"array {name} in {label}")
                      for name, value in records.items()}
    if set(normalized) != set(APPROVED_ARRAY_SHA256):
        raise ComparisonError(f"Train/validation array allowlist mismatch in {label}")
    if normalized != APPROVED_ARRAY_SHA256:
        raise ComparisonError(f"Train/validation array fingerprints do not match approved bundle in {label}")
    return normalized


def _window_hashes(run: dict[str, Any], label: str) -> dict[str, str]:
    provenance = run.get("provenance")
    values = provenance.get("window_index_sha256") if isinstance(provenance, dict) else None
    if not isinstance(values, dict) or set(values) != {"train", "validation"}:
        raise ComparisonError(f"Missing train/validation window-index identities in {label}")
    normalized = {key: _require_sha256(value, f"{key} window index in {label}")
                  for key, value in values.items()}
    if normalized != APPROVED_WINDOW_INDEX_SHA256:
        raise ComparisonError(f"Train/validation window identities do not match approved bundle in {label}")
    return normalized


def _semantic_data_identity(run: dict[str, Any], kind: str, label: str) -> dict[str, Any]:
    """Extract only cross-model data/evaluation identity, excluding run-specific paths/config labels."""
    config = run["configuration"]
    provenance = run["provenance"]
    protocol_values = {
        value for value in (config.get("dataset_protocol_id"), provenance.get("dataset_protocol_id"))
        if value is not None
    }
    if protocol_values != {"geant-h96-streaming-trainval-v1"}:
        raise ComparisonError(f"Dataset protocol identity missing or inconsistent in {label}")
    source_scope = provenance.get("source_scope")
    if source_scope != {"train": [0, 8020], "validation": [8020, 9740]}:
        raise ComparisonError(f"Train/validation source boundaries differ in {label}")
    train_windows = provenance.get("training_windows_h96", run.get("training_window_count", run.get("training_windows")))
    validation_windows = provenance.get("validation_windows_h96", run.get("validation_window_count", run.get("validation_windows")))
    if train_windows != 3924 or validation_windows != 953:
        raise ComparisonError(f"Train/validation window count mismatch in {label}")
    edge_hash = _require_sha256(
        run.get("topology", {}).get("ordered_edge_sha256") or provenance.get("ordered_edge_sha256"),
        f"ordered edge identity in {label}",
    )
    if edge_hash != APPROVED_ORDERED_EDGE_SHA256:
        raise ComparisonError(f"Directed-link ordering differs from approved bundle in {label}")
    dataset_hash = _require_sha256(run.get("dataset_metadata_sha256"), f"dataset metadata in {label}")
    array_hashes = _array_hashes(run, kind, label)
    window_hashes = _window_hashes(run, label)

    if kind == "gnn":
        bundle = config.get("colab_bundle_identity")
        if not isinstance(bundle, dict):
            raise ComparisonError(f"Missing GNN Colab dataset bundle identity in {label}")
        bundle_sha = _require_sha256(bundle.get("dataset_bundle_sha256"), f"GNN dataset bundle in {label}")
        manifest_sha = _require_sha256(bundle.get("dataset_manifest_sha256"), f"GNN dataset bundle manifest in {label}")
        if bundle_sha != DATASET_BUNDLE_SHA256 or manifest_sha != DATASET_BUNDLE_MANIFEST_SHA256:
            raise ComparisonError(f"GNN dataset bundle identity differs from approved train/validation bundle in {label}")
        if config.get("scaler_fit_scope") not in GNN_SCALER_SCOPE_DESCRIPTIONS:
            raise ComparisonError(f"Unrecognized GNN scaler configuration description in {label}")
    else:
        if config.get("scaler_fit_scope") not in DG_SCALER_SCOPE_DESCRIPTIONS:
            raise ComparisonError(f"Unrecognized DG scaler configuration description in {label}")

    normalization = run.get("normalization")
    if not isinstance(normalization, dict) or normalization.get("fit_scope") != SCALER_FIT_SCOPE:
        raise ComparisonError(f"Scaler was not produced by the approved training-only scaler in {label}")
    scaler = {
        "mean": normalization.get("mean"),
        "scale": normalization.get("scale"),
        "population_count": normalization.get("population_count"),
    }
    for field in ("mean", "scale"):
        value = scaler[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ComparisonError(f"Invalid training-only scaler {field} in {label}")
    if scaler["scale"] <= 0:
        raise ComparisonError(f"Invalid training-only scaler population in {label}")
    if isinstance(scaler["population_count"], bool) or not isinstance(scaler["population_count"], int):
        raise ComparisonError(f"Invalid training-only scaler population count in {label}")

    return {
        "dataset_metadata_sha256": dataset_hash,
        "dataset_protocol_id": "geant-h96-streaming-trainval-v1",
        "source_scope": source_scope,
        "training_windows": train_windows,
        "validation_windows": validation_windows,
        "array_file_sha256": array_hashes,
        "window_index_sha256": window_hashes,
        "ordered_edge_sha256": edge_hash,
        "normalization": scaler,
    }


def _cross_model_provenance_identity(runs: list[tuple[dict[str, Any], str, str]]) -> dict[str, Any]:
    identities = [_semantic_data_identity(run, kind, label) for run, kind, label in runs]
    mismatch_messages = {
        "dataset_metadata_sha256": "Dataset metadata identities differ across runs",
        "dataset_protocol_id": "Dataset protocol identities differ across runs",
        "source_scope": "Train/validation source boundaries differ across runs",
        "training_windows": "Training window counts differ across runs",
        "validation_windows": "Validation window counts differ across runs",
        "array_file_sha256": "Train/validation array fingerprints differ across runs",
        "window_index_sha256": "Validation-window identities differ across runs",
        "ordered_edge_sha256": "Directed edge ordering differs across runs",
        "normalization": "Training-only normalization differs across comparison runs",
    }
    for identity in identities[1:]:
        for field, message in mismatch_messages.items():
            if identity[field] != identities[0][field]:
                raise ComparisonError(message)
    metric_hashes = set()
    scaler_hashes = set()
    for run, _kind, label in runs:
        sources = run.get("source_hashes")
        if not isinstance(sources, dict):
            raise ComparisonError(f"Missing source identities in {label}")
        metric_hashes.add(_require_sha256(sources.get(SHARED_METRIC_SOURCE), f"metric implementation in {label}"))
        scaler_hashes.add(_require_sha256(sources.get(SHARED_SCALER_SOURCE), f"scaler implementation in {label}"))
    if len(metric_hashes) != 1 or len(scaler_hashes) != 1:
        raise ComparisonError("Runs do not share exact metric and scaler implementation identities")
    return identities[0]


def _sample_summary(values: list[float]) -> dict[str, float | None]:
    return {"mean": statistics.fmean(values), "sample_std_ddof1": statistics.stdev(values) if len(values) > 1 else None}


def collect_comparison(
    gnn_manifest_paths: list[str | Path],
    dg_manifest_paths: list[str | Path],
    *,
    allow_incomplete_gnn_matrix: bool = False,
) -> dict[str, Any]:
    expected_gnn_count = 8 if allow_incomplete_gnn_matrix else 9
    if len(gnn_manifest_paths) != expected_gnn_count or len(dg_manifest_paths) != 3:
        mode_text = "exactly eight GNN" if allow_incomplete_gnn_matrix else "exactly nine GNN"
        raise ComparisonError(f"Supply {mode_text} run.json paths and exactly three DG-STMTL run.json paths")
    gnn_rows: dict[tuple[str, int], tuple[dict[str, Any], dict[str, Any]]] = {}
    for path in gnn_manifest_paths:
        run = _read_run(path)
        seed, mode = run.get("seed"), run.get("adjacency_mode")
        key = (mode, seed)
        if mode not in MODES or seed not in SEEDS or key in gnn_rows:
            raise ComparisonError(f"Unexpected or duplicate frozen GNN identity at {path}")
        if run.get("status") != "complete" or run.get("protocol_id") != GNN_PROTOCOL:
            raise ComparisonError(f"GNN run is incomplete or uses another protocol: {path}")
        _validate_evaluation_contract(run, "gnn", str(path))
        if run.get("test_access", {}).get("permitted") is not False or run.get("checkpoint_reload_strict") is not True:
            raise ComparisonError(f"GNN run lacks test denial or strict checkpoint reload: {path}")
        _validate_checkpoint_identity(run, str(path))
        model_metric = _validate_metric(run, "h96_candidate", str(path))
        ewma_metric = _validate_metric(run, "ewma_alpha_0.8", str(path))
        gnn_rows[key] = (run, {"model": model_metric, "ewma": ewma_metric})
    expected_gnn = {(mode, seed) for mode in MODES for seed in SEEDS}
    missing_gnn = expected_gnn - set(gnn_rows)
    unexpected_gnn = set(gnn_rows) - expected_gnn
    if unexpected_gnn:
        raise ComparisonError(f"Unexpected GNN combinations: {sorted(unexpected_gnn)}")
    if allow_incomplete_gnn_matrix:
        if missing_gnn != {EXPECTED_MISSING_GNN}:
            raise ComparisonError(
                "Incomplete mode permits only the missing original GNN Topology seed 42; "
                f"observed missing combinations: {sorted(missing_gnn)}"
            )
    elif missing_gnn:
        raise ComparisonError("The nine GNN manifests do not cover all seed/mode combinations exactly once")

    dg_rows: dict[int, tuple[dict[str, Any], dict[str, Any]]] = {}
    for path in dg_manifest_paths:
        run = _read_run(path)
        seed = run.get("seed")
        if seed not in SEEDS or seed in dg_rows:
            raise ComparisonError(f"Unexpected or duplicate DG-STMTL seed at {path}")
        if (run.get("status") != "complete" or run.get("protocol_id") != DG_PROTOCOL
                or run.get("adjacency_mode") != "hybrid_topology_dynamic"):
            raise ComparisonError(f"DG-STMTL run is incomplete or uses another protocol: {path}")
        _validate_evaluation_contract(run, "dg", str(path))
        if run.get("test_access", {}).get("permitted") is not False or run.get("checkpoint_reload_strict") is not True:
            raise ComparisonError(f"DG-STMTL run lacks test denial or strict checkpoint reload: {path}")
        _validate_checkpoint_identity(run, str(path))
        dg_rows[seed] = (run, {
            "model": _validate_metric(run, "dg_stmtl_based", str(path)),
            "ewma": _validate_metric(run, "ewma_alpha_0.8", str(path)),
        })
    if set(dg_rows) != set(SEEDS):
        raise ComparisonError("DG-STMTL manifests do not cover the three fixed seeds")

    # Compare normalized scientific identities. Storage paths, architecture-
    # specific config fields, and equivalent schema labels are not data identity.
    labeled_runs = ([(run, "gnn", str(path)) for path, run in zip(gnn_manifest_paths, [x[0] for x in gnn_rows.values()])]
                    + [(run, "dg", str(path)) for path, run in zip(dg_manifest_paths, [x[0] for x in dg_rows.values()])])
    all_runs = [run for run, _kind, _label in labeled_runs]
    common_semantic_identity = _cross_model_provenance_identity(labeled_runs)

    # Both runners implement the same five-history, alpha=0.8 EWMA and use the
    # same float64 MAE/RMSE routine. Their final float values may differ by a
    # few rounding units because the normalized tensor paths differ slightly.
    # Keep every recorded value unchanged; only the consistency comparison is
    # tolerant, with no relative allowance.
    for seed in SEEDS:
        baseline_rows = [gnn_rows[(mode, seed)][1]["ewma"] for mode in MODES if (mode, seed) in gnn_rows]
        baseline_rows.append(dg_rows[seed][1]["ewma"])
        reference = baseline_rows[0]
        for candidate in baseline_rows[1:]:
            for field in ("mae", "rmse"):
                difference = abs(float(reference[field]) - float(candidate[field]))
                if not math.isclose(float(reference[field]), float(candidate[field]), rel_tol=0.0,
                                    abs_tol=EWMA_METRIC_ABS_TOL):
                    raise ComparisonError(
                        f"EWMA {field} differs across methods for seed {seed} "
                        f"(absolute difference {difference:.12g} > {EWMA_METRIC_ABS_TOL:g})"
                    )

    common_metadata = {_require_sha256(run.get("dataset_metadata_sha256"), "dataset metadata") for run in all_runs}
    common_edges = {common_semantic_identity["ordered_edge_sha256"]}
    common_validation = {common_semantic_identity["validation_windows"]}
    common_pairs = {
        metric.get("evaluated_pairs")
        for _run, methods in list(gnn_rows.values()) + list(dg_rows.values())
        for metric in methods.values()
    }
    if (len(common_metadata) != 1 or None in common_metadata or len(common_edges) != 1 or None in common_edges
            or common_validation != {953} or common_pairs != {PAIR_COUNT}):
        raise ComparisonError("Runs do not share exact dataset, edge ordering, or validation coverage identities")
    if any(not isinstance(run.get("normalization"), dict) or not run["normalization"] for run in all_runs):
        raise ComparisonError("Missing or malformed training-only normalization metadata")
    scaler_values = [json.dumps(run["normalization"], sort_keys=True, separators=(",", ":")) for run in all_runs]
    if any(value != scaler_values[0] for value in scaler_values[1:]):
        raise ComparisonError("Training-only normalization differs across comparison runs")

    for run, metrics in [(run, methods) for run, methods in list(gnn_rows.values()) + list(dg_rows.values())]:
        for metric in metrics.values():
            if metric.get("units") != "computed utilization ratio":
                raise ComparisonError("Validation metric units differ from computed utilization ratio")

    per_seed: list[dict[str, Any]] = []
    for seed in SEEDS:
        gnn_metrics = {mode: gnn_rows[(mode, seed)][1]["model"] for mode in MODES if (mode, seed) in gnn_rows}
        dg_metric = dg_rows[seed][1]["model"]
        # The original GNN run's recorded EWMA is the comparison reference.
        # Keep the DG-STMTL run's independently recorded baseline alongside it
        # so tolerance never rewrites or hides either manifest value.
        gnn_ewma_metric = next(gnn_rows[(mode, seed)][1]["ewma"] for mode in MODES
                               if (mode, seed) in gnn_rows)
        ewma_metric = gnn_ewma_metric
        dg_ewma_metric = dg_rows[seed][1]["ewma"]
        ewma_mae = _number(ewma_metric, "mae", "ewma")
        ewma_rmse = _number(ewma_metric, "rmse", "ewma")
        if ewma_mae <= 0.0 or ewma_rmse <= 0.0:
            raise ComparisonError(f"EWMA metric must be positive to compute relative changes for seed {seed}")

        def with_relative(value: dict[str, Any], label: str) -> dict[str, Any]:
            mae, rmse = _number(value, "mae", label), _number(value, "rmse", label)
            return {
                "mae": mae, "rmse": rmse,
                "relative_mae_improvement_vs_ewma_percent": 100.0 * (ewma_mae - mae) / ewma_mae,
                "relative_rmse_improvement_vs_ewma_percent": 100.0 * (ewma_rmse - rmse) / ewma_rmse,
            }

        per_seed.append({
            "seed": seed,
            "ewma": {"mae": ewma_mae, "rmse": ewma_rmse,
                     "relative_mae_improvement_vs_ewma_percent": 0.0,
                     "relative_rmse_improvement_vs_ewma_percent": 0.0,
                     "recorded_by_method": {
                         "original_gnn": {"mae": _number(gnn_ewma_metric, "mae", "original_gnn_ewma"),
                                          "rmse": _number(gnn_ewma_metric, "rmse", "original_gnn_ewma")},
                         "dg_stmtl_based": {"mae": _number(dg_ewma_metric, "mae", "dg_ewma"),
                                            "rmse": _number(dg_ewma_metric, "rmse", "dg_ewma")},
                     }},
            "dg_stmtl_based": {**with_relative(dg_metric, "dg"),
                               "parameter_count": dg_rows[seed][0].get("model_parameter_count"),
                               "training_seconds": dg_rows[seed][0].get("runtime_seconds"),
                               "checkpoint_sha256": dg_rows[seed][0]["checkpoint_sha256"]},
            "gnn": {mode: {**with_relative(gnn_metrics[mode], mode),
                            "parameter_count": gnn_rows[(mode, seed)][0].get("model_parameter_count"),
                            "training_seconds": gnn_rows[(mode, seed)][0].get("runtime_seconds"),
                            "checkpoint_sha256": gnn_rows[(mode, seed)][0]["checkpoint_sha256"]}
                    for mode in gnn_metrics},
        })
    methods: dict[str, list[tuple[float, float, int | None, float | None]]] = {"ewma": [], "dg_stmtl_based": []}
    methods.update({f"gnn_{mode}": [] for mode in MODES})
    for item in per_seed:
        methods["ewma"].append((item["ewma"]["mae"], item["ewma"]["rmse"], 0, None))
        methods["dg_stmtl_based"].append((item["dg_stmtl_based"]["mae"], item["dg_stmtl_based"]["rmse"],
                                          item["dg_stmtl_based"]["parameter_count"], item["dg_stmtl_based"]["training_seconds"]))
        for mode in MODES:
            if mode in item["gnn"]:
                row = item["gnn"][mode]
                methods[f"gnn_{mode}"].append((row["mae"], row["rmse"], row["parameter_count"], row["training_seconds"]))
    aggregate: dict[str, Any] = {}
    for method, rows in methods.items():
        maes, rmses = [row[0] for row in rows], [row[1] for row in rows]
        if method.startswith("gnn_"):
            selected_seeds = [item for item in per_seed if method.removeprefix("gnn_") in item["gnn"]]
        else:
            selected_seeds = per_seed
        reference_mae = statistics.fmean(item["ewma"]["mae"] for item in selected_seeds)
        reference_rmse = statistics.fmean(item["ewma"]["rmse"] for item in selected_seeds)
        aggregate[method] = {
            "mae": _sample_summary(maes), "rmse": _sample_summary(rmses),
            "n": len(rows),
            "relative_mae_improvement_vs_ewma_percent": 100.0 * (reference_mae - statistics.fmean(maes)) / reference_mae,
            "relative_rmse_improvement_vs_ewma_percent": 100.0 * (reference_rmse - statistics.fmean(rmses)) / reference_rmse,
            "parameter_count": rows[0][2] if len({row[2] for row in rows}) == 1 else [row[2] for row in rows],
            "training_seconds_mean": statistics.fmean(row[3] for row in rows) if all(row[3] is not None for row in rows) else None,
            "evaluated_pairs_per_seed": PAIR_COUNT,
        }
    partial = bool(missing_gnn)
    return {
        "status": "partial" if partial else "complete", "comparison_protocol": "geant-dg-stmtl-based-h96-v1",
        "comparison_scope": "partial_original_gnn_matrix" if partial else "complete_original_gnn_matrix",
        "gnn_matrix_complete": not partial,
        "checkpoint_identity_validation": (
            "manifest-declared best.pt filename and SHA-256 syntax; checkpoint bytes were not reopened"
        ),
        "ewma_consistency_check": {
            "absolute_tolerance": EWMA_METRIC_ABS_TOL,
            "relative_tolerance": 0.0,
            "policy": "compare separately recorded EWMA values without rewriting them; fixed absolute tolerance only",
        },
        "missing_gnn_combinations": [
            {"model": "original_gnn", "adjacency_mode": mode, "seed": seed}
            for mode, seed in sorted(missing_gnn)
        ],
        "architectural_comparison_claim": not partial,
        "split": "validation only", "seed_aggregation": "mean and sample standard deviation (ddof=1); seed variability, not confidence interval",
        "metric_units": "computed utilization ratio", "mask": "finite target/prediction pairs; exactly 953 x 72 per run",
        "common_dataset_metadata_sha256": next(iter(common_metadata)),
        "common_ordered_edge_sha256": next(iter(common_edges)),
        "common_semantic_data_identity": common_semantic_identity,
        "per_seed": per_seed, "aggregate": aggregate,
        "test_access": False,
        "interpretation": (
            "Partial validation comparison; not a full three-seed architectural comparison. "
            "The original GNN Topology seed 42 run is missing."
            if partial else "Validation comparison only; no held-out generalization claim"
        ),
    }


def write_comparison(result: dict[str, Any], output_directory: str | Path) -> Path:
    output = Path(os.path.abspath(output_directory))
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing existing comparison output directory: {output}")
    output.mkdir(parents=True, exist_ok=False)
    (output / "comparison.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (output / "comparison.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("method", "n", "MAE mean", "MAE sample SD", "RMSE mean", "RMSE sample SD", "relative MAE improvement vs EWMA (%)", "parameter count", "training seconds mean"))
        for method, values in result["aggregate"].items():
            writer.writerow((method, values["n"], values["mae"]["mean"], values["mae"]["sample_std_ddof1"],
                             values["rmse"]["mean"], values["rmse"]["sample_std_ddof1"],
                             values["relative_mae_improvement_vs_ewma_percent"], values["parameter_count"],
                             values["training_seconds_mean"]))
    heading = ("# Partial GEANT h96 validation comparison" if result["status"] == "partial"
               else "# GEANT h96 validation comparison")
    lines = [heading, "", "Validation evidence only; no test-split access.",
             f"EWMA consistency uses fixed absolute tolerance {EWMA_METRIC_ABS_TOL:g} (relative tolerance 0). "
             "The displayed EWMA is copied from the original GNN manifests; the independently recorded "
             "DG-STMTL EWMA values are preserved per seed in comparison.json.", ""]
    if result["status"] == "partial":
        lines.extend([
            "**Not a full three-seed architectural comparison.** The original GNN Topology seed 42 run is missing.",
            "Available-seed summaries use n=2 for GNN Topology and n=3 for GNN Identity, GNN Shuffled, and DG-STMTL-based.", "",
        ])
    lines.extend([
             "## Per-seed results", "",
             "| Seed | Method | MAE | RMSE | Relative MAE change vs EWMA | Relative RMSE change vs EWMA | Parameters | Training seconds |",
             "|---:|---|---:|---:|---:|---:|---:|---:|"])
    for item in result["per_seed"]:
        seed = item["seed"]
        records = [("EWMA", item["ewma"]), ("DG-STMTL-based", item["dg_stmtl_based"])]
        records.extend((f"GNN {mode}", item["gnn"][mode]) for mode in MODES if mode in item["gnn"])
        for method, values in records:
            lines.append(f"| {seed} | {method} | {values['mae']:.8g} | {values['rmse']:.8g} | "
                         f"{values['relative_mae_improvement_vs_ewma_percent']:.4f}% | "
                         f"{values['relative_rmse_improvement_vs_ewma_percent']:.4f}% | "
                         f"{values.get('parameter_count', '—')} | {values.get('training_seconds', '—')} |")
    summary_title = "Available-seed summary (not a full three-seed architectural comparison)" if result["status"] == "partial" else "Three-seed summary"
    lines.extend(["", f"## {summary_title}", "",
             "| Method | n | MAE mean ± sample SD | RMSE mean ± sample SD | MAE improvement vs EWMA | Parameters | Mean training seconds |",
             "|---|---:|---:|---:|---:|---:|---:|"])
    for method, values in result["aggregate"].items():
        lines.append(f"| {method} | {values['n']} | {values['mae']['mean']:.8g} ± {values['mae']['sample_std_ddof1'] or 0:.3g} | "
                     f"{values['rmse']['mean']:.8g} ± {values['rmse']['sample_std_ddof1'] or 0:.3g} | "
                     f"{values['relative_mae_improvement_vs_ewma_percent']:.4f}% | {values['parameter_count']} | "
                     f"{values['training_seconds_mean'] if values['training_seconds_mean'] is not None else 'n/a'} |")
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gnn-run-json", type=Path, action="append", required=True,
                        help="repeat nine times in strict mode, or eight times with the explicit incomplete-matrix option")
    parser.add_argument("--dg-run-json", type=Path, action="append", required=True,
                        help="repeat exactly three times for this DG-STMTL version")
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--allow-incomplete-gnn-matrix", action="store_true",
                        help="accept exactly eight GNN manifests only when original GNN topology seed 42 is the sole missing combination")
    parser.add_argument("--diagnostic-only", action="store_true",
                        help="read only supplied run.json manifests; validate and print semantic identities without writing a report")
    args = parser.parse_args(argv)
    if args.diagnostic_only and any(path.name != "run.json" for path in args.gnn_run_json + args.dg_run_json):
        parser.error("--diagnostic-only accepts only explicitly supplied files named run.json")
    if not args.diagnostic_only and args.output_directory is None:
        parser.error("--output-directory is required unless --diagnostic-only is selected")
    try:
        result = collect_comparison(args.gnn_run_json, args.dg_run_json,
                                    allow_incomplete_gnn_matrix=args.allow_incomplete_gnn_matrix)
    except ComparisonError as exc:
        if args.diagnostic_only:
            print(json.dumps({"status": "incompatible", "error": str(exc), "test_access": False}, indent=2))
            return 2
        raise
    if args.diagnostic_only:
        print(json.dumps({
            "status": "compatible_manifest_identity",
            "common_semantic_data_identity": result["common_semantic_data_identity"],
            "ewma_consistency_tolerance": result["ewma_consistency_check"],
            "missing_gnn_combinations": result["missing_gnn_combinations"],
            "test_access": False,
            "artifacts_written": False,
        }, indent=2))
        return 0
    output = write_comparison(result, args.output_directory)
    print(json.dumps({"status": result["status"], "output_directory": str(output), "test_access": False,
                      "missing_gnn_combinations": result["missing_gnn_combinations"]}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - future report only
    raise SystemExit(main())


__all__ = ["ComparisonError", "collect_comparison", "write_comparison"]
