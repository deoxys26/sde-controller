"""Fixed-checkpoint, validation-only B-U topology sensitivity diagnostic.

Reads only X_validation.npy, y_validation.npy, the three explicit B-U full-run
manifests/checkpoints, and the GEANT topology text. It does not train or write
predictions. The graph rewire changes connectivity only; per-column edge
histories and target correspondence remain fixed.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from phase9_geant.parse_geant import parse_topology
from phase9_geant.split_access import load_permitted_split_array
from phase9_geant.train_bu_v2 import (
    HISTORY_LENGTH,
    PROTOCOL_ID,
    TRAINING,
    V2_DATA_ROOT,
    Normalization,
    build_edge_index,
    make_bu_sequence,
)
from src.gnn.model import EdgeQoSForecaster


SCRIPT_VERSION = "1.0.0"
SEEDS = (42, 123, 2026)
TOLERANCE = 1e-7
REWIRE_SWAPS = 12
ARRAY_ROOT = V2_DATA_ROOT
X_VALIDATION = ARRAY_ROOT / "X_validation.npy"
Y_VALIDATION = ARRAY_ROOT / "y_validation.npy"
TOPOLOGY_PATH = Path(__file__).resolve().parents[1] / "data/raw/geant_topology/geant.txt"
OUTPUT_DIR = ARRAY_ROOT / "diagnostics_topology_sensitivity_20261017"


def _canonical_pair(edge: tuple[str, str]) -> frozenset[str]:
    return frozenset(edge)


def _degree_counts(nodes: Sequence[str], edges: Sequence[tuple[str, str]]) -> Counter[str]:
    degree: Counter[str] = Counter({node: 0 for node in nodes})
    for source, destination in edges:
        if source == destination:
            raise ValueError("Self-loop found in undirected topology")
        degree[source] += 1
        degree[destination] += 1
    return degree


def deterministic_degree_preserving_rewire(
    nodes: Sequence[str],
    undirected_links: Sequence[tuple[str, str]],
    *,
    accepted_swaps: int = REWIRE_SWAPS,
) -> tuple[tuple[str, str], ...]:
    """Apply deterministic two-edge swaps while preserving node degrees.

    Candidate edge-index pairs are considered lexicographically. For edges
    (a,b),(c,d), the first valid cross-pair (a,d),(c,b), then (a,c),(b,d), is
    accepted. Self-loops and duplicate undirected links are rejected. Edge-list
    slots are retained so each original directed feature/target column remains
    attached to the same slot after bidirectional expansion.
    """
    node_set = set(nodes)
    edges = [tuple(edge) for edge in undirected_links]
    if accepted_swaps <= 0:
        raise ValueError("accepted_swaps must be positive")
    if len(node_set) != len(nodes):
        raise ValueError("Node list contains duplicates")
    original_keys = [_canonical_pair(edge) for edge in edges]
    if len(set(original_keys)) != len(edges):
        raise ValueError("Input topology contains duplicate undirected links")
    if any(a not in node_set or b not in node_set for a, b in edges):
        raise ValueError("Input edge references a node outside the declared node set")
    before_degree = _degree_counts(nodes, edges)

    swap_count = 0
    for i in range(len(edges)):
        for j in range(i + 1, len(edges)):
            if swap_count >= accepted_swaps:
                break
            a, b = edges[i]
            c, d = edges[j]
            other_keys = {
                _canonical_pair(edge)
                for index, edge in enumerate(edges)
                if index not in (i, j)
            }
            accepted: tuple[tuple[str, str], tuple[str, str]] | None = None
            for candidate in (((a, d), (c, b)), ((a, c), (b, d))):
                (u1, v1), (u2, v2) = candidate
                key1, key2 = _canonical_pair((u1, v1)), _canonical_pair((u2, v2))
                if (
                    u1 != v1
                    and u2 != v2
                    and key1 != key2
                    and key1 not in other_keys
                    and key2 not in other_keys
                ):
                    accepted = candidate
                    break
            if accepted is not None:
                edges[i], edges[j] = accepted
                swap_count += 1
        if swap_count >= accepted_swaps:
            break

    if swap_count != accepted_swaps:
        raise ValueError(f"Could perform only {swap_count} of {accepted_swaps} requested swaps")
    after_keys = [_canonical_pair(edge) for edge in edges]
    if len(set(after_keys)) != len(edges):
        raise ValueError("Rewire unexpectedly created duplicate undirected links")
    if set(after_keys) == set(original_keys):
        raise ValueError("Rewire did not change the undirected topology")
    if _degree_counts(nodes, edges) != before_degree:
        raise ValueError("Rewire changed the undirected degree sequence")
    return tuple(edges)


def _validate_rewire_contract(
    nodes: Sequence[str],
    original_undirected: Sequence[tuple[str, str]],
    rewired_undirected: Sequence[tuple[str, str]],
    original_directed: Sequence[tuple[str, str]],
    rewired_directed: Sequence[tuple[str, str]],
) -> dict[str, Any]:
    if set(nodes) != {endpoint for edge in rewired_directed for endpoint in edge}:
        raise ValueError("Rewire changed the topology node set")
    if len(rewired_undirected) != len(original_undirected):
        raise ValueError("Rewire changed the undirected edge count")
    if len(rewired_directed) != 2 * len(rewired_undirected):
        raise ValueError("Rewire is not represented by two directed arcs per undirected edge")
    for index in range(0, len(rewired_directed), 2):
        forward, reverse = rewired_directed[index : index + 2]
        if forward != (reverse[1], reverse[0]):
            raise ValueError("Directed-edge expansion is not interleaved forward/reverse")
    original_degree = _degree_counts(nodes, original_undirected)
    rewired_degree = _degree_counts(nodes, rewired_undirected)
    if original_degree != rewired_degree:
        raise ValueError("Undirected degree sequence changed")
    # Bidirectional expansion makes each node's directed in- and out-degree
    # equal to its undirected degree; verify both vectors explicitly.
    original_in = Counter({node: 0 for node in nodes})
    original_out = Counter({node: 0 for node in nodes})
    rewired_in = Counter({node: 0 for node in nodes})
    rewired_out = Counter({node: 0 for node in nodes})
    for source, destination in original_directed:
        original_out[source] += 1
        original_in[destination] += 1
    for source, destination in rewired_directed:
        rewired_out[source] += 1
        rewired_in[destination] += 1
    if original_in != rewired_in or original_out != rewired_out:
        raise ValueError("Directed in/out-degree constraints changed")
    if set(map(_canonical_pair, original_undirected)) == set(map(_canonical_pair, rewired_undirected)):
        raise ValueError("Rewired graph is identical to the original graph")
    return {
        "node_count": len(nodes),
        "undirected_edge_count": len(original_undirected),
        "directed_edge_count": len(original_directed),
        "node_set_preserved": True,
        "undirected_degree_sequence_preserved": True,
        "directed_in_degree_preserved": True,
        "directed_out_degree_preserved": True,
        "reciprocal_interleaved_arcs": True,
        "topology_changed": True,
        "original_edges": len(original_undirected),
        "rewired_edges": len(rewired_undirected),
    }


def _metric(target: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    if target.shape != prediction.shape:
        raise ValueError("Validation target and prediction shapes differ")
    if not np.isfinite(target).all() or not np.isfinite(prediction).all():
        raise ValueError("Nonfinite target or prediction; refusing to alter evaluation coverage")
    error = prediction.astype(np.float64) - target.astype(np.float64)
    return {
        "mae": float(np.mean(np.abs(error), dtype=np.float64)),
        "rmse": float(np.sqrt(np.mean(np.square(error), dtype=np.float64))),
        "evaluated_pairs": int(target.size),
    }


def _prediction_difference(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float | int]:
    if reference.shape != candidate.shape:
        raise ValueError("Intervention prediction shape differs from original")
    delta = np.abs(candidate.astype(np.float64) - reference.astype(np.float64))
    if not np.isfinite(delta).all():
        raise ValueError("Nonfinite prediction difference")
    return {
        "max_absolute_prediction_difference": float(delta.max(initial=0.0)),
        "mean_absolute_prediction_difference": float(delta.mean(dtype=np.float64)),
        "changed_predictions": int(np.count_nonzero(delta > TOLERANCE)),
        "total_predictions": int(delta.size),
        "fraction_changed_beyond_tolerance": float(np.mean(delta > TOLERANCE)),
        "absolute_change_tolerance": TOLERANCE,
    }


def _load_manifest(seed: int) -> tuple[dict[str, Any], Path]:
    run_dir = V2_DATA_ROOT / f"neural_bu_v2_full_seed_{seed}"
    manifest_path = run_dir / "run_metadata.json"
    checkpoint = run_dir / "best_validation.pt"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "complete"
        or manifest.get("mode") != "full"
        or manifest.get("seed") != seed
        or manifest.get("protocol_id") != PROTOCOL_ID
        or manifest.get("candidate") != "B-U"
        or manifest.get("model") != "EdgeQoSForecaster (unchanged)"
    ):
        raise ValueError(f"Seed {seed} manifest identity/status mismatch")
    if manifest.get("model_settings") != TRAINING:
        raise ValueError(f"Seed {seed} model settings differ from the recorded B-U settings")
    if manifest.get("normalization", {}).get("fit_scope") != "all scalar values in X_train.npy and y_train.npy only":
        raise ValueError(f"Seed {seed} normalization provenance is not train-only")
    expected_checkpoint = (run_dir / "best_validation.pt").resolve()
    if Path(str(manifest.get("checkpoint", ""))).resolve() != expected_checkpoint:
        raise ValueError(f"Seed {seed} checkpoint path differs from its explicit run path")
    if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
        raise FileNotFoundError(f"Seed {seed} checkpoint is missing or empty")
    return manifest, checkpoint


def _predict(
    model: EdgeQoSForecaster,
    histories: np.ndarray,
    edge_index: torch.Tensor,
    node_count: int,
    normalization: Normalization,
) -> np.ndarray:
    batches = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(histories), TRAINING["batch_size"]):
            batch = histories[start : start + TRAINING["batch_size"]]
            sequence = make_bu_sequence(batch, edge_index, node_count, normalization)
            standardized = model(sequence).reshape(len(batch), edge_index.shape[1])
            if not torch.isfinite(standardized).all():
                raise ValueError("Model produced nonfinite validation output")
            original_units = (
                standardized.cpu().numpy().astype(np.float64) * normalization.scale
                + normalization.mean
            )
            batches.append(original_units)
    result = np.concatenate(batches, axis=0)
    if result.shape != (len(histories), edge_index.shape[1]):
        raise ValueError("Validation prediction shape does not match [windows, directed links]")
    return result


def _load_model(checkpoint: Path) -> EdgeQoSForecaster:
    model = EdgeQoSForecaster(
        edge_feature_count=1,
        target_count=1,
        hidden_channels=TRAINING["hidden_channels"],
        gru_hidden_size=TRAINING["gru_hidden_size"],
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def run(output_dir: Path = OUTPUT_DIR) -> Path:
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite topology diagnostic: {output_dir}")
    if not output_dir.resolve().is_relative_to(V2_DATA_ROOT.resolve()):
        raise ValueError("Diagnostic output must remain under the v2 data root")

    manifests: dict[int, dict[str, Any]] = {}
    checkpoints: dict[int, Path] = {}
    normalizations: dict[int, Normalization] = {}
    for seed in SEEDS:
        manifest, checkpoint = _load_manifest(seed)
        manifests[seed] = manifest
        checkpoints[seed] = checkpoint
        normalization = manifest["normalization"]
        normalizations[seed] = Normalization(
            mean=float(normalization["mean"]), scale=float(normalization["scale"])
        )
        if not np.isfinite(normalizations[seed].mean) or not np.isfinite(normalizations[seed].scale) or normalizations[seed].scale <= 0:
            raise ValueError(f"Seed {seed} has invalid recorded normalization")
    norm_values = {(value.mean, value.scale) for value in normalizations.values()}
    if len(norm_values) != 1:
        raise ValueError("Seeds do not share identical train-only normalization")

    nodes, undirected = parse_topology(str(TOPOLOGY_PATH))
    original_edge_index, node_order, original_edges = build_edge_index(nodes, undirected)
    expected_edge_records = [
        {"index": index, "source": source, "destination": destination}
        for index, (source, destination) in enumerate(original_edges)
    ]
    for seed in SEEDS:
        if manifests[seed].get("directed_edge_order") != expected_edge_records:
            raise ValueError(f"Seed {seed} edge-column order differs from topology construction")
    if len(node_order) != 22 or len(undirected) != 36 or len(original_edges) != 72:
        raise ValueError("Topology violates the established 22-node/36-link/72-arc contract")

    rewired_undirected = deterministic_degree_preserving_rewire(
        node_order, undirected, accepted_swaps=REWIRE_SWAPS
    )
    rewired_edge_index, rewired_node_order, rewired_edges = build_edge_index(
        node_order, rewired_undirected
    )
    if rewired_node_order != node_order:
        raise ValueError("Rewired topology changed sorted model node order")
    rewire_contract = _validate_rewire_contract(
        node_order, undirected, rewired_undirected, original_edges, rewired_edges
    )

    # These are the only NumPy arrays read. Histories and targets retain their
    # original 72-column order; no target or edge-attribute reordering occurs.
    histories = load_permitted_split_array(ARRAY_ROOT, "X_validation.npy")
    targets = load_permitted_split_array(ARRAY_ROOT, "y_validation.npy")
    if histories.dtype != np.float32 or targets.dtype != np.float32:
        raise ValueError("Validation arrays must retain v2 float32 dtype")
    if histories.ndim != 3 or histories.shape[1:] != (HISTORY_LENGTH, len(original_edges)):
        raise ValueError("Unexpected X_validation shape")
    if targets.shape != (histories.shape[0], len(original_edges)):
        raise ValueError("Unexpected y_validation shape")
    if histories.shape[0] == 0 or not np.isfinite(histories).all() or not np.isfinite(targets).all():
        raise ValueError("Validation data is empty or nonfinite")

    results: dict[str, Any] = {
        "script_version": SCRIPT_VERSION,
        "protocol_id": PROTOCOL_ID,
        "split": "validation only",
        "arrays_read": [str(X_VALIDATION), str(Y_VALIDATION)],
        "history_shape": list(histories.shape),
        "target_shape": list(targets.shape),
        "evaluation": {
            "metrics": "MAE and RMSE in original dimensionless computed-utilization-ratio units; float64 accumulation",
            "changed_prediction_tolerance": TOLERANCE,
            "prediction_pairs": int(targets.size),
            "history_length": HISTORY_LENGTH,
            "prediction_horizon": 1,
        },
        "intervention": {
            "name": "deterministic undirected two-edge swaps with reciprocal directed expansion",
            "accepted_swaps": REWIRE_SWAPS,
            "algorithm": "lexicographic edge-slot pairs; for (a,b),(c,d), accept first valid (a,d),(c,b), else (a,c),(b,d); reject self-loops/duplicates",
            "edge_attribute_target_correspondence": "unchanged by column: validation history and target columns are never reordered; rewire changes connectivity only",
            "rewire_contract": rewire_contract,
            "original_directed_edges": expected_edge_records,
            "rewired_directed_edges": [
                {"index": index, "source": source, "destination": destination}
                for index, (source, destination) in enumerate(rewired_edges)
            ],
        },
        "seeds": {},
        "interpretation": "Prediction sensitivity to a topology intervention is not evidence that the original topology improves forecasting. The source-level derivation predicts invariance because all node features are equal and every node has positive in-degree.",
        "limitations": [
            "Validation was used previously for checkpoint selection; this is a fixed-checkpoint sensitivity diagnostic, not a generalization estimate.",
            "A rewired endpoint assignment is a counterfactual graph; edge-history/target columns intentionally remain in the original array order.",
            "No training, model selection, clipping, imputation, or test-split access is performed.",
        ],
    }

    for seed in SEEDS:
        model = _load_model(checkpoints[seed])
        normalization = normalizations[seed]
        original_prediction = _predict(
            model, histories, original_edge_index, len(node_order), normalization
        )
        no_op_prediction = _predict(
            model, histories, original_edge_index.clone(), len(node_order), normalization
        )
        rewired_prediction = _predict(
            model, histories, rewired_edge_index, len(node_order), normalization
        )
        original_metrics = _metric(targets, original_prediction)
        no_op_metrics = _metric(targets, no_op_prediction)
        rewired_metrics = _metric(targets, rewired_prediction)
        no_op_difference = _prediction_difference(original_prediction, no_op_prediction)
        rewired_difference = _prediction_difference(original_prediction, rewired_prediction)
        if no_op_difference["max_absolute_prediction_difference"] > TOLERANCE:
            raise ValueError(f"Seed {seed} no-op topology control changed predictions")
        results["seeds"][str(seed)] = {
            "checkpoint": str(checkpoints[seed]),
            "best_validation_epoch": manifests[seed]["best_validation_epoch"],
            "recorded_validation_smooth_l1_standardized": manifests[seed]["best_validation_loss"],
            "normalization": {
                "mean": normalization.mean,
                "scale": normalization.scale,
                "fit_scope": manifests[seed]["normalization"]["fit_scope"],
            },
            "original_topology": original_metrics,
            "no_op_control": {
                "metrics": no_op_metrics,
                "prediction_difference": no_op_difference,
                "mae_change": no_op_metrics["mae"] - original_metrics["mae"],
                "rmse_change": no_op_metrics["rmse"] - original_metrics["rmse"],
            },
            "degree_preserving_rewire": {
                "metrics": rewired_metrics,
                "prediction_difference": rewired_difference,
                "mae_change": rewired_metrics["mae"] - original_metrics["mae"],
                "rmse_change": rewired_metrics["rmse"] - original_metrics["rmse"],
            },
        }

    output_dir.mkdir(parents=False, exist_ok=False)
    (output_dir / "results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(_render_report(results), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "seeds": results["seeds"]}, indent=2))
    return output_dir


def _render_report(results: dict[str, Any]) -> str:
    rows = [
        "# GEANT v2 B-U topology-sensitivity diagnostic",
        "",
        "Validation-only inference using the three retained full-mode checkpoints; no training or test access.",
        "",
        f"Validation shapes: X `{results['history_shape']}`, y `{results['target_shape']}`; {results['evaluation']['prediction_pairs']} target-link pairs.",
        f"Topology intervention: {results['intervention']['accepted_swaps']} deterministic two-edge swaps, preserving node set, undirected degrees, directed in/out degrees, reciprocal arcs, edge count, and edge-column history/target mapping.",
        f"Prediction-change tolerance: {TOLERANCE:g} absolute utilization-ratio units.",
        "",
        "| Seed | Intervention | MAE | RMSE | ΔMAE | ΔRMSE | Max Δprediction | Changed fraction |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for seed, record in results["seeds"].items():
        original = record["original_topology"]
        rows.append(f"| {seed} | Original | {original['mae']:.10g} | {original['rmse']:.10g} | — | — | 0 | 0 |")
        for label, key in (("No-op control", "no_op_control"), ("Rewired", "degree_preserving_rewire")):
            item = record[key]
            diff = item["prediction_difference"]
            rows.append(f"| {seed} | {label} | {item['metrics']['mae']:.10g} | {item['metrics']['rmse']:.10g} | {item['mae_change']:.10g} | {item['rmse_change']:.10g} | {diff['max_absolute_prediction_difference']:.10g} | {diff['fraction_changed_beyond_tolerance']:.10g} |")
    rows.extend([
        "",
        "## Source-based diagnosis",
        "",
        "For each node i, the graph convolution computes W_self x_i + W_neighbor (sum over incoming x_j / max(d_i,1)). The adapter sets every x_i to the same all-ones vector, and all topology nodes have positive in-degree. Therefore each incoming mean is the same vector; both GCN layers produce identical node embeddings regardless of neighbor identities. The edge encoder then combines identical endpoint embeddings with that edge's own edge_attr, and the GRU operates independently per directed edge. The model has no path for neighboring links' utilization histories to affect one another.",
        "",
        results["interpretation"],
        "",
        "A changed prediction would establish functional sensitivity only, not predictive benefit. Invariance is the expected result under the audited implementation. The no-op control must have zero changes within tolerance; otherwise the inference path is suspect.",
        "",
        "## Reproducibility and limits",
        "",
        "The only arrays read are the two validation arrays listed in results.json. Checkpoint normalization is reused as recorded (training-only scope); predictions are inverse-transformed before metrics. No edge attributes, targets, windows, or model weights are reordered or modified.",
        "",
        "This analysis is validation-only, and validation was previously used for checkpoint selection. It does not establish held-out generalization or justify a claim that topology improves forecasting.",
        "",
    ])
    return "\n".join(rows)


if __name__ == "__main__":
    run()
