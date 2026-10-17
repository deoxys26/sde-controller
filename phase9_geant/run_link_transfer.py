"""Controlled linear link-transfer benchmark on GEANT-derived utilization.

The official test block is intentionally not loaded or evaluated here.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np


HISTORY_LENGTH = 5
RIDGE_ALPHA = 1.0
EWMA_ALPHA = 0.8  # Frozen Prompt 5 validation-selected baseline parameter.
SHUFFLE_SEED = 20261011
MIN_PRACTICAL_RELATIVE_GAIN = 0.01
BOOTSTRAP_REPLICATES = 2000


def build_valid_starts(valid_mask, block_start: int, block_end: int, history: int = HISTORY_LENGTH):
    """Return starts whose complete history and next target stay valid/in-block."""
    if history < 2 or block_start < 0 or block_end > len(valid_mask) or block_end <= block_start:
        raise ValueError("Invalid chronological block or history length")
    starts = []
    for start in range(block_start, block_end - history):
        if bool(np.asarray(valid_mask[start : start + history + 1]).all()):
            starts.append(start)
    return np.asarray(starts, dtype=np.int64)


def build_physical_weights(edges):
    """Row-normalized directed predecessor adjacency; row=target, col=source."""
    edge_list = [tuple(edge) for edge in edges]
    count = len(edge_list)
    weights = np.zeros((count, count), dtype=np.float64)
    for target, (target_source, _) in enumerate(edge_list):
        predecessors = [
            source
            for source, (_, source_destination) in enumerate(edge_list)
            if source != target and source_destination == target_source
        ]
        if predecessors:
            weights[target, predecessors] = 1.0 / len(predecessors)
    return weights


def build_ordered_path_incidence(paths, nodes, edges, mean_od_demand):
    """Build row-normalized source-before-target OD-path weights.

    ``paths`` maps ``"src→dst"`` to a node sequence. The OD demand vector
    follows lexicographically sorted ``nodes`` order, excluding self-pairs.
    Each positive weight is the training-mean Mbps contribution from OD paths
    that traverse source link before target link.
    """
    node_list = list(nodes)
    edge_list = [tuple(edge) for edge in edges]
    edge_index = {edge: i for i, edge in enumerate(edge_list)}
    od_pairs = [(src, dst) for src in node_list for dst in node_list if src != dst]
    demand = np.asarray(mean_od_demand, dtype=np.float64)
    if demand.shape != (len(od_pairs),) or np.any(demand < 0) or not np.isfinite(demand).all():
        raise ValueError("Mean OD demand must be a finite nonnegative vector in canonical OD order")

    ordered = np.zeros((len(edge_list), len(edge_list)), dtype=np.float64)
    for od_idx, (src, dst) in enumerate(od_pairs):
        path = paths.get(f"{src}→{dst}")
        if not path or path[0] != src or path[-1] != dst:
            raise ValueError(f"Missing or malformed path for OD pair {src}->{dst}")
        path_edges = [edge_index[(a, b)] for a, b in zip(path[:-1], path[1:])]
        for pos, target in enumerate(path_edges):
            for source in path_edges[:pos]:
                ordered[target, source] += demand[od_idx]
    np.fill_diagonal(ordered, 0.0)
    row_sums = ordered.sum(axis=1, keepdims=True)
    return np.divide(ordered, row_sums, out=np.zeros_like(ordered), where=row_sums > 0)


def shuffle_incidence_weights(weights, seed: int = SHUFFLE_SEED):
    """Deterministically permute each target row's non-self source assignments.

    Per-row weight multiset, sparsity, and sum are preserved; link identities
    are shuffled independently by target. Self-links remain excluded.
    """
    values = np.asarray(weights, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("Incidence weights must be a square matrix")
    rng = np.random.default_rng(seed)
    result = np.zeros_like(values)
    for target in range(values.shape[0]):
        sources = np.flatnonzero(np.arange(values.shape[1]) != target)
        result[target, sources] = values[target, sources][rng.permutation(len(sources))]
    return result


def make_features(util, demand, starts, weights=None, history: int = HISTORY_LENGTH):
    """Create own-change, recent-demand, and optional weighted spatial features."""
    starts = np.asarray(starts, dtype=np.int64)
    util = np.asarray(util)
    demand = np.asarray(demand)
    edge_count = util.shape[1]
    if np.any(starts < 0) or np.any(starts + history >= len(util)):
        raise ValueError("A sample history/target crosses the available series boundary")
    hist = np.stack([util[s : s + history] for s in starts]).astype(np.float64)
    if not np.isfinite(hist).all():
        raise ValueError("A sample history contains an invalid observation")
    changes = np.diff(hist, axis=1)
    recent_demand_change = np.asarray(
        [demand[s + history - 1] - demand[s + history - 2] for s in starts],
        dtype=np.float64,
    )
    own = np.concatenate(
        [changes.transpose(0, 2, 1), np.broadcast_to(recent_demand_change[:, None, None], (len(starts), edge_count, 1))],
        axis=2,
    )
    target = np.stack([util[s + history] - util[s + history - 1] for s in starts]).astype(np.float64)
    target_utilization = np.stack([util[s + history] for s in starts]).astype(np.float64)
    last = hist[:, -1, :]
    spatial = None
    if weights is not None:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != (edge_count, edge_count) or not np.isfinite(weights).all():
            raise ValueError("Spatial weights have an incompatible shape or nonfinite values")
        spatial = np.stack([changes[:, lag, :] @ weights.T for lag in range(history - 1)], axis=2)
    return {"own": own, "spatial": spatial, "target_delta": target, "target_utilization": target_utilization, "last": last, "history": hist}


def fit_ridge_predict(train_x, train_y, eval_x, alpha: float = RIDGE_ALPHA):
    """Fit independent per-target standardized ridge regressions on training only."""
    train_x = np.asarray(train_x, dtype=np.float64)
    train_y = np.asarray(train_y, dtype=np.float64)
    eval_x = np.asarray(eval_x, dtype=np.float64)
    if train_x.ndim != 3 or eval_x.ndim != 3 or train_x.shape[1:] != eval_x.shape[1:]:
        raise ValueError("Expected aligned [samples, target links, features] matrices")
    if train_y.shape != train_x.shape[:2] or not all(np.isfinite(a).all() for a in (train_x, train_y, eval_x)):
        raise ValueError("Training data must be finite and shape-aligned")
    if alpha < 0:
        raise ValueError("Ridge alpha must be nonnegative")
    n_eval, edge_count, feature_count = eval_x.shape
    predictions = np.zeros((n_eval, edge_count), dtype=np.float64)
    parameters = []
    for edge in range(edge_count):
        x = train_x[:, edge, :]
        xe = eval_x[:, edge, :]
        xm = x.mean(axis=0)
        xs = x.std(axis=0)
        xs = np.where(xs > 1e-12, xs, 1.0)
        ym = float(train_y[:, edge].mean())
        ys = float(train_y[:, edge].std())
        ys = ys if ys > 1e-12 else 1.0
        xz = (x - xm) / xs
        yz = (train_y[:, edge] - ym) / ys
        xez = (xe - xm) / xs
        coef = np.linalg.solve(xz.T @ xz + alpha * np.eye(feature_count), xz.T @ yz)
        predictions[:, edge] = ym + ys * (xez @ coef)
        parameters.append({"x_mean": xm, "x_scale": xs, "y_mean": ym, "y_scale": ys, "coef": coef})
    return predictions, parameters


def ewma_prediction(history, alpha: float = EWMA_ALPHA):
    """Use the frozen Prompt 5 EWMA convention on oldest-to-newest histories."""
    history = np.asarray(history, dtype=np.float64)
    powers = np.arange(history.shape[1])
    weights = alpha * ((1.0 - alpha) ** (history.shape[1] - 1 - powers))
    weights = weights / weights.sum()
    return np.tensordot(history, weights, axes=([1], [0]))


def summarize_predictions(target, predictions, last):
    """Summarize reconstructed utilization metrics and per-link MAE."""
    result = {}
    for name, prediction in predictions.items():
        prediction = np.asarray(prediction, dtype=np.float64)
        error = np.asarray(target, dtype=np.float64) - prediction
        per_link = np.mean(np.abs(error), axis=0)
        result[name] = {
            "mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(error**2))),
            "per_link_mae": per_link.tolist(),
        }
    result["sample_count"] = int(target.shape[0])
    result["link_count"] = int(target.shape[1])
    return result


def pack_parameters(parameters):
    """Convert per-target fitted ridge scalers/coefficients to compact arrays."""
    return {
        "x_mean": np.stack([row["x_mean"] for row in parameters]),
        "x_scale": np.stack([row["x_scale"] for row in parameters]),
        "y_mean": np.asarray([row["y_mean"] for row in parameters]),
        "y_scale": np.asarray([row["y_scale"] for row in parameters]),
        "coef": np.stack([row["coef"] for row in parameters]),
    }


def describe_weights(weights):
    weights = np.asarray(weights, dtype=np.float64)
    return {
        "nonzero_directed_source_relations": int(np.count_nonzero(weights)),
        "empty_target_rows": int(np.sum(weights.sum(axis=1) == 0)),
        "row_sum_min": float(weights.sum(axis=1).min()),
        "row_sum_max": float(weights.sum(axis=1).max()),
    }


def link_bootstrap_effects(predictions, target, baseline_name, model_names, seed, reps=BOOTSTRAP_REPLICATES):
    """Link-cluster bootstrap for paired MAE gains; timepoints are not resampled as IID."""
    target = np.asarray(target, dtype=np.float64)
    edge_count = target.shape[1]
    rng = np.random.default_rng(seed)
    rows = {}
    link_ids = np.arange(edge_count)
    for model in model_names:
        base_err = np.abs(target - predictions[baseline_name])
        model_err = np.abs(target - predictions[model])
        gains = base_err - model_err
        per_link_gain = gains.mean(axis=0)
        draws = np.asarray([
            per_link_gain[rng.choice(link_ids, size=edge_count, replace=True)].mean()
            for _ in range(reps)
        ])
        rows[model] = {
            "mean_mae_gain": float(per_link_gain.mean()),
            "mean_relative_mae_gain": float(per_link_gain.mean() / max(float(base_err.mean()), 1e-15)),
            "link_bootstrap_95ci_absolute_gain": np.quantile(draws, [0.025, 0.975]).tolist(),
            "link_bootstrap_replicates": reps,
        }
    return rows


def link_bootstrap_contrasts(predictions, target, comparisons, seed, reps=BOOTSTRAP_REPLICATES):
    """Paired link-cluster bootstrap for challenger-minus-control MAE gains."""
    target = np.asarray(target, dtype=np.float64)
    edge_count = target.shape[1]
    rng = np.random.default_rng(seed)
    result = {}
    for challenger, control in comparisons:
        gain = np.abs(target - predictions[control]) - np.abs(target - predictions[challenger])
        per_link = gain.mean(axis=0)
        draws = np.asarray([
            per_link[rng.choice(edge_count, size=edge_count, replace=True)].mean()
            for _ in range(reps)
        ])
        result[f"{challenger}_vs_{control}"] = {
            "mean_mae_gain": float(per_link.mean()),
            "link_bootstrap_95ci_absolute_gain": np.quantile(draws, [0.025, 0.975]).tolist(),
            "link_bootstrap_replicates": reps,
        }
    return result


def make_protocol():
    return {
        "scope": "forecasting of computed link utilization from real GEANT traffic matrices",
        "series": "distance-shortest-path; official test block not loaded",
        "history_length": HISTORY_LENGTH,
        "target": "next utilization change; reconstructed utilization = last observed utilization + predicted change",
        "own_features": ["four most recent per-link utilization changes", "recent total-demand change"],
        "spatial_features": "same four change lags aggregated by target-row/source-column weights",
        "physical_weights": "directed predecessor-to-successor adjacency, uniform row normalization, self excluded",
        "od_incidence_weights": "mean training-fold OD Mbps contribution on paths where source link precedes target link; row-normalized",
        "shuffled_control": "per-target seeded permutation of non-self source assignments; preserves each row weight multiset, sum and sparsity",
        "regression": {"model": "independent per-target standardized ridge", "alpha": RIDGE_ALPHA, "scaler_fit": "training fold only"},
        "folds": [
            {"name": "fold_1", "train": [0, 4010], "evaluation": [4010, 5347]},
            {"name": "fold_2", "train": [0, 5347], "evaluation": [5347, 6684]},
            {"name": "fold_3", "train": [0, 6684], "evaluation": [6684, 8020]},
        ],
        "window_rule": "five consecutive valid history observations and next target; all within the corresponding fold/block",
        "loss": "MAE on reconstructed utilization; RMSE secondary; ridge fit uses standardized change-space squared error",
        "reference_baselines": {"persistence": "last observed utilization", "ewma_alpha": EWMA_ALPHA, "ewma_alpha_source": "frozen audited Prompt 5 value; not retuned"},
        "validation": "single robustness check after protocol lock; weights and coefficients fit on all train only; no selection",
        "uncertainty": "paired link-cluster bootstrap, 2000 replicates; links rather than timestep-link observations are resampled",
        "minimum_practical_gain": {"relative_mae": MIN_PRACTICAL_RELATIVE_GAIN, "rule": "OD must beat own-history, physical adjacency and shuffled-incidence by at least this fraction; link-bootstrap 95% lower bound for each paired gain must exceed zero; OD-vs-own must meet threshold in at least 2/3 folds"},
        "success_decision": "recommend spatial GNN only if the practical gain rule is met and validation robustness has the same positive direction; otherwise investigate temporal inputs/target",
        "random_seed": SHUFFLE_SEED,
    }


def load_training_od_prefix(root, train_end, valid, paths, nodes, edges):
    """Parse only valid raw training matrices and build cumulative OD demand sums."""
    from phase9_geant.parse_geant import parse_demand_file

    raw_dir = Path("data/raw/geant_topology/directed-geant-uhlig-15min-over-4months-ALL-native")
    files = sorted(raw_dir.glob("demandMatrix-geant-uhlig-15min-*.txt"))
    if len(files) < train_end:
        raise ValueError(f"Expected at least {train_end} chronological raw matrix files; found {len(files)}")
    nodes = list(nodes)
    node_index = {name: i for i, name in enumerate(nodes)}
    od_pairs = [(src, dst) for src in nodes for dst in nodes if src != dst]
    edge_index = {tuple(edge): i for i, edge in enumerate(edges)}
    incidence = np.zeros((len(od_pairs), len(edges)), dtype=np.float64)
    for od_i, (src, dst) in enumerate(od_pairs):
        path = paths[f"{src}→{dst}"]
        for a, b in zip(path[:-1], path[1:]):
            incidence[od_i, edge_index[(a, b)]] = 1.0
    traffic = np.load(Path(root) / "link_series_traffic.npy", mmap_mode="r")
    prefix = np.zeros((train_end + 1, len(od_pairs)), dtype=np.float64)
    max_load_error = 0.0
    for timestep in range(train_end):
        prefix[timestep + 1] = prefix[timestep]
        if not bool(valid[timestep]):
            continue
        timestamp, matrix, count = parse_demand_file(str(files[timestep]), nodes, node_index)
        expected_time = datetime.strptime(files[0].stem.rsplit("-", 2)[-2] + "-" + files[0].stem.rsplit("-", 2)[-1], "%Y%m%d-%H%M") + timedelta(minutes=15 * timestep)
        if timestamp != expected_time or count <= 0:
            raise ValueError(f"Raw matrix chronology/validity mismatch at timestep {timestep}")
        od = np.asarray([matrix[node_index[src], node_index[dst]] for src, dst in od_pairs], dtype=np.float64)
        prefix[timestep + 1] += od
        error = float(np.max(np.abs(od @ incidence - np.asarray(traffic[timestep], dtype=np.float64))))
        max_load_error = max(max_load_error, error)
        if error > 1e-3:
            raise ValueError(f"OD-path reconstruction mismatch at timestep {timestep}: {error} Mbps")
    return prefix, incidence, od_pairs, max_load_error


def _folds(train_end):
    expected = [0, int(round(train_end * 0.5)), int(round(train_end * 2 / 3)), int(round(train_end * 5 / 6)), train_end]
    # Stable, explicit block boundaries used by the saved experiment protocol.
    if train_end != 8020:
        raise ValueError("Frozen Prompt 11 fold boundaries expect train block [0, 8020)")
    return [
        ("fold_1", 0, 4010, 4010, 5347),
        ("fold_2", 0, 5347, 5347, 6684),
        ("fold_3", 0, 6684, 6684, 8020),
    ]


def run_benchmark(root=Path("data/processed/geant"), out=Path("data/processed/geant/link_transfer")):
    """Run the fixed training-fold experiment and one validation robustness check."""
    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    protocol = make_protocol()
    (out / "experiment_config.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    split = json.loads((root / "split.json").read_text(encoding="utf-8"))["blocks"]
    train = split["train"]
    train_start, train_end = int(train["start_index"]), int(train["end_index"])
    if train_start != 0:
        raise ValueError("Unexpected train block start")
    valid = np.load(root / "valid_mask.npy", mmap_mode="r")
    util = np.load(root / "link_series_utilization.npy", mmap_mode="r")
    demand = np.load(root / "total_demand_float64.npy", mmap_mode="r")
    slots = np.load(root / "slot_indices.npy", mmap_mode="r")
    nodes_dict, undirected = __import__("phase9_geant.parse_geant", fromlist=["parse_topology"]).parse_topology("data/raw/geant_topology/geant.txt")
    nodes = sorted(nodes_dict)
    edges = [edge for pair in undirected for edge in (pair, (pair[1], pair[0]))]
    edge_names = [f"{a}->{b}" for a, b in edges]
    protocol["directed_link_order"] = edge_names
    (out / "experiment_config.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    paths = json.loads((root / "path_table_distance.json").read_text(encoding="utf-8"))
    prefix, path_incidence, od_pairs, recon_error = load_training_od_prefix(root, train_end, valid, paths, nodes, edges)

    results = {"protocol": protocol, "raw_path_reconstruction_max_error_mbps": recon_error, "folds": {}, "validation": {}}
    saved_predictions = {}
    saved_weights = {}
    saved_parameters = {}
    fold_eval_per_link = {name: [] for name in ("own_history", "physical_adjacency", "od_incidence", "shuffled_incidence", "persistence", "ewma")}
    fold_target_parts = []

    for fold_index, (name, fit_start, fit_end, eval_start, eval_end) in enumerate(_folds(train_end)):
        train_starts = build_valid_starts(valid, fit_start, fit_end)
        eval_starts = build_valid_starts(valid, eval_start, eval_end)
        if len(train_starts) == 0 or len(eval_starts) == 0:
            raise ValueError(f"No valid samples in {name}")
        mean_od = (prefix[fit_end] - prefix[fit_start]) / max(int(np.asarray(valid[fit_start:fit_end]).sum()), 1)
        physical = build_physical_weights(edges)
        incidence_weights = build_ordered_path_incidence(paths, nodes, edges, mean_od)
        shuffled = shuffle_incidence_weights(incidence_weights, SHUFFLE_SEED + fold_index)
        saved_weights[f"{name}_physical_adjacency"] = physical
        saved_weights[f"{name}_od_incidence"] = incidence_weights
        saved_weights[f"{name}_shuffled_incidence"] = shuffled
        tr = make_features(util, demand, train_starts)
        ev = make_features(util, demand, eval_starts)
        eval_target = ev["target_utilization"]
        predictions = {
            "persistence": ev["last"],
            "ewma": ewma_prediction(ev["history"]),
        }
        for model_name, weights in (("physical_adjacency", physical), ("od_incidence", incidence_weights), ("shuffled_incidence", shuffled)):
            train_full = make_features(util, demand, train_starts, weights)
            eval_full = make_features(util, demand, eval_starts, weights)
            train_x = np.concatenate((train_full["own"], train_full["spatial"]), axis=2)
            eval_x = np.concatenate((eval_full["own"], eval_full["spatial"]), axis=2)
            delta_pred, parameters = fit_ridge_predict(train_x, train_full["target_delta"], eval_x)
            predictions[model_name] = ev["last"] + delta_pred
            for key, value in pack_parameters(parameters).items():
                saved_parameters[f"{name}_{model_name}_{key}"] = value
        own_delta, parameters = fit_ridge_predict(tr["own"], tr["target_delta"], ev["own"])
        predictions["own_history"] = ev["last"] + own_delta
        for key, value in pack_parameters(parameters).items():
            saved_parameters[f"{name}_own_history_{key}"] = value
        metrics = summarize_predictions(eval_target, predictions, ev["last"])
        results["folds"][name] = {
            "train_bounds": [fit_start, fit_end],
            "evaluation_bounds": [eval_start, eval_end],
            "train_sample_count": int(len(train_starts)),
            "evaluation_sample_count": int(len(eval_starts)),
            "weight_summary": {
                "physical_adjacency": describe_weights(physical),
                "od_incidence": describe_weights(incidence_weights),
                "shuffled_incidence": describe_weights(shuffled),
            },
            "target_indices": (eval_starts + HISTORY_LENGTH).tolist(),
            "metrics": metrics,
            "per_link_mae": {model: dict(zip(edge_names, row["per_link_mae"])) for model, row in metrics.items() if model != "sample_count" and model != "link_count"},
        }
        fold_target_parts.append(eval_target)
        saved_predictions[f"{name}_target"] = eval_target
        saved_predictions[f"{name}_target_indices"] = eval_starts + HISTORY_LENGTH
        for model, pred in predictions.items():
            saved_predictions[f"{name}_{model}"] = pred
            fold_eval_per_link[model].append(np.mean(np.abs(eval_target - pred), axis=0))

    # Aggregate uncertainty by resampling directed links as clusters across all folds.
    all_targets = np.concatenate(fold_target_parts, axis=0)
    all_preds = {model: np.concatenate([saved_predictions[f"{fold}_{model}"] for fold, *_ in _folds(train_end)], axis=0) for model in fold_eval_per_link}
    results["aggregate_folds"] = summarize_predictions(all_targets, all_preds, all_targets)
    results["aggregate_link_bootstrap"] = link_bootstrap_effects(
        all_preds, all_targets, "own_history",
        ["physical_adjacency", "od_incidence", "shuffled_incidence", "persistence", "ewma"],
        SHUFFLE_SEED,
    )
    results["aggregate_spatial_contrasts"] = link_bootstrap_contrasts(
        all_preds, all_targets,
        [("od_incidence", "physical_adjacency"), ("od_incidence", "shuffled_incidence")],
        SHUFFLE_SEED + 1,
    )
    results["aggregate_per_link_mae"] = {
        model: dict(zip(edge_names, results["aggregate_folds"][model]["per_link_mae"]))
        for model in fold_eval_per_link
    }

    # Frozen validation-only robustness: train everything on the entire official train block.
    val_block = split["val"]
    val_start, val_end = int(val_block["start_index"]), int(val_block["end_index"])
    val_starts = np.asarray(val_block["window_starts"], dtype=np.int64)
    checked_val = build_valid_starts(valid, val_start, val_end)
    if not np.array_equal(val_starts, checked_val):
        raise ValueError("Official validation starts differ from valid block-local windows")
    full_train_starts = build_valid_starts(valid, train_start, train_end)
    mean_od = prefix[train_end] / max(int(np.asarray(valid[train_start:train_end]).sum()), 1)
    physical = build_physical_weights(edges)
    incidence_weights = build_ordered_path_incidence(paths, nodes, edges, mean_od)
    shuffled = shuffle_incidence_weights(incidence_weights, SHUFFLE_SEED + 100)
    saved_weights["validation_physical_adjacency"] = physical
    saved_weights["validation_od_incidence"] = incidence_weights
    saved_weights["validation_shuffled_incidence"] = shuffled
    tr = make_features(util, demand, full_train_starts)
    ev = make_features(util, demand, val_starts)
    val_target = ev["target_utilization"]
    val_predictions = {"persistence": ev["last"], "ewma": ewma_prediction(ev["history"])}
    own_delta, parameters = fit_ridge_predict(tr["own"], tr["target_delta"], ev["own"])
    val_predictions["own_history"] = ev["last"] + own_delta
    for key, value in pack_parameters(parameters).items():
        saved_parameters[f"validation_own_history_{key}"] = value
    for model_name, weights in (("physical_adjacency", physical), ("od_incidence", incidence_weights), ("shuffled_incidence", shuffled)):
        train_full = make_features(util, demand, full_train_starts, weights)
        eval_full = make_features(util, demand, val_starts, weights)
        train_x = np.concatenate((train_full["own"], train_full["spatial"]), axis=2)
        eval_x = np.concatenate((eval_full["own"], eval_full["spatial"]), axis=2)
        delta_pred, parameters = fit_ridge_predict(train_x, train_full["target_delta"], eval_x)
        val_predictions[model_name] = ev["last"] + delta_pred
        for key, value in pack_parameters(parameters).items():
            saved_parameters[f"validation_{model_name}_{key}"] = value
    val_metrics = summarize_predictions(val_target, val_predictions, ev["last"])
    results["validation"] = {
        "train_sample_count": int(len(full_train_starts)),
        "sample_count": int(len(val_starts)),
        "weight_summary": {
            "physical_adjacency": describe_weights(physical),
            "od_incidence": describe_weights(incidence_weights),
            "shuffled_incidence": describe_weights(shuffled),
        },
        "target_indices": (val_starts + HISTORY_LENGTH).tolist(),
        "metrics": val_metrics,
        "per_link_mae": {model: dict(zip(edge_names, row["per_link_mae"])) for model, row in val_metrics.items() if model not in ("sample_count", "link_count")},
    }
    for model, pred in val_predictions.items():
        saved_predictions[f"validation_{model}"] = pred
    saved_predictions["validation_target_indices"] = val_starts + HISTORY_LENGTH
    saved_predictions["validation_target"] = val_target
    results["validation"]["link_bootstrap_vs_own"] = link_bootstrap_effects(
        val_predictions, val_target, "own_history",
        ["physical_adjacency", "od_incidence", "shuffled_incidence", "persistence", "ewma"],
        SHUFFLE_SEED + 101,
    )
    results["validation"]["spatial_contrasts"] = link_bootstrap_contrasts(
        val_predictions, val_target,
        [("od_incidence", "physical_adjacency"), ("od_incidence", "shuffled_incidence")],
        SHUFFLE_SEED + 102,
    )
    results["go_no_go"] = _decision(results)
    (out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    np.savez_compressed(out / "predictions.npz", **saved_predictions)
    np.savez_compressed(out / "weights.npz", **saved_weights)
    np.savez_compressed(out / "model_parameters.npz", **saved_parameters)
    _write_report(out / "report.md", results)
    return results


def _decision(results):
    folds = list(results["folds"].values())
    names = ["physical_adjacency", "od_incidence", "shuffled_incidence"]
    meets = True
    fold_effects = []
    for fold in folds:
        metrics = fold["metrics"]
        own_mae = metrics["own_history"]["mae"]
        row = {}
        for model in names:
            gain = (own_mae - metrics[model]["mae"]) / own_mae
            row[model] = float(gain)
        fold_effects.append(row)
        meets &= row["od_incidence"] >= MIN_PRACTICAL_RELATIVE_GAIN
    boots = results["aggregate_link_bootstrap"]
    od = boots["od_incidence"]
    own_gain_ok = od["mean_relative_mae_gain"] >= MIN_PRACTICAL_RELATIVE_GAIN and od["link_bootstrap_95ci_absolute_gain"][0] > 0
    own_mae = results["aggregate_folds"]["own_history"]["mae"]
    controls_ok = all(
        results["aggregate_spatial_contrasts"][f"od_incidence_vs_{control}"]["mean_mae_gain"] >= MIN_PRACTICAL_RELATIVE_GAIN * own_mae
        and results["aggregate_spatial_contrasts"][f"od_incidence_vs_{control}"]["link_bootstrap_95ci_absolute_gain"][0] > 0
        for control in ("physical_adjacency", "shuffled_incidence")
    )
    val = results["validation"]
    validation_direction = all(
        val["metrics"][control]["mae"] > val["metrics"]["od_incidence"]["mae"]
        for control in ("own_history", "physical_adjacency", "shuffled_incidence")
    )
    return {
        "fold_relative_gain_vs_own_history": fold_effects,
        "minimum_practical_relative_gain": MIN_PRACTICAL_RELATIVE_GAIN,
        "od_beats_own_history_practically_in_at_least_2_of_3_folds": bool(sum(x["od_incidence"] >= MIN_PRACTICAL_RELATIVE_GAIN for x in fold_effects) >= 2),
        "od_mean_gain_vs_own_history_and_link_bootstrap_lower_bound_positive": bool(own_gain_ok),
        "od_distinguished_from_both_spatial_controls": bool(controls_ok),
        "validation_robustness_same_direction": bool(validation_direction),
        "spatial_gnn_go": bool(own_gain_ok and controls_ok and validation_direction and sum(x["od_incidence"] >= MIN_PRACTICAL_RELATIVE_GAIN for x in fold_effects) >= 2),
    }


def _write_report(path, results):
    lines = [
        "# Controlled GEANT Link-Transfer Benchmark",
        "",
        "This is forecasting of computed link utilization from real GEANT traffic matrices. The official test block was not loaded or evaluated.",
        "",
        "## Fixed protocol",
        "",
        "- Five consecutive observations form the history; the target is next-step utilization change, reconstructed by adding the predicted change to the final observation.",
        "- Predictors: four own-link utilization changes and the most recent total-demand change; spatial models add four source-change aggregates.",
        "- Physical weights are uniform, row-normalized directed predecessor-to-successor adjacency. OD weights use training-fold mean demand on paths where the source edge precedes the target edge, then row-normalize. The shuffle permutes each target row's non-self weights with a fixed seed, retaining each row's weight multiset and sparsity.",
        "- Each target link has a standardized ridge regression (alpha 1.0); all centering/scaling, coefficients, and OD weights are fit on each training fold only.",
        "- Fold windows remain inside valid contiguous fold bounds. Validation is one robustness check with full-train fitting; no validation tuning is performed. Persistence and frozen EWMA alpha 0.8 are references.",
        "- Practical threshold: at least 1% relative MAE improvement over own-history and each spatial control, positive link-bootstrap uncertainty, and at least two of three folds meet the threshold.",
        "",
        "## Training-fold results",
        "",
        "MAE/RMSE are on reconstructed utilization. Positive gains favor the named spatial model.",
        "",
        "| Fold | N | Own MAE/RMSE | Physical MAE/RMSE | OD-incidence MAE/RMSE | Shuffled MAE/RMSE | Persistence MAE/RMSE | EWMA MAE/RMSE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, fold in results["folds"].items():
        m = fold["metrics"]
        lines.append(
            f"| {name} | {fold['evaluation_sample_count']} | "
            + " | ".join(f"{m[model]['mae']:.8f} / {m[model]['rmse']:.8f}" for model in ("own_history", "physical_adjacency", "od_incidence", "shuffled_incidence", "persistence", "ewma"))
            + " |"
        )
    lines.extend(["", "### Aggregate link-bootstrap effects", ""])
    for model, effect in results["aggregate_link_bootstrap"].items():
        low, high = effect["link_bootstrap_95ci_absolute_gain"]
        lines.append(f"- {model} vs own-history: MAE gain {effect['mean_mae_gain']:.8g} ({100*effect['mean_relative_mae_gain']:.3f}%); 95% link-bootstrap CI [{low:.8g}, {high:.8g}].")
    for comparison, effect in results["aggregate_spatial_contrasts"].items():
        low, high = effect["link_bootstrap_95ci_absolute_gain"]
        lines.append(f"- {comparison}: MAE gain {effect['mean_mae_gain']:.8g}; 95% link-bootstrap CI [{low:.8g}, {high:.8g}].")
    lines.extend(["", "Aggregate reconstructed-utilization metrics across the three training folds:", ""])
    for model in ("own_history", "physical_adjacency", "od_incidence", "shuffled_incidence", "persistence", "ewma"):
        metric = results["aggregate_folds"][model]
        lines.append(f"- {model}: MAE {metric['mae']:.8f}; RMSE {metric['rmse']:.8f}.")
    lines.extend(["", "Spatial weight summaries (nonzero directed source relations; empty target rows):", ""])
    for name, fold in results["folds"].items():
        summary = fold["weight_summary"]
        lines.append(f"- {name}: " + ", ".join(f"{m} {summary[m]['nonzero_directed_source_relations']} / empty rows {summary[m]['empty_target_rows']}" for m in ("physical_adjacency", "od_incidence", "shuffled_incidence")) + ".")
    lines.extend(["", "Per-link MAE for every fold and validation target is stored in results.json; time-aligned forecasts, weight matrices, and fitted per-link scalers/coefficients are saved in predictions.npz, weights.npz, and model_parameters.npz.", "", "## Validation robustness", ""])
    for model in ("own_history", "physical_adjacency", "od_incidence", "shuffled_incidence", "persistence", "ewma"):
        lines.append(f"- {model}: MAE {results['validation']['metrics'][model]['mae']:.8f}; RMSE {results['validation']['metrics'][model]['rmse']:.8f}.")
    lines.extend(["", "## Decision", "", f"`spatial_gnn_go`: **{results['go_no_go']['spatial_gnn_go']}**. This is a screening decision under the predeclared threshold, not proof that spatial dependence is absent. See `go_no_go` in results.json for component checks and link-bootstrap intervals.", "", f"Raw OD-to-link reconstruction maximum error: {results['raw_path_reconstruction_max_error_mbps']:.6g} Mbps."])
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    result = run_benchmark()
    print(json.dumps({"go_no_go": result["go_no_go"], "validation_mae": {k: v["mae"] for k, v in result["validation"]["metrics"].items() if isinstance(v, dict) and "mae" in v}}, indent=2))


if __name__ == "__main__":
    main()
