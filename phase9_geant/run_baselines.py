"""
phase9_geant/run_baselines.py — Evaluate non-neural forecasting baselines on val and test.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, mean_target: float) -> Dict[str, float]:
    """Compute overall MAE, RMSE, and normalized MAE."""
    if y_true.shape != y_pred.shape or not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError("Metrics require aligned, finite target and prediction arrays")
    diff = y_true - y_pred
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    norm_mae = float(mae / mean_target) if mean_target > 0 else 0.0
    return {
        "mae": mae,
        "rmse": rmse,
        "norm_mae": norm_mae,
    }


def compute_per_link_metrics(y_true: np.ndarray, y_pred: np.ndarray, edge_names: List[str]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[int, float]]:
    """Compute MAE per link and return best 5, worst 5, and all link MAEs."""
    link_mae = np.mean(np.abs(y_true - y_pred), axis=0)  # shape (E,)
    indexed = [{"link": edge_names[i], "edge_idx": i, "mae": float(link_mae[i])} for i in range(len(link_mae))]
    sorted_by_mae = sorted(indexed, key=lambda x: x["mae"])
    best_5 = sorted_by_mae[:5]
    worst_5 = sorted_by_mae[-5:][::-1]
    all_maes = {i: float(link_mae[i]) for i in range(len(link_mae))}
    return best_5, worst_5, all_maes


def _windows_for_block(util_matrix, starts, block_start, block_end, valid_mask):
    """Build five-history/one-target windows and enforce the split contract."""
    starts = np.asarray(starts, dtype=int)
    if np.any(starts < block_start) or np.any(starts + 5 >= block_end):
        raise ValueError("A forecasting window crosses its chronological block boundary")
    if not all(valid_mask[start:start + 6].all() for start in starts):
        raise ValueError("A forecasting window contains an invalid observation")
    return {"X": np.asarray([util_matrix[start:start + 5] for start in starts], dtype=np.float64),
            "y": np.asarray([util_matrix[start + 5] for start in starts], dtype=np.float64),
            "target_ts": starts + 5}


def _block_statistics(blocks, valid_mask, total_demand, util_matrix):
    statistics = {}
    train_mean = None
    for name, block in blocks.items():
        start, end = block["start_index"], block["end_index"]
        valid = valid_mask[start:end]
        demand = total_demand[start:end][valid].astype(np.float64)
        util = util_matrix[start:end][valid].astype(np.float64)
        if not len(demand):
            raise ValueError(f"No valid observations in {name}")
        item = {"total_demand": {"mean": float(demand.mean()), "median": float(np.median(demand)), "p95": float(np.percentile(demand, 95))},
                "link_utilization": {"mean": float(util.mean()), "p95": float(np.percentile(util, 95))}}
        if name == "train":
            train_mean = item["total_demand"]["mean"]
        else:
            item["total_demand_train_mean_ratio"] = float(item["total_demand"]["mean"] / train_mean)
        statistics[name] = item
    return statistics


def main():
    out_dir = Path("data/processed/geant")
    docs_dir = Path("docs")
    docs_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load split and masks
    with open(out_dir / "split.json") as f:
        split_data = json.load(f)

    blocks = split_data["blocks"]
    valid_mask = np.load(out_dir / "valid_mask.npy")
    total_demand = np.load(out_dir / "total_demand_float64.npy")

    # Topology edges
    # Directed edge order from build_link_series:
    from phase9_geant.parse_geant import parse_topology
    nodes, undirected = parse_topology("data/raw/geant_topology/geant.txt")
    directed_edges = [(u, v) for u, v in undirected] + [(v, u) for u, v in undirected]
    edge_names = [f"{u}->{v}" for u, v in directed_edges]

    # Compute Train 90th percentile of total demand (valid steps only)
    train_slice = slice(blocks["train"]["start_index"], blocks["train"]["end_index"])
    train_valid = valid_mask[train_slice]
    train_totals = total_demand[train_slice][train_valid]
    train_p90_demand = float(np.percentile(train_totals, 90))
    print(f"Train 90th percentile total demand (busy threshold): {train_p90_demand:.2f} Mbps")

    # Load utilization series
    series_dict = {
        "distance": np.load(out_dir / "link_series_utilization.npy"),
        "hop": np.load(out_dir / "link_series_utilization_hop.npy")
    }
    slot_indices = np.load(out_dir / "slot_indices.npy")
    if slot_indices.shape != total_demand.shape:
        raise ValueError("slot_indices and total-demand timestamps are not aligned")
    prediction_artifacts = {}

    # Evaluate for both distance and hop
    all_results = {}

    for series_name, util_matrix in series_dict.items():
        print(f"\n=======================================================")
        print(f"RUNNING BASELINES FOR SERIES: {series_name.upper()}")
        print(f"=======================================================")

        # Prepare window datasets: X: (N_win, 5, E), y: (N_win, E), target_ts: (N_win,)
        # Window start s: history is s..s+4, target is s+5
        block_windows = {}
        for b_name in ["train", "val", "test"]:
            starts = blocks[b_name]["window_starts"]
            block_windows[b_name] = _windows_for_block(
                util_matrix, starts, blocks[b_name]["start_index"], blocks[b_name]["end_index"], valid_mask)

        # Identify BUSY mask on val and test
        for b_name in ["val", "test"]:
            tgt_ts = block_windows[b_name]["target_ts"]
            demands = total_demand[tgt_ts]
            block_windows[b_name]["busy_mask"] = (demands >= train_p90_demand)
            print(f"{b_name} busy count: {np.sum(block_windows[b_name]['busy_mask'])} / {len(tgt_ts)}")

        # -------------------------------------------------------------------
        # 1. Persistence: y_pred = X[:, 4, :] (most recent history step)
        # -------------------------------------------------------------------
        preds_persistence = {
            "val": block_windows["val"]["X"][:, 4, :],
            "test": block_windows["test"]["X"][:, 4, :],
        }

        # -------------------------------------------------------------------
        # 2. EWMA over the 5-step history:
        # weights: w_k = (1 - alpha) * alpha**(4 - k) for k in 0..4 (k=4 is newest)
        # normalized weights: w / sum(w)
        # Choose alpha on VAL from {0.1, 0.2, ..., 0.9}
        # -------------------------------------------------------------------
        best_alpha = None
        best_val_mae = float("inf")
        alphas = [round(a, 1) for a in np.linspace(0.1, 0.9, 9)]
        X_val = block_windows["val"]["X"] # (N, 5, E)
        y_val = block_windows["val"]["y"]

        for a in alphas:
            # k=0 (oldest) to k=4 (newest)
            # Standard EWMA: weights decay into the past: weight of step 4-j is (1-a)^j
            # or with parameter a: weight proportional to (1-a)**(4-k) * a ?
            # Standard formula: w_k = (1 - a)**(4 - k)
            # Let's check both or standard exponential decay.
            # In time-series EWMA: y_hat_t = sum_{k=0..4} w_k * x_k / sum(w_k) where w_k = a * (1-a)**(4-k)
            # If a is the smoothing factor (0 < a <= 1): newest point gets weight a, previous a*(1-a), etc.
            # Let's define weights: w_k = (1 - a)**(4 - k) for k in 0..4
            powers = np.arange(5) # 0, 1, 2, 3, 4
            # weight for index k (k=4 is lag 1, k=0 is lag 5): w[k] = a * (1 - a)**(4 - k)
            w = a * ((1.0 - a) ** (4 - powers))
            w = w / np.sum(w) # normalize
            # X_val has shape (N, 5, E). Tensordot along axis 1:
            pred_a = np.tensordot(X_val, w, axes=([1], [0])) # (N, E)
            mae_a = np.mean(np.abs(y_val - pred_a))
            if mae_a < best_val_mae:
                best_val_mae = mae_a
                best_alpha = a

        print(f"Chosen EWMA alpha on VAL: {best_alpha} (VAL MAE: {best_val_mae:.6f})")

        # Compute EWMA predictions on val and test using best_alpha
        w_best = best_alpha * ((1.0 - best_alpha) ** (4 - np.arange(5)))
        w_best = w_best / np.sum(w_best)

        preds_ewma = {
            "val": np.tensordot(block_windows["val"]["X"], w_best, axes=([1], [0])),
            "test": np.tensordot(block_windows["test"]["X"], w_best, axes=([1], [0])),
        }

        # -------------------------------------------------------------------
        # 3. AR(5): Pooled OLS across links (shared coefficients + intercept)
        # fit on TRAIN windows only
        # Target: y[t+1]
        # Features: [X[:, 4, :], X[:, 3, :], X[:, 2, :], X[:, 1, :], X[:, 0, :], 1]
        # or history order x_0, x_1, x_2, x_3, x_4
        # -------------------------------------------------------------------
        X_tr = block_windows["train"]["X"] # (N_tr, 5, E)
        y_tr = block_windows["train"]["y"] # (N_tr, E)

        # Reshape to pool across all links: (N_tr * E, 5)
        # Let features be [x_{t}, x_{t-1}, x_{t-2}, x_{t-3}, x_{t-4}] (lags 1 to 5)
        # where x_{t} is index 4, x_{t-1} is index 3, ..., x_{t-4} is index 0
        N_tr, _, E = X_tr.shape
        # Transpose to (N_tr, E, 5) then reshape to (N_tr * E, 5)
        X_tr_pooled = np.transpose(X_tr, (0, 2, 1)).reshape(-1, 5)
        # Add intercept column of ones
        X_tr_design = np.column_stack([X_tr_pooled, np.ones(N_tr * E, dtype=np.float64)])
        y_tr_pooled = y_tr.reshape(-1).astype(np.float64)

        # OLS fit
        beta, residuals, rank, s = np.linalg.lstsq(X_tr_design, y_tr_pooled, rcond=None)
        ar_coeffs = beta[:5].tolist() # [beta_lag5, ..., beta_lag1] or [x_0, x_1, x_2, x_3, x_4]
        ar_intercept = float(beta[5])
        print(f"AR(5) coefficients (x_0..x_4): {ar_coeffs}, intercept: {ar_intercept:.6f}")

        # Compute predictions for val and test
        preds_ar = {}
        for b_name in ["val", "test"]:
            X_b = block_windows[b_name]["X"]
            N_b = X_b.shape[0]
            X_b_pooled = np.transpose(X_b, (0, 2, 1)).reshape(-1, 5).astype(np.float64)
            X_b_design = np.column_stack([X_b_pooled, np.ones(N_b * E, dtype=np.float64)])
            y_pred_pooled = X_b_design @ beta
            preds_ar[b_name] = y_pred_pooled.reshape(N_b, E)

        # -------------------------------------------------------------------
        # 4. Seasonal-Naive-96:
        # y[t+1] = y[t+1 - 96]
        # Used only where that earlier step is valid and unflagged.
        # Report coverage and persistence on that same subset.
        # -------------------------------------------------------------------
        sn96_info = {}
        preds_sn96 = {}
        for b_name in ["val", "test"]:
            tgt_ts = block_windows[b_name]["target_ts"]
            earlier_ts = tgt_ts - 96
            # valid if earlier_ts >= 0 and valid_mask[earlier_ts]
            mask_sn96 = (earlier_ts >= 0) & valid_mask[earlier_ts]
            coverage = float(np.mean(mask_sn96))
            
            # Predict
            pred_sn = np.full_like(block_windows[b_name]["y"], np.nan)
            pred_sn[mask_sn96] = util_matrix[earlier_ts[mask_sn96], :]
            preds_sn96[b_name] = (pred_sn, mask_sn96)
            sn96_info[b_name] = {
                "coverage": coverage,
                "valid_count": int(np.sum(mask_sn96)),
                "total_count": len(tgt_ts)
            }
            print(f"Seasonal-Naive-96 {b_name} coverage: {coverage:.4f} ({np.sum(mask_sn96)}/{len(tgt_ts)})")

        # -------------------------------------------------------------------
        # 5. Slot-Mean:
        # Per link and per 15-minute time-of-day slot (0..95),
        # mean utilization computed on TRAIN valid steps only.
        # Predict the slot of the target time.
        # -------------------------------------------------------------------
        # Compute slot mean on TRAIN valid steps:
        train_valid_indices = [t for t in range(blocks["train"]["start_index"], blocks["train"]["end_index"]) if valid_mask[t]]
        slot_link_means = np.zeros((96, E), dtype=np.float32)
        for slot in range(96):
            slot_t = [t for t in train_valid_indices if slot_indices[t] == slot]
            if len(slot_t) > 0:
                slot_link_means[slot, :] = np.mean(util_matrix[slot_t, :], axis=0)
            else:
                raise ValueError(f"No training observations for 15-minute slot {slot}")

        preds_slot = {}
        for b_name in ["val", "test"]:
            tgt_ts = block_windows[b_name]["target_ts"]
            tgt_slots = slot_indices[tgt_ts]
            preds_slot[b_name] = slot_link_means[tgt_slots, :]

        # -------------------------------------------------------------------
        # Compute metrics for all baselines across val and test
        # -------------------------------------------------------------------
        # Precompute 10 links with highest mean utilization on TRAIN
        train_link_means = np.mean(util_matrix[train_valid_indices, :], axis=0)
        top10_link_indices = np.argsort(train_link_means)[-10:][::-1]
        top10_links_info = [{"link": edge_names[i], "edge_idx": int(i), "train_mean_util": float(train_link_means[i])} for i in top10_link_indices]

        series_eval = {
            "ewma_alpha": best_alpha,
            "ar_coeffs": ar_coeffs,
            "ar_intercept": ar_intercept,
            "top10_busiest_links_train": top10_links_info,
            "blocks": {}
        }

        for b_name in ["val", "test"]:
            y_true = block_windows[b_name]["y"]
            busy_m = block_windows[b_name]["busy_mask"]
            mean_tgt = float(np.mean(y_true))
            mean_tgt_busy = float(np.mean(y_true[busy_m]))

            block_res = {
                "mean_target_overall": mean_tgt,
                "mean_target_busy": mean_tgt_busy,
                "baselines": {}
            }

            model_preds = {
                "persistence": preds_persistence[b_name],
                "ewma": preds_ewma[b_name],
                "ar5": preds_ar[b_name],
                "slot_mean": preds_slot[b_name]
            }

            for m_name, y_pred in model_preds.items():
                overall_metrics = compute_metrics(y_true, y_pred, mean_tgt)
                busy_metrics = compute_metrics(y_true[busy_m], y_pred[busy_m], mean_tgt_busy)
                best_5, worst_5, all_maes = compute_per_link_metrics(y_true, y_pred, edge_names)
                
                # Check top 10 busy link persistence
                top10_persistence_mae = None
                if m_name == "persistence":
                    top10_persistence_mae = {edge_names[i]: all_maes[i] for i in top10_link_indices}

                block_res["baselines"][m_name] = {
                    "overall": overall_metrics,
                    "busy": busy_metrics,
                    "per_link_mae": [{"link": edge_names[i], "edge_idx": int(i), "mae": all_maes[i]} for i in range(E)],
                    "best_5_links": best_5,
                    "worst_5_links": worst_5,
                }
                if top10_persistence_mae:
                    block_res["baselines"][m_name]["top10_links_mae"] = top10_persistence_mae

            # Special handling for seasonal-naive-96 on its valid subset
            pred_sn, mask_sn = preds_sn96[b_name]
            y_true_sn = y_true[mask_sn]
            pred_sn_sub = pred_sn[mask_sn]
            mean_tgt_sn = float(np.mean(y_true_sn))

            sn_overall = compute_metrics(y_true_sn, pred_sn_sub, mean_tgt_sn)
            best_5_sn, worst_5_sn, all_maes_sn = compute_per_link_metrics(y_true_sn, pred_sn_sub, edge_names)

            # Persistence on the SAME subset
            pred_pers_sub = preds_persistence[b_name][mask_sn]
            pers_same_sub = compute_metrics(y_true_sn, pred_pers_sub, mean_tgt_sn)

            # Busy on the seasonal naive subset
            busy_sub = busy_m[mask_sn]
            if np.sum(busy_sub) > 0:
                mean_tgt_sn_busy = float(np.mean(y_true_sn[busy_sub]))
                sn_busy = compute_metrics(y_true_sn[busy_sub], pred_sn_sub[busy_sub], mean_tgt_sn_busy)
                pers_same_sub_busy = compute_metrics(y_true_sn[busy_sub], pred_pers_sub[busy_sub], mean_tgt_sn_busy)
            else:
                sn_busy = {}
                pers_same_sub_busy = {}

            block_res["baselines"]["seasonal_naive_96"] = {
                "coverage": sn96_info[b_name]["coverage"],
                "valid_count": sn96_info[b_name]["valid_count"],
                "total_count": sn96_info[b_name]["total_count"],
                "overall": sn_overall,
                "busy": sn_busy,
                "per_link_mae": [{"link": edge_names[i], "edge_idx": int(i), "mae": all_maes_sn[i]} for i in range(E)],
                "best_5_links": best_5_sn,
                "worst_5_links": worst_5_sn,
                "persistence_same_subset": {
                    "overall": pers_same_sub,
                    "busy": pers_same_sub_busy,
                }
            }

            series_eval["blocks"][b_name] = block_res
            prediction_artifacts[f"{series_name}_{b_name}_target_ts"] = block_windows[b_name]["target_ts"]
            prediction_artifacts[f"{series_name}_{b_name}_y"] = y_true
            for model_name, prediction in model_preds.items():
                prediction_artifacts[f"{series_name}_{b_name}_{model_name}"] = prediction
            prediction_artifacts[f"{series_name}_{b_name}_seasonal_mask"] = mask_sn
            prediction_artifacts[f"{series_name}_{b_name}_seasonal_naive_96"] = pred_sn

        all_results[series_name] = series_eval

    all_results["protocol"] = {"primary_series": "distance", "robustness_series": "hop", "history_length": 5,
                               "prediction_horizon": 1, "busy_target_threshold_train_p90_total_demand": train_p90_demand,
                               "block_statistics": _block_statistics(blocks, valid_mask, total_demand, series_dict["distance"]),
                               "seasonal_naive": "Uses t-96 only when that observation is valid; no missing value is replaced with zero."}

    # Save to data/processed/geant/baselines.json
    with open(out_dir / "baselines.json", "w") as f:
        json.dump(all_results, f, indent=2)
    np.savez_compressed(out_dir / "baseline_predictions.npz", **prediction_artifacts)
    print(f"\nSaved baselines evaluation to {out_dir / 'baselines.json'}")

    # Generate docs/geant_baselines.md
    generate_markdown_report(all_results, docs_dir / "geant_baselines.md")
    print(f"Saved report to {docs_dir / 'geant_baselines.md'}")


def generate_markdown_report(results: Dict[str, Any], filepath: Path):
    lines = []
    lines.append("# GEANT Non-Neural Baselines Evaluation\n")
    lines.append("This report concerns **forecasting of computed link utilization from real GEANT traffic matrices**. The distance-shortest-path series is primary; hop-shortest-path results are robustness-only. It does not evaluate real-world QoS or routing improvement.\n")
    protocol = results["protocol"]
    lines.append("## Chronological protocol and distribution check\n")
    lines.append("Each sample has five historical 15-minute utilization steps and one next-step target; every window is valid and wholly within its assigned train, validation, or test block. EWMA alpha is selected by validation MAE only. AR(5) and per-link slot means fit only training observations. Busy targets use raw float64 total demand >= the training p90 threshold.")
    lines.append(f"\n- Train busy-target threshold: `{protocol['busy_target_threshold_train_p90_total_demand']:.6f}` Mbps")
    lines.append("\n| Block | Demand mean | Demand median | Demand p95 | Utilization mean | Utilization p95 | Demand/train mean |")
    lines.append("| :--- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for name, stat in protocol["block_statistics"].items():
        d, u = stat["total_demand"], stat["link_utilization"]
        ratio = stat.get("total_demand_train_mean_ratio", 1.0)
        lines.append(f"| {name} | {d['mean']:.6f} | {d['median']:.6f} | {d['p95']:.6f} | {u['mean']:.6f} | {u['p95']:.6f} | {ratio:.6f} |")
    lines.append("\nValidation and test demand means differ from train if the reported ratios differ from 1.0. This is descriptive evidence of a chronological distribution shift, not a causal or statistically confirmed finding.\n")

    for series_name in ["distance", "hop"]:
        s_data = results[series_name]
        lines.append(f"\n## Series: {series_name.upper()} ROUTING\n")
        lines.append(f"- **Chosen EWMA alpha (val)**: {s_data['ewma_alpha']}")
        lines.append(f"- **AR(5) Coefficients (x_0..x_4)**: `{[round(c, 5) for c in s_data['ar_coeffs']]}`")
        lines.append(f"- **AR(5) Intercept**: `{s_data['ar_intercept']:.6f}`\n")

        for b_name in ["val", "test"]:
            b_data = s_data["blocks"][b_name]
            lines.append(f"### Block: {b_name.upper()}\n")
            lines.append(f"- Mean target utilization (overall): `{b_data['mean_target_overall']:.6f}`")
            lines.append(f"- Mean target utilization (busy targets): `{b_data['mean_target_busy']:.6f}`\n")

            # Table for overall and busy
            lines.append("| Baseline | Overall MAE | Overall RMSE | Overall Norm MAE | Busy MAE | Busy RMSE | Busy Norm MAE |")
            lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")

            for m_name in ["persistence", "ewma", "ar5", "slot_mean"]:
                m = b_data["baselines"][m_name]
                ov = m["overall"]
                bu = m["busy"]
                lines.append(f"| **{m_name}** | {ov['mae']:.6f} | {ov['rmse']:.6f} | {ov['norm_mae']:.4f} | {bu['mae']:.6f} | {bu['rmse']:.6f} | {bu['norm_mae']:.4f} |")

            sn = b_data["baselines"]["seasonal_naive_96"]
            sn_ov = sn["overall"]
            sn_bu = sn["busy"]
            lines.append(f"| **seasonal_naive_96** (cov={sn['coverage']:.2%}) | {sn_ov['mae']:.6f} | {sn_ov['rmse']:.6f} | {sn_ov['norm_mae']:.4f} | {sn_bu['mae']:.6f} | {sn_bu['rmse']:.6f} | {sn_bu['norm_mae']:.4f} |")
            
            p_sub = sn["persistence_same_subset"]
            p_sub_ov = p_sub["overall"]
            p_sub_bu = p_sub["busy"]
            lines.append(f"| *persistence (same subset)* | {p_sub_ov['mae']:.6f} | {p_sub_ov['rmse']:.6f} | {p_sub_ov['norm_mae']:.4f} | {p_sub_bu['mae']:.6f} | {p_sub_bu['rmse']:.6f} | {p_sub_bu['norm_mae']:.4f} |")
            lines.append("")

            lines.append("#### Per-link metrics")
            lines.append("The complete 72-link MAE vector is retained in `data/processed/geant/baselines.json`; best/worst five are listed below.")
            for label, entries in (("Best", b_data["baselines"]["persistence"]["best_5_links"]),
                                   ("Worst", b_data["baselines"]["persistence"]["worst_5_links"])):
                lines.append(f"\n{label} five persistence links: " + ", ".join(f"{entry['link']} ({entry['mae']:.6f})" for entry in entries))
            lines.append("")

            # Top 10 persistence table (val/test)
            if "top10_links_mae" in b_data["baselines"]["persistence"]:
                lines.append("#### Persistence MAE for Top 10 Busiest Links (Train Mean Util)")
                lines.append("| Rank | Link | Train Mean Util | Persistence MAE |")
                lines.append("| :--- | :--- | :--- | :--- |")
                top10_links = s_data["top10_busiest_links_train"]
                top10_mae = b_data["baselines"]["persistence"]["top10_links_mae"]
                for rank, l_info in enumerate(top10_links, 1):
                    lname = l_info["link"]
                    mae_val = top10_mae.get(lname, 0.0)
                    lines.append(f"| {rank} | {lname} | {l_info['train_mean_util']:.6f} | {mae_val:.6f} |")
                lines.append("")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
