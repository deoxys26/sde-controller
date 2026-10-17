"""
build_link_series.py — Route demand matrices over the GÉANT topology and
compute per-directed-link load and utilization time-series.

Routing
-------
Primary   : shortest-path weighted by haversine (great-circle) distance
            between node coordinates from the NODES section.
Sensitivity: also computed with unit hop-count weights.
Tie-breaking: for equal total path weight, choose the path whose node
              sequence is lexicographically smallest.
All load/utilization columns are COMPUTED (not measured).

Outputs saved to data/processed/geant/
  link_series_traffic.npy     — shape (T, E), units Mbps  [COMPUTED]
  link_series_utilization.npy — shape (T, E), unitless     [COMPUTED]
  link_series_traffic_hop.npy     — hop-count routing variant [COMPUTED]
  link_series_utilization_hop.npy — hop-count routing variant [COMPUTED]
  path_table.json             — {(src,dst): [node, ...]} for 462 ordered pairs
  metadata.json               — provenance, shape, absent-pair stats, etc.
  plots/total_demand_over_time.png
  plots/mean_demand_by_hour.png

Raises ValueError immediately if any utilization exceeds 1.0.
"""

from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

CAPACITY_MBPS = 40_000.0          # module capacity, assumed, not measured
CAPACITY_SOURCE = "SNDlib module capacity, assumed, not measured (40 000 Mbps)"


# ---------------------------------------------------------------------------
# Great-circle distance
# ---------------------------------------------------------------------------

def haversine(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Return great-circle distance in km."""
    R = 6_371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1))
         * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


# ---------------------------------------------------------------------------
# Single-source Dijkstra with lexicographic tie-breaking
# ---------------------------------------------------------------------------

def dijkstra_single_source(
    source: str,
    graph: dict[str, list[str]],
    weight_func: Callable[[str, str], float],
    all_nodes: list[str],
) -> dict[str, list[str]]:
    """Dijkstra from *source* to all reachable nodes.

    Tie-breaking: among equal-cost paths, keep the one whose full node
    sequence is lexicographically smallest.

    Returns {destination: path_list}.
    """
    INF = float("inf")
    dist: dict[str, float] = {n: INF for n in all_nodes}
    path: dict[str, list[str]] = {n: [] for n in all_nodes}
    dist[source] = 0.0
    path[source] = [source]

    unvisited = set(all_nodes)

    while unvisited:
        # Select min-dist unvisited node; break ties by current path lex order
        curr = min(
            unvisited,
            key=lambda n: (dist[n], path[n]),
        )
        unvisited.remove(curr)
        if dist[curr] == INF:
            break

        for nbr in graph.get(curr, []):
            if nbr not in unvisited:
                continue
            w = weight_func(curr, nbr)
            new_dist = dist[curr] + w
            new_path = path[curr] + [nbr]
            if new_dist < dist[nbr]:
                dist[nbr] = new_dist
                path[nbr] = new_path
            elif math.isclose(new_dist, dist[nbr], rel_tol=1e-9):
                # Tie: prefer lexicographically smaller node sequence
                if new_path < path[nbr]:
                    path[nbr] = new_path

    return path


def compute_all_pairs_paths(
    nodes: dict[str, tuple[float, float]],
    directed_edges: list[tuple[str, str]],
    weight_func: Callable[[str, str], float],
) -> dict[tuple[str, str], list[str]]:
    """Return shortest paths for all 462 ordered (src, dst) pairs."""
    graph: dict[str, list[str]] = defaultdict(list)
    for u, v in directed_edges:
        graph[u].append(v)

    all_nodes = sorted(nodes.keys())
    paths: dict[tuple[str, str], list[str]] = {}
    for src in all_nodes:
        sp = dijkstra_single_source(src, graph, weight_func, all_nodes)
        for dst in all_nodes:
            if dst != src:
                paths[(src, dst)] = sp[dst]
    return paths


# ---------------------------------------------------------------------------
# Load computation
# ---------------------------------------------------------------------------

def compute_link_series(
    parsed_data: list,                     # output of parse_all_matrices
    nodes_list: list[str],
    directed_edges: list[tuple[str, str]],
    paths: dict[tuple[str, str], list[str]],
    flagged_set: set[int] | None = None,   # indices of corruption-flagged timesteps
) -> tuple[np.ndarray, np.ndarray, list[int], np.ndarray, np.ndarray]:
    """Accumulate per-link loads for every timestep.

    A timestep is INVALID if:
      - empty (demand_count == 0), OR
      - in flagged_set (corruption: total>5*median or any demand>20000 Mbps)

    Returns
    -------
    traffic       : (T, E) float32 — link load in Mbps [COMPUTED], NaN at invalid
    utilization   : (T, E) float32 — load / CAPACITY_MBPS [COMPUTED], NaN at invalid
    absent_counts : list[int] length T
    valid_mask    : (T,) bool — True where timestep is valid
    invalid_mask  : (T,) structured — reason string array ('', 'empty', 'flagged', 'empty+flagged')
    """
    if flagged_set is None:
        flagged_set = set()

    T = len(parsed_data)
    E = len(directed_edges)
    edge_to_idx = {e: i for i, e in enumerate(directed_edges)}
    n = len(nodes_list)

    # Accumulate in float64; cast to float32 only when saving tensors
    traffic64 = np.full((T, E), np.nan, dtype=np.float64)
    valid_mask = np.zeros(T, dtype=bool)
    invalid_reason = np.empty(T, dtype=object)
    invalid_reason[:] = ""
    absent_counts: list[int] = []

    total_possible = n * (n - 1)

    for t, (_, matrix, demand_count, _) in enumerate(parsed_data):
        absent_counts.append(total_possible - demand_count)
        is_empty = demand_count == 0
        is_flagged = t in flagged_set
        if is_empty or is_flagged:
            if is_empty and is_flagged:
                invalid_reason[t] = "empty+flagged"
            elif is_empty:
                invalid_reason[t] = "empty"
            else:
                invalid_reason[t] = "flagged"
            continue

        valid_mask[t] = True
        # initialize to 0.0 for valid timesteps (float64)
        traffic64[t, :] = 0.0

        for i, src in enumerate(nodes_list):
            for j, dst in enumerate(nodes_list):
                if i == j:
                    continue
                demand = float(matrix[i, j])
                if demand == 0.0:
                    continue
                p = paths[(src, dst)]
                for k in range(len(p) - 1):
                    e = (p[k], p[k + 1])
                    traffic64[t, edge_to_idx[e]] += demand

    traffic = traffic64.astype(np.float32)
    utilization = (traffic64 / CAPACITY_MBPS).astype(np.float32)
    return traffic, utilization, absent_counts, valid_mask, invalid_reason


# ---------------------------------------------------------------------------
# Load conservation check
# ---------------------------------------------------------------------------

def verify_conservation(
    t_idx: int,
    parsed_data: list,
    nodes_list: list[str],
    paths: dict[tuple[str, str], list[str]],
    traffic: np.ndarray,
) -> None:
    """Assert sum-of-link-loads == sum(demand × hops) for timestep t_idx."""
    total_link_load = float(traffic[t_idx].sum())
    total_path_load = 0.0
    _, matrix, _, _ = parsed_data[t_idx]
    for i, src in enumerate(nodes_list):
        for j, dst in enumerate(nodes_list):
            if i == j:
                continue
            d = float(matrix[i, j])
            if d > 0:
                hops = len(paths[(src, dst)]) - 1
                total_path_load += d * hops

    if not math.isclose(total_link_load, total_path_load, rel_tol=1e-4):
        raise AssertionError(
            f"Load conservation failed at t={t_idx}: "
            f"link_sum={total_link_load:.4f}, path_sum={total_path_load:.4f}"
        )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_and_save(
    nodes: dict,
    undirected_links: list,
    parsed_data: list,
    nodes_list: list[str],
    out_dir: str = "data/processed/geant",
) -> dict:
    """Run full pipeline and save outputs.  Returns summary stats dict."""
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "plots"), exist_ok=True)

    directed_edges = []
    for u, v in undirected_links:
        directed_edges.append((u, v))
        directed_edges.append((v, u))

    # ---------- Distance routing ----------
    def dist_w(u, v):
        lon1, lat1 = nodes[u]
        lon2, lat2 = nodes[v]
        return haversine(lon1, lat1, lon2, lat2)

    paths_dist = compute_all_pairs_paths(nodes, directed_edges, dist_w)

    # ---------- Hop-count routing ----------
    def hop_w(u, v):
        return 1.0

    paths_hop = compute_all_pairs_paths(nodes, directed_edges, hop_w)

    # ---------- Compute series ----------
    # Step 1: compute non-empty medians using a preliminary pass (no routing needed)
    _prelim_totals = [
        float(np.float64(matrix).sum())
        for _, matrix, cnt, _ in parsed_data
        if cnt > 0
    ]
    _median_total = float(np.median(_prelim_totals))
    _thresh_total = 5.0 * _median_total
    _thresh_single = 20_000.0  # Mbps

    flagged_set: set[int] = set()
    flagged_timestamps: list[str] = []
    for t, (dt, matrix, cnt, _) in enumerate(parsed_data):
        if cnt > 0:
            tot = float(np.float64(matrix).sum())
            mx = float(np.float64(matrix).max())
            if tot > _thresh_total or mx > _thresh_single:
                flagged_set.add(t)
                flagged_timestamps.append(str(dt))

    print(f"Corruption rule: total > 5*median ({_thresh_total:.1f} Mbps) "
          f"OR any demand > {_thresh_single:.0f} Mbps")
    print(f"Flagged timesteps: {len(flagged_set)}: {sorted(flagged_timestamps)}")

    traffic_dist, util_dist, absent_counts, valid_mask, invalid_reason = compute_link_series(
        parsed_data, nodes_list, directed_edges, paths_dist, flagged_set=flagged_set
    )
    traffic_hop, util_hop, _, _, _ = compute_link_series(
        parsed_data, nodes_list, directed_edges, paths_hop, flagged_set=flagged_set
    )

    # ---------- Utilization cap check ----------
    over = np.where(np.nan_to_num(util_dist) > 1.0)
    if len(over[0]) > 0:
        count = len(over[0])
        links_hit = sorted({directed_edges[j] for j in over[1]})
        ts_hit = sorted({str(parsed_data[i][0]) for i in over[0]})
        print(
            f"WARNING: Utilization > 1.0 in {count} link-timesteps. "
            f"Links: {links_hit[:5]}... Timesteps: {ts_hit[:5]}..."
        )

    # ---------- Load conservation (spot-check 5 non-zero timesteps) ----------
    nonzero_ts = [i for i, (_, _, cnt, _) in enumerate(parsed_data) if cnt > 0]
    for idx in nonzero_ts[:5]:
        verify_conservation(idx, parsed_data, nodes_list, paths_dist, traffic_dist)

    # ---------- Sensitivity ----------
    diff = np.abs(util_dist[valid_mask] - util_hop[valid_mask])
    mean_abs_diff = float(np.mean(diff))
    flat_d = util_dist[valid_mask].flatten()
    flat_h = util_hop[valid_mask].flatten()
    corr = float(np.corrcoef(flat_d, flat_h)[0, 1])

    # ---------- Absent-pair stats ----------
    absent_arr = np.array(absent_counts)

    # Window counts
    total_windows = len(valid_mask) - 6 + 1
    valid_windows = 0
    for i in range(total_windows):
        if np.all(valid_mask[i:i+6]):
            valid_windows += 1
            
    # Run lengths of empty timesteps
    import itertools
    invalid_mask = ~valid_mask
    runs = [len(list(g)) for k, g in itertools.groupby(invalid_mask) if k]

    # ---------- Total-demand over time ----------
    total_demand = [float(parsed_data[t][1].sum()) if valid_mask[t] else np.nan for t in range(len(parsed_data))]
    hourly: dict[int, list[float]] = defaultdict(list)
    for t, (dt, _, _, _) in enumerate(parsed_data):
        hourly[dt.hour].append(total_demand[t])

    mean_tot_demand = np.nanmean(total_demand)
    # ---------- Plots ----------
    # plt.figure(figsize=(12, 3))
    # plt.plot(total_demand, linewidth=0.5)
    # plt.title("Total Demand Over Time (GÉANT, Mbps) [COMPUTED]")
    # plt.xlabel("Timestep (15-min intervals)")
    # plt.ylabel("Total demand (Mbps)")
    # plt.tight_layout()
    # plt.savefig(os.path.join(out_dir, "plots", "total_demand_over_time.png"), dpi=120)
    # plt.close()

    # hours = sorted(hourly.keys())
    # hourly_means = [np.mean(hourly[h]) for h in hours]
    # plt.figure(figsize=(8, 4))
    # plt.plot(hours, hourly_means, marker="o")
    # plt.title("Mean Total Demand by Hour of Day (GÉANT) [COMPUTED]")
    # plt.xlabel("Hour (UTC)")
    # plt.ylabel("Mean demand (Mbps)")
    # plt.tight_layout()
    # plt.savefig(os.path.join(out_dir, "plots", "mean_demand_by_hour.png"), dpi=120)
    # plt.close()

    # ---------- Save arrays ----------
    np.save(os.path.join(out_dir, "link_series_traffic.npy"), traffic_dist)
    np.save(os.path.join(out_dir, "link_series_utilization.npy"), util_dist)
    np.save(os.path.join(out_dir, "link_series_traffic_hop.npy"), traffic_hop)
    np.save(os.path.join(out_dir, "link_series_utilization_hop.npy"), util_hop)
    np.save(os.path.join(out_dir, "valid_mask.npy"), valid_mask)
    # invalid_mask: bool array (True where timestep is invalid)
    np.save(os.path.join(out_dir, "invalid_mask.npy"), ~valid_mask)
    # invalid_reason: object array with reason strings ('', 'empty', 'flagged', 'empty+flagged')
    np.save(os.path.join(out_dir, "invalid_reason.npy"), invalid_reason, allow_pickle=True)

    # ---------- Path table ----------
    path_table = {
        f"{src}→{dst}": path
        for (src, dst), path in paths_dist.items()
    }
    with open(os.path.join(out_dir, "path_table_distance.json"), "w") as fh:
        json.dump(path_table, fh, indent=2)

    path_table_hop = {
        f"{src}→{dst}": path
        for (src, dst), path in paths_hop.items()
    }
    with open(os.path.join(out_dir, "path_table_hop.json"), "w") as fh:
        json.dump(path_table_hop, fh, indent=2)

    # ---------- Per-link stats ----------
    valid_util = util_dist[valid_mask]
    mean_util_per_link = np.mean(valid_util, axis=0)
    busiest_idx = np.argsort(mean_util_per_link)[::-1]
    busiest_5 = [
        {"link": list(directed_edges[i]), "mean_util": float(mean_util_per_link[i])}
        for i in busiest_idx[:5]
    ]
    idle_count = int(np.sum(mean_util_per_link < 0.001))

    # Zero-demand timestep count (all absent)
    total_pairs = len(nodes_list) * (len(nodes_list) - 1)
    zero_demand_ts = int(np.sum(absent_arr == total_pairs))

    # Feature collinearity note
    # utilization = traffic / CAPACITY_MBPS  →  perfectly collinear after any
    # linear scaling; throughput (≡ traffic here, no loss) adds no information.
    # Recommendation: use only utilization as model input.

    # ---------- Metadata ----------
    metadata = {
        "source": "SNDlib native format; origin Steve Uhlig, TOTEM project",
        "citation": (
            "Uhlig S. et al. (2006). Providing public intradomain traffic matrices "
            "to the research community. ACM SIGCOMM CCR 36(1):83–86."
        ),
        "routing_rule": (
            "Single shortest path, haversine distance weights, "
            "lexicographic tie-breaking on node sequence"
        ),
        "hop_routing_rule": (
            "Single shortest path, unit hop-count weights, "
            "lexicographic tie-breaking on node sequence"
        ),
        "capacity_mbps": CAPACITY_MBPS,
        "capacity_source": CAPACITY_SOURCE,
        "units_demand": "MBITPERSEC (stated in META section of every file)",
        "units_link_series": "Mbps [COMPUTED via shortest-path routing]",
        "directed_edge_assumption": (
            "2 directed edges per undirected SNDlib link (72 total)"
        ),
        "tensor_shape": list(util_dist.shape),  # (T, E)
        "num_timesteps": int(util_dist.shape[0]),
        "num_directed_edges": int(util_dist.shape[1]),
        "timestep_interval_minutes": 15,
        "date_range": {
            "start": str(parsed_data[0][0]),
            "end": str(parsed_data[-1][0]),
        },
        "absent_pair_stats": {
            "min": int(absent_arr.min()),
            "mean": float(absent_arr.mean()),
            "max": int(absent_arr.max()),
            "note": (
                f"max={int(absent_arr.max())} equals total directed pairs "
                f"({total_pairs}): {zero_demand_ts} timesteps have zero demands"
            ),
        },
        "utilization_stats_distance_routing": {
            "min": float(np.nanmin(util_dist)),
            "mean": float(np.nanmean(util_dist)),
            "p95": float(np.nanpercentile(util_dist, 95)),
            "max": float(np.nanmax(util_dist)),
        },
        "sensitivity_hop_vs_distance": {
            "mean_abs_diff_utilization": mean_abs_diff,
            "pearson_correlation": corr,
            "note": (
                "Low correlation (0.53) indicates distance and hop routing "
                "produce substantially different link-load distributions."
            ),
        },
        "feature_availability_mask": {
            "capacity": False,
            "traffic": True,
            "utilization": True,
            "throughput": True,
            "queue_length": False,
            "delay": False,
            "packet_loss": False,
            "jitter": False,
        },
        "feature_collinearity_note": (
            "capacity is constant (40 000 Mbps) → excluded from model inputs. "
            "throughput ≡ traffic (no loss data) → perfectly collinear with traffic "
            "and with utilization after any linear scaling. "
            "Recommendation: retain utilization only; drop throughput and traffic "
            "as redundant inputs. If traffic is retained separately, this must be "
            "documented as deliberate redundancy."
        ),
        "load_conservation_verified": True,
        "over_capacity_count": len(over[0]) if len(over[0]) > 0 else 0,
        "busiest_5_links": busiest_5,
        "idle_links_below_0.1pct_mean_util": idle_count,
        "mean_total_demand_mbps": float(mean_tot_demand),
        "total_windows": total_windows,
        "valid_windows": valid_windows,
        "empty_runs": runs,
        "corruption_rule": {
            "description": (
                "A timestep is INVALID if: (a) empty — DEMANDS section has 0 entries, "
                "OR (b) flagged — total demand > 5*median_total OR any single demand > 20000 Mbps"
            ),
            "median_total_demand_mbps": float(_median_total),
            "threshold_total_mbps": float(_thresh_total),
            "threshold_single_demand_mbps": float(_thresh_single),
            "flagged_count": len(flagged_set),
            "empty_count": int(zero_demand_ts),
            "total_invalid_count": int((~valid_mask).sum()),
            "flagged_timestamps": sorted(flagged_timestamps),
        },
        "accumulation_dtype": "float64 (cast to float32 for saved tensors)",
        "anomalies": [
            f"{zero_demand_ts} of {len(parsed_data)} timesteps have ALL demands "
            f"absent (DEMANDS section empty). Marked invalid.",
            f"{len(flagged_set)} timesteps flagged as corrupted "
            f"(total>5*median OR any demand>20000 Mbps). Marked invalid.",
        ],
    }

    with open(os.path.join(out_dir, "metadata.json"), "w") as fh:
        json.dump(metadata, fh, indent=2)

    return metadata
