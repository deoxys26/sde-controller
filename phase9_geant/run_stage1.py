#!/usr/bin/env python3
"""
Phase 9 Stage 1 — Entry point.

Parses GEANT topology + demand matrices, routes traffic, builds link-series
tensors, and saves all outputs to data/processed/geant/.

Usage (from project root):
    python phase9_geant/run_stage1.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOPO = ROOT / "data/raw/geant_topology/geant.txt"
DATA_DIR = (
    ROOT
    / "data/raw/geant_topology"
    / "directed-geant-uhlig-15min-over-4months-ALL-native"
)
OUT_DIR = ROOT / "data/processed/geant"


def main():
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    from phase9_geant.parse_geant import parse_topology, parse_all_matrices
    from phase9_geant.build_link_series import build_and_save

    print("=== Phase 9 Stage 1: GEANT data pipeline ===")

    print("Parsing topology …")
    nodes, undirected_links = parse_topology(str(TOPO))
    print(f"  Nodes: {len(nodes)}, Undirected links: {len(undirected_links)}")

    print("Parsing demand matrices …")
    parsed_data, nodes_list, _ = parse_all_matrices(str(DATA_DIR), nodes)
    print(f"  Parsed {len(parsed_data)} matrices.")

    print("Building link series (distance routing + hop routing) …")
    meta = build_and_save(
        nodes, undirected_links, parsed_data, nodes_list, str(OUT_DIR)
    )

    print("\n=== SUMMARY ===")
    print(f"Tensor shape : {meta['tensor_shape']}  (timesteps × directed edges)")
    s = meta["utilization_stats_distance_routing"]
    print(
        f"Utilization  : min={s['min']:.6f}  mean={s['mean']:.6f}"
        f"  p95={s['p95']:.6f}  max={s['max']:.6f}"
    )
    sens = meta["sensitivity_hop_vs_distance"]
    print(
        f"Sensitivity  : mean|delta_util|={sens['mean_abs_diff_utilization']:.6f}"
        f"  corr={sens['pearson_correlation']:.4f}"
    )
    ap = meta["absent_pair_stats"]
    print(f"Absent pairs : min={ap['min']}  mean={ap['mean']:.1f}  max={ap['max']}")
    for note in meta["anomalies"]:
        print(f"[ANOMALY] {note}")
    print(f"\nOutputs in: {OUT_DIR}")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()
