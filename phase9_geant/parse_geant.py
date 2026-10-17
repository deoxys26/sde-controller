"""
parse_geant.py — Parse the GÉANT SNDlib topology and demand-matrix files.

Topology source : data/raw/geant_topology/geant.txt
                  SNDlib native format, 22 nodes, 36 undirected links.
Matrices source : data/raw/geant_topology/
                    directed-geant-uhlig-15min-over-4months-ALL-native/
                  One file per 15-min interval, 11 460 files total.
                  Units: MBITPERSEC (stated in META section of every file).

Public API
----------
parse_topology(path)          → nodes: dict[str,(lon,lat)], links: list[(u,v)]
parse_demand_file(path, ...)  → (datetime, demand_matrix: np.ndarray)
parse_all_matrices(folder, nodes_dict)
    → list of (datetime, 22×22 matrix, demand_count, fpath),
      sorted by timestamp.  Raises on any violation.
"""

from __future__ import annotations

import glob
import os
import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

MATRIX_GLOB = "demandMatrix-geant-uhlig-15min-*.txt"
EXPECTED_FILES = 11460
EXPECTED_NODES = 22
EXPECTED_LINKS = 36
CAPACITY_MBPS = 40_000.0          # module capacity, assumed, not measured
CAPACITY_SOURCE = "SNDlib module capacity, assumed, not measured (40 000 Mbps)"


# ---------------------------------------------------------------------------
# Topology parsing
# ---------------------------------------------------------------------------

def parse_topology(path: str) -> tuple[dict, list]:
    """Parse NODES and LINKS from an SNDlib native topology file.

    Returns
    -------
    nodes : dict  {node_id: (longitude, latitude)}
    links : list  [(source, target), ...]  — undirected, one entry per link
    """
    path = str(path)
    with open(path, "r", encoding="utf-8") as fh:
        content = fh.read()
    return parse_topology_text(content, path=path)


def parse_topology_text(content: str, *, path: str = "<memory>") -> tuple[dict, list]:
    """Parse topology text already read from a verified byte snapshot."""
    lines = content.splitlines()

    if not lines or not lines[0].startswith("?SNDlib native format"):
        raise ValueError(
            f"{path}: first line must start with '?SNDlib native format', "
            f"got: {lines[0]!r}"
        )

    nodes: dict[str, tuple[float, float]] = {}
    links: list[tuple[str, str]] = []
    in_nodes = False
    in_links = False

    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("NODES ("):
            in_nodes, in_links = True, False
            continue
        if line.startswith("LINKS ("):
            in_nodes, in_links = False, True
            continue
        # Ignore DEMANDS and beyond
        if re.match(r"(DEMANDS|ADMISSIBLE_PATHS)\s*\(", line):
            in_nodes = in_links = False
            continue
        if line == ")":
            in_nodes = in_links = False
            continue

        if in_nodes:
            # at1.at ( 16.3729 48.2091 )
            parts = line.split()
            node_id = parts[0]
            lon, lat = float(parts[2]), float(parts[3])
            nodes[node_id] = (lon, lat)

        elif in_links:
            # L_at1.at_be1.be ( at1.at be1.be ) 0 0 0 0 ( 40000 1 )
            parts = line.split()
            src, tgt = parts[2], parts[3]
            links.append((src, tgt))

    # ---- Validate topology ----
    if EXPECTED_NODES is not None and len(nodes) != EXPECTED_NODES:
        raise ValueError(
            f"Expected {EXPECTED_NODES} nodes, found {len(nodes)} in {path}"
        )
    if EXPECTED_LINKS is not None and len(links) != EXPECTED_LINKS:
        raise ValueError(
            f"Expected {EXPECTED_LINKS} undirected links, found {len(links)} in {path}"
        )

    # No self-loops
    for u, v in links:
        if u == v:
            raise ValueError(f"Self-loop found: {u} → {v}")

    # No duplicate undirected pairs
    seen = set()
    for u, v in links:
        key = tuple(sorted([u, v]))
        if key in seen:
            raise ValueError(f"Duplicate undirected link: {u} — {v}")
        seen.add(key)

    # All link endpoints must be declared nodes
    for u, v in links:
        for n in (u, v):
            if n not in nodes:
                raise ValueError(f"Link endpoint {n!r} not in NODES section")

    return nodes, links


# ---------------------------------------------------------------------------
# Matrix parsing
# ---------------------------------------------------------------------------

_TS_RE = re.compile(r"demandMatrix-geant-uhlig-15min-(\d{8}-\d{4})\.txt$")


def parse_demand_file(
    path: str,
    nodes_list: list[str],
    node_to_idx: dict[str, int],
) -> tuple[datetime, np.ndarray, int]:
    """Parse one demand-matrix file.

    Returns
    -------
    dt           : datetime of this timestep
    matrix       : (N, N) float64 array, demands in Mbps; absent pairs = 0.0
    demand_count : number of demand entries actually present in the file
    """
    fname = os.path.basename(path)
    ts_m = _TS_RE.search(fname)
    if not ts_m:
        raise ValueError(f"Cannot parse timestamp from filename: {fname!r}")
    fname_ts = ts_m.group(1)

    try:
        with open(path, "r") as fh:
            content = fh.read()
    except OSError as exc:
        raise RuntimeError(
            f"Cloud-only or unreadable file: {path} — {exc}"
        ) from exc

    # META time cross-check
    meta_m = re.search(r"time\s*=\s*([\w-]+)", content)
    if not meta_m:
        raise ValueError(f"No META time= in {fname!r}")
    meta_ts = meta_m.group(1)
    if fname_ts != meta_ts:
        raise ValueError(
            f"Timestamp mismatch in {fname!r}: filename={fname_ts!r}, "
            f"META={meta_ts!r}"
        )

    dt = datetime.strptime(meta_ts, "%Y%m%d-%H%M")
    n = len(nodes_list)
    matrix = np.zeros((n, n), dtype=np.float64)  # float64 accumulation; cast to float32 for tensors later
    demand_count = 0

    in_demands = False
    for raw in content.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("DEMANDS ("):
            in_demands = True
            continue
        if line == ")":
            in_demands = False
            continue
        if line.startswith("ADMISSIBLE_PATHS"):
            break
        if in_demands:
            # D_<id> ( <src> <dst> ) <routing_unit> <value> <max_hops>
            # Use regex for robust extraction
            # line example: at1.at_be1.be ( at1.at be1.be ) 1 20.660211 UNLIMITED
            m = re.search(r'\(\s*(\S+)\s+(\S+)\s*\)\s+(\S+)\s+([0-9.eE+-]+)\s+(\S+)', line)
            if not m:
                continue
            src, dst, routing_unit, val_str, max_path_length = m.groups()
            assert routing_unit == '1'
            assert max_path_length == 'UNLIMITED' or max_path_length.isdigit()
            val = float(val_str)
            matrix[node_to_idx[src], node_to_idx[dst]] = val
            demand_count += 1

    return dt, matrix, demand_count


def parse_all_matrices(
    folder: str,
    nodes_dict: dict[str, tuple[float, float]],
) -> tuple[list, list[str], dict[str, int]]:
    """Parse all 11 460 demand-matrix files in *folder*.

    Returns
    -------
    parsed_data  : list of (datetime, matrix[22×22], demand_count, fpath)
                   sorted strictly by datetime
    nodes_list   : sorted list of node names (canonical order)
    node_to_idx  : {name: index}

    Raises immediately on any anomaly: wrong file count, duplicate timestamps,
    gaps, filename/META mismatch, or cloud-only files.
    """
    files = glob.glob(os.path.join(folder, MATRIX_GLOB))

    if len(files) != EXPECTED_FILES:
        raise ValueError(
            f"Expected {EXPECTED_FILES} matrix files, found {len(files)} "
            f"matching {MATRIX_GLOB!r} in {folder!r}"
        )

    nodes_list = sorted(nodes_dict.keys())
    node_to_idx = {n: i for i, n in enumerate(nodes_list)}

    parsed: list = []
    for fpath in files:
        dt, matrix, count = parse_demand_file(fpath, nodes_list, node_to_idx)
        parsed.append((dt, matrix, count, fpath))

    # Sort by timestamp
    parsed.sort(key=lambda x: x[0])

    # Check monotonicity, duplicates, and 15-min gaps
    seen_ts: dict[datetime, str] = {}
    for i, (dt, _, _, fp) in enumerate(parsed):
        if dt in seen_ts:
            raise ValueError(
                f"Duplicate timestamp {dt}: {seen_ts[dt]!r} and {fp!r}"
            )
        seen_ts[dt] = fp
        if i > 0:
            prev_dt = parsed[i - 1][0]
            diff = dt - prev_dt
            if diff != timedelta(minutes=15):
                raise ValueError(
                    f"Non-15-min step between {prev_dt} and {dt} "
                    f"(diff = {diff})"
                )

    # Check node names appear consistently — validate first non-empty matrix
    # (parsing already enforces this per file via node_to_idx KeyError)

    return parsed, nodes_list, node_to_idx
