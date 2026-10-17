import os
import glob
import math
import pytest
import numpy as np
from phase9_geant.parse_geant import parse_demand_file, parse_topology, EXPECTED_FILES

DATA_DIR = "data/raw/geant_topology/directed-geant-uhlig-15min-over-4months-ALL-native"
TOPO = "data/raw/geant_topology/geant.txt"

def test_golden_20050726_2045():
    nodes, _ = parse_topology(TOPO)
    nodes_list = sorted(nodes.keys())
    node_to_idx = {n: i for i, n in enumerate(nodes_list)}
    
    file_path = os.path.join(DATA_DIR, "demandMatrix-geant-uhlig-15min-20050726-2045.txt")
    _, matrix, count = parse_demand_file(file_path, nodes_list, node_to_idx)
    
    assert count == 424
    assert math.isclose(np.sum(matrix), 52197.2, abs_tol=0.1)
    assert math.isclose(np.max(matrix), 5715.168947, rel_tol=1e-6)
    
    val1 = matrix[node_to_idx['at1.at'], node_to_idx['be1.be']]
    assert math.isclose(val1, 20.660211, rel_tol=1e-6)
    
    val2 = matrix[node_to_idx['uk1.uk'], node_to_idx['se1.se']]
    assert math.isclose(val2, 728.014924, rel_tol=1e-6)

def test_golden_20050726_2100():
    nodes, _ = parse_topology(TOPO)
    nodes_list = sorted(nodes.keys())
    node_to_idx = {n: i for i, n in enumerate(nodes_list)}
    
    file_path = os.path.join(DATA_DIR, "demandMatrix-geant-uhlig-15min-20050726-2100.txt")
    _, matrix, count = parse_demand_file(file_path, nodes_list, node_to_idx)
    
    assert count == 419
    assert math.isclose(np.sum(matrix), 52322.8, abs_tol=0.1)
    assert math.isclose(np.max(matrix), 5569.145847, rel_tol=1e-6)

def test_golden_20050726_2115():
    nodes, _ = parse_topology(TOPO)
    nodes_list = sorted(nodes.keys())
    node_to_idx = {n: i for i, n in enumerate(nodes_list)}
    
    file_path = os.path.join(DATA_DIR, "demandMatrix-geant-uhlig-15min-20050726-2115.txt")
    _, matrix, count = parse_demand_file(file_path, nodes_list, node_to_idx)
    
    assert count == 419
    assert math.isclose(np.sum(matrix), 51763.6, abs_tol=0.1)
    assert math.isclose(np.max(matrix), 5399.759892, rel_tol=1e-6)

def test_all_files_cross_check():
    nodes, _ = parse_topology(TOPO)
    nodes_list = sorted(nodes.keys())
    node_to_idx = {n: i for i, n in enumerate(nodes_list)}
    
    files = glob.glob(os.path.join(DATA_DIR, "demandMatrix-geant-uhlig-15min-*.txt"))
    assert len(files) == EXPECTED_FILES
    
    for fpath in files:
        # main parser
        _, matrix, count = parse_demand_file(fpath, nodes_list, node_to_idx)
        parser_total = float(np.sum(matrix))
        
        # simple implementation
        simple_total = 0.0
        in_demands = False
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if line.startswith("DEMANDS"):
                    in_demands = True
                    continue
                if in_demands and line == ")":
                    break
                if in_demands and line and not line.startswith("#"):
                    parts = line.split()
                    if len(parts) >= 2:
                        simple_total += float(parts[-2])
        
        assert math.isclose(parser_total, simple_total, rel_tol=1e-5), f"Mismatch in {fpath}: {parser_total} != {simple_total}"
