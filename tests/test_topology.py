import os
import pytest
import networkx as nx
from src.topology.graph_builder import build_topology
from src.topology.topology_loader import save_topology, load_topology

def test_deterministic_topology():
    g1 = build_topology(num_nodes=10, seed=42)
    g2 = build_topology(num_nodes=10, seed=42)
    
    assert list(g1.edges()) == list(g2.edges())
    
    # Check random attributes are identical
    u, v = list(g1.edges())[0]
    assert g1[u][v]['capacity_mbps'] == g2[u][v]['capacity_mbps']

def test_graph_properties():
    g = build_topology(num_nodes=15, topology_type="barabasi_albert")
    
    # Must be DiGraph to support asymmetric directional routing
    assert isinstance(g, nx.DiGraph)
    assert nx.is_weakly_connected(g)
    assert len(g.nodes()) == 15
    
    # Required edge attributes
    for u, v, data in g.edges(data=True):
        assert 'capacity_mbps' in data
        assert 'base_delay_ms' in data
        assert 'packet_loss_rate' in data
        assert 'active' in data

def test_save_load_topology(tmp_path):
    g1 = build_topology(num_nodes=5, seed=10)
    filepath = tmp_path / "test_topo.json"
    
    save_topology(g1, filepath)
    g2 = load_topology(filepath)
    
    assert len(g1.nodes()) == len(g2.nodes())
    assert len(g1.edges()) == len(g2.edges())
    
    # Check attributes persisted
    u, v = list(g1.edges())[0]
    assert g1[u][v]['capacity_mbps'] == g2[u][v]['capacity_mbps']
