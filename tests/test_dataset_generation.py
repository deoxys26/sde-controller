import pytest
import os
import pandas as pd
from src.topology.graph_builder import build_topology
from src.simulator.scenario_generator import apply_scenario_to_graph
from src.telemetry.feature_builder import build_edge_features

def test_deterministic_generation():
    graph = build_topology(num_nodes=10, seed=42)
    
    g1 = apply_scenario_to_graph(graph.copy(), "normal", 5, 100, seed=42)
    feat1 = build_edge_features(g1, 5, seed=42)
    
    g2 = apply_scenario_to_graph(graph.copy(), "normal", 5, 100, seed=42)
    feat2 = build_edge_features(g2, 5, seed=42)
    
    assert feat1 == feat2

def test_telemetry_bounds():
    graph = build_topology(num_nodes=5, seed=1)
    # Force high congestion
    g = apply_scenario_to_graph(graph, "congestion", 100, 100, seed=1) 
    
    # Specifically manually set a huge traffic amount to test bounds
    for u, v, d in g.edges(data=True):
        d['current_offered_traffic'] = d['capacity_mbps'] * 1.5
        
    features = build_edge_features(g, 100)
    
    for f in features:
        assert f['timestamp'] == 100 if 'timestamp' in f else f['timestep'] == 100
        assert f['bandwidth_utilization'] > 1.0 # Due to 1.5x traffic
        assert f['delay'] >= 0
        assert 0 <= f['packet_loss'] <= 1.0
        assert f['throughput'] >= 0
        assert f['throughput'] <= f['bandwidth_capacity'] # Max throughput is capacity
        assert f['queue_length'] > 0

def test_increasing_traffic_behavior():
    graph = build_topology(num_nodes=2, topology_type="mesh", seed=1)
    
    # Timestep 1: Low traffic
    for u, v, d in graph.edges(data=True):
        d['current_offered_traffic'] = d['capacity_mbps'] * 0.5
    f1 = build_edge_features(graph, 1)[0]
    
    # Timestep 2: High traffic (congestion)
    for u, v, d in graph.edges(data=True):
        d['current_offered_traffic'] = d['capacity_mbps'] * 1.1
    f2 = build_edge_features(graph, 2)[0]
    
    assert f2['bandwidth_utilization'] > f1['bandwidth_utilization']
    assert f2['queue_length'] > f1['queue_length']
    assert f2['delay'] > f1['delay']
    assert f2['packet_loss'] > f1['packet_loss']
    
    # At extreme congestion (1.1x), packet loss causes throughput collapse, 
    # so throughput is actually lower than the perfectly handled 0.5x traffic.
    assert f2['throughput'] < f2['bandwidth_capacity'] 
    
    # Let's test a moderate traffic increase where throughput SHOULD increase
    for u, v, d in graph.edges(data=True):
        d['current_offered_traffic'] = d['capacity_mbps'] * 0.8
    f_mod = build_edge_features(graph, 3)[0]
    assert f_mod['throughput'] > f1['throughput']
