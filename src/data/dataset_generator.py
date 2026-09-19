import os
import yaml
import pandas as pd

from src.topology.graph_builder import build_topology
from src.topology.topology_loader import save_topology
from src.simulator.scenario_generator import apply_scenario_to_graph
from src.telemetry.feature_builder import build_edge_features, build_node_features

def load_config(config_path="configs/simulation.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def generate_dataset(config_path="configs/simulation.yaml"):
    config = load_config(config_path)
    
    seed = config.get('seed', 42)
    num_nodes = config.get('num_nodes', 15)
    topology_type = config.get('topology_type', 'barabasi_albert')
    timesteps = config.get('simulation_timesteps', 100)
    scenario = config.get('scenario', 'normal')
    output_dir = config.get('output_dir', 'data/generated')
    
    # 1. Create Output Dirs
    dirs = ['topology', 'telemetry', 'traffic', 'scenarios']
    for d in dirs:
        os.makedirs(os.path.join(output_dir, d), exist_ok=True)
        
    # 2. Build Topology
    graph = build_topology(
        num_nodes=num_nodes,
        topology_type=topology_type,
        seed=seed,
        default_capacity_mbps=config.get('default_capacity_mbps', 1000.0),
        base_delay_ms=config.get('base_delay_ms', 2.0)
    )
    
    # Save Topology
    save_topology(graph, os.path.join(output_dir, 'topology', 'base_topology.json'))
    
    # 3. Simulate Timesteps
    all_edge_telemetry = []
    all_node_telemetry = []
    
    for t in range(timesteps):
        # Apply traffic scenario
        graph = apply_scenario_to_graph(graph, scenario, t, timesteps, seed)
        
        # Build features
        edge_telem = build_edge_features(graph, t, seed)
        node_telem = build_node_features(edge_telem, graph.nodes())
        
        all_edge_telemetry.extend(edge_telem)
        all_node_telemetry.extend(node_telem)
        
    # 4. Save Telemetry
    df_edge = pd.DataFrame(all_edge_telemetry)
    df_node = pd.DataFrame(all_node_telemetry)
    
    df_edge.to_csv(os.path.join(output_dir, 'telemetry', f'{scenario}_edge_telemetry.csv'), index=False)
    df_node.to_csv(os.path.join(output_dir, 'telemetry', f'{scenario}_node_telemetry.csv'), index=False)
    
    print(f"Generated dataset for scenario '{scenario}' across {timesteps} timesteps.")
    print(f"Edge telemetry shape: {df_edge.shape}")
    print(f"Node telemetry shape: {df_node.shape}")
    
    return df_edge, df_node, graph

if __name__ == "__main__":
    generate_dataset()
