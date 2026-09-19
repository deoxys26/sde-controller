import networkx as nx
import json
import os

def save_topology(graph, filepath):
    """
    Saves a NetworkX graph to a JSON file using node-link format.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    data = nx.node_link_data(graph)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)

def load_topology(filepath):
    """
    Loads a NetworkX graph from a JSON file and validates it.
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
        
    graph = nx.node_link_graph(data)
    
    # Validation
    if len(graph.nodes) == 0:
        raise ValueError("Graph is empty.")
    
    # NetworkX is_connected only works for undirected graphs. 
    # For DiGraph, we check strong or weak connectivity. Weak connectivity is usually enough 
    # to show that the underlying physical topology is connected.
    if graph.is_directed():
        if not nx.is_weakly_connected(graph):
            raise ValueError("Graph is not weakly connected.")
    else:
        if not nx.is_connected(graph):
            raise ValueError("Graph is not connected.")
            
    # Validate required edge attributes
    for u, v, data in graph.edges(data=True):
        required = ['capacity_mbps', 'base_delay_ms', 'packet_loss_rate', 'active']
        for req in required:
            if req not in data:
                raise ValueError(f"Edge ({u}, {v}) is missing required attribute: {req}")
                
    return graph
