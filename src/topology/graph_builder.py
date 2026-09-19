import networkx as nx
import random

def build_topology(num_nodes=15, topology_type="barabasi_albert", seed=42, default_capacity_mbps=1000.0, base_delay_ms=2.0):
    """
    Builds a deterministic, reproducible network topology graph.
    """
    random.seed(seed)
    
    if topology_type == "barabasi_albert":
        # Scale-free graph, commonly used to model internet/router topologies
        # m is the number of edges to attach from a new node to existing nodes
        m = 2 if num_nodes > 2 else 1
        G = nx.barabasi_albert_graph(n=num_nodes, m=m, seed=seed)
    elif topology_type == "mesh":
        # Fully connected mesh
        G = nx.complete_graph(n=num_nodes)
    else:
        # Default to a simple connected path if unknown
        G = nx.path_graph(n=num_nodes)
    
    # Ensure it's connected (BA and complete graph usually are, but just in case)
    if not nx.is_connected(G):
        raise ValueError(f"Generated {topology_type} topology is not connected!")

    # Assign node and edge attributes
    for idx, node in enumerate(G.nodes()):
        G.nodes[node]['id'] = f"S{node}"
        G.nodes[node]['type'] = 'switch'
        
    for u, v in G.edges():
        # Edge IDs
        edge_id = f"E_{u}_{v}"
        
        # Add slight deterministic variation based on the edge endpoints to prevent uniform identical edges
        cap_variation = random.uniform(0.8, 1.2)
        delay_variation = random.uniform(0.8, 1.2)
        
        # We model directed behavior in routing, but base physical link is undirected here.
        # It's better to convert to MultiDiGraph or DiGraph for SDN routing to model asymmetric links.
        # But NetworkX standard generators produce undirected graphs. We will convert to DiGraph.
        pass

    # Convert to DiGraph so we have bidirectional asymmetric links if needed later
    DG = nx.DiGraph(G)
    
    for u, v in DG.edges():
        DG[u][v]['edge_id'] = f"E_{u}_{v}"
        # Small deterministic random offsets so links aren't 100% identical
        cap = default_capacity_mbps * (0.9 + 0.2 * random.random())
        delay = base_delay_ms * (0.9 + 0.2 * random.random())
        
        DG[u][v]['capacity_mbps'] = round(cap, 2)
        DG[u][v]['base_delay_ms'] = round(delay, 2)
        DG[u][v]['packet_loss_rate'] = 0.0 # Base physical loss
        DG[u][v]['active'] = True
        
    return DG
