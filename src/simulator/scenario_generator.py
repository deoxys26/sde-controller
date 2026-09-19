import random

def generate_scenario_multiplier(scenario_type, timestep, max_timesteps, seed=42):
    """
    Returns a traffic multiplier for the given timestep based on the scenario.
    """
    random.seed(seed + timestep) # Deterministic but varies by timestep
    
    if scenario_type == "normal":
        return random.uniform(0.4, 0.6)
    elif scenario_type == "low_load":
        return random.uniform(0.1, 0.3)
    elif scenario_type == "medium_load":
        return random.uniform(0.4, 0.6)
    elif scenario_type == "high_load":
        return random.uniform(0.7, 0.9)
    elif scenario_type in {"increasing_load", "congestion"}:
        progress = timestep / max_timesteps
        base = 0.4 + (0.8 * progress) # Goes from 0.4 to 1.2
        return base + random.uniform(-0.05, 0.05)
    elif scenario_type == "traffic_spike":
        if 0.4 * max_timesteps < timestep < 0.6 * max_timesteps:
            return random.uniform(1.0, 1.5)
        return random.uniform(0.4, 0.6)
    elif scenario_type == "link_degradation":
        return random.uniform(0.4, 0.6)
    elif scenario_type == "recovery":
        if timestep < 0.3 * max_timesteps:
            return random.uniform(0.9, 1.2)
        elif timestep < 0.5 * max_timesteps:
            return random.uniform(0.7, 0.9)
        else:
            return random.uniform(0.4, 0.6)
    elif scenario_type == "anomalous_traffic":
        if random.random() < 0.1:
            return random.uniform(1.2, 2.0)
        return random.uniform(0.3, 0.5)
    else:
        return 0.5

def apply_scenario_to_graph(graph, scenario_type, timestep, max_timesteps, seed=42):
    """
    Applies the scenario to the graph by setting base offered traffic for each edge.
    """
    multiplier = generate_scenario_multiplier(scenario_type, timestep, max_timesteps, seed)
    random.seed(seed + timestep * 1000)
    
    for u, v, data in graph.edges(data=True):
        edge_noise = random.uniform(0.8, 1.2)
        capacity = data.get('capacity_mbps', 1000.0)
        offered = capacity * multiplier * edge_noise
        
        # Link degradation scenario logic
        if scenario_type == "link_degradation" and timestep > 0.3 * max_timesteps:
            # Degrade a consistent ~10% of links by seeding on the edge tuple
            edge_seed = hash((u, v, seed))
            rnd = random.Random(edge_seed)
            if rnd.random() < 0.1:
                capacity = capacity * 0.5 # Halve the capacity temporarily
                
        data['current_offered_traffic'] = offered
        # Temporarily store adjusted capacity for this timestep's calculation if degraded
        data['current_capacity_mbps'] = capacity 
        
    return graph
