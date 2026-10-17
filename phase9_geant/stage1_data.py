import os
import glob
import re
import math
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

DATA_DIR = "data/raw/geant_topology/directed-geant-uhlig-15min-over-4months-ALL-native"
TOPO_FILE = "data/raw/geant_topology/geant.txt"
OUT_DIR = "data/processed/geant"

def haversine(lon1, lat1, lon2, lat2):
    R = 6371.0 # km
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)
    a = math.sin(dLat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dLon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return R * c

def parse_topology(path):
    nodes = {}
    links = []
    with open(path, 'r') as f:
        lines = f.readlines()
    if not lines[0].startswith("?SNDlib native format"):
        raise ValueError("First line does not start with ?SNDlib native format")
    
    in_nodes = False
    in_links = False
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'): continue
        if line.startswith("NODES ("):
            in_nodes = True
            continue
        if line.startswith("LINKS ("):
            in_nodes = False
            in_links = True
            continue
        if line.startswith("DEMANDS ("):
            in_links = False
            continue
        if line == ")":
            if in_nodes: in_nodes = False
            if in_links: in_links = False
            continue
        
        if in_nodes:
            # at1.at ( 16.3729 48.2091 )
            parts = line.split()
            node_id = parts[0]
            lon = float(parts[2])
            lat = float(parts[3])
            nodes[node_id] = (lon, lat)
        
        if in_links:
            # <link_id> ( <source> <target> ) <pre_installed_capacity> <pre_installed_capacity_cost> <routing_cost> <setup_cost> ( {<module_capacity> <module_cost>}* )
            # L_at1.at_be1.be ( at1.at be1.be ) 0 0 0 0 ( 40000 1 )
            parts = line.split()
            link_id = parts[0]
            source = parts[2]
            target = parts[3]
            links.append((source, target))
            
    return nodes, links

def dijkstra_all_pairs(nodes, edges, weight_func):
    # nodes: dict of {name: (lon,lat)}
    # edges: list of (u, v) (directed)
    # returns dict {(u,v): path_list}
    graph = {u: [] for u in nodes}
    for u, v in edges:
        graph[u].append(v)
    
    paths = {}
    for src in nodes:
        dist = {u: float('inf') for u in nodes}
        dist[src] = 0
        path_seq = {u: [] for u in nodes}
        path_seq[src] = [src]
        
        unvisited = set(nodes.keys())
        while unvisited:
            # find min dist
            curr = min(unvisited, key=lambda node: (dist[node], path_seq[node]))
            unvisited.remove(curr)
            
            if dist[curr] == float('inf'):
                break
                
            for neighbor in graph[curr]:
                if neighbor in unvisited:
                    w = weight_func(curr, neighbor)
                    new_dist = dist[curr] + w
                    new_path = path_seq[curr] + [neighbor]
                    
                    if new_dist < dist[neighbor]:
                        dist[neighbor] = new_dist
                        path_seq[neighbor] = new_path
                    elif math.isclose(new_dist, dist[neighbor], rel_tol=1e-9):
                        # tie break by lexicographic order of node sequence
                        if new_path < path_seq[neighbor]:
                            path_seq[neighbor] = new_path
                            
        for dst in nodes:
            if src != dst:
                paths[(src, dst)] = path_seq[dst]
                
    return paths

def parse_matrices(folder, nodes_dict):
    files = glob.glob(os.path.join(folder, "demandMatrix-geant-uhlig-15min-*.txt"))
    if len(files) != 11460:
        raise ValueError(f"Expected 11460 files, found {len(files)}")
    
    nodes_list = sorted(list(nodes_dict.keys()))
    node_to_idx = {n: i for i, n in enumerate(nodes_list)}
    num_nodes = len(nodes_list)
    
    # regex for timestamp
    ts_re = re.compile(r'demandMatrix-geant-uhlig-15min-(\d{8}-\d{4})\.txt')
    
    parsed_data = []
    
    for fpath in files:
        fname = os.path.basename(fpath)
        ts_match = ts_re.search(fname)
        if not ts_match:
            raise ValueError(f"Could not parse timestamp from {fname}")
        fname_ts = ts_match.group(1)
        
        with open(fpath, 'r') as f:
            content = f.read()
            
        # extract meta time
        meta_match = re.search(r'time\s*=\s*([\w-]+)', content)
        if not meta_match:
            raise ValueError(f"Could not find META time in {fname}")
        meta_ts = meta_match.group(1)
        if fname_ts != meta_ts:
            raise ValueError(f"Timestamp mismatch in {fname}: {fname_ts} != {meta_ts}")
            
        dt = datetime.strptime(meta_ts, "%Y%m%d-%H%M")
        
        # parse demands
        matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        demand_count = 0
        
        in_demands = False
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith('#'): continue
            if line.startswith('DEMANDS ('):
                in_demands = True
                continue
            if line == ')':
                in_demands = False
                continue
                
            if in_demands:
                # D_at1.at_be1.be ( at1.at be1.be ) 1 0.35400000 22
                parts = line.split()
                if len(parts) >= 6:
                    src = parts[2]
                    dst = parts[3]
                    demand_val = float(parts[5])
                    matrix[node_to_idx[src], node_to_idx[dst]] = demand_val
                    demand_count += 1
        
        parsed_data.append((dt, matrix, demand_count, fpath))
    
    # Sort by datetime
    parsed_data.sort(key=lambda x: x[0])
    
    # check for duplicates and gaps
    times = [x[0] for x in parsed_data]
    for i in range(1, len(times)):
        diff = times[i] - times[i-1]
        if diff == timedelta(0):
            raise ValueError(f"Duplicate timestamp: {times[i]} in {parsed_data[i][3]} and {parsed_data[i-1][3]}")
        if diff != timedelta(minutes=15):
            raise ValueError(f"Gap or out-of-order timestamp between {times[i-1]} and {times[i]}")
            
    return parsed_data, nodes_list, node_to_idx

def main():
    nodes, undirected_links = parse_topology(TOPO_FILE)
    if len(nodes) != 22:
        raise ValueError(f"Expected 22 nodes, found {len(nodes)}")
    if len(undirected_links) != 36:
        raise ValueError(f"Expected 36 links, found {len(undirected_links)}")
        
    directed_edges = []
    for u, v in undirected_links:
        directed_edges.append((u, v))
        directed_edges.append((v, u))
    
    def dist_weight(u, v):
        lon1, lat1 = nodes[u]
        lon2, lat2 = nodes[v]
        return haversine(lon1, lat1, lon2, lat2)
        
    def hop_weight(u, v):
        return 1.0
        
    paths_dist = dijkstra_all_pairs(nodes, directed_edges, dist_weight)
    paths_hop = dijkstra_all_pairs(nodes, directed_edges, hop_weight)
    
    print("Parsing matrices...")
    parsed_data, nodes_list, node_to_idx = parse_matrices(DATA_DIR, nodes)
    print(f"Parsed {len(parsed_data)} matrices.")
    
    num_timesteps = len(parsed_data)
    num_edges = len(directed_edges)
    
    # Edge mapping
    edge_to_idx = {e: i for i, e in enumerate(directed_edges)}
    
    capacity_mbps = 40000.0
    
    # Compute loads
    loads_dist = np.zeros((num_timesteps, num_edges), dtype=np.float32)
    loads_hop = np.zeros((num_timesteps, num_edges), dtype=np.float32)
    
    absent_pairs_counts = []
    
    total_demands_over_time = []
    mean_demand_by_hour = {h: [] for h in range(24)}
    
    for t_idx, (dt, matrix, demand_count, _) in enumerate(parsed_data):
        total_possible = 22 * 21
        absent = total_possible - demand_count
        absent_pairs_counts.append(absent)
        
        total_demand = 0.0
        
        for i, src in enumerate(nodes_list):
            for j, dst in enumerate(nodes_list):
                if i != j:
                    demand = matrix[i, j]
                    total_demand += demand
                    if demand > 0:
                        # route via dist
                        p_dist = paths_dist[(src, dst)]
                        for k in range(len(p_dist)-1):
                            e = (p_dist[k], p_dist[k+1])
                            loads_dist[t_idx, edge_to_idx[e]] += demand
                        
                        # route via hop
                        p_hop = paths_hop[(src, dst)]
                        for k in range(len(p_hop)-1):
                            e = (p_hop[k], p_hop[k+1])
                            loads_hop[t_idx, edge_to_idx[e]] += demand
                            
        total_demands_over_time.append(total_demand)
        mean_demand_by_hour[dt.hour].append(total_demand)
        
    utils_dist = loads_dist / capacity_mbps
    utils_hop = loads_hop / capacity_mbps
    
    # Check for util > 1.0
    over_capacity = np.where(utils_dist > 1.0)
    if len(over_capacity[0]) > 0:
        counts = len(over_capacity[0])
        raise ValueError(f"Utilization exceeded 1.0 in {counts} link-timesteps.")
        
    # Collinearity of utilization and throughput
    # throughput is just traffic, util is traffic / 40000. They are perfectly collinear.
    # Exclude capacity (constant 40000). Use traffic and utilization. Since they are collinear after scaling,
    # we'll keep utilization as the primary feature for ML to avoid redundancy.
    
    # Sensitivity
    diff = np.abs(utils_dist - utils_hop)
    mean_diff = np.mean(diff)
    corr = np.corrcoef(utils_dist.flatten(), utils_hop.flatten())[0,1]
    
    # Load conservation
    t_idx_test = 0
    total_link_load = np.sum(loads_dist[t_idx_test])
    total_path_load = 0
    matrix = parsed_data[t_idx_test][1]
    for i, src in enumerate(nodes_list):
        for j, dst in enumerate(nodes_list):
            if i != j:
                d = matrix[i, j]
                if d > 0:
                    hops = len(paths_dist[(src, dst)]) - 1
                    total_path_load += d * hops
                    
    assert math.isclose(total_link_load, total_path_load, rel_tol=1e-5), f"Conservation failed: {total_link_load} != {total_path_load}"
    
    print(f"Absent pairs: min={np.min(absent_pairs_counts)}, mean={np.mean(absent_pairs_counts):.2f}, max={np.max(absent_pairs_counts)}")
    print(f"Sensitivity (dist vs hop): mean abs diff = {mean_diff:.6f}, correlation = {corr:.6f}")
    print(f"Utilization: min={np.min(utils_dist):.6f}, mean={np.mean(utils_dist):.6f}, p95={np.percentile(utils_dist, 95):.6f}, max={np.max(utils_dist):.6f}")
    
    # Busiest links
    mean_utils_per_link = np.mean(utils_dist, axis=0)
    busiest_idx = np.argsort(mean_utils_per_link)[::-1][:5]
    print("Top 5 busiest links:")
    for idx in busiest_idx:
        print(f"  {directed_edges[idx]}: mean util = {mean_utils_per_link[idx]:.6f}")
        
    idle_links = np.sum(mean_utils_per_link < 0.001)
    print(f"Idle or near-idle links (<0.1% mean util): {idle_links}")
    
    # Save outputs
    # ... (Plots and JSON and NPY) ...
    plt.figure()
    plt.plot(total_demands_over_time)
    plt.title('Total Demand Over Time')
    plt.xlabel('Timestep')
    plt.ylabel('Demand (Mbps)')
    plt.savefig(os.path.join(OUT_DIR, 'plots', 'total_demand_over_time.png'))
    
    plt.figure()
    hours = sorted(list(mean_demand_by_hour.keys()))
    hourly_means = [np.mean(mean_demand_by_hour[h]) for h in hours]
    plt.plot(hours, hourly_means)
    plt.title('Mean Demand by Hour of Day')
    plt.xlabel('Hour')
    plt.ylabel('Demand (Mbps)')
    plt.savefig(os.path.join(OUT_DIR, 'plots', 'mean_demand_by_hour.png'))
    
    meta = {
        "source": "SNDlib native format, origin Steve Uhlig",
        "routing_rule": "Shortest path (Haversine distance), lexicographic tie-breaking",
        "capacity_source": "SNDlib module capacity, assumed, not measured (40000 Mbps)",
        "units": "MBITPERSEC",
        "directed_edge_assumption": "Created 2 directed edges per undirected link in topology",
        "absent_pair_stats": {
            "min": int(np.min(absent_pairs_counts)),
            "mean": float(np.mean(absent_pairs_counts)),
            "max": int(np.max(absent_pairs_counts))
        },
        "feature_availability_mask": {
            "capacity": False,
            "traffic": True,
            "utilization": True,
            "throughput": True,
            "queue_length": False,
            "delay": False,
            "packet_loss": False,
            "jitter": False
        },
        "shape": list(utils_dist.shape)
    }
    with open(os.path.join(OUT_DIR, 'metadata.json'), 'w') as f:
        json.dump(meta, f, indent=2)
        
    np.save(os.path.join(OUT_DIR, 'link_series_utilization.npy'), utils_dist)
    np.save(os.path.join(OUT_DIR, 'link_series_traffic.npy'), loads_dist)
    
    print("Done. Saved to data/processed/geant/")
    
if __name__ == '__main__':
    main()
