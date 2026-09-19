import math
import random

def build_edge_features(graph, timestep, seed=42):
    """
    Calculates dynamic telemetry metrics for each edge.
    """
    random.seed(seed + timestep)
    telemetry_records = []
    
    for u, v, data in graph.edges(data=True):
        if not data.get('active', True):
            continue
            
        # Use current_capacity if modified by scenario (like degradation), else base capacity
        capacity = data.get('current_capacity_mbps', data.get('capacity_mbps', 1000.0))
        offered = data.get('current_offered_traffic', 0.0)
        base_delay = data.get('base_delay_ms', 2.0)
        
        utilization = offered / capacity if capacity > 0 else 1.0
        
        if utilization <= 0.8:
            queue_length = 0.0
        else:
            queue_length = 100.0 * ((utilization - 0.8) ** 2)
            
        queue_delay = 0.5 * queue_length
        delay = base_delay + queue_delay
        
        noise = random.uniform(-0.1, 0.1)
        jitter = 2.0 * utilization * (1 + noise)
        if queue_length > 0:
            jitter += random.uniform(0, 0.2 * queue_length)
            
        if utilization <= 0.95:
            packet_loss = 0.0
        else:
            packet_loss = 1.0 - math.exp(-10.0 * (utilization - 0.95))
            
        packet_loss = min(max(packet_loss, 0.0), 1.0)
        
        throughput = min(offered, capacity) * (1.0 - packet_loss)
        packet_rate = (throughput * 1e6) / 12000.0
        
        record = {
            'timestep': timestep,
            'source_node': u,
            'destination_node': v,
            'bandwidth_capacity': round(capacity, 2),
            'bandwidth_utilization': round(utilization, 4),
            'delay': round(delay, 2),
            'packet_loss': round(packet_loss, 4),
            'jitter': round(jitter, 2),
            'throughput': round(throughput, 2),
            'queue_length': round(queue_length, 2),
            'packet_rate': round(packet_rate, 2)
        }
        telemetry_records.append(record)
        
    return telemetry_records

def build_node_features(edge_telemetry, graph_nodes):
    """
    Aggregates edge telemetry to generate node-level features.
    """
    nodes = {n: {'incoming_traffic': 0.0, 'outgoing_traffic': 0.0, 'queue_size': 0.0, 'flow_count': 0} for n in graph_nodes}
    
    for record in edge_telemetry:
        src = record['source_node']
        dst = record['destination_node']
        
        nodes[src]['outgoing_traffic'] += record['throughput']
        nodes[src]['queue_size'] += record['queue_length']
        nodes[src]['flow_count'] += 1
        
        nodes[dst]['incoming_traffic'] += record['throughput']
        nodes[dst]['queue_size'] += record['queue_length']
        
    node_records = []
    for node_id, data in nodes.items():
        record = {
            'timestep': edge_telemetry[0]['timestep'] if edge_telemetry else 0,
            'node': node_id,
            'incoming_traffic': round(data['incoming_traffic'], 2),
            'outgoing_traffic': round(data['outgoing_traffic'], 2),
            'queue_size': round(data['queue_size'], 2),
            'flow_count': data['flow_count'],
            'traffic_volume': round(data['incoming_traffic'] + data['outgoing_traffic'], 2),
            'node_load': round(max(0, min(1.0, data['queue_size'] / 500.0)), 4)
        }
        node_records.append(record)
        
    return node_records
