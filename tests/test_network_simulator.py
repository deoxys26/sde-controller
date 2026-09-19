import pandas as pd
import pytest

from src.simulator.network_simulator import NetworkSimulator
from src.simulator.traffic_generator import TrafficGenerator
from src.topology.graph_builder import build_topology


def _config(**updates):
    config = {"seed": 7, "simulation_timesteps": 5, "scenario": "normal",
              "traffic_flows_per_timestep": 8, "traffic_base_demand_mbps": 150,
              "queue_capacity_mbps": 200, "routing_method": "static_shortest_path"}
    config.update(updates)
    return config


def test_traffic_and_simulation_are_deterministic_and_paths_are_valid():
    graph = build_topology(num_nodes=8, seed=7, default_capacity_mbps=300)
    generator = TrafficGenerator(graph.nodes(), seed=7, timesteps=5)
    assert generator.generate(2) == generator.generate(2)
    flows1, links1 = NetworkSimulator(graph, _config()).run()
    flows2, links2 = NetworkSimulator(graph, _config()).run()
    pd.testing.assert_frame_equal(flows1, flows2)
    pd.testing.assert_frame_equal(links1, links2)
    assert {"traffic", "utilization", "queue_length", "delay", "packet_loss", "throughput"} <= set(links1)
    for path in flows1.selected_path:
        assert path[0] in graph and path[-1] in graph
        assert len(path) == len(set(path))
        assert all(graph.has_edge(u, v) for u, v in zip(path, path[1:]))


def test_traffic_is_aggregated_and_overload_degrades_qos():
    graph = build_topology(num_nodes=5, seed=2, default_capacity_mbps=100)
    flows, links = NetworkSimulator(graph, _config(scenario="congestion", simulation_timesteps=12,
                                                    traffic_base_demand_mbps=400)).run()
    for _, row in links.iterrows():
        expected = sum(flow.demand_mbps for _, flow in flows[flows.timestep == row.timestep].iterrows()
                       if (row.source, row.destination) in zip(flow.selected_path, flow.selected_path[1:]))
        assert row.traffic == pytest.approx(expected)
    overloaded = links[links.utilization > 1]
    assert not overloaded.empty
    assert (overloaded.packet_loss > 0).any()
    assert (overloaded.throughput <= overloaded.capacity).all()
    assert overloaded.delay.mean() > links[links.utilization <= 1].delay.mean()


def test_link_degradation_is_deterministic_and_reduces_actual_capacity():
    graph = build_topology(num_nodes=10, seed=12, default_capacity_mbps=100)
    config = _config(seed=12, scenario="link_degradation", simulation_timesteps=10,
                     traffic_flows_per_timestep=20, traffic_base_demand_mbps=300,
                     link_degradation_fraction=.5, link_degradation_capacity_factor=.5)
    _, first = NetworkSimulator(graph, config).run()
    _, second = NetworkSimulator(graph, config).run()
    pd.testing.assert_frame_equal(first, second)
    before = first[first.timestep < 3]
    after = first[first.timestep >= 3]
    assert not before.degraded.any()
    assert after.degraded.any()
    assert (after.loc[after.degraded, "capacity"] < after.loc[after.degraded, "base_capacity"]).all()
    assert after.loc[after.degraded, "utilization"].mean() > after.loc[~after.degraded, "utilization"].mean()
