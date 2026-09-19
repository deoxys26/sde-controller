import networkx as nx

from src.routing.baseline_methods import ReactiveRouting, StaticShortestPath


def _diamond():
    graph = nx.DiGraph()
    graph.add_edges_from([(0, 1, {"base_delay_ms": 1}), (1, 3, {"base_delay_ms": 1}),
                          (0, 2, {"base_delay_ms": 1}), (2, 3, {"base_delay_ms": 1})])
    return graph


def test_static_shortest_path_is_valid_and_handles_unreachable():
    graph = _diamond()
    router = StaticShortestPath()
    path = router.select_path(graph, 0, 3)
    assert path[0] == 0 and path[-1] == 3
    assert len(path) == len(set(path))
    assert all(graph.has_edge(u, v) for u, v in zip(path, path[1:]))
    assert router.select_path(graph, 0, 99) is None


def test_reactive_routing_uses_supplied_current_or_past_telemetry():
    graph = _diamond()
    router = ReactiveRouting({"delay": 0.1, "packet_loss": 0.1, "utilization": 0.8})
    quiet = [{"source": 0, "destination": 1, "delay": 1, "packet_loss": 0, "utilization": .05},
             {"source": 1, "destination": 3, "delay": 1, "packet_loss": 0, "utilization": .05},
             {"source": 0, "destination": 2, "delay": 1, "packet_loss": 0, "utilization": .9},
             {"source": 2, "destination": 3, "delay": 1, "packet_loss": 0, "utilization": .9}]
    assert router.select_path(graph, 0, 3, quiet) == [0, 1, 3]
    changed = [dict(row, utilization=.95) if row["destination"] == 1 or row["source"] == 1 else dict(row, utilization=.01) for row in quiet]
    assert router.select_path(graph, 0, 3, changed) == [0, 2, 3]
    # The method has no state: unsupplied future records cannot affect this decision.
    assert router.select_path(graph, 0, 3, quiet) == [0, 1, 3]
