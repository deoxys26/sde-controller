import networkx as nx
import pytest

from src.routing.ga_routing import GeneticRouter, TelemetryPathCostProvider


def _graph_and_provider():
    graph = nx.DiGraph()
    graph.add_edge(0, 1, capacity_mbps=100, base_delay_ms=2)
    graph.add_edge(1, 3, capacity_mbps=100, base_delay_ms=3)
    graph.add_edge(0, 2, capacity_mbps=100, base_delay_ms=4)
    graph.add_edge(2, 3, capacity_mbps=100, base_delay_ms=5)
    telemetry = [{"source": 0, "destination": 1, "capacity": 100, "traffic": 20, "delay": 2, "utilization": .2, "packet_loss": .1, "jitter": 1},
                 {"source": 1, "destination": 3, "capacity": 100, "traffic": 20, "delay": 3, "utilization": .2, "packet_loss": .2, "jitter": 2},
                 {"source": 0, "destination": 2, "capacity": 100, "traffic": 90, "delay": 4, "utilization": .9, "packet_loss": .0, "jitter": 3},
                 {"source": 2, "destination": 3, "capacity": 100, "traffic": 10, "delay": 5, "utilization": .1, "packet_loss": .0, "jitter": 4}]
    provider = TelemetryPathCostProvider(graph, telemetry, {(0, 1): .1, (1, 3): .3})
    return graph, provider


def _router():
    graph, provider = _graph_and_provider()
    return GeneticRouter(graph, provider, {"random_seed": 4, "population_size": 6, "generations": 3,
                                            "elite_count": 1, "max_path_length": 4})


def test_path_objectives_and_capacity_constraints():
    graph, provider = _graph_and_provider()
    values = provider.objectives([0, 1, 3])
    assert values.delay == 5 and values.jitter == 3
    assert values.utilization == .2 and values.packet_loss == pytest.approx(.28)
    assert values.security_risk == pytest.approx(.37)
    assert provider.feasible([0, 1, 3], 70)
    assert not provider.feasible([0, 2, 3], 20)
    assert not GeneticRouter(graph, provider).valid_path([0, 1, 0, 2, 3], 0, 3)


def test_population_normalized_security_weight_changes_fitness():
    graph, provider = _graph_and_provider()
    router = GeneticRouter(graph, provider, {"objective_weights": {"delay": 0, "utilization": 0,
                                               "packet_loss": 0, "jitter": 0, "security_risk": 1}})
    scored = {tuple(path): fitness for path, fitness, _ in router.scored_population([[0, 1, 3], [0, 2, 3]])}
    assert scored[(0, 1, 3)] == 1.0
    assert scored[(0, 2, 3)] == 0.0


def test_population_operators_elitism_and_determinism():
    router = _router()
    population = router.initial_population(0, 3, 10)
    assert all(router.valid_path(path, 0, 3) and len(path) == len(set(path)) for path in population)
    child = router.crossover([0, 1, 3], [0, 1, 3], 0, 3, 10)
    assert router.valid_path(child, 0, 3)
    assert router.valid_path(router.mutate([0, 1, 3], 0, 3, 10), 0, 3)
    first, second = _router().optimize(0, 3, 10), _router().optimize(0, 3, 10)
    assert first == second
    assert first["path"] == [0, 1, 3]


def test_no_capacity_feasible_path_fails_gracefully():
    with pytest.raises(nx.NetworkXNoPath):
        _router().optimize(0, 3, 95)
