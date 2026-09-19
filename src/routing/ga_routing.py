"""Deterministic multi-objective GA over valid directed NetworkX paths."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable, Mapping

import networkx as nx


@dataclass(frozen=True)
class PathObjectives:
    delay: float
    utilization: float
    packet_loss: float
    jitter: float
    security_risk: float


class TelemetryPathCostProvider:
    """Read-only per-decision QoS/risk source for current or predicted links.

    `telemetry` may be Phase 3 records or predicted records.  Missing capacity
    falls back to graph capacity. `security_risk` is optional and remains an
    uncalibrated external score; when absent its objective is zero-weighted in
    the supplied path data rather than fabricated.
    """
    def __init__(self, graph: nx.DiGraph, telemetry: Iterable[Mapping] | None = None,
                 edge_security_risk: Mapping[tuple, float] | None = None):
        self.graph = graph
        self.records = {(row["source"], row["destination"]): row for row in (telemetry or [])}
        self.edge_security_risk = dict(edge_security_risk or {})

    def edge(self, source, destination):
        if not self.graph.has_edge(source, destination):
            raise ValueError(f"Invalid directed edge: {(source, destination)}")
        attrs, record = self.graph[source][destination], self.records.get((source, destination), {})
        capacity = float(record.get("capacity", attrs.get("capacity_mbps", 0.0)))
        traffic = float(record.get("traffic", record.get("current_offered_traffic", 0.0)))
        utilization = float(record.get("utilization", record.get("bandwidth_utilization", traffic / capacity if capacity else 1.0)))
        return {"capacity": capacity, "traffic": traffic, "available_capacity": max(0.0, capacity - traffic),
                "delay": float(record.get("delay", attrs.get("base_delay_ms", 0.0))),
                "utilization": utilization, "packet_loss": float(record.get("packet_loss", 0.0)),
                "jitter": float(record.get("jitter", 0.0)),
                "security_risk": float(self.edge_security_risk.get((source, destination), record.get("security_risk", 0.0)))}

    def feasible(self, path, demand_mbps):
        return all(self.edge(u, v)["available_capacity"] >= demand_mbps for u, v in zip(path, path[1:]))

    def objectives(self, path):
        edges = [self.edge(u, v) for u, v in zip(path, path[1:])]
        if not edges:
            raise ValueError("A path must contain at least one edge")
        return PathObjectives(delay=sum(item["delay"] for item in edges),
                              utilization=max(item["utilization"] for item in edges),
                              packet_loss=1 - math.prod(1 - min(max(item["packet_loss"], 0.0), 1.0) for item in edges),
                              jitter=sum(item["jitter"] for item in edges),
                              security_risk=1 - math.prod(1 - min(max(item["security_risk"], 0.0), 1.0) for item in edges))


class GeneticRouter:
    """Minimizes normalized path QoS objectives without reading future state."""
    default_weights = {"delay": .30, "utilization": .25, "packet_loss": .20, "jitter": .15, "security_risk": .10}

    def __init__(self, graph, cost_provider, config=None):
        self.graph, self.provider, self.config = graph, cost_provider, config or {}
        self.rng = random.Random(int(self.config.get("random_seed", 42)))
        self.weights = {key: float(self.config.get("objective_weights", {}).get(key, value))
                        for key, value in self.default_weights.items()}

    def valid_path(self, path, source=None, destination=None):
        return (len(path) >= 2 and len(path) == len(set(path)) and
                (source is None or path[0] == source) and (destination is None or path[-1] == destination) and
                all(self.graph.has_edge(u, v) for u, v in zip(path, path[1:])))

    def _feasible_paths(self, source, destination, demand_mbps):
        cutoff = int(self.config.get("max_path_length", len(self.graph)))
        try:
            paths = [list(path) for path in nx.all_simple_paths(self.graph, source, destination, cutoff=cutoff)
                     if self.provider.feasible(path, demand_mbps)]
        except (nx.NodeNotFound, nx.NetworkXNoPath):
            paths = []
        return sorted(paths, key=lambda item: (len(item), item))

    def initial_population(self, source, destination, demand_mbps):
        paths = self._feasible_paths(source, destination, demand_mbps)
        if not paths:
            raise nx.NetworkXNoPath(f"No capacity-feasible directed path from {source} to {destination}")
        target = int(self.config.get("population_size", 24))
        population = paths[:] if len(paths) <= target else self.rng.sample(paths, target)
        while len(population) < target:
            population.append(paths[len(population) % len(paths)])
        return population

    @staticmethod
    def _normalize(values):
        minimum, maximum = min(values), max(values)
        return [0.0] * len(values) if maximum == minimum else [(value - minimum) / (maximum - minimum) for value in values]

    def scored_population(self, population):
        objectives = [self.provider.objectives(path) for path in population]
        normalized = {name: self._normalize([getattr(item, name) for item in objectives]) for name in self.weights}
        return [(path, sum(self.weights[name] * normalized[name][index] for name in self.weights), objectives[index])
                for index, path in enumerate(population)]

    def _tournament(self, scored):
        count = min(int(self.config.get("tournament_size", 3)), len(scored))
        return min(self.rng.sample(scored, count), key=lambda item: item[1])[0]

    def crossover(self, left, right, source, destination, demand_mbps):
        common = [node for node in left[1:-1] if node in right[1:-1]]
        if not common:
            return list(left)
        pivot = self.rng.choice(common)
        child = left[:left.index(pivot)] + right[right.index(pivot):]
        return child if self.valid_path(child, source, destination) and self.provider.feasible(child, demand_mbps) else list(left)

    def mutate(self, path, source, destination, demand_mbps):
        alternatives = [candidate for candidate in self._feasible_paths(source, destination, demand_mbps) if candidate != path]
        return self.rng.choice(alternatives) if alternatives else list(path)

    def optimize(self, source, destination, demand_mbps=0.0):
        population = self.initial_population(source, destination, demand_mbps)
        generations, elite_count = int(self.config.get("generations", 12)), int(self.config.get("elite_count", 2))
        for _ in range(generations):
            scored = sorted(self.scored_population(population), key=lambda item: item[1])
            next_population = [list(item[0]) for item in scored[:max(1, elite_count)]]
            while len(next_population) < len(population):
                first, second = self._tournament(scored), self._tournament(scored)
                child = (self.crossover(first, second, source, destination, demand_mbps)
                         if self.rng.random() < float(self.config.get("crossover_rate", .8)) else list(first))
                if self.rng.random() < float(self.config.get("mutation_rate", .2)):
                    child = self.mutate(child, source, destination, demand_mbps)
                next_population.append(child)
            population = next_population
        best_path, fitness, objectives = min(self.scored_population(population), key=lambda item: item[1])
        return {"path": list(best_path), "fitness": fitness, "objectives": objectives}
