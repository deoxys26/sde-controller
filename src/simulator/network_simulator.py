"""A small, deterministic, macroscopic discrete-time network simulator."""

from __future__ import annotations

import math
import random
from collections import defaultdict

import pandas as pd

from src.routing.baseline_methods import ReactiveRouting, StaticShortestPath
from src.simulator.traffic_generator import TrafficGenerator


class NetworkSimulator:
    def __init__(self, graph, config=None, routing_strategy=None):
        self.graph = graph.copy()
        self.config = config or {}
        self.timesteps = int(self.config.get("simulation_timesteps", 100))
        self.queue_capacity = float(self.config.get("queue_capacity_mbps", 300.0))
        self.queue_recovery = float(self.config.get("queue_recovery_fraction", 0.35))
        self.scenario = self.config.get("scenario", "normal")
        self.seed = int(self.config.get("seed", 42))
        self.degradation_start_fraction = float(self.config.get("link_degradation_start_fraction", .30))
        self.degradation_fraction = float(self.config.get("link_degradation_fraction", .15))
        self.degradation_capacity_factor = float(self.config.get("link_degradation_capacity_factor", .50))
        self.queue_state = defaultdict(float)
        self.previous_telemetry: list[dict] = []
        self.traffic_generator = TrafficGenerator(
            self.graph.nodes(), scenario=self.scenario, timesteps=self.timesteps,
            seed=self.seed,
            flows_per_timestep=int(self.config.get("traffic_flows_per_timestep", 12)),
            base_demand_mbps=float(self.config.get("traffic_base_demand_mbps", 180.0)))
        if routing_strategy is None:
            if self.config.get("routing_method", "static_shortest_path") == "reactive":
                routing_strategy = ReactiveRouting(self.config.get("reactive_routing_weights", {}))
            else:
                routing_strategy = StaticShortestPath()
        self.routing_strategy = routing_strategy

    @staticmethod
    def _edges(path):
        return list(zip(path, path[1:]))

    def _is_degraded(self, source, destination):
        """Stable edge selection without Python's process-randomized hash."""
        ordered = tuple(sorted((source, destination)))
        edge_seed = self.seed * 1_000_003 + int(ordered[0]) * 10_007 + int(ordered[1]) * 101
        return random.Random(edge_seed).random() < self.degradation_fraction

    def _effective_capacity(self, source, destination, base_capacity, timestep):
        """Temporary deterministic link-capacity reduction after the event onset."""
        event_start = self.degradation_start_fraction * self.timesteps
        if (self.scenario == "link_degradation" and timestep >= event_start and
                self._is_degraded(source, destination)):
            return base_capacity * self.degradation_capacity_factor
        return base_capacity

    def step(self, timestep: int):
        demands = self.traffic_generator.generate(timestep)
        routed, loads = [], defaultdict(float)
        # Decisions consult only completed telemetry from earlier timesteps.
        for demand in demands:
            path = self.routing_strategy.select_path(self.graph, demand["source"], demand["destination"],
                                                     self.previous_telemetry)
            record = dict(demand, selected_path=path, path_length=(len(path) - 1 if path else None))
            routed.append(record)
            if path:
                for edge in self._edges(path):
                    loads[edge] += demand["demand_mbps"]

        link_rows = []
        for u, v, attrs in self.graph.edges(data=True):
            base_capacity = float(attrs.get("capacity_mbps", 1000.0))
            capacity = self._effective_capacity(u, v, base_capacity, timestep)
            traffic = loads[(u, v)]
            previous_queue = self.queue_state[(u, v)] * self.queue_recovery
            excess = max(0.0, traffic - capacity)
            queue = min(self.queue_capacity, previous_queue + excess)
            self.queue_state[(u, v)] = queue
            utilization = traffic / capacity if capacity else 1.0
            queue_delay = 0.02 * queue
            delay = float(attrs.get("base_delay_ms", 2.0)) + queue_delay
            overflow = max(0.0, previous_queue + excess - self.queue_capacity)
            loss = 0.0 if traffic <= capacity and overflow == 0 else min(0.99, 1 - math.exp(-2.5 * max(0, utilization - 1)) + overflow / max(traffic, 1.0))
            throughput = min(traffic, capacity) * (1 - loss)
            jitter = 0.05 * delay + 0.01 * queue
            link_rows.append({"timestep": timestep, "source": u, "destination": v, "capacity": capacity,
                              "base_capacity": base_capacity, "degraded": capacity < base_capacity,
                              "traffic": traffic, "utilization": utilization, "queue_length": queue,
                              "delay": delay, "packet_loss": loss, "jitter": jitter, "throughput": throughput})

        by_edge = {(r["source"], r["destination"]): r for r in link_rows}
        flow_rows = []
        for record in routed:
            path = record["selected_path"]
            if not path:
                flow_rows.append({**record, "throughput": 0.0, "delay": float("inf"), "packet_loss": 1.0, "jitter": 0.0})
                continue
            hops = [by_edge[edge] for edge in self._edges(path)]
            success_ratio = min((row["throughput"] / row["traffic"] if row["traffic"] else 1.0) for row in hops)
            flow_rows.append({**record, "throughput": record["demand_mbps"] * success_ratio,
                              "delay": sum(row["delay"] for row in hops),
                              "packet_loss": 1 - success_ratio,
                              "jitter": sum(row["jitter"] for row in hops)})
        self.previous_telemetry = link_rows
        return flow_rows, link_rows

    def run(self, timesteps=None):
        all_flows, all_links = [], []
        for timestep in range(self.timesteps if timesteps is None else int(timesteps)):
            flows, links = self.step(timestep)
            all_flows.extend(flows)
            all_links.extend(links)
        return pd.DataFrame(all_flows), pd.DataFrame(all_links)
