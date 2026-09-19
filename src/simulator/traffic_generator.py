"""Deterministic flow demand generation for causal simulation experiments."""

from __future__ import annotations

import random

from src.simulator.scenario_generator import generate_scenario_multiplier


class TrafficGenerator:
    def __init__(self, nodes, scenario="normal", timesteps=100, seed=42,
                 flows_per_timestep=12, base_demand_mbps=180.0):
        self.nodes = tuple(nodes)
        self.scenario = scenario
        self.timesteps = timesteps
        self.seed = seed
        self.flows_per_timestep = flows_per_timestep
        self.base_demand_mbps = base_demand_mbps

    def generate(self, timestep: int) -> list[dict]:
        if len(self.nodes) < 2:
            return []
        rng = random.Random(self.seed + timestep * 1009)
        multiplier = generate_scenario_multiplier(self.scenario, timestep, self.timesteps, self.seed)
        demands = []
        for index in range(self.flows_per_timestep):
            source = rng.choice(self.nodes)
            destination = rng.choice(self.nodes)
            while destination == source:
                destination = rng.choice(self.nodes)
            demand = self.base_demand_mbps * multiplier * rng.uniform(0.65, 1.35)
            demands.append({"timestep": timestep, "source": source, "destination": destination,
                            "demand_mbps": round(demand, 6), "flow_id": f"t{timestep}_f{index}"})
        return demands
