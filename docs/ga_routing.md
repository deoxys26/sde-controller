# Phase 6: Genetic Algorithm Multi-Objective Routing

## Scope

Phase 6 implements a deterministic optimizer only. It does not yet connect the simulator, a GNN, UNSW-NB15, or an SDN controller into a final routing experiment. It can consume current telemetry, a separately supplied predicted-link telemetry table, and optional externally associated edge-risk scores through one `TelemetryPathCostProvider` interface.

## Chromosome and constraints

A chromosome is an ordered directed path `[source, ..., destination]`. Population generation enumerates only simple directed paths up to `max_path_length`, filters capacity-infeasible paths before sampling, and fails with `NetworkXNoPath` if none exist. A path has no repeated nodes and every consecutive directed edge must exist.

For requested demand `D`, an edge is feasible only when `D <= capacity - traffic`. Current Phase 3 telemetry supplies capacity and traffic. A predicted QoS input may omit capacity/traffic, in which case static graph capacity and zero traffic are the explicit fallback; Phase 7 must instead supply an appropriate future capacity/load estimate before claiming proactive capacity feasibility.

## Objective

For path `P`, lower fitness is preferred:

`fitness = w_delay*d + w_utilization*u + w_loss*l + w_jitter*j + w_security*r`

where each component is min-max normalized across the capacity-feasible candidate population at that routing decision. A constant objective is normalized to zero. Thus no telemetry outside the present routing decision is used to set normalization bounds.

`delay = sum(edge_delay)`; `packet_loss = 1 - product(1 - edge_loss)`; `utilization = max(edge_utilization)`; and `jitter = sum(edge_jitter)`. Additive jitter is a deterministic conservative proxy for accumulated hop-to-hop delay variation in this macroscopic simulator; it is not a packet-level variance model. Security risk uses the same union-style aggregation, `1 - product(1 - edge_risk)`, and is zero only when no risk values are supplied. Existing Phase 5 risk remains an uncalibrated external score, not a real-world probability.

Default weights are delay 0.30, utilization 0.25, loss 0.20, jitter 0.15, and security risk 0.10. They are initial documented values in `configs/simulation.yaml`, not tuned on final scenarios.

## Operators

Tournament selection chooses the lower-fitness member of a seeded random subset. Crossover joins two parent paths at a common internal node and accepts the child only if it remains directed, loop-free, and capacity-feasible; otherwise it deterministically retains the first parent. Mutation replaces a path with a seeded alternate feasible simple source-destination path. Elitism copies the configured best paths forward.

## Configuration and limitations

Defaults: population 24, generations 12, mutation rate 0.20, crossover rate 0.80, two elites, tournament size three, maximum path length eight, and seed 42. The implementation is appropriate for small synthetic topology validation, not large graphs where enumerating simple paths becomes expensive. It has no final routing-performance claim and does not preselect any Phase 4.5 forecasting model.
