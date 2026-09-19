# Phase 3 Routing Baselines and Causal Simulator

This phase creates **synthetic** network telemetry. It is a deterministic, discrete-time, macroscopic model; it is not a packet-level emulator or a claim about real-network performance.

## Traffic and simulator model

At each timestep, a seeded traffic generator creates source/destination flows with a scenario multiplier inherited from Phase 2 (`normal`, `increasing_load`/`congestion`, `traffic_spike`, and `recovery`). A routing plugin selects one path per flow. Flow demand is then summed on every directed link. No link receives traffic unless a selected flow traverses it.

For each link, utilization is `traffic / capacity`. Excess traffic grows a bounded queue, after which queueing delay and jitter increase. Loss is zero below capacity unless a queue overflow occurs; above capacity it grows smoothly with overload and overflow. Successful link throughput is bounded by capacity and reduced by loss. A flow's throughput is its demand times the least successful link ratio on its path. Thus higher load causally produces higher utilization, queues, delay, jitter and loss, while throughput saturates or falls.

The queue uses a configurable fractional carry-over between timesteps. This is intentionally simple and makes recovery gradual, but does not model packet scheduling, protocol retransmission, or individual packets.

For the `link_degradation` scenario, a deterministic seed-derived subset of directed links has its effective capacity reduced after a configurable fraction of the run. The base topology capacity remains unchanged; telemetry reports both `base_capacity` and effective `capacity`, plus `degraded`. Routes continue to be selected by the configured baseline and are not removed merely because capacity is reduced.

## Baselines

`StaticShortestPath` caches a valid minimum-hop NetworkX path for each source/destination pair. It returns `None` when no route exists.

`ReactiveRouting` recomputes a weighted shortest path from telemetry explicitly supplied by the simulator:

`cost = w_delay * normalized_delay + w_loss * packet_loss + w_utilization * utilization`

Delay is normalized by the largest supplied delay at that decision. Weights are configured in `configs/simulation.yaml` under `reactive_routing_weights`.

## Temporal-leakage prevention

The simulator passes only telemetry saved at the end of timestep `t-1` into the routing decision for timestep `t`. Neither baseline reads generated future telemetry. The reactive strategy itself retains no hidden telemetry state; its result depends only on topology and the passed current/past records. This conservative sequencing prevents forecasting information from leaking into either conventional baseline.

## Outputs and limitations

`NetworkSimulator.run()` returns flow telemetry (demand, selected path, path length, throughput, delay, loss and jitter) and link telemetry (capacity, traffic, utilization, queue, delay, loss, jitter and throughput) as Pandas data frames. The model omits failures, per-packet dynamics, TCP effects, heterogeneous traffic classes and real controller delay. Those constraints should be preserved when interpreting later experiments.
