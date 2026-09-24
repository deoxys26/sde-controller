# Phase 7: Causal Proactive QoS Routing Evaluation

## Scope

This experiment compares conventional shortest-path routing, reactive routing,
current-state QoS-aware GA routing, and a frozen-GNN predicted-QoS GA. It uses
synthetic telemetry from the project's discrete-time simulator; it is not a
real-network deployment study.

## Protocol

All methods use the same directed topology, topology/simulator seed (42),
scenario, generated demand records, source/destination pairs, held-out
chronological test window, and Phase 6 GA parameters. The first held-out
target timestep from the Phase 4.5 protocol begins evaluation. Earlier
timesteps establish each policy's causal queue/telemetry history but are not
scored.

At decision timestep *t*, static, reactive, and current-QoS GA methods see at
most completed telemetry through *t-1*. The predicted-QoS GA receives a
one-step forecast made from its own completed five-snapshot history ending at
*t-1*. Only after paths are selected does a policy-specific simulator instance
apply the identical timestep demand batch and produce realized link QoS.

The frozen forecast model is the Phase 4 baseline (`current`, training seed
42), selected before this experiment. It is neither retrained nor selected on
Phase 7 validation/test results. The scaler is reconstructed from the original
Phase 4.5 chronological training period only and serialized under the Phase 7
artifact directory. Evaluation observations never fit the scaler.

## Measurements

For every method and scenario, the results record link and flow sample counts,
failed-path count/rate, realized mean link delay, utilization, packet loss,
jitter, throughput, successful-flow delay, and flow throughput. Aggregate
values report the mean and population standard deviation across the ten fixed
scenarios. Results are machine-readable in
`data/experiments/phase7/results.json`.

## Synthetic security-interface check

The separate check supplies a deliberately synthetic per-edge risk map to the
GA and confirms selection of a lower-risk path when all non-security objectives
are zero-weighted. These values are not UNSW-NB15 predictions, are not mapped
to simulator links, and provide no security-performance evidence.

## Limitations

- The simulator produces synthetic QoS, so findings do not establish
  real-world generalization.
- Each routing policy changes its own queue history; the frozen GNN was trained
  on Phase 4.5 simulator rollouts and may face distribution shift.
- The evaluation has one fixed simulator/topology seed, so per-scenario
  variation is reported but is not a multi-topology confidence interval.
- No flow-risk-to-link-risk association exists; security-aware routing efficacy
  is intentionally outside the main experiment.

## Measured results

All ten scenarios completed with five held-out timesteps each: 1,200 link
observations and 300 flows per method. No method had an infeasible selected
path. Mean +/- population standard deviation across scenarios:

| Method | Link delay (ms) | Utilization | Packet loss | Jitter (ms) |
| --- | ---: | ---: | ---: | ---: |
| Shortest path | 2.0020 +/- 0.0080 | 0.1013 +/- 0.0459 | 0.00093 +/- 0.00185 | 0.1021 +/- 0.0044 |
| Reactive | 2.0016 +/- 0.0073 | 0.1008 +/- 0.0460 | 0.00079 +/- 0.00158 | 0.1019 +/- 0.0040 |
| Current-QoS GA | 2.0185 +/- 0.0323 | 0.1191 +/- 0.0561 | 0.00355 +/- 0.00596 | 0.1112 +/- 0.0178 |
| Predicted-QoS GA | 2.0117 +/- 0.0246 | 0.1113 +/- 0.0586 | 0.00313 +/- 0.00572 | 0.1075 +/- 0.0135 |

Predicted-QoS GA is lower than current-QoS GA on all four reported aggregate
QoS quantities in this fixed protocol, but neither GA method outperforms the
reactive or shortest-path baselines on these aggregate QoS measures. This is
not evidence of general GNN-routing superiority. GA's higher mean link
throughput is not interpreted as a benefit because it also counts traffic
across longer/more heavily used paths. The exact per-scenario records, flow
throughput, failure values, and protocol fields remain in the machine-readable
artifact.
