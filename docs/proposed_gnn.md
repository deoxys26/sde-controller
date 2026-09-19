# Proposed Directed Congestion-Aware DG-STMTL Adaptation

## Reference architecture

DG-STMTL builds task-specific hybrid adjacency from a static `3N x 3N` traffic prior and dense data-derived correlation, then uses gated, group-wise multi-task graph convolution for next-interval prediction.

## Limitation for SDN

When link QoS is the prediction entity, a dense correlation adjacency can allow unrelated directed links to exchange information merely because their historical features are similar. The original physical road-node prior also does not represent whether two SDN links can be consecutive hops on a route. This mismatch matters during congestion propagation: a queue on `(u,v)` is most directly relevant to downstream links leaving `v`, not every statistically similar link.

## Proposed modification

**Original DG-STMTL component:** a static road/node prior plus unconstrained dynamic all-pairs correlation, gated per task.

**Problem in our SDN setting:** road-node adjacency and dense correlations do not encode directed link continuity, while the targets are directed-link QoS measurements.

**Our modification:** represent directed links as prediction entities. Construct the static prior so link `(u,v)` can pass state to `(v,w)` (plus self-links). Reweight this prior by the recent similarity of normalized utilization and queue length. Mask the dynamic correlation by this directed, congestion-conditioned prior before task-specific hybrid gating.

**Why this modification addresses the problem:** it limits dynamic exchange to plausible path-continuous congestion propagation while retaining a learned, task-specific dynamic component.

**Expected effect:** this is hypothesized to improve forecasts during congestion buildup, spikes, degradation, and recovery. It is not assumed to improve every target or scenario.

## Proposed architecture

```text
Historical directed link QoS snapshots
        |                         |
DG-STMTL task encoders      utilization/queue state
        |                         |
dense dynamic correlation   directed link-continuity prior
        \                         /
         congestion-conditioned, task-gated hybrid adjacency
                              |
                 three-stage residual group graph convolution
                              |
                  shared multi-task output head -> t+1 QoS
```

## Models and ablation plan

- **Model A:** `DGSTMTLReferenceForecaster`, a constrained implementation-oriented adaptation: link entities, endpoint-sharing static link prior, dense dynamic adjacency, task projections/gates, three residual graph stages, and shared multi-task head. Exact reproduction is not possible because the paper uses road-node inputs, a `3N x 3N` synchronous prior, 12-step history, and unavailable original code/data.
- **Model B:** unchanged `EdgeQoSForecaster`, the existing directed graph-convolution plus GRU baseline.
- **Model C:** `DirectedCongestionDGSTMTLForecaster`, Model A with the directed-continuity and congestion-conditioned adjacency modification.

The principal ablation is A versus C; B establishes continuity with Phase 4. All use identical simulated telemetry, target columns, chronology, training-only scaler, history/horizon, optimizer, seed, and test periods. Later work must compare all scenarios without tuning on test metrics. Security remains downstream and is not an input feature here.

## Initial executed comparison (not a performance claim)

One fixed-seed CPU smoke comparison was run on the same 20-timestep normal-traffic simulation, 6-step history, one-step horizon, chronological 70/15/15 split, 1 epoch, learning rate 0.005, and batch size 4. Aggregate test MAE/RMSE were: Model B current baseline 0.05934/0.07966; Model A reference 0.06668/0.08644; Model C proposed 0.06714/0.08698. Therefore this initial small run does **not** support the hypothesis that Model C improves prediction. It is intentionally retained as a negative/inconclusive result.

The planned scenario suite is normal, increasing_load, low_load, medium_load, high_load, congestion, traffic_spike, link_degradation, recovery, and anomalous_traffic. The following section records its initial fixed-protocol execution; it is deliberately separated from future longer-training studies.

## Ten-scenario fixed-protocol study

The scenario sweep has now been executed after repairing `NetworkSimulator` degradation. Every result used the same 8-node seeded directed topology, 24 simulation timesteps, 6 flows/timestep, 140 Mbps base demand, queue capacity 250 Mbps, 5-step history, horizon 1, chronological 70/15/15 partition, training-only standardization, seed 42, Adam learning rate 0.005, batch size 4, and two epochs. These are small CPU smoke experiments, not tuned final models. `increasing_load` and `congestion` currently share the same Phase 2 multiplier definition, so their identical results are expected rather than independent evidence.

Aggregate test MAE / RMSE:

| Scenario | Model B current | Model A reference | Model C proposed |
|---|---:|---:|---:|
| normal | 0.06070 / 0.08516 | 0.05691 / 0.08405 | 0.06005 / 0.08593 |
| low_load | 0.04583 / 0.06350 | 0.04615 / 0.06495 | 0.04621 / 0.06505 |
| medium_load | 0.06070 / 0.08516 | 0.05691 / 0.08405 | 0.06005 / 0.08593 |
| high_load | 0.06362 / 0.11210 | 0.06531 / 0.11415 | 0.06532 / 0.11407 |
| increasing_load | 0.09423 / 0.16867 | 0.09350 / 0.17209 | 0.09100 / 0.17202 |
| congestion | 0.09423 / 0.16867 | 0.09350 / 0.17209 | 0.09100 / 0.17202 |
| traffic_spike | 0.05077 / 0.08214 | 0.06243 / 0.08791 | 0.06240 / 0.08790 |
| link_degradation | 0.06165 / 0.09197 | 0.05901 / 0.09171 | 0.05898 / 0.09170 |
| recovery | 0.05193 / 0.08185 | 0.05779 / 0.08533 | 0.05948 / 0.08580 |
| anomalous_traffic | 0.04504 / 0.06989 | 0.05976 / 0.08020 | 0.05970 / 0.08020 |

Per-target MAE/RMSE was measured for delay, utilization, packet loss, and jitter in every run by `evaluate_model`; no target was omitted from aggregate computation. The main qualitative pattern is consistent: Model C had the lowest aggregate MAE only in the increasing-load/congestion cases (0.09100) and was marginally best in repaired link-degradation (0.05898/0.09170), but did not beat Model B broadly and did not have the lowest RMSE in congestion. The pre-registered hypothesis is therefore **not supported by this small fixed-protocol study**. This is a negative/inconclusive result, not evidence that the modification is universally ineffective.

The degradation run contained 32 degraded directed-link telemetry rows after onset, with a 50% effective capacity factor. That event is now a genuine capacity-change scenario. No test-set tuning or security input was used.

## Strengthened multi-seed conclusion

The later 10-epoch, validation-only, three-training-seed study is documented in `docs/phase4_5_experiment_report.md` and its JSON artifacts under `data/experiments/`. It preserves the preliminary negative/inconclusive result and strengthens the caution: Model C did not provide a broad, stable improvement relative to the current GCN-GRU baseline. The Model C ablation likewise did not establish that combining directed continuity with congestion conditioning consistently reduces error. This project therefore has no experimental basis yet to claim that the proposed modification supports proactive routing better than the existing baseline.
