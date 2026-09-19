# Phase 4.5 Strengthened Multi-Seed Experiment

## Scope and integrity controls

This report extends, rather than replaces, the earlier fixed-seed two-epoch experiment recorded in `proposed_gnn.md`. It does not introduce GA routing, security features in the GNN, controller work, or test-set tuning.

Synthetic SDN telemetry was generated with fixed simulator and topology seeds of 42. The training seed alone varied across 42, 123, and 2026, isolating initialization and training variability from changes in generated data. Every run used 24 timesteps, 8 nodes, 6 flows per timestep, 140 Mbps base demand, a five-snapshot history, horizon one, targets delay/utilization/packet loss/jitter, chronological 70/15/15 train/validation/test periods, and training-only feature normalization. Adam used learning rate 0.005 and batch size 4. Training was capped at 10 epochs; validation loss alone selected checkpoints and could trigger patience-4 early stopping. Test metrics were computed only after that checkpoint was loaded.

The ten required scenarios were normal, low_load, medium_load, high_load, increasing_load, congestion, traffic_spike, link_degradation, recovery, and anomalous_traffic. `increasing_load` and `congestion` share the project’s existing multiplier implementation and therefore remain duplicated experimental conditions, not independent evidence.

The primary artifact is `data/experiments/phase4_5_multiseed/results.json`: 90 unique tuples (3 models x 3 seeds x 10 scenarios). An interrupted first process had already written 78 checkpoints but no JSON; those weights were evaluated without retraining. Exactly 12 missing tuples were trained: the three `recovery/*/2026` tuples and all nine `anomalous_traffic` tuples. A transient OneDrive lock caused the interruption; checkpoint saving now retries locks. The ablation artifact is `data/experiments/phase4_5_ablation/results.json` and contains 60 new tuples (2 variants x 3 seeds x 10 scenarios). Reference and full Model C values are reused from the verified main artifact, giving four ablation arms.

## Main models: aggregate test MAE / RMSE (mean +/- population standard deviation across seeds)

| Scenario | A: reference | B: current GCN-GRU | C: proposed |
|---|---:|---:|---:|
| normal | 0.04141 +/- 0.00570 / 0.06962 +/- 0.00308 | 0.04364 +/- 0.00157 / 0.07157 +/- 0.00193 | 0.04116 +/- 0.00501 / 0.06920 +/- 0.00300 |
| low_load | 0.02473 +/- 0.00521 / 0.03726 +/- 0.00568 | 0.02963 +/- 0.00307 / 0.04368 +/- 0.00524 | 0.02410 +/- 0.00274 / 0.03680 +/- 0.00343 |
| medium_load | 0.04141 +/- 0.00570 / 0.06962 +/- 0.00308 | 0.04364 +/- 0.00157 / 0.07157 +/- 0.00193 | 0.04116 +/- 0.00501 / 0.06920 +/- 0.00300 |
| high_load | 0.04590 +/- 0.00019 / 0.10053 +/- 0.00008 | 0.04392 +/- 0.00124 / 0.10100 +/- 0.00094 | 0.04645 +/- 0.00032 / 0.10054 +/- 0.00014 |
| increasing_load | 0.11837 +/- 0.02492 / 0.24861 +/- 0.04728 | 0.08053 +/- 0.00296 / 0.16421 +/- 0.00136 | 0.09281 +/- 0.00763 / 0.21652 +/- 0.01866 |
| congestion | 0.11837 +/- 0.02492 / 0.24861 +/- 0.04728 | 0.08053 +/- 0.00296 / 0.16421 +/- 0.00136 | 0.09281 +/- 0.00763 / 0.21652 +/- 0.01866 |
| traffic_spike | 0.04784 +/- 0.00161 / 0.07710 +/- 0.00137 | 0.04148 +/- 0.00195 / 0.07063 +/- 0.00295 | 0.04841 +/- 0.00123 / 0.07791 +/- 0.00086 |
| link_degradation | 0.04354 +/- 0.00236 / 0.07820 +/- 0.00165 | 0.04406 +/- 0.00405 / 0.08151 +/- 0.00242 | 0.04350 +/- 0.00397 / 0.07853 +/- 0.00153 |
| recovery | 0.04558 +/- 0.00202 / 0.07476 +/- 0.00138 | 0.03857 +/- 0.00508 / 0.06901 +/- 0.00404 | 0.04589 +/- 0.00215 / 0.07534 +/- 0.00144 |
| anomalous_traffic | 0.04566 +/- 0.00181 / 0.06991 +/- 0.00149 | 0.03884 +/- 0.00083 / 0.06278 +/- 0.00180 | 0.04622 +/- 0.00234 / 0.06984 +/- 0.00136 |

The full JSON records per-target MAE/RMSE for every seed/scenario. Pooled descriptive values across all 30 model observations (ten scenarios x three seeds) are below; they summarize scenario variability as well as seed variability and must not be read as a seed-only confidence interval.

| Model | Delay MAE/RMSE | Utilization MAE/RMSE | Loss MAE/RMSE | Jitter MAE/RMSE |
|---|---:|---:|---:|---:|
| A | 0.05022 / 0.09247 | 0.12550 / 0.16069 | 0.04824 / 0.07376 | 0.00515 / 0.01677 |
| B | 0.03028 / 0.05254 | 0.12403 / 0.15778 | 0.03508 / 0.04254 | 0.00455 / 0.01596 |
| C | 0.04868 / 0.09429 | 0.12484 / 0.15823 | 0.03030 / 0.04866 | 0.00517 / 0.01680 |

## Model C mechanism ablation: aggregate test MAE / RMSE

| Scenario | A: neither mechanism | Directed only | Congestion only | Full C |
|---|---:|---:|---:|---:|
| normal | 0.04141 / 0.06962 | 0.04079 / 0.06894 | 0.03959 / 0.06886 | 0.04116 / 0.06920 |
| low_load | 0.02473 / 0.03726 | 0.02489 / 0.03787 | 0.02349 / 0.03650 | 0.02410 / 0.03680 |
| high_load | 0.04590 / 0.10053 | 0.04624 / 0.10062 | 0.04612 / 0.10054 | 0.04645 / 0.10054 |
| increasing_load / congestion | 0.11837 / 0.24861 | 0.09331 / 0.21188 | 0.09210 / 0.21290 | 0.09281 / 0.21652 |
| traffic_spike | 0.04784 / 0.07710 | 0.04836 / 0.07789 | 0.04729 / 0.07644 | 0.04841 / 0.07791 |
| link_degradation | 0.04354 / 0.07820 | 0.04489 / 0.07919 | 0.04466 / 0.07961 | 0.04350 / 0.07853 |
| recovery | 0.04558 / 0.07476 | 0.04566 / 0.07490 | 0.04523 / 0.07435 | 0.04589 / 0.07534 |
| anomalous_traffic | 0.04566 / 0.06991 | 0.04454 / 0.06919 | 0.04560 / 0.06988 | 0.04622 / 0.06984 |

Each ablation cell is a mean across seeds; per-seed values and standard deviations are in the JSON artifacts. Medium load duplicates normal under the existing scenario implementation and is omitted from this compact ablation table. The ablation does not show a consistent added benefit from combining both mechanisms: directed-only and congestion-only each reduce error relative to A in increasing-load/congestion, but full C has a higher RMSE there. Full C is slightly lower in MAE for repaired degradation, but that difference is within the three-seed variation.

## Computational cost

The first interrupted run did not persist elapsed time for 78 recovered checkpoints, so those times are intentionally not reconstructed. Instead, a dedicated normal-scenario, 10-epoch, three-seed timing run was performed after all accuracy experiments. Values are seconds, mean +/- population standard deviation; inference is the complete held-out test loop.

| Variant | Parameters | Training seconds | Inference seconds | Best validation epochs (42, 123, 2026) |
|---|---:|---:|---:|---|
| A reference | 8,036 | 7.678 +/- 3.005 | 0.07434 +/- 0.02449 | 10, 10, 10 |
| B current | 2,932 | 2.846 +/- 0.851 | 0.02853 +/- 0.00487 | 10, 10, 9 |
| directed only | 8,036 | 4.802 +/- 0.488 | 0.05029 +/- 0.00720 | 10, 10, 10 |
| congestion only | 8,036 | 6.718 +/- 2.005 | 0.05320 +/- 0.01095 | 10, 10, 10 |
| full C | 8,036 | 4.747 +/- 0.849 | 0.04515 +/- 0.00449 | 10, 10, 10 |

## Interpretation and conclusion

The strengthened study changes the earlier evidence from a single two-epoch, one-seed indication into a 90-configuration, three-seed result. It strengthens the earlier caution rather than reversing it. Model C has lower aggregate MAE than Model B in normal, low/medium load, and repaired degradation, but Model B has markedly lower MAE/RMSE in increasing-load/congestion, traffic spike, recovery, and anomalous traffic. Model C also has roughly 2.7 times as many parameters as Model B and longer observed normal-scenario timing.

The research hypothesis is **not supported as a broad improvement claim** under this controlled synthetic study. The evidence is scenario-dependent: the directed/congestion modifications can reduce some errors, especially compared with the reference in increasing-load/congestion, but their joint full form is neither consistently beneficial nor the lowest-RMSE configuration. This negative result is retained. It does not establish that the idea can never help; it establishes that this specific small-data implementation and protocol have not provided enough consistent evidence to justify claiming it improves future QoS forecasting for proactive routing.

Limitations remain: simulator telemetry is synthetic; ten scenarios include duplicated normal/medium and increasing/congestion definitions; only three training seeds and 24 timesteps were used; no real-network data or statistical hypothesis testing was performed; and security risk remains intentionally outside the GNN.
