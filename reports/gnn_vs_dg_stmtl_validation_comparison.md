# GNN vs DG-STMTL-inspired vs EWMA: validation comparison

## Executive summary

This is a descriptive synthesis of previously recorded validation metrics supplied for this report; it is not a newly executed or provenance-verified benchmark. Across the reported three-seed averages, Identity GNN has the lowest MAE, while Topology GNN has the lowest RMSE. The DG-STMTL-inspired adaptation is worse than the recorded EWMA baseline on both average metrics. Identity GNN seed 123 has the lowest individual MAE; Topology GNN seed 123 has the lowest individual RMSE. Both are below EWMA on their reported validation metrics. These observations do not establish statistical significance or test-set generalization.

The mean Topology GNN improvement versus EWMA is 6.411% MAE and 7.511% RMSE using all three historical values; seed 42's individual run manifest is unavailable in the accessible backups. On the independently available seeds 123 and 2026 only, Topology GNN averages MAE 0.00604642 (sample SD 0.00012172, n=2) and RMSE 0.01628508 (sample SD 0.00023562, n=2).

## Dataset and experimental setup

- Network representation: GEANT, 22 nodes and 72 directed links.
- Target: estimated link utilization derived from GEANT origin-destination traffic matrices; these are not measured link-level QoS observations.
- Sampling interval: 15 minutes; history length: 672 observations (7 days); horizon: 96 observations (24 hours ahead).
- Recorded data counts: 3,924 training windows and 953 validation windows; expected validation coverage is 68,616 window-link pairs (953 × 72).
- Split: chronological train/validation split. The official test split was sealed and not evaluated for the results summarized here.
- Metrics: the supplied validation MAE and RMSE values are reported in utilization-ratio units. No uncertainty interval is calculated here.

## Models compared

- **Original GNN:** EdgeQoS/GNN result groups with Topology, Identity, or Shuffled adjacency. The graph variants are presented as recorded; they are not averaged together.
- **DG-STMTL-inspired adaptation:** hybrid static/dynamic adjacency, gating, and grouped spatio-temporal processing adapted to single-target link-utilization forecasting, with an EWMA residual head. It is an adaptation, not an exact reproduction of the published DG-STMTL paper. The stated model size is 52,513 parameters.
- **EWMA:** fixed alpha 0.8 over the five observations available to each forecast. The canonical original-GNN evaluation baseline is used in comparison calculations: MAE 0.006502658485291, RMSE 0.017641687598330.

The DG-STMTL runs recorded EWMA MAE 0.006502658393768, differing from the canonical value by 9.15e-11. The recorded values are preserved; for the summary percentage comparisons, the single canonical EWMA value above is used unchanged. No EWMA error bars are shown because only one baseline aggregate was supplied.

## Complete per-seed results

Improvement is calculated as 100 × (EWMA error − model error) / EWMA error. Positive values indicate lower error than EWMA. The EWMA row is the same fixed baseline, not a seed-specific observation.

| Method | Adjacency | Seed | MAE | RMSE | MAE improvement vs EWMA | RMSE improvement vs EWMA | Parameters | Training seconds |
|---|---|---|---|---|---|---|---|---|
| GNN | Topology | 42 | 0.006164560901 | 0.016379645112 | +5.199% | +7.154% | not supplied | not supplied |
| GNN | Topology | 123 | 0.005960349073 | 0.016118466690 | +8.340% | +8.634% | not supplied | not supplied |
| GNN | Topology | 2026 | 0.006132490537 | 0.016451684668 | +5.693% | +6.745% | not supplied | not supplied |
| GNN | Identity | 42 | 0.005961605636 | 0.016297320223 | +8.320% | +7.620% | not supplied | not supplied |
| GNN | Identity | 123 | 0.005859218496 | 0.016333039149 | +9.895% | +7.418% | not supplied | not supplied |
| GNN | Identity | 2026 | 0.006022622229 | 0.016397655871 | +7.382% | +7.052% | not supplied | not supplied |
| GNN | Shuffled | 42 | 0.006164661038 | 0.016287520553 | +5.198% | +7.676% | not supplied | not supplied |
| GNN | Shuffled | 123 | 0.005959080818 | 0.016327115040 | +8.359% | +7.452% | not supplied | not supplied |
| GNN | Shuffled | 2026 | 0.006182522990 | 0.016629894350 | +4.923% | +5.735% | not supplied | not supplied |
| DG-STMTL-inspired | Hybrid static/dynamic | 42 | 0.006656095017 | 0.017735000432 | -2.360% | -0.529% | 52,513 | 332.403 |
| DG-STMTL-inspired | Hybrid static/dynamic | 123 | 0.006721429628 | 0.017765220882 | -3.364% | -0.700% | 52,513 | 291.214 |
| DG-STMTL-inspired | Hybrid static/dynamic | 2026 | 0.006550973043 | 0.017715223442 | -0.743% | -0.417% | 52,513 | 345.836 |
| EWMA | alpha=0.8, 5 observations | — | 0.006502658485 | 0.017641687598 | 0.000% | 0.000% | 0 trainable | not supplied |

## Three-seed aggregate comparison

Values are mean ± sample standard deviation (ddof=1) across the listed seeds; these are descriptive seed-to-seed summaries, not formal confidence intervals. Topology's n=3 historical summary includes seed 42's previously recorded values despite its missing individual run manifest. The other neural groups have n=3 reported runs.

| Model group | n | MAE mean ± SD | RMSE mean ± SD | MAE improvement vs EWMA | RMSE improvement vs EWMA | Parameters | Mean training seconds |
|---|---|---|---|---|---|---|---|
| Topology GNN (historical n=3) | 3 | 0.00608580 ± 0.00010982 | 0.01631660 ± 0.00017533 | +6.411% | +7.511% | not supplied | not supplied |
| Identity GNN | 3 | 0.00594782 ± 0.00008257 | 0.01634267 ± 0.00005086 | +8.533% | +7.363% | not supplied | not supplied |
| Shuffled GNN | 3 | 0.00610209 ± 0.00012417 | 0.01641484 ± 0.00018729 | +6.160% | +6.954% | not supplied | not supplied |
| DG-STMTL-inspired | 3 | 0.00664283 ± 0.00008600 | 0.01773848 ± 0.00002518 | -2.156% | -0.549% | 52,513 | 323.151 |

Ranking by mean MAE (lower is better): Identity GNN → Topology GNN (historical n=3) → Shuffled GNN → DG-STMTL-inspired. Ranking by mean RMSE (lower is better): Topology GNN (historical n=3) → Identity GNN → Shuffled GNN → DG-STMTL-inspired.

Topology-versus-DG-STMTL-inspired relative difference, defined as 100 × (DG mean − Topology mean) / DG mean: MAE +8.385%; RMSE +8.016%. Positive means the Topology mean is lower. This is descriptive and does not compare the published DG-STMTL model.

Paired-seed Topology-versus-DG-STMTL-inspired differences use the same formula; positive values favor the Topology GNN:

| Seed | Topology MAE reduction vs DG | Topology RMSE reduction vs DG | Evidence note |
|---|---|---|---|
| 42 | +7.385% | +7.642% | historical Topology metric; manifest unavailable |
| 123 | +11.323% | +9.270% | reported metric |
| 2026 | +6.388% | +7.133% | reported metric |

## Two-seed manifest-available Topology sensitivity comparison

The following sensitivity summary excludes Topology seed 42 because its individual `run.json` is missing from accessible backups. It must not be described as the full three-seed Topology mean.

| Model group | Seeds | n | MAE mean ± SD | RMSE mean ± SD | MAE improvement vs EWMA | RMSE improvement vs EWMA |
|---|---|---|---|---|---|---|
| Topology GNN (available manifests) | 123, 2026 | 2 | 0.00604642 ± 0.00012172 | 0.01628508 ± 0.00023562 | +7.016% | +7.690% |

## Best individual runs and runtime

- Best individual MAE among the supplied neural runs: GNN / Identity / seed 123, MAE 0.005859218496.
- Best individual RMSE among the supplied neural runs: GNN / Topology / seed 123, RMSE 0.016118466690.
- Mean DG-STMTL-inspired training runtime: 323.151 seconds across three runs. No comparable GNN runtime values were supplied, so runtime ranking is unavailable.

## Graph interpretation

Identity GNN achieved lower three-seed mean MAE than Topology GNN, while Topology GNN achieved lower three-seed mean RMSE. Thus Topology does not dominate every GNN variant on every metric. These results alone do not identify why the adjacency variants differ, and they do not establish that network structure is or is not useful in general.

The selected Topology GNN seed 123 result (MAE 0.0059603491; RMSE 0.0161184667) is below the supplied EWMA values on validation. This is one selected seed among reported runs; it is not evidence of a generalizable advantage.

## Why the DG-STMTL-inspired adaptation may have underperformed

All three supplied adaptation runs selected their best checkpoint at epoch 1 and their validation errors were slightly above EWMA. Plausible explanations include optimization behavior, the residual-head initialization or learning dynamics, model complexity relative to the available windows, and characteristics of this estimated-utilization task. None of these explanations has been experimentally isolated here; they are hypotheses, not findings.

## Implications for choosing a forecasting model for the GA

If a downstream GA needs a provisional validation-scored forecast, these results can inform a candidate shortlist, but they do not demonstrate routing benefit. Estimated utilization forecasts are not evidence of improved latency, packet loss, security, or GA outcomes. Any integration decision should be evaluated separately under a fixed and comparable protocol, and validation-based selection must remain distinct from final-test claims.

## Limitations and reproducibility status

- This is a descriptive comparison of supplied historical numbers, not a newly executed provenance-verified benchmark.
- The strict Colab manifest comparator did not produce a validated report because of metadata/provenance compatibility checks.
- Original Topology GNN seed 42 metrics were previously recorded, but its individual `run.json` is missing from accessible backups; the historical three-seed Topology aggregate therefore has that verification caveat. The n=2 sensitivity result reports seeds 123 and 2026 only.
- The DG-STMTL implementation is an architectural adaptation, not an exact reproduction of the paper.
- Original GNN parameter counts were not supplied; they are marked unavailable rather than inferred. The adaptation's supplied count is 52,513 parameters.
- Validation performance is not sealed-test performance. No sealed test evaluation was performed for this report.
- No measured latency comparison, packet-loss improvement, security improvement, or GA routing benefit has been established.
- The reported best seed must not be interpreted as a predeclared model-selection result. Final-test/model-selection claims remain separate.
- Three seeds do not justify statistical-significance claims; no confidence intervals are fabricated.
- These results do not demonstrate superiority over the published DG-STMTL model.

## Conclusion

On the supplied GEANT 15-minute estimated link-utilization validation results, Identity GNN has the lowest mean MAE, Topology GNN has the lowest mean RMSE, and the DG-STMTL-inspired adaptation is worse than EWMA on both mean metrics. Identity seed 123 has the best individual MAE, while Topology seed 123 has the best individual RMSE; each beats EWMA on its respective reported metric. The missing Topology seed-42 manifest limits independent verification of its full three-seed aggregate. The evidence is useful for documenting these runs, but it is not a provenance-verified head-to-head benchmark, does not establish test-set generalization, and does not establish downstream GA benefit.
