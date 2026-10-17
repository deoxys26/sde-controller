# Phase 9 Prompt 6 — Reduced-Feature Neural Forecasting

## Scope and status

This experiment evaluates **forecasting of computed link utilization from real GEANT traffic matrices**. It predicts the next utilization value for each directed link from the previous five utilization snapshots. This is not a measurement of real network link utilization and does not demonstrate production generalization or routing improvement.

All nine requested model/seed runs completed. The original synthetic Model B checkpoint and two earlier B-U checkpoints were preserved and not used. The earlier B-U artifacts used validation MAE selection, which does not satisfy this prompt's validation-loss selection rule; they are excluded from the reported experiment. The Prompt 6 checkpoints were initialized from scratch.

## Protocol and data-integrity checks

- Primary target series: distance-shortest-path utilization series, unchanged from the audited Prompt 5 baseline.
- Inputs/targets: one feature and target, `utilization`; history length 5; horizon 1.
- Windows use the frozen chronological split and valid observations only. No input includes its target or later timesteps.
- Counts and target index ranges (inclusive): train 7,344 windows, targets 7–8,019 (671 invalid candidates); validation 1,715, targets 8,025–9,739 (0 invalid); test 1,651, targets 9,745–11,395 (64 invalid). Every retained history+target window stays inside its assigned block.
- One global utilization standardization was fitted only on valid train-block timesteps: mean `0.031582471922407776`, standard deviation `0.04205260732036718`. This scaler was shared by all models; metrics are inverse-transformed to utilization units.
- Busy classification uses target-step total demand in raw float64, with the frozen train-period p90 threshold `59652.6650878`. Each test split contains 7 busy target timesteps.
- All models use the same valid starts, target/link ordering, scaler, optimizer, loss, epoch budget, and stopping rule. Seeds are 42, 123, and 2026. Adam, learning rate 0.005, batch size 32, SmoothL1 loss, maximum 5 epochs, and patience 2. The best checkpoint is selected by validation SmoothL1 loss only. Test metrics are calculated after selection and do not affect training or model selection.
- Initialization: fresh seeded initialization for all runs; no warm start. Checkpoints record model state only; run configuration, history, checkpoint paths, and seed-level metrics are in the results JSON.

## Model adaptations

- **B-U:** the existing edge QoS forecaster with utilization-only edge features: directed graph convolution, edge encoder, GRU, and prediction head.
- **C-U:** the existing DG-STMTL-inspired directed congestion forecaster with one utilization feature. Feature columns are resolved by name (`utilization`), not a positional index. Its existing input-derived link attention and task pathway are retained; congestion conditioning consequently uses utilization alone.
- **D-U:** the existing B architecture with temporal attention over GRU outputs.

For batch execution, B-U/D-U use disconnected copies of the graph. C-U computes independent per-sample link attention. The batching path was smoke-tested for expected output shapes and finite values.

## Seed-level results

MAE/RMSE and normalized MAE are for utilization. Busy metrics use the frozen demand threshold; `n` is the number of busy target timesteps (the metric aggregates link values over those timesteps).

| Model | Seed | Best epoch | Val MAE | Val RMSE | Test MAE | Test RMSE | Test norm-MAE | Busy test MAE | Busy test RMSE | Busy norm-MAE | Train seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B-U | 42 | 2 | 0.00216199 | 0.00508021 | 0.00177012 | 0.00352907 | 0.07869422 | 0.00251225 | 0.00561499 | 0.06856210 | 28.4 |
| B-U | 123 | 3 | 0.00223984 | 0.00507865 | 0.00187144 | 0.00362944 | 0.08319878 | 0.00275606 | 0.00593396 | 0.07521594 | 46.0 |
| B-U | 2026 | 2 | 0.00202011 | 0.00503682 | 0.00163344 | 0.00349511 | 0.07261795 | 0.00239605 | 0.00559430 | 0.06539088 | 29.6 |
| C-U | 42 | 2 | 0.00251933 | 0.00622504 | 0.00209003 | 0.00444648 | 0.09291635 | 0.00317262 | 0.00688649 | 0.08658428 | 40.5 |
| C-U | 123 | 4 | 0.00247195 | 0.00599665 | 0.00206456 | 0.00433122 | 0.09178437 | 0.00317911 | 0.00727277 | 0.08676143 | 81.4 |
| C-U | 2026 | 2 | 0.00240427 | 0.00598799 | 0.00198287 | 0.00427850 | 0.08815240 | 0.00307854 | 0.00723117 | 0.08401680 | 52.9 |
| D-U | 42 | 4 | 0.00197343 | 0.00499709 | 0.00159122 | 0.00350878 | 0.07074085 | 0.00243891 | 0.00574045 | 0.06656044 | 107.2 |
| D-U | 123 | 4 | 0.00206578 | 0.00511690 | 0.00167710 | 0.00355613 | 0.07455875 | 0.00249792 | 0.00563417 | 0.06817112 | 69.3 |
| D-U | 2026 | 4 | 0.00205350 | 0.00501120 | 0.00166464 | 0.00352826 | 0.07400514 | 0.00249898 | 0.00587195 | 0.06819991 | 52.3 |

`test norm-MAE = test MAE / mean test target`. The per-link test MAE for all 72 directed links is saved for every model and seed in `data/processed/geant/neural_prompt6/results.json`.

## Aggregate and non-neural comparisons

Across seeds, test MAE mean ± sample standard deviation:

| Method | Test MAE mean ± seed SD | Test RMSE mean | Seeds beating persistence | Acceptance criterion |
|---|---:|---:|---:|---|
| Persistence | 0.00156469 (single fixed baseline) | 0.00349363 | — | reference |
| B-U | 0.00175833 ± 0.00011944 | 0.00355121 ± 0.00006985 | 0/3 | Not met |
| C-U | 0.00204582 ± 0.00005598 | 0.00435207 ± 0.00008591 | 0/3 | Not met |
| D-U | 0.00164432 ± 0.00004641 | 0.00353106 ± 0.00002379 | 0/3 | Not met |

The audited Prompt 5 distance-series baseline test MAEs are persistence `0.00156469`, EWMA `0.00155746`, AR(5) `0.00162066`, seasonal-naive-96 `0.00518933`, and slot mean `0.01440454`. The neural mean MAEs are respectively worse than persistence, EWMA, and AR(5), and better than seasonal-naive-96 and slot mean. Seasonal-naive coverage was 100% in Prompt 5 (its exact coverage and same-subset persistence result are copied into the machine-readable comparison). Validation baseline metrics are also recorded there.

The acceptance rule requires beating persistence in at least two of three seeds and a mean absolute MAE gain larger than the seed-level standard deviation. No model beat persistence in any seed, so none is considered useful under that rule. D-U is closest to persistence, but still has a higher three-seed mean MAE; this is not evidence of superiority.

## Verification and artifacts

- The full Phase 9 test suite: **32 passed in 272.88s**. It includes five new Prompt 6 tests for block-local chronological windows, train-only scaler statistics, future-target noninterference with inputs, validation-loss checkpoint selection/results coverage, and model output shape/finiteness.
- An independent re-evaluation of B-U seed 42 on the full test set from its saved checkpoint and raw targets reproduced MAE `0.0017701187086053194`, RMSE `0.0035290662850075596`, normalized MAE `0.07869421854874258` exactly (difference 0.0 from the saved metrics).
- All nine checkpoint paths exist; the results file contains all three models and all three seeds.
- Results/configuration/checkpoints: `data/processed/geant/neural_prompt6/results.json`, `data/processed/geant/neural_prompt6/experiment_config.json`, and `data/processed/geant/neural_prompt6/checkpoints/`.

## Interpretation and limitations

Under this short, CPU-friendly training budget, none of B-U/C-U/D-U met the predeclared persistence-based usefulness criterion. D-U was numerically closest to persistence, but did not beat it across seeds. C-U had the highest test MAE. These are results for this GEANT-derived computed-utilization series and this protocol only. The small epoch budget, one utilization-only input, a single topology and chronological test period, and only three training seeds limit conclusions. No claims are made about real-world QoS, production generalization, routing gains, or security.

## Prompt 8 — D-Δ residual forecasting

### Hypothesis and protocol

The hypothesis was that predicting the next-step change, `Δ[t+1] = u[t+1] - u[t]`, then reconstructing `u_hat[t+1] = u[t] + Δ_hat[t+1]`, would help the D-U temporal-attention model focus on deviations from persistence. This is a separate experiment; the Prompt 6 D-U implementation, results, checkpoints, and split were left unchanged.

D-Δ reuses the exact D-U architecture, 5-step history, one-step horizon, distance-series valid windows, train/validation/test blocks, seeds 42/123/2026, and Prompt 6 training budget: Adam, learning rate 0.005, batch size 32, SmoothL1 loss, up to 5 epochs and patience 2. The input scaler remains fit only on valid train-period utilization observations. The residual target scaler is fit on residual targets from valid training windows only: mean `-0.0000205914`, standard deviation `0.0066314323`. Checkpoints are selected using validation SmoothL1 loss in standardized residual space; validation/test utilization metrics are computed after adding the predicted residual to the final observed input. No predictions are clipped, no targets are imputed, and all 72 directed links and valid windows are retained.

The split remains 7,344 train, 1,715 validation, and 1,651 test windows, with target indices and invalid-window counts matching Prompt 6. Baseline target arrays and target timestamps were checked for exact equality before evaluation. D-Δ outputs and raw-unit predictions are stored separately under `data/processed/geant/neural_prompt8_delta/`.

### Measured results

MAE/RMSE are in utilization units. Test normalized MAE is test MAE divided by the test target mean. Change diagnostics use reconstructed forecast change relative to the final input; direction accuracy excludes exactly-zero actual changes.

| Seed | Best epoch | Validation MAE | Validation RMSE | Test MAE | Test RMSE | Test norm-MAE | Signed test bias | Change correlation | Change direction accuracy | Predicted/actual change SD |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 5 | 0.00195333 | 0.00498021 | 0.00157453 | 0.00348793 | 0.06999871 | -0.00010876 | 0.12480 | 0.54077 | 0.23134 |
| 123 | 5 | 0.00193307 | 0.00495699 | 0.00154877 | 0.00344861 | 0.06885378 | -0.00005781 | 0.16135 | 0.56149 | 0.17392 |
| 2026 | 5 | 0.00196376 | 0.00497445 | 0.00158354 | 0.00347354 | 0.07039943 | -0.00014212 | 0.13132 | 0.52768 | 0.19556 |
| Mean ± sample SD | — | — | — | **0.00156895 ± 0.00001804** | **0.00347003 ± 0.00001989** | — | -0.00010289 (mean bias) | 0.13915 (mean) | 0.54331 (mean) | 0.20027 (mean ratio) |

The predicted changes remain small relative to actual test changes: mean predicted/actual change standard-deviation ratio is about 0.20, with weak mean change correlation (about 0.14). Residual targeting therefore did not establish improved change responsiveness. Out-of-range forecasts were retained; for example, the saved seed-42 test prediction minimum is `-0.00012565`.

### Same-test baseline comparison and decision

All methods use the same 1,651 target timestamps and 72 directed links. Persistence test MAE is `0.00156469` (RMSE `0.00349363`); EWMA is `0.00155746` (`0.00347652`); AR(5) is `0.00162066` (`0.00354311`).

| Seed | D-Δ MAE | Persistence MAE | EWMA MAE | AR(5) MAE | Beats persistence? |
|---:|---:|---:|---:|---:|---|
| 42 | 0.00157453 | 0.00156469 | 0.00155746 | 0.00162066 | No |
| 123 | 0.00154877 | 0.00156469 | 0.00155746 | 0.00162066 | Yes |
| 2026 | 0.00158354 | 0.00156469 | 0.00155746 | 0.00162066 | No |

D-Δ beats persistence in only 1/3 seeds. Its mean MAE gain over persistence is `-0.0000042590` (i.e. mean MAE is slightly worse), with seed-level sample SD `0.0000180429`; the gain is not positive and does not exceed that SD. **The predeclared preliminary usefulness gate is not passed.** D-Δ’s mean MAE is better than AR(5), but worse than both persistence and EWMA; it does not beat the best measured baseline, EWMA. No superiority claim is supported.

### Integrity, artifacts, and limitations

- Added six D-Δ tests for residual alignment, reconstruction (including out-of-range values to verify no clipping), train-only residual scaling, validation-loss checkpoint selection, test-target noninterference, and prediction/target alignment.
- Full Phase 9 suite: `python -m pytest -q .\phase9_geant\tests --basetemp .pytest-prompt8-final-20261011` — **38 passed in 117.13s**, including all six D-Δ tests.
- Independent recomputation from saved test predictions exactly reproduced the stored MAE, RMSE, and signed bias for all three seeds. Checkpoints exist, validation loss selects the recorded epoch, and saved target timestamps/values equal the Prompt 5 baseline archive. Prompt 6 artifacts remain separate.
- Machine-readable configuration/results: `data/processed/geant/neural_prompt8_delta/experiment_config.json` and `results.json`; aligned raw predictions: `predictions.npz`; model states: `checkpoints/D-delta_seed{42,123,2026}.pt`.
- Limitations: one GEANT-derived computed-utilization series, one fixed topology/test period, three seeds, and the same short five-epoch budget. D-Δ lowers mean MAE relative to Prompt 6 D-U, but this comparison does not change the predeclared persistence gate and does not demonstrate improvement over EWMA.

Interpretation remains limited to **forecasting of computed link utilization from real GEANT traffic matrices**; no real-world QoS, production generalization, routing, or security claim is made.
