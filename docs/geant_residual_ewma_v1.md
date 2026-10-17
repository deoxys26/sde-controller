# GEANT v2 neural residual-to-EWMA experiment — protocol

Status: the three train/validation-only residual runs and three B-U runs completed on the isolated rebuild. The path-form mismatch in evaluator preflight was corrected without changing run metadata, and the validation-only comparison completed. No test evaluation was run.

## Method

Each prebuilt GEANT v2 window contains five consecutive historical observations for each of 72 directed links and one next-step target. For history `u[t-4:t+1]`, fixed EWMA alpha `0.8` gives

`w[k] = 0.8 * (1 - 0.8) ** (4 - k) / sum_j(0.8 * (1 - 0.8) ** (4 - j))`, for `k=0..4` oldest to newest,

`ewma[t+1] = sum_k(w[k] * u[t-4+k])`.

These weights and their order match `phase9_geant/run_baselines_v2_validation.py`. They use only the five inputs, never the target. The supervised target is `residual[t+1] = actual[t+1] - ewma[t+1]`. Final utilization prediction is `ewma[t+1] + inverse_scale(predicted_residual[t+1])`.

The model is a shared, per-directed-link MLP (`5 -> 16 -> ReLU -> 1`). Its only inputs are that link's five historical utilization values. It receives no topology, neighboring-link features, or future target. Inputs are standardized with the existing v2 scalar utilization normalization fit on training arrays only. Residual targets have a separate mean/standard-deviation scaler fitted only on training residuals. No clipping or imputation is used.

## Training and selection plan

Seeds: 42, 123, 2026. Each run is fresh and writes to its own fixed directory under the approved, separate rebuild root:

- `neural_ewma_residual_v1_seed_42`
- `neural_ewma_residual_v1_seed_123`
- `neural_ewma_residual_v1_seed_2026`

The intended root is `data/processed/geant_validity_v2_rebuild_trainval_20261017_v1`; it must be confirmed by the project owner before building. Run names remain `neural_ewma_residual_v1_seed_{42,123,2026}` beneath that new root. Existing destinations are refused. Settings: at most 5 epochs, patience 2, Adam at 0.005, batch size 32, SmoothL1 loss on standardized residuals, initialization from scratch. Only training windows contribute gradients. Validation SmoothL1 on standardized residuals is the sole early-stopping and checkpoint-selection objective. The best state dict is strictly reloaded before validation metrics are computed. The intended per-seed command is `python -m phase9_geant.train_residual_ewma_v1 --run --seed SEED --data-root data/processed/geant_validity_v2_rebuild_trainval_20261017_v1`; it has not been executed.

Chronological boundaries remain train `[0, 8020)`, validation `[8020, 9740)`, test `[9740, 11460)`. Existing v2 construction uses five history steps and a one-step target, with all six observations valid, consecutive, and within one split. The runner consumes only the four train/validation arrays plus the dedicated `window_provenance_train_validation.json` sidecar. It never reads combined `window_starts.json`. The sidecar has exactly train and validation entries, includes split-local timestamps/validity and start/target indices, and fingerprints the exact array shapes, dtypes, and bytes. Before output creation or model construction, the runner validates both splits against the official `validity_v2.valid_window_starts` rule and checks both array fingerprints. Existing processed arrays were not inspected; their provenance is unverified, and the runner refuses to train if the sidecar is missing or mismatched.

## Planned validation comparison and decision rule

For each seed, compare the residual-to-EWMA model against fixed alpha-0.8 EWMA and the corresponding retained B-U checkpoint on the same validation windows, directed-link columns, targets, and finite pairs. B-U checkpoints are fixed; they are not retrained. Report original utilization-ratio MAE and RMSE per seed, mean and sample standard deviation across the three seeds, and paired per-window/per-link absolute-error differences versus EWMA. Report mean paired absolute MAE difference separately from seed-level relative reductions. Seed spread is descriptive and not a confidence interval.

Primary metric: validation MAE in dimensionless computed-utilization-ratio units. Secondary metric: RMSE. The predeclared exploratory usefulness gate is met only if the new model has lower validation MAE than EWMA in at least two of three seeds and the arithmetic mean of the three seed-level relative MAE reductions is at least 1%. This is not a significance test or held-out generalization claim; validation also selects checkpoints. The B-U comparison is contextual and does not change this gate.

Individual residual-run manifests contain train/validation-only validation metrics, but the combined residual/EWMA/B-U validation comparison did not complete because evaluator preflight rejected the residual seed-42 input-array path allowlist. No test evaluation is authorized by this protocol.

## B-U contextual rerun amendment

Historical B-U runs are legacy artifacts and are not evidence for the rebuilt-array comparison. Do not edit or retrofit their manifests. Before any residual run or validation evaluator run, rerun the current `phase9_geant/train_bu_v2.py` workflow on the same newly rebuilt train/validation arrays using seeds 42, 123, and 2026 and the runner's unchanged frozen configuration. Save each run beneath the same approved rebuild root in its seed-specific fresh output directory. B-U remains contextual only: its results cannot select the residual model, change settings, or affect the residual-versus-EWMA usefulness gate above.

## Known limitations

Path-based checks are not race-free and do not guarantee directory identity against concurrent replacement. The threat model is accidental redirection or configuration error, not a hostile local user or privileged administrator. Junction handling is environment-dependent: report it as verified only if the junction tests actually create and inspect junctions. If both `_winapi.CreateJunction` and `mklink /J` fail, report the tests as skipped with both precise reasons; OS-level junction behavior then remains unverified. Cloud-backed/OneDrive reparse-point behavior is not claimed to be covered by synthetic tests.

OS junction behavior verified on 2026-10-17, method winapi, directory junctions only. Directory symlink tests were skipped with WinError 1314 (a required privilege is not held by the client).

The `--splits train,val` builder mode fits the validity cutoff from the 8,020 training matrices only, then parses the remaining holdout source matrices—including raw matrices assigned to the test period—to construct the full in-memory link series and validity mask. Those test-period observations are not used in the training or validation windows: windows are generated only for the requested train and validation bounds, and the train/validation provenance sidecar contains only those splits. This mode omits test-window starts, test arrays, and test fingerprints from its generated artifacts, but it is not a source-reader that avoids parsing test-period raw files. The recorded train/validation-only validity counts are restricted to indices before the validation end; the training cutoff is not refit on holdout data.

## Result

The validation-only evaluator compared the three fixed residual-to-EWMA checkpoints with alpha-0.8 EWMA and the three fixed B-U checkpoints on the same 1,715 validation windows, 72 directed links, and 123,480 target-link pairs per method and seed. Units are dimensionless computed utilization ratios.

| Seed | Residual EWMA MAE | Residual EWMA RMSE | EWMA MAE | EWMA RMSE | B-U MAE | B-U RMSE |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.001939910461539591 | 0.004965832701022696 | 0.001942426522904879 | 0.004983379572390751 | 0.002162289291808859 | 0.005080259091384350 |
| 123 | 0.001936592132204046 | 0.004963483238161294 | 0.001942426522904879 | 0.004983379572390751 | 0.001960981494272134 | 0.005017382046291975 |
| 2026 | 0.001937344308908096 | 0.004956950434262218 | 0.001942426522904879 | 0.004983379572390751 | 0.002020156631569476 | 0.005036975851913592 |
| Mean ± sample SD (ddof=1) | 0.001937948967550578 ± 0.000001739838053 | 0.004962088791148736 ± 0.000004602393677 | 0.001942426522904879 ± 0 | 0.004983379572390751 ± 0 | 0.002047809139216823 ± 0.000103463535710 | 0.005044872329863306 ± 0.000032173692248 |

The frozen usefulness gate **FAILED**. Residual MAE was lower than EWMA in 3/3 seeds, but the mean seed-relative MAE reduction was 0.2305135%, below the required 1.0%. B-U was contextual only and was worse than EWMA in 3/3 seeds; it did not affect the gate.

Paired absolute-error differences used a window-level percentile bootstrap: 2,000 resamples, retaining each sampled window's full directed-link error vector, with RNG seed 20261017. This accounts for within-window link pairing, but overlapping windows are treated as independent resampling units, so the resulting intervals are likely too narrow.

Evaluator source SHA-256 that produced the result artifact: `023EFDBDD65D28DF11C7ADFF722E44085AA4C76EB8B244C500C179283DB7853E`.

This is validation evidence only, not held-out generalization.
