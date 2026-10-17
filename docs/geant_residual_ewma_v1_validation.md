# GEANT v2 residual-to-EWMA validation evaluator

Status: source and synthetic tests only. The evaluator was not run, and no real arrays, checkpoints, or validation metrics were accessed in this implementation task.

`phase9_geant/evaluate_residual_ewma_v1_validation.py` compares the three fixed residual-to-EWMA checkpoints, fixed EWMA (`alpha=0.8`), and each seed's retained B-U checkpoint on the same finite validation window/link pairs. It reads only `X_validation.npy`, `y_validation.npy`, the train/validation-only provenance sidecar, and explicitly named run manifests/checkpoints. It does not train or write checkpoints.

Metrics are MAE and RMSE in dimensionless computed-utilization-ratio units, accumulated in float64. A shape mismatch or any nonfinite input/prediction is fatal; no pairwise intersection or imputation is used. Across seeds, means and sample standard deviations use `ddof=1`.

For each seed and candidate (`residual_ewma` and B-U), paired absolute-error difference is `abs(candidate - target) - abs(EWMA - target)`; negative values favor the candidate. The percentile interval uses a paired bootstrap of validation windows with replacement, retaining each selected window's full directed-link error vector. The frozen default is 2,000 replicates, 95% interval, RNG seed 20261017. The report includes per-link mean paired differences and the aggregate difference.

The usefulness gate applies only to residual-to-EWMA versus EWMA: strict lower MAE in at least two seeds and arithmetic mean seed-level relative MAE reduction of at least 1%. Equality is not a win. B-U is contextual and cannot affect the gate. This is validation evidence, not a significance claim or held-out generalization.

The evaluator fails closed unless every residual and B-U run manifest binds the same train/validation provenance, all four permitted train/validation array fingerprints, split/window metadata, ordered directed edges, configuration/source identity, normalization provenance, and checkpoint identity. Source-hash drift is rejected by default; `--allow-source-drift` is an explicit non-strict opt-in and is recorded in the output. Historical B-U manifests were not inspected for this documentation update and are not retroactively trusted or attested; only newly generated runs with complete provenance may qualify. No output is overwritten; results are written only to the fresh `neural_ewma_residual_v1_validation_eval` directory.
