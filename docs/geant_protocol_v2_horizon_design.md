# GEANT multi-step forecasting protocol v2 — design only

**Status:** proposed design; no builder or model code is authorized or implemented here. The v1 experiment remains frozen and closed.

## Research question and data boundary

At horizons of 4 steps (1 hour) and 96 steps (1 day), does a seasonal or graph-based residual forecaster improve on a fixed EWMA forecast of computed directed-link utilization from GEANT traffic matrices? Sampling is every 900 seconds. The v1 builder emits only a horizon-1 target; these horizons require a separately reviewed builder version and a new, previously nonexistent output root, for example `data/processed/geant_validity_horizons_v2_trainval_20261019/`. The builder must emit train/validation data only and select train/validation source files before parsing; it must not parse test-period matrices, emit test arrays, or create test-window metadata. Test remains sealed.

Proposed boundary convention: forecast origin `t` is the last observed step. Inputs and all seasonal source observations must be available at `t`; the two targets are `u[t+4]` and `u[t+96]`. Use a 672-step history so both daily and weekly seasonal references are constructible. Require history and target to be valid and within the same official split, with no invalid or nonconsecutive steps in the history. Do not require future intermediate values to be valid: they are not model inputs. Do not let a history or target window cross an invalid gap or train/validation boundary. The existing official train and validation blocks remain unchanged; no test observations are accessed.

## Baselines and identical comparison support

For link `e`, let the five most recent observed values be `u[t-4,e], …, u[t,e]`. With fixed `alpha=0.8`, define the v1 normalized EWMA level

`S[t,e] = sum(k=0..4) w[k] u[t-4+k,e]`, where `w[k] = alpha*(1-alpha)^(4-k) / sum(j=0..4 alpha*(1-alpha)^(4-j))`.

For either horizon `h ∈ {4,96}`, the horizon-h EWMA forecast is explicitly the flat carry-forward `EWMA[t+h|t,e] = S[t,e]`. It does not update with unavailable future observations or recursively invent future inputs. Compare against:

- Persistence: `u[t,e]` at either horizon.
- Same-time-yesterday: `u[t+h-96,e]`, only when that source is observed, valid, and in the same split.
- Same-time-last-week: `u[t+h-672,e]`, under the same availability and split rules.

Seasonal references are unavailable rather than imputed when their source is missing, invalid, nonconsecutive, or outside the split. Report each baseline's coverage over all canonical horizon targets. For the primary paired comparison, evaluate every method on the exact same `(origin, horizon, directed link)` intersection where the target and every compared prediction are finite and valid; report the intersection count and method-specific omitted coverage. No zero fill, clipping, or selective per-method mask is allowed. Primary metric is MAE in dimensionless computed-utilization-ratio units; RMSE is secondary. Report horizons separately.

## Predeclared train-only signal diagnostic — before any model

Before training any neural model, evaluate one fixed pooled Ridge residual screen separately at each horizon. The residual target is `u[t+h,e] - S[t,e]`. At origin `t`, use only fixed features available by `t`: (1) valid daily and weekly seasonal forecasts minus `S[t,e]`; (2) for links adjacent in the line graph (links sharing an endpoint, including reverse directions), the mean neighboring `u[t,j]-S[t,j]`; and (3) the mean neighboring four-step change `u[t,j]-u[t-4,j]`. These definitions and topology-derived neighbor sets are frozen before reading outcomes. Pool link-origin examples without link IDs. Standardize features and residual targets using fit-period training statistics only; fit Ridge with fixed `alpha=1.0`, no tuning.

Use an internal chronological split of the official training block: first 70% of eligible origins for fitting, final 30% for the screen holdout, with a 768-step embargo between them (672-step maximum history plus 96-step maximum target horizon). No validation or test observation may choose features, thresholds, lags, or coefficients. Report the Ridge-plus-EWMA relative MAE reduction `(MAE_EWMA - MAE_RidgeEWMA)/MAE_EWMA` on identical finite holdout pairs, plus counts and residual R-squared as a secondary diagnostic. Decision thresholds, fixed now for each horizon:

- Below 0.5% relative MAE reduction: stop; do not build forecasting models for that horizon.
- At least 1.0%: the diagnostic supports proceeding to the candidate-design review for that horizon; it does not itself authorize training.
- From 0.5% up to but not including 1.0%: report the result and ask for a decision; do not proceed automatically.

## Candidate models, only if the diagnostic supports them

1. **Per-link residual MLP:** shared per-link MLP using only that link's 672-step history and the predeclared seasonal residual features; predict two residuals (horizons 4 and 96), then add each to its corresponding flat EWMA forecast. This is the non-graph learned model and the minimum comparator.
2. **Line-graph residual model:** links are line-graph vertices; fixed adjacency connects directed links sharing a router, including reverse-direction links. The model uses the same per-link temporal inputs and outputs as the MLP, with a residual head over EWMA. It must not be described as useful spatial learning unless it beats controls.

Line-graph controls are (a) identity adjacency, which disables cross-link message passing, and (b) a deterministic degree-preserving shuffled adjacency with a recorded fixed permutation/seed and verified node set, direction mapping, and degree sequence. The principal model to beat is the per-link residual MLP; EWMA remains the primary operational baseline. Hold edge/link-to-target correspondence fixed in every control.

## Frozen candidate-training and evaluation rules

Only after diagnostic support and explicit review: use seeds `42, 123, 2026`; train one multi-horizon model per candidate/seed, maximum 20 epochs, early-stopping patience 4, Adam at `0.005`, batch size 32. Fit all scalers on training data only. Validation runs in evaluation mode with gradients disabled and is used only for checkpoint selection and early stopping; test is never used for model or setting selection. Use SmoothL1 with fixed `beta=0.001` in original residual-utilization units (report its exact implementation and units); this is robust to occasional large residuals while retaining a local quadratic penalty. Initialize the final residual layer to zero so initial predictions equal the EWMA baseline. Save protocol/config/source/data provenance before execution and checkpoint identity after finalization; refuse existing output directories.

For each horizon, the validation usefulness gate is: (1) candidate MAE is at least 1% lower than EWMA in at least two of three seeds; and (2) the paired candidate-minus-per-link-MLP MAE difference has a 95% contiguous-block bootstrap interval wholly below zero. If the candidate is the MLP, condition (2) is not applicable; its comparison is against EWMA plus the same seed-level rule. The line-graph model must also outperform identity-adjacency and shuffled-adjacency controls under the predeclared paired comparison. A lower training loss alone is not success.

Use a paired moving-block bootstrap over chronological forecast origins, retaining all 72 link errors per sampled origin: 2,000 replicates, fixed RNG seed `20261019`, primary block length 96 origins (one day); report a sensitivity analysis at 192 origins (two days). Resample within the validation block only and preserve temporal order inside each sampled block. These intervals quantify validation uncertainty, not held-out generalization; overlapping windows and the limited number of long blocks remain limitations.

## Required integrity tests before any build or training

- Builder selects and validates train/validation source filenames before parsing; test filenames and test split tokens are rejected by every read allowlist.
- Versioned output root is new and outside legacy roots; no overwrite or automatic cleanup.
- Exact horizon-target alignment (`t+4`, `t+96`), history length 672, timestamp continuity, invalid-gap behavior, and split-boundary containment.
- Seasonal source index arithmetic, source availability, and no future reads for both horizons; unavailable sources remain unavailable, never imputed.
- Train-only fitting of all scalers and Ridge/model statistics; changing validation fixtures cannot change fitted values.
- Identical target/link ordering, masks, and paired counts across baselines, models, and controls; coverage is explicit.
- Deterministic degree-preserving line-graph shuffle; identity control has no cross-link messages; no-op control preserves predictions.
- Validation checkpoint selection uses validation loss only, gradients are disabled during validation, and test paths are rejected before any file access.
- Provenance fingerprints, source/configuration hashes, lifecycle states, checkpoint hashes, and atomic manifests; altered or missing identity fails closed.
- Bootstrap uses contiguous origin blocks and fixed seed/lengths; synthetic metric tests verify signs, paired differences, interval logic, and denominators.

## What would make us stop

- Either horizon's train-only Ridge screen is below 0.5%: stop that horizon with a negative signal-screen result; no model build.
- The screen is between 0.5% and 1.0%: pause for review; no automatic build.
- Any leakage, provenance, target alignment, path, coverage, or reproducibility check fails: stop before training and repair only under a separately reviewed protocol.
- After approved validation experiments, neither candidate clears the predeclared usefulness gate, or line-graph results fail to beat identity/shuffled controls: record a negative result and stop; no tuning, reseeding, or repeated validation search.
- No final test evaluation is authorized by this design. The test split stays sealed until a separately approved, single evaluation of one frozen model/protocol; no test-driven selection or reruns.

**Interpretation limit:** a negative result applies to these horizons, inputs, baselines, and thresholds only. It does not establish that seasonal or graph methods cannot work on other data or objectives.
