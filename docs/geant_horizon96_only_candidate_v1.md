# GEANT horizon-96-only candidate protocol v1

**Status:** isolated implementation prepared for review; no model training or real-data validation was run as part of this implementation. This is a separate amendment, not a rewrite of the two-horizon `geant_protocol_v2_horizon_design.md`.

## Decision and scope

The prior fixed train-only Ridge screen reported an 8.692% relative EWMA-MAE reduction at horizon 96 and classified that horizon as candidate-design review only. Horizon 4 was classified STOP. This amendment admits only h=96 to candidate preparation. The Ridge result is screening evidence, not evidence that a neural model will improve forecasting.

The experiment protocol ID is `geant-horizon96-only-candidate-v1`. Its history is 672 15-minute samples; its sole target is `u[t+96]`, 24 hours after origin `t`. Official train `[0,8020)` and validation `[8020,9740)` boundaries remain unchanged. History and target indices must lie wholly inside their assigned split and satisfy the existing h=96 window/provenance contract.

**Horizon 4 is excluded by construction.** The candidate model is instantiated with `horizons=(96,)`, has one residual output channel, and returns `[batch,1,72]`. Its data reader requests only `history_train_h96.npy`, `target_train_h96.npy`, `history_validation_h96.npy`, and `target_validation_h96.npy`. The training/scaler/evaluation interfaces accept one history array and one target array per split, not a horizon-indexed mapping. No h=4 target, prediction, metric, loss, ranking, or checkpoint-selection value enters this protocol. The underlying dataset metadata still records the original builder's `(4,96)` window definition; that is provenance, not an h=4 model objective.

## Model, features, and objective

Use the existing `SpatioTemporalLinkForecaster` unchanged, but construct it only with `history_length=672` and `horizons=(96,)`. Each directed link's 672 utilization values feed a shared GRU temporal encoder. The graph encoder uses a fixed line-graph adjacency between directed links whose endpoint sets intersect (including reverse directions); topology, identity, and shuffled controls use the same model class, learned modules, parameter count, input arrays, and schedule. The forecast head predicts one correction to the fixed EWMA level.

For the five latest history observations, define `S[t,e] = Σ(k=0..4) w[k] u[t-4+k,e]`, with `w[k] = 0.8 × 0.2^(4-k) / Σ(j=0..4) 0.8 × 0.2^(4-j)`. The h=96 model's two seasonal residual channels are exactly:

1. `u[t+96-96,e] - S[t,e] = u[t,e] - S[t,e]` (history index 671).
2. `u[t+96-672,e] - S[t,e] = u[t-576,e] - S[t,e]` (history index 95).

Both sources are in the 672-step observed history. The implementation derives them internally from that history; no target or future value is accepted by the model's forward method. The zero-initialized residual head makes the initial h=96 forecast exactly the alpha-0.8 five-step EWMA for every adjacency mode.

Let `r[t,e] = u[t+96,e] - S[t,e]`. Training minimizes `SmoothL1(estimated residual, r; beta=0.001)` in original computed-utilization-ratio units. Inputs and h=96 targets are standardized with one affine scaler fit only on h=96 training histories and h=96 training targets; validation uses that frozen scaler. Validation is in evaluation mode without gradient updates. The checkpoint and early-stopping criterion is the h=96 validation residual SmoothL1 objective only. No h=4 value can enter it.

Fixed candidate settings: seeds 42, 123, and 2026; maximum 20 epochs; patience 4; Adam, learning rate 0.005; batch size 32; hidden size 16; alpha 0.8 over five observations; SmoothL1 beta 0.001. The final residual head starts at zero. These are candidate-preparation settings, not a claim of optimality.

## Graph controls and provenance

The three fixed adjacency modes are topology, identity, and shuffled. The shuffled adjacency uses seed `20261019` and the existing deterministic degree-preserving line-graph rewire. The run record stores the ordered 72-edge digest, realized normalized adjacency SHA-256, mode, and shuffle seed. Directed-edge order is taken from the validated dataset metadata and compared to the current parsed local topology before any h=96 array is loaded. The topology contract is 22 nodes, 36 undirected links, 72 interleaved directed links. The local topology digest identifies local file bytes only; upstream topology/data provenance remains unverified.

For every future run, the training data arrays are memory-mapped read-only and accessed through exact allow-listed h=96 filenames. Train-only scaler accumulation and fingerprints operate in bounded chunks; mini-batches are copied to tensors as needed. Validation uses the same 72 link columns and exact stored h=96 window ordering. Nonfinite values, mismatched shapes, bad provenance, unexpected artifact names, split crossing, source drift, or an existing output directory fail closed. The output namespace is the new `data/processed/geant_horizon96_only_v1/`, with distinct mode/seed run paths; no historical run path is reused.

Run manifests are written through same-directory atomic replacement and track incomplete/failed/complete lifecycle, training-only normalization, config/data/source fingerprints, topology/adjacency identity, epoch histories, validation-only selection, checkpoint hash, strict reload, and runtime. Source hashes describe listed file bytes and do not prove which already-imported Python code executed. Filesystem path checks are path-based and are not race-free against a hostile concurrent writer or a privileged local actor.

The training API and CLI are disabled by default and require both an explicit run action and an explicit horizon-96 candidate authorization flag, plus explicit data/repository roots, seed, and adjacency mode. The module reuses scaler/metric helpers from the historical joint-experiment module, but calls neither its Ridge screen nor its joint-training entry point. Importing the module does not train. The synthetic CPU benchmark is a separate bounded action and uses generated tensors only. This document does not authorize training; a separate approval is required after review.

## Candidate interpretation and evaluation rules

For a later separately authorized controlled evaluation, compare each seed/mode against flat EWMA (`S[t]`), persistence, same-time-yesterday, and same-time-last-week on identical h=96 validation target windows and directed links. The original h=96 Ridge screen is a train-only screening comparator; do not rerun or retune it under this protocol. Report h=96 validation SmoothL1 for checkpoint selection separately from original-unit MAE/RMSE. Keep baseline comparison, per-seed results, and seed mean/sample standard deviation explicit. The three topology modes are controls, not independent candidate methods selected on validation.

No validation/test result may alter architecture, seeds, thresholds, optimizer, loss, preprocessing, masks, or protocol. Do not claim improvement without a future controlled evaluation. The official test split remains inaccessible; any final test evaluation would require a separate frozen protocol and explicit authorization.

## Integrity and limitation status

The implementation is intended to enforce: h=96-only arrays; exact train/validation bounds; matching data-builder source/config identities; payload fingerprints; 72-link order and local topology identity; train-only scaling; one h=96 loss/selection channel; and a dual explicit training gate. Synthetic tests exercise these properties without opening project datasets, checkpoints, or manifests.

Not established here: actual build/run feasibility on real arrays, actual training quality, full-run peak memory/duration, OneDrive or Windows concurrent-path race guarantees, authenticated upstream GEANT/topology provenance, or test-set generalization. The Ridge screen's 8.692% result only supports candidate preparation. No result in this document establishes that this GNN beats EWMA, Ridge, or DG-STMTL.
