# GEANT h=96 streaming data and bounded smoke preparation

## Scope and protocol boundary

This isolated implementation prepares one h=96-only dataset format and a
separate, explicit two-epoch smoke path. It does not change the existing joint
h=4/h=96 builder, full candidate trainer, historical v1 results, model
architecture, split boundaries, or the scientific gate. Horizon 4 remains
stopped and is absent from this format. Importing either new module has no
data or training side effect. Real building and smoke training were **not run**
in this implementation stage.

The fixed series intervals are train `[0,8020)` and validation `[8020,9740)`.
The source anchor is the documented SNDlib local-time filename convention
starting at `2005-05-04 15:00`; the local archive checksum/upstream release is
not independently authenticated. The topology contract is 22 nodes, 36
undirected links, and 72 directed edges in the existing interleaved order.
Its local file hash is only a local-byte identity; upstream topology provenance
remains unverified. Utilization means shortest-path-computed link load divided
by the assumed 40,000 Mbps capacity, not measured telemetry.

## Builder and boundedness

`phase9_geant/horizon96_streaming_data.py` is a new, separate entry point. It
uses the audited one-scan `expected_split_paths_for_splits(..., ("train",
"validation"))` selector. That selector necessarily enumerates directory-entry
names, including names outside the requested splits, but performs selected-file
metadata checks only for the 8,020 train and 1,720 validation paths. It does not
stat, resolve, hash, open, or parse the nonselected test-period paths. No test
matrix contents or test array paths are used. The builder never invokes the
legacy all-series parser or the two-horizon builder.

The training cutoff is five times the median total demand over nonempty
training matrices only. Training matrices are parsed in a first pass to fit
that cutoff and a second pass to compute link utilization; each matrix is
discarded before the next. Validation matrices are parsed once with the frozen
training cutoff. Each selected source file is fingerprinted and checked again
around parsing; training fingerprints must match between both passes.
Per-matrix parsing is capped at 8 MiB, and the builder checks 512 MiB minimum
OS-available RAM and free-disk reserves before parsing. The output is compact
`float32 [T,72]` utilization plus a validity vector and timestamp vector for
each permitted split. Invalid utilization rows remain NaN. No `[N,672,72]`
history stack is built or saved.

Window records store global indices and are enumerated using the existing
index contract: history `[origin-671,...,origin]`, target `origin+96`; all
history observations and the target must be valid, timestamps must be
consecutive from history start through target, and both must remain within the
same official split. Intermediate observations are not inputs or targets and
retain the existing contract's validity treatment. The emitted data format has
no h=4 artifacts and no test arrays.

The lazy dataset slices one `[672,72]` history and one `[72]` target from a
read-only memory map per sample. Scaling is fit only from valid h=96 training
windows and their targets. To match the existing candidate method
`fit_h96_training_scaler`, each overlapping history occurrence is counted
again, and each training target occurrence is included; row multiplicities
allow the same population moments to be computed from the compact series
without stacking windows. Validation values never enter the scaler. The model
derives h=96 seasonal residual features causally from the history: `u[t,e] -
EWMA[t,e]` and `u[t-576,e] - EWMA[t,e]`; it receives no target tensor in its
forward method.

## Smoke-only execution contract

`phase9_geant/horizon96_smoke.py` contains a separate fixed configuration:
seed 42, topology adjacency only, one h=96 output channel, at most two epochs,
batch size 2, zero loader workers, Adam at 0.005, hidden size 16, alpha-0.8
five-step EWMA, and residual SmoothL1 beta 0.001 in original utilization-ratio
units. The operational smoke uses a fixed maximum of 32 evenly spaced valid
training windows and 32 evenly spaced validation windows, with selection rule
and available/selected counts recorded in its manifest. This subset is only a
bounded interface/checkpoint test; its validation numbers are not full
validation evidence and cannot support scientific comparison. Validation
residual SmoothL1 alone selects the checkpoint within that fixed subset.
Training normalization is fit over the full permitted train-window
history/target population and frozen for smoke validation. Checkpoints are
atomically written, byte-hashed, strictly reloaded, and only then can the
manifest be marked complete.

There is no default training action. Future smoke execution requires both
`--run-smoke` and `--authorize-h96-smoke`, plus explicit data and repository
roots. The output namespace is the previously unused dedicated child
`data/processed/geant_h96_smoke_stage20_v1/seed42_topology`; any existing or
partial namespace causes refusal. It cannot be used for other seeds, graph
modes, epoch counts, or full candidate training. A synthetic benchmark is a
separate action and accepts no data roots or authorization options.

The resource preflight uses OS-available RAM (`psutil.virtual_memory().available`),
which is not the process's peak working set or total committed memory. It
requires a conservative 640 MiB smoke peak estimate plus 384 MiB reserve, at
least 1 GiB OS-available RAM, and 512 MiB disk headroom. These are refusal
guards, not guarantees that Windows, Python, Torch, or the filesystem will not
allocate more. The synthetic benchmark records observed process RSS/peak field
when available and optimizer-step timing. Its full-epoch projection is a
linear-throughput estimate only, excludes validation and I/O, and is not an
observed training runtime. It also reports an estimate for the fixed 32/32,
two-epoch smoke workload; neither estimate is an observed real-data runtime.
The separate read-only resource query during this implementation observed
2,638,966,784 bytes OS-available RAM and 82,071,756,800 bytes free on the
destination volume. Both builder and smoke preflight thresholds passed at
that moment; these values are volatile and do not guarantee future headroom.

## Integrity and known limitations

- Tests use synthetic arrays/tensors and temporary paths only. The real builder,
  smoke runner, and benchmark CLI were not invoked as real-data workflows.
- Source hashes identify listed file bytes and are checked before/after; they do
  not prove which source Python imported or executed.
- Source discovery sees directory-entry names outside train/validation as part
  of the one bounded scan. It does not perform per-file operations or content
  reads on nonselected names. The output manifest records this distinction.
- Path checks are path-based and not race-free against concurrent replacement,
  privileged writers, hard-link manipulation, cloud hydration, or all Windows
  reparse semantics. The threat model is accidental redirection, not a hostile
  local administrator. No OS junction behavior is claimed as newly verified by
  this implementation.
- Upstream source archive and topology provenance are unverified. The local
  topology hash is not upstream authentication.
- The full real data build and smoke execution still require separate explicit
  authorization. A two-epoch smoke is operational evidence only, not a
  predictive-quality result or generalization claim.

## Synthetic verification

The dedicated tests cover h=96 index counts/alignment, invalid and
nonconsecutive windows, lazy sample extraction, cross-split rejection,
window-occurrence scaler equivalence to the materialized candidate scaler,
training-only scaler API, filename allowlists, bounded resource checks,
authorization-before-path-access, existing-output refusal, and a synthetic
CPU optimizer-step benchmark using `[2,672,72]` tensors. See the final task
report for exact commands and observed test outcomes.

In the final bounded synthetic run, three measured optimizer steps were
0.3184, 0.3318, and 0.3288 seconds (median 0.3288 seconds). Process RSS was
278,065,152 bytes before model steps and 349,245,440 bytes afterward; the
platform-reported peak working set/RSS field was 386,576,384 bytes. The
synthetic full-epoch extrapolation is 1,192 seconds for 7,253 assumed windows
at batch two, excluding validation and I/O. The fixed 32-train/32-validation,
two-epoch smoke extrapolation is 21.04 seconds. These are estimates from
synthetic workload measurements, not real-data training observations.

Pytest's built-in `tmp_path` had previously failed in the virtualized AppData
Temp mount while attempting its numbered-directory `current` symlink. The test
conftest now creates a unique direct child with `tempfile.mkdtemp`, avoiding
that symlink dependency. Test commands used a short, unique basetemp under the
external Codex visualization scratch root; using a deeply nested/long basetemp
caused five pre-existing synthetic fixture paths to exceed Windows path length.
The final relevant suite passed with the short external path. This is not a
claim about OS junction handling.
