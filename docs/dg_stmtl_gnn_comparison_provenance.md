# DG-STMTL / GNN h96 comparison identity rule

The comparison treats protocol-specific configuration and storage fields as
run-specific, not as shared data identity. For example, the original GNN may
record the dataset protocol in `configuration` and per-file hashes in
`provenance.array_file_records`; DG-STMTL records the protocol in `provenance`
and its permitted file hashes in `permitted_array_file_hashes`. The comparator
normalizes only these documented layouts.

Acceptance still requires all of the following to match the reviewed
train/validation bundle and one another: the six permitted train/validation
series, validity-mask, and timestamp file hashes; train and validation h96
window-index hashes; protocol and split boundaries; 3,924/953 window counts;
dataset metadata identity; the 72-edge ordered-link digest; exact scaler mean,
scale, and population count; and the shared scaler and float64 metric source
hashes. The GNN's Colab bundle and bundle-manifest hashes must match the
approved train/validation package. Every metric must retain the computed
utilization-ratio units and exactly 68,616 evaluated pairs. EWMA values remain
as recorded and are checked with the pre-existing absolute tolerance of
`1e-9` and zero relative tolerance.

The source-defined scaler is the same function for both methods. It streams
the population formed by every scalar in each training history occurrence plus
each h96 training target, counting overlapping history occurrences repeatedly;
validation values are not inputs to the scaler. The differing config strings
are accepted only from the two exact, source-reviewed descriptions, while the
manifest's actual scaler record must carry the implementation's exact fit-scope
label and identical numeric parameters.

Model-specific values are intentionally not compared for equality: model
architecture/config hashes, checkpoint hashes, best-validation-loss field
names and values, and validation-prediction hashes. Each run's own config hash,
checkpoint name/hash syntax, completion status, test denial, and strict reload
flag are still independently validated. Prediction hashes should differ
between forecasting methods.

The `--diagnostic-only` mode uses only the explicitly supplied `run.json`
manifests and compiled-in identities copied from the verified train/validation
bundle manifest. It opens no arrays, predictions, checkpoint tensors, or test
artifacts and writes no report files. A passing diagnostic establishes
manifest-declared compatibility; it does not independently prove that the
recorded run actually consumed those bytes.

In Colab, after uploading this comparator source, mount Drive and pass the same
eight GNN and three DG manifest paths intended for reporting, each as a
separate `--gnn-run-json` or `--dg-run-json` argument. Include
`--allow-incomplete-gnn-matrix --diagnostic-only`. Do not use globbing or
directory discovery; the command accepts only explicit paths whose basename
is `run.json`. A successful check prints the normalized identity and
`artifacts_written: false`; a nonzero exit with `status: incompatible` means
the reporting run must not proceed until the reported mismatch is resolved.
