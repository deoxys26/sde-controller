# GEANT h96 Matrix3 recovery — Review18 candidate audit

Status: **unapproved candidate**. Independent review and a successful no-training Colab preflight are required before any separate training authorization. This package is restricted to the two remaining runs: shuffled seeds 123 and 2026.

## Source-order and data-identity checks

The recovery path now registers `RUN_SOURCE_FILES` through `_verified_recovery_inputs()` before the first call to `trainer._verified_inputs()`. This ensures that recovery helpers are part of the source identity observed during data preflight. The regression test asserts that both recovery source paths and the migration runtime files are present at the moment the verifier is called.

For each remaining run, `_execute()` calls the train/validation-only verifier again and compares the resulting identity with the initial identity using `launch_matrix._same_input_identity()`. A difference raises `RecoveryError` before `_planned_run()` or `trainer.train_one()`. A synthetic regression computes SHA-256 values from two different fabricated train-array byte payloads, changes the `utilization_train.npy` fingerprint, and verifies the run fails before training. No GEANT array or observation is used.

## Synthetic verification

- Focused recovery tests: **18 passed**.
- Complete notebook-configured synthetic suite: **123 passed** across portability, matrix launcher/workflow, recovery, notebook staging, streaming experiment, v2 streaming experiment, and spatio-temporal model tests.
- `py_compile` passed for the recovery launcher, session staging helper, and their test modules.
- Every notebook code cell parsed successfully; the JSON pins were parsed and matched against Review18, the fresh preparation and Drive namespaces contain `review18` plus a UUID, and authorization flags remain false by default.
- Recovery launcher `--help` exited 0. `git diff --check` exited 0; only existing line-ending warnings were printed.

Tests used a fresh pytest basetemp under the user's external local temp directory. The six completed Drive backups were not visible to this local session and were not accessed; they must be verified read-only by the notebook in Colab before any authorization. Colab runtime execution and Drive FUSE durability remain unverified locally.

## Review18 files and independent archive verification

- Source ZIP: `data/transfer_bundles/geant_h96_colab_matrix_v1_review18/geant_h96_colab_source_v1.zip`
  - SHA-256: `81aa2b765b72e4eab00822010414d2446dfa2e1b8164891cc232dbc146172f20`
- Source manifest: `data/transfer_bundles/geant_h96_colab_matrix_v1_review18/source_snapshot_manifest.json`
  - SHA-256: `592715fa389d53624bdce2749267d669ed40c1d209369f0e1d255e2baf8dffee`
- Independent check: 44 manifest entries plus the embedded manifest (45 ZIP members), exact inventory, no duplicate/case-colliding names, no directory/symlink entries, and all member lengths and SHA-256 values matched. The source archive contains no array or checkpoint files. The approved dataset ZIP and manifest pins are unchanged.
- Recovery notebook: `notebooks/geant_h96_colab_matrix3_recovery.ipynb`
  - SHA-256: `0cfee46df6f8776a534d6878df3120414dff7984e329f82f0c94d048a7429ade`

Review15, Review16, and Review17 were not overwritten. Review18 is not marked approved by the packaging tool. No training, real-data preflight, inference, dataset rebuild, Drive upload, or official test-split access occurred.

## Next step

An independent reviewer should review this audit and the exact Review18 hashes. Only after that review should the user upload the Review18 source ZIP/manifest and the already-approved dataset ZIP/manifest, execute the notebook’s synthetic suite and read-only backup/preflight checks in Colab, and separately authorize each remaining run if all checks pass. The notebook and launcher still cannot confer authorization automatically.
