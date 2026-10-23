# GEANT h96 Matrix3 recovery — source candidate audit

Status: **candidate for independent review; not approved for training**.

This bundle adds a recovery-only workflow for exactly two runs: shuffled seed 123, then shuffled seed 2026. It never resumes a partial recovery, and it will not authorize the second run until the first run's Drive backup is copied and read-back verified. The existing six-run Matrix3 backups are not included in or modified by this source bundle. Their actual contents could not be inspected from this local session; the Colab preflight validates them read-only before any training.

## Bundle identities

- Source bundle: `data/transfer_bundles/geant_h96_colab_matrix_v1_review10/geant_h96_colab_source_v1.zip`
- Source ZIP SHA-256: `0f9aeaff442a5b77e449992770607e99a33038450f9e2b2eebc12c69504b9f31`
- External manifest: `data/transfer_bundles/geant_h96_colab_matrix_v1_review10/source_snapshot_manifest.json`
- Manifest SHA-256: `6356d15eacdf8322baeda0b5818bee59e0e35d5ec59da3037d12e1d3de17b6bc`
- Approved unchanged train/validation dataset package: ZIP `df2bac044c9b9a577d5117310a99022cce402b9370e2ba483ea18b8d7045521e`; manifest `70f60f09b71c0cb89003ac2987af4433df0eeeb64b696845faa25d0fbadb8a95`.
- Recovery notebook: `notebooks/geant_h96_colab_matrix3_recovery.ipynb` (separate operator artifact; not embedded in the source ZIP, avoiding self-referential hashing).

## Independent archive audit

The source ZIP and manifest were hashed separately after package generation. The ZIP's bound archive digest equals its independently computed SHA-256. All 42 external entries matched their raw archive-member SHA-256 and byte counts; the archive contained exactly those entries plus `PORTABLE_SOURCE_MANIFEST.json`, with no duplicate/case-colliding members. The internal and external entry inventories matched. These checks establish package integrity, not source approval or runtime behavior on Colab.

Compared with the previously reviewed Matrix3 source ZIP `f4a35e77968d80069a04d027a793d118f6ecaa8ae9feb0a50e2fdd8c4736643d` (manifest `53789ea2deca2416781bd79c957e2598596ed7a55dd95555f932a62015d7bb15`): three files were added—`launch_matrix_recovery.py`, `matrix_recovery.py`, and `tests/test_matrix_recovery.py`; no files were removed; `package.py` changed only to inventory/package the recovery launcher, helper, and tests. No model, dataset builder, optimizer, loss, split, or training implementation was changed in this recovery work.

## Guard behavior

The launcher verifies source/dataset package identities, the v3 Colab memory policy, exact train/validation dataset inventory, and train/validation window/provenance identity before the run loop. It checks the original six backup directories and their exact four-file inventories (`run.json`, opaque `best.pt`, opaque `validation_predictions.npy`, `backup_manifest.json`), validates byte hashes/readback records, fixed h96 scientific configuration, train-only scaler provenance, shared data/source identities, strict checkpoint reload, validation coverage, and explicit test-access denial. Only the two shuffled runs are planned. No code path selects a test artifact.

Each remaining run requires an exact, separate typed phrase. A fresh local output namespace and a unique, previously nonexistent Drive namespace are required; existing namespaces are refused. After each run, the only copied payloads are its run manifest, checkpoint, and validation predictions. The copy is hash-readback verified before registration and before the next authorization. A combined eight-run JSON summary and its SHA-256 are bound to the final immutable lifecycle snapshot.

## Synthetic verification

Notebook-configured suite (local synthetic-only execution): 110 passed, 0 failed, 0 skipped. `py_compile`, notebook JSON/code-cell AST parsing, and `git diff --check` passed. No training, real-data preflight, Drive access, checkpoint deserialization, or test-split access occurred. This does **not** verify that the six actual Drive backups satisfy the schema; that remains a required Colab preflight result.

## Colab use after independent review

1. Upload the review10 source ZIP and manifest, plus the already-approved train/validation dataset ZIP and manifest when prompted.
2. Review this audit and independently approve/pin the source ZIP and manifest hashes above.
3. Run notebook cells in order. Confirm the mounted backup location is the stated original Matrix3 Drive folder. Stop if any one of the six backup validations fails.
4. Confirm the preflight lists exactly shuffled seeds 123 and 2026 and a fresh Drive destination. The preflight itself does not train.
5. Only after separate authorization, set both notebook flags to `True`, execute the final cell, and enter the displayed per-run phrases. Never rerun a failed/interrupted recovery namespace.

The recovery is not yet approved for training. Colab/Drive behavior, the real six backups, and cloud durability remain unverified here.
