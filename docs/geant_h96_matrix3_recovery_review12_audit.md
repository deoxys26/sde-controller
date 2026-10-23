# GEANT h96 Matrix3 recovery — review12 source candidate

Status: **candidate for independent review; not approved for training**. Use review12 only; earlier review9–review11 recovery bundles are superseded drafts and remain preserved.

The workflow is limited to the two missing runs: shuffled seed 123, then shuffled seed 2026. The six existing Matrix3 Drive backups are read-only. They could not be inspected in this local environment; the Colab preflight must validate them before training.

## Source and dataset identities

- Bundle directory: `data/transfer_bundles/geant_h96_colab_matrix_v1_review12`
- Source ZIP SHA-256: `ada8f309e07b7e5eeb3a8cf6381c8a36783fe13590594da6313eb9adbe274720`
- Source manifest SHA-256: `1c936ac6a44b35021453e1dbde05b93d420931be875e4d6e1ffdb0a626050fd3`
- Unchanged approved dataset ZIP: `df2bac044c9b9a577d5117310a99022cce402b9370e2ba483ea18b8d7045521e`
- Unchanged dataset manifest: `70f60f09b71c0cb89003ac2987af4433df0eeeb64b696845faa25d0fbadb8a95`
- Notebook: `notebooks/geant_h96_colab_matrix3_recovery.ipynb`; SHA-256 `a10a4577e471d05ad4ceabed573d7b9cf2cabeab9a4b7b603407d68361aa90af`. Its pinned source hashes match review12. It is intentionally outside the ZIP to avoid a self-referential digest.

## Independent archive audit

The independently computed ZIP and manifest SHA-256 values match the package record, and the manifest binds the ZIP digest. The ZIP contains exactly 42 manifest-listed members plus its reserved internal manifest. All listed member sizes and SHA-256 hashes match; internal/external manifests match; no extra/duplicate archive entries were found. Compared with review8 (ZIP `f4a35e77968d80069a04d027a793d118f6ecaa8ae9feb0a50e2fdd8c4736643d`), the package adds `launch_matrix_recovery.py`, `matrix_recovery.py`, and `tests/test_matrix_recovery.py`; removes nothing; and changes `package.py` to enumerate these support files. No model, trainer, scientific settings, data builder, or split definitions changed.

## Safeguards and lifecycle

Before running, the launcher verifies the approved source and dataset bundles, source import identity, exact train/validation inventory and window counts, v3 Colab memory policy, and all six existing backups. Backup validation checks exact directory/file allowlists; run and backup manifest identity; byte hashes and sizes for `run.json`, opaque `best.pt`, and opaque validation predictions; fixed protocol/configuration; train-only normalization; shared data/provenance/source identity; approved dataset identity; validation pair counts; and explicit test-access denial. Checkpoint and prediction payloads are never deserialized during old-backup validation.

The fresh local output root is `geant_horizon96_streaming_experiment_v3_repro_matrix3_recovery1`; a new UUID-named Drive destination is selected per notebook session. Existing paths are refused. Each remaining run requires an exact separate typed authorization. Per-run source, data, resource, and output identities are rechecked before training. Each completed run is copied and read-back hash-verified on Drive before registration and before the next authorization. Interrupted/partial namespaces are preserved and not resumable. Completion writes an eight-run summary whose SHA-256 is bound into the immutable completion snapshot.

## Verification

- Notebook-configured synthetic suite: **110 passed, 0 failed, 0 skipped**.
- `py_compile`, notebook JSON/code-cell AST parsing, independent archive inventory/hash checks, and `git diff --check`: passed.
- No training, inference, Drive access, real backup preflight, checkpoint deserialization, or test-split access occurred locally. The six actual backups and Colab filesystem behavior remain unverified until the user runs the notebook.
- Drive FUSE read-back does not guarantee cloud replication durability; review the persistent Drive copy after completion.

## User-controlled next steps

1. Independently review and approve the review12 source ZIP and manifest identities above.
2. Open the recovery notebook in Colab, upload the review12 source ZIP/manifest and unchanged approved train/validation dataset ZIP/manifest.
3. Confirm Drive is mounted and the old read-only backup directory is `MyDrive/SDNResearch/geant_h96_matrix3_5fe371f8aab74f42983569579477c328/runs`.
4. Run preparation, synthetic tests, and preflight. Stop if any of the six backups fail or the remaining plan differs from shuffled seeds 123/2026.
5. Only after separate approval, enable both explicit authorization flags and enter both per-run phrases. Do not retry/resume a failed namespace.
