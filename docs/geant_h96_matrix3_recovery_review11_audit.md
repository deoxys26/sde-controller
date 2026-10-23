# GEANT h96 Matrix3 recovery — review11 source candidate

Status: **candidate for independent review; not approved for training**. This review11 bundle supersedes the earlier local review9/review10 drafts; use review11 only.

The recovery plan is restricted to shuffled seed 123 followed by shuffled seed 2026. The six reported completed Matrix3 backups remain read-only and are not included in the source package. Their actual Drive contents could not be inspected from this local session. The Colab preflight is designed to validate the exact six folders and their artifact hashes before any training, and must pass in the user’s mounted Drive session.

## Package identity

- Source ZIP: [geant_h96_colab_source_v1.zip](/C:/Users/Admin/OneDrive/Desktop/DN_proj/data/transfer_bundles/geant_h96_colab_matrix_v1_review11/geant_h96_colab_source_v1.zip)
- ZIP SHA-256: `29b4546918b40917640f19cb0580fce45d2a197a1c9f333d3374f09ba9534a04`
- External source manifest: [source_snapshot_manifest.json](/C:/Users/Admin/OneDrive/Desktop/DN_proj/data/transfer_bundles/geant_h96_colab_matrix_v1_review11/source_snapshot_manifest.json)
- Manifest SHA-256: `934fc07f1cc5a9910ea705f7bc28af8121a9008bfca8768a5d4a221039a80dcb`
- `SOURCE_HASHES.txt` is in the same bundle directory.
- Dataset bundle is unchanged: ZIP `df2bac044c9b9a577d5117310a99022cce402b9370e2ba483ea18b8d7045521e`; manifest `70f60f09b71c0cb89003ac2987af4433df0eeeb64b696845faa25d0fbadb8a95`.
- Operator notebook: [geant_h96_colab_matrix3_recovery.ipynb](/C:/Users/Admin/OneDrive/Desktop/DN_proj/notebooks/geant_h96_colab_matrix3_recovery.ipynb). It is outside the ZIP to avoid a self-referential source hash; its pinned ZIP/manifest hashes match review11.

## Independent archive audit

Independent raw-byte SHA-256 of the source ZIP and external manifest matched the package record. The manifest binds the ZIP digest. All 42 manifest entries matched archive-member byte lengths and hashes. The archive has exactly the 42 entries plus the reserved internal manifest (43 members total), no duplicate/case-colliding names, and the internal and external entry lists are identical.

Compared with the earlier review8 Matrix3 candidate (ZIP `f4a35e77968d80069a04d027a793d118f6ecaa8ae9feb0a50e2fdd8c4736643d`, manifest `53789ea2deca2416781bd79c957e2598596ed7a55dd95555f932a62015d7bb15`), the recovery source adds `launch_matrix_recovery.py`, `matrix_recovery.py`, and `tests/test_matrix_recovery.py`; removes no files; and changes `package.py` only to include the recovery runtime/test files in the versioned portable inventory. No trainer, model, scientific configuration, split, or dataset builder was changed.

## Recovery guarantees and limits

Before any run, the isolated launcher verifies source and dataset identities, imported-source locations, the v3 Colab resource policy, exact train/validation artifact inventory/window counts, and the six completed backups. For each old run it validates exact artifact names, backup/run-manifest identity, file-byte hashes and sizes for `best.pt` and `validation_predictions.npy` without deserializing either, strict checkpoint-reload claim, full h96 scientific configuration, train-only scaler provenance, common provenance/source identities, approved dataset metadata identity, validation pair count, and explicit test-access denial. Only the two missing shuffled runs are permitted.

The recovery uses a fresh local root `geant_horizon96_streaming_experiment_v3_repro_matrix3_recovery1` and a fresh UUID-named Drive root under `MyDrive/SDNResearch`. Existing roots/runs are refused. Each run has a separate authorization phrase. Its `run.json`, checkpoint, and validation prediction are copied to Drive and read-back hash-verified before lifecycle registration and before prompting for the next run. An interrupted or partial root is never resumed. At completion, a combined eight-run JSON summary is written exclusively, with its hash bound into the final immutable lifecycle snapshot.

The Colab notebook defaults both authorization booleans to false, runs the synthetic suite and a dry preflight without training, then requires global and per-run typed authorization. The Drive check is preflight-only from this authoring environment; the actual six backups remain unverified until the notebook runs in Colab. Drive FUSE read-back is not proof of cloud replication durability. No test-split files are packaged or read.

## Verification performed locally

- `python -m py_compile` on recovery helper, launcher, tests, and package module: passed.
- Notebook JSON and all code-cell AST parsing: passed.
- Notebook-configured synthetic suite: **110 passed, 0 failed, 0 skipped**.
- Independent archive inventory/hash verification: passed.
- `git diff --check`: passed (only existing line-ending warnings for unrelated modified files).
- No training, inference, real-data preflight, Drive access, checkpoint deserialization, or test-split access occurred locally.

### User-controlled next steps

1. Independently review this document and the review11 source ZIP/manifest hashes before approving transfer/use.
2. Open the recovery notebook in Colab; upload review11 source ZIP/manifest and the unchanged approved train/validation dataset ZIP/manifest.
3. Confirm the mounted original backup directory is `MyDrive/SDNResearch/geant_h96_matrix3_5fe371f8aab74f42983569579477c328/runs` and that the proposed recovery Drive destination is new.
4. Run all preparation, synthetic test, and preflight cells. Stop on any mismatch; preflight must report exactly the six completed backups and remaining shuffled seeds 123/2026.
5. Only after a separate approval, set both training flags to `True`, execute the final cell, and enter both per-run phrases. Do not retry or resume any failed/interrupted namespace.
