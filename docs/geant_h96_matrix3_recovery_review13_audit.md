# GEANT h96 Matrix3 recovery — review13 candidate audit

Status: **candidate for independent review only; not approved for training**. This workflow is limited to the two missing experiments: shuffled adjacency, seeds 123 and 2026. The original six Drive backups, Matrix1/Matrix2 outputs, and seed-42 topology result are not modified or reused as destinations.

## Deliverables and identities

- Notebook: `notebooks/geant_h96_colab_matrix3_recovery.ipynb` — SHA-256 `224afba51cd03e13a1dcdebdc4cfa53868575ca3392dcd1a5708e0ce520a1c69`.
- Source bundle directory: `data/transfer_bundles/geant_h96_colab_matrix_v1_review13`.
- Source ZIP: `geant_h96_colab_source_v1.zip` — SHA-256 `c51927e428ebb85915379ed9a301e86930454f9eb61dbaa6148624ab4f75e286`.
- External source manifest: `source_snapshot_manifest.json` — SHA-256 `4385ee54f5319dc32cf0ed295c3c61c8b14ff0d293a35fb14daa92d406471d7a`.
- Approved dataset ZIP and manifest identities remain unchanged: `df2bac044c9b9a577d5117310a99022cce402b9370e2ba483ea18b8d7045521e` and `70f60f09b71c0cb89003ac2987af4433df0eeeb64b696845faa25d0fbadb8a95`.

The source archive contains 42 listed source members plus its reserved internal manifest (43 ZIP members total). An independent `zipfile`/`hashlib` pass verified the exact archive inventory, no duplicate or case-conflicting names, no directory or symlink entries, agreement of external and internal manifests, and each listed member's byte count and SHA-256. The package generator reports `dataset_artifacts_opened: false`; it packaged source only. The notebook is outside the ZIP to avoid a self-referential digest.

## Recovery behavior

The notebook re-verifies uploaded source and dataset packages by their pinned SHA-256 values, uses a new UUID session and Drive recovery directory, validates the exact train/validation-only dataset inventory, runs synthetic tests, and invokes the recovery launcher's no-training preflight. It requires six exact original run directories and checks their backup manifests, run-manifest/checkpoint/prediction byte hashes and sizes, protocol/configuration, train-only scaler claim, shared provenance/source identity, dataset identity, and realized adjacency hash against the ordered graph reconstructed from the approved link order. Only after preflight does the separate authorization cell permit recovery; it is disabled by default and requires an exact approval phrase before each remaining run. Each newly completed run is copied and hash/read-back verified on Drive before authorization of the next. Any incomplete or failed namespace is preserved and cannot be resumed.

Recovery uses a fresh local namespace `geant_horizon96_streaming_experiment_v3_repro_matrix3_recovery1` and a fresh per-session Drive destination. Matrix1, Matrix2, the six completed experiments, and existing failed outputs are not destinations. The fixed scientific configuration and test-split denial are retained. Validation predictions and checkpoints are treated as opaque bytes for integrity checks; this workflow does not inspect test data.

## Bootstrap correction

Review12's launcher had a preflight ordering defect: it referenced `matrix` and `recovery` before those names were imported. Review13 fixes this by validating the source archive/manifest and exact `launch.py` bytes using stdlib-only bootstrap code, executing only those verified bootstrap bytes, and then calling the bootstrap's full source verification before importing project modules. Recovery/matrix helpers are imported only after that verification. A synthetic regression test confirms the bootstrap rejects a modified source file before executing it.

## Verification performed locally

- `python -m py_compile phase9_geant/colab_h96_v1/matrix_recovery.py phase9_geant/colab_h96_v1/launch_matrix_recovery.py phase9_geant/colab_h96_v1/tests/test_matrix_recovery.py phase9_geant/colab_h96_v1/package.py` — passed.
- Notebook-configured synthetic suite (the seven test files listed in the notebook), run with an external temporary base directory — **113 passed, 0 failed, 0 skipped**.
- `python -I -S phase9_geant/colab_h96_v1/launch_matrix_recovery.py --help` — exit 0; no data or project imports.
- Notebook JSON and code-cell AST validation — passed (8 cells).
- Independent ZIP/manifest/member verification — passed.

Tests use synthetic files/metadata and do not inspect Drive backups or GEANT observations. No Drive-backed preflight was run locally; this environment cannot establish that the six real backups exist or pass their checks. No training, inference, real-data evaluation, or test-split access occurred. The candidate must be independently reviewed and then preflighted in the actual Colab session before any separate training authorization. A Drive FUSE read-back confirms visible bytes and hashes, not cloud replication durability. Local path-based checks are not race-free against a privileged or hostile local process.

## Colab handoff

1. Independently review this audit, the source ZIP, and its external manifest; approve these exact source hashes separately.
2. Open `notebooks/geant_h96_colab_matrix3_recovery.ipynb` in Colab and run its cells in order. Upload exactly the source ZIP, source manifest, approved train/validation dataset ZIP, and dataset manifest; the notebook resolves them by digest, not filename.
3. Confirm Drive mount and the expected original `runs` directory. The notebook preflight must verify all six backups and report only shuffled seeds 123 and 2026 as remaining, plus the v3 resource policy.
4. Do not edit the authorization cell until independent review and explicit authorization. If authorized, enable both flags and provide the global phrase and each per-run phrase. Stop on any error; never reuse an incomplete destination.

This local review does not certify Colab execution or authorize training.
