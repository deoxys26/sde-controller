# GEANT h96 Matrix3 recovery — Review15 candidate audit

Status: **independent review required; not approved for training**. The notebook is restricted to recovery of the two outstanding Matrix3 runs: shuffled adjacency, seeds 123 and 2026. The six completed experiments remain read-only inputs; no old run is a destination.

## Candidate identities

- Notebook: `notebooks/geant_h96_colab_matrix3_recovery.ipynb` — SHA-256 `55741a901972d35945b543196f73a12416e3f2e9bc48bacb2b805eaef23208b8`.
- Source bundle: `data/transfer_bundles/geant_h96_colab_matrix_v1_review15/geant_h96_colab_source_v1.zip` — SHA-256 `c9a47faaa8e9dbd3554548e63cbfa9226c9ed4fb7b503f572be19c99989da034`.
- External source manifest: `data/transfer_bundles/geant_h96_colab_matrix_v1_review15/source_snapshot_manifest.json` — SHA-256 `23d0a8bc997fba5eb03b84a370e3b6a9e349851f201d858c163d82ad136eb77d`.
- Approved train/validation dataset package hashes are unchanged: ZIP `df2bac044c9b9a577d5117310a99022cce402b9370e2ba483ea18b8d7045521e`; manifest `70f60f09b71c0cb89003ac2987af4433df0eeeb64b696845faa25d0fbadb8a95`.

The archive has 44 manifest-listed source members plus the reserved internal manifest (45 ZIP members). Independent `zipfile`/`hashlib` verification passed: exact inventory, internal/external manifest agreement, no duplicate/case-conflicting paths, no directory or symlink members, and every member's byte count and digest matched. The source archive contains no arrays or checkpoints. Review13 and Review14 are preserved; Review15 is the current candidate and is explicitly not pre-approved.

## Preparation-cell repair

The preparation cell imports `sys` locally before `sys.path` use. Upload resolution no longer pins the preparation to a directory created in the upload cell. Instead, each execution of preparation verifies the source ZIP/manifest in memory, executes the staging helper only after matching it to the verified archive manifest, and creates a fresh UUID staging session with exclusive file creation and read-back hashes. A partial prior session is intentionally retained; rerunning preparation uses a new session and never deletes, reuses, or overwrites the old one. Before importing the package, retry logic removes only cached project modules and path entries from prior Matrix3 staging roots; unexpected first-party imports fail closed.

ZIP hash, exact member inventory, safe member-path, symlink, manifest, package, and dataset checks remain fail-closed. The notebook keeps the approved CPU/resource policy, data identity, scientific configuration and test-split restrictions. Its training cell remains disabled until independent source review, successful Colab preflight, global authorization, and a separate exact typed authorization for each of the two runs. Every completed run must be backed up and read-back verified in Drive before the next authorization is requested.

## Verification

- `python -m py_compile phase9_geant/colab_h96_v1/notebook_stage.py phase9_geant/colab_h96_v1/package.py phase9_geant/colab_h96_v1/tests/test_notebook_stage.py phase9_geant/tests/test_matrix3_recovery_notebook.py` — passed.
- Notebook JSON and all code-cell AST parsing — passed; 8 cells.
- Full notebook-configured synthetic suite (portability, launch matrix, matrix workflow, recovery, staging, h96 integration and model synthetic tests), external `--basetemp` — **119 passed, 0 failed, 0 skipped**.
- Local notebook-cell regression tests, external `--basetemp` — **2 passed**. They check the preparation-cell `sys` import, per-attempt fresh staging call, and that upload staging does not reserve/reuse a session.
- Review15 bundle was independently verified as described above.

No Google Drive data was available to this local authoring session, so the six actual backup manifests/checkpoints/predictions have **not** been verified here. The notebook's Colab preflight must validate them before any run. No real GEANT observations, checkpoints, predictions, test artifacts, or Drive backup contents were accessed locally; no training, inference, or real-data evaluation was run. The workflow is prepared for independent review, not authorization. Drive read-back hashes verify visible bytes but do not guarantee cloud replication durability; local path checks are not race-free against a privileged/hostile process.

## Next Colab actions

1. Independently review this audit and approve these exact Review15 source hashes; do not use the superseded Review13/Review14 candidates.
2. Open `notebooks/geant_h96_colab_matrix3_recovery.ipynb` and run cells in order. Upload exactly the two Review15 source files and the unchanged approved dataset ZIP/manifest. The notebook identifies uploads by digest.
3. Mount Drive and let no-training preflight verify the exact six old backups, their run/checkpoint/prediction hashes, shared data/protocol/source identity, reconstructed adjacency digests, resource policy, and the two-run recovery plan.
4. Stop on any failed check. Only after independent review and successful preflight should the separate authorization cell be enabled. Each shuffled run asks for its own exact approval. Never resume a partial recovery namespace.

This candidate has not been executed in Colab and does not authorize training.
