# GEANT h96 Matrix3 recovery — Review16 audit

Status: **candidate only; independent review and a fresh Colab preflight are required. No training authorization is granted.**

## Resource-policy defect and correction

The failure was a result-envelope schema mismatch, not a missing trainer attribute. `linux_resources.preflight()` returns a wrapper with `status`, `snapshot`, and nested `policy`; the policy ID is at `result["policy"]["policy_id"]`, not `result["policy_id"]`. The recovery launcher inspected the outer mapping at the failing condition, so it raised before `_bind_resource_policy()` ran. Consequently the trainer still showed its pre-binding 671,088,640-byte default. The trainer is not designed to expose a top-level `RESOURCE_POLICY_ID`; the binder records policy identity under `trainer.CONFIG["linux_resource_policy"]` and updates the process/memory settings.

The first-run launcher passes `resource_preflight["policy"]` to the same binder. The existing Matrix3 launcher likewise runs preflight and then binds `policy["policy"]` before its train/validation input reader. Review16 follows that contract: it requires a passing envelope, checks the nested policy ID, invokes the unchanged binder, and verifies the effective 2-GiB process limit, 384-MiB host reserve, and bound policy record before dataset verification/loading. The binder continues enforcing the 384-MiB cgroup reserve and 512-MiB disk reserve. No memory limit, reserve, or scientific setting changed.

## Synthetic verification

- Focused recovery tests: `python -m pytest -q phase9_geant/colab_h96_v1/tests/test_matrix_recovery.py --basetemp <external-temp>` — **16 passed**.
- Complete notebook-configured synthetic suite: portability, matrix, recovery, notebook-stage, streaming experiment, v2 streaming experiment, and spatio-temporal model test modules — **121 passed**.
- `py_compile` passed for the changed launcher and recovery test module.
- Recovery launcher `--help` exited 0; notebook JSON loaded, every code cell parsed with Python AST, source pins matched Review16, and both training authorization flags remained false by default.
- `git diff --check` exited 0 (only pre-existing line-ending warnings were printed).

The suite was run with pytest basetemp under the user's external local temp directory. A first attempt with a deep external basetemp encountered Windows temporary-path creation failure; a short basetemp under the repository was rejected by the suite's intentional outside-repository guard. The final external-temp invocation passed. No Colab runtime or Google Drive was available here, so Drive backup verification and actual notebook execution in Colab remain unverified.

## Review16 artifacts

- Source-only bundle: `data/transfer_bundles/geant_h96_colab_matrix_v1_review16/geant_h96_colab_source_v1.zip`
  - SHA-256: `e8cb243e2c37720b2381ed8f2872c65b411e1d2285f000587302e5a1ef0913e1`
- External source manifest: `data/transfer_bundles/geant_h96_colab_matrix_v1_review16/source_snapshot_manifest.json`
  - SHA-256: `1d70304ae17349803c97b0776d5e0debc529211b427610b2325e1815fe2c5443`
- Independently checked inventory: 44 listed source entries plus the reserved embedded manifest (45 ZIP members); exact inventory, no duplicate/case-colliding names, no directory/symlink entries, and all listed sizes/digests matched.
- Updated recovery notebook: `notebooks/geant_h96_colab_matrix3_recovery.ipynb`
  - SHA-256: `bdfea52b86e5c57bc43f95a1c4eb70a2260d5ca149ef79e524c6a2d916aa4a83`

The notebook now pins Review16 and explicitly labels it a review candidate. Review15 and its audit were not edited. Dataset bundle pins are unchanged. The notebook still limits recovery to shuffled seeds 123 and 2026, requires separate authorization for each, and does not package or access the official test split. The six existing Drive backups were not accessible or modified in this local work; they must be reverified read-only in Colab before any authorization.

## Scope and stop condition

No training, preflight against real artifacts, inference, dataset rebuild, checkpoint access, Drive upload, or sealed-test access occurred. Review16 is not approved automatically. Next step: independently review this audit and exact Review16 hashes, then run the notebook's synthetic/preflight stages in the target Colab session. Any training still requires a separate explicit authorization after those checks pass.
