# GEANT horizon-96 Colab migration package — review6 resource policy v3

## Scope and authorization

This package ports the existing horizon-96, batch-4 candidate without changing its model or scientific settings. It is preparation only: no Colab session, upload, real training, or validation experiment was executed while preparing it. The first authorized Colab run, if separately approved, is seed 42 with topology adjacency only. The other eight runs remain gated on review of that first run. Official test-period data stays sealed.

The first Colab attempt failed in `train_epoch_1_batch`: process high-water was 866,205,696 bytes, sampled RSS was 830,087,168 bytes, and the configured process limit was 671,088,640 bytes. At that observation, system-wide available RAM was 9,328,300,032 bytes; no epoch completed and the attempt was reported as `failed`. This is evidence that the 640 MiB process ceiling was too low for the observed process, not evidence of host-wide memory pressure or a proven memory leak. This report is based on the supplied run evidence; the failed run directory was not opened, resumed, or reused.

## Reproducibility and bundle review

The source snapshot is assembled from the v2 trainer's actual `RUN_SOURCE_FILES`, the local topology file required to verify edge ordering, the portable adapter, the synthetic tests, notebook, dependency pins, and this guide. The generated source manifest records exact relative paths, sizes, SHA-256 file-byte digests, a per-file first-party import inventory, current Git HEAD/tag references, and that the working tree was dirty. A newly cloned v1-frozen commit alone is not the execution source: some required code is untracked or modified. `src/gnn/model.py` is included exactly as it existed when packaging and is not edited by this migration.

The review6-final3 source-only package is created independently of any dataset packaging. Its `SOURCE_HASHES.txt` companion records the source ZIP and external source-manifest SHA-256 values after creation. It is outside the ZIP to avoid self-reference; those generated values still require independent human review and approval. The notebook's expected-digest cells remain blank by default. Use the review6-final3 source ZIP/manifest with the already-reviewed train/validation ZIP/manifest from `review5_final2`; Stage 30 does not read or regenerate that dataset bundle. Earlier review6 drafts were generated before the final root-membership test and test-suite verification; retain them, but do not upload them.

The review2 source snapshot omitted `phase9_geant/split_access.py` even though the packaged `phase9_geant/tests/conftest.py` imports it. Review3 added that exact tracked helper (current SHA-256 `D3F460948AD23262087DA65F2C8FC8B7334A726E82E5B251BB28DCF50923C93F`; unchanged from `v1-frozen`) because it supplies the shared explicit train/validation array-name allowlist. The package builder now statically checks first-party Python imports against the source allowlist and records those edges in both source-manifest views; it fails packaging if an imported first-party source module is absent. A later import-origin probe found that some Python installations expose an unrelated top-level `src` namespace from site-packages. Review4's isolated launcher binds `src` to the approved snapshot directory before importing project modules and retains an exact namespace-path check, so site-package namespace locations cannot merge into project imports.

The source and dataset manifest/archive hashes generated locally are **not approval tokens**. Before transfer, an independent reviewer must inspect the source inventory, relevant diffs, notebook cells, package contents, and dataset manifest, then record/approve the exact four sidecar/archive SHA-256 values. The notebook intentionally leaves those expected digest fields blank and refuses to proceed until the operator enters reviewed values. Source hashes establish snapshot bytes, not the identity of a hostile Python runtime or native library.

The train/validation archive contains exactly these nine members, derived from `horizon96_streaming_data.BASE_ARTIFACT_NAMES` and checked against the builder's metadata:

1. `metadata.json`
2. `timestamps_train.npy`
3. `timestamps_validation.npy`
4. `utilization_train.npy`
5. `utilization_validation.npy`
6. `valid_train.npy`
7. `valid_validation.npy`
8. `window_index_train_h96.json`
9. `window_index_validation_h96.json`

It excludes raw demand matrices, test-period arrays/windows/masks/targets, all run outputs, checkpoints, predictions, and credentials. `data/raw/geant_topology/geant.txt` is included in the separate source snapshot because the trainer verifies the ordered 72-link identity; its local SHA-256 is only a byte identity, and upstream provenance remains unverified.

## Local preparation commands

From the repository root, run synthetic tests with a fresh basetemp outside the repository, then compile the portable source:

```powershell
python -m py_compile phase9_geant/colab_h96_v1/package.py phase9_geant/colab_h96_v1/environment.py phase9_geant/colab_h96_v1/linux_resources.py phase9_geant/colab_h96_v1/launch.py
python -m pytest -q phase9_geant/colab_h96_v1/tests/test_portability.py phase9_geant/tests/test_horizon96_streaming_experiment_v2.py phase9_geant/tests/test_horizon96_streaming_experiment.py phase9_geant/tests/test_spatiotemporal_link.py --basetemp <fresh-external-temp>
```

For this resource-policy revision, prepare only a new source snapshot. This command does not inspect or package datasets, and it refuses an existing destination:

```powershell
python -m phase9_geant.colab_h96_v1.package --prepare-source-only --repository-root . --output-root data/transfer_bundles/geant_h96_colab_v1_review6_final3
```

All review2 through review5 outputs and prior review6 drafts remain preserved. The source-only review6-final3 destination is fresh and separate. It contains only the source ZIP, source manifest, and external `SOURCE_HASHES.txt`; it contains no train/validation data bundle, arrays, checkpoints, predictions, or run outputs. The already-reviewed train/validation bundle is a separate artifact from review5_final2 and is not touched by this operation. No upload is performed.

## Colab runtime and environment

The notebook is `notebooks/geant_h96_colab_v1.ipynb`. It uses `/content/colab_upload` for manually uploaded archives, `/content/sdn_project` for a fresh verified source extraction, and the trainer's isolated output root `data/processed/geant_horizon96_streaming_experiment_v2_colab/` beneath that snapshot. All paths are configurable in the notebook; no Windows, OneDrive, or user-home path is embedded.

The recorded local reference pins are Python 3.12.x, NumPy 1.26.4, PyTorch 2.12.0 (CPU build locally), psutil 7.2.2, and pytest 9.1.1. The environment module verifies these pins and records OS/kernel, CPU, CUDA availability/build, package versions, deterministic-algorithm setting, runtime memory and disk. The notebook does not install packages automatically. If the Colab image differs, stop and review the pinned install plan; do not silently accept the difference. Bitwise equivalence across Windows CPU, Linux CPU, and Colab GPU is not claimed. The launcher selects CPU only; GPU presence is reported, not used.

The launcher is started in a fresh Python subprocess with isolated flags. It checks the externally supplied expected manifest digests, source archive/member hashes, exact extracted source snapshot, and every loaded first-party project-module origin before accepting preflight. It rejects preloaded first-party modules and imports only from the extracted bundle. System site-package directories are added explicitly after source verification; `.pth` and `sitecustomize` processing is avoided by the recommended `python -I -S` invocation. Trust still rests on the reviewed launcher/notebook, approved digest values, Python executable, OS, native libraries, and package installations; this is not a defense against a privileged Colab administrator or arbitrary code in the process.

## Resource policy v3 (Linux/Colab)

Policy identifier: `colab-linux-process-memory-v3-visible-cgroup`. The Colab-only process high-water limit remains 2 GiB. The earlier 640 MiB limit was inherited from the Windows-side budget and was below the supplied Colab measurement: high-water 866,205,696 bytes, sampled RSS 830,087,168 bytes, and OS-available RAM 9,328,300,032 bytes when the attempt stopped in its first epoch with no epoch completed. This supports that the process ceiling was too low for that run; it does not establish a leak or guarantee that 2 GiB is sufficient for future runs.

The adapter reports instantaneous process RSS, process high-water (`resource.getrusage`, Linux KiB converted to bytes), system-wide available RAM, cgroup current usage and visible limits/headroom, and free disk as distinct measurements. Policy v3 accepts a literal `max` for cgroup v2 `memory.max` or `memory.high` (and the documented large unlimited sentinel for v1) as unlimited at that inspected level. It walks every discoverable ancestor between the process cgroup and the visible mount boundary and applies the tightest finite `memory.max`, `memory.high`, or v1 hard limit. Missing, unreadable, malformed, negative, or inconsistent essential values fail closed. When cgroup membership contains traversal components or mount namespace information is unavailable, it does not resolve those components into host paths; it inspects only the safe visible root and records the inaccessible-ancestor/mount uncertainty. Thus `visible_unlimited` means only that every inspected visible limit was explicitly unlimited, not that no hidden parent limit exists.

The unchanged safety checks remain: process high-water must be measurable and at or below 2 GiB; startup host-available RAM must be at least 2 GiB + 384 MiB; during training/validation host-available RAM must remain at least 384 MiB; free disk must remain at least 512 MiB. For a finite visible cgroup, startup headroom must be at least 2 GiB + 384 MiB and in-run headroom at least 384 MiB. For an explicitly visible-unlimited cgroup, cgroup headroom is not fabricated; host and process measurements remain mandatory and its uncertainty is recorded. RSS is not mislabeled as high-water. Samples can miss brief spikes; resource behavior beyond the visible mount and actual Colab runtime must be checked independently.

The launcher records the v3 policy and visibility uncertainty in the trainer configuration/manifest and uses the separate `data/processed/geant_horizon96_streaming_experiment_v2_colab_memv3/` namespace. It refuses an existing namespace. This revision does not reuse the failed attempt or v2 output. It does not alter the horizon, graph mode, seed, batch size, optimizer, loss, scaler, or stopping rules.

## Scientific and data integrity

The verified bundle retains train `[0,8020)` and validation `[8020,9740)`, history 672, h96-only target, 72 interleaved directed links, train-window-occurrence-weighted scaler, alpha 0.8 EWMA residual target, SmoothL1 beta 0.001, hidden size 16, batch 4, Adam 0.005, at most 20 epochs, patience 4, and validation-only residual SmoothL1 checkpoint selection. The graph controls use the original model and exact adjacency creation function. The portable adapter changes only output namespace, source/data bundle provenance, environment recording, resource monitoring, and fresh-process import checks. It does not change model code, sampling, scaler, loss, optimizer, epochs, or targets.

`horizon96_streaming_experiment_v2._verified_inputs` rechecks the builder source inventory, compact array hashes, window-index hashes and alignment, split shapes/counts, training-only cutoff/scaler, topology bytes, and exact ordered-edge digest. The loader allowlist is train/validation-only. The bundle contains no test artifact. The old raw builder is not invoked. The source code's build-time source selector may enumerate source directory names, but the Colab transfer and trainer do not include raw source matrices. Upstream topology/source provenance remains unverified.

## Operator workflow

1. Review the locally generated source and dataset manifests, archive listings, hashes, and this guide independently. Approve the exact digest values; do not trust values solely because the packager printed them.
2. Run the local synthetic tests and package verifier. Transfer only the two approved archives and their sidecars through an explicitly user-controlled upload. No credentials are stored here.
3. In a fresh Colab runtime, run notebook cells in order. The upload cell is user initiated. The notebook verifies packages, sidecar digests and archive members, extracts into previously nonexistent roots, verifies source import origins and dataset identity, checks Linux resource limits, runs synthetic tests, and performs dry-run preflight.
4. The final notebook cell defaults to `TRAINING_AUTHORIZED = False` and performs no training. Gate C is not authorized by this migration. A separate explicit approval is required; only then may the cell be changed/run for seed 42/topology. The subprocess also requires `--authorize-training`. Do not authorize identity/shuffled or other seeds before reviewing that run.
5. Choose persistence explicitly before any future training. `/content` is ephemeral. If persisting to Drive, the user must manually mount/select it and extract into a new path; no automatic mount, upload, checkpoint resume, or retry is implemented.
6. Official test evaluation is not included or authorized.

## Limitations / readiness

The source snapshot is prepared from a dirty local tree and requires independent digest review. No Colab runtime was available here, no upload was initiated, and Colab-specific dependency availability, cgroup layout, and filesystem semantics have only synthetic/source verification. The review6-final3 source-only bundle is suitable for **independent review before Colab transfer**, not proof that a real Colab preflight or run will pass. The observed `/proc/self/cgroup` path `/../../jupyter-children` is not normalized into a host path. The reported `memory.max=max` and `memory.high=max` permit the v3 visible-unlimited branch only if they are readable, well-formed, and all other startup checks pass; the cgroup ancestor outside that visible namespace remains unknown.

### User-controlled Colab upload and verification

1. On the local machine, independently hash `geant_h96_colab_source_v1.zip` and `source_snapshot_manifest.json` in review6-final3 and compare with its `SOURCE_HASHES.txt`; do not treat generated digests as approvals. For the unchanged train/validation ZIP and manifest, use the independently reviewed Stage 29 values from review5_final2. The source and data pairs are separate and each expected digest must be reviewed.
2. In a fresh Colab runtime, manually upload exactly the review6-final3 source ZIP/manifest and the previously reviewed review5_final2 train/validation ZIP/manifest, keeping their expected filenames in `/content/colab_upload/`. Do not upload run directories, checkpoints, predictions, or test artifacts.
3. After separate human review/approval, enter the two new source digests from the companion record and the two previously approved dataset digests into the notebook's blank expected-digest fields. Run cells in order through the synthetic suite and dry-run. Require policy ID `colab-linux-process-memory-v3-visible-cgroup`, process high-water at or below 2 GiB, host availability at least 2 GiB + 384 MiB at startup, finite visible cgroup startup headroom at least that value when finite, explicitly visible-unlimited status otherwise, 512 MiB disk, and `preflight_passed_no_training`. Missing/malformed cgroup readings, insufficient finite headroom, insufficient host/process/disk margin, digest mismatch, existing output namespace, or test failure means stop.
4. The final notebook cell remains `TRAINING_AUTHORIZED = False`. Do not edit it or infer authorization from package readiness. A separate approval is still required; the launcher also requires `--authorize-training`. If separately authorized, use only the new `geant_horizon96_streaming_experiment_v2_colab_memv3` namespace and seed-42/topology first run. No automatic upload, training, retries, or other modes/seeds are enabled here.
5. Review6 is **ready only for separate Colab resource/data preflight review**, not proof of Colab runtime behavior and not authorization to train. Cgroup ancestors outside the visible namespace cannot be established locally; if the actual runtime reports any finite visible parent, enforce it, and if essential readings are unreadable/invalid, stop. Test evaluation remains out of scope.
