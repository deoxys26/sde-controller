# GEANT final-evaluator source freeze (implementation review draft)

## Status

This document describes an implementation, not approval to evaluate the test split. The evaluator's `SOURCE_FREEZE_REVIEWED` flag remains `False`; no approved manifest, immutable bundle, dependency lock, or real bundle hashes have been created. Do not run preflight or final evaluation until a separate static review and filesystem assessment are complete.

## First-party source inventory

The frozen bundle must contain exactly the first-party files listed by `FIRST_PARTY_FILES` in `phase9_geant/source_freeze.py`:

- `phase9_geant/__init__.py`
- `phase9_geant/evaluate_bu_v2_final.py`
- `phase9_geant/parse_geant.py`
- `phase9_geant/source_freeze.py`
- `src/gnn/__init__.py`
- `src/gnn/dataset.py`
- `src/gnn/preprocessing.py`
- `src/gnn/model.py`
- `data/processed/geant_validity_v2/final_test_protocol_v1/PROTOCOL.md`

The evaluator imports the topology parser during preflight. Model loading imports `src.gnn.model`; Python first executes `src/gnn/__init__.py`, which imports the dataset and preprocessing helper as well. `numpy` and `torch` are required third-party distributions. The source inventory deliberately excludes all datasets, checkpoints, predictions, labels, and metrics.

## Bundle layout and manifest

An independently prepared bundle has this layout:

```text
<frozen-bundle>/
  source-freeze-manifest.json
  phase9_geant/                 # exact first-party inventory above
  src/gnn/                      # exact first-party inventory above
  data/processed/geant_validity_v2/final_test_protocol_v1/PROTOCOL.md
```

The manifest schema is `geant-source-freeze-v1`. It records `approval_status`, an external review-record reference, a bundle identity, exact path-to-SHA-256 entries, and a runtime record. `source_freeze.py` requires the operator to supply the manifest SHA-256 from the separate review record. It never creates, selects, or approves expected hashes. The manifest's `files` mapping must equal the fixed first-party allow-list; any missing or additional file fails closed.

Current runtime checking records the Python implementation/version, interpreter executable SHA-256, and installed `numpy`/`torch` distribution versions. These checks are useful compatibility evidence but **do not establish the byte identity of all installed dependency code or native libraries**. An exact dependency lock and independently reviewed runtime/package content identity are still required. The source-freeze implementation must not be treated as complete runtime attestation until that evidence is supplied and the verifier is reviewed against it.

## Verification and execution sequence

1. An independent reviewer selects and inspects the exact source revision, enumerates transitive first-party imports, prepares a candidate bundle and manifest, and verifies that no protected data artifacts are included.
2. A separate reviewer compares the manifest paths and digests against the reviewed source/bundle, checks the runtime/dependency identity, and records approval outside the bundle. Candidate generation is not approval.
3. Preserve the approved bundle in a dedicated immutable/read-only location. The launcher requires the independently recorded manifest digest and verifies the exact source inventory before starting a child process.
4. The launcher starts a fresh Python process with isolated mode (`-I`) and bytecode writes disabled (`-B`). The child checks that first-party project modules were not preloaded, re-verifies the manifest and source files before importing the evaluator, places only the bundle root ahead of ordinary import paths, then imports the evaluator from that bundle. The evaluator checks its own, package, parser, and model module paths against the bundle.
5. The evaluator still revalidates the manifest and source bundle before preflight/execution. Its existing runtime gate remains disabled. Any missing, extra, altered, redirected, or mismatched source/runtime input aborts; there is no retry or fallback to the mutable checkout.

The intended future invocation shape is:

```text
<approved-bundle-python> -I -B <frozen-bundle>/phase9_geant/source_freeze.py \
  --bundle-root <frozen-bundle> \
  --approved-manifest-sha256 <digest-from-independent-review> \
  --launch-evaluator -- --authorize-final-test-evaluation
```

This command is documentation only and must not be run until separate authorization and all gates are reviewed. No such real bundle or digest currently exists.

## Trust assumptions and limitations

- The operator is assumed to launch the reviewed launcher from the approved bundle; the filesystem and bundle remain immutable between verification and import. The Python program cannot protect itself against an administrator, arbitrary code in the same process, or an actor able to replace the interpreter, bundle, manifest approval record, or filesystem state.
- `-I` reduces environment and import-path contamination but is not a sandbox. The launcher checks module origins; a fully trusted, reviewed interpreter/runtime remains an external prerequisite.
- Path-component and file-identity checks are not a substitute for a Windows-specific assessment of junctions, reparse tags, cloud placeholders, OneDrive hydration, sharing modes, or in-place writes. The source bundle should be on a reviewed local non-synced path. The current implementation rejects reparse points conservatively; actual Windows behavior remains unverified.
- Hash-before-import still has a time-of-check/time-of-use window if the bundle is mutable. Read-only deployment and independent filesystem review are prerequisites, not claims proven by synthetic tests.
- The final evaluator must remain blocked until source identity, complete runtime/dependency identity, the bundle's Windows filesystem behavior, and the evaluator integration have passed independent review. Historical test-derived artifacts remain prohibited inputs.

## Safe synthetic verification

`phase9_geant/tests/test_source_freeze.py` uses disposable temporary bundles only. It tests changed source/protocol rejection, missing/unlisted files, an externally supplied synthetic manifest digest, fresh child-process execution from a synthetic bundle despite parent `sys.modules` and `PYTHONPATH` conflicts, and that a probe standing in for a loader is not reached after verification failure. These tests do not open GEANT files, checkpoints, arrays, predictions, labels, or metrics. They do not establish actual Windows junction/reparse/cloud behavior or dependency runtime identity.
