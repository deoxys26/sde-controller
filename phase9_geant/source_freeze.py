"""Verify and launch the GEANT evaluator from an independently approved source bundle.

This module uses only the Python standard library. A manifest is not approval by
itself: callers must supply its digest from a separately reviewed record.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import platform
import runpy
import subprocess
import sys
from typing import Mapping, Sequence


MANIFEST_NAME = "source-freeze-manifest.json"
SCHEMA = "geant-source-freeze-v1"
FIRST_PARTY_FILES = (
    "phase9_geant/__init__.py",
    "phase9_geant/evaluate_bu_v2_final.py",
    "phase9_geant/parse_geant.py",
    "phase9_geant/source_freeze.py",
    "src/gnn/__init__.py",
    "src/gnn/dataset.py",
    "src/gnn/preprocessing.py",
    "src/gnn/model.py",
    "data/processed/geant_validity_v2/final_test_protocol_v1/PROTOCOL.md",
)
REQUIRED_DISTRIBUTIONS = ("numpy", "torch")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_relative(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or not path.parts or any(part in ("", ".", "..") for part in path.parts):
        raise ValueError(f"Unsafe bundle-relative path: {name!r}")
    if "\\" in name or ":" in name:
        raise ValueError(f"Non-portable bundle-relative path: {name!r}")
    return path


def _reject_reparse_components(root: Path, candidate: Path) -> None:
    """Conservative source-bundle check; actual Windows tag semantics need review."""
    supplied_root = root.absolute()
    cursor = Path(supplied_root.anchor)
    for part in supplied_root.parts[1:]:
        cursor = cursor / part
        if cursor.is_symlink() or (cursor.exists() and getattr(cursor, "is_junction", lambda: False)()):
            raise ValueError(f"Redirected source-bundle root component: {cursor}")
        if cursor.exists() and getattr(cursor.lstat(), "st_file_attributes", 0) & 0x0400:
            raise ValueError(f"Reparse point in source-bundle root: {cursor}")
    root = supplied_root.resolve(strict=True)
    if os.path.normcase(str(root)) != os.path.normcase(str(supplied_root)):
        raise ValueError(f"Source-bundle root is not its canonical path: {supplied_root}")
    candidate_abs = candidate.absolute()
    try:
        candidate_abs.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Bundle path escapes approved root: {candidate}") from exc
    cursor = root
    for part in candidate_abs.relative_to(root).parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(f"Symlink in frozen bundle path: {cursor}")
        if cursor.exists() and getattr(cursor, "is_junction", lambda: False)():
            raise ValueError(f"Junction in frozen bundle path: {cursor}")
        attrs = getattr(cursor.lstat(), "st_file_attributes", 0) if cursor.exists() else 0
        if attrs & 0x0400:
            raise ValueError(f"Reparse point in frozen bundle path: {cursor}")


def _runtime_record() -> dict[str, object]:
    executable = Path(sys.executable).resolve(strict=True)
    return {
        "implementation": sys.implementation.name,
        "python_version": platform.python_version(),
        "executable_sha256": _sha256(executable.read_bytes()),
        "distributions": {
            name: importlib.metadata.version(name) for name in REQUIRED_DISTRIBUTIONS
        },
    }


def _verify_tree(root: Path, declared: Mapping[str, object], required_sources: Sequence[str]) -> None:
    entries = set(declared)
    required = set(required_sources)
    if entries != required:
        raise ValueError(
            f"Manifest file set must exactly match reviewed first-party inventory; "
            f"missing={sorted(required - entries)}, unexpected={sorted(entries - required)}"
        )
    for name, expected in declared.items():
        relative = _safe_relative(name)
        path = root.joinpath(*relative.parts)
        _reject_reparse_components(root, path)
        info = path.lstat()
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Manifest entry is not a regular file: {name}")
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError(f"Invalid SHA-256 in manifest for {name}")
        before = (info.st_size, info.st_mtime_ns, getattr(info, "st_ino", 0))
        content = path.read_bytes()
        after_info = path.stat()
        after = (after_info.st_size, after_info.st_mtime_ns, getattr(after_info, "st_ino", 0))
        if before != after or _sha256(content) != expected.lower():
            raise ValueError(f"Frozen source missing, changed, or ambiguous: {name}")

    actual = set()
    for current, dirs, files in os.walk(root, followlinks=False):
        base = Path(current)
        for directory in dirs:
            item = base / directory
            if item.is_symlink() or getattr(item, "is_junction", lambda: False)():
                raise ValueError(f"Directory redirection in source bundle: {item}")
        for filename in files:
            item = base / filename
            if item == root / MANIFEST_NAME:
                continue
            relative = item.relative_to(root).as_posix()
            actual.add(relative)
    if actual != entries:
        raise ValueError(
            f"Bundle contains missing or unlisted files; missing={sorted(entries - actual)}, "
            f"unlisted={sorted(actual - entries)}"
        )


def verify_source_bundle(
    bundle_root: str | Path,
    approved_manifest_sha256: str,
    *,
    required_sources: Sequence[str] = FIRST_PARTY_FILES,
) -> dict[str, object]:
    """Verify an approved manifest and exact source tree before project imports.

    The expected manifest digest must come from an independent review record;
    this function never creates or blesses that digest.
    """
    if not isinstance(approved_manifest_sha256, str) or len(approved_manifest_sha256) != 64:
        raise ValueError("An independently approved manifest SHA-256 is required")
    supplied_root = Path(bundle_root).absolute()
    _reject_reparse_components(supplied_root, supplied_root)
    root = supplied_root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Source bundle root must be a directory")
    manifest_path = root / MANIFEST_NAME
    _reject_reparse_components(root, manifest_path)
    raw = manifest_path.read_bytes()
    if _sha256(raw).lower() != approved_manifest_sha256.lower():
        raise ValueError("Manifest digest does not match the independent approval record")
    manifest = json.loads(raw.decode("utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA:
        raise ValueError("Unsupported source-freeze manifest schema")
    if manifest.get("approval_status") != "independently-reviewed" or not manifest.get("review_record"):
        raise ValueError("Manifest lacks an independent review record")
    if not manifest.get("bundle_id"):
        raise ValueError("Manifest lacks a reviewed bundle identity")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("Manifest files must be a path-to-SHA-256 mapping")
    _verify_tree(root, files, required_sources)
    runtime = manifest.get("runtime")
    if runtime != _runtime_record():
        raise ValueError("Python runtime or dependency versions differ from the approved manifest")
    return manifest


def _worker_main(
    bundle_root: str,
    approved_manifest_sha256: str,
    required_sources: Sequence[str],
    entrypoint: str,
    worker_args: Sequence[str],
) -> int:
    supplied_bundle = Path(bundle_root).absolute()
    _reject_reparse_components(supplied_bundle, supplied_bundle)
    verify_source_bundle(supplied_bundle, approved_manifest_sha256, required_sources=required_sources)
    bundle = supplied_bundle.resolve(strict=True)
    module, function = entrypoint.split(":", 1)
    sys.path.insert(0, str(bundle))
    loaded = __import__(module, fromlist=[function])
    expected = (bundle / (module.replace(".", "/") + ".py")).resolve(strict=True)
    if Path(loaded.__file__).resolve(strict=True) != expected:
        raise RuntimeError("Evaluator entrypoint did not load from the verified bundle")
    if module == "phase9_geant.evaluate_bu_v2_final":
        return int(getattr(loaded, function)(worker_args, frozen_source_root=bundle,
                                             approved_manifest_sha256=approved_manifest_sha256))
    return int(getattr(loaded, function)(worker_args))


def launch_verified_bundle(
    bundle_root: str | Path,
    approved_manifest_sha256: str,
    evaluator_args: Sequence[str],
    *,
    _test_entrypoint: str | None = None,
    _required_sources: Sequence[str] = FIRST_PARTY_FILES,
) -> int:
    """Verify first, then dispatch into a fresh isolated interpreter from bundle."""
    supplied_root = Path(bundle_root).absolute()
    _reject_reparse_components(supplied_root, supplied_root)
    root = supplied_root.resolve(strict=True)
    verify_source_bundle(root, approved_manifest_sha256, required_sources=_required_sources)
    entrypoint = _test_entrypoint or "phase9_geant.evaluate_bu_v2_final:main"
    script = """
import json, runpy, sys
from pathlib import Path
root = Path(sys.argv[1]).resolve(strict=True)
digest = sys.argv[2]
required = json.loads(sys.argv[3])
entry = sys.argv[4]
args = sys.argv[5:]
if any(n == 'phase9_geant' or n.startswith('phase9_geant.') or n == 'src' or n.startswith('src.') for n in sys.modules):
    raise SystemExit('project module preloaded before frozen verification')
ns = runpy.run_path(str(root / 'phase9_geant' / 'source_freeze.py'), run_name='_geant_frozen_child')
raise SystemExit(ns['_worker_main'](str(root), digest, required, entry, args))
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-c", script, str(root), approved_manifest_sha256,
         json.dumps(list(_required_sources)), entrypoint, *list(evaluator_args)],
        cwd=str(root), check=False,
    )
    return int(completed.returncode)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", required=True)
    parser.add_argument("--approved-manifest-sha256", required=True)
    parser.add_argument("--launch-evaluator", action="store_true")
    parser.add_argument("evaluator_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    verify_source_bundle(args.bundle_root, args.approved_manifest_sha256)
    if not args.launch_evaluator:
        print("Source bundle verified; evaluator was not launched.")
        return 0
    evaluator_args = args.evaluator_args[1:] if args.evaluator_args[:1] == ["--"] else args.evaluator_args
    return launch_verified_bundle(args.bundle_root, args.approved_manifest_sha256, evaluator_args)


if __name__ == "__main__":
    raise SystemExit(main())
