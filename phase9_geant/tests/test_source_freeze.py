"""Synthetic-only source-freeze tests; never import or invoke the real evaluator."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import types
from pathlib import Path

import pytest

from phase9_geant import source_freeze as freeze


TEST_FILES = (
    "phase9_geant/source_freeze.py",
    "phase9_geant/freeze_probe.py",
    "data/processed/geant_validity_v2/final_test_protocol_v1/PROTOCOL.md",
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _bundle(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / "reviewed-bundle"
    (root / "phase9_geant").mkdir(parents=True)
    (root / "data/processed/geant_validity_v2/final_test_protocol_v1").mkdir(parents=True)
    shutil.copyfile(Path(freeze.__file__), root / TEST_FILES[0])
    (root / TEST_FILES[1]).write_text(
        "from pathlib import Path\n"
        "def main(args):\n"
        "    Path(args[0]).write_text('bundle-worker-ran', encoding='utf-8')\n"
        "    return 0\n",
        encoding="utf-8",
    )
    (root / TEST_FILES[2]).write_text("synthetic protocol fixture\n", encoding="utf-8")
    manifest = {
        "schema": freeze.SCHEMA,
        "approval_status": "independently-reviewed",
        "review_record": "SYNTHETIC-TEST-FIXTURE-NOT-A-REAL-APPROVAL",
        "bundle_id": "synthetic-test-bundle",
        "runtime": freeze._runtime_record(),
        "files": {name: _sha((root / name).read_bytes()) for name in TEST_FILES},
    }
    raw = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    (root / freeze.MANIFEST_NAME).write_bytes(raw)
    return root, _sha(raw)


def test_approved_manifest_verifies_exact_synthetic_bundle(tmp_path):
    root, digest = _bundle(tmp_path)
    manifest = freeze.verify_source_bundle(root, digest, required_sources=TEST_FILES)
    assert manifest["bundle_id"] == "synthetic-test-bundle"


def test_changed_source_fails_before_worker_or_test_loader_can_run(tmp_path):
    root, digest = _bundle(tmp_path)
    probe = root / TEST_FILES[1]
    probe.write_text("def main(args):\n    raise AssertionError('loader invoked')\n", encoding="utf-8")
    marker = tmp_path / "loader-called.txt"
    with pytest.raises(ValueError, match="Frozen source missing, changed, or ambiguous"):
        freeze.launch_verified_bundle(
            root, digest, [str(marker)], _test_entrypoint="phase9_geant.freeze_probe:main",
            _required_sources=TEST_FILES,
        )
    assert not marker.exists()


def test_protocol_modification_fails_before_worker(tmp_path):
    root, digest = _bundle(tmp_path)
    (root / TEST_FILES[2]).write_text("modified synthetic protocol\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Frozen source missing, changed, or ambiguous"):
        freeze.verify_source_bundle(root, digest, required_sources=TEST_FILES)


def test_protocol_changed_after_parent_verification_is_rejected_in_child(tmp_path, monkeypatch):
    root, digest = _bundle(tmp_path)
    marker = tmp_path / "loader-called.txt"
    real_run = freeze.subprocess.run
    changed = False

    def mutate_then_launch(command, **kwargs):
        nonlocal changed
        if not changed:
            (root / TEST_FILES[2]).write_text("changed in verification/launch gap", encoding="utf-8")
            changed = True
        return real_run(command, **kwargs)

    monkeypatch.setattr(freeze.subprocess, "run", mutate_then_launch)
    status = freeze.launch_verified_bundle(
        root, digest, [str(marker)], _test_entrypoint="phase9_geant.freeze_probe:main",
        _required_sources=TEST_FILES,
    )
    assert changed and status != 0
    assert not marker.exists()


def test_missing_and_unexpected_bundle_files_fail_closed(tmp_path):
    missing_root, missing_digest = _bundle(tmp_path / "missing")
    (missing_root / TEST_FILES[1]).unlink()
    with pytest.raises((OSError, ValueError)):
        freeze.verify_source_bundle(missing_root, missing_digest, required_sources=TEST_FILES)

    extra_root, extra_digest = _bundle(tmp_path / "extra")
    (extra_root / "unexpected.txt").write_text("not allow-listed", encoding="utf-8")
    with pytest.raises(ValueError, match="unlisted"):
        freeze.verify_source_bundle(extra_root, extra_digest, required_sources=TEST_FILES)


def test_manifest_digest_must_match_separate_approval_value(tmp_path):
    root, _digest = _bundle(tmp_path)
    with pytest.raises(ValueError, match="independent approval record"):
        freeze.verify_source_bundle(root, "0" * 64, required_sources=TEST_FILES)


def test_runtime_or_dependency_version_mismatch_fails_closed(tmp_path):
    root, _digest = _bundle(tmp_path)
    path = root / freeze.MANIFEST_NAME
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["runtime"]["distributions"]["numpy"] = "not-the-installed-version"
    raw = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="runtime or dependency versions differ"):
        freeze.verify_source_bundle(root, _sha(raw), required_sources=TEST_FILES)


def test_symlink_bundle_root_is_rejected_when_supported(tmp_path):
    root, digest = _bundle(tmp_path)
    alias = tmp_path / "bundle-alias"
    try:
        alias.symlink_to(root, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlink creation is unavailable on this filesystem")
    with pytest.raises(ValueError, match="Redirected source-bundle root component"):
        freeze.verify_source_bundle(alias, digest, required_sources=TEST_FILES)


def _run_probe(root: Path, digest: str, marker: Path) -> int:
    return freeze.launch_verified_bundle(
        root, digest, [str(marker)], _test_entrypoint="phase9_geant.freeze_probe:main",
        _required_sources=TEST_FILES,
    )


def test_conflicting_module_preloaded_in_parent_is_not_inherited(tmp_path, monkeypatch):
    root, digest = _bundle(tmp_path)
    shadow = tmp_path / "shadow"
    (shadow / "phase9_geant").mkdir(parents=True)
    (shadow / "phase9_geant/freeze_probe.py").write_text(
        "from pathlib import Path\n"
        "def main(args):\n"
        "    Path(args[0]).write_text('shadow-ran', encoding='utf-8')\n"
        "    return 0\n",
        encoding="utf-8",
    )
    fake = types.ModuleType("phase9_geant.freeze_probe")
    fake.__file__ = str(shadow / "phase9_geant/freeze_probe.py")
    monkeypatch.setitem(sys.modules, "phase9_geant.freeze_probe", fake)
    marker = tmp_path / "worker-origin.txt"
    status = _run_probe(root, digest, marker)
    assert status == 0
    assert marker.read_text(encoding="utf-8") == "bundle-worker-ran"


def test_pythonpath_shadow_cannot_replace_bundle_module(tmp_path, monkeypatch):
    root, digest = _bundle(tmp_path)
    shadow = tmp_path / "shadow"
    (shadow / "phase9_geant").mkdir(parents=True)
    (shadow / "phase9_geant/freeze_probe.py").write_text(
        "from pathlib import Path\n"
        "def main(args):\n"
        "    Path(args[0]).write_text('shadow-ran', encoding='utf-8')\n"
        "    return 0\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("PYTHONPATH", str(shadow))
    marker = tmp_path / "worker-origin.txt"
    status = _run_probe(root, digest, marker)
    assert status == 0
    assert marker.read_text(encoding="utf-8") == "bundle-worker-ran"
