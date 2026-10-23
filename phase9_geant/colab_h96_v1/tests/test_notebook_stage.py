"""Synthetic-only tests for Matrix3 preparation staging and retry behavior."""

from __future__ import annotations

import hashlib
from pathlib import Path
import types

import pytest

from phase9_geant.colab_h96_v1 import notebook_stage


def _uploads():
    payloads = {
        name: ("synthetic:" + name).encode("ascii")
        for name in notebook_stage.UPLOAD_NAMES
    }
    hashes = {name: hashlib.sha256(value).hexdigest() for name, value in payloads.items()}
    return payloads, hashes


def test_exact_synthetic_uploads_stage_under_fresh_session(tmp_path):
    payloads, hashes = _uploads()
    session_id, session, incoming = notebook_stage.create_fresh_preparation_session(
        tmp_path, payloads, hashes,
    )
    assert session.name == f"{notebook_stage.SESSION_DIR_PREFIX}{session_id}"
    assert {path.name for path in incoming.iterdir()} == notebook_stage.UPLOAD_NAMES
    assert all((incoming / name).read_bytes() == payload for name, payload in payloads.items())


def test_partial_stage_is_preserved_and_retry_uses_a_new_session(tmp_path, monkeypatch):
    payloads, hashes = _uploads()
    original_open = Path.open
    failed_once = False

    def fail_during_stage(path, *args, **kwargs):
        nonlocal failed_once
        if path.name == "source_snapshot_manifest.json" and not failed_once:
            failed_once = True
            raise OSError("synthetic interrupted staging write")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_during_stage)
    with pytest.raises(OSError, match="synthetic interrupted"):
        notebook_stage.create_fresh_preparation_session(tmp_path, payloads, hashes)
    monkeypatch.setattr(Path, "open", original_open)

    prior_sessions = list(tmp_path.glob(f"{notebook_stage.SESSION_DIR_PREFIX}*"))
    assert len(prior_sessions) == 1
    prior = prior_sessions[0]
    prior_snapshot = {path.name: path.read_bytes() for path in prior.rglob("*") if path.is_file()}

    _session_id, fresh, incoming = notebook_stage.create_fresh_preparation_session(
        tmp_path, payloads, hashes,
    )
    assert fresh != prior
    assert fresh.exists() and incoming.is_dir()
    assert {path.name: path.read_bytes() for path in prior.rglob("*") if path.is_file()} == prior_snapshot
    assert {path.name for path in incoming.iterdir()} == notebook_stage.UPLOAD_NAMES


def test_existing_session_collision_is_not_reused(tmp_path, monkeypatch):
    payloads, hashes = _uploads()
    fixed = "0123456789abcdef0123456789abcdef"
    monkeypatch.setattr(notebook_stage.uuid, "uuid4", lambda: type("U", (), {"hex": fixed})())
    notebook_stage.create_fresh_preparation_session(tmp_path, payloads, hashes)
    with pytest.raises(FileExistsError):
        notebook_stage.create_fresh_preparation_session(tmp_path, payloads, hashes)


def test_invalid_upload_inventory_or_hash_fails_before_staging(tmp_path):
    payloads, hashes = _uploads()
    wrong = dict(hashes)
    wrong["geant_h96_colab_source_v1.zip"] = "0" * 64
    with pytest.raises(notebook_stage.StagingError, match="digest mismatch"):
        notebook_stage.create_fresh_preparation_session(tmp_path, payloads, wrong)
    with pytest.raises(notebook_stage.StagingError, match="Exactly the four"):
        notebook_stage.create_fresh_preparation_session(
            tmp_path, {k: v for k, v in payloads.items() if k != "dataset_bundle_manifest.json"}, hashes,
        )
    assert list(tmp_path.iterdir()) == []


def test_retry_removes_only_project_modules_from_prior_staging_session(tmp_path):
    base = tmp_path / "content"
    old_repo = base / "geant_h96_matrix3_recovery_old" / "approved_source_snapshot"
    old_package = old_repo / "phase9_geant" / "colab_h96_v1" / "package.py"
    old_src = old_repo / "src"
    sys_path = [str(old_repo), "C:/python/site-packages"]
    modules = {
        "phase9_geant": types.SimpleNamespace(__file__=str(old_repo / "phase9_geant" / "__init__.py")),
        "phase9_geant.colab_h96_v1.package": types.SimpleNamespace(__file__=str(old_package)),
        "src": types.SimpleNamespace(__file__=None, __path__=[str(old_src)]),
    }
    expected_removed = set(modules)
    removed = notebook_stage.reset_prior_staged_project_imports(sys_path, modules, base)
    assert set(removed) == expected_removed
    assert modules == {}
    assert sys_path == ["C:/python/site-packages"]


def test_retry_refuses_unrelated_preloaded_project_module(tmp_path):
    foreign = types.SimpleNamespace(__file__="C:/unrelated/phase9_geant/__init__.py")
    with pytest.raises(notebook_stage.StagingError, match="unexpected preloaded"):
        notebook_stage.reset_prior_staged_project_imports([], {"phase9_geant": foreign}, tmp_path)
