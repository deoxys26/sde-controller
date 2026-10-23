"""Synthetic-only tests for Colab bundle, path, environment and resource guards."""

from __future__ import annotations

import hashlib
import io
import json
import stat
import zipfile
from types import SimpleNamespace
from pathlib import Path

import pytest

from phase9_geant.colab_h96_v1 import environment, launch, linux_resources, package


def _zip(members: list[tuple[str, bytes, int | None]]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, payload, mode in members:
            info = zipfile.ZipInfo(name)
            info.external_attr = ((mode if mode is not None else stat.S_IFREG | 0o644) << 16)
            archive.writestr(info, payload)
    return buffer.getvalue()


def _entry(name: str, payload: bytes) -> dict[str, object]:
    return {"path": name, "size_bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _external_manifest(kind: str, archive: bytes, entries: list[dict[str, object]]) -> tuple[dict, bytes]:
    if kind == "data":
        manifest = {
            "schema": "geant-h96-colab-trainval-bundle-v1",
            "archive_sha256": hashlib.sha256(archive).hexdigest(),
            "artifacts": entries,
        }
    else:
        imports = {entry["path"]: entry.get("first_party_imports", [])
                   for entry in entries if str(entry["path"]).endswith(".py")}
        manifest = {
            "schema": "geant-h96-colab-source-snapshot-v1",
            "archive_sha256": hashlib.sha256(archive).hexdigest(),
            "entries": entries,
            "first_party_import_inventory": imports,
        }
    return manifest, package.canonical_json_bytes(manifest)


def test_dataset_bundle_allowlist_is_exactly_nine_train_validation_files():
    assert set(package.DATA_ARTIFACTS) == {
        "metadata.json", "utilization_train.npy", "utilization_validation.npy",
        "valid_train.npy", "valid_validation.npy", "timestamps_train.npy",
        "timestamps_validation.npy", "window_index_train_h96.json",
        "window_index_validation_h96.json",
    }
    payloads = [(name, f"synthetic:{name}".encode(), None) for name in package.DATA_ARTIFACTS]
    archive_bytes = _zip(payloads)
    entries = [_entry(name, content) for name, content, _ in payloads]
    manifest = {"archive_sha256": hashlib.sha256(archive_bytes).hexdigest(), "artifacts": entries}
    raw_manifest = package.canonical_json_bytes(manifest)
    result = package._verify_archive_payload(
        archive_bytes, manifest, expected_archive_sha256=manifest["archive_sha256"],
        expected_manifest_sha256=hashlib.sha256(raw_manifest).hexdigest(),
        manifest_payload=raw_manifest, kind="data",
    )
    assert len(result["members"]) == 9
    assert not any("test" in name.casefold() for name in result["members"])


@pytest.mark.parametrize("bad_name", ["X_test.npy", "nested/test/window.json", "../escape.npy", "C:/outside.npy"])
def test_dataset_bundle_rejects_unknown_test_or_traversal_members(bad_name):
    payloads = [(name, b"synthetic", None) for name in package.DATA_ARTIFACTS]
    payloads.append((bad_name, b"forbidden", None))
    archive_bytes = _zip(payloads)
    entries = [_entry(name, content) for name, content, _ in payloads]
    manifest = {"archive_sha256": hashlib.sha256(archive_bytes).hexdigest(), "artifacts": entries}
    raw_manifest = package.canonical_json_bytes(manifest)
    with pytest.raises(package.BundleError):
        package._verify_archive_payload(
            archive_bytes, manifest, expected_archive_sha256=manifest["archive_sha256"],
            expected_manifest_sha256=hashlib.sha256(raw_manifest).hexdigest(),
            manifest_payload=raw_manifest, kind="data",
        )


def test_archive_verification_rejects_duplicate_case_names_and_symlinks():
    duplicate = _zip([("a.txt", b"one", None), ("A.txt", b"two", None)])
    with zipfile.ZipFile(io.BytesIO(duplicate)) as archive:
        with pytest.raises(package.BundleError, match="Duplicate or case-conflicting"):
            package._zip_member_names(archive)
    symlink = _zip([("link", b"target", stat.S_IFLNK | 0o777)])
    with zipfile.ZipFile(io.BytesIO(symlink)) as archive:
        with pytest.raises(package.BundleError, match="Symlink archive member"):
            package._zip_member_names(archive)
    with pytest.raises(package.BundleError, match="Unsafe archive path"):
        package._relative_path("C:/outside.py")


def test_source_bundle_manifest_and_source_bytes_are_verified():
    source_payload = b"synthetic approved source"
    entry = {**_entry("phase9_geant/example.py", source_payload), "first_party_imports": []}
    import_inventory = {"phase9_geant/example.py": []}
    internal = {"schema": "geant-h96-colab-source-snapshot-v1", "entries": [entry],
                "first_party_import_inventory": import_inventory}
    archive_bytes = _zip([
        ("phase9_geant/example.py", source_payload, None),
        (package.SOURCE_MANIFEST_MEMBER, package.canonical_json_bytes(internal), None),
    ])
    manifest = {**internal, "archive_sha256": hashlib.sha256(archive_bytes).hexdigest()}
    raw_manifest = package.canonical_json_bytes(manifest)
    result = package._verify_archive_payload(
        archive_bytes, manifest, expected_archive_sha256=manifest["archive_sha256"],
        expected_manifest_sha256=hashlib.sha256(raw_manifest).hexdigest(),
        manifest_payload=raw_manifest, kind="source",
    )
    assert result["kind"] == "source"
    tampered = _zip([
        ("phase9_geant/example.py", b"modified source", None),
        (package.SOURCE_MANIFEST_MEMBER, package.canonical_json_bytes(internal), None),
    ])
    with pytest.raises(package.BundleError, match="Archive SHA-256 mismatch"):
        package._verify_archive_payload(
            tampered, manifest, expected_archive_sha256=manifest["archive_sha256"],
            expected_manifest_sha256=hashlib.sha256(raw_manifest).hexdigest(),
            manifest_payload=raw_manifest, kind="source",
        )


def test_source_only_packager_writes_review6_hash_record_without_dataset_reads(tmp_path, monkeypatch):
    repository = tmp_path / "synthetic_repo"
    repository.mkdir()
    source = b"synthetic source bundle"
    entry = {**_entry("phase9_geant/synthetic.py", source), "first_party_imports": []}
    internal = {
        "schema": "geant-h96-colab-source-snapshot-v1",
        "entries": [entry],
        "first_party_import_inventory": {entry["path"]: []},
    }
    payload = _zip([
        (entry["path"], source, None),
        (package.SOURCE_MANIFEST_MEMBER, package.canonical_json_bytes(internal), None),
    ])
    manifest = {
        **internal,
        "archive_name": package.SOURCE_BUNDLE_NAME,
        "archive_sha256": hashlib.sha256(payload).hexdigest(),
        "internal_manifest_member": package.SOURCE_MANIFEST_MEMBER,
    }
    monkeypatch.setattr(package, "make_source_bundle", lambda _repo: (payload, manifest))
    monkeypatch.setattr(package, "make_dataset_bundle", lambda *_args: pytest.fail("dataset packaging called"))
    output = tmp_path / "review6"
    result = package.prepare_source_only(repository, output_root=output)
    assert result["dataset_artifacts_opened"] is False
    assert result["source_member_count"] == 1
    assert set(item.name for item in output.iterdir()) == {
        package.SOURCE_BUNDLE_NAME, package.SOURCE_MANIFEST_NAME, "SOURCE_HASHES.txt",
    }
    hashes = (output / "SOURCE_HASHES.txt").read_text(encoding="ascii")
    assert result["source_archive_sha256"] in hashes
    assert result["source_manifest_sha256"] in hashes
    with zipfile.ZipFile(output / package.SOURCE_BUNDLE_NAME) as archive:
        names = archive.namelist()
    assert not any(name.endswith((".npy", ".npz", ".pt")) or "test" in name.casefold() for name in names)
    with pytest.raises(FileExistsError, match="Refusing existing source-only"):
        package.prepare_source_only(repository, output_root=output)


def test_source_snapshot_rejects_missing_first_party_import_dependency(tmp_path):
    package_dir = tmp_path / "phase9_geant"
    tests_dir = package_dir / "tests"
    package_dir.mkdir()
    tests_dir.mkdir()
    init = package_dir / "__init__.py"
    conftest = tests_dir / "conftest.py"
    split_access = package_dir / "split_access.py"
    safe_paths = package_dir / "safe_paths.py"
    init.write_text("", encoding="utf-8")
    conftest.write_text("from phase9_geant.split_access import validate_split_array_name\n", encoding="utf-8")
    split_access.write_text("from phase9_geant.safe_paths import require_regular_file\n", encoding="utf-8")
    safe_paths.write_text("def require_regular_file(*args): pass\n", encoding="utf-8")
    selected = (init, conftest, split_access, safe_paths)
    inventory = package._first_party_import_inventory(tmp_path, selected)
    assert "phase9_geant.split_access" in inventory["phase9_geant/tests/conftest.py"]
    assert "phase9_geant.safe_paths" in inventory["phase9_geant/split_access.py"]
    with pytest.raises(package.BundleError, match="absent from source allowlist"):
        package._first_party_import_inventory(tmp_path, (init, conftest, safe_paths))


def test_external_manifest_tampering_fails_before_extraction(tmp_path):
    source_file = tmp_path / "source.zip"
    manifest_file = tmp_path / "manifest.json"
    source_file.write_bytes(b"synthetic zip bytes")
    manifest_file.write_text('{"archive_sha256":"wrong"}', encoding="utf-8")
    with pytest.raises(package.BundleError, match="External manifest"):
        package.verify_dataset_bundle(source_file, manifest_file, "x" * 64, "y" * 64)


def test_safe_extraction_refuses_existing_destination_and_traversal(tmp_path):
    destination = tmp_path / "existing"
    destination.mkdir()
    with pytest.raises(FileExistsError):
        package._safe_extract(b"", destination, [])
    traversal = _zip([("../outside.txt", b"no", None)])
    with pytest.raises(package.BundleError, match="Unsafe archive path"):
        package._safe_extract(traversal, tmp_path / "fresh", {"../outside.txt"})
    assert not (tmp_path / "outside.txt").exists()


def _write_v2_level(path: Path, *, current: str = "100", maximum: str = "max", high: str = "max"):
    path.mkdir(parents=True, exist_ok=True)
    (path / "memory.current").write_text(current, encoding="ascii")
    (path / "memory.max").write_text(maximum, encoding="ascii")
    (path / "memory.high").write_text(high, encoding="ascii")


def _cgroup_fixture(tmp_path, *, membership="/job", mount_root="/", version="cgroup_v2"):
    root = tmp_path / "visible"
    root.mkdir(parents=True)
    proc = tmp_path / "proc_cgroup"
    mountinfo = tmp_path / "mountinfo"
    if version == "cgroup_v2":
        proc.write_text(f"0::{membership}\n", encoding="ascii")
        base = root
        mountinfo.write_text(f"1 0 0:1 {mount_root} /sys/fs/cgroup rw - cgroup2 cgroup rw\n", encoding="ascii")
    else:
        proc.write_text(f"5:cpu,memory:{membership}\n", encoding="ascii")
        base = root / "memory"
        base.mkdir()
        mountinfo.write_text(f"1 0 0:2 {mount_root} /sys/fs/cgroup/memory rw - cgroup cgroup rw,memory\n", encoding="ascii")
    return root, base, proc, mountinfo


def test_cgroup_v2_and_v1_paths_include_visible_parent_limits(tmp_path):
    root, base, proc, mountinfo = _cgroup_fixture(tmp_path, membership="/parent/job")
    _write_v2_level(base, current="200", maximum="max", high="max")
    _write_v2_level(base / "parent", current="900", maximum="1000", high="950")
    _write_v2_level(base / "parent" / "job", current="100", maximum="max", high="max")
    result = linux_resources.read_cgroup_memory(root, proc_cgroup_path=proc, mountinfo_path=mountinfo)
    assert result["source"] == "cgroup_v2"
    assert result["status"] == "finite"
    assert result["limit_bytes"] == 950
    assert result["headroom_bytes"] == 50  # parent memory.high binds before memory.max

    v1root, v1base, v1proc, v1mount = _cgroup_fixture(tmp_path / "v1", membership="/parent/job", version="cgroup_v1")
    for directory, usage, limit in ((v1base, "500", "max"), (v1base / "parent", "700", "1000"), (v1base / "parent" / "job", "100", "max")):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "memory.usage_in_bytes").write_text(usage, encoding="ascii")
        (directory / "memory.limit_in_bytes").write_text(str(linux_resources.V1_UNLIMITED_SENTINEL if limit == "max" else limit), encoding="ascii")
    v1result = linux_resources.read_cgroup_memory(v1root, proc_cgroup_path=v1proc, mountinfo_path=v1mount)
    assert v1result["source"] == "cgroup_v1"
    assert v1result["status"] == "finite"
    assert v1result["headroom_bytes"] == 300


def test_literal_unlimited_visible_cgroup_is_distinct_from_hidden_ancestors(tmp_path):
    root, base, proc, mountinfo = _cgroup_fixture(
        tmp_path, membership="/../../jupyter-children", mount_root="/",
    )
    _write_v2_level(base, current=str(1024**3), maximum="max", high="max")
    result = linux_resources.read_cgroup_memory(root, proc_cgroup_path=proc, mountinfo_path=mountinfo)
    assert result["status"] == "visible_unlimited"
    assert result["headroom_bytes"] is None
    assert any("traversal components" in item for item in result["visibility_uncertainty"])
    assert any("outside the visible" in item for item in result["visibility_uncertainty"])
    assert result["visible_levels"][0]["usage_bytes"] == 1024**3


@pytest.mark.parametrize("bad_field,bad_value", [
    ("memory.current", "unknown"), ("memory.max", "unlimited"),
    ("memory.high", "-1"),
])
def test_cgroup_v2_malformed_essential_values_fail_closed(tmp_path, bad_field, bad_value):
    root, base, proc, mountinfo = _cgroup_fixture(tmp_path)
    _write_v2_level(base)
    _write_v2_level(base / "job")
    (base / bad_field).write_text(bad_value, encoding="ascii")
    result = linux_resources.read_cgroup_memory(root, proc_cgroup_path=proc, mountinfo_path=mountinfo)
    assert result["status"] == "invalid"
    assert result["headroom_bytes"] is None


def test_cgroup_v2_missing_essential_limit_fails_closed(tmp_path):
    root, base, proc, mountinfo = _cgroup_fixture(tmp_path)
    (base / "memory.current").write_text("1", encoding="ascii")
    (base / "memory.max").write_text("max", encoding="ascii")
    result = linux_resources.read_cgroup_memory(root, proc_cgroup_path=proc, mountinfo_path=mountinfo)
    assert result["status"] == "invalid"
    assert "memory.high" in result["visibility_uncertainty"][-1]


def test_cgroup_missing_membership_and_unreadable_mount_mapping_fail_closed(tmp_path):
    root, base, proc, mountinfo = _cgroup_fixture(tmp_path)
    _write_v2_level(base)
    _write_v2_level(base / "job")
    missing = linux_resources.read_cgroup_memory(root, proc_cgroup_path=tmp_path / "absent", mountinfo_path=mountinfo)
    assert missing["status"] == "invalid"
    badmount = tmp_path / "bad_mountinfo"
    badmount.write_text("not mountinfo\n", encoding="ascii")
    malformed_table = linux_resources.read_cgroup_memory(root, proc_cgroup_path=proc, mountinfo_path=badmount)
    assert malformed_table["status"] == "invalid"
    # A genuinely unavailable mount table permits only direct visible-root
    # inspection, with the mapping uncertainty retained in the result.
    fallback = linux_resources.read_cgroup_memory(root, proc_cgroup_path=proc, mountinfo_path=tmp_path / "unreadable_mountinfo")
    assert fallback["status"] == "visible_unlimited"
    assert any("mountinfo" in item for item in fallback["visibility_uncertainty"])


def test_unreadable_mountinfo_accepts_explicit_visible_root_membership(tmp_path):
    root = tmp_path / "visible"
    _write_v2_level(root)
    proc = tmp_path / "proc_cgroup"
    proc.write_text("0::/\n", encoding="ascii")
    result = linux_resources.read_cgroup_memory(
        root, proc_cgroup_path=proc, mountinfo_path=tmp_path / "missing_mountinfo",
    )
    assert result["status"] == "visible_unlimited"
    assert len(result["visible_levels"]) == 1


def _resource_snapshot(*, high=500 * 1024**2, available=4 * 1024**3,
                       cgroup_headroom=4 * 1024**3, disk=2 * 1024**3):
    unlimited = cgroup_headroom is None
    return {
        "process_rss_bytes": 350 * 1024**2,
        "process_peak_bytes": high,
        "process_peak_source": "synthetic ru_maxrss",
        "os_available_ram_bytes": available,
        "disk_free_bytes": disk,
        "cgroup_source": "synthetic cgroup_v2",
        "cgroup_status": "visible_unlimited" if unlimited else "finite",
        "cgroup_memory_usage_bytes": 1,
        "cgroup_memory_limit_bytes": None if unlimited else cgroup_headroom + 1,
        "cgroup_memory_headroom_bytes": cgroup_headroom,
        "cgroup_visibility_uncertainty": ["synthetic hidden parent"],
        "cgroup_visible_levels": [{
            "path": "synthetic-cgroup", "usage_bytes": 1,
            "limits": ([
                {"name": "memory.max", "limit_bytes": None, "kind": "unlimited"},
                {"name": "memory.high", "limit_bytes": None, "kind": "unlimited"},
            ] if unlimited else [
                {"name": "memory.max", "limit_bytes": cgroup_headroom + 1, "kind": "finite"},
                {"name": "memory.high", "limit_bytes": cgroup_headroom + 1, "kind": "finite"},
            ]),
        }],
        "cgroup_memory_events": None,
    }


def test_linux_resource_preflight_and_monitor_enforce_separate_limits():
    passed = linux_resources.preflight(snapshot_fn=lambda: _resource_snapshot())
    assert passed["status"] == "pass"
    assert passed["policy"]["policy_id"] == "colab-linux-process-memory-v3-visible-cgroup"
    assert passed["policy"]["process_peak_limit_bytes"] == 2 * 1024**3
    assert passed["policy"]["startup_headroom_bytes"] == (2 * 1024**3) + (384 * 1024**2)
    with pytest.raises(linux_resources.LinuxResourceError, match="OS-available"):
        linux_resources.preflight(snapshot_fn=lambda: _resource_snapshot(available=2 * 1024**3))
    with pytest.raises(linux_resources.LinuxResourceError, match="Process high-water exceeds"):
        linux_resources.preflight(snapshot_fn=lambda: _resource_snapshot(high=2 * 1024**3 + 1))
    over_limit_monitor = linux_resources.LinuxMemoryMonitor(
        snapshot_fn=lambda: _resource_snapshot(high=2 * 1024**3 + 1),
    )
    with pytest.raises(linux_resources.LinuxResourceError, match="Process high-water exceeds"):
        over_limit_monitor.sample("train_epoch_1_batch")
    with pytest.raises(linux_resources.LinuxResourceError, match="Cgroup memory headroom"):
        linux_resources.preflight(snapshot_fn=lambda: _resource_snapshot(cgroup_headroom=2 * 1024**3))
    monitor = linux_resources.LinuxMemoryMonitor(snapshot_fn=lambda: _resource_snapshot(cgroup_headroom=200 * 1024**2))
    with pytest.raises(linux_resources.LinuxResourceError, match="Cgroup memory headroom"):
        monitor.sample("synthetic-batch")
    assert monitor.summary()["min_cgroup_headroom_bytes"] == 200 * 1024**2


def test_colab_resource_policy_allows_visible_unlimited_but_fails_bad_state_or_high_water():
    passed = linux_resources.preflight(snapshot_fn=lambda: _resource_snapshot(cgroup_headroom=None))
    assert passed["status"] == "pass"
    assert passed["snapshot"]["cgroup_status"] == "visible_unlimited"
    with pytest.raises(linux_resources.LinuxResourceError, match="high-water is unavailable"):
        linux_resources.preflight(snapshot_fn=lambda: _resource_snapshot(high=None))
    malformed = _resource_snapshot()
    malformed["cgroup_status"] = "invalid"
    malformed["cgroup_visibility_uncertainty"] = ["synthetic malformed cgroup.current"]
    with pytest.raises(linux_resources.LinuxResourceError, match="Cgroup measurements are invalid"):
        linux_resources.preflight(snapshot_fn=lambda: malformed)
    monitor = linux_resources.LinuxMemoryMonitor(snapshot_fn=lambda: _resource_snapshot(cgroup_headroom=None))
    assert monitor.sample("synthetic-batch")["cgroup_status"] == "visible_unlimited"
    assert monitor.summary()["last_snapshot"]["cgroup_memory_headroom_bytes"] is None


def test_unlimited_cgroup_still_enforces_host_process_and_disk_limits():
    with pytest.raises(linux_resources.LinuxResourceError, match="OS-available memory"):
        linux_resources.preflight(snapshot_fn=lambda: _resource_snapshot(cgroup_headroom=None, available=1))
    with pytest.raises(linux_resources.LinuxResourceError, match="Process high-water exceeds"):
        linux_resources.preflight(snapshot_fn=lambda: _resource_snapshot(cgroup_headroom=None, high=2 * 1024**3 + 1))
    with pytest.raises(linux_resources.LinuxResourceError, match="Disk free space"):
        linux_resources.preflight(snapshot_fn=lambda: _resource_snapshot(cgroup_headroom=None, disk=1))
    monitor = linux_resources.LinuxMemoryMonitor(
        snapshot_fn=lambda: _resource_snapshot(
            cgroup_headroom=None, available=linux_resources.OS_AVAILABLE_RESERVE_BYTES - 1,
        ),
    )
    with pytest.raises(linux_resources.LinuxResourceError, match="OS-available memory is below reserve"):
        monitor.sample("train_epoch_1_batch")


def test_startup_and_in_run_apply_distinct_documented_reserves():
    startup = linux_resources.LinuxMemoryMonitor(
        snapshot_fn=lambda: _resource_snapshot(available=2 * 1024**3, cgroup_headroom=4 * 1024**3),
    )
    with pytest.raises(linux_resources.LinuxResourceError, match="startup headroom"):
        startup.sample("startup_before_output")
    in_run = linux_resources.LinuxMemoryMonitor(
        snapshot_fn=lambda: _resource_snapshot(available=500 * 1024**2, cgroup_headroom=500 * 1024**2),
    )
    assert in_run.sample("train_epoch_1_batch")["os_available_ram_bytes"] == 500 * 1024**2


def test_resource_monitor_rejects_incomplete_or_malformed_snapshots():
    malformed = _resource_snapshot()
    malformed.pop("cgroup_memory_headroom_bytes")
    with pytest.raises(linux_resources.LinuxResourceError, match="snapshot is incomplete"):
        linux_resources.preflight(snapshot_fn=lambda: malformed)
    with pytest.raises(linux_resources.LinuxResourceError, match="snapshot is incomplete"):
        linux_resources.LinuxMemoryMonitor(snapshot_fn=lambda: malformed).sample("synthetic-batch")


def test_runtime_pins_and_training_authorization_are_fail_closed(monkeypatch, tmp_path):
    assert environment.PINNED_VERSIONS == {"numpy": "1.26.4", "torch": "2.12.0", "psutil": "7.2.2", "pytest": "9.1.1"}
    monkeypatch.setattr(environment, "collect_environment", lambda: {
        "python_major_minor": [3, 12], "numpy": "1.26.4", "torch_base_version": "2.12.0",
        "psutil": "7.2.2", "pytest": "9.1.1",
    })
    assert environment.verify_environment()["torch_base_version"] == "2.12.0"
    monkeypatch.setattr(environment, "collect_environment", lambda: {
        "python_major_minor": [3, 12], "numpy": "9.9", "torch_base_version": "2.12.0",
        "psutil": "7.2.2", "pytest": "9.1.1",
    })
    with pytest.raises(environment.EnvironmentMismatch, match="numpy expected"):
        environment.verify_environment()
    args = [
        "--run-training", "--repository-root", str(tmp_path), "--source-bundle", str(tmp_path / "s.zip"),
        "--source-manifest", str(tmp_path / "s.json"), "--expected-source-bundle-sha256", "a" * 64,
        "--expected-source-manifest-sha256", "b" * 64, "--dataset-bundle", str(tmp_path / "d.zip"),
        "--dataset-manifest", str(tmp_path / "d.json"), "--expected-dataset-bundle-sha256", "c" * 64,
        "--expected-dataset-manifest-sha256", "d" * 64,
    ]
    with pytest.raises(SystemExit, match="separately approved"):
        launch.main(args)
    with pytest.raises(SystemExit):
        launch.build_parser().parse_args(["--repository-root", str(tmp_path)])


def test_launcher_binds_colab_policy_and_uses_new_output_namespace():
    trainer = SimpleNamespace(
        CONFIG={"memory_policy": {"workers": 0, "old_limit": 640}},
        lazy=SimpleNamespace(SMOKE_ESTIMATED_PEAK_RAM=640 * 1024**2),
        PROCESS_PEAK_LIMIT_BYTES=640 * 1024**2,
        SYSTEM_AVAILABLE_RESERVE_BYTES=384 * 1024**2,
    )
    policy = {
        "policy_id": linux_resources.RESOURCE_POLICY_ID,
        "process_peak_limit_bytes": linux_resources.PROCESS_PEAK_LIMIT_BYTES,
        "startup_headroom_bytes": linux_resources.STARTUP_HEADROOM_BYTES,
        "in_run_os_available_reserve_bytes": linux_resources.OS_AVAILABLE_RESERVE_BYTES,
        "in_run_cgroup_headroom_reserve_bytes": linux_resources.OS_AVAILABLE_RESERVE_BYTES,
        "minimum_free_disk_bytes": linux_resources.MIN_FREE_DISK_BYTES,
    }
    launch._bind_resource_policy(trainer, policy)
    assert trainer.CONFIG["memory_policy"]["policy_id"] == linux_resources.RESOURCE_POLICY_ID
    assert trainer.CONFIG["memory_policy"]["process_peak_limit_bytes"] == 2 * 1024**3
    assert trainer.CONFIG["linux_resource_policy"] == policy
    assert trainer.lazy.SMOKE_ESTIMATED_PEAK_RAM == 2 * 1024**3
    assert trainer.PROCESS_PEAK_LIMIT_BYTES == 2 * 1024**3
    assert launch.COLAB_OUTPUT_ROOT.endswith("_memv3")
    assert package.OUTPUT_RELATIVE.as_posix().endswith("geant_h96_colab_v1_review6_final3")
    with pytest.raises(launch.LaunchIntegrityError, match="Unexpected Colab resource policy"):
        launch._bind_resource_policy(trainer, {**policy, "policy_id": "wrong"})
    with pytest.raises(launch.LaunchIntegrityError, match="values differ"):
        launch._bind_resource_policy(trainer, {**policy, "process_peak_limit_bytes": 640 * 1024**2})


def test_source_verification_failure_precedes_any_project_import(monkeypatch, tmp_path):
    source_archive = tmp_path / "source.zip"
    source_manifest = tmp_path / "source.json"
    source_archive.write_bytes(b"bad archive")
    source_manifest.write_text("{}", encoding="utf-8")
    called = []
    monkeypatch.setattr(launch, "_import_verified_runtime", lambda *_args: called.append("import"))
    with pytest.raises(launch.LaunchIntegrityError, match="Source manifest differs"):
        launch._verify_source_before_import(
            tmp_path, source_archive, source_manifest,
            expected_archive_sha256="0" * 64, expected_manifest_sha256="1" * 64,
        )
    assert called == []


def test_loaded_project_modules_must_all_be_in_the_approved_snapshot(tmp_path):
    root = tmp_path / "approved"
    allowed = root / "phase9_geant" / "entry.py"
    allowed.parent.mkdir(parents=True)
    allowed.write_text("# synthetic", encoding="utf-8")
    unexpected = tmp_path / "shadow" / "phase9_geant" / "shadow.py"
    unexpected.parent.mkdir(parents=True)
    unexpected.write_text("# synthetic", encoding="utf-8")
    approved_module = SimpleNamespace(__file__=str(allowed))
    launch._verify_loaded_project_modules(root, {allowed.resolve()}, modules={"phase9_geant.synthetic": approved_module})
    with pytest.raises(launch.LaunchIntegrityError, match="Unexpected loaded project module"):
        launch._verify_loaded_project_modules(root, {allowed.resolve()}, modules={
            "phase9_geant.synthetic": approved_module,
            "phase9_geant.shadow": SimpleNamespace(__file__=str(unexpected)),
        })


def test_loaded_project_module_not_listed_is_rejected_even_under_bundle_root(tmp_path):
    root = tmp_path / "approved"
    unlisted = root / "src" / "extra.py"
    unlisted.parent.mkdir(parents=True)
    unlisted.write_text("# synthetic", encoding="utf-8")
    with pytest.raises(launch.LaunchIntegrityError, match="not in the approved manifest"):
        launch._verify_loaded_project_modules(root, set(), modules={"src.extra": SimpleNamespace(__file__=str(unlisted))})


def test_preloaded_project_module_is_rejected_before_bundle_import():
    with pytest.raises(launch.LaunchIntegrityError, match="imported before source verification"):
        launch._assert_clean_project_import_state(modules={"src.gnn.shadow": object()})


def test_controlled_src_namespace_excludes_site_package_shadow_paths(tmp_path):
    root = tmp_path / "approved"
    gnn_init = root / "src" / "gnn" / "__init__.py"
    gnn_init.parent.mkdir(parents=True)
    gnn_init.write_text("", encoding="utf-8")
    shadow = tmp_path / "site-packages" / "src"
    shadow.mkdir(parents=True)
    (shadow / "unexpected.py").write_text("VALUE = 1\n", encoding="utf-8")
    modules = {}
    namespace = launch._install_controlled_src_namespace(root, modules=modules)
    assert namespace.__path__ == [str((root / "src").resolve())]
    launch._verify_loaded_project_modules(root, {gnn_init.resolve()}, modules=modules)
    namespace.__path__.append(str(shadow.resolve()))
    with pytest.raises(launch.LaunchIntegrityError, match="Unexpected namespace package locations"):
        launch._verify_loaded_project_modules(root, {gnn_init.resolve()}, modules=modules)
