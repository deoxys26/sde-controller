from __future__ import annotations

import hashlib
import json
import stat
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from datetime import timedelta

from phase9_geant import horizon_v2_data as data
from phase9_geant import horizon_v2_experiment as experiment


EDGES = (("A", "B"), ("B", "A"))


def _synthetic_dataset(tmp_path):
    repo = tmp_path / "repo"
    (repo / "phase9_geant").mkdir(parents=True)
    (repo / "data" / "processed").mkdir(parents=True)
    root = repo / "data" / "processed" / f"{data.OUTPUT_PREFIX}synthetic"
    root.mkdir()
    edges = tuple((f"source-{i}", f"target-{i}") for i in range(72))
    names_by_split = {
        split: [
            f"demandMatrix-geant-uhlig-15min-{(data.EXPECTED_START + index * data.STEP).strftime('%Y%m%d-%H%M')}.txt"
            for index in range(*bounds)
        ]
        for split, bounds in data.OFFICIAL_SPLIT_BOUNDS.items()
    }
    artifact_names = sorted(data._ARRAY_NAMES | data._JSON_NAMES)
    split_records = {}
    window_indexes = {}
    arrays = {}
    for split, bounds in data.OFFICIAL_SPLIT_BOUNDS.items():
        split_records[split] = {
            "bounds": list(bounds),
            "source_file_names": names_by_split[split],
            "parsed_source_semantic_sha256": "b" * 64,
            "horizons": {},
        }
        window_indexes[split] = {}
        for horizon in data.HORIZONS:
            origins = [bounds[0] + 671, bounds[0] + 672]
            history = np.full((2, 672, 72), 0.25 + horizon / 1000, dtype=np.float32)
            target = np.full((2, 72), 0.3 + horizon / 1000, dtype=np.float32)
            arrays[(split, "history", horizon)] = history
            arrays[(split, "target", horizon)] = target
            split_records[split]["horizons"][str(horizon)] = {
                "history_shape": list(history.shape),
                "target_shape": list(target.shape),
                "history_sha256": hashlib.sha256(history.tobytes()).hexdigest(),
                "target_sha256": hashlib.sha256(target.tobytes()).hexdigest(),
            }
            window_indexes[split][str(horizon)] = [
                {"history_start": origin - 671, "origin": origin, "target": origin + horizon}
                for origin in origins
            ]
    for (split, kind, horizon), array in arrays.items():
        np.save(root / f"{kind}_{split}_h{horizon}.npy", array, allow_pickle=False)
    for split, index in window_indexes.items():
        (root / f"window_index_{split}.json").write_text(json.dumps(index), encoding="utf-8")
    source_hashes = {
        "phase9_geant/horizon_v2_data.py": "c" * 64,
        "phase9_geant/horizon_windows.py": "c" * 64,
        "phase9_geant/validity_v2.py": "c" * 64,
        "phase9_geant/parse_geant.py": "c" * 64,
        "phase9_geant/build_link_series.py": "c" * 64,
    }
    metadata = {
        "protocol_id": data.PROTOCOL_ID,
        "status": "complete",
        "source_scope": {split: list(bounds) for split, bounds in data.OFFICIAL_SPLIT_BOUNDS.items()},
        "source_hashes_at_build": source_hashes,
        "build_configuration": data.BUILD_CONFIGURATION,
        "build_configuration_sha256": hashlib.sha256(
            json.dumps(data.BUILD_CONFIGURATION, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "window": {"history_length": 672, "horizons": [4, 96], "step_seconds": 900},
        "topology": {
            "nodes": [f"node-{i}" for i in range(22)],
            "undirected_links": 36,
            "directed_links": 72,
            "sha256_local_file_identity_only": "a" * 64,
            "directed_edges_interleaved": [list(edge) for edge in edges],
            "ordered_edge_sha256": data._edge_digest(edges),
        },
        "splits": split_records,
        "artifact_names": artifact_names,
    }
    (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return repo, root, arrays


def _patch_topology(monkeypatch, edges):
    def synthetic_topology(_path, _repo):
        return {f"node-{i}": (0.0, 0.0) for i in range(22)}, [("a", "b")] * 36, edges, "a" * 64

    monkeypatch.setattr(experiment, "_topology_snapshot", synthetic_topology)


def _patch_loader_source_identity(monkeypatch):
    expected = {
        "phase9_geant/horizon_v2_data.py": "c" * 64,
        "phase9_geant/horizon_windows.py": "c" * 64,
        "phase9_geant/validity_v2.py": "c" * 64,
        "phase9_geant/parse_geant.py": "c" * 64,
        "phase9_geant/build_link_series.py": "c" * 64,
    }
    monkeypatch.setattr(data, "_current_build_source_hashes", lambda _repo: expected.copy())
    monkeypatch.setattr(experiment, "_current_build_source_hashes", data._current_build_source_hashes)
    return expected


def test_split_path_selection_constructs_only_train_or_validation_names(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    raw = repo / data.RAW_DIRECTORY_RELATIVE
    raw.mkdir(parents=True)
    monkeypatch.setattr(data, "OFFICIAL_SPLIT_BOUNDS", {"train": (0, 3), "validation": (3, 5), "test": (5, 8)})
    names = [
        f"demandMatrix-geant-uhlig-15min-{(data.EXPECTED_START + i * data.STEP):%Y%m%d-%H%M}.txt"
        for i in range(5)
    ]
    entries = [_FakeDirEntry(name) for name in names]
    scans = []
    monkeypatch.setattr(data.os, "scandir", lambda _path: _FakeScan(entries, scans))
    selected = data.expected_split_paths_for_splits(raw, ("train", "validation"), repository_root=repo)
    assert len(scans) == 1
    assert len(selected["train"]) == 3
    assert len(selected["validation"]) == 2
    assert [p.name for p in selected["train"]] == names[:3]
    assert [p.name for p in selected["validation"]] == names[3:]


class _FakeDirEntry:
    def __init__(self, name, *, mode=stat.S_IFREG | 0o600, attributes=0):
        self.name = name
        self.mode = mode
        self.attributes = attributes
        self.stat_calls = 0

    def stat(self, *, follow_symlinks=True):
        self.stat_calls += 1
        return SimpleNamespace(st_mode=self.mode, st_file_attributes=self.attributes)

    def is_symlink(self):
        return False


class _FakeScan:
    def __init__(self, entries, calls):
        self.entries = entries
        self.calls = calls

    def __enter__(self):
        self.calls.append("scan")
        return iter(self.entries)

    def __exit__(self, *_args):
        return False


def _small_source_fixture(tmp_path, monkeypatch, bounds=(0, 3), entries=None):
    repo = tmp_path / "repo"
    (repo / "phase9_geant").mkdir(parents=True)
    raw = repo / data.RAW_DIRECTORY_RELATIVE
    raw.mkdir(parents=True)
    monkeypatch.setattr(data, "OFFICIAL_SPLIT_BOUNDS", {"train": bounds, "validation": (3, 5), "test": (5, 8)})
    names = [
        f"demandMatrix-geant-uhlig-15min-{(data.EXPECTED_START + i * data.STEP):%Y%m%d-%H%M}.txt"
        for i in range(8)
    ]
    entries = list(entries) if entries is not None else [_FakeDirEntry(name) for name in names[:bounds[1]]]
    scans = []
    monkeypatch.setattr(data.os, "scandir", lambda _path: _FakeScan(entries, scans))
    return repo, raw, names, entries, scans


def _synthetic_matrix_names():
    return [
        f"demandMatrix-geant-uhlig-15min-{(data.EXPECTED_START + i * data.STEP):%Y%m%d-%H%M}.txt"
        for i in range(8)
    ]


def test_single_scan_sorts_out_of_order_and_does_not_stat_nonselected_matrix_names(tmp_path, monkeypatch):
    names = _synthetic_matrix_names()
    repo, raw, names, entries, scans = _small_source_fixture(
        tmp_path,
        monkeypatch,
        entries=[_FakeDirEntry(names[2]), _FakeDirEntry(names[0]), _FakeDirEntry(names[1]), _FakeDirEntry(names[7])],
    )
    audit = {}
    result = data.expected_split_paths(raw, "train", repository_root=repo, access_audit=audit)
    assert [path.name for path in result] == names[:3]
    assert len(scans) == 1
    assert [entry.stat_calls for entry in entries[:3]] == [1, 1, 1]
    assert entries[3].stat_calls == 0
    assert audit["directory_scans"] == 1
    assert audit["directory_entry_names_enumerated"] == 4
    assert audit["nonselected_geant_matrix_names_seen_without_per_file_stat"] == 1
    assert audit["selected_file_metadata_checks"] == 3
    assert audit["content_reads_during_preflight"] == 0
    assert audit["filename_preflight_seconds"] >= 0


def test_single_scan_fails_closed_for_missing_file_duplicate_name_and_unknown_entry(tmp_path, monkeypatch):
    names = _synthetic_matrix_names()
    repo, raw, names, entries, _ = _small_source_fixture(tmp_path, monkeypatch, entries=[_FakeDirEntry(names[0])])
    with pytest.raises(FileNotFoundError, match="Missing"):
        data.expected_split_paths(raw, "train", repository_root=repo)

    repo, raw, names, entries, _ = _small_source_fixture(
        tmp_path / "duplicate", monkeypatch,
        entries=[_FakeDirEntry(names[0]), _FakeDirEntry(names[0]), _FakeDirEntry(names[1]), _FakeDirEntry(names[2])],
    )
    with pytest.raises(ValueError, match="Duplicate/case-colliding"):
        data.expected_split_paths(raw, "train", repository_root=repo)

    repo, raw, names, entries, _ = _small_source_fixture(
        tmp_path / "unknown", monkeypatch,
        entries=[_FakeDirEntry(name) for name in names[:3]] + [_FakeDirEntry("unrelated.txt")],
    )
    with pytest.raises(ValueError, match="Unexpected entry"):
        data.expected_split_paths(raw, "train", repository_root=repo)


def test_single_scan_rejects_selected_reparse_point_before_return(tmp_path, monkeypatch):
    names = _synthetic_matrix_names()
    repo, raw, names, entries, _ = _small_source_fixture(
        tmp_path,
        monkeypatch,
        entries=[_FakeDirEntry(names[0]), _FakeDirEntry(names[1]), _FakeDirEntry(names[2])],
    )
    entries[1].attributes = 0x400
    with pytest.raises(ValueError, match="redirected or not a regular file"):
        data.expected_split_paths(raw, "train", repository_root=repo)


def test_single_scan_slow_directory_is_scanned_once_and_timing_is_recorded(tmp_path, monkeypatch):
    repo, raw, names, entries, scans = _small_source_fixture(tmp_path, monkeypatch)
    # Keep the fixture's enumeration synthetic while delaying that one operation.
    monkeypatch.setattr(data.os, "scandir", lambda _path: _DelayedFakeScan(entries, scans, 0.01))
    audit = {}
    data.expected_split_paths(raw, "train", repository_root=repo, access_audit=audit)
    assert len(scans) == 1
    assert audit["directory_scans"] == 1
    assert audit["filename_preflight_seconds"] >= 0.01


class _DelayedFakeScan(_FakeScan):
    def __init__(self, entries, calls, delay_seconds):
        super().__init__(entries, calls)
        self.delay_seconds = delay_seconds

    def __enter__(self):
        time.sleep(self.delay_seconds)
        return super().__enter__()


def test_training_reader_selects_anchored_train_files_before_parser_calls():
    from phase9_geant import validity_v2

    raw = __import__("pathlib").Path("synthetic-raw")
    paths = tuple(
        raw / f"demandMatrix-geant-uhlig-15min-{(data.EXPECTED_START + i * data.STEP):%Y%m%d-%H%M}.txt"
        for i in range(8020)
    )
    parser_paths = []
    nodes = ("A", "B")
    node_to_index = {"A": 0, "B": 1}
    shared = np.zeros((2, 2), dtype=np.float64)

    def parser(path, _nodes, _mapping):
        parser_paths.append(path)
        stamp = validity_v2.timestamp_from_filename(path)
        return validity_v2.MatrixObservation(stamp, shared, 1, path.name)

    result = validity_v2.read_training_period(
        raw, nodes, node_to_index,
        expected_start_timestamp=data.EXPECTED_START,
        expected_start_timestamp_source="synthetic fixture anchor",
        discovery=lambda _directory: paths,
        parser=parser,
    )
    assert tuple(parser_paths) == paths
    assert len(parser_paths) == 8020
    assert result.metadata["training_indices"] == list(range(8020))
    assert result.metadata["threshold_scope"] == "nonempty training matrices only"
    assert all("test" not in path.name.casefold() for path in parser_paths)


def test_test_split_is_rejected_before_any_path_validation(tmp_path, monkeypatch):
    called = False

    def forbidden(*_args):
        nonlocal called
        called = True
        raise AssertionError("test split must be rejected before source enumeration")

    monkeypatch.setattr(data.os, "scandir", forbidden)
    with pytest.raises(ValueError, match="test is unsupported"):
        data.expected_split_paths(tmp_path, "test", repository_root=tmp_path)
    assert not called


@pytest.mark.parametrize("name", [
    "history_test_h4.npy",
    "target_Test_h96.npy",
    "unknown_train_h4.npy",
    "../history_train_h4.npy",
])
def test_horizon_artifact_allowlist_rejects_test_unknown_and_aliases(name):
    with pytest.raises(ValueError):
        data.validate_artifact_name(name)


def test_horizon_artifact_allowlist_accepts_only_fixed_train_validation_names():
    for name in sorted(data._ARRAY_NAMES | data._JSON_NAMES):
        assert data.validate_artifact_name(name) == name


def test_invalid_loader_name_is_rejected_before_array_loader(tmp_path):
    called = False

    def loader(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("loader must not be called")

    with pytest.raises(ValueError, match="test"):
        data.load_horizon_array(tmp_path, "history_test_h4.npy", repository_root=tmp_path, array_loader=loader)
    assert not called


def test_ridge_features_use_only_history_and_keep_edge_order():
    rng = np.random.default_rng(41)
    history = rng.normal(size=(4, 672, len(EDGES)))
    features4, baseline4 = experiment.build_ridge_features(history, EDGES, 4)
    features96, baseline96 = experiment.build_ridge_features(history, EDGES, 96)
    assert features4.shape == (4, 2, 4)
    assert features96.shape == (4, 2, 4)
    assert baseline4.shape == baseline96.shape == (4, 2)
    assert np.isfinite(features4).all() and np.isfinite(features96).all()
    changed = history.copy()
    changed[:, 0, :] += 1000.0  # outside every seasonal/EWMA/neighbor-change source
    changed_features, changed_baseline = experiment.build_ridge_features(changed, EDGES, 4)
    np.testing.assert_array_equal(features4, changed_features)
    np.testing.assert_array_equal(baseline4, changed_baseline)


def test_ridge_screen_uses_chronological_fit_prefix_and_embargo():
    rng = np.random.default_rng(7)
    n = 100
    history = rng.uniform(0, 1, size=(n, 672, len(EDGES)))
    target = rng.uniform(0, 1, size=(n, len(EDGES)))
    origins = np.arange(n, dtype=np.int64) * 32 + 671
    result = experiment.chronological_ridge_screen(history, target, origins, EDGES, 4)
    assert result.fit_origins == int(np.floor(n * 0.7))
    first_fit = origins[result.fit_origins - 1]
    expected_holdout = np.count_nonzero(
        origins[result.fit_origins:] >= first_fit + experiment.SCREEN_EMBARGO_STEPS
    )
    assert result.holdout_origins == expected_holdout
    assert result.embargo_steps == 768
    assert result.holdout_pairs == result.holdout_origins * len(EDGES)
    assert 0 <= result.holdout_origins < n - result.fit_origins
    assert np.isfinite([result.ewma_mae, result.ridge_ewma_mae, result.ewma_rmse, result.ridge_ewma_rmse]).all()


def test_ridge_screen_rejects_nonchronological_or_nonfinite_data():
    history = np.zeros((5, 672, 2), dtype=np.float64)
    target = np.zeros((5, 2), dtype=np.float64)
    origins = np.arange(5)
    with pytest.raises(ValueError, match="strictly chronological"):
        experiment.chronological_ridge_screen(history, target, origins[::-1], EDGES, 4)
    history[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        experiment.chronological_ridge_screen(history, target, origins, EDGES, 4)


def test_series_feature_construction_does_not_use_future_target_as_an_input():
    series = np.tile(np.linspace(0.05, 0.95, 8020, dtype=np.float64)[:, None], (1, 2))
    origins = np.asarray([671, 1200])
    features_before, residual_before = experiment.ridge_features_from_training_series(
        series, origins, EDGES, 4
    )
    changed = series.copy()
    changed[origins[0] + 4, 0] += 3.0
    features_after, residual_after = experiment.ridge_features_from_training_series(
        changed, origins, EDGES, 4
    )
    np.testing.assert_array_equal(features_before, features_after)
    assert not np.array_equal(residual_before[0], residual_after[0])


def test_baselines_align_to_each_horizon_and_metrics_reject_missing_pairs():
    history = np.arange(672, dtype=np.float64)[None, :, None]
    h4 = experiment.baseline_forecasts(history, 4)
    h96 = experiment.baseline_forecasts(history, 96)
    assert h4["same_time_yesterday"][0, 0] == 579
    assert h4["same_time_last_week"][0, 0] == 3
    assert h96["same_time_yesterday"][0, 0] == 671
    assert h96["same_time_last_week"][0, 0] == 95
    summary = experiment.metric_summary(np.asarray([0.0, 1.0]), np.asarray([0.0, 2.0]))
    assert summary["evaluated_target_link_pairs"] == 2
    assert summary["mae"] == 0.5
    with pytest.raises(ValueError, match="nonfinite"):
        experiment.metric_summary(np.asarray([0.0, np.nan]), np.asarray([0.0, 0.0]))


def test_checkpoint_roundtrip_uses_only_synthetic_tensors(tmp_path):
    path = tmp_path / "synthetic.pt"
    expected = {"weight": torch.arange(6, dtype=torch.float32).reshape(2, 3)}
    experiment._atomic_torch_save(expected, path)
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    torch.testing.assert_close(loaded["weight"], expected["weight"])


def test_training_gate_defaults_disabled_before_any_data_path_access(tmp_path):
    with pytest.raises(PermissionError, match="gate is disabled"):
        experiment.train_from_horizon_root(
            tmp_path / "not-a-dataset",
            repository_root=tmp_path,
            seed=42,
            adjacency_mode="topology",
        )


def test_joint_candidate_training_refuses_stopped_h4_before_path_or_array_access(monkeypatch, tmp_path):
    def unexpected(*_args, **_kwargs):
        raise AssertionError("path validation or data loading must not run for the stopped joint objective")

    monkeypatch.setattr(experiment, "canonical_repository_root", unexpected)
    with pytest.raises(PermissionError, match="h=4 as STOP"):
        experiment.train_from_horizon_root(
            tmp_path / "synthetic-placeholder",
            repository_root=tmp_path,
            seed=42,
            adjacency_mode="topology",
            enabled=True,
        )


def test_low_level_joint_trainer_refuses_stopped_h4_before_output_creation(tmp_path):
    output = tmp_path / "must-not-exist"
    with pytest.raises(PermissionError, match="h=4 is STOP"):
        experiment._train_arrays(
            histories={}, targets={}, validation_histories={}, validation_targets={},
            edge_order=EDGES, output_directory=output, seed=42, adjacency_mode="identity",
            run_provenance={"data_root": str(tmp_path)},
        )
    assert not output.exists()


def test_checkpoint_selection_is_validation_minimum_only():
    assert experiment.validation_loss_improved(0.4, float("inf"))
    assert experiment.validation_loss_improved(0.3, 0.4)
    assert not experiment.validation_loss_improved(0.4, 0.4)
    assert not experiment.validation_loss_improved(0.5, 0.4)
    assert not experiment.validation_loss_improved(float("nan"), 0.4)


def test_training_scaler_is_fitted_from_supplied_training_arrays_only():
    x = {4: np.full((3, 672, 2), 2.0), 96: np.full((2, 672, 2), 4.0)}
    y = {4: np.full((3, 2), 6.0), 96: np.full((2, 2), 8.0)}
    first = experiment.fit_training_scaler(x, y)
    # No validation argument exists in the API, so changing a holdout fixture
    # cannot influence the fit.
    validation_a = np.full((1, 672, 2), -1000.0)
    validation_b = np.full((1, 672, 2), 1000.0)
    second = experiment.fit_training_scaler(x, y)
    assert validation_a.mean() != validation_b.mean()
    assert first == second


def test_training_scaler_streaming_matches_population_reference_across_chunks():
    # Cross the scaler's fixed chunk boundary with synthetic values; compare
    # against the former exact population-statistic definition.
    rng = np.random.default_rng(4815)
    histories = {4: rng.normal(0.03, 0.02, size=(800, 672, 2)).astype(np.float32)}
    targets = {4: rng.normal(0.03, 0.02, size=(800, 2)).astype(np.float32)}
    reference = np.concatenate(
        [histories[4].astype(np.float64).reshape(-1), targets[4].astype(np.float64).reshape(-1)]
    )
    fitted = experiment.fit_training_scaler(histories, targets)
    assert fitted.fit_scope == "training arrays only"
    assert fitted.mean == pytest.approx(float(reference.mean()), rel=1e-13, abs=1e-14)
    assert fitted.scale == pytest.approx(float(reference.std(ddof=0)), rel=1e-12, abs=1e-14)


def test_source_identity_covers_first_party_import_and_data_path_modules():
    expected = set(experiment.SOURCE_IDENTITY_FILES)
    root = experiment.Path(experiment.__file__).resolve().parents[1]
    recorded = {
        experiment.Path(path).resolve().relative_to(root).as_posix(): digest
        for path, digest in experiment._source_identity().items()
    }
    assert expected == set(recorded)
    assert all(len(digest) == 64 for digest in recorded.values())


def test_source_identity_fails_closed_for_missing_required_source(monkeypatch):
    monkeypatch.setattr(experiment, "SOURCE_IDENTITY_FILES", ("phase9_geant/not-present.py",))
    with pytest.raises(ValueError, match="Required first-party source identity file"):
        experiment._source_identity()


def test_loader_accepts_synthetic_train_and_validation_with_exact_window_identity(tmp_path, monkeypatch):
    repo, root, arrays = _synthetic_dataset(tmp_path)
    metadata_edges = tuple((f"source-{i}", f"target-{i}") for i in range(72))
    _patch_topology(monkeypatch, metadata_edges)
    _patch_loader_source_identity(monkeypatch)
    train_hx, train_y, edges, train_index = experiment._load_split_arrays(
        root, "train", repository_root=repo
    )
    val_hx, val_y, val_edges, val_index = experiment._load_split_arrays(
        root, "validation", repository_root=repo
    )
    assert train_hx[4].shape == val_hx[4].shape == (2, 672, 72)
    assert train_y[96].shape == val_y[96].shape == (2, 72)
    assert edges == val_edges and len(edges) == 72
    assert len(train_index["4"]) == len(val_index["96"]) == 2
    np.testing.assert_array_equal(train_hx[4], arrays[("train", "history", 4)])


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda m: m.update(source_scope={"train": [0, 8020], "validation": [8020, 9740], "test": [9740, 11460]}), "source split provenance"),
        (lambda m: m["source_hashes_at_build"].update({"phase9_geant/validity_v2.py": "d" * 64}), "source identity drift"),
        (lambda m: m["build_configuration"].update(history_length=5), "configuration identity"),
        (lambda m: m["topology"].update(ordered_edge_sha256="d" * 64), "directed-edge identity"),
        (lambda m: m["splits"]["train"].update(source_file_names=["not-the-anchored-training-file.txt"]), "source filename provenance"),
        (lambda m: m.update(artifact_names=[*m["artifact_names"], "target_test_h4.npy"]), "artifact allowlist"),
    ],
)
def test_loader_rejects_metadata_tampering_before_array_or_checkpoint_load(
    tmp_path, monkeypatch, mutate, message
):
    repo, root, _ = _synthetic_dataset(tmp_path)
    edges = tuple((f"source-{i}", f"target-{i}") for i in range(72))
    _patch_topology(monkeypatch, edges)
    _patch_loader_source_identity(monkeypatch)
    metadata_path = root / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    mutate(metadata)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    array_load_called = False

    def forbidden_array_load(*_args, **_kwargs):
        nonlocal array_load_called
        array_load_called = True
        raise AssertionError("metadata mismatch must fail before reading an array")

    monkeypatch.setattr(data.np, "load", forbidden_array_load)
    with pytest.raises((ValueError, KeyError), match=message):
        experiment._load_split_arrays(root, "train", repository_root=repo)
    assert not array_load_called


def test_loader_rejects_mismatched_train_array_fingerprint_on_synthetic_data(tmp_path, monkeypatch):
    repo, root, _ = _synthetic_dataset(tmp_path)
    edges = tuple((f"source-{i}", f"target-{i}") for i in range(72))
    _patch_topology(monkeypatch, edges)
    _patch_loader_source_identity(monkeypatch)
    path = root / "history_train_h4.npy"
    altered = np.load(path, allow_pickle=False)
    altered[0, 0, 0] += 0.5
    np.save(path, altered, allow_pickle=False)
    with pytest.raises(ValueError, match="history fingerprint mismatch"):
        experiment._load_split_arrays(root, "train", repository_root=repo)


def test_train_only_raw_screen_preserves_horizon_and_embargo_contract():
    # Tiny link count, but the exact official training time axis is retained.
    time = np.arange(8020, dtype=np.float64)
    utilization = np.column_stack((0.2 + 1e-6 * time, 0.3 + 2e-6 * time))
    timestamps = time.astype(np.int64) * 900
    results, counts = experiment.screen_training_utilization(
        utilization, [True] * 8020, timestamps, EDGES
    )
    assert set(results) == {4, 96}
    for horizon, result in results.items():
        assert result.fit_origins >= 2
        assert result.holdout_origins > 0
        assert result.holdout_pairs == result.holdout_origins * len(EDGES)
        assert counts[horizon]["fit_origins"] == result.fit_origins
        assert counts[horizon]["holdout_origins"] == result.holdout_origins
        assert counts[horizon]["embargoed_eligible_origins"] >= 0


def test_builder_refuses_existing_output_before_source_parser_or_paths(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / "phase9_geant").mkdir(parents=True)
    processed = repo / "data" / "processed"
    processed.mkdir(parents=True)
    output = processed / f"{data.OUTPUT_PREFIX}existing"
    output.mkdir()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("existing output must be refused before any source selection/parsing")

    monkeypatch.setattr(data, "expected_split_paths_for_splits", forbidden)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        data.build_train_validation(repository_root=repo, output_root=output)


def test_synthetic_training_cycle_writes_complete_checkpoint_after_validation_only_selection(
    tmp_path, monkeypatch
):
    import torch.nn as nn

    calls = []

    class TinyModel(nn.Module):
        def __init__(self, edge_order, *, history_length, horizons, hidden_size, alpha,
                     ewma_steps, include_seasonal_features, adjacency_mode):
            super().__init__()
            self.edge_order = tuple(edge_order)
            self.horizons = tuple(horizons)
            self.ewma_steps = ewma_steps
            self.shuffle_seed = 20261019
            self.bias = nn.Parameter(torch.zeros(1))
            raw = torch.tensor([alpha * (1 - alpha) ** (ewma_steps - 1 - k)
                                for k in range(ewma_steps)], dtype=torch.float32)
            self.register_buffer("ewma_weights", raw / raw.sum())
            self.register_buffer("topology_adjacency", torch.eye(len(edge_order)))
            self.register_buffer("identity_adjacency", torch.eye(len(edge_order)))
            self.register_buffer("shuffled_adjacency", torch.eye(len(edge_order)))

        def _adjacency(self, mode):
            return getattr(self, f"{mode}_adjacency")

        def _ewma(self, history):
            return torch.einsum(
                "bte,t->be", history[:, -self.ewma_steps :, :], self.ewma_weights.to(history)
            )

        def forward(self, history):
            calls.append((self.training, torch.is_grad_enabled()))
            baseline = self._ewma(history)
            return baseline[:, None, :].expand(-1, len(self.horizons), -1) + self.bias

    monkeypatch.setattr(experiment, "SpatioTemporalLinkForecaster", TinyModel)
    monkeypatch.setattr(experiment, "HORIZONS", (96,))
    monkeypatch.setattr(experiment, "TRAINING_CONFIG", {
        "epochs": 1, "patience": 1, "learning_rate": 0.005, "batch_size": 2,
        "optimizer": "Adam", "loss": "synthetic SmoothL1", "hidden_size": 2,
        "alpha": 0.8, "ewma_steps": 5, "seeds": [42],
    })
    rng = np.random.default_rng(57)
    histories = {h: rng.uniform(0.1, 0.5, size=(3, 672, 2)).astype(np.float32) for h in (96,)}
    targets = {h: (histories[h][:, -1, :] + 0.02).astype(np.float32) for h in (96,)}
    validation_histories = {h: rng.uniform(0.1, 0.5, size=(2, 672, 2)).astype(np.float32) for h in (96,)}
    validation_targets = {h: (validation_histories[h][:, -1, :] + 0.03).astype(np.float32) for h in (96,)}
    data_root = tmp_path / "synthetic-data"
    data_root.mkdir()
    output = data_root / "runs" / "horizon_v2_identity_seed_42"
    result = experiment._train_arrays(
        histories=histories, targets=targets,
        validation_histories=validation_histories, validation_targets=validation_targets,
        edge_order=EDGES, output_directory=output, seed=42, adjacency_mode="identity",
        run_provenance={"data_root": str(data_root), "ordered_edge_sha256": "f" * 64},
    )
    assert result["status"] == "complete"
    assert output.joinpath("best.pt").is_file() and output.joinpath("best.pt").stat().st_size > 0
    assert result["adjacency_identity"]["mode"] == "identity"
    assert len(result["adjacency_identity"]["normalized_matrix_sha256"]) == 64
    assert result["best_epoch"] == 1 and result["epochs_completed"] == 1
    assert any(training and grad for training, grad in calls)
    assert any((not training) and (not grad) for training, grad in calls)
    assert all(not grad for training, grad in calls if not training)


def test_loader_rejects_extra_test_artifact_before_array_loading(tmp_path, monkeypatch):
    repo, root, _ = _synthetic_dataset(tmp_path)
    (root / "target_test_h4.npy").write_bytes(b"synthetic forbidden filename only")
    called = False

    def forbidden_loader(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("array loader must not run for invalid inventory")

    monkeypatch.setattr(data.np, "load", forbidden_loader)
    with pytest.raises(ValueError, match="inventory mismatch"):
        experiment._load_split_arrays(root, "train", repository_root=repo)
    assert not called


def test_protocol_constants_match_frozen_horizon_design():
    assert experiment.HISTORY_LENGTH == 672
    assert experiment.HORIZONS == (4, 96)
    assert experiment.RIDGE_ALPHA == 1.0
    assert experiment.SCREEN_EMBARGO_STEPS == 768
    assert experiment.SCREEN_THRESHOLDS == {"stop_below": 0.005, "review_at_or_above": 0.01}
