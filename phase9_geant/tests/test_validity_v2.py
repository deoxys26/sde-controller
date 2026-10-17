"""Synthetic-only tests for the isolated GEANT validity protocol v2 code."""

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from phase9_geant import validity_v2 as v2

START = datetime(2000, 1, 1)
SYNTHETIC_SOURCE = "synthetic fixture anchor; not a GEANT timestamp assertion"


def _name(index: int, *, start: datetime = START) -> Path:
    stamp = (start + timedelta(minutes=15 * index)).strftime("%Y%m%d-%H%M")
    return Path(f"demandMatrix-geant-uhlig-15min-{stamp}.txt")


def _selection(count: int = v2.TRAIN_END):
    return v2.select_training_files(
        [_name(i) for i in range(count)],
        expected_start_timestamp=START,
        expected_start_timestamp_source=SYNTHETIC_SOURCE,
    )


def _observation(index: int, total: float, count: int = 1) -> v2.MatrixObservation:
    return v2.MatrixObservation(
        START + timedelta(minutes=15 * index),
        np.asarray([[0.0, total], [0.0, 0.0]], dtype=np.float64),
        count,
        _name(index).name,
    )


def _full_train_observations(*, empty: bool = False):
    return tuple(
        _observation(i, 0.0 if empty else float(i % 11), count=0 if empty else 1)
        for i in range(v2.TRAIN_END)
    )


def _read_synthetic(observations, *, extra_paths=(), parser=None):
    """Exercise the production reader with synthetic names and parser output."""
    observations_by_name = {item.filename: item for item in observations}

    def synthetic_parser(path, nodes, node_to_index):
        if parser is not None:
            return parser(path, nodes, node_to_index)
        return observations_by_name[Path(path).name]

    return v2.read_training_period(
        Path("synthetic-raw-directory"),
        ("a", "b"),
        {"a": 0, "b": 1},
        expected_start_timestamp=START,
        expected_start_timestamp_source=SYNTHETIC_SOURCE,
        discovery=lambda _directory: tuple(
            [_name(i) for i in range(v2.TRAIN_END)] + list(extra_paths)
        ),
        parser=synthetic_parser,
    )


def test_validation_test_fixtures_cannot_change_training_cutoff(monkeypatch):
    """Run the production reader twice; only its training parser calls may fit."""
    paths = [_name(i) for i in range(v2.TRAIN_END + 2)]
    monkeypatch.setattr(v2, "_discover_matrix_paths", lambda directory: tuple(paths))
    holdout_payloads = {paths[-2].name: 1e9, paths[-1].name: 2e9}

    def run_reader():
        parsed_names = []

        def synthetic_parser(path, nodes, node_to_index):
            name = Path(path).name
            parsed_names.append(name)
            # These values stand in for holdout matrix contents; if the reader
            # ever passes a holdout filename, the changed value affects parsing.
            if name in holdout_payloads:
                return _observation(v2.TRAIN_END, holdout_payloads[name])
            index = len(parsed_names) - 1
            return _observation(index, float(index % 11))

        read = v2.read_training_period(
            Path("synthetic-raw-directory"),
            ("a", "b"),
            {"a": 0, "b": 1},
            expected_start_timestamp=START,
            expected_start_timestamp_source=SYNTHETIC_SOURCE,
            discovery=lambda _directory: tuple(paths),
            parser=synthetic_parser,
        )
        return read, parsed_names

    first, first_parsed = run_reader()
    holdout_payloads.update({paths[-2].name: 8e99, paths[-1].name: 9e99})
    second, second_parsed = run_reader()
    expected_training_names = [path.name for path in paths[: v2.TRAIN_END]]
    assert first_parsed == second_parsed == expected_training_names
    assert first.cutoff_mbps == second.cutoff_mbps
    assert first.cutoff_fit_filenames == second.cutoff_fit_filenames
    assert first.metadata == second.metadata


def test_legacy_all_series_parser_is_not_called(monkeypatch):
    paths = [_name(i) for i in range(v2.TRAIN_END)]

    def forbidden(*args, **kwargs):
        raise AssertionError("legacy all-series parser must never be called")

    from phase9_geant import parse_geant

    monkeypatch.setattr(parse_geant, "parse_all_matrices", forbidden)
    monkeypatch.setattr(v2, "parse_all_matrices", forbidden, raising=False)
    seen_names = []

    def parser(path, nodes, mapping):
        seen_names.append(Path(path).name)
        return _observation(len(seen_names) - 1, 1.0)

    read = v2.read_training_period(
        Path("synthetic-raw-directory"),
        ("a", "b"),
        {"a": 0, "b": 1},
        expected_start_timestamp=START,
        expected_start_timestamp_source=SYNTHETIC_SOURCE,
        discovery=lambda _directory: tuple(paths),
        parser=parser,
    )
    assert len(read.observations) == v2.TRAIN_END
    assert seen_names == [path.name for path in paths]


@pytest.mark.parametrize(
    ("paths", "error"),
    [
        ([_name(1), _name(2), _name(3)], "start mismatch"),
        ([_name(0), _name(1), _name(1), _name(3)], "Duplicate"),
        ([_name(0), _name(1), _name(3), _name(4)], "gap"),
        ([Path("demandMatrix-geant-uhlig-15min-not-a-time.txt")], "Unexpected"),
        ([_name(0), _name(1)], "Insufficient"),
    ],
)
def test_bad_start_duplicate_gap_malformed_name_and_insufficient_files_fail(
    paths, error
):
    with pytest.raises(ValueError, match=error):
        v2.select_training_files(
            paths,
            expected_start_timestamp=START,
            expected_start_timestamp_source=SYNTHETIC_SOURCE,
            training_end=3,
        )


def test_reader_rejects_initially_missing_file_before_parser_call(monkeypatch):
    paths = [_name(i) for i in range(1, v2.TRAIN_END + 1)]
    called = []
    with pytest.raises(ValueError, match="start mismatch"):
        v2.read_training_period(
            Path("synthetic-raw-directory"),
            ("a", "b"),
            {"a": 0, "b": 1},
            expected_start_timestamp=START,
            expected_start_timestamp_source=SYNTHETIC_SOURCE,
            discovery=lambda _directory: tuple(paths),
            parser=lambda *args: called.append(args),
        )
    assert called == []


def test_reader_requires_explicit_start_and_nonempty_provenance():
    with pytest.raises(TypeError):
        v2.read_training_period(Path("synthetic-raw-directory"), ("a",), {"a": 0})
    with pytest.raises(ValueError, match="provenance"):
        v2.select_training_files(
            [_name(0)],
            expected_start_timestamp=START,
            expected_start_timestamp_source="  ",
            training_end=1,
        )


def test_production_reader_selects_exactly_8020_files_before_parsing(monkeypatch):
    paths = [_name(i) for i in range(v2.TRAIN_END + 2)]
    parsed_names = []

    def parser(path, nodes, mapping):
        name = Path(path).name
        parsed_names.append(name)
        return _observation(len(parsed_names) - 1, 1.0)

    result = v2.read_training_period(
        Path("synthetic-raw-directory"),
        ("a", "b"),
        {"a": 0, "b": 1},
        expected_start_timestamp=START,
        expected_start_timestamp_source=SYNTHETIC_SOURCE,
        discovery=lambda _directory: tuple(paths),
        parser=parser,
    )
    assert parsed_names == [path.name for path in paths[: v2.TRAIN_END]]
    assert not set(parsed_names).intersection(path.name for path in paths[v2.TRAIN_END :])
    assert len(result.observations) == v2.TRAIN_END


def _matrix_text(timestamp: datetime, demands: str) -> str:
    stamp = timestamp.strftime("%Y%m%d-%H%M")
    return f"META ( time={stamp} )\nDEMANDS (\n{demands}\n)\nADMISSIBLE_PATHS (\n)\n"


def test_parser_accepts_synthetic_empty_matrix_and_valid_demand(tmp_path):
    empty_path = tmp_path / _name(0).name
    valid_path = tmp_path / _name(1).name
    empty_path.write_text(_matrix_text(START, ""), encoding="utf-8")
    valid_path.write_text(
        _matrix_text(
            START + timedelta(minutes=15),
            "od_a_b ( a b ) 1 20000 UNLIMITED",
        ),
        encoding="utf-8",
    )
    empty = v2.parse_one_training_matrix(empty_path, ("a", "b"), {"a": 0, "b": 1})
    valid = v2.parse_one_training_matrix(valid_path, ("a", "b"), {"a": 0, "b": 1})
    assert empty.demand_count == 0
    assert empty.total_demand_mbps == 0.0
    assert v2.is_empty_matrix_v2(empty)
    assert v2.invalidity_reason(empty, 100_000.0) == ("empty",)
    assert valid.demand_count == 1
    assert valid.total_demand_mbps == 20_000.0


@pytest.mark.parametrize(
    ("filename", "content", "error"),
    [
        (_name(0).name, _matrix_text(START, "not a demand row"), "Malformed demand row"),
        (_name(0).name, "META ( time=20000101-0000 )\nADMISSIBLE_PATHS (\n)\n", "Missing or incomplete DEMANDS"),
        (_name(0).name, "META ( time=20000101-0000 )\nDEMANDS (\n)\nDEMANDS (\n)\n", "Duplicate DEMANDS section"),
        (_name(0).name, "META ( time=20000101-0000 )\nDEMANDS (\nADMISSIBLE_PATHS (\n)\n", "Unclosed DEMANDS section"),
        (_name(0).name, _matrix_text(START + timedelta(minutes=15), ""), "timestamp mismatch"),
    ],
)
def test_parser_fails_closed_on_malformed_or_mismatched_matrix(
    tmp_path, filename, content, error
):
    path = tmp_path / filename
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=error):
        v2.parse_one_training_matrix(path, ("a", "b"), {"a": 0, "b": 1})


def test_missing_file_is_distinct_from_malformed_matrix(tmp_path):
    missing_path = tmp_path / _name(0).name
    with pytest.raises(RuntimeError, match="Unreadable training matrix"):
        v2.parse_one_training_matrix(missing_path, ("a", "b"), {"a": 0, "b": 1})

    malformed_path = tmp_path / _name(1).name
    malformed_path.write_text(
        "META ( time=20000101-0015 )\nDEMANDS (\nnot a demand row\n)\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Malformed demand row"):
        v2.parse_one_training_matrix(
            malformed_path, ("a", "b"), {"a": 0, "b": 1}
        )


def test_strict_thresholds_and_individual_limit_equality():
    equal_total = _observation(0, 100.0)
    greater_total = _observation(1, 100.01)
    equal_individual = v2.MatrixObservation(
        equal_total.timestamp,
        np.asarray([[0.0, 20_000.0], [0.0, 0.0]], dtype=np.float64),
        1,
        equal_total.filename,
    )
    greater_individual = v2.MatrixObservation(
        equal_total.timestamp,
        np.asarray([[0.0, 20_000.01], [0.0, 0.0]], dtype=np.float64),
        1,
        equal_total.filename,
    )
    assert v2.invalidity_reason(equal_total, 100.0) == ()
    assert v2.invalidity_reason(greater_total, 100.0) == ("total_demand",)
    assert not v2.exceeds_individual_demand_limit_v2(equal_individual)
    assert v2.exceeds_individual_demand_limit_v2(greater_individual)
    assert v2.invalidity_reason(_observation(2, 0.0, count=0), 100.0) == ("empty",)


def test_production_reader_fits_cutoff_and_records_truthful_provenance():
    observations = list(_full_train_observations())
    observations[0] = _observation(0, 0.0, count=0)
    result = _read_synthetic(observations)
    assert result.metadata["protocol_id"] == v2.PROTOCOL_ID
    assert result.metadata["training_indices"] == list(range(v2.TRAIN_END))
    assert result.metadata["training_boundary"] == [0, v2.TRAIN_END]
    assert result.metadata["expected_start_timestamp_source"] == SYNTHETIC_SOURCE
    assert result.metadata["cutoff_fit_filenames"] == [
        item.filename for item in observations[1:]
    ]
    assert result.metadata["cutoff_fit_nonempty_count"] == v2.TRAIN_END - 1
    assert result.observations[0].demand_count == 0
    assert result.observations[0].filename not in result.cutoff_fit_filenames
    assert result.metadata["total_demand_median_mbps"] == 5.0
    assert result.metadata["total_demand_multiplier"] == 5.0
    assert result.cutoff_mbps == 25.0

    with pytest.raises(ValueError, match="no nonempty"):
        _read_synthetic(_full_train_observations(empty=True))


def test_cutoff_cannot_be_fitted_from_caller_supplied_observation_list():
    assert not hasattr(v2, "fit_training_cutoff")
    observations = _full_train_observations()
    with pytest.raises(TypeError):
        v2.read_training_period(
            observations,
            None,
            None,
            expected_start_timestamp=START,
            expected_start_timestamp_source=SYNTHETIC_SOURCE,
            selection=_selection(),
        )


@pytest.mark.parametrize("mismatch", ["filename", "timestamp"])
def test_reader_rejects_parser_output_not_matching_selected_file(mismatch):
    observations = _full_train_observations()

    def parser(path, nodes, mapping):
        index = len(calls)
        calls.append(Path(path).name)
        result = observations[index]
        if index == 0 and mismatch == "filename":
            return v2.MatrixObservation(
                result.timestamp, result.matrix_mbps, result.demand_count, "wrong.txt"
            )
        if index == 0 and mismatch == "timestamp":
            return v2.MatrixObservation(
                result.timestamp + timedelta(minutes=15),
                result.matrix_mbps,
                result.demand_count,
                result.filename,
            )
        return result

    calls = []
    with pytest.raises(ValueError, match="Parsed (filename|timestamp) mismatch"):
        _read_synthetic(observations, parser=parser)


def test_invalid_link_utilization_remains_nan():
    observations = [_observation(0, 0.0, count=0), _observation(1, 50.0)]
    traffic, utilization, reasons = v2.build_link_series_v2(
        observations,
        (("a", "b"),),
        {("a", "b"): ("a", "b")},
        100.0,
        nodes=("a", "b"),
    )
    assert np.isnan(traffic[0, 0])
    assert np.isnan(utilization[0, 0])
    assert traffic[1, 0] == 50.0
    assert utilization[1, 0] == pytest.approx(50.0 / v2.CAPACITY_MBPS)
    assert reasons[0] == ("empty",)


def test_windows_enforce_exact_official_splits_and_reject_invalid_or_gapped_data():
    timestamps = [START + i * timedelta(minutes=15) for i in range(v2.TEST_END)]
    valid = [True] * len(timestamps)
    train = v2.valid_window_starts(timestamps, valid, split_start=0, split_end=8020)
    validation = v2.valid_window_starts(
        timestamps, valid, split_start=8020, split_end=9740
    )
    test = v2.valid_window_starts(timestamps, valid, split_start=9740, split_end=11460)
    assert train[0] == 0 and train[-1] == 8014
    assert validation[0] == 8020 and validation[-1] == 9734
    assert test[0] == 9740 and test[-1] == 11454
    assert 8015 not in train and 8015 not in validation
    assert 9735 not in validation and 9735 not in test
    with pytest.raises(ValueError, match="official split"):
        v2.valid_window_starts(timestamps, valid, split_start=0, split_end=v2.TEST_END)

    invalid = valid.copy()
    invalid[8022] = False
    assert 8020 not in v2.valid_window_starts(
        timestamps, invalid, split_start=8020, split_end=9740
    )
    gap = timestamps.copy()
    gap[9743] += timedelta(minutes=15)
    assert 9740 not in v2.valid_window_starts(
        gap, valid, split_start=9740, split_end=11460
    )


def test_output_guard_rejects_legacy_dotdot_and_symlink_aliases(tmp_path):
    project = tmp_path / "repo"
    allowed = project / "data/processed/geant_validity_v2"
    allowed.parent.mkdir(parents=True)
    assert v2.validate_output_root(allowed, project_root=project) == allowed.resolve()
    for value, message in (
        (project / "data/processed/geant", "Output root must be the original v2 root"),
        (project / "data/processed/geant/nested", "Output root must be the original v2 root"),
        (project / "data/processed/geant_validity_v2/../geant/child", "Parent traversal"),
    ):
        with pytest.raises(ValueError, match=message):
            v2.validate_output_root(value, project_root=project)

    legacy = project / "data/processed/geant"
    legacy.mkdir(parents=True)
    allowed.mkdir(parents=True)
    alias = allowed / "legacy-alias"
    try:
        alias.symlink_to(legacy, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("Symlink creation is not supported by this test environment")
    with pytest.raises(ValueError, match="legacy"):
        v2.validate_output_root(alias / "child", project_root=project)


def test_cutoff_and_metadata_are_deterministic_for_identical_training_inputs():
    observations = _full_train_observations()
    first = _read_synthetic(observations)
    second = _read_synthetic(observations)
    assert first.cutoff_mbps == second.cutoff_mbps
    assert first.metadata == second.metadata
    assert v2.metadata_json(first) == v2.metadata_json(second)
