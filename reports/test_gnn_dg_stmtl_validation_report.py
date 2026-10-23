from __future__ import annotations

import csv

import pytest

import generate_gnn_dg_stmtl_validation_report as report


def test_aggregates_use_sample_standard_deviation_and_correct_counts():
    mean, sd, n = report.summarize(report.GROUPS["Identity GNN"], "mae")
    assert n == 3
    assert mean == pytest.approx(0.00594781545373562)
    assert sd == pytest.approx(0.00008257010216730141)
    assert report.topology_two_seed_summary("mae")[2] == 2


def test_improvement_formula_and_dg_ewma_precision_difference():
    assert report.improvement_percent(10.0, 9.0) == pytest.approx(10.0)
    assert abs(report.DG_RECORDED_EWMA_MAE - report.EWMA["mae"]) == pytest.approx(9.1523301e-11, rel=1e-7)


def test_report_discloses_provenance_caveats_and_not_significance():
    text = report.build_report()
    assert "individual run manifest is unavailable" in text
    assert "not a newly executed or provenance-verified benchmark" in text
    assert "not formal confidence intervals" in text
    assert "not an exact reproduction" in text
    assert "sealed-test" in text
    assert "GA routing benefit" in text
    assert "Identity GNN seed 123 has the lowest individual MAE" in text
    assert "Topology GNN seed 123 has the lowest individual RMSE" in text
    assert "Parameters" in text and "Mean training seconds" in text


def test_generate_writes_requested_csv_report_and_four_figures(tmp_path):
    generated = report.generate(tmp_path / "new-report")
    assert len(generated) == 6
    assert all(path.is_file() and path.stat().st_size > 0 for path in generated)
    csv_path = tmp_path / "new-report" / "gnn_vs_dg_stmtl_validation_comparison.csv"
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    topology = [row for row in rows if row["record_type"] == "aggregate" and row["method"] == "Topology GNN (historical n=3)"][0]
    assert topology["n"] == "3"
    assert any(row["record_type"] == "baseline" and row["method"] == "EWMA" for row in rows)
    paired = [row for row in rows if row["record_type"] == "paired_topology_vs_dg"]
    assert {row["seed"] for row in paired} == {"42", "123", "2026"}


def test_generate_refuses_to_overwrite_existing_artifact(tmp_path):
    destination = tmp_path / "new-report"
    destination.mkdir()
    report_path = destination / "gnn_vs_dg_stmtl_validation_comparison.md"
    report_path.write_text("keep me", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        report.generate(destination)
    assert report_path.read_text(encoding="utf-8") == "keep me"

