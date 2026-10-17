"""Synthetic-only tests for the GEANT v2 diagnostic metric helpers."""

import inspect

import numpy as np
import pytest

from phase9_geant.diagnose_v2_baselines import (
    build_baselines,
    burst_window_mask,
    metric_bundle,
    per_link_metrics,
    subgroup_masks,
    training_thresholds,
)


def test_baselines_use_only_causal_history_and_preserve_alignment():
    history = np.arange(2 * 5 * 3, dtype=np.float64).reshape(2, 5, 3)
    predictions = build_baselines(history)

    assert predictions["persistence"].shape == (2, 3)
    np.testing.assert_array_equal(predictions["persistence"], history[:, 4, :])
    np.testing.assert_array_equal(predictions["seasonal_naive_4"], history[:, 1, :])
    np.testing.assert_allclose(predictions["moving_average_3"], history[:, 2:, :].mean(axis=1))

    alpha = 0.8
    weights = alpha * (1 - alpha) ** (4 - np.arange(5))
    weights = weights / weights.sum()
    np.testing.assert_allclose(predictions["ewma_alpha_0.8"], np.tensordot(history, weights, axes=([1], [0])))


def test_forecast_api_has_no_future_target_argument():
    history = np.ones((1, 5, 2), dtype=np.float64)
    assert list(inspect.signature(build_baselines).parameters) == ["history"]
    with pytest.raises(TypeError):
        build_baselines(history, np.full((1, 2), 1e9))


def test_metric_alignment_rejects_shape_mismatch():
    target = np.zeros((3, 2), dtype=np.float32)
    prediction = np.zeros((3, 1), dtype=np.float32)
    with pytest.raises(ValueError, match="shape mismatch"):
        metric_bundle(target, prediction)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_missing_or_nonfinite_predictions_fail_instead_of_reducing_coverage(bad_value):
    target = np.zeros((2, 2), dtype=np.float64)
    prediction = np.zeros_like(target)
    prediction[0, 1] = bad_value
    with pytest.raises(ValueError, match="nonfinite"):
        metric_bundle(target, prediction)


def test_masked_metric_reports_explicit_subgroup_coverage():
    target = np.asarray([[0.0, 0.2], [0.1, 0.3]], dtype=np.float64)
    prediction = target + 0.1
    mask = np.asarray([[True, False], [True, True]])
    result = metric_bundle(target, prediction, mask)
    assert result["evaluated_pairs"] == 3
    assert result["available_pairs"] == 3
    assert result["coverage"] == pytest.approx(1.0)
    assert result["full_split_pairs"] == 4
    assert result["mae"] == pytest.approx(0.1)


def test_subgroup_thresholds_are_fit_from_training_targets_and_counts_align():
    training = np.arange(100, dtype=np.float64).reshape(20, 5)
    thresholds = training_thresholds(training)
    validation = np.asarray([[0.0, 1.0, 49.0, 60.0, 99.0]], dtype=np.float64)
    masks = subgroup_masks(validation, thresholds)
    assert thresholds["p50"] == pytest.approx(49.5)
    assert int(masks["low_le_train_p50"].sum()) == 3
    assert int(masks["mid_train_p50_to_p90"].sum()) == 1
    assert int(masks["high_ge_train_p90"].sum()) == 1
    assert int(masks["burst_cells_ge_train_p99"].sum()) == 1
    assert burst_window_mask(validation, thresholds).tolist() == [True]


def test_per_link_metrics_are_paired_on_same_rows():
    target = np.asarray([[0.0, 1.0], [1.0, 2.0]], dtype=np.float64)
    prediction = np.asarray([[0.5, 1.0], [1.5, 3.0]], dtype=np.float64)
    rows = per_link_metrics(target, prediction)
    assert [row["edge_index"] for row in rows] == [0, 1]
    assert [row["evaluated_pairs"] for row in rows] == [2, 2]
    assert rows[0]["mae"] == pytest.approx(0.5)
    assert rows[1]["mae"] == pytest.approx(0.5)

