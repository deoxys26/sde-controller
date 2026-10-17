import numpy as np
import pytest

from phase9_geant.run_baselines import _windows_for_block, compute_metrics


def test_windows_have_five_history_steps_one_target_and_stay_in_block():
    series = np.arange(20 * 2, dtype=float).reshape(20, 2)
    valid = np.ones(20, dtype=bool)
    windows = _windows_for_block(series, [5, 6], 5, 12, valid)
    assert windows["X"].shape == (2, 5, 2)
    assert windows["y"].shape == (2, 2)
    assert windows["target_ts"].tolist() == [10, 11]
    assert np.array_equal(windows["X"][0, -1], series[9])
    assert np.array_equal(windows["y"][0], series[10])


def test_windows_reject_invalid_or_cross_block_observations():
    series = np.zeros((20, 2), dtype=float)
    valid = np.ones(20, dtype=bool)
    valid[7] = False
    with pytest.raises(ValueError, match="invalid"):
        _windows_for_block(series, [5], 5, 12, valid)
    valid[7] = True
    with pytest.raises(ValueError, match="boundary"):
        _windows_for_block(series, [7], 5, 12, valid)


def test_metrics_reject_unaligned_or_nonfinite_predictions():
    with pytest.raises(ValueError):
        compute_metrics(np.ones((2, 2)), np.ones((1, 2)), 1.0)
    with pytest.raises(ValueError):
        compute_metrics(np.ones((2, 2)), np.full((2, 2), np.nan), 1.0)
