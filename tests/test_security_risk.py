import numpy as np

from src.security.risk import risk_from_probability, threshold_risk


def test_risk_is_clipped_and_thresholding_is_deterministic():
    risk = risk_from_probability([-1, .2, .5, 2])
    np.testing.assert_array_equal(risk, [0, .2, .5, 1])
    np.testing.assert_array_equal(threshold_risk(risk, .5), [0, 0, 1, 1])
