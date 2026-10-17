import pytest
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

def test_feature_variance():
    # just checking that if we use utilization/traffic they have variance
    if os.path.exists("data/processed/geant/link_series_utilization.npy"):
        utils = np.load("data/processed/geant/link_series_utilization.npy")
        var = np.nanvar(utils, axis=0)
        assert np.all(var > 0), "Some links have zero variance in utilization."
    else:
        pytest.skip("Data not yet processed")
