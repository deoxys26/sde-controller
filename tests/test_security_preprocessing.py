import numpy as np
import pandas as pd
from scipy import sparse

from src.security.preprocessing import UNSWPreprocessor


def _frame(rows=30):
    labels = np.array([0, 1] * (rows // 2))
    return pd.DataFrame({"id": range(rows), "dur": np.linspace(.1, 2, rows), "rate": np.arange(rows),
                         "proto": ["tcp", "udp"] * (rows // 2), "service": ["http", "dns"] * (rows // 2),
                         "state": ["FIN", "CON"] * (rows // 2), "attack_cat": np.where(labels, "Generic", "Normal"),
                         "label": labels})


def test_preprocessing_has_train_only_finite_and_deterministic_transformations():
    train, test = _frame(), _frame()
    test.loc[:, "dur"] = 999999  # Test statistics must never affect the fitted scaler.
    first = UNSWPreprocessor().fit(train)
    second = UNSWPreprocessor().fit(train)
    x_train, x_test = first.transform(train), first.transform(test)
    assert first.categorical_columns == ("proto", "service", "state")
    assert x_train.shape[1] == x_test.shape[1]
    values = x_test.data if sparse.issparse(x_test) else x_test
    assert np.isfinite(values).all()
    np.testing.assert_allclose(first.transformer.named_transformers_["numeric"].named_steps["scaler"].mean_, [train.dur.mean(), train.rate.mean()])
    repeat = first.transform(train) - second.transform(train)
    repeat_values = repeat.data if sparse.issparse(repeat) else repeat
    assert np.allclose(repeat_values, 0)
