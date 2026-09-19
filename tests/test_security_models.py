import numpy as np
import pandas as pd

from src.security.models import build_logistic_regression, build_random_forest
from src.security.preprocessing import UNSWPreprocessor
from src.security.risk import attack_probability


def _data():
    labels = np.array([0, 1] * 25)
    frame = pd.DataFrame({"id": range(50), "dur": labels + np.linspace(.01, .1, 50), "proto": np.where(labels, "tcp", "udp"),
                          "attack_cat": np.where(labels, "Generic", "Normal"), "label": labels})
    preprocessor = UNSWPreprocessor().fit(frame)
    return preprocessor.transform(frame), labels


def test_seeded_models_train_and_return_bounded_probabilities():
    features, labels = _data()
    config = {"random_seed": 9, "logistic_max_iter": 100, "random_forest_estimators": 5}
    for builder in (build_logistic_regression, build_random_forest):
        model = builder(config).fit(features, labels)
        probability = attack_probability(model, features)
        assert probability.shape == labels.shape
        assert ((0 <= probability) & (probability <= 1)).all()
    first = attack_probability(build_random_forest(config).fit(features, labels), features)
    second = attack_probability(build_random_forest(config).fit(features, labels), features)
    np.testing.assert_array_equal(first, second)
