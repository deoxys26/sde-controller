"""End-to-end, train-only-fitted UNSW-NB15 experiment runner."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from src.security.evaluate import category_breakdown, evaluate_binary
from src.security.models import build_logistic_regression, build_random_forest
from src.security.preprocessing import UNSWPreprocessor
from src.security.risk import attack_probability, risk_from_probability


def _sample(frame, size, seed):
    return frame if not size or len(frame) <= size else frame.sample(n=int(size), random_state=seed)


def run_security_experiment(train_path, test_path, config):
    """Fit only training-derived transformations/models and evaluate the untouched test CSV."""
    seed = int(config.get("random_seed", 42))
    train_frame, test_frame = pd.read_csv(train_path), pd.read_csv(test_path)
    sampled_train = _sample(train_frame, config.get("training_sample_size"), seed)
    fit_frame, validation_frame = train_test_split(sampled_train, test_size=float(config.get("validation_ratio", .15)),
                                                   random_state=seed, stratify=sampled_train["label"])
    preprocessor = UNSWPreprocessor().fit(fit_frame)
    x_fit, x_validation, x_test = preprocessor.transform(fit_frame), preprocessor.transform(validation_frame), preprocessor.transform(test_frame)
    y_fit, y_validation, y_test = preprocessor.labels(fit_frame), preprocessor.labels(validation_frame), preprocessor.labels(test_frame)
    output = {"class_distribution": {"fit": pd.Series(y_fit).value_counts().sort_index().to_dict(),
                                      "test": pd.Series(y_test).value_counts().sort_index().to_dict()},
              "validation": {}, "test": {}, "preprocessor": preprocessor}
    for name, builder in (("logistic_regression", build_logistic_regression), ("random_forest", build_random_forest)):
        model = builder(config).fit(x_fit, y_fit)
        validation_risk = risk_from_probability(attack_probability(model, x_validation))
        test_risk = risk_from_probability(attack_probability(model, x_test))
        output["validation"][name] = evaluate_binary(y_validation, validation_risk, config.get("risk_threshold", .5))
        output["test"][name] = {**evaluate_binary(y_test, test_risk, config.get("risk_threshold", .5)),
                                "attack_categories": category_breakdown(test_frame.reset_index(drop=True), test_risk, config.get("risk_threshold", .5))}
    return output
