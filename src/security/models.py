"""Small reproducible classical classifiers for the binary security task."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def build_logistic_regression(config):
    return LogisticRegression(max_iter=int(config.get("logistic_max_iter", 300)),
                              class_weight=config.get("class_weight"), random_state=int(config.get("random_seed", 42)))


def build_random_forest(config):
    return RandomForestClassifier(n_estimators=int(config.get("random_forest_estimators", 50)),
                                  class_weight=config.get("class_weight"), random_state=int(config.get("random_seed", 42)),
                                  n_jobs=-1)
