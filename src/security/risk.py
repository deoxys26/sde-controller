"""Continuous risk mapping for later consumers; no routing integration occurs here."""

from __future__ import annotations

import numpy as np


def attack_probability(model, features):
    """Return P(label=1) from a probabilistic binary classifier."""
    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError("Model does not expose the attack class (label 1)")
    return model.predict_proba(features)[:, classes.index(1)]


def risk_from_probability(probabilities):
    """Risk is the model's uncalibrated P(attack), clipped to [0, 1]."""
    return np.clip(np.asarray(probabilities, dtype=float), 0.0, 1.0)


def threshold_risk(risk, threshold=.5):
    return (risk_from_probability(risk) >= threshold).astype(int)
